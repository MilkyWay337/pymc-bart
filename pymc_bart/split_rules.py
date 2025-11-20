#   Copyright 2022 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from abc import abstractmethod

from numba import njit
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from functools import partial

def target_split_rule(X, y, feature, min_samples_leaf=1, alpha=1.0, **kwargs):
    """
    Split rule for categorical features based on target statistics.
    
    Args:
        X: Input data
        y: Target values
        feature: Feature index to split on
        min_samples_leaf: Minimum samples per leaf
        alpha: Smoothing parameter for target statistics
    
    Returns:
        Tuple (split_value, gain) or None if no good split found
    """
    feature_values = X[:, feature]
    unique_vals = jnp.unique(feature_values)
    
    # Skip if not enough unique values
    if len(unique_vals) <= 1:
        return None
    
    # Calculate target statistics for each category
    category_stats = _calculate_category_stats(feature_values, y, unique_vals, alpha)
    
    if len(category_stats) < 2:
        return None
        
    # Order categories by target mean
    ordered_categories = _order_categories_by_target(category_stats)
    
    # Find best binary split
    best_split = _find_best_binary_split(feature_values, y, ordered_categories, 
                                       category_stats, min_samples_leaf)
    
    return best_split

def _calculate_category_stats(feature_values, y, unique_vals, alpha):
    """Calculate target statistics for each category with smoothing."""
    stats = {}
    global_mean = jnp.mean(y)
    
    for category in unique_vals:
        mask = feature_values == category
        n_category = jnp.sum(mask)
        
        if n_category == 0:
            continue
            
        category_mean = jnp.mean(y[mask])
        
        # Apply smoothing (like in CatBoost)
        smoothed_mean = (n_category * category_mean + alpha * global_mean) / (n_category + alpha)
        
        stats[category] = {
            'mean': smoothed_mean,
            'n': n_category,
            'raw_mean': category_mean
        }
        
    return stats

def _order_categories_by_target(category_stats):
    """Order categories by their target mean."""
    categories = list(category_stats.keys())
    means = [category_stats[cat]['mean'] for cat in categories]
    
    # Sort categories by mean
    sorted_indices = jnp.argsort(jnp.array(means))
    return [categories[i] for i in sorted_indices]

def _find_best_binary_split(feature_values, y, ordered_categories, 
                          category_stats, min_samples_leaf):
    """Find best binary split among ordered categories."""
    n_categories = len(ordered_categories)
    best_gain = -jnp.inf
    best_split_value = None
    
    for split_idx in range(1, n_categories):
        left_categories = set(ordered_categories[:split_idx])
        right_categories = set(ordered_categories[split_idx:])
        
        left_mask = jnp.isin(feature_values, jnp.array(list(left_categories)))
        right_mask = jnp.isin(feature_values, jnp.array(list(right_categories)))
        
        # Check minimum samples constraint
        n_left = jnp.sum(left_mask)
        n_right = jnp.sum(right_mask)
        
        if n_left < min_samples_leaf or n_right < min_samples_leaf:
            continue
        
        gain = _calculate_variance_reduction(y, left_mask, right_mask)
        
        if gain > best_gain and gain > 0:
            best_gain = gain
            # Use the mean between the last left and first right category
            last_left = category_stats[ordered_categories[split_idx-1]]['mean']
            first_right = category_stats[ordered_categories[split_idx]]['mean']
            best_split_value = (last_left + first_right) / 2
    
    if best_split_value is not None:
        return best_split_value, best_gain
    return None

def _calculate_variance_reduction(y, left_mask, right_mask):
    """Calculate variance reduction for a split."""
    y_left = y[left_mask]
    y_right = y[right_mask]
    
    if len(y_left) == 0 or len(y_right) == 0:
        return -jnp.inf
    
    n_left, n_right, n_all = len(y_left), len(y_right), len(y)
    
    # Calculate variances
    var_left = jnp.var(y_left) if len(y_left) > 1 else 0.0
    var_right = jnp.var(y_right) if len(y_right) > 1 else 0.0
    var_all = jnp.var(y)
    
    # Weighted variance after split
    weighted_var = (n_left * var_left + n_right * var_right) / n_all
    
    # Variance reduction
    return var_all - weighted_var

class SplitRule:
    """
    Abstract template class for a split rule
    """

    @staticmethod
    @abstractmethod
    def get_split_value(available_splitting_values):
        pass

    @staticmethod
    @abstractmethod
    def divide(available_splitting_values, split_value):
        pass


class ContinuousSplitRule(SplitRule):
    """
    Standard continuous split rule: pick a pivot value and split
    depending on if variable is smaller or greater than the value picked.
    """

    @staticmethod
    def get_split_value(available_splitting_values):
        split_value = None
        if available_splitting_values.size > 1:
            idx_selected_splitting_values = int(
                np.random.random() * len(available_splitting_values)
            )
            split_value = available_splitting_values[idx_selected_splitting_values]
        return split_value

    @staticmethod
    @njit
    def divide(available_splitting_values, split_value):
        return available_splitting_values <= split_value


class OneHotSplitRule(SplitRule):
    """Choose a single categorical value and branch on if the variable is that value or not"""

    @staticmethod
    def get_split_value(available_splitting_values):
        split_value = None
        if available_splitting_values.size > 1 and not np.all(
            available_splitting_values == available_splitting_values[0]
        ):
            idx_selected_splitting_values = int(
                np.random.random() * len(available_splitting_values)
            )
            split_value = available_splitting_values[idx_selected_splitting_values]
        return split_value

    @staticmethod
    @njit
    def divide(available_splitting_values, split_value):
        return available_splitting_values == split_value


class SubsetSplitRule(SplitRule):
    """
    Choose a random subset of the categorical values and branch on belonging to that set.
    This is the approach taken by Sameer K. Deshpande.
    flexBART: Flexible Bayesian regression trees with categorical predictors. arXiv,
    `link <https://arxiv.org/abs/2211.04459>`__
    """

    @staticmethod
    def get_split_value(available_splitting_values):
        split_value = None
        if available_splitting_values.size > 1 and not np.all(
            available_splitting_values == available_splitting_values[0]
        ):
            unique_values = np.unique(available_splitting_values)
            while True:
                sample = np.random.randint(0, 2, size=len(unique_values)).astype(bool)
                if np.any(sample):
                    break
            split_value = unique_values[sample]
        return split_value

    @staticmethod
    def divide(available_splitting_values, split_value):
        return np.isin(available_splitting_values, split_value)
