import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from functools import partial


def continuous_split_rule(X, y, feature, min_samples_leaf=1, **kwargs):
    """
    Original continuous split rule
    """
    feature_values = X[:, feature]
    sorted_indices = jnp.argsort(feature_values)
    sorted_features = feature_values[sorted_indices]
    sorted_y = y[sorted_indices]
    
    n = len(sorted_features)
    best_gain = -jnp.inf
    best_split_value = None
    
    for i in range(min_samples_leaf, n - min_samples_leaf):
        if sorted_features[i] == sorted_features[i + 1]:
            continue
            
        left_y = sorted_y[:i + 1]
        right_y = sorted_y[i + 1:]
        
        var_left = jnp.var(left_y) if len(left_y) > 1 else 0.0
        var_right = jnp.var(right_y) if len(right_y) > 1 else 0.0
        var_total = jnp.var(sorted_y)
        
        n_left, n_right = len(left_y), len(right_y)
        weighted_var = (n_left * var_left + n_right * var_right) / n
        
        gain = var_total - weighted_var
        
        if gain > best_gain:
            best_gain = gain
            best_split_value = (sorted_features[i] + sorted_features[i + 1]) / 2
    
    if best_split_value is not None:
        return best_split_value, best_gain
    return None


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
    
    # Skip if not enough unique values or samples
    if len(unique_vals) <= 1 or len(y) < 2 * min_samples_leaf:
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
        
        stats[category.item()] = {
            'mean': smoothed_mean,
            'n': n_category,
            'raw_mean': category_mean
        }
        
    return stats


def _order_categories_by_target(category_stats):
    """Order categories by their target mean."""
    categories = list(category_stats.keys())
    means = jnp.array([category_stats[cat]['mean'] for cat in categories])
    
    # Sort categories by mean
    sorted_indices = jnp.argsort(means)
    return [categories[i] for i in sorted_indices]


def _find_best_binary_split(feature_values, y, ordered_categories, 
                          category_stats, min_samples_leaf):
    """Find best binary split among ordered categories."""
    n_categories = len(ordered_categories)
    best_gain = -jnp.inf
    best_split_value = None
    
    for split_idx in range(1, n_categories):
        left_categories = ordered_categories[:split_idx]
        right_categories = ordered_categories[split_idx:]
        
        left_mask = jnp.zeros_like(feature_values, dtype=bool)
        right_mask = jnp.zeros_like(feature_values, dtype=bool)
        
        for cat in left_categories:
            left_mask = left_mask | (feature_values == cat)
        for cat in right_categories:
            right_mask = right_mask | (feature_values == cat)
        
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
            best_split_value = (last_left + first_right) / 2.0
    
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
