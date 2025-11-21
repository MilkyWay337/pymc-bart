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

import numpy as np
from numba import njit


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
# pymc_bart/split_rules.py

from abc import abstractmethod
import numpy as np
from numba import njit

class TargetSplitRule(SplitRule):
    """
    Target-based split rule for categorical features inspired by CatBoost.
    Transforms categorical features into numerical values based on target statistics.
    """

    @staticmethod
    def get_split_value(available_splitting_values, y=None, node_indices=None, alpha=1.0):
        """
        Calculate target statistics and pick a split value.
        
        Parameters
        ----------
        available_splitting_values : np.ndarray
            Categorical feature values for current node
        y : np.ndarray, optional
            Target values for all observations
        node_indices : np.ndarray, optional  
            Indices of observations in current node
        alpha : float
            Smoothing parameter for target statistics
            
        Returns
        -------
        split_value : float or None
            The split threshold for transformed values
        """
        if available_splitting_values.size <= 1:
            return None
            
        if y is None or node_indices is None:
            # Fall back to random selection if target info not available
            idx = int(np.random.random() * len(available_splitting_values))
            return available_splitting_values[idx]
        
        # Get data for current node
        x_node = available_splitting_values
        y_node = y[node_indices]
        
        # Calculate target statistics for each unique category
        unique_cats = np.unique(x_node)
        if len(unique_cats) < 2:
            return None
            
        # Transform categorical values to target statistics
        transformed_values = np.zeros(len(x_node), dtype=float)
        global_mean = np.mean(y_node)
        
        for cat in unique_cats:
            cat_mask = (x_node == cat)
            cat_count = np.sum(cat_mask)
            
            if cat_count > 0:
                cat_target_mean = np.mean(y_node[cat_mask])
                # Apply smoothing (CatBoost style)
                smoothed_value = (cat_count * cat_target_mean + alpha * global_mean) / (cat_count + alpha)
                transformed_values[cat_mask] = smoothed_value
        
        # Now use continuous split logic on transformed values
        if len(np.unique(transformed_values)) < 2:
            return None
            
        # Pick a random split value from unique transformed values
        unique_transformed = np.unique(transformed_values)
        if len(unique_transformed) > 1:
            # Avoid endpoints to ensure proper split
            idx = int(np.random.random() * (len(unique_transformed) - 1))
            split_value = (unique_transformed[idx] + unique_transformed[idx + 1]) / 2.0
            return split_value
            
        return None

    @staticmethod
    @njit
    def divide(available_splitting_values, split_value, y=None, node_indices=None, alpha=1.0):
        """
        Split based on target-transformed values.
        
        Parameters
        ----------
        available_splitting_values : np.ndarray
            Categorical feature values
        split_value : float
            Threshold for transformed values
        y : np.ndarray, optional
            Target values  
        node_indices : np.ndarray, optional
            Indices of observations in node
        alpha : float
            Smoothing parameter
            
        Returns
        -------
        mask : np.ndarray
            Boolean mask for left/right split
        """
        if y is None or node_indices is None:
            # Fallback: use random split
            return available_splitting_values == available_splitting_values[0]
        
        # Transform categorical values using same logic as get_split_value
        x_node = available_splitting_values
        y_node = y[node_indices]
        unique_cats = np.unique(x_node)
        transformed_values = np.zeros(len(x_node), dtype=np.float64)
        global_mean = np.mean(y_node)
        
        for cat in unique_cats:
            cat_mask = x_node == cat
            cat_count = np.sum(cat_mask)
            
            if cat_count > 0:
                # Calculate mean for this category
                cat_sum = 0.0
                count = 0
                for i in range(len(x_node)):
                    if x_node[i] == cat:
                        cat_sum += y_node[i]
                        count += 1
                if count > 0:
                    cat_target_mean = cat_sum / count
                    smoothed_value = (count * cat_target_mean + alpha * global_mean) / (count + alpha)
                    for i in range(len(x_node)):
                        if x_node[i] == cat:
                            transformed_values[i] = smoothed_value
        
        # Split based on transformed values
        return transformed_values <= split_value
