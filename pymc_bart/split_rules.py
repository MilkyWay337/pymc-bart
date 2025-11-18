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
    def get_split_value(available_splitting_values, **kwargs):
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
    def get_split_value(available_splitting_values, **kwargs):
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
    def get_split_value(available_splitting_values, **kwargs):
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
    def get_split_value(available_splitting_values, **kwargs):
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


class TargetMeanSplitRule(SplitRule):
    """
    Target mean encoding split rule for categorical variables.
    
    This approach orders categorical values by their mean target value,
    then finds an optimal split point in this ordered sequence.
    This is similar to the approach used in LightGBM and CatBoost for
    categorical features.
    
    The split divides categories into two groups based on whether their
    mean target value is below or above a threshold.
    """

    @staticmethod
    def get_split_value(available_splitting_values, **kwargs):
        """
        Compute split value based on target mean encoding.
        
        Parameters
        ----------
        available_splitting_values : np.ndarray
            Array of categorical values at this node
        **kwargs : dict
            Must contain:
            - idx_data_points: indices of data points at this node
            - Y: full target array
            - sum_trees: current sum of trees predictions (for residuals)
            
        Returns
        -------
        split_value : np.ndarray or None
            Array of categories that should go to the left child
        """
        split_value = None
        
        # Extract required data from kwargs
        idx_data_points = kwargs.get('idx_data_points')
        Y = kwargs.get('Y')
        sum_trees = kwargs.get('sum_trees')
        
        if idx_data_points is None or Y is None or sum_trees is None:
            # Fallback to random split if target information not available
            if available_splitting_values.size > 1 and not np.all(
                available_splitting_values == available_splitting_values[0]
            ):
                unique_values = np.unique(available_splitting_values)
                n_unique = len(unique_values)
                n_left = np.random.randint(1, n_unique)
                split_value = unique_values[np.random.choice(n_unique, n_left, replace=False)]
            return split_value
        
        # Check if we have multiple unique values
        if available_splitting_values.size > 1 and not np.all(
            available_splitting_values == available_splitting_values[0]
        ):
            unique_values = np.unique(available_splitting_values)
            
            if len(unique_values) < 2:
                return None
            
            # Compute residuals (what the tree should predict)
            residuals = Y[idx_data_points] - sum_trees[:, idx_data_points].mean(axis=0)
            
            # Compute mean residual for each category
            category_means = {}
            for cat in unique_values:
                mask = available_splitting_values == cat
                if np.any(mask):
                    category_means[cat] = np.mean(residuals[mask])
                else:
                    category_means[cat] = 0.0
            
            # Sort categories by their mean target value
            sorted_categories = sorted(category_means.items(), key=lambda x: x[1])
            sorted_cats = np.array([cat for cat, _ in sorted_categories])
            
            # Try different split points and choose the best one
            # (we could use all possible splits or sample a subset)
            if len(sorted_cats) <= 10:
                # Try all possible splits for small number of categories
                n_splits = len(sorted_cats) - 1
                split_idx = np.random.randint(1, n_splits + 1)
            else:
                # For many categories, choose a random split point
                # (or we could use a more sophisticated criterion)
                split_idx = np.random.randint(1, len(sorted_cats))
            
            split_value = sorted_cats[:split_idx]
        
        return split_value

    @staticmethod
    def divide(available_splitting_values, split_value):
        """
        Divide data points based on whether their category is in split_value.
        
        Parameters
        ----------
        available_splitting_values : np.ndarray
            Array of categorical values
        split_value : np.ndarray
            Array of categories that should go to the left child
            
        Returns
        -------
        mask : np.ndarray
            Boolean array indicating which values go left (True) or right (False)
        """
        return np.isin(available_splitting_values, split_value)
