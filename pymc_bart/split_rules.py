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


class TargetMeanSplitRule(SplitRule):
    """
    Split rule for categorical variables that chooses the split subset 
    that maximizes the difference in target means between the two resulting groups.
    
    This rule uses target variable information to find the most predictive splits
    for categorical features by evaluating different category subsets.
    """

    @staticmethod
    def get_split_value(available_splitting_values, target_values=None):
        """
        For categorical variables, find the subset of categories that maximizes
        the difference in target means between the subset and its complement.
        
        Parameters
        ----------
        available_splitting_values : np.ndarray
            Categorical values for splitting
        target_values : np.ndarray
            Target values corresponding to the splitting values
        """
        split_value = None
        
        if (available_splitting_values.size > 1 and 
            target_values is not None and 
            target_values.size > 0):
            
            # Remove NaN values
            valid_mask = ~np.isnan(available_splitting_values)
            if np.sum(valid_mask) <= 1:
                return None
                
            available_splitting_values = available_splitting_values[valid_mask]
            target_values = target_values[valid_mask]
            
            # Get unique categorical values
            unique_categories = np.unique(available_splitting_values)
            
            if len(unique_categories) <= 1:
                return None
            
            # For categorical variables, find the best subset of categories
            # that maximizes the absolute difference in target means
            
            best_gain = -np.inf
            best_subset = None
            
            # If there are too many categories, use efficient search
            max_categories_for_exhaustive = 8
            if len(unique_categories) <= max_categories_for_exhaustive:
                # Exhaustive search for small number of categories
                # Generate all non-empty proper subsets (skip empty and full sets)
                n_categories = len(unique_categories)
                for i in range(1, 2 ** n_categories - 1):
                    # Create subset mask using binary representation
                    subset_mask = np.array([(i >> j) & 1 for j in range(n_categories)], dtype=bool)
                    current_subset = unique_categories[subset_mask]
                    
                    subset_mask_data = np.isin(available_splitting_values, current_subset)
                    
                    # Ensure both subsets have at least one element
                    if np.sum(subset_mask_data) > 0 and np.sum(~subset_mask_data) > 0:
                        left_mean = np.mean(target_values[subset_mask_data])
                        right_mean = np.mean(target_values[~subset_mask_data])
                        gain = abs(left_mean - right_mean)
                        
                        if gain > best_gain:
                            best_gain = gain
                            best_subset = current_subset
            else:
                # For many categories, use heuristic search
                # Try single categories first (equivalent to OneHot)
                for cat in unique_categories:
                    subset_mask = available_splitting_values == cat
                    if np.sum(subset_mask) > 0 and np.sum(~subset_mask) > 0:
                        left_mean = np.mean(target_values[subset_mask])
                        right_mean = np.mean(target_values[~subset_mask])
                        gain = abs(left_mean - right_mean)
                        
                        if gain > best_gain:
                            best_gain = gain
                            best_subset = np.array([cat])
                
                # Try random subsets
                n_random_tries = min(50, 2 ** (len(unique_categories) - 1))
                for _ in range(n_random_tries):
                    # Random subset size between 1 and n_categories-1
                    subset_size = np.random.randint(1, len(unique_categories))
                    random_subset = np.random.choice(
                        unique_categories, size=subset_size, replace=False
                    )
                    
                    subset_mask = np.isin(available_splitting_values, random_subset)
                    if np.sum(subset_mask) > 0 and np.sum(~subset_mask) > 0:
                        left_mean = np.mean(target_values[subset_mask])
                        right_mean = np.mean(target_values[~subset_mask])
                        gain = abs(left_mean - right_mean)
                        
                        if gain > best_gain:
                            best_gain = gain
                            best_subset = random_subset
            
            split_value = best_subset
        
        return split_value

    @staticmethod
    def divide(available_splitting_values, split_value):
        """
        For categorical TargetMeanSplitRule, split_value is a subset of categories
        that go to the left branch.
        """
        if split_value is None:
            return np.zeros_like(available_splitting_values, dtype=bool)
        return np.isin(available_splitting_values, split_value)
