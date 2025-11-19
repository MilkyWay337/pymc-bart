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
    def get_split_value(available_splitting_values, y=None):
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
    def get_split_value(available_splitting_values, y=None):
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
    def get_split_value(available_splitting_values, y=None):
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
    def get_split_value(available_splitting_values, y=None):
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
    Choose a split based on ordering categorical values by their target mean.
    Modified to be fully compatible with OneHotSplitRule and SubsetSplitRule.
    """

    @staticmethod
    def get_split_value(available_splitting_values, y=None):
        """
        Returns a subset of categories forming the left branch.
        """
        if y is None:
            # Fall back to random subset splitting if no y provided
            return SubsetSplitRule.get_split_value(available_splitting_values)
        
        # Check input validity
        if len(y) != len(available_splitting_values):
            raise ValueError("y must have the same length as available_splitting_values")
        
        # Unique categories
        cats = np.unique(available_splitting_values)

        # Mean target per category
        means = []
        for cat in cats:
            mask = available_splitting_values == cat
            if np.sum(mask) > 0:
                cat_mean = y[mask].mean()
                means.append(cat_mean)
        
        if len(cats) <= 1:
            return None
            
        means = np.array(means)

        # Order categories by target mean
        order = np.argsort(means)
        ordered = cats[order]

        # Choose one prefix split at random
        if len(ordered) <= 1:
            return None

        # Randomly choose split point (prefix)
        k = np.random.randint(1, len(ordered))
        split_value = ordered[:k]

        return split_value

    @staticmethod
    def divide(available_splitting_values, split_value):
        """
        Return boolean array indicating left branch membership for each element.
        Fully compatible with OneHotSplitRule and SubsetSplitRule.
        """
        # Используем ту же логику, что и в SubsetSplitRule
        return np.isin(available_splitting_values, split_value)
