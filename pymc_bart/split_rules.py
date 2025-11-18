# В файле split_rules.py (исправленная версия)

from abc import abstractmethod

import numpy as np
from numba import njit


class SplitRule:
    """
    Abstract template class for a split rule
    """

    @abstractmethod
    def get_split_value(self, available_splitting_values, y_values=None):
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

    def get_split_value(self, available_splitting_values, y_values=None):
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

    def get_split_value(self, available_splitting_values, y_values=None):
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

    def get_split_value(self, available_splitting_values, y_values=None):
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
    Упрощённый split rule на основе target encoding только для categorical признаков.
    Кодирует категории по mean Y (с smoothing), затем split_value = mean по encoded.
    Divide: encoded <= split_value vs. > (эффективно разделяет группы категорий).
    """
    

    def get_split_value(self, available_splitting_values, y_values=None):
        if available_splitting_values.size <= 1:
            return None

        unique_cats, inverse = np.unique(available_splitting_values, return_inverse=True)
        if len(unique_cats) <= 1:
            return None

        encoded = np.zeros_like(available_splitting_values, dtype=float)
        global_mean = np.mean(y_values) if y_values is not None else 0
        for i, cat in enumerate(unique_cats):
            mask = (available_splitting_values == cat)
            cat_mean = np.mean(y_values[mask]) if y_values is not None and mask.sum() > 0 else global_mean
            n = mask.sum()
            smoothed_mean = (n * cat_mean + self.smoothing_alpha * global_mean) / (n + self.smoothing_alpha)
            encoded[mask] = smoothed_mean

        split_value = np.mean(encoded)
        return split_value

    @staticmethod
    @njit
    def divide(available_splitting_values, split_value):
        return available_splitting_values <= split_value
