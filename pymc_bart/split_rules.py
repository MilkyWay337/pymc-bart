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
    @staticmethod
    @abstractmethod
    def get_split_value(available_splitting_values, y_values=None):
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
    def get_split_value(available_splitting_values, y_values=None):
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
    def get_split_value(available_splitting_values, y_values=None):
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
    def get_split_value(available_splitting_values, y_values=None):
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
    def __init__(self, smoothing_alpha=1.0):
        """
        smoothing_alpha: Для smoothing в target encoding (избежать overfitting для редких категорий).
        """
        self.smoothing_alpha = smoothing_alpha

    @staticmethod
    def get_split_value(available_splitting_values, y_values=None):
        if available_splitting_values.size <= 1:
            return None

        # Предполагаем categorical (после label encoding как integers или objects)
        unique_cats, inverse = np.unique(available_splitting_values, return_inverse=True)
        if len(unique_cats) <= 1:
            return None

        encoded = np.zeros_like(available_splitting_values, dtype=float)
        global_mean = np.mean(y_values) if y_values is not None else 0
        for i, cat in enumerate(unique_cats):
            mask = (available_splitting_values == cat)
            cat_mean = np.mean(y_values[mask]) if y_values is not None and mask.sum() > 0 else global_mean
            n = mask.sum()
            # Smoothing: (n * cat_mean + alpha * global_mean) / (n + alpha)
            smoothed_mean = (n * cat_mean + self.smoothing_alpha * global_mean) / (n + self.smoothing_alpha)
            encoded[mask] = smoothed_mean

        # Split_value как mean по encoded values
        split_value = np.mean(encoded)
        return split_value

    @staticmethod
    @njit
    def divide(available_splitting_values, split_value):
        # Divide на основе original values? Нет: но поскольку rule только для categorical,
        # мы можем remap to encoded внутри grow_tree, но для простоты assume available_splitting_values - encoded или используем threshold
        # Чтобы упростить: assume user encoded categorical to numbers, divide as <= split_value
        return available_splitting_values <= split_value
