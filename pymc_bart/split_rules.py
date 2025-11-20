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
from typing import Dict, Optional

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


class TargetSplitRule(SplitRule):
    """CatBoost-style target encoding split rule for categorical features.

    This rule transforms categorical features into numerical ones using the
    formula:
        ctr = (countInClass + prior) / (totalCount + 1)

    It supports regression, classification, and multi-classification scenarios.
    """

    def __init__(self, prior: float = 1.0) -> None:
        self.prior = prior
        self.encoding_map: Dict[object, float] = {}

    def compute_target_encoding(
        self,
        categories: np.ndarray,
        targets: np.ndarray,
        residuals: Optional[np.ndarray] = None,
    ) -> Dict[object, float]:
        """
        Compute target encoding for categorical values using the CatBoost formula.

        Parameters
        ----------
        categories : np.ndarray
            Categorical values (e.g., ["red", "green", "blue"]).
        targets : np.ndarray
            Target values (e.g., [0.1, 0.5, 0.9]).
        residuals : Optional[np.ndarray]
            Residuals from current tree sum (for iterative BART fitting).
            If provided, used instead of targets.

        Returns
        -------
        Dict[object, float]
            Mapping from category to encoded numerical value.
        """
        encoding_targets = residuals if residuals is not None else targets
        unique = np.unique(categories)
        encoding: Dict[object, float] = {}
        for u in unique:
            mask = categories == u
            count_in_class = np.sum(encoding_targets[mask])
            total_count = np.sum(mask)
            encoding[u] = (count_in_class + self.prior) / (total_count + 1)
        return encoding

    def get_split_value(
        self,
        available_splitting_values: np.ndarray,
        targets: Optional[np.ndarray] = None,
        residuals: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Optional[float]:
        """
        Get split value using target encoding.

        Parameters
        ----------
        available_splitting_values : np.ndarray
            Categorical values for current split.
        targets : Optional[np.ndarray]
            Target values (for encoding computation).
        residuals : Optional[np.ndarray]
            Residuals for encoding (takes precedence over targets).

        Returns
        -------
        Optional[float]
            Split threshold on encoded values, or None if no split possible.
        """
        if targets is None and residuals is None:
            raise ValueError("Either 'targets' or 'residuals' must be provided for TargetSplitRule")
        if available_splitting_values.size <= 1:
            return None

        self.encoding_map = self.compute_target_encoding(
            available_splitting_values, targets if targets is not None else residuals, residuals
        )
        numerical_values = np.array([self.encoding_map[x] for x in available_splitting_values])
        unique_vals = np.unique(numerical_values)
        if unique_vals.size <= 1:
            return None
        idx = int(np.random.random() * len(unique_vals))
        return unique_vals[idx]

    def divide(self, available_splitting_values: np.ndarray, split_value: float) -> np.ndarray:
        """
        Divide data based on target encoded values.

        Parameters
        ----------
        available_splitting_values : np.ndarray
            Original categorical values.
        split_value : float
            Threshold on encoded values (from get_split_value).

        Returns
        -------
        np.ndarray
            Boolean array indicating left branch (encoded_value <= split_value).
        """
        if not self.encoding_map:
            raise ValueError("Encoding map not computed. Call get_split_value first.")
        numerical_values = np.array([self.encoding_map[x] for x in available_splitting_values])
        return numerical_values <= split_value
