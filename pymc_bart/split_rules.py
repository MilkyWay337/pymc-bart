#   Copyright 2022 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
"""Split rule implementations used by the tree-growing routines.

This module provides a minimal, well-typed implementation of split rules
used by BART: Continuous, One-Hot, Subset (categorical subset splits),
and a CatBoost-style Target encoding split rule.

Only these four rules are exported from the package.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


class SplitRule:
    """Base interface for split rules."""

    def get_split_value(self, available_splitting_values: np.ndarray, **kwargs) -> Optional[np.ndarray]:
        raise NotImplementedError

    def divide(self, available_splitting_values: np.ndarray, split_value: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ContinuousSplitRule(SplitRule):
    """Split rule for continuous variables.

    Chooses a split threshold among unique values and splits by <= threshold.
    """

    def get_split_value(self, available_splitting_values: np.ndarray, **kwargs) -> Optional[float]:
        vals = np.unique(available_splitting_values)
        if vals.size <= 1:
            return None
        # choose random threshold among unique values (not the max)
        idx = int(np.random.random() * (len(vals) - 1))
        # pick threshold between vals[idx] and vals[idx+1]
        return 0.5 * (vals[idx] + vals[idx + 1])

    def divide(self, available_splitting_values: np.ndarray, split_value: float) -> np.ndarray:
        return available_splitting_values <= split_value


class OneHotSplitRule(SplitRule):
    """Split rule for one-hot / binary categorical variables.

    Chooses one of the categories as left branch (equality test).
    """

    def get_split_value(self, available_splitting_values: np.ndarray, **kwargs) -> Optional[object]:
        vals = np.unique(available_splitting_values)
        if vals.size <= 1:
            return None
        # choose a category value to split on
        idx = int(np.random.random() * len(vals))
        return vals[idx]

    def divide(self, available_splitting_values: np.ndarray, split_value: object) -> np.ndarray:
        return available_splitting_values == split_value


class SubsetSplitRule(SplitRule):
    """Split rule that divides categories by membership in a random subset."""

    def get_split_value(self, available_splitting_values: np.ndarray, **kwargs) -> Optional[np.ndarray]:
        vals = np.unique(available_splitting_values)
        if vals.size <= 1:
            return None
        # build a boolean mask representing the left subset
        # ensure neither side is empty: pick random subset size between 1 and len(vals)-1
        k = np.random.randint(1, len(vals))
        chosen = np.random.choice(vals, size=k, replace=False)
        return chosen

    def divide(self, available_splitting_values: np.ndarray, split_value: Sequence) -> np.ndarray:
        chosen = set(split_value)
        return np.array([x in chosen for x in available_splitting_values])


class TargetSplitRule(SplitRule):
    """CatBoost-style target encoding split rule for categorical features.

    The encoding used is a smoothed target mean per category:
        enc(c) = (sum_targets[c] + prior) / (count[c] + 1)

    During tree growth the algorithm must pass either `targets` or `residuals`
    as a keyword argument to `get_split_value` so the encoding can be computed.
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
        encoding_targets = residuals if residuals is not None else targets
        unique = np.unique(categories)
        encoding: Dict[object, float] = {}
        for u in unique:
            mask = categories == u
            cnt = np.sum(mask)
            if cnt == 0:
                encoding[u] = 0.0
            else:
                s = np.sum(encoding_targets[mask])
                encoding[u] = (s + self.prior) / (cnt + 1)
        return encoding

    def get_split_value(
        self,
        available_splitting_values: np.ndarray,
        targets: Optional[np.ndarray] = None,
        residuals: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Optional[float]:
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
        if not self.encoding_map:
            raise ValueError("Encoding map not computed. Call get_split_value first.")
        numerical_values = np.array([self.encoding_map[x] for x in available_splitting_values])
        return numerical_values <= split_value
        
