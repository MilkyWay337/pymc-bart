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
from typing import Optional

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
    """
    Target encoding for categorical variables (CatBoost-style).
    
    Transforms categorical feature values to numerical using target statistics
    with Bayesian smoothing. Formula:
    
    avg_target = (countInClass + prior) / (totalCount + 1)
    
    where:
    - countInClass: sum of target values for current categorical feature value
    - prior: smoothing parameter
    - totalCount: count of objects with current feature value
    
    References:
    -----------
    CatBoost: Transforming categorical features to numerical features
    https://catboost.ai/docs/concepts/algorithm-main-principles_cat-to-num
    """
    
    def __init__(self, prior: float = 1.0, n_buckets: int = 10):
        """
        Parameters
        ----------
        prior : float
            Smoothing parameter for target encoding (default: 1.0)
        n_buckets : int
            Number of buckets for bucketing mode (default: 10)
        """
        self.prior = prior
        self.n_buckets = n_buckets
        self.encoding_map = {}
        self.global_mean = 0.0
        
    def compute_target_encoding(self, categories: np.ndarray, targets: np.ndarray,
                                residuals: Optional[np.ndarray] = None) -> dict:
        """
        Compute target encoding for categorical values with smoothing.
        
        Parameters
        ----------
        categories : np.ndarray
            Categorical values
        targets : np.ndarray  
            Target values (y)
        residuals : Optional[np.ndarray]
            Residuals from current tree sum (for iterative fitting)
            
        Returns
        -------
        dict
            Mapping from category to encoded numerical value
        """
        # Use residuals if available (for BART iterative fitting), otherwise use targets
        encoding_targets = residuals if residuals is not None else targets
        
        self.global_mean = np.mean(encoding_targets)
        unique_cats = np.unique(categories)
        encoding_map = {}
        
        for cat in unique_cats:
            mask = categories == cat
            total_count = np.sum(mask)
            
            # Sum of target values for this category
            count_in_class = np.sum(encoding_targets[mask])
            
            # Apply smoothing formula: (countInClass + prior) / (totalCount + 1)
            encoded_value = (count_in_class + self.prior) / (total_count + 1)
            encoding_map[cat] = encoded_value
            
        return encoding_map
    
    def get_split_value(self, available_splitting_values: np.ndarray,
                       targets: Optional[np.ndarray] = None,
                       residuals: Optional[np.ndarray] = None) -> Optional[float]:
        """
        Get split value using target encoding.
        
        Parameters
        ----------
        available_splitting_values : np.ndarray
            Categorical values for current split
        targets : Optional[np.ndarray]
            Target values
        residuals : Optional[np.ndarray]  
            Residuals for encoding
            
        Returns
        -------
        Optional[float]
            Split threshold on encoded values, or None if no split possible
        """
        if targets is None and residuals is None:
            raise ValueError("Either targets or residuals must be provided for target encoding")
            
        if available_splitting_values.size <= 1:
            return None
            
        # Compute target encoding
        self.encoding_map = self.compute_target_encoding(
            available_splitting_values, 
            targets if targets is not None else residuals,
            residuals
        )
        
        # Convert categorical to numerical using encoding
        numerical_values = np.array([self.encoding_map[x] for x in available_splitting_values])
        unique_values = np.unique(numerical_values)
        
        # Use continuous split on encoded values
        if len(unique_values) > 1:
            idx = int(np.random.random() * len(unique_values))
            return unique_values[idx]
        
        return None

    def divide(self, available_splitting_values: np.ndarray, split_value: float) -> np.ndarray:
        """
        Divide data based on target encoded values.
        
        Parameters
        ----------
        available_splitting_values : np.ndarray
            Categorical values
        split_value : float
            Threshold on encoded values
            
        Returns
        -------
        np.ndarray
            Boolean array indicating left branch (encoded value <= split_value)
        """
        if not self.encoding_map:
            raise ValueError("Encoding map not computed. Call get_split_value first.")
            
        numerical_values = np.array([self.encoding_map[x] for x in available_splitting_values])
        return numerical_values <= split_value


class CounterSplitRule(SplitRule):
    """
    Counter encoding for categorical variables (CatBoost-style).
    
    Frequency-based encoding that doesn't depend on target values.
    Formula:
    
    ctr = (curCount + prior) / (maxCount + 1)
    
    where:
    - curCount: frequency of current categorical value
    - maxCount: maximum frequency across all categories
    - prior: smoothing parameter
    
    References:
    -----------
    CatBoost: Counter transformation method
    https://catboost.ai/docs/concepts/algorithm-main-principles_cat-to-num
    """
    
    def __init__(self, prior: float = 1.0, calculation_method: str = "Full"):
        """
        Parameters
        ----------
        prior : float
            Smoothing parameter (default: 1.0)
        calculation_method : str
            'Full' - use all training + test data for encoding
            'SkipTest' - use only training data
        """
        self.prior = prior
        self.calculation_method = calculation_method
        self.encoding_map = {}
        self.max_count = 0
        
    def compute_counter_encoding(self, categories: np.ndarray,
                                training_categories: Optional[np.ndarray] = None) -> dict:
        """
        Compute counter encoding based on frequencies.
        
        Parameters
        ----------
        categories : np.ndarray
            Current categorical values
        training_categories : Optional[np.ndarray]
            Full training set categories for consistent encoding
            
        Returns
        -------
        dict
            Mapping from category to counter encoded value
        """
        if training_categories is not None:
            # Use full training set for consistent encoding
            all_cats = training_categories
        else:
            all_cats = categories
            
        unique_cats, counts = np.unique(all_cats, return_counts=True)
        self.max_count = np.max(counts)
        
        encoding_map = {}
        
        # Count occurrences in current set
        unique_current, counts_current = np.unique(categories, return_counts=True)
        current_counts = dict(zip(unique_current, counts_current))
        
        for cat in unique_cats:
            if self.calculation_method == "Full":
                # Use total count from training set + current count
                total_cat_count = counts[unique_cats == cat][0] if cat in unique_cats else 0
            else:  # SkipTest
                # Use only current count (from current node)
                total_cat_count = current_counts.get(cat, 0)
                
            # Apply formula: (curCount + prior) / (maxCount + 1)
            encoded_value = (total_cat_count + self.prior) / (self.max_count + 1)
            encoding_map[cat] = encoded_value
            
        return encoding_map
    
    def get_split_value(self, available_splitting_values: np.ndarray,
                       training_categories: Optional[np.ndarray] = None) -> Optional[float]:
        """
        Get split value using counter encoding.
        
        Parameters
        ----------
        available_splitting_values : np.ndarray
            Categorical values for current split
        training_categories : Optional[np.ndarray]
            Full training set categories for consistent encoding
            
        Returns
        -------
        Optional[float]
            Split threshold on encoded values, or None if no split possible
        """
        if available_splitting_values.size <= 1:
            return None
            
        self.encoding_map = self.compute_counter_encoding(
            available_splitting_values, 
            training_categories
        )
        
        numerical_values = np.array([self.encoding_map[x] for x in available_splitting_values])
        unique_values = np.unique(numerical_values)
        
        if len(unique_values) > 1:
            idx = int(np.random.random() * len(unique_values))
            return unique_values[idx]
        
        return None

    def divide(self, available_splitting_values: np.ndarray, split_value: float) -> np.ndarray:
        """
        Divide data based on counter encoded values.
        
        Parameters
        ----------
        available_splitting_values : np.ndarray
            Categorical values
        split_value : float
            Threshold on encoded values
            
        Returns
        -------
        np.ndarray
            Boolean array indicating left branch (encoded value <= split_value)
        """
        if not self.encoding_map:
            raise ValueError("Encoding map not computed. Call get_split_value first.")
            
        numerical_values = np.array([self.encoding_map[x] for x in available_splitting_values])
        return numerical_values <= split_value
