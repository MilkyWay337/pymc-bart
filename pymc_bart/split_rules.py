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
    Target encoding for categorical variables (CatBoost-style: BinarizedTargetMeanValue).
    
    Transforms categorical feature values to numerical using target statistics
    with Bayesian smoothing. 
    
    Formula for Regression:
    avg_target = (countInClass + prior) / (totalCount + 1)
    
    where:
    - countInClass: sum of all target values for current categorical feature value
    - prior: smoothing parameter (default 1.0 in CatBoost)
    - totalCount: count of objects WITH current feature value
    
    Example:
    If category A appears 3 times with targets [0.1, 0.2, 0.3]:
    avg_target_A = (0.1 + 0.2 + 0.3 + 1.0) / (3 + 1) = 1.6 / 4 = 0.4
    
    References:
    -----------
    CatBoost: Transforming categorical features to numerical features
    Type: BinarizedTargetMeanValue
    https://catboost.ai/docs/concepts/algorithm-main-principles_cat-to-num
    """
    
    def __init__(self, prior: float = 1.0, n_buckets: int = 10):
        """
        Parameters
        ----------
        prior : float
            Smoothing parameter (default: 1.0 per CatBoost for regression).
            Prevents overfitting on small categories.
        n_buckets : int
            Number of buckets for quantization (not used in current BART integration)
        """
        self.prior = prior
        self.n_buckets = n_buckets
        self.encoding_map = {}
        self.global_mean = 0.0
        
    def compute_target_encoding(self, categories: np.ndarray, targets: np.ndarray,
                                residuals: Optional[np.ndarray] = None) -> dict:
        """
        Compute target encoding for categorical values using CatBoost formula.
        
        Parameters
        ----------
        categories : np.ndarray
            Categorical values (e.g., [0, 0, 1, 1, 2, 2])
        targets : np.ndarray  
            Target values (e.g., [0.1, 0.2, 0.5, 0.6, 0.9, 1.0])
        residuals : Optional[np.ndarray]
            Residuals from current tree sum (for iterative BART fitting).
            If provided, used instead of targets.
            
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
            
            # SUM of target values for this category (not mean!)
            count_in_class = np.sum(encoding_targets[mask])
            
            # CatBoost formula: (sum_of_targets + prior) / (count + 1)
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
            Target values (for encoding computation)
        residuals : Optional[np.ndarray]  
            Residuals for encoding (takes precedence over targets)
            
        Returns
        -------
        Optional[float]
            Split threshold on encoded values, or None if no split possible
        """
        if targets is None and residuals is None:
            raise ValueError("Either targets or residuals must be provided for TargetSplitRule")
            
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
        
        # Pick random split threshold from unique encoded values
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
            Original categorical values
        split_value : float
            Threshold on encoded values (from get_split_value)
            
        Returns
        -------
        np.ndarray
            Boolean array indicating left branch (encoded_value <= split_value)
        """
        if not self.encoding_map:
            raise ValueError("Encoding map not computed. Call get_split_value first.")
            
        numerical_values = np.array([self.encoding_map[x] for x in available_splitting_values])
        return numerical_values <= split_value


class CounterSplitRule(SplitRule):
    """
    Counter encoding for categorical variables (CatBoost-style: Counter method).
    
    Frequency-based encoding that depends ONLY on feature value frequencies,
    NOT on target values. Useful when target dependency is not desired.
    
    Formula (Training dataset):
    ctr = (curCount + prior) / (maxCount + 1)
    
    where:
    - curCount: frequency of current categorical value in training set
    - maxCount: maximum frequency among all categorical values in training set
    - prior: smoothing parameter (default 1.0)
    
    Example:
    If training set has categories with frequencies: A=50, B=30, C=20
    maxCount = 50
    Encodings: A = (50+1)/(50+1) = 1.0, B = (30+1)/(50+1) = 0.61, C = (20+1)/(50+1) = 0.41
    
    References:
    -----------
    CatBoost: Counter transformation method
    Type: Counter
    https://catboost.ai/docs/concepts/algorithm-main-principles_cat-to-num
    """
    
    def __init__(self, prior: float = 1.0, calculation_method: str = "Full"):
        """
        Parameters
        ----------
        prior : float
            Smoothing parameter (default: 1.0)
        calculation_method : str
            'Full' - use all training + validation data for maxCount and frequencies
            'SkipTest' - use only training data (more conservative)
        """
        self.prior = prior
        self.calculation_method = calculation_method
        self.encoding_map = {}
        self.max_count = 0
        self.training_freqs = {}  # Store frequencies for consistency
        
    def compute_counter_encoding(self, categories: np.ndarray,
                                training_categories: Optional[np.ndarray] = None) -> dict:
        """
        Compute counter encoding based on CatBoost Counter method.
        
        Formula: ctr = (curCount + prior) / (maxCount + 1)
        
        Parameters
        ----------
        categories : np.ndarray
            Current categorical values (e.g., from a tree node)
        training_categories : Optional[np.ndarray]
            Full training set categories for reference counts.
            If None, uses current categories as reference.
            
        Returns
        -------
        dict
            Mapping from category to counter encoded value
        """
        # Determine reference distribution for maxCount
        if training_categories is not None:
            reference_cats = training_categories
        else:
            reference_cats = categories
            
        # Get unique values and their frequencies in reference set
        unique_cats, counts = np.unique(reference_cats, return_counts=True)
        max_count = np.max(counts)
        self.max_count = max_count
        
        # Store training frequencies for consistency
        self.training_freqs = dict(zip(unique_cats, counts))
        
        encoding_map = {}
        
        # For each category in current data
        unique_current = np.unique(categories)
        for cat in unique_current:
            # Get frequency from training/reference set if available
            if cat in self.training_freqs:
                cat_count = self.training_freqs[cat]
            else:
                # If category doesn't exist in training, use current count
                cat_count = np.sum(categories == cat)
                
            # CatBoost Counter formula: (count + prior) / (maxCount + 1)
            encoded_value = (cat_count + self.prior) / (max_count + 1)
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
            Full training set categories (for computing maxCount)
            
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
        
        # Pick random split threshold from unique encoded values
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
            Original categorical values
        split_value : float
            Threshold on encoded values (from get_split_value)
            
        Returns
        -------
        np.ndarray
            Boolean array indicating left branch (encoded_value <= split_value)
        """
        if not self.encoding_map:
            raise ValueError("Encoding map not computed. Call get_split_value first.")
            
        numerical_values = np.array([self.encoding_map[x] for x in available_splitting_values])
        return numerical_values <= split_value
