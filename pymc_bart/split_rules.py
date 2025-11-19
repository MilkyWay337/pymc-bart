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


class TargetEncodingSplitRule(SplitRule):
    """
    CatBoost-style target encoding for categorical variables.
    
    Transforms categorical values to numerical using target statistics with smoothing.
    """
    
    def __init__(self, prior: float = 1.0, noise_level: float = 0.01, 
                 target_type: str = "regression", n_buckets: int = 10):
        self.prior = prior
        self.noise_level = noise_level
        self.target_type = target_type
        self.n_buckets = n_buckets
        self.encoding_map = {}
        self.global_mean = 0.0
        
    def compute_target_stats(self, categories: np.ndarray, targets: np.ndarray, 
                           residuals: Optional[np.ndarray] = None) -> dict:
        """
        Compute target statistics for each category with smoothing.
        
        Parameters
        ----------
        categories : np.ndarray
            Categorical values
        targets : np.ndarray  
            Target values
        residuals : Optional[np.ndarray]
            Current residuals for adaptive encoding
            
        Returns
        -------
        dict
            Mapping from category to encoded value
        """
        if residuals is not None:
            # Use residuals for encoding (BART-specific)
            encoding_targets = residuals
        else:
            encoding_targets = targets
            
        self.global_mean = np.mean(encoding_targets)
        unique_cats = np.unique(categories)
        encoding_map = {}
        
        for cat in unique_cats:
            mask = categories == cat
            count = np.sum(mask)
            
            if self.target_type == "regression":
                # For regression: mean target value with smoothing
                cat_sum = np.sum(encoding_targets[mask])
                encoded_value = (cat_sum + self.prior * self.global_mean) / (count + self.prior)
                
            elif self.target_type == "binary":
                # For binary classification: probability with smoothing
                cat_sum = np.sum(encoding_targets[mask])
                encoded_value = (cat_sum + self.prior) / (count + 2 * self.prior)
                
            elif self.target_type == "multiclass":
                # For multiclass: one-vs-all encoding (simplified)
                cat_sum = np.sum(encoding_targets[mask])
                encoded_value = (cat_sum + self.prior) / (count + self.prior * len(np.unique(targets)))
            
            # Add noise to prevent overfitting
            if self.noise_level > 0:
                noise = np.random.normal(0, self.noise_level * np.abs(encoded_value))
                encoded_value += noise
                
            encoding_map[cat] = encoded_value
            
        return encoding_map
    
    def get_split_value(self, available_splitting_values, targets: Optional[np.ndarray] = None,
                       residuals: Optional[np.ndarray] = None) -> Optional[float]:
        """
        Get split value using target encoding.
        
        Parameters
        ----------
        available_splitting_values : np.ndarray
            Categorical values
        targets : Optional[np.ndarray]
            Target values for encoding
        residuals : Optional[np.ndarray]  
            Current residuals for encoding
            
        Returns
        -------
        Optional[float]
            Split threshold on encoded values
        """
        if targets is None and residuals is None:
            raise ValueError("Either targets or residuals must be provided for target encoding")
            
        if available_splitting_values.size <= 1:
            return None
            
        # Compute target encoding
        self.encoding_map = self.compute_target_stats(
            available_splitting_values, 
            targets if targets is not None else residuals,
            residuals
        )
        
        # Convert categorical to numerical using encoding
        numerical_values = np.array([self.encoding_map[x] for x in available_splitting_values])
        
        # Use continuous split rule on encoded values
        if len(np.unique(numerical_values)) > 1:
            idx = int(np.random.random() * len(numerical_values))
            return numerical_values[idx]
        
        return None

    def divide(self, available_splitting_values, split_value: float) -> np.ndarray:
        """
        Divide data based on encoded values.
        
        Parameters
        ----------
        available_splitting_values : np.ndarray
            Categorical values
        split_value : float
            Threshold on encoded values
            
        Returns
        -------
        np.ndarray
            Boolean array indicating left branch
        """
        if not self.encoding_map:
            raise ValueError("Encoding map not computed. Call get_split_value first.")
            
        numerical_values = np.array([self.encoding_map[x] for x in available_splitting_values])
        return numerical_values <= split_value


class CounterEncodingSplitRule(SplitRule):
    """
    Counter encoding for categorical variables - frequency-based encoding.
    Similar to CatBoost's Counter method.
    """
    
    def __init__(self, prior: float = 1.0, calculation_method: str = "Full"):
        self.prior = prior
        self.calculation_method = calculation_method
        self.encoding_map = {}
        self.max_count = 0
        
    def compute_counter_stats(self, categories: np.ndarray, training_categories: Optional[np.ndarray] = None) -> dict:
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
            Mapping from category to counter value
        """
        if training_categories is not None:
            # Use full training set for consistent encoding
            all_cats = training_categories
        else:
            all_cats = categories
            
        unique_cats, counts = np.unique(all_cats, return_counts=True)
        self.max_count = np.max(counts)
        
        encoding_map = {}
        current_counts = {}
        
        # Count occurrences in current set
        unique_current, counts_current = np.unique(categories, return_counts=True)
        for cat, count in zip(unique_current, counts_current):
            current_counts[cat] = count
            
        for cat in unique_cats:
            cur_count = current_counts.get(cat, 0)
            
            if self.calculation_method == "Full":
                total_count = counts[unique_cats == cat][0] if cat in unique_cats else 0
            else:  # SkipTest
                total_count = cur_count
                
            encoded_value = (cur_count + self.prior) / (self.max_count + self.prior)
            encoding_map[cat] = encoded_value
            
        return encoding_map
    
    def get_split_value(self, available_splitting_values, 
                       training_categories: Optional[np.ndarray] = None) -> Optional[float]:
        """
        Get split value using counter encoding.
        """
        if available_splitting_values.size <= 1:
            return None
            
        self.encoding_map = self.compute_counter_stats(
            available_splitting_values, 
            training_categories
        )
        
        numerical_values = np.array([self.encoding_map[x] for x in available_splitting_values])
        
        if len(np.unique(numerical_values)) > 1:
            idx = int(np.random.random() * len(numerical_values))
            return numerical_values[idx]
        
        return None

    def divide(self, available_splitting_values, split_value: float) -> np.ndarray:
        """
        Divide data based on counter encoded values.
        """
        if not self.encoding_map:
            raise ValueError("Encoding map not computed. Call get_split_value first.")
            
        numerical_values = np.array([self.encoding_map[x] for x in available_splitting_values])
        return numerical_values <= split_value
