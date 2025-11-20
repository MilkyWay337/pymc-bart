import jax
import jax.numpy as jnp
import numpy as np
import pymc as pm
from typing import List, Optional, Callable, Dict, Any
from functools import partial

from pymc_bart.split_rules import continuous_split_rule, target_split_rule

class BART:
    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        feature_types: Optional[List[str]] = None,
        split_rules: Optional[List[Callable]] = None,
        m: int = 50,
        alpha: float = 0.95,
        beta: float = 2.0,
        **kwargs
    ):
        self.X = X
        self.Y = Y
        self.m = m
        self.alpha = alpha
        self.beta = beta
        
        # Handle feature types and split rules
        self.feature_types = feature_types or self._infer_feature_types(X)
        self.split_rules = split_rules or self._create_split_rules()
        
        # Validate dimensions
        if len(self.feature_types) != X.shape[1]:
            raise ValueError(f"feature_types length ({len(self.feature_types)}) must match number of features ({X.shape[1]})")
        if len(self.split_rules) != X.shape[1]:
            raise ValueError(f"split_rules length ({len(self.split_rules)}) must match number of features ({X.shape[1]})")
        
        self.kwargs = kwargs
        self.model = None
        self.trees = []
        
        self._build_model()
    
    def _infer_feature_types(self, X: np.ndarray) -> List[str]:
        """Infer feature types based on data characteristics."""
        feature_types = []
        n_samples, n_features = X.shape
        
        for i in range(n_features):
            unique_vals = len(np.unique(X[:, i]))
            # Simple heuristic for categorical features
            if unique_vals <= min(20, n_samples / 10) and not self._is_likely_continuous(X[:, i]):
                feature_types.append('categorical')
            else:
                feature_types.append('continuous')
        
        return feature_types
    
    def _is_likely_continuous(self, feature_values: np.ndarray) -> bool:
        """Check if feature is likely continuous."""
        unique_vals = np.unique(feature_values)
        if len(unique_vals) > 20:
            return True
        
        # Check if values are numeric and have reasonable spread
        if np.issubdtype(feature_values.dtype, np.number):
            value_range = np.max(feature_values) - np.min(feature_values)
            if value_range > 1e-10:
                return True
        
        return False
    
    def _create_split_rules(self) -> List[Callable]:
        """Create appropriate split rules for each feature."""
        split_rules = []
        for feature_type in self.feature_types:
            if feature_type == 'categorical':
                # Use target-based split rule for categorical features
                split_rules.append(partial(target_split_rule, alpha=1.0))
            else:
                # Use continuous split rule for continuous features
                split_rules.append(continuous_split_rule)
        return split_rules
    
    def _build_model(self):
        """Build the BART model with categorical feature support."""
        with pm.Model() as self.model:
            # Priors for tree parameters
            tree_depth = pm.Gamma("tree_depth", alpha=self.alpha, beta=self.beta)
            
            # Convert to JAX arrays
            X_jax = jnp.array(self.X)
            Y_jax = jnp.array(self.Y)
            
            # Initialize trees
            self.trees = []
            for i in range(self.m):
                tree_output = pm.Normal(f"tree_{i}", 0, 1 / self.m)
                
                # Store tree information with split rules
                tree_info = {
                    'output': tree_output,
                    'split_rules': self.split_rules,
                    'feature_types': self.feature_types,
                    'X': X_jax,
                    'Y': Y_jax
                }
                self.trees.append(tree_info)
            
            # Sum of trees
            tree_sum = sum(tree['output'] for tree in self.trees)
            
            # Likelihood
            y_obs = pm.Normal("y_obs", mu=tree_sum, observed=self.Y)
    
    def fit(self, n_iter: int = 1000, **kwargs):
        """Fit the BART model."""
        if self.model is None:
            self._build_model()
        
        with self.model:
            trace = pm.sample(n_iter, **kwargs)
        
        return trace
    
    def predict(self, X_new: np.ndarray):
        """Make predictions using the fitted model."""
        # This is a simplified version - you'd need to implement
        # the actual prediction logic using the sampled trees
        X_new_jax = jnp.array(X_new)
        
        # Placeholder for prediction logic
        # In practice, you'd use the sampled trees to make predictions
        predictions = jnp.zeros(X_new.shape[0])
        
        return np.array(predictions)

    def _build_tree_structure(self, X, y, split_rules, feature_types, depth=0, max_depth=5):
        """
        Recursively build tree structure using appropriate split rules.
        This is a simplified version for demonstration.
        """
        if depth >= max_depth or len(y) < 2:
            return {'leaf': jnp.mean(y)}
        
        best_gain = -jnp.inf
        best_split = None
        best_feature = None
        
        # Try all features
        for feature in range(X.shape[1]):
            split_rule = split_rules[feature]
            split_result = split_rule(X, y, feature, min_samples_leaf=1)
            
            if split_result is not None:
                split_value, gain = split_result
                if gain > best_gain:
                    best_gain = gain
                    best_split = split_value
                    best_feature = feature
        
        if best_split is None:
            return {'leaf': jnp.mean(y)}
        
        # Create split
        if feature_types[best_feature] == 'categorical':
            # For categorical, split based on target-ordered categories
            left_mask = X[:, best_feature] <= best_split
        else:
            # For continuous, normal split
            left_mask = X[:, best_feature] <= best_split
        
        right_mask = ~left_mask
        
        if jnp.sum(left_mask) == 0 or jnp.sum(right_mask) == 0:
            return {'leaf': jnp.mean(y)}
        
        # Recursively build subtrees
        left_subtree = self._build_tree_structure(
            X[left_mask], y[left_mask], split_rules, feature_types, depth + 1, max_depth
        )
        right_subtree = self._build_tree_structure(
            X[right_mask], y[right_mask], split_rules, feature_types, depth + 1, max_depth
        )
        
        return {
            'feature': best_feature,
            'split_value': best_split,
            'left': left_subtree,
            'right': right_subtree
        }
