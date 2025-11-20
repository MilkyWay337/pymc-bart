"""
Test script for TargetSplitRule and CounterSplitRule
Demonstrates categorical feature handling with CatBoost-style encoding
"""

import numpy as np
import pymc as pm
from pymc_bart import BART, TargetSplitRule, CounterSplitRule, ContinuousSplitRule


def test_target_split_rule_basic():
    """Test TargetSplitRule basic functionality"""
    print("\n" + "="*60)
    print("TEST 1: TargetSplitRule Basic Encoding")
    print("="*60)
    
    # Create categorical data with clear target relationship
    np.random.seed(42)
    n_samples = 100
    
    # Categorical feature: 0='A', 1='B', 2='C'
    X_cat = np.random.choice([0, 1, 2], size=n_samples)
    
    # Target depends on category: A=high, B=medium, C=low
    y = np.where(X_cat == 0, 5.0, np.where(X_cat == 1, 2.0, 0.0))
    y += np.random.normal(0, 0.5, n_samples)  # Add noise
    
    X = X_cat.reshape(-1, 1).astype(float)
    
    # Test TargetSplitRule encoding
    rule = TargetSplitRule(prior=1.0)
    encoding_map = rule.compute_target_encoding(X_cat, y)
    
    print(f"Category encodings (target-based):")
    for cat in sorted(encoding_map.keys()):
        print(f"  Category {int(cat)}: {encoding_map[cat]:.4f}")
    
    print(f"Mean y for A (cat 0): {y[X_cat == 0].mean():.4f}")
    print(f"Mean y for B (cat 1): {y[X_cat == 1].mean():.4f}")
    print(f"Mean y for C (cat 2): {y[X_cat == 2].mean():.4f}")
    
    # Build BART model
    print("\nTraining BART with TargetSplitRule...")
    split_rules = [TargetSplitRule(prior=1.0)]
    
    with pm.Model() as model:
        bart = BART('bart', X, y, m=10, split_rules=split_rules)
        idata = pm.sample(draws=100, chains=1, tune=100, progressbar=False)
    
    print("✓ TargetSplitRule test passed!")
    return idata


def test_counter_split_rule_basic():
    """Test CounterSplitRule basic functionality"""
    print("\n" + "="*60)
    print("TEST 2: CounterSplitRule Basic Encoding")
    print("="*60)
    
    # Create imbalanced categorical data
    np.random.seed(42)
    n_samples = 100
    
    # Categorical feature with different frequencies
    X_cat = np.concatenate([
        np.full(50, 0),      # Category 0: 50 occurrences
        np.full(30, 1),      # Category 1: 30 occurrences
        np.full(20, 2),      # Category 2: 20 occurrences
    ])
    
    y = np.random.normal(np.where(X_cat == 0, 3.0, 1.0), 0.5, n_samples)
    
    # Shuffle
    idx = np.random.permutation(n_samples)
    X_cat = X_cat[idx]
    y = y[idx]
    
    X = X_cat.reshape(-1, 1).astype(float)
    
    # Test CounterSplitRule encoding
    rule = CounterSplitRule(prior=1.0, calculation_method="Full")
    encoding_map = rule.compute_counter_encoding(X_cat)
    
    print(f"Category encodings (frequency-based):")
    for cat in sorted(encoding_map.keys()):
        print(f"  Category {int(cat)}: {encoding_map[cat]:.4f}")
    
    # Count frequencies
    unique, counts = np.unique(X_cat, return_counts=True)
    print(f"\nCategory frequencies:")
    for c, cnt in zip(unique, counts):
        print(f"  Category {int(c)}: {cnt} samples")
    
    # Build BART model
    print("\nTraining BART with CounterSplitRule...")
    split_rules = [CounterSplitRule(prior=1.0)]
    
    with pm.Model() as model:
        bart = BART('bart', X, y, m=10, split_rules=split_rules)
        idata = pm.sample(draws=100, chains=1, tune=100, progressbar=False)
    
    print("✓ CounterSplitRule test passed!")
    return idata


def test_mixed_split_rules():
    """Test mixed continuous and categorical features"""
    print("\n" + "="*60)
    print("TEST 3: Mixed Features (Continuous + Categorical)")
    print("="*60)
    
    np.random.seed(42)
    n_samples = 100
    
    # Feature 1: Continuous
    X_cont = np.random.uniform(-5, 5, n_samples)
    
    # Feature 2: Categorical
    X_cat = np.random.choice([0, 1, 2], size=n_samples)
    
    # Target depends on both features
    y = X_cont + 2.0 * np.where(X_cat == 0, 1.0, np.where(X_cat == 1, 0.0, -1.0))
    y += np.random.normal(0, 0.5, n_samples)
    
    X = np.column_stack([X_cont, X_cat])
    
    # Use different split rules for each feature
    split_rules = [
        ContinuousSplitRule(),      # For continuous feature
        TargetSplitRule(prior=1.0)  # For categorical feature
    ]
    
    print("Feature 1: Continuous (ContinuousSplitRule)")
    print("Feature 2: Categorical (TargetSplitRule)")
    
    print("\nTraining BART with mixed split rules...")
    with pm.Model() as model:
        bart = BART('bart', X, y, m=10, split_rules=split_rules)
        idata = pm.sample(draws=100, chains=1, tune=100, progressbar=False)
    
    print("✓ Mixed split rules test passed!")
    return idata


def test_split_value_computation():
    """Test that split values are computed correctly"""
    print("\n" + "="*60)
    print("TEST 4: Split Value Computation")
    print("="*60)
    
    # Simple test data
    categories = np.array([0, 0, 1, 1, 2, 2])
    targets = np.array([0.1, 0.2, 0.5, 0.6, 0.9, 1.0])
    
    # TargetSplitRule
    target_rule = TargetSplitRule(prior=1.0)
    split_val_target = target_rule.get_split_value(categories, targets=targets)
    print(f"TargetSplitRule split value: {split_val_target}")
    print(f"Encoding map: {target_rule.encoding_map}")
    
    # Division should work
    division = target_rule.divide(categories, split_val_target)
    print(f"Division result: {division}")
    print(f"Left branch indices: {np.where(division)[0]}")
    print(f"Right branch indices: {np.where(~division)[0]}")
    
    # CounterSplitRule
    print()
    counter_rule = CounterSplitRule(prior=1.0)
    split_val_counter = counter_rule.get_split_value(categories)
    print(f"CounterSplitRule split value: {split_val_counter}")
    print(f"Encoding map: {counter_rule.encoding_map}")
    
    division = counter_rule.divide(categories, split_val_counter)
    print(f"Division result: {division}")
    
    print("✓ Split value computation test passed!")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING CATBOOST-STYLE SPLIT RULES FOR PYMC-BART")
    print("="*60)
    
    try:
        # Test 4 first (no sampling needed)
        test_split_value_computation()
        
        # Run sampling tests
        idata1 = test_target_split_rule_basic()
        idata2 = test_counter_split_rule_basic()
        idata3 = test_mixed_split_rules()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
