#!/usr/bin/env python3
"""
Comprehensive Data Integrity Test Suite

This script validates that EVERY SINGLE CELL is preserved perfectly
across all JSON-Tables implementations and configurations.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add jsontables to path
sys.path.insert(0, '.')

from jsontables.data_integrity import (
    DataIntegrityValidator, 
    DataIntegrityError, 
    create_extreme_test_dataframe,
    validate_conversion_integrity
)
from jsontables import df_to_jt, df_from_jt, df_to_jt_hp

def create_real_world_test_data():
    """Create realistic test data with common edge cases."""
    np.random.seed(42)  # Reproducible results
    
    # Customer data with realistic edge cases
    customers = []
    for i in range(1000):
        customer = {
            'customer_id': f'CUST_{i:06d}',
            'name': f'Customer {i}' if i % 10 != 0 else None,  # 10% missing names
            'age': np.random.randint(18, 80) if i % 15 != 0 else None,  # ~7% missing ages
            'balance': round(np.random.normal(1000, 500), 2) if i % 8 != 0 else None,  # ~12% missing balances
            'is_active': bool(np.random.choice([True, False], p=[0.8, 0.2])),
            'signup_date': f'2023-{np.random.randint(1, 13):02d}-{np.random.randint(1, 28):02d}',
            'category': np.random.choice(['Premium', 'Standard', 'Basic'], p=[0.2, 0.6, 0.2]),
        }
        
        # Add some extreme values
        if i == 100:
            customer['balance'] = np.inf
        elif i == 101:
            customer['balance'] = -np.inf  
        elif i == 102:
            customer['balance'] = np.nan
        elif i == 103:
            customer['age'] = 0  # Edge case: zero value
        elif i == 104:
            customer['name'] = ''  # Edge case: empty string
        
        customers.append(customer)
    
    return pd.DataFrame(customers)

def test_cell_by_cell_validation():
    """Test that our validation actually checks every single cell."""
    print("🔬 CELL-BY-CELL VALIDATION TEST")
    print("=" * 50)
    
    # Create test DataFrame with known values
    df_original = pd.DataFrame({
        'col1': [1, 2, np.nan, 4, np.inf],
        'col2': ['a', None, 'c', '', 'e'],
        'col3': [True, False, None, True, False],
        'col4': [1.1, 2.2, np.nan, -np.inf, 0.0]
    })
    
    print(f"📊 Original DataFrame:")
    print(df_original)
    print(f"Shape: {df_original.shape}")
    
    # Test 1: Perfect conversion (should pass)
    print(f"\n✅ Test 1: Perfect conversion")
    json_table = df_to_jt(df_original)
    df_restored = df_from_jt(json_table)
    
    try:
        DataIntegrityValidator.validate_dataframe_equality(
            df_original, df_restored, operation_name="Perfect conversion test"
        )
        print("   ✅ PASSED: All cells match perfectly")
    except DataIntegrityError as e:
        print(f"   ❌ FAILED: {e}")
        return False
    
    # Test 2: Deliberately corrupt a cell (should fail)
    print(f"\n❌ Test 2: Deliberately corrupted data (should fail)")
    df_corrupted = df_restored.copy()
    df_corrupted.iloc[0, 0] = 999  # Change first cell from 1 to 999
    
    try:
        DataIntegrityValidator.validate_dataframe_equality(
            df_original, df_corrupted, operation_name="Corruption detection test"
        )
        print("   ❌ FAILED: Should have detected corruption!")
        return False
    except DataIntegrityError as e:
        print(f"   ✅ PASSED: Correctly detected corruption: {str(e)[:100]}...")
    
    # Test 3: Type mismatch (should fail)
    print(f"\n❌ Test 3: Type change (should fail)")
    df_type_changed = df_restored.copy()
    df_type_changed.iloc[1, 1] = 123  # Change string 'None' to integer 123
    
    try:
        DataIntegrityValidator.validate_dataframe_equality(
            df_original, df_type_changed, operation_name="Type change detection test"
        )
        print("   ❌ FAILED: Should have detected type change!")
        return False
    except DataIntegrityError as e:
        print(f"   ✅ PASSED: Correctly detected type change: {str(e)[:100]}...")
    
    print(f"\n🎉 Cell-by-cell validation system works perfectly!")
    return True

def test_extreme_edge_cases():
    """Test every possible edge case we can think of."""
    print("\n🌊 EXTREME EDGE CASE TESTING")
    print("=" * 50)
    
    # Create DataFrame with every nasty edge case
    df_extreme = create_extreme_test_dataframe()
    print(f"📊 Extreme test DataFrame shape: {df_extreme.shape}")
    print(f"   Total cells to validate: {df_extreme.shape[0] * df_extreme.shape[1]:,}")
    
    # Test all implementations
    implementations = [
        ("Standard JSON-Tables", lambda df: df_to_jt(df), df_from_jt),
        ("High-Performance", lambda df: df_to_jt_hp(df), df_from_jt),
        ("HP Skip Numpy", lambda df: df_to_jt_hp(df, skip_numpy_conversion=True), df_from_jt),
        ("HP Chunked", lambda df: df_to_jt_hp(df, max_workers=4, chunk_size=100), df_from_jt),
        ("Columnar Format", lambda df: df_to_jt(df, columnar=True), df_from_jt),
    ]
    
    for impl_name, convert_func, restore_func in implementations:
        print(f"\n🧪 Testing {impl_name}...")
        
        try:
            # Perform conversion
            json_table = convert_func(df_extreme)
            df_restored = restore_func(json_table)
            
            # Validate EVERY SINGLE CELL
            DataIntegrityValidator.validate_dataframe_equality(
                df_extreme, df_restored, operation_name=impl_name
            )
            
            print(f"   ✅ PERFECT: All {df_extreme.shape[0] * df_extreme.shape[1]:,} cells preserved exactly")
            
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            return False
    
    print(f"\n🎉 ALL EXTREME EDGE CASES PASSED!")
    return True

def test_real_world_data():
    """Test realistic data that mimics real-world usage."""
    print("\n🌍 REAL-WORLD DATA TESTING")
    print("=" * 50)
    
    # Create realistic test data
    df_real = create_real_world_test_data()
    print(f"📊 Real-world test data shape: {df_real.shape}")
    print(f"   Total cells to validate: {df_real.shape[0] * df_real.shape[1]:,}")
    
    # Show some statistics
    print(f"   Missing values: {df_real.isnull().sum().sum():,}")
    print(f"   Unique customers: {df_real['customer_id'].nunique():,}")
    print(f"   Categories: {df_real['category'].value_counts().to_dict()}")
    
    # Test high-performance implementation on realistic data
    print(f"\n🚀 Testing high-performance implementation...")
    
    try:
        # Convert using high-performance mode
        json_table = df_to_jt_hp(df_real)
        df_restored = df_from_jt(json_table)
        
        # Validate every single cell
        DataIntegrityValidator.validate_dataframe_equality(
            df_real, df_restored, operation_name="Real-world high-performance"
        )
        
        print(f"   ✅ PERFECT: All {df_real.shape[0] * df_real.shape[1]:,} cells preserved exactly")
        print(f"   ✅ All {df_real.isnull().sum().sum():,} missing values preserved")
        print(f"   ✅ All infinity and edge case values preserved")
        
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False
    
    print(f"\n🎉 REAL-WORLD DATA TEST PASSED!")
    return True

def test_floating_point_precision():
    """Test that floating point precision is preserved perfectly."""
    print("\n🔢 FLOATING POINT PRECISION TESTING")
    print("=" * 50)
    
    # Create DataFrame with challenging floating point values
    df_precision = pd.DataFrame({
        'tiny_numbers': [1e-323, 5e-324, 2.225e-308, 1e-308, 0.0],
        'huge_numbers': [1.797e308, 1e308, 9.999e307, 1e200, 1e100],
        'precise_decimals': [0.1, 0.2, 0.3, 1/3, 22/7],
        'edge_floats': [np.finfo(float).eps, np.finfo(float).tiny, np.finfo(float).max, -0.0, 0.0],
        'special_values': [np.nan, np.inf, -np.inf, np.pi, np.e]
    })
    
    print(f"📊 Precision test DataFrame shape: {df_precision.shape}")
    
    # Test with extremely tight tolerance
    tolerance = 1e-17  # Even tighter than default
    
    try:
        # Convert and restore
        json_table = df_to_jt_hp(df_precision)
        df_restored = df_from_jt(json_table)
        
        # Validate with tight precision
        DataIntegrityValidator.validate_dataframe_equality(
            df_precision, df_restored, tolerance=tolerance, 
            operation_name="Floating point precision"
        )
        
        print(f"   ✅ PERFECT: All floating point values preserved with tolerance {tolerance}")
        
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False
    
    print(f"\n🎉 FLOATING POINT PRECISION TEST PASSED!")
    return True

def main():
    """Run the complete comprehensive data integrity test suite."""
    print("🔬 COMPREHENSIVE DATA INTEGRITY TEST SUITE")
    print("=" * 60)
    print("This test validates that EVERY SINGLE CELL is preserved perfectly")
    print("across all JSON-Tables implementations and edge cases.")
    print("=" * 60)
    
    tests = [
        ("Cell-by-cell validation system", test_cell_by_cell_validation),
        ("Extreme edge cases", test_extreme_edge_cases),
        ("Real-world data", test_real_world_data),
        ("Floating point precision", test_floating_point_precision),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"🧪 RUNNING: {test_name}")
        print(f"{'='*60}")
        
        try:
            if test_func():
                passed += 1
                print(f"✅ PASSED: {test_name}")
            else:
                print(f"❌ FAILED: {test_name}")
        except Exception as e:
            print(f"💥 ERROR in {test_name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"📊 FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print(f"🎉 ALL TESTS PASSED!")
        print(f"✅ Data integrity is bulletproof across all implementations")
        print(f"✅ Every single cell validated in every test case")
        print(f"✅ All edge cases (nan, inf, types, nulls) handled perfectly")
        return True
    else:
        print(f"❌ {total - passed} test(s) failed!")
        print(f"🚨 Data integrity issues detected!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 