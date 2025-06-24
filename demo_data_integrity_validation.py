#!/usr/bin/env python3
"""
Demo: Data Integrity Validation

This demo shows how to use JSON-Tables' data integrity validation
to ensure that EVERY SINGLE CELL is preserved perfectly during conversion.
"""

import pandas as pd
import numpy as np
from jsontables import (
    df_to_jt, df_from_jt,
    DataIntegrityValidator, DataIntegrityError,
    validate_conversion_integrity
)

def demo_basic_validation():
    """Demo basic data integrity validation."""
    print("🔍 BASIC DATA INTEGRITY VALIDATION")
    print("=" * 50)
    
    # Create test DataFrame
    df = pd.DataFrame({
        'name': ['Alice', 'Bob', None],
        'age': [30, 25, np.nan],
        'balance': [1000.50, -250.75, np.inf],
        'active': [True, False, None]
    })
    
    print("📊 Original DataFrame:")
    print(df)
    print(f"Shape: {df.shape}")
    
    # Convert to JSON-Tables and back
    json_table = df_to_jt(df)
    df_restored = df_from_jt(json_table)
    
    print(f"\n📊 Restored DataFrame:")
    print(df_restored)
    print(f"Shape: {df_restored.shape}")
    
    # Validate every single cell
    try:
        DataIntegrityValidator.validate_dataframe_equality(
            df, df_restored, operation_name="Basic conversion"
        )
        print(f"\n✅ PERFECT: All {df.shape[0] * df.shape[1]} cells preserved exactly!")
        
    except DataIntegrityError as e:
        print(f"\n❌ Data integrity violation: {e}")
        return False
    
    return True

def demo_extreme_edge_cases():
    """Demo validation on extreme edge cases."""
    print("\n🌊 EXTREME EDGE CASE VALIDATION")
    print("=" * 50)
    
    # Create DataFrame with every possible edge case
    df = pd.DataFrame({
        'strings': ['hello', 'world', None, '', 'unicode: café 北京'],
        'integers': [42, -17, 0, None, 999999999],
        'floats': [3.14159, -2.71828, 0.0, None, 1e-308],
        'booleans': [True, False, None, True, False],
        'numpy_nans': [np.nan, 1.0, np.nan, 2.0, np.nan],
        'infinities': [np.inf, -np.inf, 1.0, np.inf, -np.inf],
        'mixed': [42, 'text', True, None, 3.14]
    })
    
    print(f"📊 Extreme test DataFrame: {df.shape}")
    print(f"   Total cells to validate: {df.shape[0] * df.shape[1]:,}")
    
    # Show some of the extreme values
    print(f"\n🔬 Sample extreme values:")
    print(f"   Infinities: {df['infinities'].tolist()}")
    print(f"   Mixed types: {df['mixed'].tolist()}")
    
    # Test conversion
    print(f"\n🚀 Testing conversion...")
    
    try:
        json_table = df_to_jt(df)
        df_restored = df_from_jt(json_table)
        
        # Validate every single cell
        DataIntegrityValidator.validate_dataframe_equality(
            df, df_restored, operation_name="Extreme edge cases"
        )
        print(f"   ✅ PERFECT: All {df.shape[0] * df.shape[1]:,} cells preserved exactly!")
        print(f"   ✅ All numpy.nan, ±inf, and extreme values handled perfectly!")
        
    except DataIntegrityError as e:
        print(f"   ❌ Data integrity violation: {e}")
        return False
    
    return True

def demo_deliberate_corruption_detection():
    """Demo that validation catches data corruption."""
    print("\n🚨 CORRUPTION DETECTION DEMO")
    print("=" * 50)
    
    # Create test DataFrame
    df_original = pd.DataFrame({
        'values': [1, 2, 3, 4, 5],
        'names': ['a', 'b', 'c', 'd', 'e']
    })
    
    # Create deliberately corrupted version
    df_corrupted = df_original.copy()
    df_corrupted.iloc[2, 0] = 999  # Change 3 to 999
    df_corrupted.iloc[1, 1] = 'WRONG'  # Change 'b' to 'WRONG'
    
    print("📊 Original DataFrame:")
    print(df_original)
    print(f"\n📊 Corrupted DataFrame:")
    print(df_corrupted)
    
    print(f"\n🔍 Running validation (should detect corruption)...")
    
    try:
        DataIntegrityValidator.validate_dataframe_equality(
            df_original, df_corrupted, operation_name="Corruption test"
        )
        print("   ❌ FAILED: Should have detected corruption!")
        return False
        
    except DataIntegrityError as e:
        print(f"   ✅ SUCCESS: Correctly detected corruption!")
        print(f"   Details: {str(e)[:150]}...")
        return True

def demo_real_world_validation():
    """Demo validation on realistic data."""
    print("\n🌍 REAL-WORLD DATA VALIDATION")
    print("=" * 50)
    
    # Create realistic customer data
    np.random.seed(42)
    customers = []
    for i in range(100):
        customer = {
            'customer_id': f'CUST_{i:06d}',
            'name': f'Customer {i}' if i % 15 != 0 else None,  # ~7% missing
            'age': np.random.randint(18, 80) if i % 10 != 0 else None,  # 10% missing
            'balance': round(np.random.normal(1000, 500), 2) if i % 8 != 0 else None,
            'is_premium': bool(np.random.choice([True, False], p=[0.3, 0.7])),
            'region': np.random.choice(['US', 'EU', 'APAC', None], p=[0.4, 0.3, 0.2, 0.1]),
        }
        
        # Add some edge cases
        if i == 10:
            customer['balance'] = np.inf
        elif i == 11:
            customer['balance'] = -np.inf
        elif i == 12:
            customer['balance'] = np.nan
            
        customers.append(customer)
    
    df = pd.DataFrame(customers)
    
    print(f"📊 Customer data: {df.shape}")
    print(f"   Missing values: {df.isnull().sum().sum()}")
    print(f"   Unique customers: {df['customer_id'].nunique()}")
    
    # Test conversion with validation
    print(f"\n🚀 Converting and validating...")
    
    try:
        # Use the convenience validation function
        validate_conversion_integrity(
            original_df=df,
            conversion_func=df_to_jt,
            back_conversion_func=df_from_jt,
            operation_name="Real-world customer data"
        )
        
        print(f"   ✅ PERFECT: All {df.shape[0] * df.shape[1]:,} cells preserved!")
        print(f"   ✅ All {df.isnull().sum().sum()} missing values preserved!")
        print(f"   ✅ All infinity values and edge cases preserved!")
        
    except DataIntegrityError as e:
        print(f"   ❌ Data integrity violation: {e}")
        return False
    
    return True

def demo_format_comparison():
    """Demo validation across different formats."""
    print("\n⚡ FORMAT COMPARISON")
    print("=" * 50)
    
    # Create test data
    df = pd.DataFrame({
        'col1': range(100),
        'col2': [f'text_{i}' for i in range(100)],
        'col3': np.random.randn(100),
        'col4': np.random.choice([True, False, None], 100, p=[0.4, 0.4, 0.2])
    })
    
    print(f"📊 Test data: {df.shape} = {df.shape[0] * df.shape[1]} cells")
    
    # Test different formats
    formats = [
        ("Row format (default)", lambda d: df_to_jt(d)),
        ("Columnar format", lambda d: df_to_jt(d, columnar=True)),
    ]
    
    for format_name, convert_func in formats:
        print(f"\n🧪 Testing {format_name}...")
        
        try:
            # Convert and validate
            json_table = convert_func(df)
            df_restored = df_from_jt(json_table)
            
            # Validate every cell
            DataIntegrityValidator.validate_dataframe_equality(
                df, df_restored, operation_name=format_name
            )
            
            print(f"   ✅ PERFECT: All {df.shape[0] * df.shape[1]} cells preserved")
            
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            return False
    
    print(f"\n🎉 ALL FORMATS PRESERVE DATA PERFECTLY!")
    return True

def main():
    """Run all data integrity validation demos."""
    print("🔬 JSON-TABLES DATA INTEGRITY VALIDATION DEMO")
    print("=" * 60)
    print("This demo shows how to verify that EVERY SINGLE CELL")
    print("is preserved perfectly during JSON-Tables conversion.")
    print("=" * 60)
    
    demos = [
        ("Basic validation", demo_basic_validation),
        ("Extreme edge cases", demo_extreme_edge_cases),
        ("Corruption detection", demo_deliberate_corruption_detection),
        ("Real-world data", demo_real_world_validation),
        ("Format comparison", demo_format_comparison),
    ]
    
    passed = 0
    for demo_name, demo_func in demos:
        print(f"\n{'='*60}")
        print(f"🧪 DEMO: {demo_name}")
        print(f"{'='*60}")
        
        try:
            if demo_func():
                passed += 1
            else:
                print(f"❌ Demo failed: {demo_name}")
        except Exception as e:
            print(f"💥 Demo error: {e}")
    
    print(f"\n{'='*60}")
    print(f"📊 DEMO RESULTS: {passed}/{len(demos)} successful")
    print(f"{'='*60}")
    
    if passed == len(demos):
        print("🎉 ALL DEMOS PASSED!")
        print("✅ JSON-Tables provides bulletproof data integrity")
        print("✅ Every single cell is validated and preserved")
        print("✅ All edge cases (nan, inf, nulls, types) handled perfectly")
        
        print(f"\n💡 HOW TO USE IN YOUR CODE:")
        print(f"```python")
        print(f"from jsontables import DataIntegrityValidator, df_to_jt, df_from_jt")
        print(f"")
        print(f"# Convert your DataFrame")
        print(f"json_table = df_to_jt(your_df)")
        print(f"df_restored = df_from_jt(json_table)")
        print(f"")
        print(f"# Validate every single cell")
        print(f"DataIntegrityValidator.validate_dataframe_equality(")
        print(f"    your_df, df_restored, operation_name='Your conversion'")
        print(f")")
        print(f"# ✅ Throws DataIntegrityError if ANY cell differs!")
        print(f"```")
        
    else:
        print("❌ Some demos failed - investigate data integrity issues")

if __name__ == "__main__":
    main() 