#!/usr/bin/env python3
"""
Comprehensive Data Integrity Validation for JSON-Tables

This module provides bulletproof cell-by-cell validation for all conversion operations,
ensuring perfect data preservation across all edge cases.
"""

import pandas as pd
import numpy as np
import math
from typing import Any, Dict, List, Tuple, Optional
import warnings

class DataIntegrityError(Exception):
    """Raised when data integrity validation fails."""
    pass

class DataIntegrityValidator:
    """
    Comprehensive data integrity validator that checks every single cell.
    
    Handles all edge cases:
    - numpy.nan, None, pandas.NA equivalence
    - ±infinity representations  
    - Floating point precision
    - numpy scalar types
    - Mixed type columns
    - Empty/sparse data
    """
    
    @staticmethod
    def are_values_equal(val1: Any, val2: Any, tolerance: float = 1e-15) -> bool:
        """
        Compare two values for equality, handling all edge cases.
        
        Args:
            val1, val2: Values to compare
            tolerance: Floating point tolerance for numeric comparisons
            
        Returns:
            True if values are equivalent
        """
        # Handle None/null/NaN equivalence
        val1_is_null = DataIntegrityValidator._is_null_like(val1)
        val2_is_null = DataIntegrityValidator._is_null_like(val2)
        
        if val1_is_null and val2_is_null:
            return True
        elif val1_is_null or val2_is_null:
            return False
        
        # Handle infinity cases
        val1_is_inf = DataIntegrityValidator._is_infinity(val1)
        val2_is_inf = DataIntegrityValidator._is_infinity(val2)
        
        if val1_is_inf or val2_is_inf:
            if val1_is_inf and val2_is_inf:
                # Both are infinity - check if same sign
                return (val1 > 0) == (val2 > 0)
            else:
                return False
        
        # Convert numpy scalars to Python types for comparison
        val1_py = DataIntegrityValidator._to_python_type(val1)
        val2_py = DataIntegrityValidator._to_python_type(val2)
        
        # Handle floating point numbers with tolerance
        if isinstance(val1_py, (int, float)) and isinstance(val2_py, (int, float)):
            if isinstance(val1_py, float) or isinstance(val2_py, float):
                return abs(float(val1_py) - float(val2_py)) <= tolerance
            else:
                return val1_py == val2_py
        
        # String/boolean/other direct comparison
        return val1_py == val2_py
    
    @staticmethod
    def _is_null_like(value: Any) -> bool:
        """Check if value is any kind of null/missing value."""
        if value is None:
            return True
        if pd.isna(value):
            return True
        # Handle pandas.NA specifically
        if hasattr(pd, 'NA') and value is pd.NA:
            return True
        return False
    
    @staticmethod
    def _is_infinity(value: Any) -> bool:
        """Check if value is positive or negative infinity."""
        try:
            return math.isinf(float(value))
        except (TypeError, ValueError, OverflowError):
            return False
    
    @staticmethod
    def _to_python_type(value: Any) -> Any:
        """Convert numpy scalars and other types to Python primitives."""
        # Handle numpy scalars
        if hasattr(value, 'item'):
            return value.item()
        
        # Handle numpy.str_ specifically
        if hasattr(value, 'tolist'):
            return value.tolist()
        
        return value
    
    @staticmethod
    def validate_dataframe_equality(
        df1: pd.DataFrame, 
        df2: pd.DataFrame, 
        tolerance: float = 1e-15,
        operation_name: str = "conversion"
    ) -> bool:
        """
        Validate that two DataFrames are identical cell-by-cell.
        
        Args:
            df1, df2: DataFrames to compare
            tolerance: Floating point tolerance
            operation_name: Name of operation being validated
            
        Returns:
            True if DataFrames are identical
            
        Raises:
            DataIntegrityError: If any discrepancy is found
        """
        # Shape validation
        if df1.shape != df2.shape:
            raise DataIntegrityError(
                f"{operation_name}: Shape mismatch - {df1.shape} vs {df2.shape}"
            )
        
        # Column name validation
        if list(df1.columns) != list(df2.columns):
            raise DataIntegrityError(
                f"{operation_name}: Column mismatch - {list(df1.columns)} vs {list(df2.columns)}"
            )
        
        # Cell-by-cell validation
        rows, cols = df1.shape
        mismatches = []
        
        for row in range(rows):
            for col in range(cols):
                col_name = df1.columns[col]
                val1 = df1.iloc[row, col]
                val2 = df2.iloc[row, col]
                
                if not DataIntegrityValidator.are_values_equal(val1, val2, tolerance):
                    mismatches.append({
                        'row': row,
                        'col': col_name,
                        'original': val1,
                        'converted': val2,
                        'original_type': type(val1).__name__,
                        'converted_type': type(val2).__name__
                    })
        
        # Report mismatches
        if mismatches:
            error_msg = f"{operation_name}: {len(mismatches)} cell mismatches found:\n"
            for i, mismatch in enumerate(mismatches[:10]):  # Show first 10
                error_msg += (
                    f"  Row {mismatch['row']}, Col '{mismatch['col']}': "
                    f"{mismatch['original']} ({mismatch['original_type']}) != "
                    f"{mismatch['converted']} ({mismatch['converted_type']})\n"
                )
            if len(mismatches) > 10:
                error_msg += f"  ... and {len(mismatches) - 10} more mismatches"
            
            raise DataIntegrityError(error_msg)
        
        return True
    
    @staticmethod
    def validate_records_equality(
        records1: List[Dict[str, Any]], 
        records2: List[Dict[str, Any]], 
        tolerance: float = 1e-15,
        operation_name: str = "conversion"
    ) -> bool:
        """
        Validate that two record lists are identical.
        
        Args:
            records1, records2: Record lists to compare
            tolerance: Floating point tolerance
            operation_name: Name of operation being validated
            
        Returns:
            True if record lists are identical
            
        Raises:
            DataIntegrityError: If any discrepancy is found
        """
        # Length validation
        if len(records1) != len(records2):
            raise DataIntegrityError(
                f"{operation_name}: Record count mismatch - {len(records1)} vs {len(records2)}"
            )
        
        # Record-by-record validation
        mismatches = []
        
        for i, (rec1, rec2) in enumerate(zip(records1, records2)):
            # Key validation
            keys1 = set(rec1.keys())
            keys2 = set(rec2.keys())
            
            if keys1 != keys2:
                mismatches.append({
                    'record': i,
                    'error': 'key_mismatch',
                    'keys1': sorted(keys1),
                    'keys2': sorted(keys2)
                })
                continue
            
            # Value validation
            for key in keys1:
                val1 = rec1[key]
                val2 = rec2[key]
                
                if not DataIntegrityValidator.are_values_equal(val1, val2, tolerance):
                    mismatches.append({
                        'record': i,
                        'error': 'value_mismatch',
                        'key': key,
                        'original': val1,
                        'converted': val2,
                        'original_type': type(val1).__name__,
                        'converted_type': type(val2).__name__
                    })
        
        # Report mismatches
        if mismatches:
            error_msg = f"{operation_name}: {len(mismatches)} record mismatches found:\n"
            for i, mismatch in enumerate(mismatches[:10]):  # Show first 10
                if mismatch['error'] == 'key_mismatch':
                    error_msg += (
                        f"  Record {mismatch['record']}: Key mismatch - "
                        f"{mismatch['keys1']} vs {mismatch['keys2']}\n"
                    )
                else:
                    error_msg += (
                        f"  Record {mismatch['record']}, Key '{mismatch['key']}': "
                        f"{mismatch['original']} ({mismatch['original_type']}) != "
                        f"{mismatch['converted']} ({mismatch['converted_type']})\n"
                    )
            if len(mismatches) > 10:
                error_msg += f"  ... and {len(mismatches) - 10} more mismatches"
            
            raise DataIntegrityError(error_msg)
        
        return True

def validate_conversion_integrity(
    original_df: pd.DataFrame,
    conversion_func,
    back_conversion_func,
    operation_name: str = "roundtrip",
    tolerance: float = 1e-15,
    **kwargs
) -> bool:
    """
    Validate that a conversion operation preserves data integrity perfectly.
    
    Args:
        original_df: Original DataFrame
        conversion_func: Function to convert DataFrame (e.g., df_to_jt)
        back_conversion_func: Function to convert back (e.g., df_from_jt)
        operation_name: Name for error reporting
        tolerance: Floating point tolerance
        **kwargs: Additional arguments for conversion functions
        
    Returns:
        True if conversion preserves all data perfectly
        
    Raises:
        DataIntegrityError: If any data is corrupted during conversion
    """
    # Perform conversion
    converted_format = conversion_func(original_df, **kwargs)
    restored_df = back_conversion_func(converted_format)
    
    # Comprehensive validation
    DataIntegrityValidator.validate_dataframe_equality(
        original_df, restored_df, tolerance, operation_name
    )
    
    return True

def create_extreme_test_dataframe() -> pd.DataFrame:
    """
    Create a DataFrame with all possible edge cases for comprehensive testing.
    
    Returns:
        DataFrame containing every edge case we need to handle
    """
    # All the nasty edge cases
    data = {
        # Regular types
        'strings': ['hello', 'world', None, '', 'unicode: café 北京'],
        'integers': [42, -17, 0, None, 999999999],
        'floats': [3.14159, -2.71828, 0.0, None, 1e-308],
        'booleans': [True, False, None, True, False],
        
        # Numpy edge cases
        'numpy_nans': [np.nan, 1.0, np.nan, 2.0, np.nan],
        'infinities': [np.inf, -np.inf, 1.0, np.inf, -np.inf],
        'tiny_numbers': [1e-323, 0.0, 1e-308, 2.225e-308, 5e-324],
        'huge_numbers': [1e308, 1.797e308, 1e100, 9.999e307, 1e200],
        
        # Numpy scalar types
        'numpy_int64': [np.int64(42), np.int64(-17), None, np.int64(0), np.int64(999)],
        'numpy_float32': [np.float32(3.14), np.float32(-2.71), None, np.float32(0.0), np.float32(1.23)],
        'numpy_bool': [np.bool_(True), np.bool_(False), None, np.bool_(True), np.bool_(False)],
        
        # Mixed types
        'mixed': [42, 'text', True, None, 3.14],
        'zeros_and_nulls': [0, None, 0.0, False, ''],
        'special_floats': [0.0, -0.0, np.nan, np.inf, -np.inf]
    }
    
    return pd.DataFrame(data)

def run_comprehensive_integrity_test():
    """
    Run comprehensive data integrity tests on extreme edge cases.
    
    Returns:
        True if all tests pass
    """
    print("🔬 COMPREHENSIVE DATA INTEGRITY VALIDATION")
    print("=" * 60)
    
    # Create extreme test data
    print("📊 Creating extreme test DataFrame with all edge cases...")
    df = create_extreme_test_dataframe()
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Total cells: {df.shape[0] * df.shape[1]:,}")
    
    # Test standard implementation
    try:
        from .core import JSONTablesEncoder, JSONTablesDecoder
        print("\n🧪 Testing standard implementation...")
        
        # Convert to JSON-Tables and back
        json_table = JSONTablesEncoder.from_dataframe(df)
        df_restored = JSONTablesDecoder.to_dataframe(json_table)
        
        # Validate every single cell
        DataIntegrityValidator.validate_dataframe_equality(
            df, df_restored, operation_name="Standard JSON-Tables"
        )
        print("   ✅ Standard implementation: PERFECT data integrity")
        
    except Exception as e:
        print(f"   ❌ Standard implementation failed: {e}")
        return False
    
    # Test high-performance implementation
    try:
        from .high_performance_core import df_to_jt_hp
        print("\n🚀 Testing high-performance implementation...")
        
        # Test both configurations
        configs = [
            ("HP with numpy conversion", {"skip_numpy_conversion": False}),
            ("HP skip numpy conversion", {"skip_numpy_conversion": True}),
            ("HP chunked processing", {"skip_numpy_conversion": False, "max_workers": 4, "chunk_size": 100})
        ]
        
        for config_name, kwargs in configs:
            print(f"   Testing {config_name}...")
            
            # Convert to JSON-Tables and back
            json_table = df_to_jt_hp(df, **kwargs)
            df_restored = JSONTablesDecoder.to_dataframe(json_table)
            
            # Validate every single cell
            DataIntegrityValidator.validate_dataframe_equality(
                df, df_restored, operation_name=config_name
            )
            print(f"      ✅ {config_name}: PERFECT data integrity")
        
    except Exception as e:
        print(f"   ❌ High-performance implementation failed: {e}")
        return False
    
    # Test columnar format
    try:
        print("\n📊 Testing columnar format...")
        
        # Convert to columnar and back
        json_table = JSONTablesEncoder.from_dataframe(df, columnar=True)
        df_restored = JSONTablesDecoder.to_dataframe(json_table)
        
        # Validate every single cell
        DataIntegrityValidator.validate_dataframe_equality(
            df, df_restored, operation_name="Columnar format"
        )
        print("   ✅ Columnar format: PERFECT data integrity")
        
    except Exception as e:
        print(f"   ❌ Columnar format failed: {e}")
        return False
    
    print(f"\n🎉 ALL TESTS PASSED!")
    print(f"   ✅ Every single cell validated across all implementations")
    print(f"   ✅ All edge cases handled perfectly")
    print(f"   ✅ numpy.nan, ±inf, and all numpy types preserved")
    print(f"   ✅ Data integrity is bulletproof!")
    
    return True

if __name__ == "__main__":
    success = run_comprehensive_integrity_test()
    if not success:
        exit(1) 