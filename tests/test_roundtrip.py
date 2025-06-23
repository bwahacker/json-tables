#!/usr/bin/env python3
"""
Comprehensive round-trip tests for JSON-Tables.

Tests that data can be converted to JSON-Tables format and back
without any loss of information or data corruption.
"""

import sys
import os
import unittest
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from jsontables.core import JSONTablesEncoder, JSONTablesDecoder


class TestJSONTablesRoundTrip(unittest.TestCase):
    """Test round-trip conversion: data → JSON-Tables → data"""
    
    def assertDataEqual(self, original: List[Dict[str, Any]], converted: List[Dict[str, Any]], msg: str = None):
        """Assert that two datasets are equivalent, handling type conversions."""
        import pandas as pd
        
        self.assertEqual(len(original), len(converted), f"Length mismatch: {msg}")
        
        for i, (orig_row, conv_row) in enumerate(zip(original, converted)):
            self.assertEqual(set(orig_row.keys()), set(conv_row.keys()), 
                           f"Row {i} key mismatch: {msg}")
            
            for key in orig_row.keys():
                orig_val = orig_row[key]
                conv_val = conv_row[key]
                
                # Handle None/null/NaN equivalence using pandas
                orig_is_null = pd.isna(orig_val)
                conv_is_null = pd.isna(conv_val)
                
                if orig_is_null and conv_is_null:
                    continue  # Both are null-like values - equivalent
                elif orig_is_null or conv_is_null:
                    self.fail(f"Row {i}, key '{key}': null mismatch - orig: {orig_val}, conv: {conv_val}")
                else:
                    self.assertEqual(orig_val, conv_val, 
                                   f"Row {i}, key '{key}': {orig_val} != {conv_val}")

    def test_basic_round_trip(self):
        """Test basic data types round-trip correctly."""
        original_data = [
            {"name": "Alice", "age": 30, "active": True, "score": 95.5},
            {"name": "Bob", "age": 25, "active": False, "score": 87.2},
            {"name": "Carol", "age": 35, "active": True, "score": 92.8}
        ]
        
        # Convert to JSON-Tables
        json_tables = JSONTablesEncoder.from_records(original_data)
        
        # Convert back to records
        converted_data = JSONTablesDecoder.to_records(json_tables)
        
        # Verify data integrity
        self.assertDataEqual(original_data, converted_data, "Basic data types")

    def test_null_values_round_trip(self):
        """Test that null/None values are preserved correctly."""
        original_data = [
            {"id": "U001", "name": "Alice", "age": 30, "city": "NYC", "notes": None},
            {"id": "U002", "name": "Bob", "age": None, "city": "LA", "notes": "Important"},
            {"id": "U003", "name": None, "age": 25, "city": None, "notes": None}
        ]
        
        # Convert to JSON-Tables
        json_tables = JSONTablesEncoder.from_records(original_data)
        
        # Convert back to records
        converted_data = JSONTablesDecoder.to_records(json_tables)
        
        # Verify null preservation
        self.assertDataEqual(original_data, converted_data, "Null values")

    def test_mixed_data_types_round_trip(self):
        """Test mixed data types including strings, numbers, booleans, nulls."""
        original_data = [
            {
                "string_field": "text",
                "int_field": 42,
                "float_field": 3.14159,
                "bool_field": True,
                "null_field": None,
                "zero_field": 0,
                "empty_string": "",
                "false_field": False
            },
            {
                "string_field": "another text",
                "int_field": -17,
                "float_field": 0.0,
                "bool_field": False,
                "null_field": "not null",
                "zero_field": 100,
                "empty_string": "not empty",
                "false_field": True
            }
        ]
        
        # Convert to JSON-Tables
        json_tables = JSONTablesEncoder.from_records(original_data)
        
        # Convert back to records
        converted_data = JSONTablesDecoder.to_records(json_tables)
        
        # Verify mixed types preserved
        self.assertDataEqual(original_data, converted_data, "Mixed data types")

    def test_categorical_data_round_trip(self):
        """Test data with repeated categorical values."""
        original_data = [
            {"user_id": "U001", "status": "Premium", "region": "US", "active": True},
            {"user_id": "U002", "status": "Standard", "region": "EU", "active": True},
            {"user_id": "U003", "status": "Premium", "region": "US", "active": False},
            {"user_id": "U004", "status": "Standard", "region": "APAC", "active": True},
            {"user_id": "U005", "status": "Premium", "region": "EU", "active": True}
        ]
        
        # Convert to JSON-Tables
        json_tables = JSONTablesEncoder.from_records(original_data)
        
        # Convert back to records
        converted_data = JSONTablesDecoder.to_records(json_tables)
        
        # Verify categorical data preserved
        self.assertDataEqual(original_data, converted_data, "Categorical data")

    def test_sparse_data_round_trip(self):
        """Test sparse data with many missing values."""
        original_data = [
            {"id": "1", "name": "Alice", "email": "alice@example.com", "phone": None, "address": None},
            {"id": "2", "name": "Bob", "email": None, "phone": "555-1234", "address": None},
            {"id": "3", "name": "Carol", "email": None, "phone": None, "address": "123 Main St"},
            {"id": "4", "name": "David", "email": "david@example.com", "phone": "555-5678", "address": "456 Oak Ave"}
        ]
        
        # Convert to JSON-Tables
        json_tables = JSONTablesEncoder.from_records(original_data)
        
        # Convert back to records
        converted_data = JSONTablesDecoder.to_records(json_tables)
        
        # Verify sparse data preserved
        self.assertDataEqual(original_data, converted_data, "Sparse data")

    def test_edge_case_values_round_trip(self):
        """Test edge case values like empty strings, zeros, special characters."""
        original_data = [
            {
                "empty_string": "",
                "zero_int": 0,
                "zero_float": 0.0,
                "negative": -42,
                "special_chars": "!@#$%^&*()_+-=[]{}|;:,.<>?",
                "unicode": "café Naïve résumé 北京",
                "very_long": "x" * 1000
            },
            {
                "empty_string": "not empty",
                "zero_int": 1,
                "zero_float": 1.0,
                "negative": 42,
                "special_chars": "normal text",
                "unicode": "ASCII only",
                "very_long": "short"
            }
        ]
        
        # Convert to JSON-Tables
        json_tables = JSONTablesEncoder.from_records(original_data)
        
        # Convert back to records
        converted_data = JSONTablesDecoder.to_records(json_tables)
        
        # Verify edge cases preserved
        self.assertDataEqual(original_data, converted_data, "Edge case values")

    def test_single_row_round_trip(self):
        """Test single row data."""
        original_data = [
            {"id": "single", "value": "only one row", "count": 1}
        ]
        
        # Convert to JSON-Tables
        json_tables = JSONTablesEncoder.from_records(original_data)
        
        # Convert back to records
        converted_data = JSONTablesDecoder.to_records(json_tables)
        
        # Verify single row preserved
        self.assertDataEqual(original_data, converted_data, "Single row")

    def test_empty_data_round_trip(self):
        """Test empty dataset."""
        original_data = []
        
        # Convert to JSON-Tables
        json_tables = JSONTablesEncoder.from_records(original_data)
        
        # Convert back to records
        converted_data = JSONTablesDecoder.to_records(json_tables)
        
        # Verify empty data handled correctly
        self.assertEqual(original_data, converted_data, "Empty dataset")

    def test_inconsistent_schema_round_trip(self):
        """Test data with inconsistent schemas (missing fields in some rows)."""
        original_data = [
            {"id": "1", "name": "Alice", "age": 30, "city": "NYC"},
            {"id": "2", "name": "Bob", "city": "LA"},  # Missing age
            {"id": "3", "name": "Carol", "age": 25},   # Missing city
            {"id": "4", "age": 35, "city": "Chicago"}  # Missing name
        ]
        
        # Convert to JSON-Tables
        json_tables = JSONTablesEncoder.from_records(original_data)
        
        # Convert back to records
        converted_data = JSONTablesDecoder.to_records(json_tables)
        
        # For inconsistent schemas, missing fields should become None/null
        expected_data = [
            {"id": "1", "name": "Alice", "age": 30, "city": "NYC"},
            {"id": "2", "name": "Bob", "age": None, "city": "LA"},
            {"id": "3", "name": "Carol", "age": 25, "city": None},
            {"id": "4", "name": None, "age": 35, "city": "Chicago"}
        ]
        
        # Verify inconsistent schema handled correctly
        self.assertDataEqual(expected_data, converted_data, "Inconsistent schema")

    def test_large_dataset_round_trip(self):
        """Test larger dataset for performance and correctness."""
        # Generate test data
        original_data = []
        for i in range(1000):
            original_data.append({
                "id": f"ID_{i:06d}",
                "name": f"User_{i}",
                "score": i * 0.1,
                "active": i % 2 == 0,
                "group": f"Group_{i % 5}",
                "optional": None if i % 7 == 0 else f"Value_{i}"
            })
        
        # Convert to JSON-Tables
        json_tables = JSONTablesEncoder.from_records(original_data)
        
        # Convert back to records
        converted_data = JSONTablesDecoder.to_records(json_tables)
        
        # Verify large dataset preserved
        self.assertDataEqual(original_data, converted_data, "Large dataset (1000 rows)")

    def test_dataframe_round_trip(self):
        """Test round-trip through pandas DataFrame."""
        import pandas as pd
        
        original_data = [
            {"name": "Alice", "age": 30, "score": 95.5},
            {"name": "Bob", "age": 25, "score": 87.2},
            {"name": "Carol", "age": 35, "score": 92.8}
        ]
        
        # Original → DataFrame → JSON-Tables → DataFrame → Records
        df1 = pd.DataFrame(original_data)
        json_tables = JSONTablesEncoder.from_dataframe(df1)
        df2 = JSONTablesDecoder.to_dataframe(json_tables)
        converted_data = df2.to_dict('records')
        
        # Verify DataFrame round-trip preserved
        self.assertDataEqual(original_data, converted_data, "DataFrame round-trip")

    def test_numpy_types_round_trip(self):
        """Test automatic numpy type handling and round-trip preservation."""
        try:
            import numpy as np
        except ImportError:
            self.skipTest("Numpy not available")
        
        original_data = [
            {
                "int64_field": np.int64(42),
                "float64_field": np.float64(3.14159),
                "bool_field": np.bool_(True),
                "nan_field": np.nan,
                "str_field": np.str_("numpy string"),
                "int32_field": np.int32(100),
                "normal_field": "regular python string"
            },
            {
                "int64_field": np.int64(99),
                "float64_field": np.float64(2.71828),
                "bool_field": np.bool_(False),
                "nan_field": np.float64(42.0),
                "str_field": np.str_("another string"),
                "int32_field": np.int32(200),
                "normal_field": "another regular string"
            }
        ]
        
        # Convert to JSON-Tables with automatic numpy handling
        json_tables = JSONTablesEncoder.from_records(original_data, auto_numpy=True)
        
        # Verify numpy metadata was stored
        self.assertIn("_numpy_metadata", json_tables)
        self.assertIn("dtypes", json_tables["_numpy_metadata"])
        
        # Convert back to records
        converted_data = JSONTablesDecoder.to_records(json_tables, auto_numpy=True)
        
        # Verify data integrity (numpy types converted to Python types)
        expected_data = [
            {
                "int64_field": 42,
                "float64_field": 3.14159,
                "bool_field": True,
                "nan_field": None,  # numpy.nan → None
                "str_field": "numpy string",
                "int32_field": 100,
                "normal_field": "regular python string"
            },
            {
                "int64_field": 99,
                "float64_field": 2.71828,
                "bool_field": False,
                "nan_field": 42.0,
                "str_field": "another string", 
                "int32_field": 200,
                "normal_field": "another regular string"
            }
        ]
        
        self.assertDataEqual(expected_data, converted_data, "Numpy types with auto-conversion")

    def test_pandas_dataframe_with_numpy_round_trip(self):
        """Test pandas DataFrame with numpy types round-trip."""
        try:
            import pandas as pd
            import numpy as np
        except ImportError:
            self.skipTest("Pandas/Numpy not available")
        
        # Create DataFrame with various numpy types
        df = pd.DataFrame({
            'int_col': pd.array([1, 2, None], dtype='Int64'),
            'float_col': [1.1, np.nan, 3.3],
            'bool_col': pd.array([True, False, None], dtype='boolean'),
            'str_col': ['a', None, 'c']
        })
        
        # Convert to JSON-Tables
        json_tables = JSONTablesEncoder.from_dataframe(df, auto_numpy=True)
        
        # Verify numpy metadata was stored
        self.assertIn("_numpy_metadata", json_tables)
        
        # Convert back to DataFrame
        restored_df = JSONTablesDecoder.to_dataframe(json_tables, auto_numpy=True)
        
        # Convert to records for comparison
        original_records = df.to_dict('records')
        restored_records = restored_df.to_dict('records')
        
        # Verify data integrity
        self.assertDataEqual(original_records, restored_records, "Pandas DataFrame with numpy types")


def run_round_trip_tests():
    """Run all round-trip tests and report results."""
    print("🔄 JSON-Tables Round-Trip Data Integrity Tests")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestJSONTablesRoundTrip)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print(f"\n📊 Test Results Summary:")
    print(f"  Total tests: {result.testsRun}")
    print(f"  Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  Failed: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print(f"\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    if len(result.failures) == 0 and len(result.errors) == 0:
        print(f"\n✅ All round-trip tests passed! Data integrity confirmed.")
        return True
    else:
        print(f"\n❌ Some tests failed. Data integrity issues detected.")
        return False


if __name__ == "__main__":
    success = run_round_trip_tests()
    sys.exit(0 if success else 1) 