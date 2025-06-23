#!/usr/bin/env python3
"""
Comprehensive round-trip tests for JSON-Tables.

Tests that data can be converted to JSON-Tables format and back
without any loss of information or data corruption.

NOW WITH BULLETPROOF CELL-BY-CELL VALIDATION!
"""

import sys
import os
import unittest
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from jsontables.core import JSONTablesEncoder, JSONTablesDecoder
from jsontables.data_integrity import DataIntegrityValidator, DataIntegrityError
import pandas as pd


class TestJSONTablesRoundTrip(unittest.TestCase):
    """Test round-trip conversion: data → JSON-Tables → data"""
    
    def validate_dataframe_integrity(self, original_df: pd.DataFrame, restored_df: pd.DataFrame, test_name: str):
        """Use bulletproof cell-by-cell validation instead of weak shape comparison."""
        try:
            DataIntegrityValidator.validate_dataframe_equality(
                original_df, restored_df, operation_name=test_name
            )
        except DataIntegrityError as e:
            self.fail(f"Data integrity validation failed for {test_name}: {e}")
    
    def validate_records_integrity(self, original_records: List[Dict], restored_records: List[Dict], test_name: str):
        """Use bulletproof record-by-record validation."""
        try:
            DataIntegrityValidator.validate_records_equality(
                original_records, restored_records, operation_name=test_name
            )
        except DataIntegrityError as e:
            self.fail(f"Record integrity validation failed for {test_name}: {e}")

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
        
        # Bulletproof validation
        self.validate_records_integrity(original_data, converted_data, "Basic data types")

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
        self.validate_records_integrity(original_data, converted_data, "Null values")

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
        self.validate_records_integrity(original_data, converted_data, "Mixed data types")

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
        self.validate_records_integrity(original_data, converted_data, "Categorical data")

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
        self.validate_records_integrity(original_data, converted_data, "Sparse data")

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
        self.validate_records_integrity(original_data, converted_data, "Edge case values")

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
        self.validate_records_integrity(original_data, converted_data, "Single row")

    def test_empty_data_round_trip(self):
        """Test empty dataset."""
        original_data = []
        
        # Convert to JSON-Tables
        json_tables = JSONTablesEncoder.from_records(original_data)
        
        # Convert back to records
        converted_data = JSONTablesDecoder.to_records(json_tables)
        
        # Verify empty data handled correctly
        self.validate_records_integrity(original_data, converted_data, "Empty dataset")

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
        self.validate_records_integrity(expected_data, converted_data, "Inconsistent schema")

    def test_dataframe_round_trip_bulletproof(self):
        """Test DataFrame round-trip with bulletproof cell-by-cell validation."""
        # Create DataFrame with challenging data
        df = pd.DataFrame({
            'strings': ['hello', 'world', None, '', 'unicode: café'],
            'integers': [42, -17, 0, None, 999999999],
            'floats': [3.14159, -2.71828, 0.0, None, 1e-308],
            'booleans': [True, False, None, True, False],
            'numpy_nans': [pd.NA, 1.0, float('nan'), 2.0, None],
            'infinities': [float('inf'), float('-inf'), 1.0, float('inf'), -float('inf')],
        })
        
        # Convert to JSON-Tables and back
        json_tables = JSONTablesEncoder.from_dataframe(df)
        df_restored = JSONTablesDecoder.to_dataframe(json_tables)
        
        # Bulletproof cell-by-cell validation
        self.validate_dataframe_integrity(df, df_restored, "DataFrame with challenging data")

    def test_extreme_edge_cases_bulletproof(self):
        """Test extreme edge cases with bulletproof validation."""
        import numpy as np
        
        # Create DataFrame with every possible edge case
        df = pd.DataFrame({
            'tiny_numbers': [1e-323, 5e-324, 2.225e-308, 1e-308, 0.0],
            'huge_numbers': [1.797e308, 1e308, 9.999e307, 1e200, 1e100],
            'special_values': [np.nan, np.inf, -np.inf, np.pi, np.e],
            'numpy_types': [np.int64(42), np.float32(3.14), np.bool_(True), None, np.str_('test')],
            'mixed_nulls': [None, pd.NA, np.nan, 0, ''],
            'edge_strings': ['', ' ', '\n', '\t', 'unicode: 北京 café'],
        })
        
        # Test multiple implementations
        implementations = [
            ("Standard", lambda df: JSONTablesEncoder.from_dataframe(df)),
            ("Columnar", lambda df: JSONTablesEncoder.from_dataframe(df, columnar=True)),
        ]
        
        for impl_name, convert_func in implementations:
            with self.subTest(implementation=impl_name):
                json_tables = convert_func(df)
                df_restored = JSONTablesDecoder.to_dataframe(json_tables)
                
                # Bulletproof validation of every single cell
                self.validate_dataframe_integrity(df, df_restored, f"{impl_name} extreme edge cases")

    def test_high_performance_implementation_bulletproof(self):
        """Test high-performance implementation with bulletproof validation."""
        try:
            from jsontables.high_performance_core import df_to_jt_hp
        except ImportError:
            self.skipTest("High-performance implementation not available")
        
        import numpy as np
        
        # Create challenging test data
        df = pd.DataFrame({
            'mixed_data': [42, 'text', True, None, 3.14, np.nan, np.inf],
            'numpy_scalars': [np.int64(100), np.float32(2.5), np.bool_(False), None, np.str_('hello'), 0, -1],
            'edge_floats': [0.0, -0.0, 1e-300, 1e300, np.finfo(float).eps, np.pi, np.e],
        })
        
        # Test different HP configurations
        configs = [
            ("HP default", {}),
            ("HP skip numpy", {"skip_numpy_conversion": True}),
            ("HP chunked", {"max_workers": 4, "chunk_size": 100}),
        ]
        
        for config_name, kwargs in configs:
            with self.subTest(config=config_name):
                json_tables = df_to_jt_hp(df, **kwargs)
                df_restored = JSONTablesDecoder.to_dataframe(json_tables)
                
                # Bulletproof validation
                self.validate_dataframe_integrity(df, df_restored, f"High-performance {config_name}")

    def test_large_dataset_bulletproof(self):
        """Test large dataset with bulletproof validation."""
        import numpy as np
        
        # Generate large test data (1000 rows)
        np.random.seed(42)
        df = pd.DataFrame({
            'id': [f'ID_{i:06d}' for i in range(1000)],
            'values': np.random.randn(1000),
            'categories': np.random.choice(['A', 'B', 'C', None], 1000, p=[0.4, 0.3, 0.2, 0.1]),
            'booleans': np.random.choice([True, False, None], 1000, p=[0.5, 0.4, 0.1]),
            'integers': np.random.randint(-1000, 1000, 1000),
        })
        
        # Add some edge cases
        df.loc[100, 'values'] = np.inf
        df.loc[101, 'values'] = -np.inf
        df.loc[102, 'values'] = np.nan
        
        # Convert and validate
        json_tables = JSONTablesEncoder.from_dataframe(df)
        df_restored = JSONTablesDecoder.to_dataframe(json_tables)
        
        # Bulletproof validation of all 5,000 cells
        self.validate_dataframe_integrity(df, df_restored, "Large dataset (1000x5=5000 cells)")

    def test_real_world_scenario_bulletproof(self):
        """Test realistic real-world data scenario."""
        import numpy as np
        
        # Simulate customer data with realistic patterns
        customers = []
        for i in range(500):
            customer = {
                'customer_id': f'CUST_{i:06d}',
                'name': f'Customer {i}' if i % 20 != 0 else None,  # 5% missing
                'age': np.random.randint(18, 80) if i % 15 != 0 else None,  # ~7% missing
                'balance': round(np.random.normal(1000, 500), 2) if i % 12 != 0 else None,
                'is_premium': bool(np.random.choice([True, False], p=[0.3, 0.7])),
                'signup_year': np.random.randint(2020, 2024),
            }
            customers.append(customer)
        
        df = pd.DataFrame(customers)
        
        # Convert and validate
        json_tables = JSONTablesEncoder.from_dataframe(df)
        df_restored = JSONTablesDecoder.to_dataframe(json_tables)
        
        # Bulletproof validation
        self.validate_dataframe_integrity(df, df_restored, "Real-world customer data")


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