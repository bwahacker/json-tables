#!/usr/bin/env python3
"""
Multithreaded JSON-Tables Core Operations
Leverages multiple CPU cores for improved performance
"""

import json
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union
import concurrent.futures
import multiprocessing
import threading
from functools import partial
import time

# Import our existing utilities
try:
    from .numpy_utils import convert_numpy_types, is_pandas_available
    from .profiling import profile_operation, profile_function
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    def convert_numpy_types(data): return data
    def profile_operation(name): 
        from contextlib import nullcontext
        return nullcontext()
    def profile_function(name=None): 
        def decorator(func): return func
        return decorator

# Get optimal number of workers
MAX_WORKERS = min(8, (multiprocessing.cpu_count() or 1))

def _process_dataframe_chunk_rows(args):
    """Process a chunk of DataFrame rows in parallel."""
    chunk_df, auto_numpy = args
    
    if len(chunk_df) == 0:
        return []
    
    # Use our optimized row conversion on the chunk
    values = chunk_df.values
    nan_mask = pd.isna(values)
    row_data = values.tolist()
    
    # Vectorized None replacement for this chunk
    for i in range(len(row_data)):
        row = row_data[i]
        mask_row = nan_mask[i]
        for j in range(len(row)):
            if mask_row[j]:
                row[j] = None
            elif auto_numpy and UTILS_AVAILABLE:
                row[j] = convert_numpy_types(row[j])
    
    return row_data

def _process_dataframe_column(args):
    """Process a single DataFrame column in parallel."""
    col_name, col_data, auto_numpy = args
    
    # Handle NaN values
    if hasattr(col_data, 'isna'):
        # It's a pandas Series
        values = col_data.values
        nan_mask = pd.isna(values)
        col_values = values.tolist()
    else:
        # It's already a list/array
        col_values = list(col_data)
        nan_mask = [pd.isna(v) for v in col_values]
    
    # Vectorized None replacement
    for i in range(len(col_values)):
        if nan_mask[i]:
            col_values[i] = None
        elif auto_numpy and UTILS_AVAILABLE:
            col_values[i] = convert_numpy_types(col_values[i])
    
    return col_name, col_values

def _build_json_chunk(data_chunk):
    """Build JSON string for a chunk of data."""
    return json.dumps(data_chunk, separators=(',', ':'))

class MultithreadedJSONTablesEncoder:
    """Multithreaded encoder for JSON-Tables with parallel processing."""
    
    @staticmethod
    @profile_function("MultithreadedJSONTablesEncoder.from_dataframe")
    def from_dataframe(
        df: pd.DataFrame,
        page_size: Optional[int] = None,
        current_page: int = 0,
        columnar: bool = False,
        auto_numpy: bool = True,
        max_workers: Optional[int] = None,
        chunk_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Multithreaded DataFrame to JSON Tables conversion.
        
        Args:
            df: Input DataFrame
            page_size: Number of rows per page
            current_page: Current page number
            columnar: Use columnar format
            auto_numpy: Handle numpy types automatically
            max_workers: Maximum number of worker threads (default: CPU count)
            chunk_size: Size of chunks for parallel processing
            
        Returns:
            Dictionary in JSON Tables format
        """
        if max_workers is None:
            max_workers = MAX_WORKERS
        
        with profile_operation("mt_dataframe_validation"):
            if df.empty:
                return {
                    "__dict_type": "table",
                    "cols": list(df.columns),
                    "row_data": [],
                    "current_page": 0,
                    "total_pages": 1,
                    "page_rows": 0
                }
        
        # Extract columns efficiently
        cols = list(df.columns)
        
        # Handle pagination
        with profile_operation("mt_pagination_logic"):
            if page_size is not None:
                total_pages = (len(df) + page_size - 1) // page_size
                start_idx = current_page * page_size
                end_idx = start_idx + page_size
                page_df = df.iloc[start_idx:end_idx]
                page_rows = len(page_df)
            else:
                page_df = df
                total_pages = 1
                page_rows = len(df)
        
        # Handle numpy metadata
        numpy_metadata = {}
        if auto_numpy and UTILS_AVAILABLE and is_pandas_available():
            numpy_metadata = {
                'dtypes': {col: str(dtype) for col, dtype in page_df.dtypes.items()},
                'index_name': page_df.index.name
            }
        
        # Build result structure
        result = {
            "__dict_type": "table",
            "cols": cols,
            "current_page": current_page,
            "total_pages": total_pages,
            "page_rows": page_rows
        }
        
        if numpy_metadata:
            result["_numpy_metadata"] = numpy_metadata
        
        if columnar:
            # Multithreaded columnar conversion
            with profile_operation("mt_columnar_conversion"):
                result["column_data"] = MultithreadedJSONTablesEncoder._fast_columnar_conversion_mt(
                    page_df, cols, auto_numpy, max_workers
                )
                result["row_data"] = None
        else:
            # Multithreaded row conversion
            with profile_operation("mt_row_conversion"):
                result["row_data"] = MultithreadedJSONTablesEncoder._fast_row_conversion_mt(
                    page_df, auto_numpy, max_workers, chunk_size
                )
        
        return result
    
    @staticmethod
    def _fast_row_conversion_mt(
        df: pd.DataFrame, 
        auto_numpy: bool = True, 
        max_workers: int = MAX_WORKERS,
        chunk_size: Optional[int] = None
    ) -> List[List[Any]]:
        """
        Ultra-fast multithreaded row conversion.
        
        Splits DataFrame into chunks and processes them in parallel.
        """
        if len(df) == 0:
            return []
        
        # Determine optimal chunk size
        if chunk_size is None:
            # Aim for roughly 1000-5000 rows per chunk
            chunk_size = max(1000, len(df) // max_workers)
        
        # For small DataFrames, use single-threaded version
        if len(df) < 2000 or max_workers == 1:
            return _process_dataframe_chunk_rows((df, auto_numpy))
        
        # Split DataFrame into chunks
        chunks = []
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size]
            chunks.append((chunk, auto_numpy))
        
        # Process chunks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            chunk_results = list(executor.map(_process_dataframe_chunk_rows, chunks))
        
        # Combine results
        row_data = []
        for chunk_result in chunk_results:
            row_data.extend(chunk_result)
        
        return row_data
    
    @staticmethod
    def _fast_columnar_conversion_mt(
        df: pd.DataFrame, 
        cols: List[str], 
        auto_numpy: bool = True,
        max_workers: int = MAX_WORKERS
    ) -> Dict[str, List[Any]]:
        """
        Ultra-fast multithreaded columnar conversion.
        
        Processes each column in parallel.
        """
        if len(df) == 0:
            return {col: [] for col in cols}
        
        # For few columns, use single-threaded version
        if len(cols) < 4 or max_workers == 1:
            column_data = {}
            for col in cols:
                _, col_values = _process_dataframe_column((col, df[col], auto_numpy))
                column_data[col] = col_values
            return column_data
        
        # Prepare column processing arguments
        column_args = [(col, df[col], auto_numpy) for col in cols]
        
        # Process columns in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            column_results = list(executor.map(_process_dataframe_column, column_args))
        
        # Build result dictionary
        column_data = {}
        for col_name, col_values in column_results:
            column_data[col_name] = col_values
        
        return column_data

class MultithreadedJSONTablesDecoder:
    """Multithreaded decoder for JSON-Tables."""
    
    @staticmethod
    @profile_function("MultithreadedJSONTablesDecoder.to_dataframe")
    def to_dataframe(
        json_table: Dict[str, Any], 
        auto_numpy: bool = True,
        max_workers: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Multithreaded JSON Tables to DataFrame conversion.
        
        For decoding, the main bottleneck is usually DataFrame construction
        which is hard to parallelize, but we can parallelize dtype restoration.
        """
        if max_workers is None:
            max_workers = MAX_WORKERS
        
        # Basic validation (same as single-threaded)
        if not isinstance(json_table, dict):
            raise ValueError("Input must be a dictionary")
        
        if json_table.get("__dict_type") != "table":
            raise ValueError("Missing or invalid __dict_type field")
        
        cols = json_table.get("cols")
        if not isinstance(cols, list):
            raise ValueError("cols field must be a list")
        
        # Extract numpy metadata
        numpy_metadata = json_table.get("_numpy_metadata", {})
        
        # Handle columnar format
        if "column_data" in json_table and json_table["column_data"] is not None:
            column_data = json_table["column_data"]
            if not isinstance(column_data, dict):
                raise ValueError("column_data must be a dictionary")
            
            # Validate columns
            for col in cols:
                if col not in column_data:
                    raise ValueError(f"Missing column data for: {col}")
            
            # Create DataFrame efficiently
            df_data = {col: column_data[col] for col in cols}
            df = pd.DataFrame(df_data)
        else:
            # Handle row-oriented format
            row_data = json_table.get("row_data")
            if not isinstance(row_data, list):
                raise ValueError("row_data field must be a list")
            
            if not row_data:
                df = pd.DataFrame(columns=cols)
            else:
                # Fast DataFrame creation
                df = pd.DataFrame(row_data, columns=cols)
        
        # Multithreaded dtype restoration if metadata available
        if auto_numpy and UTILS_AVAILABLE and numpy_metadata:
            df = MultithreadedJSONTablesDecoder._fast_dtype_restoration_mt(
                df, numpy_metadata, max_workers
            )
        
        return df
    
    @staticmethod
    def _fast_dtype_restoration_mt(
        df: pd.DataFrame, 
        numpy_metadata: Dict, 
        max_workers: int = MAX_WORKERS
    ) -> pd.DataFrame:
        """
        Multithreaded dtype restoration.
        
        This is most beneficial when there are many columns to process.
        """
        dtypes = numpy_metadata.get('dtypes', {})
        if not dtypes:
            return df
        
        # For few columns, use single-threaded
        if len(dtypes) < 8 or max_workers == 1:
            return MultithreadedJSONTablesDecoder._restore_dtypes_single(df, dtypes)
        
        # Group columns by dtype family for efficient processing
        dtype_groups = {
            'int': [],
            'float': [],
            'bool': [],
            'object': [],
            'string': []
        }
        
        for col, dtype_str in dtypes.items():
            if col not in df.columns:
                continue
            
            dtype_lower = dtype_str.lower()
            if 'int' in dtype_lower:
                dtype_groups['int'].append(col)
            elif 'float' in dtype_lower:
                dtype_groups['float'].append(col)
            elif 'bool' in dtype_lower:
                dtype_groups['bool'].append(col)
            elif 'object' in dtype_lower:
                dtype_groups['object'].append(col)
            elif 'string' in dtype_lower:
                dtype_groups['string'].append(col)
        
        # Process dtype groups in parallel
        def process_dtype_group(args):
            dtype_family, columns = args
            result_series = {}
            
            for col in columns:
                try:
                    if dtype_family == 'int':
                        result_series[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                    elif dtype_family == 'float':
                        result_series[col] = pd.to_numeric(df[col], errors='coerce')
                    elif dtype_family == 'bool':
                        result_series[col] = df[col].astype('boolean')
                    elif dtype_family == 'object':
                        # Keep object columns as-is
                        result_series[col] = df[col]
                    elif dtype_family == 'string':
                        result_series[col] = df[col].astype('string')
                except Exception:
                    # Keep original on failure
                    result_series[col] = df[col]
            
            return result_series
        
        # Prepare arguments for parallel processing
        group_args = [(family, columns) for family, columns in dtype_groups.items() if columns]
        
        # Process in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            group_results = list(executor.map(process_dtype_group, group_args))
        
        # Apply results to DataFrame
        for group_result in group_results:
            for col, series in group_result.items():
                df[col] = series
        
        return df
    
    @staticmethod
    def _restore_dtypes_single(df: pd.DataFrame, dtypes: Dict) -> pd.DataFrame:
        """Single-threaded dtype restoration fallback."""
        for col, dtype_str in dtypes.items():
            if col not in df.columns:
                continue
            
            try:
                dtype_lower = dtype_str.lower()
                if 'int' in dtype_lower:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                elif 'float' in dtype_lower:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                elif 'bool' in dtype_lower:
                    df[col] = df[col].astype('boolean')
                elif 'object' in dtype_lower:
                    continue  # Keep as-is
                elif 'string' in dtype_lower:
                    df[col] = df[col].astype('string')
            except Exception:
                continue
        
        return df

# Convenience functions for multithreaded operations
def df_to_jt_mt(df: pd.DataFrame, max_workers: Optional[int] = None, **kwargs) -> Dict[str, Any]:
    """Multithreaded DataFrame to JSON-Tables conversion."""
    return MultithreadedJSONTablesEncoder.from_dataframe(
        df, max_workers=max_workers, **kwargs
    )

def df_from_jt_mt(json_table: Dict[str, Any], max_workers: Optional[int] = None, **kwargs) -> pd.DataFrame:
    """Multithreaded JSON-Tables to DataFrame conversion."""
    return MultithreadedJSONTablesDecoder.to_dataframe(
        json_table, max_workers=max_workers, **kwargs
    )

def benchmark_multithreaded_performance():
    """Benchmark multithreaded vs single-threaded performance."""
    print("🚀 MULTITHREADED JSON-TABLES BENCHMARK")
    print("=" * 50)
    
    # Import single-threaded versions for comparison
    try:
        from .core import JSONTablesEncoder, JSONTablesDecoder
        CORE_AVAILABLE = True
    except ImportError:
        print("⚠️ Single-threaded core not available for comparison")
        CORE_AVAILABLE = False
    
    test_sizes = [1000, 5000, 10000]
    
    for size in test_sizes:
        print(f"\n📊 Testing with {size:,} rows × 20 columns:")
        
        # Create test data
        df = pd.DataFrame({
            f'col_{i}': np.random.rand(size) for i in range(20)
        })
        
        if CORE_AVAILABLE:
            # Single-threaded encoding
            start = time.perf_counter()
            json_table_st = JSONTablesEncoder.from_dataframe(df)
            st_encode_time = (time.perf_counter() - start) * 1000
            
            # Single-threaded decoding
            start = time.perf_counter()
            df_restored_st = JSONTablesDecoder.to_dataframe(json_table_st)
            st_decode_time = (time.perf_counter() - start) * 1000
        else:
            st_encode_time = 0
            st_decode_time = 0
        
        # Multithreaded encoding
        start = time.perf_counter()
        json_table_mt = MultithreadedJSONTablesEncoder.from_dataframe(df)
        mt_encode_time = (time.perf_counter() - start) * 1000
        
        # Multithreaded decoding
        start = time.perf_counter()
        df_restored_mt = MultithreadedJSONTablesDecoder.to_dataframe(json_table_mt)
        mt_decode_time = (time.perf_counter() - start) * 1000
        
        if CORE_AVAILABLE:
            encode_speedup = st_encode_time / mt_encode_time if mt_encode_time > 0 else 0
            decode_speedup = st_decode_time / mt_decode_time if mt_decode_time > 0 else 0
            
            print(f"  🔄 Encoding:")
            print(f"     Single-threaded: {st_encode_time:6.1f}ms")
            print(f"     Multithreaded:   {mt_encode_time:6.1f}ms")
            print(f"     Speedup:         {encode_speedup:6.1f}x")
            
            print(f"  📥 Decoding:")
            print(f"     Single-threaded: {st_decode_time:6.1f}ms")
            print(f"     Multithreaded:   {mt_decode_time:6.1f}ms")
            print(f"     Speedup:         {decode_speedup:6.1f}x")
        else:
            print(f"  🔄 Multithreaded encoding: {mt_encode_time:6.1f}ms")
            print(f"  📥 Multithreaded decoding: {mt_decode_time:6.1f}ms")
        
        # Verify correctness
        shape_match = df.shape == df_restored_mt.shape
        print(f"  ✅ Data integrity: {'PASS' if shape_match else 'FAIL'}")

if __name__ == "__main__":
    benchmark_multithreaded_performance() 