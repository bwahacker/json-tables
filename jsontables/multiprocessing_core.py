#!/usr/bin/env python3
"""
Multiprocessing JSON-Tables Core Operations
Uses multiple processes to overcome Python's GIL limitations
"""

import json
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
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

def _process_chunk_worker(chunk_data):
    """
    Worker function that processes a chunk of DataFrame data.
    This runs in a separate process to avoid GIL limitations.
    """
    chunk_values, auto_numpy = chunk_data
    
    if len(chunk_values) == 0:
        return []
    
    # Convert to list and handle NaN values
    row_data = []
    for row in chunk_values:
        processed_row = []
        for value in row:
            if pd.isna(value):
                processed_row.append(None)
            elif auto_numpy and UTILS_AVAILABLE:
                processed_row.append(convert_numpy_types(value))
            else:
                processed_row.append(value)
        row_data.append(processed_row)
    
    return row_data

def _process_column_worker(column_data):
    """
    Worker function that processes a single column.
    This runs in a separate process for true parallelism.
    """
    col_name, col_values, auto_numpy = column_data
    
    # Handle NaN values and convert types
    processed_values = []
    for value in col_values:
        if pd.isna(value):
            processed_values.append(None)
        elif auto_numpy and UTILS_AVAILABLE:
            processed_values.append(convert_numpy_types(value))
        else:
            processed_values.append(value)
    
    return col_name, processed_values

class MultiprocessingJSONTablesEncoder:
    """Multiprocessing encoder for JSON-Tables with true parallelism."""
    
    @staticmethod
    @profile_function("MultiprocessingJSONTablesEncoder.from_dataframe")
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
        Multiprocessing DataFrame to JSON Tables conversion.
        
        Args:
            df: Input DataFrame
            page_size: Number of rows per page
            current_page: Current page number
            columnar: Use columnar format
            auto_numpy: Handle numpy types automatically
            max_workers: Maximum number of worker processes
            chunk_size: Size of chunks for parallel processing
            
        Returns:
            Dictionary in JSON Tables format
        """
        if max_workers is None:
            max_workers = MAX_WORKERS
        
        with profile_operation("mp_dataframe_validation"):
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
        with profile_operation("mp_pagination_logic"):
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
            # Multiprocessing columnar conversion
            with profile_operation("mp_columnar_conversion"):
                result["column_data"] = MultiprocessingJSONTablesEncoder._fast_columnar_conversion_mp(
                    page_df, cols, auto_numpy, max_workers
                )
                result["row_data"] = None
        else:
            # Multiprocessing row conversion
            with profile_operation("mp_row_conversion"):
                result["row_data"] = MultiprocessingJSONTablesEncoder._fast_row_conversion_mp(
                    page_df, auto_numpy, max_workers, chunk_size
                )
        
        return result
    
    @staticmethod
    def _fast_row_conversion_mp(
        df: pd.DataFrame, 
        auto_numpy: bool = True, 
        max_workers: int = MAX_WORKERS,
        chunk_size: Optional[int] = None
    ) -> List[List[Any]]:
        """
        Ultra-fast multiprocessing row conversion.
        
        Splits DataFrame into chunks and processes them in separate processes.
        """
        if len(df) == 0:
            return []
        
        # Determine optimal chunk size
        if chunk_size is None:
            # Aim for roughly 2000-5000 rows per chunk for multiprocessing
            chunk_size = max(2000, len(df) // max_workers)
        
        # For small DataFrames, use single process
        if len(df) < 5000 or max_workers == 1:
            return _process_chunk_worker((df.values, auto_numpy))
        
        # Split DataFrame into chunks (convert to raw values for pickling)
        chunks = []
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size]
            chunks.append((chunk.values, auto_numpy))
        
        # Process chunks in parallel using separate processes
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                chunk_results = list(executor.map(_process_chunk_worker, chunks))
            
            # Combine results
            row_data = []
            for chunk_result in chunk_results:
                row_data.extend(chunk_result)
            
            return row_data
        except Exception as e:
            # Fallback to single-threaded if multiprocessing fails
            print(f"Multiprocessing failed, falling back to single process: {e}")
            return _process_chunk_worker((df.values, auto_numpy))
    
    @staticmethod
    def _fast_columnar_conversion_mp(
        df: pd.DataFrame, 
        cols: List[str], 
        auto_numpy: bool = True,
        max_workers: int = MAX_WORKERS
    ) -> Dict[str, List[Any]]:
        """
        Ultra-fast multiprocessing columnar conversion.
        
        Processes each column in separate processes.
        """
        if len(df) == 0:
            return {col: [] for col in cols}
        
        # For few columns, use single process
        if len(cols) < 6 or max_workers == 1:
            column_data = {}
            for col in cols:
                _, col_values = _process_column_worker((col, df[col].values, auto_numpy))
                column_data[col] = col_values
            return column_data
        
        # Prepare column processing arguments (use raw values for pickling)
        column_args = [(col, df[col].values, auto_numpy) for col in cols]
        
        # Process columns in parallel using separate processes
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                column_results = list(executor.map(_process_column_worker, column_args))
            
            # Build result dictionary
            column_data = {}
            for col_name, col_values in column_results:
                column_data[col_name] = col_values
            
            return column_data
        except Exception as e:
            # Fallback to single process if multiprocessing fails
            print(f"Multiprocessing failed, falling back to single process: {e}")
            column_data = {}
            for col in cols:
                _, col_values = _process_column_worker((col, df[col].values, auto_numpy))
                column_data[col] = col_values
            return column_data

# Convenience functions for multiprocessing operations
def df_to_jt_mp(df: pd.DataFrame, max_workers: Optional[int] = None, **kwargs) -> Dict[str, Any]:
    """Multiprocessing DataFrame to JSON-Tables conversion."""
    return MultiprocessingJSONTablesEncoder.from_dataframe(
        df, max_workers=max_workers, **kwargs
    )

def benchmark_multiprocessing_vs_threading():
    """Compare multiprocessing vs threading performance."""
    print("🔥 MULTIPROCESSING vs THREADING COMPARISON")
    print("=" * 60)
    
    # Import threading version for comparison
    try:
        from .multithreaded_core import df_to_jt_mt as df_to_jt_threading
    except ImportError:
        print("⚠️ Threading version not available")
        return
    
    # Import single-threaded version
    try:
        from .core import JSONTablesEncoder
    except ImportError:
        print("⚠️ Single-threaded version not available")
        return
    
    test_sizes = [5000, 10000, 20000]
    
    for size in test_sizes:
        print(f"\n📊 Testing {size:,} rows × 50 columns:")
        
        # Create test data
        df = pd.DataFrame({
            f'col_{i}': np.random.rand(size) for i in range(50)
        })
        
        # Single-threaded
        start = time.perf_counter()
        json_table_st = JSONTablesEncoder.from_dataframe(df)
        st_time = (time.perf_counter() - start) * 1000
        
        # Threading
        start = time.perf_counter()
        json_table_threading = df_to_jt_threading(df, columnar=True)
        threading_time = (time.perf_counter() - start) * 1000
        
        # Multiprocessing
        start = time.perf_counter()
        json_table_mp = df_to_jt_mp(df, columnar=True)
        mp_time = (time.perf_counter() - start) * 1000
        
        # Calculate speedups
        threading_speedup = st_time / threading_time if threading_time > 0 else 0
        mp_speedup = st_time / mp_time if mp_time > 0 else 0
        mp_vs_threading = threading_time / mp_time if mp_time > 0 else 0
        
        print(f"   Single-threaded:  {st_time:6.1f}ms")
        print(f"   Threading:        {threading_time:6.1f}ms ({threading_speedup:.1f}x)")
        print(f"   Multiprocessing:  {mp_time:6.1f}ms ({mp_speedup:.1f}x)")
        print(f"   MP vs Threading:  {mp_vs_threading:.1f}x faster")

if __name__ == "__main__":
    benchmark_multiprocessing_vs_threading() 