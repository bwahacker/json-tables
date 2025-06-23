#!/usr/bin/env python3
"""
Smart Parallel JSON-Tables Implementation
Uses multithreading only when it provides actual benefits
"""

import time
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union
import concurrent.futures
import threading
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# Import our existing utilities
try:
    from .core import JSONTablesEncoder, JSONTablesDecoder
    from .numpy_utils import convert_numpy_types, is_pandas_available
    from .profiling import profile_operation, profile_function
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    def convert_numpy_types(data): return data
    def profile_operation(name): 
        from contextlib import nullcontext
        return nullcontext()
    def profile_function(name=None): 
        def decorator(func): return func
        return decorator

# Optimal thresholds for when parallelization helps
PARALLEL_ROW_THRESHOLD = 10000      # Parallelize if >10K rows
PARALLEL_COLUMN_THRESHOLD = 50      # Parallelize if >50 columns  
WIDE_DATA_THRESHOLD = 100           # Very wide data (columnar wins big)
MAX_WORKERS = min(6, (multiprocessing.cpu_count() or 1))

def _should_use_parallel_rows(df: pd.DataFrame) -> bool:
    """Determine if row-wise parallelization would help."""
    return len(df) > PARALLEL_ROW_THRESHOLD

def _should_use_parallel_columns(df: pd.DataFrame) -> bool:
    """Determine if column-wise parallelization would help."""
    return len(df.columns) > PARALLEL_COLUMN_THRESHOLD

def _is_wide_data(df: pd.DataFrame) -> bool:
    """Check if data is very wide (many columns, fewer rows)."""
    return len(df.columns) > WIDE_DATA_THRESHOLD

def _process_column_chunk(args):
    """Process a chunk of columns in parallel."""
    columns, df_chunk, auto_numpy = args
    
    result = {}
    for col in columns:
        if col not in df_chunk.columns:
            continue
            
        # Convert column to list, handling NaN
        col_data = df_chunk[col]
        if hasattr(col_data, 'isna'):
            values = col_data.values
            nan_mask = pd.isna(values)
            col_values = values.tolist()
        else:
            col_values = list(col_data)
            nan_mask = [pd.isna(v) for v in col_values]
        
        # Handle NaN and numpy types
        for i in range(len(col_values)):
            if nan_mask[i]:
                col_values[i] = None
            elif auto_numpy:
                col_values[i] = convert_numpy_types(col_values[i])
        
        result[col] = col_values
    
    return result

class SmartParallelJSONTablesEncoder:
    """Smart parallel encoder that uses threading only when beneficial."""
    
    @staticmethod
    @profile_function("SmartParallelJSONTablesEncoder.from_dataframe")
    def from_dataframe(
        df: pd.DataFrame,
        page_size: Optional[int] = None,
        current_page: int = 0,
        columnar: bool = False,
        auto_numpy: bool = True,
        force_parallel: bool = False
    ) -> Dict[str, Any]:
        """
        Smart DataFrame to JSON Tables conversion with intelligent parallelization.
        
        Args:
            df: Input DataFrame
            page_size: Number of rows per page
            current_page: Current page number
            columnar: Use columnar format
            auto_numpy: Handle numpy types automatically
            force_parallel: Force parallel processing (for testing)
            
        Returns:
            Dictionary in JSON Tables format
        """
        
        # Handle empty DataFrame
        if df.empty:
            return {
                "__dict_type": "table",
                "cols": list(df.columns),
                "row_data": [],
                "current_page": 0,
                "total_pages": 1,
                "page_rows": 0
            }
        
        # Determine if we should use parallel processing
        use_column_parallel = force_parallel or _should_use_parallel_columns(df)
        use_row_parallel = force_parallel or _should_use_parallel_rows(df)
        is_wide = _is_wide_data(df)
        
        # Auto-select best format for very wide data
        if is_wide and not columnar:
            columnar = True
            print(f"🚀 Auto-selected columnar format for wide data ({len(df.columns)} columns)")
        
        # Use single-threaded for small datasets (it's faster!)
        if not force_parallel and not use_column_parallel and not use_row_parallel:
            if CORE_AVAILABLE:
                return JSONTablesEncoder.from_dataframe(
                    df, page_size, current_page, columnar, auto_numpy
                )
        
        # Continue with parallel processing for large datasets
        with profile_operation("smart_parallel_validation"):
            cols = list(df.columns)
            
            # Handle pagination
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
            if auto_numpy and is_pandas_available():
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
            # Smart columnar conversion
            with profile_operation("smart_columnar_conversion"):
                if use_column_parallel and len(cols) > 6:
                    result["column_data"] = SmartParallelJSONTablesEncoder._parallel_columnar_conversion(
                        page_df, cols, auto_numpy
                    )
                    print(f"⚡ Used parallel columnar processing ({len(cols)} columns)")
                else:
                    result["column_data"] = SmartParallelJSONTablesEncoder._fast_columnar_conversion(
                        page_df, cols, auto_numpy
                    )
                result["row_data"] = None
        else:
            # Smart row conversion
            with profile_operation("smart_row_conversion"):
                result["row_data"] = SmartParallelJSONTablesEncoder._smart_row_conversion(
                    page_df, auto_numpy, use_row_parallel
                )
        
        return result
    
    @staticmethod
    def _parallel_columnar_conversion(
        df: pd.DataFrame, 
        cols: List[str], 
        auto_numpy: bool = True
    ) -> Dict[str, List[Any]]:
        """Parallel columnar conversion for wide datasets."""
        
        if len(df) == 0:
            return {col: [] for col in cols}
        
        # Split columns into chunks for parallel processing
        chunk_size = max(8, len(cols) // MAX_WORKERS)
        column_chunks = []
        for i in range(0, len(cols), chunk_size):
            chunk_cols = cols[i:i + chunk_size]
            column_chunks.append((chunk_cols, df, auto_numpy))
        
        # Process chunks in parallel
        column_data = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            chunk_results = list(executor.map(_process_column_chunk, column_chunks))
        
        # Combine results
        for chunk_result in chunk_results:
            column_data.update(chunk_result)
        
        return column_data
    
    @staticmethod
    def _fast_columnar_conversion(
        df: pd.DataFrame, 
        cols: List[str], 
        auto_numpy: bool = True
    ) -> Dict[str, List[Any]]:
        """Fast single-threaded columnar conversion."""
        
        if len(df) == 0:
            return {col: [] for col in cols}
        
        column_data = {}
        for col in cols:
            # Fast vectorized conversion
            col_series = df[col]
            values = col_series.values
            nan_mask = pd.isna(values)
            col_values = values.tolist()
            
            # Vectorized None replacement
            for i in range(len(col_values)):
                if nan_mask[i]:
                    col_values[i] = None
                elif auto_numpy:
                    col_values[i] = convert_numpy_types(col_values[i])
            
            column_data[col] = col_values
        
        return column_data
    
    @staticmethod
    def _smart_row_conversion(
        df: pd.DataFrame, 
        auto_numpy: bool = True,
        use_parallel: bool = False
    ) -> List[List[Any]]:
        """Smart row conversion with optional parallelization."""
        
        if len(df) == 0:
            return []
        
        if use_parallel and len(df) > PARALLEL_ROW_THRESHOLD:
            print(f"⚡ Using parallel row processing ({len(df):,} rows)")
            return SmartParallelJSONTablesEncoder._parallel_row_conversion(df, auto_numpy)
        else:
            return SmartParallelJSONTablesEncoder._fast_row_conversion(df, auto_numpy)
    
    @staticmethod
    def _fast_row_conversion(df: pd.DataFrame, auto_numpy: bool = True) -> List[List[Any]]:
        """Optimized single-threaded row conversion."""
        
        # Use vectorized operations for speed
        values = df.values
        nan_mask = pd.isna(values)
        row_data = values.tolist()
        
        # Vectorized None replacement
        for i in range(len(row_data)):
            row = row_data[i]
            mask_row = nan_mask[i]
            for j in range(len(row)):
                if mask_row[j]:
                    row[j] = None
                elif auto_numpy:
                    row[j] = convert_numpy_types(row[j])
        
        return row_data
    
    @staticmethod
    def _parallel_row_conversion(df: pd.DataFrame, auto_numpy: bool = True) -> List[List[Any]]:
        """Parallel row conversion for very large datasets."""
        
        # Split into chunks
        chunk_size = max(2000, len(df) // MAX_WORKERS)
        chunks = []
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size]
            chunks.append(chunk)
        
        def process_chunk(chunk_df):
            return SmartParallelJSONTablesEncoder._fast_row_conversion(chunk_df, auto_numpy)
        
        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            chunk_results = list(executor.map(process_chunk, chunks))
        
        # Combine results
        row_data = []
        for chunk_result in chunk_results:
            row_data.extend(chunk_result)
        
        return row_data

# Convenience functions
def df_to_jt_smart(df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """Smart DataFrame to JSON-Tables conversion with intelligent parallelization."""
    return SmartParallelJSONTablesEncoder.from_dataframe(df, **kwargs)

def benchmark_smart_parallel():
    """Benchmark the smart parallel implementation."""
    print("🧠 SMART PARALLEL JSON-TABLES BENCHMARK")
    print("=" * 60)
    
    test_cases = [
        ("Small (1K×5)", 1000, 5),
        ("Medium (5K×20)", 5000, 20), 
        ("Large (15K×20)", 15000, 20),
        ("Wide (2K×100)", 2000, 100),
        ("Very Wide (500×200)", 500, 200)
    ]
    
    for name, rows, cols in test_cases:
        print(f"\n📊 {name} dataset:")
        
        # Create test data
        np.random.seed(42)
        df = pd.DataFrame({
            f'col_{i}': np.random.rand(rows) for i in range(cols)
        })
        
        # Single-threaded baseline
        if CORE_AVAILABLE:
            start = time.perf_counter()
            json_table_st = JSONTablesEncoder.from_dataframe(df, columnar=True)
            st_time = (time.perf_counter() - start) * 1000
        else:
            st_time = 0
        
        # Smart parallel
        start = time.perf_counter()
        json_table_smart = df_to_jt_smart(df, columnar=True)
        smart_time = (time.perf_counter() - start) * 1000
        
        # Calculate speedup
        if st_time > 0:
            speedup = st_time / smart_time
            print(f"   Single-threaded: {st_time:6.1f}ms")
            print(f"   Smart parallel:  {smart_time:6.1f}ms ({speedup:.1f}x speedup)")
        else:
            print(f"   Smart parallel:  {smart_time:6.1f}ms")

if __name__ == "__main__":
    benchmark_smart_parallel() 