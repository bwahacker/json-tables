#!/usr/bin/env python3
"""
High-Performance JSON-Tables Core
Implements optimized threading with minimal numpy conversion overhead
"""

import pandas as pd
import numpy as np
import time
import concurrent.futures
from typing import Any, Dict, List, Optional
import multiprocessing

# Import existing utilities
try:
    from .profiling import profile_operation, profile_function
    PROFILING_AVAILABLE = True
except ImportError:
    PROFILING_AVAILABLE = False
    def profile_operation(name): 
        from contextlib import nullcontext
        return nullcontext()
    def profile_function(name=None): 
        def decorator(func): return func
        return decorator

# Optimal configuration
MAX_WORKERS = min(12, multiprocessing.cpu_count())  # Increase cap for larger datasets
OPTIMAL_CHUNK_SIZE = 500

# Adaptive threading thresholds
SINGLE_THREAD_THRESHOLD = 5000  # Use single-thread for datasets < 5K rows
MULTI_THREAD_THRESHOLD = 15000  # Use max threads for datasets > 15K rows

def _convert_numpy_minimal(value):
    """Minimal numpy conversion - only when absolutely necessary."""
    if hasattr(value, 'item'):  # numpy scalar - convert to Python type
        return value.item()
    return value  # Already a Python type

def _process_chunk_optimized(args):
    """
    Optimized chunk processor that minimizes numpy conversions.
    Does both NaN replacement AND minimal numpy conversion in worker thread.
    """
    chunk_start, chunk_rows, chunk_mask, skip_numpy = args
    
    for i in range(len(chunk_rows)):
        row = chunk_rows[i]
        mask_row = chunk_mask[i]
        for j in range(len(row)):
            if mask_row[j]:
                row[j] = None
            elif not skip_numpy:
                # Only convert if it's actually a numpy type
                row[j] = _convert_numpy_minimal(row[j])
    
    return chunk_start, chunk_rows

class HighPerformanceJSONTablesEncoder:
    """
    High-performance JSON-Tables encoder optimized for speed.
    Uses intelligent chunking and minimal numpy conversion.
    """
    
    @staticmethod
    @profile_function("HighPerformanceJSONTablesEncoder.from_dataframe")
    def from_dataframe(
        df: pd.DataFrame,
        page_size: Optional[int] = None,
        current_page: int = 0,
        columnar: bool = False,
        skip_numpy_conversion: bool = False,
        max_workers: Optional[int] = None,
        chunk_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        High-performance DataFrame to JSON Tables conversion.
        
        Args:
            df: Input DataFrame
            page_size: Number of rows per page
            current_page: Current page number  
            columnar: Use columnar format
            skip_numpy_conversion: Skip numpy type conversion for speed
            max_workers: Maximum worker threads (default: CPU count, capped at 8)
            chunk_size: Chunk size for parallel processing (default: 500)
            
        Returns:
            Dictionary in JSON Tables format
        """
        if max_workers is None:
            max_workers = MAX_WORKERS
        if chunk_size is None:
            chunk_size = OPTIMAL_CHUNK_SIZE
            
        with profile_operation("hp_dataframe_validation"):
            if df.empty:
                return {
                    "__dict_type": "table",
                    "cols": list(df.columns),
                    "row_data": [],
                    "current_page": 0,
                    "total_pages": 1,
                    "page_rows": 0
                }
        
        # Fast column extraction
        cols = list(df.columns)
        
        # Handle pagination efficiently
        with profile_operation("hp_pagination"):
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
        
        # Build result structure
        result = {
            "__dict_type": "table",
            "cols": cols,
            "current_page": current_page,
            "total_pages": total_pages,
            "page_rows": page_rows
        }
        
        if columnar:
            # Optimized columnar conversion
            with profile_operation("hp_columnar_conversion"):
                result["column_data"] = HighPerformanceJSONTablesEncoder._fast_columnar_conversion(
                    page_df, cols, skip_numpy_conversion, max_workers, chunk_size
                )
                result["row_data"] = None
        else:
            # Optimized row conversion with chunking
            with profile_operation("hp_row_conversion"):
                result["row_data"] = HighPerformanceJSONTablesEncoder._fast_row_conversion(
                    page_df, skip_numpy_conversion, max_workers, chunk_size
                )
        
        return result
    
    @staticmethod
    def _fast_row_conversion(
        df: pd.DataFrame,
        skip_numpy: bool = False,
        max_workers: int = MAX_WORKERS,
        chunk_size: int = OPTIMAL_CHUNK_SIZE
    ) -> List[List[Any]]:
        """
        Ultra-fast row conversion with adaptive threading.
        Uses single-thread for small datasets, multi-thread for large datasets.
        """
        if len(df) == 0:
            return []
        
        # Get raw data once
        values = df.values
        nan_mask = pd.isna(values)
        row_data = values.tolist()
        
        # Adaptive threading based on dataset size
        dataset_size = len(df)
        
        if dataset_size < SINGLE_THREAD_THRESHOLD:
            # Small datasets: single-threaded is faster (no threading overhead)
            return HighPerformanceJSONTablesEncoder._process_single_threaded(
                row_data, nan_mask, skip_numpy
            )
        elif dataset_size < MULTI_THREAD_THRESHOLD:
            # Medium datasets: use limited threading
            effective_workers = min(4, max_workers)
        else:
            # Large datasets: use full threading power
            effective_workers = max_workers
        
        # Chunked parallel processing for medium/large datasets
        chunks = []
        for i in range(0, len(row_data), chunk_size):
            end_idx = min(i + chunk_size, len(row_data))
            chunk_rows = [row[:] for row in row_data[i:end_idx]]  # Deep copy for thread safety
            chunk_mask = nan_mask[i:end_idx]
            chunks.append((i, chunk_rows, chunk_mask, skip_numpy))
        
        # Process chunks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as executor:
            chunk_results = list(executor.map(_process_chunk_optimized, chunks))
        
        # Reassemble results in correct order
        final_result = [None] * len(row_data)
        for start_idx, processed_rows in chunk_results:
            for i, row in enumerate(processed_rows):
                final_result[start_idx + i] = row
        
        return final_result
    
    @staticmethod
    def _process_single_threaded(row_data, nan_mask, skip_numpy):
        """Single-threaded fallback for small datasets."""
        result = []
        for i in range(len(row_data)):
            row = row_data[i][:]
            mask_row = nan_mask[i]
            for j in range(len(row)):
                if mask_row[j]:
                    row[j] = None
                elif not skip_numpy:
                    row[j] = _convert_numpy_minimal(row[j])
            result.append(row)
        return result
    
    @staticmethod
    def _fast_columnar_conversion(
        df: pd.DataFrame,
        cols: List[str],
        skip_numpy: bool = False,
        max_workers: int = MAX_WORKERS,
        chunk_size: int = OPTIMAL_CHUNK_SIZE
    ) -> Dict[str, List[Any]]:
        """
        Fast columnar conversion with column-wise threading.
        """
        if len(df) == 0:
            return {col: [] for col in cols}
        
        # For few columns, single-threaded is faster
        if len(cols) < 4:
            return HighPerformanceJSONTablesEncoder._columnar_single_threaded(df, cols, skip_numpy)
        
        # Column-wise parallel processing
        def process_column(col):
            col_series = df[col]
            values = col_series.values
            nan_mask = pd.isna(values)
            col_values = values.tolist()
            
            # Process this column
            for i in range(len(col_values)):
                if nan_mask[i]:
                    col_values[i] = None
                elif not skip_numpy:
                    col_values[i] = _convert_numpy_minimal(col_values[i])
            
            return col, col_values
        
        # Process columns in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, len(cols))) as executor:
            column_results = list(executor.map(process_column, cols))
        
        # Build result dictionary
        return {col_name: col_values for col_name, col_values in column_results}
    
    @staticmethod
    def _columnar_single_threaded(df, cols, skip_numpy):
        """Single-threaded columnar fallback."""
        column_data = {}
        for col in cols:
            col_series = df[col]
            values = col_series.values
            nan_mask = pd.isna(values)
            col_values = values.tolist()
            
            for i in range(len(col_values)):
                if nan_mask[i]:
                    col_values[i] = None
                elif not skip_numpy:
                    col_values[i] = _convert_numpy_minimal(col_values[i])
            
            column_data[col] = col_values
        
        return column_data

# Convenience function
def df_to_jt_hp(df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """High-performance DataFrame to JSON-Tables conversion."""
    return HighPerformanceJSONTablesEncoder.from_dataframe(df, **kwargs)

def benchmark_high_performance():
    """Benchmark the high-performance implementation."""
    print("🚀 HIGH-PERFORMANCE JSON-TABLES BENCHMARK")
    print("=" * 60)
    
    # Create test data
    np.random.seed(42)
    df = pd.DataFrame({
        'id': range(10000),
        'name': [f'User_{i}' for i in range(10000)],
        'score1': np.random.rand(10000) * 100,
        'score2': np.random.rand(10000) * 100,
        'score3': np.random.rand(10000) * 100,
        'active': np.random.choice([True, False], 10000),
        'category': np.random.choice(['A', 'B', 'C', 'D'], 10000)
    })
    
    print(f"📊 Test dataset: {df.shape} = {df.shape[0] * df.shape[1]:,} cells")
    
    # Import standard implementation for comparison
    try:
        from .core import JSONTablesEncoder
        standard_available = True
    except ImportError:
        standard_available = False
    
    # Test configurations
    configs = [
        ("High-Perf (skip numpy)", {"skip_numpy_conversion": True}),
        ("High-Perf (with numpy)", {"skip_numpy_conversion": False}),
        ("High-Perf (chunked)", {"skip_numpy_conversion": False, "max_workers": 8, "chunk_size": 500}),
    ]
    
    if standard_available:
        # Standard baseline
        start = time.perf_counter()
        result_std = JSONTablesEncoder.from_dataframe(df)
        std_time = (time.perf_counter() - start) * 1000
        configs.insert(0, ("Standard", None))
        print(f"📏 Standard implementation: {std_time:.1f}ms")
    
    print(f"\n⚡ HIGH-PERFORMANCE RESULTS:")
    
    for name, kwargs in configs:
        if name == "Standard":
            continue
            
        start = time.perf_counter()
        result_hp = df_to_jt_hp(df, **kwargs)
        hp_time = (time.perf_counter() - start) * 1000
        
        if standard_available:
            speedup = std_time / hp_time
            print(f"   {name:<25}: {hp_time:6.1f}ms ({speedup:.2f}x speedup)")
        else:
            print(f"   {name:<25}: {hp_time:6.1f}ms")

if __name__ == "__main__":
    benchmark_high_performance() 