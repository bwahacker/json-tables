#!/usr/bin/env python3
"""
Profiling and timing utilities for JSON-Tables operations.

Provides detailed timing breakdown of where time is spent during
encoding, decoding, and append operations.
"""

import time
import functools
import json
from contextlib import contextmanager
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import threading

@dataclass
class TimingStats:
    """Statistics for a timed operation."""
    name: str
    count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    
    @property
    def avg_time(self) -> float:
        return self.total_time / self.count if self.count > 0 else 0.0
    
    def add_measurement(self, duration: float):
        """Add a timing measurement."""
        self.count += 1
        self.total_time += duration
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'count': self.count,
            'total_time_ms': self.total_time * 1000,
            'avg_time_ms': self.avg_time * 1000,
            'min_time_ms': self.min_time * 1000 if self.min_time != float('inf') else 0,
            'max_time_ms': self.max_time * 1000
        }

class JSONTablesProfiler:
    """Comprehensive profiler for JSON-Tables operations."""
    
    def __init__(self):
        self.stats: Dict[str, TimingStats] = {}
        self.call_stack: List[str] = []
        self.nested_timers: Dict[str, float] = {}
        self._lock = threading.Lock()
        self.enabled = True
    
    def reset(self):
        """Reset all timing statistics."""
        with self._lock:
            self.stats.clear()
            self.call_stack.clear()
            self.nested_timers.clear()
    
    def enable(self):
        """Enable profiling."""
        self.enabled = True
    
    def disable(self):
        """Disable profiling."""
        self.enabled = False
    
    @contextmanager
    def time_operation(self, operation_name: str):
        """Context manager to time an operation."""
        if not self.enabled:
            yield
            return
        
        start_time = time.perf_counter()
        
        with self._lock:
            self.call_stack.append(operation_name)
            full_name = " -> ".join(self.call_stack)
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            with self._lock:
                # Record timing for both the specific operation and full call path
                for name in [operation_name, full_name]:
                    if name not in self.stats:
                        self.stats[name] = TimingStats(name)
                    self.stats[name].add_measurement(duration)
                
                self.call_stack.pop()
    
    def time_function(self, func_name: Optional[str] = None):
        """Decorator to time function calls."""
        def decorator(func: Callable) -> Callable:
            name = func_name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.time_operation(name):
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def get_summary(self, sort_by: str = 'total_time') -> Dict[str, Any]:
        """Get a summary of timing statistics."""
        with self._lock:
            stats_list = [stat.to_dict() for stat in self.stats.values()]
        
        # Sort by specified metric
        if sort_by in ['total_time', 'total_time_ms']:
            stats_list.sort(key=lambda x: x['total_time_ms'], reverse=True)
        elif sort_by in ['avg_time', 'avg_time_ms']:
            stats_list.sort(key=lambda x: x['avg_time_ms'], reverse=True)
        elif sort_by == 'count':
            stats_list.sort(key=lambda x: x['count'], reverse=True)
        
        # Calculate totals
        total_time = sum(stat['total_time_ms'] for stat in stats_list if ' -> ' not in stat['name'])
        total_calls = sum(stat['count'] for stat in stats_list if ' -> ' not in stat['name'])
        
        return {
            'summary': {
                'total_time_ms': total_time,
                'total_calls': total_calls,
                'avg_time_per_call_ms': total_time / total_calls if total_calls > 0 else 0
            },
            'operations': stats_list,
            'call_paths': [stat for stat in stats_list if ' -> ' in stat['name']]
        }
    
    def print_summary(self, sort_by: str = 'total_time', show_call_paths: bool = False):
        """Print a formatted summary of timing statistics."""
        summary = self.get_summary(sort_by)
        
        print("\n🕐 JSON-Tables Performance Profile")
        print("=" * 50)
        
        print(f"📊 Overall Summary:")
        print(f"  Total Time: {summary['summary']['total_time_ms']:.2f}ms")
        print(f"  Total Calls: {summary['summary']['total_calls']}")
        print(f"  Avg per Call: {summary['summary']['avg_time_per_call_ms']:.2f}ms")
        
        print(f"\n⚡ Top Operations (sorted by {sort_by}):")
        print("┌" + "─" * 40 + "┬" + "─" * 12 + "┬" + "─" * 10 + "┬" + "─" * 10 + "┐")
        print("│ Operation" + " " * 31 + "│ Total (ms) │ Avg (ms) │ Count    │")
        print("├" + "─" * 40 + "┼" + "─" * 12 + "┼" + "─" * 10 + "┼" + "─" * 10 + "┤")
        
        for stat in summary['operations'][:10]:  # Top 10
            if ' -> ' in stat['name'] and not show_call_paths:
                continue
            
            name = stat['name'][:38] + ".." if len(stat['name']) > 40 else stat['name']
            print(f"│ {name:<40} │ {stat['total_time_ms']:>10.2f} │ {stat['avg_time_ms']:>8.2f} │ {stat['count']:>8} │")
        
        print("└" + "─" * 40 + "┴" + "─" * 12 + "┴" + "─" * 10 + "┴" + "─" * 10 + "┘")
        
        if show_call_paths and summary['call_paths']:
            print(f"\n🔍 Call Path Analysis:")
            for stat in summary['call_paths'][:5]:  # Top 5 call paths
                print(f"  {stat['name']}: {stat['total_time_ms']:.2f}ms ({stat['count']} calls)")
    
    def save_results(self, file_path: str):
        """Save timing results to a JSON file."""
        summary = self.get_summary()
        with open(file_path, 'w') as f:
            json.dump(summary, f, indent=2)


# Global profiler instance
_profiler = JSONTablesProfiler()

def get_profiler() -> JSONTablesProfiler:
    """Get the global profiler instance."""
    return _profiler

def profile_operation(operation_name: str):
    """Decorator/context manager for profiling operations."""
    return _profiler.time_operation(operation_name)

def profile_function(func_name: Optional[str] = None):
    """Decorator for profiling functions."""
    return _profiler.time_function(func_name)

def reset_profiling():
    """Reset all profiling data."""
    _profiler.reset()

def enable_profiling():
    """Enable profiling."""
    _profiler.enable()

def disable_profiling():
    """Disable profiling."""
    _profiler.disable()

def print_profile_summary(sort_by: str = 'total_time', show_call_paths: bool = False):
    """Print profiling summary."""
    _profiler.print_summary(sort_by, show_call_paths)

def save_profile_results(file_path: str):
    """Save profiling results to file."""
    _profiler.save_results(file_path)

# Monkey-patch critical library functions for detailed timing
def instrument_library_calls():
    """Instrument common library calls for timing."""
    
    # Patch pandas operations
    try:
        import pandas as pd
        
        original_dataframe_init = pd.DataFrame.__init__
        original_to_dict = pd.DataFrame.to_dict
        original_iterrows = pd.DataFrame.iterrows
        
        @profile_function("pandas.DataFrame.__init__")
        def timed_dataframe_init(self, *args, **kwargs):
            return original_dataframe_init(self, *args, **kwargs)
        
        @profile_function("pandas.DataFrame.to_dict")
        def timed_to_dict(self, *args, **kwargs):
            return original_to_dict(self, *args, **kwargs)
        
        @profile_function("pandas.DataFrame.iterrows")
        def timed_iterrows(self, *args, **kwargs):
            return original_iterrows(self, *args, **kwargs)
        
        pd.DataFrame.__init__ = timed_dataframe_init
        pd.DataFrame.to_dict = timed_to_dict
        pd.DataFrame.iterrows = timed_iterrows
        
    except ImportError:
        pass
    
    # Patch JSON operations
    original_json_dumps = json.dumps
    original_json_loads = json.loads
    original_json_load = json.load
    original_json_dump = json.dump
    
    @profile_function("json.dumps")
    def timed_json_dumps(*args, **kwargs):
        return original_json_dumps(*args, **kwargs)
    
    @profile_function("json.loads")
    def timed_json_loads(*args, **kwargs):
        return original_json_loads(*args, **kwargs)
    
    @profile_function("json.load")
    def timed_json_load(*args, **kwargs):
        return original_json_load(*args, **kwargs)
    
    @profile_function("json.dump")
    def timed_json_dump(*args, **kwargs):
        return original_json_dump(*args, **kwargs)
    
    json.dumps = timed_json_dumps
    json.loads = timed_json_loads
    json.load = timed_json_load
    json.dump = timed_json_dump


# Context manager for comprehensive profiling sessions
@contextmanager
def profiling_session(name: str = "session", enable_library_instrumentation: bool = True):
    """Context manager for a complete profiling session."""
    print(f"🔍 Starting profiling session: {name}")
    
    reset_profiling()
    enable_profiling()
    
    if enable_library_instrumentation:
        instrument_library_calls()
    
    start_time = time.perf_counter()
    
    try:
        yield _profiler
    finally:
        end_time = time.perf_counter()
        total_time = (end_time - start_time) * 1000
        
        print(f"✅ Profiling session '{name}' completed in {total_time:.2f}ms")
        print_profile_summary() 