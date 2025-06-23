"""
JSON-Tables: A minimal, readable format for tabular data in JSON.

Provides human-readable table rendering, clear semantics for tooling,
and seamless loading into analytics libraries.
"""

from .core import (
    # Core classes
    JSONTablesEncoder,
    JSONTablesDecoder, 
    JSONTablesRenderer,
    JSONTablesAppender,
    JSONTablesV2Encoder,
    JSONTablesV2Decoder,
    
    # Utility functions
    to_json_table,
    from_json_table,
    render_json_table,
    append_to_json_table_file,
    is_json_table,
    detect_table_in_json,
    
    # DataFrame convenience functions
    df_to_jt,
    df_from_jt,
    
    # Exceptions
    JSONTablesError
)

# Profiling functionality (optional import)
try:
    from .profiling import (
        profiling_session,
        profile_operation,
        profile_function,
        print_profile_summary,
        save_profile_results,
        enable_profiling,
        disable_profiling,
        reset_profiling,
        get_profiler
    )
    _PROFILING_AVAILABLE = True
except ImportError:
    _PROFILING_AVAILABLE = False

# Import multithreaded operations
try:
    from .multithreaded_core import (
        MultithreadedJSONTablesEncoder,
        MultithreadedJSONTablesDecoder,
        df_to_jt_mt,
        df_from_jt_mt
    )
    MULTITHREADED_AVAILABLE = True
except ImportError:
    MULTITHREADED_AVAILABLE = False

# Import smart parallel operations
try:
    from .smart_parallel import (
        SmartParallelJSONTablesEncoder,
        df_to_jt_smart
    )
    SMART_PARALLEL_AVAILABLE = True
except ImportError:
    SMART_PARALLEL_AVAILABLE = False

# Import high-performance operations
try:
    from .high_performance_core import (
        HighPerformanceJSONTablesEncoder,
        df_to_jt_hp
    )
    HIGH_PERFORMANCE_AVAILABLE = True
except ImportError:
    HIGH_PERFORMANCE_AVAILABLE = False

__version__ = "1.0.0"
__author__ = "Mitch Haile"
__email__ = "mitch.haile@gmail.com"
__license__ = "MIT"

# Public API
__all__ = [
    # Core functionality
    'JSONTablesEncoder',
    'JSONTablesDecoder',
    'JSONTablesRenderer', 
    'JSONTablesAppender',
    'JSONTablesV2Encoder',
    'JSONTablesV2Decoder',
    
    # Utility functions
    'to_json_table',
    'from_json_table',
    'render_json_table',
    'append_to_json_table_file',
    'is_json_table',
    'detect_table_in_json',
    
    # DataFrame convenience functions
    'df_to_jt',
    'df_from_jt',
    
    # Exceptions
    'JSONTablesError',
]

# Add profiling to public API if available
if _PROFILING_AVAILABLE:
    __all__.extend([
        'profiling_session',
        'profile_operation', 
        'profile_function',
        'print_profile_summary',
        'save_profile_results',
        'enable_profiling',
        'disable_profiling',
        'reset_profiling',
        'get_profiler'
    ])

# Add multithreaded functions to public API if available
if MULTITHREADED_AVAILABLE:
    __all__.extend([
        'MultithreadedJSONTablesEncoder',
        'MultithreadedJSONTablesDecoder',
        'df_to_jt_mt',
        'df_from_jt_mt'
    ])

# Add smart parallel functions to public API if available
if SMART_PARALLEL_AVAILABLE:
    __all__.extend([
        'SmartParallelJSONTablesEncoder',
        'df_to_jt_smart'
    ])

# Add high-performance functions to public API if available
if HIGH_PERFORMANCE_AVAILABLE:
    __all__.extend([
        'HighPerformanceJSONTablesEncoder',
        'df_to_jt_hp'
    ])

def get_version():
    """Get the current version of JSON-Tables."""
    return __version__

def is_profiling_available():
    """Check if profiling functionality is available."""
    return _PROFILING_AVAILABLE 