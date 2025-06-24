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
    JSONTablesError,
    
    # Simple utility functions
    to_json_table,
    from_json_table,
    render_json_table,
    is_json_table,
    detect_table_in_json,
)

# Import the FAST implementations as the default
from .high_performance_core import HighPerformanceJSONTablesEncoder

# Data integrity validation
from .data_integrity import (
    DataIntegrityValidator,
    DataIntegrityError,
    validate_conversion_integrity,
)

def df_to_jt(df, **kwargs):
    """
    Convert pandas DataFrame to JSON-Tables format (high-performance).
    
    Args:
        df: pandas DataFrame
        **kwargs: Additional options (columnar, page_size, etc.)
    
    Returns:
        Dictionary in JSON-Tables format
    """
    return HighPerformanceJSONTablesEncoder.from_dataframe(df, **kwargs)

def df_from_jt(json_table):
    """
    Convert JSON-Tables format to pandas DataFrame.
    
    Args:
        json_table: Dictionary in JSON-Tables format
        
    Returns:
        pandas DataFrame
    """
    return JSONTablesDecoder.to_dataframe(json_table)

def json_to_jt(json_records, **kwargs):
    """
    Convert standard JSON (list of records) to JSON-Tables format.
    
    Args:
        json_records: List of dictionaries (standard JSON records)
        **kwargs: Additional options (columnar, page_size, etc.)
    
    Returns:
        Dictionary in JSON-Tables format
    """
    import pandas as pd
    df = pd.DataFrame(json_records)
    return df_to_jt(df, **kwargs)

def jt_to_json(json_table):
    """
    Convert JSON-Tables format to standard JSON (list of records).
    
    Args:
        json_table: Dictionary in JSON-Tables format
        
    Returns:
        List of dictionaries (standard JSON records)
    """
    df = df_from_jt(json_table)
    return df.to_dict('records')

def csv_to_jt(csv_path, **kwargs):
    """
    Read CSV file and convert to JSON-Tables format.
    
    Args:
        csv_path: Path to CSV file
        **kwargs: Additional options for pandas.read_csv() and df_to_jt()
    
    Returns:
        Dictionary in JSON-Tables format
    """
    import pandas as pd
    
    # Separate pandas read_csv args from df_to_jt args
    jt_kwargs = {}
    csv_kwargs = {}
    
    jt_options = {'columnar', 'page_size'}
    for key, value in kwargs.items():
        if key in jt_options:
            jt_kwargs[key] = value
        else:
            csv_kwargs[key] = value
    
    df = pd.read_csv(csv_path, **csv_kwargs)
    return df_to_jt(df, **jt_kwargs)

def jt_to_csv(json_table, csv_path, **kwargs):
    """
    Write JSON-Tables format to CSV file.
    
    Args:
        json_table: Dictionary in JSON-Tables format
        csv_path: Path for output CSV file
        **kwargs: Additional options for pandas.to_csv()
    
    Returns:
        None (writes file)
    """
    df = df_from_jt(json_table)
    df.to_csv(csv_path, index=False, **kwargs)

def jt_to_sqlite(json_table, db_path, table_name='data', **kwargs):
    """
    Export JSON-Tables format to SQLite database.
    
    Args:
        json_table: Dictionary in JSON-Tables format
        db_path: Path for SQLite database file
        table_name: Name for database table (default: 'data')
        **kwargs: Additional options for pandas.to_sql()
    
    Returns:
        None (writes database)
    """
    import sqlite3
    df = df_from_jt(json_table)
    
    with sqlite3.connect(db_path) as conn:
        df.to_sql(table_name, conn, index=False, if_exists='replace', **kwargs)

__version__ = "0.2.0"
__author__ = "Mitch Haile"
__email__ = "mitch.haile@gmail.com"
__license__ = "MIT"

# Simple, clean public API
__all__ = [
    # Core functionality
    'JSONTablesEncoder',
    'JSONTablesDecoder',
    'JSONTablesRenderer',
    'JSONTablesError',
    
    # Simple utility functions
    'to_json_table',
    'from_json_table', 
    'render_json_table',
    'is_json_table',
    'detect_table_in_json',
    
    # Main DataFrame API - FAST by default
    'df_to_jt',  # High-performance by default
    'df_from_jt',
    
    # Format conversion functions
    'json_to_jt',
    'jt_to_json', 
    'csv_to_jt',
    'jt_to_csv',
    'jt_to_sqlite',
    
    # Data integrity validation
    'DataIntegrityValidator',
    'DataIntegrityError',
    'validate_conversion_integrity',
]

def get_version():
    """Get the current version of JSON-Tables."""
    return __version__ 