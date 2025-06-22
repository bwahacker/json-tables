"""
JSON-Tables (JSON-T): A minimal format for representing tabular data in JSON.

This package provides tools for converting between pandas DataFrames, lists of dictionaries,
and the JSON-Tables format, with human-readable rendering capabilities.
"""

__version__ = "0.1.0"
__author__ = "Mitch Haile"
__email__ = "mitch.haile@gmail.com"

from .core import (
    JSONTablesError,
    JSONTablesEncoder,
    JSONTablesDecoder, 
    JSONTablesRenderer,
    is_json_table,
    detect_table_in_json,
    to_json_table,
    from_json_table,
    render_json_table
)

__all__ = [
    "JSONTablesError",
    "JSONTablesEncoder", 
    "JSONTablesDecoder",
    "JSONTablesRenderer",
    "is_json_table",
    "detect_table_in_json", 
    "to_json_table",
    "from_json_table",
    "render_json_table"
] 