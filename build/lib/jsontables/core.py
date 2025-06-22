#!/usr/bin/env python3
"""
Core functionality for JSON-Tables (JSON-T).

A minimal, readable, and backward-compatible format for representing
structured tabular data in JSON.
"""

import json
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from io import StringIO


class JSONTablesError(Exception):
    """Base exception for JSON Tables operations."""
    pass


class JSONTablesEncoder:
    """Encoder for converting data to JSON Tables format."""
    
    @staticmethod
    def from_dataframe(
        df: pd.DataFrame,
        page_size: Optional[int] = None,
        current_page: int = 0,
        columnar: bool = False
    ) -> Dict[str, Any]:
        """
        Convert a pandas DataFrame to JSON Tables format.
        
        Args:
            df: Input DataFrame
            page_size: Number of rows per page (None for no pagination)
            current_page: Current page number (0-based)
            columnar: Use columnar format instead of row-oriented
            
        Returns:
            Dictionary in JSON Tables format
        """
        if df.empty:
            return {
                "__dict_type": "table",
                "cols": list(df.columns),
                "row_data": [],
                "current_page": 0,
                "total_pages": 1,
                "page_rows": 0
            }
        
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
        
        # Convert to JSON Tables format
        result = {
            "__dict_type": "table",
            "cols": cols,
            "current_page": current_page,
            "total_pages": total_pages,
            "page_rows": page_rows
        }
        
        if columnar:
            # Columnar format
            column_data = {}
            for col in cols:
                # Convert to native Python types, handle NaN/None
                values = page_df[col].tolist()
                # Replace NaN with None for JSON serialization
                values = [None if pd.isna(v) else v for v in values]
                column_data[col] = values
            
            result["column_data"] = column_data
            result["row_data"] = None
        else:
            # Row-oriented format
            row_data = []
            for _, row in page_df.iterrows():
                # Convert to native Python types, handle NaN/None
                row_values = [None if pd.isna(v) else v for v in row.tolist()]
                row_data.append(row_values)
            
            result["row_data"] = row_data
        
        return result
    
    @staticmethod
    def from_records(
        records: List[Dict[str, Any]],
        page_size: Optional[int] = None,
        current_page: int = 0,
        columnar: bool = False
    ) -> Dict[str, Any]:
        """
        Convert a list of dictionaries to JSON Tables format.
        
        Args:
            records: List of record dictionaries
            page_size: Number of rows per page (None for no pagination)
            current_page: Current page number (0-based)
            columnar: Use columnar format instead of row-oriented
            
        Returns:
            Dictionary in JSON Tables format
        """
        if not records:
            return {
                "__dict_type": "table",
                "cols": [],
                "row_data": [],
                "current_page": 0,
                "total_pages": 1,
                "page_rows": 0
            }
        
        # Extract column names from first record
        cols = list(records[0].keys())
        
        # Handle pagination
        if page_size is not None:
            total_pages = (len(records) + page_size - 1) // page_size
            start_idx = current_page * page_size
            end_idx = start_idx + page_size
            page_records = records[start_idx:end_idx]
            page_rows = len(page_records)
        else:
            page_records = records
            total_pages = 1
            page_rows = len(records)
        
        result = {
            "__dict_type": "table",
            "cols": cols,
            "current_page": current_page,
            "total_pages": total_pages,
            "page_rows": page_rows
        }
        
        if columnar:
            # Columnar format
            column_data = {col: [] for col in cols}
            for record in page_records:
                for col in cols:
                    column_data[col].append(record.get(col))
            
            result["column_data"] = column_data
            result["row_data"] = None
        else:
            # Row-oriented format
            row_data = []
            for record in page_records:
                row_values = [record.get(col) for col in cols]
                row_data.append(row_values)
            
            result["row_data"] = row_data
        
        return result


class JSONTablesDecoder:
    """Decoder for converting JSON Tables format to standard data structures."""
    
    @staticmethod
    def to_dataframe(json_table: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert JSON Tables format to pandas DataFrame.
        
        Args:
            json_table: Dictionary in JSON Tables format
            
        Returns:
            pandas DataFrame
            
        Raises:
            JSONTablesError: If the input is not valid JSON Tables format
        """
        if not isinstance(json_table, dict):
            raise JSONTablesError("Input must be a dictionary")
        
        if json_table.get("__dict_type") != "table":
            raise JSONTablesError("Missing or invalid __dict_type field")
        
        cols = json_table.get("cols")
        if not isinstance(cols, list):
            raise JSONTablesError("cols field must be a list")
        
        # Handle columnar format
        if "column_data" in json_table and json_table["column_data"] is not None:
            column_data = json_table["column_data"]
            if not isinstance(column_data, dict):
                raise JSONTablesError("column_data must be a dictionary")
            
            # Validate all columns are present
            for col in cols:
                if col not in column_data:
                    raise JSONTablesError(f"Missing column data for: {col}")
            
            # Create DataFrame from columnar data
            df_data = {col: column_data[col] for col in cols}
            return pd.DataFrame(df_data)
        
        # Handle row-oriented format
        row_data = json_table.get("row_data")
        if not isinstance(row_data, list):
            raise JSONTablesError("row_data field must be a list")
        
        if not row_data:
            # Empty table
            return pd.DataFrame(columns=cols)
        
        # Validate row data structure
        for i, row in enumerate(row_data):
            if not isinstance(row, list):
                raise JSONTablesError(f"Row {i} must be a list")
            if len(row) != len(cols):
                raise JSONTablesError(f"Row {i} has {len(row)} values but expected {len(cols)}")
        
        # Create DataFrame
        return pd.DataFrame(row_data, columns=cols)
    
    @staticmethod
    def to_records(json_table: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert JSON Tables format to list of dictionaries.
        
        Args:
            json_table: Dictionary in JSON Tables format
            
        Returns:
            List of record dictionaries
        """
        df = JSONTablesDecoder.to_dataframe(json_table)
        return df.to_dict('records')


class JSONTablesRenderer:
    """Renderer for human-friendly display of JSON Tables."""
    
    @staticmethod
    def render_aligned(
        json_table: Dict[str, Any],
        max_width: int = 300,
        indent: int = 0
    ) -> str:
        """
        Render JSON Tables in aligned, human-readable format.
        
        Args:
            json_table: Dictionary in JSON Tables format
            max_width: Maximum width for rendering
            indent: Indentation level
            
        Returns:
            Human-readable string representation
        """
        try:
            df = JSONTablesDecoder.to_dataframe(json_table)
        except JSONTablesError:
            # Fall back to regular JSON rendering
            return json.dumps(json_table, indent=2)
        
        if df.empty:
            return "[]"
        
        # Convert DataFrame to records for alignment
        records = df.to_dict('records')
        
        # Check if suitable for aligned rendering
        if not JSONTablesRenderer._should_align(records, max_width):
            return json.dumps(records, indent=2)
        
        return JSONTablesRenderer._render_aligned_records(records, indent)
    
    @staticmethod
    def _should_align(records: List[Dict[str, Any]], max_width: int) -> bool:
        """Check if records should be rendered in aligned format."""
        if not records:
            return False
        
        # Check if all records have the same keys
        first_keys = set(records[0].keys())
        if not all(set(record.keys()) == first_keys for record in records):
            return False
        
        # Check if all values are primitives
        for record in records:
            for value in record.values():
                if not isinstance(value, (str, int, float, bool, type(None))):
                    return False
        
        # Estimate rendered width
        estimated_width = JSONTablesRenderer._estimate_width(records)
        return estimated_width <= max_width
    
    @staticmethod
    def _estimate_width(records: List[Dict[str, Any]]) -> int:
        """Estimate the rendered width of aligned records."""
        if not records:
            return 0
        
        # Calculate maximum width for each column
        cols = list(records[0].keys())
        col_widths = {}
        
        for col in cols:
            max_width = len(str(col))
            for record in records:
                value_str = json.dumps(record[col]) if isinstance(record[col], str) else str(record[col])
                max_width = max(max_width, len(value_str))
            col_widths[col] = max_width
        
        # Estimate total width: sum of column widths + separators + brackets
        total_width = sum(col_widths.values()) + len(cols) * 4 + 10
        return total_width
    
    @staticmethod
    def _render_aligned_records(records: List[Dict[str, Any]], indent: int = 0) -> str:
        """Render records in aligned format."""
        if not records:
            return "[]"
        
        cols = list(records[0].keys())
        
        # Calculate column widths
        col_widths = {}
        for col in cols:
            max_width = len(col)
            for record in records:
                value_str = json.dumps(record[col]) if isinstance(record[col], str) else str(record[col])
                max_width = max(max_width, len(value_str))
            col_widths[col] = max_width
        
        # Build output
        lines = ["["]
        
        for i, record in enumerate(records):
            line_parts = ["  { "]
            
            for j, col in enumerate(cols):
                value = record[col]
                if isinstance(value, str):
                    value_str = json.dumps(value)
                else:
                    value_str = str(value)
                
                # Add column with proper alignment
                if j == 0:
                    line_parts.append(f"{col}: {value_str:<{col_widths[col] - len(col) + len(value_str)}}")
                else:
                    line_parts.append(f" , {col}: {value_str:<{col_widths[col] - len(col) + len(value_str)}}")
            
            line_parts.append(" }")
            if i < len(records) - 1:
                line_parts.append(",")
            
            lines.append("".join(line_parts))
        
        lines.append("]")
        
        # Apply indentation
        if indent > 0:
            indent_str = " " * indent
            lines = [indent_str + line for line in lines]
        
        return "\n".join(lines)


def is_json_table(data: Any) -> bool:
    """Check if data is in JSON Tables format."""
    return (
        isinstance(data, dict) and
        data.get("__dict_type") == "table" and
        "cols" in data and
        isinstance(data["cols"], list)
    )


def detect_table_in_json(data: Any) -> bool:
    """
    Detect if JSON data could be represented as a table.
    
    Returns True if data is a list of objects with identical keys
    and primitive values.
    """
    if not isinstance(data, list) or not data:
        return False
    
    if not all(isinstance(item, dict) for item in data):
        return False
    
    # Check if all objects have the same keys
    first_keys = set(data[0].keys())
    if not all(set(item.keys()) == first_keys for item in data):
        return False
    
    # Check if all values are primitives
    for item in data:
        for value in item.values():
            if not isinstance(value, (str, int, float, bool, type(None))):
                return False
    
    return True


# CLI-style functions for easy usage
def to_json_table(data: Union[pd.DataFrame, List[Dict[str, Any]]], **kwargs) -> Dict[str, Any]:
    """Convert data to JSON Tables format."""
    if isinstance(data, pd.DataFrame):
        return JSONTablesEncoder.from_dataframe(data, **kwargs)
    elif isinstance(data, list):
        return JSONTablesEncoder.from_records(data, **kwargs)
    else:
        raise JSONTablesError(f"Unsupported data type: {type(data)}")


def from_json_table(json_table: Dict[str, Any]) -> pd.DataFrame:
    """Convert JSON Tables format to DataFrame."""
    return JSONTablesDecoder.to_dataframe(json_table)


def render_json_table(json_table: Dict[str, Any], **kwargs) -> str:
    """Render JSON Tables in human-readable format."""
    return JSONTablesRenderer.render_aligned(json_table, **kwargs) 