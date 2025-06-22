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

# Import profiling utilities
try:
    from .profiling import profile_operation, profile_function
except ImportError:
    # Fallback if profiling module is not available
    from contextlib import nullcontext
    def profile_operation(name): return nullcontext()
    def profile_function(name=None): 
        def decorator(func): return func
        return decorator


class JSONTablesError(Exception):
    """Base exception for JSON Tables operations."""
    pass


class JSONTablesEncoder:
    """Encoder for converting data to JSON Tables format."""
    
    @staticmethod
    @profile_function("JSONTablesEncoder.from_dataframe")
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
        with profile_operation("dataframe_validation"):
            if df.empty:
                return {
                    "__dict_type": "table",
                    "cols": list(df.columns),
                    "row_data": [],
                    "current_page": 0,
                    "total_pages": 1,
                    "page_rows": 0
                }
        
        with profile_operation("extract_columns"):
            cols = list(df.columns)
        
        # Handle pagination
        with profile_operation("pagination_logic"):
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
        with profile_operation("build_result_structure"):
            result = {
                "__dict_type": "table",
                "cols": cols,
                "current_page": current_page,
                "total_pages": total_pages,
                "page_rows": page_rows
            }
        
        if columnar:
            # Columnar format
            with profile_operation("columnar_conversion"):
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
            with profile_operation("row_oriented_conversion"):
                row_data = []
                for _, row in page_df.iterrows():
                    # Convert to native Python types, handle NaN/None
                    row_values = [None if pd.isna(v) else v for v in row.tolist()]
                    row_data.append(row_values)
                
                result["row_data"] = row_data
        
        return result
    
    @staticmethod
    @profile_function("JSONTablesEncoder.from_records")
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
        with profile_operation("records_validation"):
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
        with profile_operation("extract_column_names"):
            cols = list(records[0].keys())
        
        # Handle pagination
        with profile_operation("records_pagination"):
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
        
        with profile_operation("records_result_structure"):
            result = {
                "__dict_type": "table",
                "cols": cols,
                "current_page": current_page,
                "total_pages": total_pages,
                "page_rows": page_rows
            }
        
        if columnar:
            # Columnar format
            with profile_operation("records_columnar_conversion"):
                column_data = {col: [] for col in cols}
                for record in page_records:
                    for col in cols:
                        column_data[col].append(record.get(col))
                
                result["column_data"] = column_data
                result["row_data"] = None
        else:
            # Row-oriented format
            with profile_operation("records_row_conversion"):
                row_data = []
                for record in page_records:
                    row_values = [record.get(col) for col in cols]
                    row_data.append(row_values)
                
                result["row_data"] = row_data
        
        return result


class JSONTablesDecoder:
    """Decoder for converting JSON Tables format to standard data structures."""
    
    @staticmethod
    @profile_function("JSONTablesDecoder.to_dataframe")
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
        with profile_operation("decode_validation"):
            if not isinstance(json_table, dict):
                raise JSONTablesError("Input must be a dictionary")
            
            if json_table.get("__dict_type") != "table":
                raise JSONTablesError("Missing or invalid __dict_type field")
            
            cols = json_table.get("cols")
            if not isinstance(cols, list):
                raise JSONTablesError("cols field must be a list")
        
        # Handle columnar format
        if "column_data" in json_table and json_table["column_data"] is not None:
            with profile_operation("decode_columnar_format"):
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
        with profile_operation("decode_row_oriented"):
            row_data = json_table.get("row_data")
            if not isinstance(row_data, list):
                raise JSONTablesError("row_data field must be a list")
            
            if not row_data:
                # Empty table
                return pd.DataFrame(columns=cols)
            
            # Validate row data structure
            with profile_operation("validate_row_structure"):
                for i, row in enumerate(row_data):
                    if not isinstance(row, list):
                        raise JSONTablesError(f"Row {i} must be a list")
                    if len(row) != len(cols):
                        raise JSONTablesError(f"Row {i} has {len(row)} values but expected {len(cols)}")
            
            # Create DataFrame
            with profile_operation("create_dataframe"):
                return pd.DataFrame(row_data, columns=cols)
    
    @staticmethod
    @profile_function("JSONTablesDecoder.to_records")
    def to_records(json_table: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert JSON Tables format to list of dictionaries.
        
        Args:
            json_table: Dictionary in JSON Tables format
            
        Returns:
            List of record dictionaries
        """
        with profile_operation("decode_to_records"):
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


class JSONTablesV2Encoder:
    """Advanced encoder for JSON Tables v2 with optimizations."""
    
    @staticmethod
    def analyze_data_patterns(records: List[Dict[str, Any]], max_schemas: int = 8, sample_size: int = 1000) -> Dict:
        """Analyze data to determine if advanced optimizations are worthwhile."""
        if not records:
            return {'use_schemas': False, 'use_categoricals': False}
        
        # Sample data for analysis
        sample = records[:min(sample_size, len(records))]
        
        # Analyze null patterns
        null_patterns = {}
        categorical_candidates = {}
        
        # Get all possible fields
        all_fields = set()
        for record in sample:
            all_fields.update(record.keys())
        all_fields = sorted(all_fields)
        
        # Analyze each record's null pattern and categorical values
        for record in sample:
            null_pattern = tuple(field for field in all_fields if record.get(field) is None or record.get(field) == '')
            
            if null_pattern not in null_patterns:
                null_patterns[null_pattern] = []
            null_patterns[null_pattern].append(record)
            
            # Track potential categorical fields
            for field, value in record.items():
                if isinstance(value, str) and len(value) < 50:  # Potential categorical
                    if field not in categorical_candidates:
                        categorical_candidates[field] = set()
                    categorical_candidates[field].add(value)
        
        # Determine if schemas are worth it
        schema_savings = 0
        schema_overhead = 50  # Estimated JSON overhead for schema definitions
        
        if len(null_patterns) <= max_schemas and len(null_patterns) > 1:
            # Calculate potential savings from schema variations
            for pattern, pattern_records in null_patterns.items():
                # Savings = (num_nulls * avg_field_name_length * num_records_with_pattern)
                avg_field_length = sum(len(f) for f in pattern) / max(len(pattern), 1)
                savings = len(pattern) * avg_field_length * len(pattern_records)
                schema_savings += savings
        
        # Determine categorical fields
        useful_categoricals = {}
        for field, values in categorical_candidates.items():
            if 2 <= len(values) <= 50 and len(values) < len(sample) * 0.8:  # Good categorical candidate
                # Estimate savings: (avg_string_length - 1) * frequency
                avg_length = sum(len(str(v)) for v in values) / len(values)
                if avg_length > 2:  # Worth encoding
                    useful_categoricals[field] = sorted(values)
        
        return {
            'use_schemas': schema_savings > schema_overhead,
            'schemas': null_patterns if schema_savings > schema_overhead else {},
            'use_categoricals': len(useful_categoricals) > 0,
            'categoricals': useful_categoricals,
            'analysis': {
                'num_null_patterns': len(null_patterns),
                'estimated_schema_savings': schema_savings,
                'schema_overhead': schema_overhead,
                'categorical_fields': len(useful_categoricals)
            }
        }
    
    @staticmethod
    def from_records_v2(
        records: List[Dict[str, Any]], 
        auto_optimize: bool = True,
        max_schemas: int = 8
    ) -> Dict[str, Any]:
        """Convert records to JSON Tables v2 format with advanced optimizations."""
        if not records:
            return {
                "__dict_type": "table",
                "__version": "2.0",
                "cols": [],
                "row_data": []
            }
        
        # Analyze data patterns if auto-optimization is enabled
        if auto_optimize:
            analysis = JSONTablesV2Encoder.analyze_data_patterns(records, max_schemas)
        else:
            analysis = {'use_schemas': False, 'use_categoricals': False}
        
        cols = list(records[0].keys())
        
        result = {
            "__dict_type": "table", 
            "__version": "2.0",
            "cols": cols
        }
        
        # Apply categorical encoding if beneficial
        categorical_mappings = {}
        if analysis.get('use_categoricals', False):
            categorical_mappings = analysis['categoricals']
            result["categoricals"] = categorical_mappings
        
        # Apply schema variations if beneficial
        if analysis.get('use_schemas', False):
            return JSONTablesV2Encoder._encode_with_schemas(records, cols, categorical_mappings, analysis['schemas'])
        else:
            return JSONTablesV2Encoder._encode_simple_v2(records, cols, categorical_mappings, result)
    
    @staticmethod
    def _encode_simple_v2(records: List[Dict[str, Any]], cols: List[str], categoricals: Dict, result: Dict) -> Dict:
        """Encode with categorical optimization but no schema variations."""
        row_data = []
        
        for record in records:
            row = []
            for col in cols:
                value = record.get(col)
                
                # Apply categorical encoding
                if col in categoricals and value in categoricals[col]:
                    value = categoricals[col].index(value)
                
                row.append(value)
            row_data.append(row)
        
        result["row_data"] = row_data
        return result
    
    @staticmethod 
    def _encode_with_schemas(records: List[Dict[str, Any]], cols: List[str], categoricals: Dict, null_patterns: Dict) -> Dict:
        """Encode with both schema variations and categorical optimizations."""
        # Create schema definitions
        schemas = {}
        schema_map = {}  # pattern -> schema_id
        
        for i, (pattern, pattern_records) in enumerate(null_patterns.items()):
            schema_id = str(i)
            schemas[schema_id] = {
                "defaults": [None if col in pattern else "REQUIRED" for col in cols]
            }
            schema_map[pattern] = schema_id
        
        result = {
            "__dict_type": "table",
            "__version": "2.0", 
            "cols": cols,
            "schemas": schemas,
            "row_data": []
        }
        
        if categoricals:
            result["categoricals"] = categoricals
        
        # Encode records with schema references
        for record in records:
            # Determine null pattern
            null_pattern = tuple(col for col in cols if record.get(col) is None or record.get(col) == '')
            schema_id = schema_map.get(null_pattern, "0")
            
            row = ["__jt_sid", schema_id]
            
            # Only include non-default values
            for col in cols:
                value = record.get(col)
                if value is not None and value != '':
                    # Apply categorical encoding
                    if col in categoricals and value in categoricals[col]:
                        value = categoricals[col].index(value)
                    row.extend([cols.index(col), value])
            
            result["row_data"].append(row)
        
        return result


class JSONTablesV2Decoder:
    """Decoder for JSON Tables v2 format."""
    
    @staticmethod
    def to_records(json_table: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert JSON Tables v2 format to list of dictionaries."""
        if json_table.get("__version") != "2.0":
            # Fall back to v1 decoder
            return JSONTablesDecoder.to_records(json_table)
        
        cols = json_table.get("cols", [])
        schemas = json_table.get("schemas", {})
        categoricals = json_table.get("categoricals", {})
        row_data = json_table.get("row_data", [])
        
        if not row_data:
            return []
        
        records = []
        
        for row in row_data:
            if len(row) >= 2 and row[0] == "__jt_sid":
                # Schema variation format
                schema_id = row[1]
                record = JSONTablesV2Decoder._decode_schema_row(row[2:], cols, schemas.get(schema_id, {}), categoricals)
            else:
                # Simple v2 format
                record = {}
                for i, value in enumerate(row):
                    if i < len(cols):
                        col = cols[i]
                        # Apply categorical decoding
                        if col in categoricals and isinstance(value, int) and 0 <= value < len(categoricals[col]):
                            value = categoricals[col][value]
                        record[col] = value
            
            records.append(record)
        
        return records
    
    @staticmethod
    def _decode_schema_row(data: List, cols: List[str], schema: Dict, categoricals: Dict) -> Dict[str, Any]:
        """Decode a row with schema variations."""
        defaults = schema.get("defaults", [])
        
        # Start with default values
        record = {}
        for i, col in enumerate(cols):
            default = defaults[i] if i < len(defaults) else None
            record[col] = None if default == "REQUIRED" else default
        
        # Apply specific values
        i = 0
        while i < len(data) - 1:
            col_idx = data[i]
            value = data[i + 1]
            
            if 0 <= col_idx < len(cols):
                col = cols[col_idx]
                # Apply categorical decoding
                if col in categoricals and isinstance(value, int) and 0 <= value < len(categoricals[col]):
                    value = categoricals[col][value]
                record[col] = value
            
            i += 2
        
        return record 


class JSONTablesAppender:
    """Efficient append operations for JSON-Tables files."""
    
    @staticmethod
    @profile_function("JSONTablesAppender.append_rows")
    def append_rows(file_path: str, new_rows: List[Dict[str, Any]]) -> bool:
        """
        Efficiently append rows to an existing JSON-Tables file.
        
        Args:
            file_path: Path to the JSON-Tables file
            new_rows: List of new row dictionaries to append
            
        Returns:
            True if successful, False otherwise
        """
        if not new_rows:
            return True
            
        try:
            # Read and parse the existing file
            with profile_operation("read_existing_file"):
                with open(file_path, 'r') as f:
                    table_data = json.load(f)
            
            with profile_operation("validate_table_structure"):
                if table_data.get("__dict_type") != "table":
                    return False
                    
                cols = table_data.get("cols", [])
                if not cols:
                    return False
            
            # Convert new rows to the table format
            with profile_operation("convert_new_rows"):
                new_row_data = []
                for row in new_rows:
                    row_values = [row.get(col) for col in cols]
                    new_row_data.append(row_values)
            
            # Try efficient append first, fall back to full rewrite if needed
            with profile_operation("efficient_append_attempt"):
                return JSONTablesAppender._try_efficient_append(file_path, table_data, new_row_data, cols)
            
        except Exception as e:
            print(f"Append failed: {e}")
            return False
    
    @staticmethod
    @profile_function("JSONTablesAppender._try_efficient_append")
    def _try_efficient_append(file_path: str, table_data: Dict, new_row_data: List[List], cols: List[str]) -> bool:
        """Try to efficiently append by modifying file content directly."""
        try:
            # Read the file as text
            with profile_operation("read_file_as_text"):
                with open(file_path, 'r') as f:
                    content = f.read()
            
            # Find the row_data array ending
            with profile_operation("parse_json_structure"):
                import re
                
                # Find the row_data section more carefully
                pattern = r'"row_data"\s*:\s*\['
                match = re.search(pattern, content)
                
                if not match:
                    # Fall back to full rewrite
                    return JSONTablesAppender._full_rewrite_append(file_path, table_data, new_row_data)
                
                # Find the matching closing bracket for the row_data array
                start_pos = match.end() - 1  # Position of the opening [
                bracket_count = 1
                pos = start_pos + 1
                
                while pos < len(content) and bracket_count > 0:
                    if content[pos] == '[':
                        bracket_count += 1
                    elif content[pos] == ']':
                        bracket_count -= 1
                    pos += 1
                
                if bracket_count > 0:
                    # Couldn't find matching bracket, fall back
                    return JSONTablesAppender._full_rewrite_append(file_path, table_data, new_row_data)
                
                # pos is now just after the closing ]
                end_bracket_pos = pos - 1
                
                # Check if array is empty or has content
                array_content = content[start_pos+1:end_bracket_pos].strip()
            
            # Create the JSON for new rows
            with profile_operation("serialize_new_rows"):
                new_rows_json = []
                for row in new_row_data:
                    new_rows_json.append(json.dumps(row))
                
                new_rows_str = ', '.join(new_rows_json)
            
            with profile_operation("modify_file_content"):
                if array_content and new_rows_str:
                    # Array has content, add comma before new rows
                    new_content = content[:end_bracket_pos] + ', ' + new_rows_str + content[end_bracket_pos:]
                elif new_rows_str:
                    # Array is empty, just add the new rows
                    new_content = content[:end_bracket_pos] + new_rows_str + content[end_bracket_pos:]
                else:
                    # Nothing to add
                    return True
            
            # Write the new content
            with profile_operation("write_modified_file"):
                with open(file_path, 'w') as f:
                    f.write(new_content)
            
            return True
            
        except Exception as e:
            # If anything goes wrong, fall back to full rewrite
            print(f"Efficient append failed, falling back to full rewrite: {e}")
            return JSONTablesAppender._full_rewrite_append(file_path, table_data, new_row_data)
    
    @staticmethod
    @profile_function("JSONTablesAppender._full_rewrite_append")
    def _full_rewrite_append(file_path: str, table_data: Dict, new_row_data: List[List]) -> bool:
        """Fallback to full rewrite when efficient append isn't possible."""
        try:
            with profile_operation("full_rewrite_operation"):
                # Add new rows to existing data
                existing_rows = table_data.get("row_data", [])
                existing_rows.extend(new_row_data)
                
                # Write back to file
                with open(file_path, 'w') as f:
                    json.dump(table_data, f)
            
            return True
        except Exception as e:
            print(f"Full rewrite failed: {e}")
            return False

# Add convenience functions
@profile_function("append_to_json_table_file")
def append_to_json_table_file(file_path: str, new_rows: List[Dict[str, Any]]) -> bool:
    """Convenience function to append rows to a JSON-Tables file."""
    return JSONTablesAppender.append_rows(file_path, new_rows) 