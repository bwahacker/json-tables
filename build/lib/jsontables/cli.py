#!/usr/bin/env python3
"""
Command-line interface for JSON-Tables.
"""

import argparse
import json
import sys
from typing import Any, Dict

from .core import (
    detect_table_in_json,
    to_json_table, 
    render_json_table,
    is_json_table,
    JSONTablesRenderer
)


def render_json_data(data: Any, max_width: int = 300, force_table: bool = False) -> str:
    """
    Render JSON data, using table format if appropriate.
    
    Args:
        data: JSON data to render
        max_width: Maximum width for aligned rendering
        force_table: Force conversion to table format if possible
        
    Returns:
        Rendered string
    """
    # If already a JSON table, render it
    if is_json_table(data):
        return render_json_table(data, max_width=max_width)
    
    # If it's a list of dicts that could be a table
    if detect_table_in_json(data):
        if force_table:
            # Convert to JSON table format first
            json_table = to_json_table(data)
            return render_json_table(json_table, max_width=max_width)
        else:
            # Try to render aligned directly
            return JSONTablesRenderer._render_aligned_records(data)
    
    # Fall back to regular JSON rendering
    return json.dumps(data, indent=2)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="JSON-Tables CLI: Pretty-print JSON tables in aligned format"
    )
    
    parser.add_argument(
        "file", 
        nargs="?", 
        help="JSON file to process (default: stdin)"
    )
    
    parser.add_argument(
        "--max-width", 
        type=int, 
        default=300,
        help="Maximum width for aligned rendering (default: 300)"
    )
    
    parser.add_argument(
        "--force-table",
        action="store_true",
        help="Force conversion to JSON-Tables format"
    )
    
    parser.add_argument(
        "--to-json-table",
        action="store_true", 
        help="Convert to JSON-Tables format and output as JSON"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0"
    )
    
    args = parser.parse_args()
    
    try:
        # Read input
        if args.file:
            with open(args.file, 'r') as f:
                data = json.load(f)
        else:
            data = json.load(sys.stdin)
        
        # Process and output
        if args.to_json_table:
            # Convert to JSON-Tables format if possible
            if detect_table_in_json(data):
                json_table = to_json_table(data)
                print(json.dumps(json_table, indent=2))
            else:
                print("Error: Input data cannot be converted to table format", file=sys.stderr)
                sys.exit(1)
        else:
            # Render for display
            output = render_json_data(
                data, 
                max_width=args.max_width,
                force_table=args.force_table
            )
            print(output)
            
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON - {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: File '{args.file}' not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main() 