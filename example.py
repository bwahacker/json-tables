#!/usr/bin/env python3
"""
Example usage of the jsontables package.
"""

import json
from jsontables import to_json_table, render_json_table, detect_table_in_json

def main():
    # Example data from the README
    test_data = [
        {"name": "Alessandra", "age": 3, "score": 812},
        {"name": "Bo", "age": 14, "score": 5},
        {"name": "Christine", "age": 103, "score": 1000}
    ]
    
    print("🧩 JSON-Tables Example")
    print("=" * 50)
    
    print("\n1. Original JSON data:")
    print(json.dumps(test_data, indent=2))
    
    print("\n2. Detect if data can be a table:")
    is_table = detect_table_in_json(test_data)
    print(f"   Can be table: {is_table}")
    
    print("\n3. Convert to JSON-Tables format:")
    json_table = to_json_table(test_data)
    print(json.dumps(json_table, indent=2))
    
    print("\n4. Render in aligned format:")
    aligned = render_json_table(json_table)
    print(aligned)
    
    print("\n✅ Example completed successfully!")

if __name__ == "__main__":
    main() 