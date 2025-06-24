#!/usr/bin/env python3
"""
Demo: JSON-Tables as Universal Format Converter

Shows how JSON-Tables can convert between different tabular data formats:
- Standard JSON ↔ JSON-Tables
- CSV ↔ JSON-Tables  
- SQLite ← JSON-Tables
- Pandas DataFrames ↔ JSON-Tables
"""

import json
import os
import sqlite3
import tempfile
from jsontables import (
    json_to_jt, jt_to_json,
    csv_to_jt, jt_to_csv,
    jt_to_sqlite,
    df_to_jt, df_from_jt
)

def demo_json_conversion():
    """Demo converting between standard JSON and JSON-Tables."""
    print("🔄 JSON ↔ JSON-Tables Conversion")
    print("=" * 50)
    
    # Sample data as standard JSON
    standard_json = [
        {"name": "Alice", "age": 30, "city": "NYC", "salary": 85000},
        {"name": "Bob", "age": 25, "city": "LA", "salary": 72000},
        {"name": "Carol", "age": 35, "city": "Chicago", "salary": 95000},
        {"name": "David", "age": 28, "city": "Austin", "salary": 68000}
    ]
    
    print(f"📊 Original JSON size: {len(json.dumps(standard_json))} bytes")
    
    # Convert to JSON-Tables
    json_table = json_to_jt(standard_json)
    jt_size = len(json.dumps(json_table))
    print(f"🗜️  JSON-Tables size: {jt_size} bytes")
    
    # Calculate savings
    orig_size = len(json.dumps(standard_json))
    savings = (orig_size - jt_size) / orig_size * 100
    print(f"💰 Space savings: {savings:.1f}%")
    
    # Convert back to standard JSON
    restored_json = jt_to_json(json_table)
    print(f"✅ Perfect restoration: {standard_json == restored_json}")
    print()

def demo_csv_conversion():
    """Demo converting between CSV and JSON-Tables."""
    print("📄 CSV ↔ JSON-Tables Conversion")
    print("=" * 50)
    
    # Create sample CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("name,age,city,salary\n")
        f.write("Alice,30,NYC,85000\n")
        f.write("Bob,25,LA,72000\n") 
        f.write("Carol,35,Chicago,95000\n")
        f.write("David,28,Austin,68000\n")
        csv_path = f.name
    
    try:
        # Read CSV and convert to JSON-Tables
        print(f"📖 Reading CSV: {os.path.basename(csv_path)}")
        json_table = csv_to_jt(csv_path)
        print(f"✅ Converted to JSON-Tables format")
        
        # Write back to new CSV
        output_csv = csv_path.replace('.csv', '_output.csv')
        jt_to_csv(json_table, output_csv)
        print(f"💾 Exported to: {os.path.basename(output_csv)}")
        
        # Verify round-trip
        with open(csv_path) as f1, open(output_csv) as f2:
            original = f1.read()
            restored = f2.read()
            print(f"✅ Perfect CSV round-trip: {original == restored}")
            
        os.unlink(output_csv)
        
    finally:
        os.unlink(csv_path)
    
    print()

def demo_sqlite_export():
    """Demo exporting JSON-Tables to SQLite."""
    print("🗄️  JSON-Tables → SQLite Export")
    print("=" * 50)
    
    # Create JSON-Tables data
    json_table = {
        "__dict_type": "table",
        "cols": ["product", "price", "category", "in_stock"],
        "row_data": [
            ["Laptop", 999.99, "Electronics", True],
            ["Coffee Mug", 12.50, "Kitchen", True],
            ["Book", 15.99, "Education", False],
            ["Headphones", 79.99, "Electronics", True]
        ]
    }
    
    # Export to SQLite
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    try:
        jt_to_sqlite(json_table, db_path, table_name='products')
        print(f"💾 Exported to SQLite: {os.path.basename(db_path)}")
        
        # Verify the data
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            print(f"📋 Tables created: {[t[0] for t in tables]}")
            
            cursor.execute("SELECT COUNT(*) FROM products")
            count = cursor.fetchone()[0]
            print(f"📊 Records inserted: {count}")
            
            cursor.execute("SELECT * FROM products LIMIT 2")
            sample = cursor.fetchall()
            print(f"🔍 Sample data: {sample}")
            
    finally:
        os.unlink(db_path)
    
    print()

def demo_format_hub():
    """Demo JSON-Tables as central format hub."""
    print("🌐 JSON-Tables as Format Hub")
    print("=" * 50)
    
    # Start with pandas DataFrame
    import pandas as pd
    df = pd.DataFrame({
        'employee': ['Alice', 'Bob', 'Carol'],
        'department': ['Engineering', 'Sales', 'Marketing'], 
        'years': [5, 3, 7],
        'remote': [True, False, True]
    })
    print("📊 Starting with pandas DataFrame")
    
    # Convert to JSON-Tables (central format)
    json_table = df_to_jt(df)
    print("🔄 Converted to JSON-Tables (central hub)")
    
    # Export to multiple formats
    formats_created = []
    
    # To standard JSON
    records = jt_to_json(json_table)
    formats_created.append(f"📄 Standard JSON ({len(records)} records)")
    
    # To CSV
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        csv_path = f.name
    jt_to_csv(json_table, csv_path)
    formats_created.append(f"📄 CSV file ({os.path.basename(csv_path)})")
    
    # To SQLite
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    jt_to_sqlite(json_table, db_path, table_name='employees')
    formats_created.append(f"🗄️  SQLite database ({os.path.basename(db_path)})")
    
    # Back to DataFrame
    df_restored = df_from_jt(json_table)
    formats_created.append(f"🐼 Pandas DataFrame ({len(df_restored)} rows)")
    
    print("🎯 Formats created from single JSON-Tables source:")
    for fmt in formats_created:
        print(f"   {fmt}")
    
    # Cleanup
    try:
        os.unlink(csv_path)
        os.unlink(db_path)
    except:
        pass
    
    print("✅ One format → Many destinations!")
    print()

def main():
    """Run all format conversion demos."""
    print("🚀 JSON-Tables Universal Format Converter Demo")
    print("=" * 60)
    print()
    
    demo_json_conversion()
    demo_csv_conversion() 
    demo_sqlite_export()
    demo_format_hub()
    
    print("🎉 Demo complete! JSON-Tables is your tabular data Swiss Army knife.")

if __name__ == "__main__":
    main() 