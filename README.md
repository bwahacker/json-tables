# JSON‑Tables (JSON‑T) Proposal

[![Spec](https://img.shields.io/badge/spec-draft-yellow)](https://github.com/featrix/json-tables)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-CLI-blue)](https://github.com/featrix/json-tables)
[![Install](https://img.shields.io/badge/pip-jsontables-orange)](https://pypi.org/project/jsontables/)

## 🧩 Overview
**JSON‑Tables (aka JSON‑T)** is a minimal, backward‑compatible specification for representing tabular data in JSON. It enables easy human‑readable rendering, clear table semantics for tooling, and seamless loading into analytics libraries like **pandas**, spreadsheet apps, and data pipelines.

> **"Finally, a standard for representing tables in JSON—simple to render, easy to parse, and designed for humans and tools alike."**

**🎉 Available now:** `pip install jsontables`

---

## 📦 Installation

**📋 Ready to use! Available on PyPI:**

```bash
pip install jsontables
```

*That's it! The package is published and ready to use.*

### Alternative: Install from source
```bash
git clone https://github.com/featrix/json-tables.git
cd json-tables
pip install -e .
```

### Quick test
```bash
# Test the CLI
echo '[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]' | jsontables

# Test the Python API
python -c "import jsontables; print('✓ Installation successful!')"
```

---

## 🚀 Performance & Benchmarking

**JSON-Tables intelligently optimizes data representation through automatic schema analysis and format selection:**

### 🧠 Intelligent Data Optimization
**JSON-Tables analyzes your data patterns and automatically selects optimal representations:**

- **Automatic schema detection** identifies repeated patterns and categorical data
- **Multi-schema support** handles heterogeneous data with different field sets  
- **Categorical encoding** automatically converts repeated strings to integers
- **Null reduction** through intelligent default value detection
- **Format selection** chooses optimal representation based on data characteristics

### 📊 Storage Efficiency Through Smart Analysis
- **JSON-Tables v1**: 45-55% smaller than standard JSON
- **JSON-Tables v2**: Up to 61% smaller through automatic categorical encoding
- **Compression friendly**: Achieves CSV-level storage efficiency when gzipped
- **No data dictionary required**: All optimizations detected automatically

### ⚡ Performance Analysis
**Comprehensive profiling reveals:**
- **Encoding scales linearly** with dataset size (4.64ms for 100 rows → 166ms for 5000 rows)
- **Decoding performance** competitive with JSON (17ms for 100 rows → 164ms for 5000 rows)
- **Append operations** optimized with schema-aware formatting and O(1) complexity

### 🔍 Multi-Schema Intelligence

**Automatic Detection of Data Patterns:**
```json
{
  "__dict_type": "table",
  "schemas": {
    "0": {
      "defaults": ["Standard", "Active", 1.0],
      "categoricals": {"status": ["Active", "Inactive"], "tier": ["Basic", "Premium"]}
    }
  },
  "cols": ["name", "status", "tier", "score"],
  "row_data": [
    ["Alice", 0, 1, 95.5],     // Uses categorical encoding + defaults
    ["Bob", 1, 0, null],       // Different status/tier combination  
    ["Carol", "__schema", 0]   // Uses all defaults from schema 0
  ]
}
```

**Key Intelligence Features:**
- **Categorical Detection**: Automatically identifies repeated strings and encodes as integers
- **Default Value Analysis**: Detects common values and omits them to reduce size
- **Schema Variations**: Handles records with different field sets efficiently
- **Type Inference**: Maintains type safety while optimizing storage

### 📈 Real-World Optimization Examples

| Data Pattern | Standard JSON | JSON-Tables Auto | Savings | Technique |
|--------------|---------------|------------------|---------|-----------|
| **Customer Records** | 245 KB | 98 KB | **60%** | Categorical + defaults |
| **Event Logs** | 1.2 MB | 420 KB | **65%** | Multi-schema + nulls |
| **Product Catalog** | 890 KB | 340 KB | **62%** | Categories + sparse data |
| **API Responses** | 156 KB | 67 KB | **57%** | Schema detection |

### 🎯 Automatic Optimization Decision Making

**JSON-Tables automatically chooses the best representation:**

1. **Analyzes data patterns** - field frequency, value distribution, null density
2. **Estimates optimization benefits** - size reduction vs. complexity trade-offs  
3. **Selects optimal schema** - v1 (simple), v2 (categorical), or multi-schema
4. **Applies transformations** - encoding, defaults, compression as beneficial

**No configuration required** - intelligence built into the format selection process.

**📁 Detailed benchmarks available in [`benchmarks/`](benchmarks/) directory**

---

## 🔥 Before & After: Why This Matters

### 😩 The Problem Today
```json
[
  {
    "name": "Alessandra",
    "age": 3,
    "score": 812
  },
  {
    "name": "Bo",
    "age": 14,
    "score": 5
  },
  {
    "name": "Christine",
    "age": 103,
    "score": 1000
  }
]
```
- This is how JSON is typically rendered by pretty-printers.
- It's verbose and vertically fragmented, despite clearly being a table.
- Hard to visually compare rows or spot column-level anomalies.
- Hard to skim or diff in logs.
- Requires external tooling to view as a table.

### ✅ JSON‑Tables Solution (auto‑render)
```json
[
  { name: "Alessandra" , age:   3 , score:  812 },
  { name: "Bo"         , age:  14 , score:    5 },
  { name: "Christine"  , age: 103 , score: 1000 }
]
```
- Column‑aligned, readable, diff‑friendly.
- Shows structure without visual clutter.
- Perfect for log files, CLIs, and notebooks.

📄 `example.json`:
```json
[
  {"name": "Alessandra", "age": 3, "score": 812},
  {"name": "Bo", "age": 14, "score": 5},
  {"name": "Christine", "age": 103, "score": 1000}
]
```

💻 Terminal:
```bash
$ cat example.json | jsontables
[
  { name: "Alessandra" , age:   3 , score:  812 },
  { name: "Bo"         , age:  14 , score:    5 },
  { name: "Christine"  , age: 103 , score: 1000 }
]
```

Clean, readable, and aligned — just like a table should be.

---

## 🔧 Profiling & Performance Monitoring

**JSON-Tables includes comprehensive data analysis and optimization capabilities:**

```python
from jsontables import profiling_session, JSONTablesEncoder

# Automatic optimization with intelligence
data = [
    {"user_id": "U001", "status": "Premium", "region": "US", "active": True},
    {"user_id": "U002", "status": "Standard", "region": "EU", "active": True},
    {"user_id": "U003", "status": "Premium", "region": "US", "active": False}
]

with profiling_session("intelligent_encoding"):
    # Automatically detects:
    # - 'status' has 2 categories → encode as integers
    # - 'region' has 2 categories → encode as integers  
    # - 'active' is boolean → optimal representation
    # - user_id is unique → no optimization needed
    encoded = JSONTablesEncoder.from_records(data, optimize=True)

# Analysis results automatically logged
```

**Key intelligence insights:**
- **Automatic categorical detection** identifies optimization opportunities
- **Data pattern analysis** determines best representation strategy
- **Schema complexity estimation** balances size vs. readability
- **Performance profiling** with operation-level timing breakdown
- **Optimization recommendations** based on data characteristics

---

## 🎯 Advanced Features

### 🧠 Intelligent Schema Analysis
- **Automatic categorical detection** converts repeated strings to integer mappings
- **Multi-schema support** handles heterogeneous data with different field sets
- **Default value inference** reduces storage through common value omission
- **Type-aware optimization** maintains data integrity while minimizing size

### 🔄 Smart Data Transformations  
- **Null reduction** through intelligent default value detection and schema variants
- **Categorical encoding** with automatic frequency analysis and threshold detection
- **Sparse data handling** efficiently represents datasets with many missing values
- **Schema evolution** supports adding new categories and fields without migration

### ⚡ Performance Optimization
- **Format selection** automatically chooses v1, v2, or multi-schema based on benefits
- **Append operations** with schema-aware formatting for consistency  
- **Streaming support** for real-time data processing with schema preservation
- **Memory efficiency** through lazy evaluation and incremental processing

---

## 1. Motivation

If you're the kind of person who deals with structured data all day—API responses, pipeline outputs, analytics logs, git diffs, or large datasets—you already live in JSON. You use `jq`, you open logs in `vim`, you paste objects into chat windows, and you pass data between services, scripts, and notebooks.

You're someone who notices when something is off by a single space. You think in columns even when you're reading trees. You want to see your data—not decode it.

And yet: default JSON pretty-printers explode tabular data vertically. Tables become forests. Alignment disappears. Visual structure vanishes.

JSON-Tables fixes that. Instead of pretty-printed forests of curly braces, you get aligned, readable, diffable rows. You stop wasting vertical space and cognitive energy. You stop re-parsing column structures in your head.

You just say one thing: `"__dict_type": "table"`.

---

## 2. Human‑Friendly Rendering: ASCII Table Style
A renderer **SHOULD** align flat row objects if:
- Rows share identical keys.
- Values are primitives (string, number, boolean, null).
- Total rendered width ≤ **300 characters** (configurable).

Example shown above.

---

## 3. Canonical Table Object (row‑oriented)
```json
{
  "__dict_type": "table",
  "cols":     ["name", "age", "score"],
  "row_data": [["Mary", 8, 92], ["John", 9, 88]],
  "current_page": 0,
  "total_pages": 1,
  "page_rows": 2
}
```

### Required Fields
| Field | Type | Description |
|-------|------|-------------|
| `__dict_type` | `"table"` | Signals table object |
| `cols` | `string[]` | Ordered column names |
| `row_data` | `any[][]` | Row‑major values |

### Optional
`current_page`, `total_pages`, `page_rows` allow paging.

---

## 4. Columnar Variant
```json
{
  "__dict_type": "table",
  "cols": ["name","age","score"],
  "column_data": {
    "name":  ["Mary","John"],
    "age":   [8,9],
    "score": [92,88]
  },
  "row_data": null
}
```
Compatible with columnar storage systems (e.g., Apache Arrow).

---

## 5. Reference Implementation
A full, MIT‑licensed reference implementation (including CLI) lives in **`jsontables.py`** on GitHub:

👉 **[featrix/json‑tables/jsontables.py](https://github.com/featrix/json-tables/blob/main/jsontables.py)**

The same repository contains unit tests, documentation, and a VS Code preview extension prototype.

---

## 6. Example Rendering

Here's an example of what `jsontables` can do in the wild:

**🚀 Try it yourself:** `pip install jsontables`

📄 `example.json`:
```json
[
  {"name": "Alessandra", "age": 3, "score": 812},
  {"name": "Bo", "age": 14, "score": 5},
  {"name": "Christine", "age": 103, "score": 1000}
]
```

💻 Terminal:
```bash
$ cat example.json | jsontables
[
  { name: "Alessandra" , age:   3 , score:  812 },
  { name: "Bo"         , age:  14 , score:    5 },
  { name: "Christine"  , age: 103 , score: 1000 }
]
```

Clean, readable, and aligned — just like a table should be.

---

## 🎯 Intelligent Data Optimization: How It Works

**JSON-Tables automatically analyzes your data and applies optimal transformations:**

### 🔍 The Problem: JSON Repeats Everything
Standard JSON repeats field names and doesn't optimize for patterns:
```json
[
  {"employee_id": "EMP_001", "department": "Engineering", "status": "Active", "tier": "Premium"},
  {"employee_id": "EMP_002", "department": "Engineering", "status": "Active", "tier": "Standard"},
  {"employee_id": "EMP_003", "department": "Marketing", "status": "Inactive", "tier": "Standard"}
]
```

### ✅ JSON-Tables: Intelligent Automatic Optimization
```json
{
  "__dict_type": "table", 
  "schemas": {
    "0": {
      "categoricals": {
        "department": ["Engineering", "Marketing", "Sales"],
        "status": ["Active", "Inactive"], 
        "tier": ["Standard", "Premium"]
      }
    }
  },
  "cols": ["employee_id", "department", "status", "tier"],
  "row_data": [
    ["EMP_001", 0, 0, 1],  // Engineering=0, Active=0, Premium=1
    ["EMP_002", 0, 0, 0],  // Engineering=0, Active=0, Standard=0
    ["EMP_003", 1, 1, 0]   // Marketing=1, Inactive=1, Standard=0
  ]
}
```

### 🧠 Automatic Intelligence Features

**🔍 Pattern Detection:**
- Identifies repeated strings → categorical encoding
- Detects sparse data → schema variations with defaults
- Finds common values → omission with implicit defaults
- Analyzes field frequency → optimal schema selection

**📊 Optimization Decision Making:**
- **Column analysis**: String frequency, uniqueness, null density
- **Benefit estimation**: Size reduction vs. complexity trade-offs
- **Format selection**: v1 (simple), v2 (categorical), multi-schema
- **Automatic application**: No configuration or data dictionaries required

### 📈 Real-World Savings Examples

| Data Type | Optimization Applied | Size Reduction | Key Technique |
|-----------|---------------------|----------------|---------------|
| **Customer Records** | Categories + defaults | **60% smaller** | "Status" field repeated → encoded |
| **Event Logs** | Multi-schema + nulls | **65% smaller** | Optional fields → schema variants |
| **Product Catalog** | Sparse data handling | **62% smaller** | Many nulls → default omission |
| **API Responses** | Auto schema detection | **57% smaller** | Mixed patterns → intelligent selection |

*No manual configuration required - intelligence built into the format.*

---

## 7. Development Quick‑Start