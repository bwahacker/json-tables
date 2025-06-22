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

**JSON-Tables has been comprehensively benchmarked against CSV, JSON, and Parquet across multiple dimensions:**

### 📊 Storage Efficiency
- **JSON-Tables v1**: 45-55% smaller than standard JSON
- **JSON-Tables v2**: Up to 61% smaller than JSON, competitive with CSV when compressed
- **Compression friendly**: Achieves CSV-level storage efficiency when gzipped

### ⚡ Performance Analysis
**Comprehensive profiling reveals:**
- **pandas operations** dominate execution time (DataFrame.to_dict is the main bottleneck)
- **Encoding scales linearly** with dataset size
- **Append operations** can be optimized from O(n) to **O(1)** with format design
- **JSONL variant** achieves **173x faster appends** than traditional JSON-Tables

### 🔍 Key Benchmarking Results

| Operation | CSV | JSON | JSON-Tables v1 | JSON-Tables v2 |
|-----------|-----|------|----------------|----------------|
| **Storage (uncompressed)** | Baseline | 2.5-5.1x larger | 1.3-1.7x larger | **1.1-1.2x larger** |
| **Storage (gzipped)** | Baseline | 16-81% larger | 3-14% larger | **0-2% larger** |
| **Encode Speed** | Fastest | 2.74x slower | 1.71x slower | **2.05x slower** |
| **Decode Speed** | Baseline | 0.93x faster | 2.56x slower | **1.10x slower** |
| **Append Speed** | 0.057ms (O(1)) | 4-20ms (O(n)) | 4-20ms (O(n)) | **0.092ms (O(1) JSONL)** |

### 📈 Real-World Performance Insights
- **JSON-Tables v2 + gzip ≈ CSV + gzip** in storage size
- **Only 10% slower decode** than CSV while maintaining full human readability
- **JSONL variant achieves near-CSV append performance** (2x slower) with O(1) complexity
- **Categorical encoding** in v2 provides significant space savings for real-world data

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
- Column‑aligned, readable, diff‑friendly.
- Perfect for log files, CLIs, and notebooks.

---

## 🔧 Profiling & Performance Monitoring

**JSON-Tables includes comprehensive profiling capabilities:**

```python
from jsontables import profiling_session

with profiling_session("my_operation"):
    # Your JSON-Tables operations here
    encoded = JSONTablesEncoder.from_records(data)
    decoded = JSONTablesDecoder.to_records(encoded)

# Detailed timing breakdown automatically printed
```

**Key profiling insights:**
- Identifies performance bottlenecks in real-time
- Tracks library overhead (pandas, json) vs pure Python
- Provides operation-level timing with call path analysis
- Enables data-driven optimization decisions

---

## 🎯 Advanced Features

### JSON-Tables v2 Optimizations
- **Algorithmic schema analysis** determines when optimizations provide net benefits
- **Categorical encoding** converts repeated strings to integers
- **Schema variations** handle sparse data efficiently
- **Default value omission** for homogeneous datasets

### Append-Friendly Formats
- **Optimized append operations** with 3.2x speedup over naive approaches
- **JSONL variant** for true O(1) append performance
- **Streaming support** for real-time data processing

---

## 1. Motivation
If you're the kind of person who deals with structured data all day—API responses, pipeline outputs, analytics logs, git diffs, or large datasets—you already live in JSON. You use `jq`, you open logs in `vim`, you paste objects into chat windows, and you pass data between services, scripts, and notebooks.

You're someone who notices when something is off by a single space. You think in columns even when you're reading trees. You want to see your data—not decode it.

And yet: default JSON pretty-printers explode tabular data vertically. Tables become forests. Alignment disappears. Visual structure vanishes.

Let's put it this way:

If you're:
- Skimming logs or scanning API outputs,
- Wrangling data frames or debugging pipelines,
- Building devtools or inspecting traces in `jq`,
- Sharing samples with teammates or dropping JSON into ChatGPT...

You're already reading tables. You just don't get to *see* them as tables.

JSON-Tables fixes that.

Instead of pretty-printed forests of curly braces, you get aligned, readable, diffable rows.
You stop wasting vertical space and cognitive energy.
You stop re-parsing column structures in your head.
You stop reimplementing the same table renderers or naming hacks.

You just say one thing: `"__dict_type": "table"`.

To be blunt: if you regularly work with tabular JSON and this doesn't seem useful to you—*that's weird*.

We built the modern data world on JSON, and yet there's never been a common way to say "this is a table." This proposal fixes that.

---

## 2. Human‑Friendly Rendering: ASCII Table Style
A renderer **SHOULD** align flat row objects if:
- Rows share identical keys.
- Values are primitives (string, number, boolean, null).
- Total rendered width ≤ **300 characters** (configurable).

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

## 🎯 Format Comparison: How Savings Scale with Data Size

**The more columns you have, the more JSON-Tables saves!** Here's why:

### 🔍 The Problem: JSON Repeats Field Names
Standard JSON repeats every field name for every row:
```json
[
  {"employee_id": "EMP_001", "first_name": "Alice", "department": "Engineering", "salary": 85000},
  {"employee_id": "EMP_002", "first_name": "Bob", "department": "Engineering", "salary": 92000},
  {"employee_id": "EMP_003", "first_name": "Carol", "department": "Marketing", "salary": 78000}
]
```

### ✅ JSON-Tables Solution: Store Schema Once
```json
{
  "__dict_type": "table",
  "cols": ["employee_id", "first_name", "department", "salary"],
  "row_data": [
    ["EMP_001", "Alice", "Engineering", 85000],
    ["EMP_002", "Bob", "Engineering", 92000], 
    ["EMP_003", "Carol", "Marketing", 78000]
  ]
}
```

### 📈 Savings Scale with Column Count

| Dataset | JSON Size | JSON-T Size | Space Saved | % Reduction |
|---------|-----------|-------------|-------------|-------------|
| **1K rows × 5 cols** | 141 KB | 78 KB | 63 KB | **44.8%** |
| **1K rows × 10 cols** | 254 KB | 130 KB | 124 KB | **48.7%** |
| **1K rows × 15 cols** | 378 KB | 167 KB | 212 KB | **55.9%** |
| **1K rows × 20 cols** | 505 KB | 207 KB | 298 KB | **59.0%** |

### 🚀 Real-World Impact at Scale

| Dataset | JSON Size | JSON-T Size | Space Saved | % Reduction |
|---------|-----------|-------------|-------------|-------------|
| **5K rows × 5 cols** | 719 KB | 402 KB | 317 KB | **44.1%** |
| **5K rows × 10 cols** | 1.3 MB | 0.6 MB | 0.6 MB | **48.3%** |
| **5K rows × 15 cols** | 1.9 MB | 0.8 MB | 1.0 MB | **55.6%** |
| **5K rows × 20 cols** | 2.5 MB | 1.0 MB | 1.5 MB | **58.8%** |

### 🎯 Key Insights:

**📊 Column Count Matters Most:**
- 5 columns → **45% savings**
- 10 columns → **49% savings**  
- 15 columns → **56% savings**
- 20 columns → **59% savings**

**🔥 The Sweet Spot:**
- **10+ columns**: JSON-Tables saves ~50% or more
- **15+ columns**: JSON-Tables saves ~55% or more
- **Real databases/APIs**: Often have 20-50+ columns = **massive savings**

**💰 Production Impact:**
- Employee database (50K × 25 cols) → **~25MB saved**
- Product catalog (100K × 30 cols) → **~60MB saved**  
- Transaction logs (1M × 15 cols) → **~200MB saved**

*JSON-Tables isn't just an optimization—it's a fundamental improvement for tabular data!*

---

## 🚀 Advanced Optimizations (Roadmap)

**The current format is just the beginning.** Advanced JSON-Tables can achieve even greater savings for real-world data patterns:

### 🔍 The Opportunity: Sparse & Categorical Data
Most real-world datasets have:
- **Sparse data**: Many null/missing values
- **Repeated categories**: "Active", "Premium", "Engineering" appear thousands of times  
- **Default values**: 80% of customers have "Standard" status
- **Schema variations**: Different object types need different fields

### ✨ Advanced Features

**🏷️ Schema Variations (`__jt_sid`)**
```json
{
  "__dict_type": "table",
  "schemas": {
    "0": {"defaults": [null, null, "Active", "Standard", 1.0]},
    "1": {"defaults": [null, null, "Inactive", "Premium", 2.0]}
  },
  "row_data": [
    ["user_001", "Alice", "__jt_sid", 0],           // Uses schema 0 defaults
    ["user_002", "Bob", "__jt_sid", 1, "Special"], // Schema 1, custom value
  ]
}
```

**🔢 Categorical Encoding**
```json
{
  "schemas": {
    "0": {
      "categoricals": {
        "status": ["Active", "Inactive", "Pending"],
        "tier": ["Basic", "Standard", "Premium"]
      }
    }
  },
  "row_data": [
    ["user_001", 0, 1],  // "Active", "Standard"
    ["user_002", 2, 2]   // "Pending", "Premium"  
  ]
}
```

**⚡ Default Value Omission**
Only store values that differ from schema defaults—massive savings for homogeneous data.

### 📊 Advanced Savings Example

| Optimization Level | Size | vs Standard | Use Case |
|-------------------|------|-------------|----------|
| **Standard JSON** | 213 KB | - | Baseline |
| **JSON-Tables v1** | 110 KB | **-48%** | Basic tabular data |
| **JSON-Tables v2** | 83 KB | **-61%** | Sparse + categorical data |

**🎯 Where Advanced Optimizations Excel:**
- **Customer databases**: Lots of optional fields → **~60% savings**
- **Product catalogs**: Many categories, sparse attributes → **~65% savings**
- **Event logs**: Repeated patterns, optional metadata → **~70% savings**
- **API responses**: Conditional fields, enums → **~55% savings**

### 🔬 Technical Benefits

**Beyond Size Savings:**
- ✅ **Type safety**: Categorical encoding prevents typos
- ✅ **Schema evolution**: Add new categories without data migration  
- ✅ **Query optimization**: Integer categories = faster filtering
- ✅ **Compression friendly**: Highly repetitive structure compresses better

**🌟 Still JSON:**
- ✅ **Human readable** (with proper tooling)
- ✅ **Diff-friendly** (schema changes are visible)
- ✅ **Tool compatible** (any JSON parser works)
- ✅ **No binary dependencies**

*Advanced JSON-Tables: Getting close to binary efficiency while staying in the JSON ecosystem.*

---

## 7. Development Quick‑Start
```