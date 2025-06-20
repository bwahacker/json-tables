# JSON-Tables (JSON-T) Proposal

## 🧩 Overview
**JSON-Tables (aka JSON-T)** is a minimal, backward-compatible standard to represent tabular data in JSON. It supports easier rendering, tool interoperability, and direct integration with tabular tools like Pandas, spreadsheets, and data pipelines.

> **“Finally, a standard for representing tables in JSON—simple to render, easy to parse, and designed for humans and tools alike.”**

---

## 🔥 Before & After: Why This Matters

### 😩 The Problem Today
```json
[
  {"name": "Mary", "age": 8, "score": 92},
  {"name": "John", "age": 9, "score": 88}
]
```
- Unstructured visually
- Hard to skim or diff in logs
- Requires external tools to view as a table

### ✅ JSON-Tables Solution
```json
[
  { name: "Mary" , age:  8 , score: 92 },
  { name: "John" , age:  9 , score: 88 }
]
```
- Cleaner, column-aligned, diffable
- Perfect for monospaced logs, tools, terminals

---

## 1. Motivation
JSON lacks ergonomic support for tabular (row-column) data, making it cumbersome for tools and humans alike to parse or visualize. Current workarounds vary widely and lack standard semantics or structure.

This proposal introduces:
- A rendering convention for debuggers and formatters
- A standardized JSON table object with metadata and extensibility

---

## 2. Human-Friendly Rendering: ASCII Table Style

When a list of flat dicts with consistent keys and short primitive values is detected, formatters SHOULD render it as an aligned ASCII table-style array:

```json
[
  { name: "Mary" , age:  8 , score: 92 },
  { name: "John" , age:  9 , score: 88 }
]
```

### Constraints:
- All items must be flat dicts (no nesting)
- Keys must be consistent across all items
- Values must be short strings/numbers/booleans

This rendering allows quick inspection of tabular JSON data in monospaced terminals, logs, and editors.

---

## 3. Canonical Table Object Format (JSON-T)

```json
{
  "__dict_type": "table",
  "cols": ["name", "age", "score"],
  "row_data": [
    ["Mary", 8, 92],
    ["John", 9, 88]
  ],
  "current_page": 0,
  "total_pages": 1,
  "page_rows": 2
}
```

### Required Fields
| Field           | Type       | Description                                  |
|----------------|------------|----------------------------------------------|
| `__dict_type`  | `"table"`  | Identifier for tooling                        |
| `cols`         | `string[]` | Column names in order                        |
| `row_data`     | `any[][]`  | Rows of data, each row is an array of values |

### Optional Fields
| Field           | Type   | Default | Description                                |
|----------------|--------|---------|--------------------------------------------|
| `current_page` | int    | 0       | For paginated tables                        |
| `total_pages`  | int    | 1       | Total number of pages                       |
| `page_rows`    | int    | n/a     | Number of rows on this page                |

---

## 4. Columnar Format (Optional Alternative)

```json
{
  "__dict_type": "table",
  "cols": ["name", "age", "score"],
  "column_data": {
    "name": ["Mary", "John"],
    "age": [8, 9],
    "score": [92, 88]
  },
  "row_data": null
}
```

### Benefits
- Efficient for memory and disk
- Compatible with columnar storage systems like Apache Arrow
- Easier to transform for analytical queries

---

## 5. Compatibility
- Non-table-aware tools treat this as a normal JSON dict
- `row_data` and `column_data` are interchangeable via transposition
- Schema inference is possible using `cols` and value inspection

---

## 6. Future Extensions
- `col_types`: Explicit type hints (e.g., `"string"`, `"int"`, `"date"`)
- `col_nullable`: Boolean/nullability flags
- `col_meta`: Per-column metadata (units, labels, formats)
- `meta`: Global metadata (source, timestamp, license, etc.)

---

## 7. Real-World Use Cases
- Web APIs returning tabular data with pagination and schema
- AI tools generating charts or summaries from `JSON-T`
- CLI tools or loggers pretty-printing structured rows
- Pandas `.to_json(table_format="json-t")`
- Low-code platforms or spreadsheet import/export

---

## 8. Status
This is an open proposal for discussion and implementation by developers, formatter authors, data tool creators, and the broader JSON tooling community.

Feedback, improvements, and adoption interest are welcome.

---

**Name**: JSON-T / JSON-Tables  
**Author**: Mitch Haile / Featrix.ai  
**License**: Public Domain / CC0

---

## 🔗 Related Work
- [W3C CSV on the Web / JSON Table Schema](https://specs.frictionlessdata.io/table-schema/)
- [Apache Arrow JSON Format](https://arrow.apache.org/)
- [Google Visualization API Table Format](https://developers.google.com/chart/interactive/docs/reference#datatable-object)
- [JSON-stat](https://json-stat.org/)
