# JSON‑Tables (JSON‑T) Proposal

## 🧩 Overview
**JSON‑Tables (aka JSON‑T)** is a minimal, backward‑compatible specification for representing tabular data in JSON. It enables easy human‑readable rendering, clear table semantics for tooling, and seamless loading into analytics libraries like **pandas**, spreadsheet apps, and data pipelines.

> **“Finally, a standard for representing tables in JSON—simple to render, easy to parse, and designed for humans and tools alike.”**

---

## 🔥 Before & After: Why This Matters

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
- It’s verbose and vertically fragmented, despite clearly being a table.
- Hard to visually compare rows or spot column-level anomalies.
- Hard to skim or diff in logs.
- Requires external tooling to view as a table.
- This is how JSON is typically rendered by pretty-printers.
- It’s verbose and vertically fragmented, despite clearly being a table.
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

## 1. Motivation
If you're the kind of person who deals with structured data all day—API responses, pipeline outputs, analytics logs, git diffs, or large datasets—you already live in JSON. You use `jq`, you open logs in `vim`, you paste objects into chat windows, and you pass data between services, scripts, and notebooks.

You're someone who notices when something is off by a single space. You think in columns even when you're reading trees. You want to see your data—not decode it.

And yet: default JSON pretty-printers explode tabular data vertically. Tables become forests. Alignment disappears. Visual structure vanishes.

Let’s put it this way:

If you're:
- Skimming logs or scanning API outputs,
- Wrangling data frames or debugging pipelines,
- Building devtools or inspecting traces in `jq`,
- Sharing samples with teammates or dropping JSON into ChatGPT...

You’re already reading tables. You just don’t get to *see* them as tables.

JSON-Tables fixes that.

Instead of pretty-printed forests of curly braces, you get aligned, readable, diffable rows.
You stop wasting vertical space and cognitive energy.
You stop re-parsing column structures in your head.
You stop reimplementing the same table renderers or naming hacks.

You just say one thing: `"__dict_type": "table"`.

To be blunt: if you regularly work with tabular JSON and this doesn’t seem useful to you—*that’s weird*.

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
| `__dict_type` | `"table"` | Signals table object |
| `cols` | `string[]` | Ordered column names |
| `row_data` | `any[][]` | Row‑major values |

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
Compatible with columnar storage systems (e.g., Apache Arrow).

---

## 5. Reference Implementation
A full, MIT‑licensed reference implementation (including CLI) lives in **`jsontables.py`** on GitHub:

👉 **[featrix/json‑tables/jsontables.py](https://github.com/featrix/json-tables/blob/main/jsontables.py)**

The same repository contains unit tests, documentation, and a VS Code preview extension prototype.

---

## 6. Packaging Quick‑Start
```bash
# Clone & install in editable mode
$ git clone https://github.com/featrix/json-tables.git
$ cd json-tables
$ pip install -e .

# CLI usage: pretty‑print tables in any JSON file
$ cat data.json | jsontables           # autodetect & render
$ jsontables --max-width 120 file.json # narrow terminals
```

---

## 7. Future Extensions (Roadmap)
- `col_types`, `col_nullable`, `col_meta`, dataset‑level `meta`.
- Binary/Arrow encoding.
- VS Code / Jupyter syntax‑aware renderers.
- Streaming & chunked table support.

---

## 8. Real‑World Use Cases
- Web APIs returning paged tables with schema.
- AI agents auto‑charting JSON‑T responses.
- Logger/CLI debug dumps.
- Pandas: `df.to_json(table_format="json-t")`.
- Low‑code & spreadsheet import/export.

---

## 9. Status
Open proposal — feedback, issues, and PRs welcome!

---

**Name**: JSON‑T / JSON‑Tables  
**Author**: Mitch Haile, Featrix.ai  
**License**: Public Domain / CC0

---

## 🔗 Related Work
- [W3C “CSV on the Web” / JSON Table Schema](https://specs.frictionlessdata.io/table-schema/)
- [Apache Arrow JSON Format](https://arrow.apache.org/docs/format/Columnar.html#json)
- [Google Visualization API Table Format](https://developers.google.com/chart/interactive/docs/reference#DataTable)
- [JSON-stat](https://json-stat.org/)
- [OpenRefine Export Format](https://docs.openrefine.org/manual/exporting#json)
- [CKAN JSON Table Schema usage](https://docs.ckan.org/en/latest/maintaining/datastore.html#the-json-table-schema)
