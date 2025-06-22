# JSON-Tables Benchmarks

This directory contains comprehensive benchmarks comparing JSON-Tables with other data formats.

## 📁 Directory Structure

```
benchmarks/
├── data/                    # Test datasets
│   └── 8000-boston.csv     # Real-world Boston real estate data (8K rows, 90 cols)
├── scripts/                # Benchmark scripts
│   ├── benchmark_performance.py      # Comprehensive performance + compression analysis
│   └── compute_time_comparison.py    # Focused compute time comparison vs baselines
├── results/                # Generated results (created when running benchmarks)
└── README.md               # This file
```

## 🚀 Quick Start

### Prerequisites

Make sure you have the required dependencies:
```bash
pip install pandas pyarrow numpy
```

### Running Benchmarks

From the project root directory:

```bash
# Comprehensive benchmark (performance + compression + storage)
cd benchmarks/scripts
python benchmark_performance.py

# Focused compute time comparison
python compute_time_comparison.py
```

## 📊 Benchmark Scripts

### 1. `benchmark_performance.py`
**Complete performance analysis including:**
- Encode/decode/add_row timing (with statistical confidence intervals)
- Gzip compression/decompression timing  
- Storage efficiency (uncompressed and compressed sizes)
- All formats: CSV, JSON, JSON-Tables v1, JSON-Tables v2

**Key Features:**
- Multiple iterations for statistical reliability
- Optimized CSV append (O(1) vs O(n))
- Real-world and synthetic datasets
- Compression ratio analysis

### 2. `compute_time_comparison.py`
**Focused compute time analysis:**
- Direct performance comparison vs CSV and JSON baselines
- Scaling efficiency analysis
- Statistical analysis of performance ratios
- Identifies consistent performance leaders

## 📋 Test Datasets

### Synthetic Data
- **Medium:** 1K rows × 7 columns (typical API response size)
- **Large:** 5K rows × 7 columns (batch processing size)

### Real-World Data
- **Boston Real Estate:** 8K rows × 90 columns
  - Source: Real estate assessment data
  - Mixed data types (strings, numbers, nulls)
  - Representative of complex real-world datasets

## 🎯 Key Findings Summary

### Performance (Compute Time)

**Encode Performance vs CSV:**
- JSON-Tables v1: **1.71x slower** on average
- JSON-Tables v2: **2.05x slower** on average (due to optimization analysis)
- JSON: **2.74x slower** on average

**Decode Performance vs CSV:**
- JSON-Tables v1: **2.56x slower** on average  
- JSON-Tables v2: **1.10x slower** on average (only 10% slower!)
- JSON: **0.93x faster** on average

**Encode Performance vs JSON:**
- JSON-Tables v1: **0.64x faster** (36% faster) on average
- JSON-Tables v2: **0.77x faster** (23% faster) on average
- CSV: **0.38x faster** (62% faster) on average

**Decode Performance vs JSON:**
- JSON-Tables v1: **2.75x slower** on average
- JSON-Tables v2: **1.19x slower** on average
- CSV: **1.14x faster** on average

### Storage Efficiency (Uncompressed)

**vs CSV baseline:**
- JSON: 2.5-5.1x larger
- JSON-Tables v1: 1.3-1.7x larger  
- JSON-Tables v2: 1.1-1.2x larger

### Storage Efficiency (Compressed with gzip)

**vs CSV (gzipped):**
- JSON: 16-81% larger
- JSON-Tables v1: 3-14% larger
- JSON-Tables v2: **0-2% larger** (essentially identical!)

**vs JSON (gzipped):**
- CSV: 14-45% smaller
- JSON-Tables v1: 10-37% smaller
- JSON-Tables v2: **14-46% smaller**

### Compression Effectiveness

**Compression ratios (original → compressed):**
- JSON: 87-96% reduction (excellent compression due to redundancy)
- JSON-Tables v2: 76-92% reduction (very good compression)  
- JSON-Tables v1: 78-93% reduction (very good compression)
- CSV: 72-90% reduction (good baseline compression)

### Add Row Performance

**Optimized CSV append:** 0.057-0.307ms (**O(1) operation** - 70x faster improvement!)
**JSON/JSON-Tables:** 4-20ms (**O(n) operation** - requires full rewrite)

### Gzip Compression/Decompression Speed

**Compression time (gzip):**
- CSV: Fastest baseline
- JSON-Tables v2: ~1.7x slower than CSV
- JSON-Tables v1: ~2.0x slower than CSV  
- JSON: ~2.7x slower than CSV

**Decompression time (gunzip):**
- All formats perform similarly (0.2-0.8ms range for medium datasets)

## 🔬 Methodology

### Statistical Reliability
- Multiple iterations per test (3-5 depending on dataset size)
- Mean ± standard deviation reported
- Confidence intervals for timing measurements

### Test Environment
- Python 3.11
- pandas, pyarrow, numpy libraries
- macOS (darwin 24.2.0)

### Measurement Categories
1. **Encode Time:** Write data to file format
2. **Decode Time:** Read data from file format  
3. **Add Row Time:** Append single row to existing file
4. **Compression Time:** Gzip compression of file
5. **Decompression Time:** Gzip decompression of file
6. **Storage Size:** File size (uncompressed and compressed)

## 💡 Practical Recommendations

Based on comprehensive benchmark results:

### Choose CSV when:
- **Maximum raw performance** is critical (fastest encode/decode)
- **Frequent append operations** (O(1) append vs O(n) for JSON formats)
- **Universal compatibility** is required
- **Smallest compressed size** is needed (though JSON-Tables v2 is very close)

### Choose JSON-Tables v1 when:
- Need **human-readable data** with good performance
- **Simpler implementation** is preferred (no optimization analysis overhead)
- Moderate performance cost acceptable (1.7x encode, 2.6x decode vs CSV)
- **Better than JSON** for encode speed (36% faster)

### Choose JSON-Tables v2 when:
- Need **best balance** of readability and efficiency
- Working with **larger datasets** (1K+ rows) where optimizations matter
- **Storage efficiency critical** (compresses to CSV size while staying readable)
- **Advanced optimizations** provide value (auto-analysis, categorical encoding)
- **Excellent decode performance** (only 10% slower than CSV!)

### Choose Standard JSON when:
- **Maximum compatibility** with existing tools is required
- **Small datasets** where performance differences are negligible  
- **Decode speed** is more important than storage efficiency (faster than JSON-Tables for decode)
- **Established tooling** and workflows require standard JSON

## 🎯 The Sweet Spots

### JSON-Tables v2 Advantages:
- **Compresses to CSV size** (0-2% difference when gzipped)
- **46% smaller than JSON** when compressed
- **Human-readable** unlike compressed alternatives  
- **23% faster encode** than standard JSON
- **Only 10% slower decode** than CSV

### When Performance Trade-offs Are Worth It:
- **2x encode cost vs CSV** → Acceptable for most applications
- **1.1x decode cost vs CSV** → Minimal impact (10% slower)
- **Massive readability gain** → Invaluable for debugging, APIs, logs
- **Storage competitive** → No longer a penalty when compressed

### Real-World Scenarios:
- **APIs:** JSON-Tables v2 + gzip ≈ CSV + gzip in size, infinitely more readable
- **Logging:** JSON-Tables v2 provides readable logs with reasonable storage cost  
- **Data Exchange:** Self-contained, optimized, human-inspectable
- **Development:** Only readable format that compresses efficiently

## 🔄 Reproducing Results

To reproduce the benchmark results:

1. **Clone the repository and install dependencies**
2. **Navigate to benchmarks/scripts**
3. **Run the benchmark scripts**
4. **Results will be displayed in terminal and can be saved to results/ directory**

The benchmarks use deterministic random seeds for reproducible synthetic data generation.

## ⚡ Append Optimization Analysis

**Major Discovery:** JSON-Tables can achieve **CSV-level append performance** through format design!

For detailed analysis of append optimization strategies, see:
📄 **[Append Optimization Analysis](results/append_optimization_analysis.md)**

### Key Findings:
- **JSONL variant**: 173x faster appends than traditional JSON-Tables
- **O(1) complexity**: True constant-time appends like CSV
- **Smaller file size**: JSONL format is 13% more compact
- **Full compatibility**: Still valid JSON with ecosystem support

### Quick Comparison:
| Format | 5K Row Append | Complexity | 
|--------|---------------|------------|
| Traditional JSON-Tables | 15.93ms | O(n) |
| Optimized JSON-Tables | 4.98ms | O(n) |
| **JSONL JSON-Tables** | **0.092ms** | **O(1)** |
| CSV (baseline) | 0.057ms | O(1) |

**Bottom Line:** JSONL JSON-Tables achieves near-CSV performance (2x slower) while maintaining full human readability and JSON compatibility.

## 📈 Future Enhancements

Potential additions to the benchmark suite:
- Memory usage analysis
- Concurrent access patterns
- Network transfer simulation
- Integration with data processing frameworks
- Language-specific implementations comparison 