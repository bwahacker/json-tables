# Compute Time Analysis Summary

## 📊 Performance vs CSV Baseline (Average Across All Datasets)

| Format | Encode Time | Decode Time |
|--------|------------|-------------|
| **JSON-Tables v1** | **1.71x slower** | **2.56x slower** |
| **JSON-Tables v2** | **2.05x slower** | **1.10x slower** |
| **JSON** | **2.74x slower** | **0.93x faster** |

## 📊 Performance vs JSON Baseline (Average Across All Datasets)

| Format | Encode Time | Decode Time |
|--------|------------|-------------|
| **JSON-Tables v1** | **0.64x faster** | **2.75x slower** |
| **JSON-Tables v2** | **0.77x faster** | **1.19x slower** |
| **CSV** | **0.38x faster** | **1.14x faster** |

## 🎯 Key Insights

### Performance Leaders
- **Encode:** CSV wins consistently across all datasets
- **Decode:** Mixed results - JSON wins small/medium datasets, CSV wins large datasets

### JSON-Tables v1 vs v2
- **v1 Encode:** Faster than v2 (analysis overhead in v2)
- **v2 Decode:** Much faster than v1 (optimizations pay off)
- **v2 Overall:** Better balanced performance profile

### Scaling Characteristics
- **CSV:** Most predictable scaling, good efficiency
- **JSON:** Good scaling on medium datasets, struggles with wide data
- **JSON-Tables v2:** Best overall scaling efficiency
- **JSON-Tables v1:** Moderate scaling performance

## 💡 Practical Implications

### When Performance Cost is Acceptable
- **JSON-Tables v1:** 1.7x encode cost, 2.6x decode cost vs CSV → Acceptable for many use cases
- **JSON-Tables v2:** 2.0x encode cost, 1.1x decode cost vs CSV → Very reasonable for readable format

### When JSON-Tables Outperforms JSON
- **Encode:** Both v1 and v2 are 20-40% faster than JSON
- **Decode:** v2 is competitive with JSON, v1 is slower due to conversion overhead

### The Sweet Spot
JSON-Tables v2 provides the best balance:
- Only 10% slower decode than CSV
- 23% faster encode than JSON
- Human-readable format
- Excellent storage efficiency when compressed

This analysis confirms that JSON-Tables successfully bridges the gap between performance and readability, with v2 offering the optimal balance for most use cases. 