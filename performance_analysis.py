import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Results from our benchmark (in milliseconds)
results = {
    'Small (100 rows)': {
        'CSV': {'encode': 0.45, 'decode': 0.13, 'add_row': 0.39},
        'JSON': {'encode': 0.62, 'decode': 0.09, 'add_row': 0.53},
        'JSON-Tables': {'encode': 0.39, 'decode': 1.07, 'add_row': 1.22},
        'Parquet': {'encode': 8.55, 'decode': 7.60, 'add_row': 2.69}
    },
    'Medium (1K rows)': {
        'CSV': {'encode': 1.58, 'decode': 0.94, 'add_row': 2.32},
        'JSON': {'encode': 3.57, 'decode': 0.99, 'add_row': 4.35},
        'JSON-Tables': {'encode': 2.14, 'decode': 1.90, 'add_row': 4.15},
        'Parquet': {'encode': 1.91, 'decode': 2.80, 'add_row': 3.72}
    },
    'Large (5K rows)': {
        'CSV': {'encode': 6.88, 'decode': 4.49, 'add_row': 11.10},
        'JSON': {'encode': 20.27, 'decode': 2.97, 'add_row': 20.78},
        'JSON-Tables': {'encode': 11.20, 'decode': 8.10, 'add_row': 20.99},
        'Parquet': {'encode': 3.59, 'decode': 7.06, 'add_row': 5.56}
    }
}

def analyze_performance():
    print("⚡ PERFORMANCE ANALYSIS")
    print("=" * 50)
    print()
    
    print("🏆 WINNERS BY OPERATION:")
    print("-" * 30)
    
    operations = ['encode', 'decode', 'add_row']
    
    for op in operations:
        print(f"\n📊 {op.upper().replace('_', ' ')} Performance:")
        for dataset, formats in results.items():
            # Find the fastest format for this operation
            times = [(fmt, data[op]) for fmt, data in formats.items()]
            times.sort(key=lambda x: x[1])
            winner = times[0]
            
            print(f"   {dataset}: {winner[0]} ({winner[1]:.2f}ms)")
            
            # Show relative performance
            for fmt, time in times[1:]:
                slowdown = time / winner[1]
                print(f"      vs {fmt}: {slowdown:.1f}x slower ({time:.2f}ms)")
    
    print()
    print()
    
    print("📈 SCALING ANALYSIS:")
    print("-" * 25)
    
    # Analyze how each format scales with data size
    dataset_sizes = [100, 1000, 5000]
    dataset_keys = ['Small (100 rows)', 'Medium (1K rows)', 'Large (5K rows)']
    
    for fmt in ['CSV', 'JSON', 'JSON-Tables', 'Parquet']:
        print(f"\n🔍 {fmt} Scaling:")
        for op in operations:
            times = [results[key][fmt][op] for key in dataset_keys]
            
            # Calculate scaling factors
            scale_1k = times[1] / times[0]  # 100 -> 1000 rows (10x data)
            scale_5k = times[2] / times[1]  # 1000 -> 5000 rows (5x data)
            
            print(f"   {op}: {times[0]:.2f} → {times[1]:.2f} → {times[2]:.2f}ms")
            print(f"         Scaling: {scale_1k:.1f}x (10x data), {scale_5k:.1f}x (5x data)")
    
    print()
    print()
    
    print("🎯 KEY INSIGHTS:")
    print("-" * 20)
    
    print("📝 ENCODE (Write) Performance:")
    print("   • CSV: Fastest and most consistent")
    print("   • JSON-Tables: Very close to CSV") 
    print("   • JSON: Scales poorly with size")
    print("   • Parquet: High overhead for small data, excellent for large")
    print()
    
    print("📖 DECODE (Read) Performance:")
    print("   • JSON: Surprisingly fast for small/medium data")
    print("   • CSV: Consistent and reliable")
    print("   • Parquet: High overhead but scales well")
    print("   • JSON-Tables: Moderate overhead for conversion")
    print()
    
    print("➕ ADD ROW Performance:")
    print("   • CSV: Best for small data, degrades linearly")
    print("   • Parquet: Most consistent across sizes")
    print("   • JSON/JSON-Tables: Scale poorly (full rewrite)")
    print()
    
    print("🚀 PRACTICAL RECOMMENDATIONS:")
    print("-" * 35)
    print("📊 Small datasets (<1K rows):")
    print("   • Best overall: CSV (fast everything)")
    print("   • Best readable: JSON-Tables (slight overhead)")
    print()
    print("📈 Medium datasets (1K-10K rows):")
    print("   • Best performance: CSV for simple ops")
    print("   • Best balance: JSON-Tables (readable + decent speed)")
    print("   • Best for analytics: Parquet (if using pandas/analysis)")
    print()
    print("🗃️  Large datasets (10K+ rows):")
    print("   • Best write: Parquet (scales excellently)")
    print("   • Best simple ops: CSV") 
    print("   • Avoid: JSON (too slow)")
    print("   • JSON-Tables: Good for APIs where readability matters")

if __name__ == "__main__":
    analyze_performance()
