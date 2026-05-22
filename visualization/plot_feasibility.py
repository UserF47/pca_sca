import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('feasibility_data.csv')

print(f"Loaded {len(df)} data points")
print(f"Dimensions: {sorted(df['dimension'].unique())}")

# Create individual plots for each dimension
for d in sorted(df['dimension'].unique()):
    df_d = df[df['dimension'] == d].sort_values('n')
    
    plt.figure(figsize=(10, 6))
    plt.plot(df_d['n'], df_d['vertex_us'], 'o-', linewidth=2.5, markersize=8, 
             label='Vertex-Based', color='#e74c3c')
    plt.plot(df_d['n'], df_d['bucket_us'], 's-', linewidth=2.5, markersize=8, 
             label='Centroid-Bucketing', color='#3498db')
    
    plt.xlabel('Number of Vertices (n)', fontsize=12, fontweight='bold')
    plt.ylabel('CPU Time (μs)', fontsize=12, fontweight='bold')
    plt.title(f'Feasibility Checking Performance: d={d}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'feasibility_d{d}.png', dpi=300, bbox_inches='tight')
    print(f"Saved: feasibility_d{d}.png")
    plt.close()

# Combined plot with all dimensions
plt.figure(figsize=(14, 8))

colors = plt.cm.tab10(np.linspace(0, 1, 10))

for idx, d in enumerate(sorted(df['dimension'].unique())):
    df_d = df[df['dimension'] == d].sort_values('n')
    
    plt.plot(df_d['n'], df_d['vertex_us'], 'o-', linewidth=2, markersize=6, 
             color=colors[idx], label=f'd={d} Vertex', alpha=0.8)
    plt.plot(df_d['n'], df_d['bucket_us'], 's--', linewidth=2, markersize=6, 
             color=colors[idx], label=f'd={d} Bucket', alpha=0.8)

plt.xlabel('Number of Vertices (n)', fontsize=12, fontweight='bold')
plt.ylabel('CPU Time (μs)', fontsize=12, fontweight='bold')
plt.title('Feasibility Checking: All Dimensions (d=2 to d=10)', fontsize=14, fontweight='bold')
plt.legend(fontsize=8, ncol=2, loc='upper left')
plt.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('feasibility_all_dimensions.png', dpi=300, bbox_inches='tight')
print("Saved: feasibility_all_dimensions.png")
plt.close()

# Cross-dimension comparison (median n for each dimension)
plt.figure(figsize=(12, 7))

dims = []
vertex_times = []
bucket_times = []
n_values = []

for d in sorted(df['dimension'].unique()):
    df_d = df[df['dimension'] == d].sort_values('n')
    mid_idx = len(df_d) // 2
    
    dims.append(d)
    vertex_times.append(df_d.iloc[mid_idx]['vertex_us'])
    bucket_times.append(df_d.iloc[mid_idx]['bucket_us'])
    n_values.append(df_d.iloc[mid_idx]['n'])

plt.plot(dims, vertex_times, 'o-', linewidth=2.5, markersize=8, 
         label='Vertex-Based', color='#e74c3c')
plt.plot(dims, bucket_times, 's-', linewidth=2.5, markersize=8, 
         label='Centroid-Bucketing', color='#3498db')

plt.xlabel('Dimension (d)', fontsize=12, fontweight='bold')
plt.ylabel('CPU Time (μs)', fontsize=12, fontweight='bold')
plt.title('Feasibility Checking Across Dimensions (d=2 to d=10)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(dims)

# Add n values as annotations
for i, (d, n) in enumerate(zip(dims, n_values)):
    max_time = max(vertex_times[i], bucket_times[i])
    plt.annotate(f'n={n}', xy=(d, max_time), xytext=(0, 10), 
                textcoords='offset points', ha='center', fontsize=8, alpha=0.7)

plt.tight_layout()
plt.savefig('feasibility_d_sweep.png', dpi=300, bbox_inches='tight')
print("Saved: feasibility_d_sweep.png")
plt.close()

# Speedup bar chart
plt.figure(figsize=(12, 7))

speedups = [vertex_times[i] / bucket_times[i] if bucket_times[i] > 0 else 0 
            for i in range(len(dims))]

colors_bar = ['#e74c3c' if s < 1.0 else '#27ae60' for s in speedups]
bars = plt.bar(dims, speedups, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1.5)

plt.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, label='No Speedup (1x)')

plt.xlabel('Dimension (d)', fontsize=12, fontweight='bold')
plt.ylabel('Speedup (Vertex / Bucket)', fontsize=12, fontweight='bold')
plt.title('Speedup of Centroid-Bucketing over Vertex-Based', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3, linestyle='--', axis='y')
plt.xticks(dims)

# Add speedup values on bars
for i, (d, s) in enumerate(zip(dims, speedups)):
    if s > 0:
        plt.text(d, s + 0.1, f'{s:.2f}x', ha='center', va='bottom', 
                fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('feasibility_speedup.png', dpi=300, bbox_inches='tight')
print("Saved: feasibility_speedup.png")
plt.close()

print("\n" + "="*60)
print("SUMMARY: Generated plots:")
print("="*60)
for d in sorted(df['dimension'].unique()):
    print(f"  - feasibility_d{d}.png")
print("\nCombined plots:")
print("  - feasibility_all_dimensions.png")
print("  - feasibility_d_sweep.png")
print("  - feasibility_speedup.png")
print("="*60)
