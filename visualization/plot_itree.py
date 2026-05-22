import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load d=2 and d=3 data
try:
    df_d2 = pd.read_csv('itree_d2_data.csv')
    print(f"Loaded d=2: {len(df_d2)} test cases")
except:
    print("WARNING: itree_d2_data.csv not found")
    df_d2 = pd.DataFrame()

try:
    df_d3 = pd.read_csv('itree_d3_data.csv')
    print(f"Loaded d=3: {len(df_d3)} test cases")
except:
    print("WARNING: itree_d3_data.csv not found")
    df_d3 = pd.DataFrame()

# Create bar chart for d=2
if not df_d2.empty:
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    labels = [f"n={row['n']}\n{row['num_funcs']} funcs" for _, row in df_d2.iterrows()]
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df_d2['vertex_ms'], width, label='Vertex-Based', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, df_d2['bucket_ms'], width, label='Centroid-Bucketing', color='#3498db', alpha=0.8)
    
    ax.set_xlabel('Test Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('CPU Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title('I-Tree Construction d=2: Recursive Index Propagation', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
  
    plt.tight_layout()
    plt.savefig('itree_d2_bar.png', dpi=300, bbox_inches='tight')
    print("Saved: itree_d2_bar.png")
    plt.close()

# Create bar chart for d=3
if not df_d3.empty:
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    labels = [f"n={row['n']}\n{row['num_funcs']} funcs" for _, row in df_d3.iterrows()]
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df_d3['vertex_ms'], width, label='Vertex-Based', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, df_d3['bucket_ms'], width, label='Centroid-Bucketing', color='#3498db', alpha=0.8)
    
    ax.set_xlabel('Test Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('CPU Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title('I-Tree Construction d=3: Recursive Index Propagation', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    
    plt.tight_layout()
    plt.savefig('itree_d3_bar.png', dpi=300, bbox_inches='tight')
    print("Saved: itree_d3_bar.png")
    plt.close()

# Combined comparison
if not df_d2.empty and not df_d3.empty:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # d=2 subplot
    labels_d2 = [f"n={row['n']}" for _, row in df_d2.iterrows()]
    x_d2 = np.arange(len(labels_d2))
    width = 0.35
    
    ax1.bar(x_d2 - width/2, df_d2['vertex_ms'], width, label='Vertex-Based', color='#e74c3c', alpha=0.8)
    ax1.bar(x_d2 + width/2, df_d2['bucket_ms'], width, label='Centroid-Bucketing', color='#3498db', alpha=0.8)
    
    ax1.set_xlabel('Polytope Size (n)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('CPU Time (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('d=2 (2D Polygons)', fontsize=13, fontweight='bold')
    ax1.set_xticks(x_d2)
    ax1.set_xticklabels(labels_d2)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # d=3 subplot
    labels_d3 = [f"n={row['n']}" for _, row in df_d3.iterrows()]
    x_d3 = np.arange(len(labels_d3))
    
    ax2.bar(x_d3 - width/2, df_d3['vertex_ms'], width, label='Vertex-Based', color='#e74c3c', alpha=0.8)
    ax2.bar(x_d3 + width/2, df_d3['bucket_ms'], width, label='Centroid-Bucketing', color='#3498db', alpha=0.8)
    
    ax2.set_xlabel('Polytope Size (n)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('CPU Time (ms)', fontsize=12, fontweight='bold')
    ax2.set_title('d=3 (3D Prisms)', fontsize=13, fontweight='bold')
    ax2.set_xticks(x_d3)
    ax2.set_xticklabels(labels_d3)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    fig.suptitle('I-Tree Construction with Recursive Index Propagation', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('itree_combined.png', dpi=300, bbox_inches='tight')
    print("Saved: itree_combined.png")
    plt.close()

print("\n" + "="*60)
print("SUMMARY: Generated I-tree plots")
print("="*60)
if not df_d2.empty:
    print("  - itree_d2_bar.png")
if not df_d3.empty:
    print("  - itree_d3_bar.png")
if not df_d2.empty and not df_d3.empty:
    print("  - itree_combined.png")
print("="*60)
