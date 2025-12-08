import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

def plot_batch_size_analysis(csv_path, output_dir=None):
    """
    Create comprehensive plots from batch size experiment results.
    
    Args:
        csv_path: Path to CSV file with batch size experiment results
        output_dir: Directory to save plots (optional, defaults to same dir as CSV)
    """
    # Read data
    df = pd.read_csv(csv_path)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # Create output directory
    if output_dir is None:
        output_dir = Path(csv_path).parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get unique kernel types
    kernel_types = df['kernel_type'].unique()
    colors = plt.cm.Set2(range(len(kernel_types)))
    color_map = dict(zip(kernel_types, colors))
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Batch Size vs Performance Metrics', fontsize=16, fontweight='bold')
    
    # 1. TTFT Metrics
    ax = axes[0, 0]
    for kernel in kernel_types:
        data = df[df['kernel_type'] == kernel]
        ax.plot(data['batch_size'], data['mean_ttft_ms'], 
               marker='o', label=f'{kernel} (mean)', 
               color=color_map[kernel], linewidth=2)
        ax.fill_between(data['batch_size'], 
                        data['median_ttft_ms'], 
                        data['p99_ttft_ms'], 
                        alpha=0.2, color=color_map[kernel])
    ax.set_xlabel('Batch Size', fontsize=11)
    ax.set_ylabel('Time to First Token (ms)', fontsize=11)
    ax.set_title('TTFT vs Batch Size', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. TPOT Metrics
    ax = axes[0, 1]
    for kernel in kernel_types:
        data = df[df['kernel_type'] == kernel]
        ax.plot(data['batch_size'], data['mean_tpot_ms'], 
               marker='s', label=f'{kernel} (mean)', 
               color=color_map[kernel], linewidth=2)
        ax.fill_between(data['batch_size'], 
                        data['median_tpot_ms'], 
                        data['p99_tpot_ms'], 
                        alpha=0.2, color=color_map[kernel])
    ax.set_xlabel('Batch Size', fontsize=11)
    ax.set_ylabel('Time per Output Token (ms)', fontsize=11)
    ax.set_title('TPOT vs Batch Size', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Request Throughput
    ax = axes[0, 2]
    for kernel in kernel_types:
        data = df[df['kernel_type'] == kernel]
        ax.plot(data['batch_size'], data['request_throughput'], 
               marker='^', label=kernel, 
               color=color_map[kernel], linewidth=2)
    ax.set_xlabel('Batch Size', fontsize=11)
    ax.set_ylabel('Request Throughput (req/s)', fontsize=11)
    ax.set_title('Request Throughput vs Batch Size', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Output Throughput
    ax = axes[1, 0]
    for kernel in kernel_types:
        data = df[df['kernel_type'] == kernel]
        ax.plot(data['batch_size'], data['output_throughput'], 
               marker='D', label=kernel, 
               color=color_map[kernel], linewidth=2)
    ax.set_xlabel('Batch Size', fontsize=11)
    ax.set_ylabel('Output Throughput (tokens/s)', fontsize=11)
    ax.set_title('Output Throughput vs Batch Size', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Imbalance Score
    ax = axes[1, 1]
    for kernel in kernel_types:
        data = df[df['kernel_type'] == kernel]
        ax.plot(data['batch_size'], data['imbalance_score'], 
               marker='*', markersize=10, label=kernel, 
               color=color_map[kernel], linewidth=2)
    ax.set_xlabel('Batch Size', fontsize=11)
    ax.set_ylabel('Load Imbalance Score (CV)', fontsize=11)
    ax.set_title('Load Imbalance vs Batch Size', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Median TTFT vs Median TPOT (scatter)
    ax = axes[1, 2]
    for kernel in kernel_types:
        data = df[df['kernel_type'] == kernel]
        ax.scatter(data['median_ttft_ms'], data['median_tpot_ms'], 
                  s=data['batch_size']*20, alpha=0.6, 
                  label=kernel, color=color_map[kernel])
        # Add batch size labels
        for _, row in data.iterrows():
            ax.annotate(f"{int(row['batch_size'])}", 
                       (row['median_ttft_ms'], row['median_tpot_ms']),
                       fontsize=8, alpha=0.7)
    ax.set_xlabel('Median TTFT (ms)', fontsize=11)
    ax.set_ylabel('Median TPOT (ms)', fontsize=11)
    ax.set_title('TTFT vs TPOT (bubble size = batch size)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the comprehensive plot
    output_file = output_dir / 'batch_size_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved comprehensive plot: {output_file}")
    
    # Create individual high-quality plots
    
    # Individual plot 1: TTFT comparison
    plt.figure(figsize=(10, 6))
    for kernel in kernel_types:
        data = df[df['kernel_type'] == kernel]
        plt.plot(data['batch_size'], data['mean_ttft_ms'], 
                marker='o', label=kernel, color=color_map[kernel], 
                linewidth=2.5, markersize=8)
    plt.xlabel('Batch Size', fontsize=13)
    plt.ylabel('Mean Time to First Token (ms)', fontsize=13)
    plt.title('TTFT vs Batch Size', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'ttft_vs_batch_size.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'ttft_vs_batch_size.png'}")
    plt.close()
    
    # Individual plot 2: TPOT comparison
    plt.figure(figsize=(10, 6))
    for kernel in kernel_types:
        data = df[df['kernel_type'] == kernel]
        plt.plot(data['batch_size'], data['mean_tpot_ms'], 
                marker='s', label=kernel, color=color_map[kernel], 
                linewidth=2.5, markersize=8)
    plt.xlabel('Batch Size', fontsize=13)
    plt.ylabel('Mean Time per Output Token (ms)', fontsize=13)
    plt.title('TPOT vs Batch Size', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'tpot_vs_batch_size.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'tpot_vs_batch_size.png'}")
    plt.close()
    
    # Individual plot 3: Throughput comparison
    plt.figure(figsize=(10, 6))
    for kernel in kernel_types:
        data = df[df['kernel_type'] == kernel]
        plt.plot(data['batch_size'], data['output_throughput'], 
                marker='^', label=kernel, color=color_map[kernel], 
                linewidth=2.5, markersize=8)
    plt.xlabel('Batch Size', fontsize=13)
    plt.ylabel('Output Throughput (tokens/s)', fontsize=13)
    plt.title('Throughput vs Batch Size', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'throughput_vs_batch_size.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'throughput_vs_batch_size.png'}")
    plt.close()
    
    # Individual plot 4: Imbalance score
    plt.figure(figsize=(10, 6))
    for kernel in kernel_types:
        data = df[df['kernel_type'] == kernel]
        plt.plot(data['batch_size'], data['imbalance_score'], 
                marker='*', markersize=12, label=kernel, 
                color=color_map[kernel], linewidth=2.5)
    plt.xlabel('Batch Size', fontsize=13)
    plt.ylabel('Load Imbalance Score (CV)', fontsize=13)
    plt.title('Load Imbalance vs Batch Size (Lower is Better)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'imbalance_vs_batch_size.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'imbalance_vs_batch_size.png'}")
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    for kernel in kernel_types:
        kernel_data = df[df['kernel_type'] == kernel]
        print(f"\n{kernel.upper()} Kernel:")
        print(f"  Batch sizes tested: {sorted(kernel_data['batch_size'].tolist())}")
        print(f"  Mean TTFT range: {kernel_data['mean_ttft_ms'].min():.2f} - {kernel_data['mean_ttft_ms'].max():.2f} ms")
        print(f"  Mean TPOT range: {kernel_data['mean_tpot_ms'].min():.2f} - {kernel_data['mean_tpot_ms'].max():.2f} ms")
        print(f"  Throughput range: {kernel_data['output_throughput'].min():.2f} - {kernel_data['output_throughput'].max():.2f} tokens/s")
        print(f"  Imbalance range: {kernel_data['imbalance_score'].min():.4f} - {kernel_data['imbalance_score'].max():.4f}")
    
    print("\n" + "="*60)
    print(f"All plots saved to: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_batch_metrics.py <csv_file> [output_directory]")
        print("Example: python plot_batch_metrics.py batch_size_metrics.csv ./plots")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    plot_batch_size_analysis(csv_path, output_dir)