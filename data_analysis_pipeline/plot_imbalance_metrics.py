import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import sys

def plot_metrics(csv_file, output_dir='.'):
    """Plot TTFT and TPOT metrics against imbalance score."""

    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)
    
    # Validate required columns
    required_cols = ['imbalance_score', 'mean_ttft_ms', 'p99_ttft_ms', 
                     'mean_tpot_ms', 'p99_tpot_ms']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        sys.exit(1)
    
    # Remove rows with missing values in key columns
    df_clean = df[required_cols].dropna()
    if len(df_clean) == 0:
        print("Error: No valid data rows found")
        sys.exit(1)
    
    print(f"Plotting {len(df_clean)} data points...")
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('TTFT and TPOT Metrics vs Imbalance Score', fontsize=16, fontweight='bold')
    
    # Sort by imbalance score for better line plots
    df_sorted = df_clean.sort_values('imbalance_score')
    
    # Plot 1: Mean TTFT vs Imbalance Score
    ax1 = axes[0, 0]
    ax1.scatter(df_sorted['imbalance_score'], df_sorted['mean_ttft_ms'], 
                alpha=0.6, s=50, color='#1f77b4')
    ax1.plot(df_sorted['imbalance_score'], df_sorted['mean_ttft_ms'], 
             alpha=0.3, linewidth=1, color='#1f77b4')
    ax1.set_xlabel('Imbalance Score', fontsize=11)
    ax1.set_ylabel('Mean TTFT (ms)', fontsize=11)
    ax1.set_title('Mean Time to First Token', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: P99 TTFT vs Imbalance Score
    ax2 = axes[0, 1]
    ax2.scatter(df_sorted['imbalance_score'], df_sorted['p99_ttft_ms'], 
                alpha=0.6, s=50, color='#ff7f0e')
    ax2.plot(df_sorted['imbalance_score'], df_sorted['p99_ttft_ms'], 
             alpha=0.3, linewidth=1, color='#ff7f0e')
    ax2.set_xlabel('Imbalance Score', fontsize=11)
    ax2.set_ylabel('P99 TTFT (ms)', fontsize=11)
    ax2.set_title('P99 Time to First Token', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Mean TPOT vs Imbalance Score
    ax3 = axes[1, 0]
    ax3.scatter(df_sorted['imbalance_score'], df_sorted['mean_tpot_ms'], 
                alpha=0.6, s=50, color='#2ca02c')
    ax3.plot(df_sorted['imbalance_score'], df_sorted['mean_tpot_ms'], 
             alpha=0.3, linewidth=1, color='#2ca02c')
    ax3.set_xlabel('Imbalance Score', fontsize=11)
    ax3.set_ylabel('Mean TPOT (ms)', fontsize=11)
    ax3.set_title('Mean Time Per Output Token', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: P99 TPOT vs Imbalance Score
    ax4 = axes[1, 1]
    ax4.scatter(df_sorted['imbalance_score'], df_sorted['p99_tpot_ms'], 
                alpha=0.6, s=50, color='#d62728')
    ax4.plot(df_sorted['imbalance_score'], df_sorted['p99_tpot_ms'], 
             alpha=0.3, linewidth=1, color='#d62728')
    ax4.set_xlabel('Imbalance Score', fontsize=11)
    ax4.set_ylabel('P99 TPOT (ms)', fontsize=11)
    ax4.set_title('P99 Time Per Output Token', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = output_path / 'metrics_vs_imbalance.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_file}")
    
    # Create a combined plot (mean and p99 on same axes)
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle('TTFT and TPOT Metrics vs Imbalance Score (Combined)', 
                  fontsize=16, fontweight='bold')
    
    # TTFT combined plot
    ax1 = axes2[0]
    ax1.scatter(df_sorted['imbalance_score'], df_sorted['mean_ttft_ms'], 
                alpha=0.6, s=50, color='#1f77b4', label='Mean TTFT')
    ax1.plot(df_sorted['imbalance_score'], df_sorted['mean_ttft_ms'], 
             alpha=0.3, linewidth=1, color='#1f77b4')
    ax1.scatter(df_sorted['imbalance_score'], df_sorted['p99_ttft_ms'], 
                alpha=0.6, s=50, color='#ff7f0e', label='P99 TTFT')
    ax1.plot(df_sorted['imbalance_score'], df_sorted['p99_ttft_ms'], 
             alpha=0.3, linewidth=1, color='#ff7f0e')
    ax1.set_xlabel('Imbalance Score', fontsize=11)
    ax1.set_ylabel('TTFT (ms)', fontsize=11)
    ax1.set_title('Time to First Token', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # TPOT combined plot
    ax2 = axes2[1]
    ax2.scatter(df_sorted['imbalance_score'], df_sorted['mean_tpot_ms'], 
                alpha=0.6, s=50, color='#2ca02c', label='Mean TPOT')
    ax2.plot(df_sorted['imbalance_score'], df_sorted['mean_tpot_ms'], 
             alpha=0.3, linewidth=1, color='#2ca02c')
    ax2.scatter(df_sorted['imbalance_score'], df_sorted['p99_tpot_ms'], 
                alpha=0.6, s=50, color='#d62728', label='P99 TPOT')
    ax2.plot(df_sorted['imbalance_score'], df_sorted['p99_tpot_ms'], 
             alpha=0.3, linewidth=1, color='#d62728')
    ax2.set_xlabel('Imbalance Score', fontsize=11)
    ax2.set_ylabel('TPOT (ms)', fontsize=11)
    ax2.set_title('Time Per Output Token', fontsize=12, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file2 = output_path / 'metrics_vs_imbalance_combined.png'
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    print(f"Saved combined plot: {output_file2}")
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"\nImbalance Score:")
    print(f"  Range: {df_clean['imbalance_score'].min():.4f} - {df_clean['imbalance_score'].max():.4f}")
    print(f"  Mean: {df_clean['imbalance_score'].mean():.4f}")
    
    print(f"\nMean TTFT (ms):")
    print(f"  Range: {df_clean['mean_ttft_ms'].min():.4f} - {df_clean['mean_ttft_ms'].max():.4f}")
    print(f"  Mean: {df_clean['mean_ttft_ms'].mean():.4f}")
    
    print(f"\nP99 TTFT (ms):")
    print(f"  Range: {df_clean['p99_ttft_ms'].min():.4f} - {df_clean['p99_ttft_ms'].max():.4f}")
    print(f"  Mean: {df_clean['p99_ttft_ms'].mean():.4f}")
    
    print(f"\nMean TPOT (ms):")
    print(f"  Range: {df_clean['mean_tpot_ms'].min():.4f} - {df_clean['mean_tpot_ms'].max():.4f}")
    print(f"  Mean: {df_clean['mean_tpot_ms'].mean():.4f}")
    
    print(f"\nP99 TPOT (ms):")
    print(f"  Range: {df_clean['p99_tpot_ms'].min():.4f} - {df_clean['p99_tpot_ms'].max():.4f}")
    print(f"  Mean: {df_clean['p99_tpot_ms'].mean():.4f}")
    
    print(f"\n=== Plots saved to {output_path} ===")


def main():
    parser = argparse.ArgumentParser(
        description='Plot TTFT and TPOT metrics against imbalance score',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_metrics.py metrics.csv
  python plot_metrics.py metrics.csv --output ./plots
        """
    )
    parser.add_argument('csv_file', help='Path to CSV file containing metrics')
    parser.add_argument('--output', '-o', default='.', 
                        help='Output directory for plots (default: current directory)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.csv_file).exists():
        print(f"Error: CSV file not found: {args.csv_file}")
        sys.exit(1)
    
    plot_metrics(args.csv_file, args.output)


if __name__ == '__main__':
    main()


