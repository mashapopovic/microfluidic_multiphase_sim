import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIGURATION ---
INPUT_CSV = 'Bubble_Spiral_Sorted.csv'  
OUTPUT_STATS_CSV = 'Channel_Bubble_Length_Statistics.csv'
OUTPUT_PLOT_IMAGE = 'Channel_Bubble_Length_Distributions.png'  # Clear name for the saved image file

def analyze_and_save_bubble_lengths():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Could not find '{INPUT_CSV}'. Please ensure the sorting script has run successfully.")
        return

    # 1. Load the sorted data
    df = pd.read_csv(INPUT_CSV)
    
    required_cols = ['Channel_ID', 'Horizontal_Length_Microns']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV must contain the following columns: {required_cols}")
        return

    print(f"Loaded {len(df)} total bubble measurements. Processing statistics and plotting...")

    # 2. Compute Summary Statistics per Channel and Save to CSV
    stats_summary = df.groupby('Channel_ID')['Horizontal_Length_Microns'].describe().rename(columns={
        'count': 'Total_Bubbles_Count', 'mean': 'Mean_Length_um', 'std': 'Std_Dev_um',
        'min': 'Min_Length_um', '25%': '25th_Percentile_um', '50%': 'Median_Length_um',
        '75%': '75th_Percentile_um', 'max': 'Max_Length_um'
    }).round(2)
    stats_summary.to_csv(OUTPUT_STATS_CSV)
    print(f" -> Statistics table saved to: {OUTPUT_STATS_CSV}")

    # 3. GENERATE HORIZONTAL PLOT
    plt.figure(figsize=(10, 12)) 
    sns.set_theme(style="whitegrid")
    
    df_sorted_channels = df.sort_values('Channel_ID', ascending=True)
    channel_order = sorted(df['Channel_ID'].unique())

    # Create Horizontal Boxplot
    ax = sns.boxplot(
        y='Channel_ID', 
        x='Horizontal_Length_Microns', 
        data=df_sorted_channels,
        order=channel_order,
        palette='viridis',
        width=0.5,
        fliersize=4,
        orient='h'
    )
    
    # Overlay individual data points
    sns.stripplot(
        y='Channel_ID', 
        x='Horizontal_Length_Microns', 
        data=df_sorted_channels,
        order=channel_order,
        color='black', 
        alpha=0.3, 
        size=3,
        jitter=0.15,
        orient='h'
    )

    # Label adjustments matching the flipped orientation
    plt.title("Bubble Length Distributions Across the Channel Array\n(Arranged Top-to-Bottom by Physical Layout)", 
              fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Horizontal Bubble Length (µm)", fontsize=11, fontweight='bold', labelpad=10)
    plt.ylabel("Channel Position (Top to Bottom)", fontsize=11, fontweight='bold', labelpad=10)
    
    plt.tight_layout()
    
    # --- NEW: SAVE THE PLOT FIGURE TO IMAGE FILE ---
    # dpi=300 guarantees high resolution; bbox_inches='tight' prevents label clipping
    plt.savefig(OUTPUT_PLOT_IMAGE, dpi=300, bbox_inches='tight')
    print(f" -> High-resolution plot figure saved to: {OUTPUT_PLOT_IMAGE}")

    # 4. Display the visual plot window
    print("\nLaunching Flipped Verification Plot window...")
    plt.show()

if __name__ == "__main__":
    analyze_and_save_bubble_lengths()