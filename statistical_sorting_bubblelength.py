import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIGURATION ---
GT_INPUT_CSV = 'Bubble_Spiral_Sorted_1and2.csv'          # Manual Ground Truth
AI_INPUT_CSV = 'BB_AI_Bubble_Analysis_Sorted_RAW.csv'       # Model Outputs
OUTPUT_STATS_CSV = 'Channel_Bubble_Comparison_Statistics_RAW.csv'
OUTPUT_PLOT_IMAGE = 'Channel_Bubble_SideBySide_Distributions_RAW.png'

def analyze_and_compare_bubble_lengths():
    # 1. Verification Block & Data Load
    gt_exists = os.path.exists(GT_INPUT_CSV)
    ai_exists = os.path.exists(AI_INPUT_CSV)
    
    if not gt_exists and not ai_exists:
        print("Error: Neither Manual GT nor AI sorted CSV files were found.")
        return

    dfs_to_combine = []
    
    if gt_exists:
        df_gt = pd.read_csv(GT_INPUT_CSV)
        df_gt['Data_Source'] = 'Manual GT'
        dfs_to_combine.append(df_gt)
        print(f"Loaded {len(df_gt)} Manual Ground Truth records.")
        
    if ai_exists:
        df_ai = pd.read_csv(AI_INPUT_CSV)
        df_ai['Data_Source'] = 'AI Predicted'
        dfs_to_combine.append(df_ai)
        print(f"Loaded {len(df_ai)} AI Prediction records.")

    # Merge everything into a master plotting frame
    df_master = pd.concat(dfs_to_combine, ignore_index=True)
    
    required_cols = ['Channel_ID', 'Horizontal_Length_Microns', 'Data_Source']
    if not all(col in df_master.columns for col in required_cols):
        print(f"Error: Datasets are missing required evaluation fields: {required_cols}")
        return

    # 2. Compute Segmented Summary Comparison Statistics
    stats_summary = df_master.groupby(['Channel_ID', 'Data_Source'])['Horizontal_Length_Microns'].describe().rename(columns={
        'count': 'Total_Bubbles_Count', 'mean': 'Mean_Length_um', 'std': 'Std_Dev_um',
        'min': 'Min_Length_um', '25%': '25th_Percentile_um', '50%': 'Median_Length_um',
        '75%': '75th_Percentile_um', 'max': 'Max_Length_um'
    }).round(2)
    stats_summary.to_csv(OUTPUT_STATS_CSV)
    print(f" -> Comparative statistical summary table saved to: {OUTPUT_STATS_CSV}")

    # 3. GENERATE DUAL COLOR BOXPLOT CONTEXT
    plt.figure(figsize=(11, 10)) 
    sns.set_theme(style="whitegrid")
    
    # Keep channel arrays sorting correctly from top to bottom
    df_sorted = df_master.sort_values(by=['Channel_ID', 'Data_Source'], ascending=[True, False])
    channel_order = sorted(df_master['Channel_ID'].unique())

    # Build side-by-side grouped distribution bars using hue
    ax = sns.boxplot(
        y='Channel_ID', 
        x='Horizontal_Length_Microns', 
        hue='Data_Source',
        data=df_sorted,
        order=channel_order,
        palette={'Manual GT': '#3174A1', 'AI Predicted': '#D37960'},  # Matches your custom color profile exactly
        width=0.6,
        dodge=True,        # Forces separate bars side-by-side per channel
        fliersize=3,
        linewidth=1.2,
        orient='h'
    )
    
    # Overlap jittered strip plots to show underlying data density profile
    sns.stripplot(
        y='Channel_ID', 
        x='Horizontal_Length_Microns', 
        hue='Data_Source',
        data=df_sorted,
        order=channel_order,
        palette={'Manual GT': '#1A435E', 'AI Predicted': '#853F2E'},
        alpha=0.3, 
        size=2.5,
        jitter=0.15,
        dodge=True,        # Aligns strip dots perfectly to their matching box container
        legend=False,      # Prevents duplicate entries from cluttering the legend card
        orient='h'
    )

    # Clean styling layout matching your graphic
    plt.title("Channel-by-Channel Profiling: Manual vs. AI Performance", 
              fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Horizontal Bubble Length (µm)", fontsize=11, fontweight='bold', labelpad=10)
    plt.ylabel("Channel ID", fontsize=11, fontweight='bold', labelpad=10)
    
    # Position the legend top right inside layout boundary box
    plt.legend(title='Data Source', loc='upper right', frameon=True, facecolor='white', edgecolor='lightgrey')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_IMAGE, dpi=300, bbox_inches='tight')
    print(f" -> High-resolution comparison figure saved to: {OUTPUT_PLOT_IMAGE}")

    print("\nDisplaying Verification Figure window...")
    plt.show()

if __name__ == "__main__":
    analyze_and_compare_bubble_lengths()