import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
GT_SORTED_CSV = 'Bubble_Spiral_Sorted_1and2.csv'
AI_SORTED_CSV = 'BB_AI_Bubble_Analysis_Sorted_ALL.csv'
OUTPUT_REPORT_CSV = 'AI_vs_Manual_Microfluidic_Benchmark.csv'

def generate_error_benchmark_report():
    if not os.path.exists(GT_SORTED_CSV) or not os.path.exists(AI_SORTED_CSV):
        print("Error: Missing sorted files! Ensure both sorting scripts have run.")
        return

    # Load datasets
    df_gt = pd.read_csv(GT_SORTED_CSV)
    df_ai = pd.read_csv(AI_SORTED_CSV)

    # 1. Calculate per-channel summary statistics (Mean and Standard Deviation)
    gt_stats = df_gt.groupby('Channel_ID')['Horizontal_Length_Microns'].agg(
        Manual_Count='count', 
        Manual_Mean_um='mean',
        Manual_Std_um='std'
    )
    
    ai_stats = df_ai.groupby('Channel_ID')['Horizontal_Length_Microns'].agg(
        AI_Count='count', 
        AI_Mean_um='mean',
        AI_Std_um='std'
    )

    # 2. Outer join to make sure no channel is left behind
    report_df = pd.merge(gt_stats, ai_stats, on='Channel_ID', how='outer').fillna(0)
    report_df = report_df.sort_index()

    # 3. Calculate Physical Residual Errors per channel
    # Residual = AI Value - Ground Truth Value (Positive = AI overshot, Negative = AI undershot)
    report_df['Count_Bias'] = report_df['AI_Count'] - report_df['Manual_Count']
    report_df['Size_Bias_um'] = report_df['AI_Mean_um'] - report_df['Manual_Mean_um']
    report_df['Size_MAE_um'] = np.abs(report_df['Size_Bias_um'])

    # Clean display conversions
    report_df['Manual_Count'] = report_df['Manual_Count'].astype(int)
    report_df['AI_Count'] = report_df['AI_Count'].astype(int)
    report_df['Manual_Mean_um'] = report_df['Manual_Mean_um'].round(1)
    report_df['AI_Mean_um'] = report_df['AI_Mean_um'].round(1)
    report_df['Manual_Std_um'] = report_df['Manual_Std_um'].round(1)
    report_df['AI_Std_um'] = report_df['AI_Std_um'].round(1)
    report_df['Size_Bias_um'] = report_df['Size_Bias_um'].round(1)
    report_df['Size_MAE_um'] = report_df['Size_MAE_um'].round(1)

    # Save summary report to CSV
    report_df.to_csv(OUTPUT_REPORT_CSV)

    # 4. COMPUTE GLOBAL STATISTICAL METRICS Across Active Channels
    active_channels = report_df[report_df['Manual_Mean_um'] > 0]

    # Sizing Metrics
    size_residuals = active_channels['AI_Mean_um'] - active_channels['Manual_Mean_um']
    sizing_mbe = size_residuals.mean()               # Mean Bias Error (Systematic Shift)
    sizing_mae = np.abs(size_residuals).mean()        # Mean Absolute Error
    sizing_rmse = np.sqrt((size_residuals**2).mean()) # Root Mean Squared Error
    sizing_std_err = size_residuals.std()             # Precision / Variability of error

    # Count Metrics
    count_residuals = report_df['AI_Count'] - report_df['Manual_Count']
    count_mbe = count_residuals.mean()
    count_mae = np.abs(count_residuals).mean()
    count_rmse = np.sqrt((count_residuals**2).mean())

    total_manual = report_df['Manual_Count'].sum()
    total_ai = report_df['AI_Count'].sum()

    # 5. PRINT THE PUBLICATION-READY METROLOGY REPORT
    print("="*102)
    print("                      AI MICROFLUIDIC BENCHMARK REPORT (METROLOGY METRICS)")
    print("="*102)
    print(f"{'Channel_ID':<12} {'Manual_Count':<13} {'AI_Count':<10} {'Count_Bias':<12} {'Manual_Mean':<15} {'AI_Mean':<15} {'Size_Bias':<12} {'Size_MAE'}")
    print(f"{'':<12} {'':<13} {'':<10} {'(bubbles)':<12} {'(Mean ± σ µm)':<15} {'(Mean ± σ µm)':<15} {'(µm)':<12} {'(µm)'}")
    print("-"*102)

    for ch_id, row in report_df.iterrows():
        man_str = f"{row['Manual_Mean_um']:.1f} ± {row['Manual_Std_um']:.1f}"
        ai_str = f"{row['AI_Mean_um']:.1f} ± {row['AI_Std_um']:.1f}"
        bias_str = f"{row['Count_Bias']:+d}"
        size_bias_str = f"{row['Size_Bias_um']:+.1f}"

        print(f"{ch_id:<12} "
              f"{int(row['Manual_Count']):<13} "
              f"{int(row['AI_Count']):<10} "
              f"{bias_str:<12} "
              f"{man_str:<15} "
              f"{ai_str:<15} "
              f"{size_bias_str:<12} "
              f"{row['Size_MAE_um']:<10.1f}")

    print("="*102)
    print("OVERALL SYSTEM STATISTICAL EVALUATION:")
    print("-" * 50)
    print(" COUNT PERFORMANCE:")
    print(f"  -> Total Manual / AI Count       : {total_manual} / {total_ai} bubbles")
    print(f"  -> Count Mean Bias Error (MBE)   : {count_mbe:+.2f} bubbles/channel")
    print(f"  -> Count Mean Absolute Error     : {count_mae:.2f} bubbles/channel")
    print(f"  -> Count Root Mean Square Error  : {count_rmse:.2f} bubbles")
    print("-" * 50)
    print(" SIZING METRICS (Bubble Length Dynamics):")
    print(f"  -> Mean Bias Error (MBE)         : {sizing_mbe:+.2f} µm (Directional System Drift)")
    print(f"  -> Mean Absolute Error (MAE)     : {sizing_mae:.2f} µm")
    print(f"  -> Root Mean Squared Error (RMSE): {sizing_rmse:.2f} µm (Penalty Metric for Outliers)")
    print(f"  -> Standard Deviation of Error   : {sizing_std_err:.2f} µm (Measurement Precision)")
    print("="*102 + "\n")

if __name__ == "__main__":
    generate_error_benchmark_report()