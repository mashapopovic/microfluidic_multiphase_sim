import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# --- CONFIG ---
REFERENCE_IMAGE = '/home/masas/Frames/S1_H4_L1_p14_C04_T60_Seq001_Img000051.bmp'

# Dataset 1: Manual Ground Truth
GT_INPUT_CSV = 'Bubble_Spatial_6Frame_Pairs_1and2.csv'
GT_OUTPUT_CSV = 'Bubble_Spiral_Sorted_1and2.csv'

# Dataset 2: AI Predictions
AI_INPUT_CSV = 'BB_AI_Bubble_Analysis_Results_adjustedwtestset.csv'
AI_OUTPUT_CSV = 'BB_AI_Bubble_Analysis_Sorted.csv'

class ChannelTrackDefiner:
    def __init__(self, image_path):
        self.img = cv2.imread(image_path)
        if self.img is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")
        self.img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        
        self.channels = []          # List of lists containing (x,y) tracks
        self.current_channel = []   # Points for the active track being drawn
        
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.imshow(self.img_rgb)
        
        # --- FORCE FULLSCREEN / MAXIMIZED WINDOW ---
        fig_manager = plt.get_current_fig_manager()
        try:
            fig_manager.window.state('zoomed')
        except AttributeError:
            try:
                fig_manager.window.showMaximized()
            except AttributeError:
                try:
                    fig_manager.frame.Maximize(True)
                except Exception:
                    pass
        
        print("--- CHANNEL DRAWING INSTRUCTIONS ---")
        print("1. Left-click points along the path/centerline of Channel 1 (follow the curve).")
        print("2. When finished with Channel 1, press the 'n' key on your keyboard to start Channel 2.")
        print("3. Repeat for all channels.")
        print("4. Close the window when you are completely finished drawing all tracks.\n")
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.update_title()
        plt.show()

    def update_title(self):
        ch_num = len(self.channels) + 1
        self.ax.set_title(f"Drawing Track for Channel {ch_num} | Click along curve | Press 'n' for Next Channel", 
                          fontsize=12, fontweight='bold')
        self.fig.canvas.draw()

    def on_click(self, event):
        if event.xdata is not None and event.ydata is not None and event.button == 1:
            x, y = event.xdata, event.ydata
            self.current_channel.append((x, y))
            
            self.ax.plot(x, y, 'ro', markersize=4)
            if len(self.current_channel) > 1:
                p1 = self.current_channel[-2]
                self.ax.plot([p1[0], x], [p1[1], y], 'r-', linewidth=1.5)
            self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == 'n':
            if self.current_channel:
                self.channels.append(list(self.current_channel))
                print(f"Stored Channel {len(self.channels)} with {len(self.current_channel)} guide points.")
                
                xs, ys = zip(*self.current_channel)
                self.ax.plot(xs, ys, 'g-', linewidth=2)
                self.ax.text(xs[0], ys[0], f"Ch {len(self.channels)}", color='lime', 
                             fontsize=12, fontweight='bold', bbox=dict(facecolor='black', alpha=0.7))
                
                self.current_channel.clear()
                self.update_title()
            else:
                print("Click some points before trying to move to the next channel!")

def point_to_segment_distance(p, a, b):
    p = np.array(p)
    a = np.array(a)
    b = np.array(b)
    ab = b - a
    ap = p - a
    ab_len_sq = np.sum(ab**2)
    if ab_len_sq == 0:
        return np.linalg.norm(ap)
    t = np.dot(ap, ab) / ab_len_sq
    t = max(0.0, min(1.0, t))
    closest_point = a + t * ab
    return np.linalg.norm(p - closest_point)

def point_to_path_distance(point, path):
    min_dist = float('inf')
    for i in range(len(path) - 1):
        dist = point_to_segment_distance(point, path[i], path[i+1])
        if dist < min_dist:
            min_dist = dist
    return min_dist

def process_and_sort_csv(input_path, output_path, drawn_channels):
    """Reads a bubble CSV file, maps coordinates to closest channels, and saves a sorted copy."""
    if not os.path.exists(input_path):
        print(f"⚠️ Skipping: File '{input_path}' not found.")
        return None

    print(f"Processing classifications for '{input_path}'...")
    df = pd.read_csv(input_path)
    assigned_channels = []
    
    for idx, row in df.iterrows():
        bx = row['Center_X_px']
        by = row['Center_Y_px']
        bubble_point = (bx, by)
        
        # Calculate minimum geometric distance to all drawn channels
        distances = [point_to_path_distance(bubble_point, path) for path in drawn_channels]
        closest_channel_idx = np.argmin(distances)
        assigned_channels.append(f"Channel_{closest_channel_idx + 1:02d}")

    df['Channel_ID'] = assigned_channels
    
    # Sort logically by channel space, frame context, and horizontal flow direction
    df_sorted = df.sort_values(by=['Channel_ID', 'Frame_Name', 'Center_X_px'])
    df_sorted.to_csv(output_path, index=False)
    print(f" -> Successfully saved sorted data to: {output_path}")
    return df_sorted

def main():
    # Print exactly what the script is looking for to debug instantly
    print(f"Checking for GT File: '{GT_INPUT_CSV}' -> Found: {os.path.exists(GT_INPUT_CSV)}")
    print(f"Checking for AI File: '{AI_INPUT_CSV}' -> Found: {os.path.exists(AI_INPUT_CSV)}")

    if not os.path.exists(GT_INPUT_CSV) and not os.path.exists(AI_INPUT_CSV):
        print("Error: Neither Ground Truth nor AI output files were found. Nothing to sort!")
        return

    # Step 1: User draws coordinates on reference background
    try:
        drawer = ChannelTrackDefiner(REFERENCE_IMAGE)
    except Exception as e:
        print(e)
        return

    if drawer.current_channel:
        drawer.channels.append(list(drawer.current_channel))

    if not drawer.channels:
        print("No channels were drawn. Sorting aborted.")
        return

    print(f"\nTracks locked. Processing {len(drawer.channels)} microfluidic channels...")

    # Step 2: Run classification pipeline on BOTH files
    df_gt_sorted = process_and_sort_csv(GT_INPUT_CSV, GT_OUTPUT_CSV, drawer.channels)
    print("--- Ground Truth Processing Finished ---")
    
    df_ai_sorted = process_and_sort_csv(AI_INPUT_CSV, AI_OUTPUT_CSV, drawer.channels)
    print("--- AI Processing Finished ---")
    
    # We will create side-by-side or layered plots depending on what files were actually processed
    fig_ver, ax_ver = plt.subplots(figsize=(12, 8))
    ax_ver.imshow(drawer.img_rgb)
    
    # Maximize window
    fig_ver_manager = plt.get_current_fig_manager()
    try:
        fig_ver_manager.window.state('zoomed')
    except AttributeError:
        try:
            fig_ver_manager.window.showMaximized()
        except AttributeError:
            try:
                fig_ver_manager.frame.Maximize(True)
            except Exception:
                pass

    cmap = plt.colormaps.get_cmap('tab10')

    # Draw original channel paths for reference
    for ch_idx in range(len(drawer.channels)):
        color = cmap(ch_idx % 10)
        tx, ty = zip(*drawer.channels[ch_idx])
        ax_ver.plot(tx, ty, color=color, linestyle='--', alpha=0.5, linewidth=1.5)

    # Plot Sorted Ground Truth points if they exist (Solid circles with black borders)
    if df_gt_sorted is not None:
        unique_channels = sorted(df_gt_sorted['Channel_ID'].unique())
        for ch_idx, ch_id in enumerate(unique_channels):
            ch_data = df_gt_sorted[df_gt_sorted['Channel_ID'] == ch_id]
            color = cmap(ch_idx % 10)
            ax_ver.scatter(ch_data['Center_X_px'], ch_data['Center_Y_px'], 
                           label=f"GT {ch_id}", color=color, s=40, edgecolor='black', marker='o', zorder=5)

    # Plot Sorted AI points if they exist (Translucent diamond stars)
    if df_ai_sorted is not None:
        unique_channels_ai = sorted(df_ai_sorted['Channel_ID'].unique())
        for ch_idx, ch_id in enumerate(unique_channels_ai):
            ch_data = df_ai_sorted[df_ai_sorted['Channel_ID'] == ch_id]
            color = cmap(ch_idx % 10)
            ax_ver.scatter(ch_data['Center_X_px'], ch_data['Center_Y_px'], 
                           label=f"AI {ch_id}", color=color, s=35, edgecolor='white', marker='D', alpha=0.7, zorder=6)

    ax_ver.set_title("DUAL VISUAL VERIFICATION: Review Ground Truth (Circles) vs AI Predictions (Diamonds)\n(Close window to finalize)", 
                    fontsize=12, fontweight='bold')
    ax_ver.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
    plt.tight_layout()
    plt.show()

    print("\nProcessing complete! Both sets are ready for statistical evaluation.")

if __name__ == "__main__":
    main()