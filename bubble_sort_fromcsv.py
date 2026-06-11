import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# --- CONFIG ---
INPUT_CSV = 'Bubble_Spatial_6Frame_Pairs.csv'  # Your raw clicked data
REFERENCE_IMAGE = '/home/masas/Frames/S1_H4_L1_p14_C04_T60_Seq001_Img000001.bmp'     # An image showing the channel layout
OUTPUT_CSV = 'Bubble_Spiral_Sorted.csv'

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
            
            # draw point and connecting lines live
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
                
                # Redraw the finalized channel line in green to lock it visually
                xs, ys = zip(*self.current_channel)
                self.ax.plot(xs, ys, 'g-', linewidth=2)
                self.ax.text(xs[0], ys[0], f"Ch {len(self.channels)}", color='lime', 
                             fontsize=12, fontweight='bold', bbox=dict(facecolor='black', alpha=0.7))
                
                self.current_channel.clear()
                self.update_title()
            else:
                print("Click some points before trying to move to the next channel!")

def point_to_segment_distance(p, a, b):
    """Calculates the shortest distance from point p to line segment ab."""
    p = np.array(p)
    a = np.array(a)
    b = np.array(b)
    ab = b - a
    ap = p - a
    ab_len_sq = np.sum(ab**2)
    if ab_len_sq == 0:
        return np.linalg.norm(ap)
    t = np.dot(ap, ab) / ab_len_sq
    t = max(0.0, min(1.0, t))  # Clamp to segment bounds
    closest_point = a + t * ab
    return np.linalg.norm(p - closest_point)

def point_to_path_distance(point, path):
    """Finds the absolute minimum distance from a point to a piecewise linear path."""
    min_dist = float('inf')
    for i in range(len(path) - 1):
        dist = point_to_segment_distance(point, path[i], path[i+1])
        if dist < min_dist:
            min_dist = dist
    return min_dist

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Missing data file '{INPUT_CSV}'")
        return

    # Step 1: Have the user draw the boundaries/tracks on the reference image
    try:
        drawer = ChannelTrackDefiner(REFERENCE_IMAGE)
    except Exception as e:
        print(e)
        return

    # Catch if window was closed while drawing the final track
    if drawer.current_channel:
        drawer.channels.append(list(drawer.current_channel))

    if not drawer.channels:
        print("No channels were drawn. Sorting aborted.")
        return

    print(f"\nProcessing classifications for {len(drawer.channels)} defined channels...")

    # Step 2: Read raw bubble dataset
    df = pd.read_csv(INPUT_CSV)
    
    assigned_channels = []
    
    # Step 3: Math classification based on proximity to tracks
    for idx, row in df.iterrows():
        bx = row['Center_X_px']
        by = row['Center_Y_px']
        bubble_point = (bx, by)
        
        # Calculate distance from this bubble to every drawn channel track
        distances = [point_to_path_distance(bubble_point, path) for path in drawer.channels]
        
        # Find the index of the closest channel track
        closest_channel_idx = np.argmin(distances)
        assigned_channels.append(f"Channel_{closest_channel_idx + 1:02d}")

    df['Channel_ID'] = assigned_channels

    # Logically sort the data
    df_sorted = df.sort_values(by=['Channel_ID', 'Frame_Name', 'Center_X_px'])

    # Step 4: VISUAL VERIFICATION STEP
    print("\nLoading Visual Verification Plot... Review accuracy before saving.")
    fig_ver, ax_ver = plt.subplots(figsize=(12, 8))
    ax_ver.imshow(drawer.img_rgb)
    
    # Generate unique colors for each channel using a colormap
    unique_channels = sorted(df_sorted['Channel_ID'].unique())
    cmap = plt.cm.get_cmap('tab10', len(unique_channels))
    
    for ch_idx, ch_id in enumerate(unique_channels):
        ch_data = df_sorted[df_sorted['Channel_ID'] == ch_id]
        color = cmap(ch_idx)
        
        # Plot all bubbles belonging to this channel as a scatter cluster
        ax_ver.scatter(ch_data['Center_X_px'], ch_data['Center_Y_px'], 
                       label=ch_id, color=color, s=25, edgecolor='black', zorder=5)
        
        # Plot the original track line underneath for comparison
        track_idx = int(ch_id.split('_')[1]) - 1
        tx, ty = zip(*drawer.channels[track_idx])
        ax_ver.plot(tx, ty, color=color, linestyle='--', alpha=0.7, linewidth=1.5)

    ax_ver.set_title("VISUAL VERIFICATION: Review Classified Bubbles\n(Close window to finalize and save CSV)", 
                     fontsize=12, fontweight='bold')
    ax_ver.legend(loc='upper right')
    plt.show()  # Pauses script execution so you can look closely at the sorted groupings

    # Step 5: Final Save Execution
    df_sorted.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSorting complete! Clean data file written to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()