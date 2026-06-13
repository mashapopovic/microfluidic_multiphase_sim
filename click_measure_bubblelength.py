import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd

# --- CONFIGURATION ---
IMAGE_FOLDER = '/home/masas/Frames/*.bmp' 
OUTPUT_CSV = 'Bubble_Spatial_6Frame_Pairs_2.csv'

# Your exact diagonal-to-diagonal calibration values
TRUE_CHANNEL_WIDTH_UM = 1281.8  
PIXELS_PER_CHANNEL = 30.286 
ANGLE_DEG = 15

# Updated history depth to exactly 6 frames
GHOST_HISTORY_LIMIT = 6  

microns_per_pixel = TRUE_CHANNEL_WIDTH_UM / PIXELS_PER_CHANNEL
angle_rad = np.radians(ANGLE_DEG)

class SpatialPairHistoryMeasurer:
    def __init__(self, history_limit=6):
        self.all_data = []
        self.current_frame = ""
        self.clicks = []
        
        # Spatial Memory Queues for complete pairs
        self.current_frame_pairs = []   # List of ((x1, y1), (x2, y2)) tuples for this frame
        self.history_pairs = []         # List of lists storing past frames' coordinate pairs
        self.history_limit = history_limit
        
        self.fig = None
        self.ax = None

    def on_click(self, event):
        if event.xdata is not None and event.ydata is not None:
            self.clicks.append((event.xdata, event.ydata))
            
            # Plot active click points (Red = Start, Yellow = End)
            colors = ['ro', 'yo']
            self.ax.plot(event.xdata, event.ydata, colors[len(self.clicks)-1])
            self.fig.canvas.draw()
            
            if len(self.clicks) == 2:
                x1, y1 = self.clicks[0]
                x2, y2 = self.clicks[1]
                
                # 1. Calculate length metrics
                horizontal_px = abs(x2 - x1)
                horizontal_um = horizontal_px * microns_per_pixel
                
                # Calculate center purely for data classification & label placement
                cx = int(round((x1 + x2) / 2))
                cy = int(round((y1 + y2) / 2))
                
                print(f"[{self.current_frame}] Logged Bubble -> Center: ({cx}, {cy}) | Length: {horizontal_um:.1f} um")
                
                # Cache BOTH exact clicked points as a pair for the historical trail
                self.current_frame_pairs.append(((x1, y1), (x2, y2)))
                
                # 2. Append data entry
                self.all_data.append({
                    'Frame_Name': self.current_frame,
                    'Center_X_px': cx,
                    'Center_Y_px': cy,
                    'Horizontal_Length_Microns': round(horizontal_um, 2)
                })
                
                # 3. Active text label overlay
                self.ax.text(cx, cy - 15, f"({cx}, {cy})", color='lime', 
                             fontsize=10, fontweight='bold',
                             bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'))
                self.fig.canvas.draw()
                
                self.clicks.clear()

    def finalize_frame(self):
        # Push this frame's complete click pairs to history
        self.history_pairs.append(list(self.current_frame_pairs))
        
        # Maintain strict 6-frame rolling queue depth
        if len(self.history_pairs) > self.history_limit:
            self.history_pairs.pop(0)
            
        self.current_frame_pairs.clear()

# --- MAIN RUNTIME LOOP ---
measurer = SpatialPairHistoryMeasurer(history_limit=GHOST_HISTORY_LIMIT)
image_files = sorted(glob.glob(IMAGE_FOLDER))

if not image_files:
    print(f"Error: No images found matching the path: {IMAGE_FOLDER}")
else:
    print(f"Found {len(image_files)} frames. Starting 6-frame pair loop...\n")
    print(f"INSTRUCTIONS:")
    print(f"1. Fading blue dots display BOTH boundary points clicked up to 6 frames ago.")
    print(f"2. A subtle dashed blue line links past point pairs to clarify bubble spans.")
    print(f"3. Close the window to advance frames. Ctrl+C in terminal saves and quits.\n")

try:
    for file_path in image_files:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None: 
            continue
        
        measurer.current_frame = os.path.basename(file_path)
        measurer.clicks.clear()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        measurer.fig = fig
        measurer.ax = ax
        
        ax.imshow(img, cmap='gray')
        
        # --- PLOT 6-FRAME GHOST BOUNDARY PAIRS ---
        total_history_depth = len(measurer.history_pairs)
        for history_idx, past_pairs in enumerate(measurer.history_pairs):
            frame_age = total_history_depth - history_idx
            
            # Calculate fading alpha value (age 1 = sharpest, age 6 = faintest)
            alpha_val = 0.6 / frame_age 
            
            for p1, p2 in past_pairs:
                # Render both past boundary clicks as ghost dots
                ax.plot(p1[0], p1[1], 'bo', alpha=alpha_val, markersize=5)
                ax.plot(p2[0], p2[1], 'bo', alpha=alpha_val, markersize=5)
                
                # Draw a faint connecting line to show the historic bubble width context
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b--', alpha=alpha_val * 0.4, linewidth=1)
                
                # Only show text overlay for the immediate previous frame to keep things clean
                if frame_age == 1:
                    cx_prev = int((p1[0] + p2[0]) / 2)
                    cy_prev = int((p1[1] + p2[1]) / 2)
                    ax.text(cx_prev, cy_prev - 18, f"Prev: ({cx_prev},{cy_prev})", color='deepskyblue', 
                            alpha=0.5, fontsize=8, fontweight='bold',
                            bbox=dict(facecolor='black', alpha=0.3, edgecolor='none', boxstyle='round,pad=0.1'))
        
        # 15-Degree Reference Line
        height, width = img.shape
        x_end = height * np.tan(angle_rad)
        ax.plot([0, x_end], [height, 0], color='red', linewidth=1, linestyle='--', label='15° Ref Line')
        
        plt.title(f"Frame: {measurer.current_frame} | Close window to advance")
        plt.legend(loc='upper right')
        
        cid = fig.canvas.mpl_connect('button_press_event', measurer.on_click)
        plt.show()
        
        measurer.finalize_frame()

except KeyboardInterrupt:
    print("\n[INTERRUPTED] Stopping early. Saving compiled data entries...")

# --- EXPORT TO CSV ---
if measurer.all_data:
    df = pd.DataFrame(measurer.all_data)
    df = df.sort_values(by=['Frame_Name', 'Center_Y_px', 'Center_X_px'])
    df.to_csv(OUTPUT_CSV, index=False)
    print("\n=====================================")
    print("ANALYSIS COMPLETE!")
    print(f"Data saved cleanly to: {OUTPUT_CSV}")
    print("=====================================")
else:
    print("\nNo measurements recorded.")