import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd

# --- CONFIGURATION ---
IMAGE_FOLDER = '/home/masas/Frames/*.bmp' 
OUTPUT_EXCEL = 'Bubble_Spatial_History_Measurements.xlsx'

# Your exact diagonal-to-diagonal calibration values
TRUE_CHANNEL_WIDTH_UM = 1281.8  
PIXELS_PER_CHANNEL = 30.286 
ANGLE_DEG = 15

# How many frames back the ghost dots should persist
GHOST_HISTORY_LIMIT = 4  

microns_per_pixel = TRUE_CHANNEL_WIDTH_UM / PIXELS_PER_CHANNEL
angle_rad = np.radians(ANGLE_DEG)

class SpatialHistoryBubbleMeasurer:
    def __init__(self, history_limit=4):
        self.all_data = []
        self.current_frame = ""
        self.clicks = []
        
        # Spatial Memory Queues
        self.current_frame_centers = []   # List of (cx, cy) for the active frame
        self.history_centers = []         # List of lists storing past frames' centers
        self.history_limit = history_limit
        
        self.fig = None
        self.ax = None

    def on_click(self, event):
        if event.xdata is not None and event.ydata is not None:
            self.clicks.append((event.xdata, event.ydata))
            
            # Plot raw click points (Red = Start, Yellow = End)
            colors = ['ro', 'yo']
            self.ax.plot(event.xdata, event.ydata, colors[len(self.clicks)-1])
            self.fig.canvas.draw()
            
            if len(self.clicks) == 2:
                x1, y1 = self.clicks[0]
                x2, y2 = self.clicks[1]
                
                # 1. Calculate length component
                horizontal_px = abs(x2 - x1)
                horizontal_um = horizontal_px * microns_per_pixel
                
                # 2. Classify by exact pixel center coordinate
                cx = int(round((x1 + x2) / 2))
                cy = int(round((y1 + y2) / 2))
                
                print(f"[{self.current_frame}] Center px: ({cx}, {cy}) -> {horizontal_um:.1f} um")
                
                # Cache this coordinate for the rolling history tracking queue
                self.current_frame_centers.append((cx, cy))
                
                # 3. Append to dataset
                self.all_data.append({
                    'Frame_Name': self.current_frame,
                    'Center_X_px': cx,
                    'Center_Y_px': cy,
                    'Horizontal_Length_Microns': round(horizontal_um, 2)
                })
                
                # 4. Live text overlay for current click
                self.ax.text(cx, cy - 15, f"({cx}, {cy})", color='lime', 
                             fontsize=10, fontweight='bold',
                             bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'))
                self.fig.canvas.draw()
                
                self.clicks.clear()

    def finalize_frame(self):
        # Append current frame's centers to the rolling historical timeline
        self.history_centers.append(list(self.current_frame_centers))
        
        # Trim history if it exceeds our frame depth limit (e.g., keeping last 4 frames)
        if len(self.history_centers) > self.history_limit:
            self.history_centers.pop(0)
            
        self.current_frame_centers.clear()

# --- MAIN RUNTIME LOOP ---
measurer = SpatialHistoryBubbleMeasurer(history_limit=GHOST_HISTORY_LIMIT)
image_files = sorted(glob.glob(IMAGE_FOLDER))

if not image_files:
    print(f"Error: No images found matching the path: {IMAGE_FOLDER}")
else:
    print(f"Found {len(image_files)} frames. Starting spatial history loop...\n")
    print(f"INSTRUCTIONS:")
    print(f"1. Fading blue dots show a tracking trail up to {GHOST_HISTORY_LIMIT} frames back.")
    print(f"2. Only the most recent historical frame displays its coordinate text to minimize clutter.")
    print(f"3. Close the window to advance frames. Ctrl+C in terminal saves progress and quits.\n")

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
        
        # Display main frame
        ax.imshow(img, cmap='gray')
        
        # --- PLOT ROLLING GHOST TRAILS FROM PAST FRAMES ---
        total_history_depth = len(measurer.history_centers)
        for history_idx, past_centers in enumerate(measurer.history_centers):
            # Calculate frame age (1 = immediate previous frame, 4 = oldest frame in memory)
            frame_age = total_history_depth - history_idx
            
            # Scale transparency dynamically (older frames = more transparent)
            alpha_val = 0.55 / frame_age 
            
            for px, py in past_centers:
                # Draw historical trail markers
                ax.plot(px, py, 'bo', alpha=alpha_val, markersize=5)
                
                # Only overlay text labels for the immediate previous frame (Age == 1)
                if frame_age == 1:
                    ax.text(px, py - 18, f"Prev: ({px},{py})", color='deepskyblue', 
                            alpha=0.5, fontsize=8, fontweight='bold',
                            bbox=dict(facecolor='black', alpha=0.3, edgecolor='none', boxstyle='round,pad=0.1'))
        
        # Reference Line (15-degrees)
        height, width = img.shape
        x_end = height * np.tan(angle_rad)
        ax.plot([0, x_end], [height, 0], color='red', linewidth=1, linestyle='--', label='15° Ref Line')
        
        plt.title(f"Frame: {measurer.current_frame} | Close window to advance")
        plt.legend(loc='upper right')
        
        cid = fig.canvas.mpl_connect('button_press_event', measurer.on_click)
        plt.show()
        
        measurer.finalize_frame()

except KeyboardInterrupt:
    print("\n[INTERRUPTED] Stopping early. Compiling history data entries...")

# --- EXPORT TO SPREADSHEET ---
if measurer.all_data:
    df = pd.DataFrame(measurer.all_data)
    
    # Sort logically by frame timeline, then spatial placement
    df = df.sort_values(by=['Frame_Name', 'Center_Y_px', 'Center_X_px'])
    
    df.to_excel(OUTPUT_EXCEL, index=False)
    print("\n=====================================")
    print("ANALYSIS COMPLETE!")
    print(f"Spatial history data saved to: {OUTPUT_EXCEL}")
    print("=====================================")
else:
    print("\nNo measurements recorded.")