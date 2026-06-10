import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd

# --- CONFIGURATION ---
IMAGE_FOLDER = '/home/masas/Frames/*.bmp' 
OUTPUT_EXCEL = 'Bubble_Tracked_Horizontal_Measurements.xlsx'

# Your exact diagonal-to-diagonal calibration values
TRUE_CHANNEL_WIDTH_UM = 1281.8  
PIXELS_PER_CHANNEL = 30.286 
ANGLE_DEG = 15

# TRACKING THRESHOLD: 
# Max pixel distance a bubble can travel between consecutive frames.
MAX_TRACKING_DISTANCE_PX = 200  

microns_per_pixel = TRUE_CHANNEL_WIDTH_UM / PIXELS_PER_CHANNEL
angle_rad = np.radians(ANGLE_DEG)

class TrackedBubbleMeasurer:
    def __init__(self):
        self.all_data = []
        self.current_frame = ""
        self.clicks = []
        
        # Identity Tracking States
        self.next_unique_id = 1
        self.previous_frame_bubbles = []       # List of {'id': X, 'center': (x, y)} from last frame
        self.current_frame_bubbles = []        # Accumulator for current frame's finalized bubbles
        self.current_frame_matched_ids = set() # Avoid assigning the same ID to two different selections
        
        self.fig = None
        self.ax = None

    def on_click(self, event):
        if event.xdata is not None and event.ydata is not None:
            self.clicks.append((event.xdata, event.ydata))
            
            # Plot the raw click markers on screen (Red for start, Yellow for end)
            colors = ['ro', 'yo']
            self.ax.plot(event.xdata, event.ydata, colors[len(self.clicks)-1])
            self.fig.canvas.draw()
            
            if len(self.clicks) == 2:
                x1, y1 = self.clicks[0]
                x2, y2 = self.clicks[1]
                
                # 1. Calculate length and bubble center coordinate
                horizontal_px = abs(x2 - x1)
                horizontal_um = horizontal_px * microns_per_pixel
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                # 2. Centroid Tracking Logic
                assigned_id = None
                min_dist = float('inf')
                best_match_idx = -1
                
                # Look through previous frame's coordinates for the closest match
                for idx, prev_b in enumerate(self.previous_frame_bubbles):
                    if prev_b['id'] in self.current_frame_matched_ids:
                        continue 
                    
                    dist = np.sqrt((cx - prev_b['center'][0])**2 + (cy - prev_b['center'][1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        best_match_idx = idx
                
                # If a close neighbor is found within constraints, persist the ID
                if best_match_idx != -1 and min_dist < MAX_TRACKING_DISTANCE_PX:
                    assigned_id = self.previous_frame_bubbles[best_match_idx]['id']
                    self.current_frame_matched_ids.add(assigned_id)
                    print(f"[{self.current_frame}] Matched Bubble ID {assigned_id} (Moved {min_dist:.1f} px)")
                else:
                    # Otherwise, instantiate a completely new unique bubble ID
                    assigned_id = self.next_unique_id
                    self.next_unique_id += 1
                    print(f"[{self.current_frame}] New Bubble Detected -> Assigned ID {assigned_id}")
                
                # Cache this position for the next frame's comparison map
                self.current_frame_bubbles.append({'id': assigned_id, 'center': (cx, cy)})
                
                # 3. Save entry data
                self.all_data.append({
                    'Frame_Name': self.current_frame,
                    'Bubble_ID': assigned_id,
                    'Horizontal_Length_Microns': round(horizontal_um, 2)
                })
                
                # 4. Text overlay on plot UI so you can see current tracking live
                self.ax.text(cx, cy - 15, f"ID {assigned_id}", color='lime', 
                             fontsize=11, fontweight='bold',
                             bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'))
                self.fig.canvas.draw()
                
                self.clicks.clear()

    def finalize_frame(self):
        # Pass the current frame's bubble coordinates to be the history for the next frame
        self.previous_frame_bubbles = list(self.current_frame_bubbles)
        self.current_frame_bubbles.clear()
        self.current_frame_matched_ids.clear()

# --- MAIN RUNTIME LOOP ---
measurer = TrackedBubbleMeasurer()
image_files = sorted(glob.glob(IMAGE_FOLDER))

if not image_files:
    print(f"Error: No images found matching the path: {IMAGE_FOLDER}")
else:
    print(f"Found {len(image_files)} frames. Starting tracking analysis loop...\n")
    print("INSTRUCTIONS:")
    print("1. Faint blue tags show where bubbles were in the PREVIOUS frame.")
    print("2. Click the new start/end positions of the corresponding bubble.")
    print("3. Close the window to save and advance frames.")
    print("4. Press Ctrl+C in the terminal to save and exit early.\n")

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
        
        # Render the fresh frame
        ax.imshow(img, cmap='gray')
        
        # --- NEW: PLOT GHOST MARKERS FROM THE PREVIOUS FRAME ---
        for prev_b in measurer.previous_frame_bubbles:
            px, py = prev_b['center']
            # Draw a faint blue circle at its old position
            ax.plot(px, py, 'bo', alpha=0.4, markersize=6)
            # Add a faint blue text label right above it
            ax.text(px, py - 18, f"Last ID {prev_b['id']}", color='deepskyblue', 
                    alpha=0.5, fontsize=9, fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.3, edgecolor='none', boxstyle='round,pad=0.1'))
        
        # Reference Line (15-degrees)
        height, width = img.shape
        x_end = height * np.tan(angle_rad)
        ax.plot([0, x_end], [height, 0], color='red', linewidth=1, linestyle='--', label='15° Ref Line')
        
        plt.title(f"Frame: {measurer.current_frame} | Close window to advance")
        plt.legend(loc='upper right')
        
        cid = fig.canvas.mpl_connect('button_press_event', measurer.on_click)
        plt.show()  # Pauses until window is closed
        
        # Shift current tracking coordinates to history slot
        measurer.finalize_frame()

except KeyboardInterrupt:
    print("\n[INTERRUPTED] Script stopped manually. Saving compiled tracking data...")

# --- EXPORT TO SPREADSHEET ---
if measurer.all_data:
    df = pd.DataFrame(measurer.all_data)
    
    # Sort data logically by Bubble ID then by Frame Name
    df = df.sort_values(by=['Bubble_ID', 'Frame_Name'])
    
    df.to_excel(OUTPUT_EXCEL, index=False)
    print("\n=====================================")
    print("ANALYSIS COMPLETE!")
    print(f"Tracking data exported cleanly to: {OUTPUT_EXCEL}")
    print("=====================================")
else:
    print("\nNo measurements recorded.")