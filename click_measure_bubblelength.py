import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd

# --- CONFIGURATION ---
# Change this path to point to the folder where your frames are stored
IMAGE_FOLDER = '/home/masas/Frames/*.bmp' 
OUTPUT_EXCEL = 'Bubble_Horizontal_Measurements.xlsx'

# Your exact diagonal-to-diagonal calibration values
TRUE_CHANNEL_WIDTH_UM = 1281.8  
PIXELS_PER_CHANNEL = 30.286 
ANGLE_DEG = 15

microns_per_pixel = TRUE_CHANNEL_WIDTH_UM / PIXELS_PER_CHANNEL
angle_rad = np.radians(ANGLE_DEG)

class BubbleMeasurer:
    def __init__(self):
        self.all_data = []
        self.current_frame = ""
        self.clicks = []
        self.bubble_count = 0
        self.fig = None
        self.ax = None

    def calculate_horizontal_distance(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return abs(x2 - x1)

    def on_click(self, event):
        if event.xdata is not None and event.ydata is not None:
            self.clicks.append((event.xdata, event.ydata))
            
            # Draw visual markers on the image (Red for start, Yellow for end)
            colors = ['ro', 'yo']
            self.ax.plot(event.xdata, event.ydata, colors[len(self.clicks)-1])
            self.fig.canvas.draw()
            
            if len(self.clicks) == 2:
                # Calculate the manual horizontal distance component
                horizontal_px = self.calculate_horizontal_distance(self.clicks[0], self.clicks[1])
                horizontal_um = horizontal_px * microns_per_pixel
                
                self.bubble_count += 1
                print(f"[{self.current_frame}] Bubble {self.bubble_count}: {horizontal_um:.1f} um")
                
                # Store the measurement
                self.all_data.append({
                    'Frame_Name': self.current_frame,
                    'Bubble_Index': self.bubble_count,
                    'Horizontal_Distance_Microns': round(horizontal_um, 2)
                })
                
                self.clicks.clear()

# --- MAIN RUNTIME LOOP ---
measurer = BubbleMeasurer()
image_files = sorted(glob.glob(IMAGE_FOLDER))

if not image_files:
    print(f"Error: No images found matching the path: {IMAGE_FOLDER}")
else:
    print(f"Found {len(image_files)} frames. Starting manual analysis loop...\n")
    print("Press Ctrl+C in the terminal at any time to SAVE and QUIT early.\n")

try:
    for file_path in image_files:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None: 
            continue
        
        measurer.current_frame = os.path.basename(file_path)
        measurer.bubble_count = 0
        measurer.clicks.clear()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        measurer.fig = fig
        measurer.ax = ax
        
        ax.imshow(img, cmap='gray')
        
        height, width = img.shape
        x_end = height * np.tan(angle_rad)
        ax.plot([0, x_end], [height, 0], color='red', linewidth=1, label='15° Reference Line')
        
        plt.title(f"Frame: {measurer.current_frame} (Close window to next frame)")
        plt.legend(loc='upper right')
        
        cid = fig.canvas.mpl_connect('button_press_event', measurer.on_click)
        plt.show()

except KeyboardInterrupt:
    print("\n[INTERRUPTED] Stopping loop early. Saving collected data...")

# --- SAVE DATA TO EXCEL ---
# This line now runs even if you press Ctrl+C!
if measurer.all_data:
    df = pd.DataFrame(measurer.all_data)
    df.to_excel(OUTPUT_EXCEL, index=False)
    print("\n=====================================")
    print(f"Data saved cleanly to: {OUTPUT_EXCEL}")
    print("=====================================")