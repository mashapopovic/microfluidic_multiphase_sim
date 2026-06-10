import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
IMAGE_PATH = 'S1_H4_L1_p14_C04_T60_Seq001_Img000001.bmp'
TRUE_CHANNEL_WIDTH_UM = 1000  
PIXELS_PER_CHANNEL = 18

microns_per_pixel = TRUE_CHANNEL_WIDTH_UM / PIXELS_PER_CHANNEL
clicks = []

def calculate_vertical_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(y2 - y1)

def on_click(event):
    if event.xdata is not None and event.ydata is not None:
        clicks.append((event.xdata, event.ydata))
        
        colors = ['ro', 'yo']
        ax.plot(event.xdata, event.ydata, colors[len(clicks)-1])
        fig.canvas.draw()
        
        if len(clicks) == 2:
            vertical_px = calculate_vertical_distance(clicks[0], clicks[1])
            vertical_um = vertical_px * microns_per_pixel
            
            print(f"\n--- VERTICAL DISTANCE MEASUREMENT ---")
            print(f"Vertical Distance: {vertical_um:.1f} um")
            print("=====================================")
            clicks.clear()

# load img in grayscale
img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Could not find your image file at: {IMAGE_PATH}")

height, width = img.shape

# trigonometry for the 15 degree line from the vertical
angle_rad = np.radians(15)
x_end = height * np.tan(angle_rad)

# single plotting window for clicks and image
fig, ax = plt.subplots(figsize=(12, 8))

# cmap='gray' stops the image from turning green when plot w matlib
ax.imshow(img, cmap='gray') 

# red line starts from the bottom left corner at a degree of 15 from the vertical
ax.plot([0, x_end], [height, 0], color='red', linewidth=1, label='15° Reference Line')

# Terminal Instructions
print("INSTRUCTIONS FOR VERTICAL DISTANCE MEASUREMENT:")
print("For each bubble, click 2 points along its center path:")
print("1st Click: Bubble START (Red)")
print("2nd Click: Bubble END (Yellow) -> Instantly prints vertical distance!\n")

plt.title("Taylor Bubble Vertical Distance Measurement (15° Vertical Bound)")
plt.legend(loc='upper right')

# Connect the click behavior to this specific window
cid = fig.canvas.mpl_connect('button_press_event', on_click)
plt.show()