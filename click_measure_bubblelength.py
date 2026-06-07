import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
IMAGE_PATH = 'S1_H4_L1_p14_C04_T60_Seq001_Img000001.bmp'
TRUE_CHANNEL_WIDTH_UM = 1000  
PIXELS_PER_CHANNEL = 45       # Adjust based on your visual channel calibration

microns_per_pixel = TRUE_CHANNEL_WIDTH_UM / PIXELS_PER_CHANNEL
clicks = []

def calculate_arc_length(p1, p2, p3):
    """
    Calculates the exact arc length passing through 3 points.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    
    # Mathematical determinant to find the unique circle center (xc, yc)
    A = x1*(y2 - y3) - y1*(x2 - x3) + x2*y3 - x3*y2
    B = (x1**2 + y1**2)*(y3 - y2) + (x2**2 + y2**2)*(y1 - y3) + (x3**2 + y3**2)*(y2 - y1)
    C = (x1**2 + y1**2)*(x2 - x3) + (x2**2 + y2**2)*(x3 - x1) + (x3**2 + y3**2)*(x1 - x2)
    
    if A == 0:
        # Points are a perfectly straight line
        return np.sqrt((x3 - x1)**2 + (y3 - y1)**2)
        
    xc = -B / (2 * A)
    yc = -C / (2 * A)
    radius = np.sqrt((x1 - xc)**2 + (y1 - yc)**2)
    
    # Calculate vectors from center to start (p1) and end (p3)
    v1 = np.array([x1 - xc, y1 - yc])
    v3 = np.array([x3 - xc, y3 - yc])
    
    # Find the angle alpha between them using dot product
    cos_angle = np.dot(v1, v3) / (np.linalg.norm(v1) * np.linalg.norm(v3))
    cos_angle = np.clip(cos_angle, -1.0, 1.0) # Prevent floating point errors
    alpha = np.arccos(cos_angle)
    
    # True curved arc length in pixels
    arc_length_px = radius * alpha
    return arc_length_px

def on_click(event):
    if event.xdata is not None and event.ydata is not None:
        clicks.append((event.xdata, event.ydata))
        
        # Color code clicks: 1st=Red, 2nd=Yellow, 3rd=Green
        colors = ['ro', 'yo', 'go']
        plt.plot(event.xdata, event.ydata, colors[len(clicks)-1])
        plt.draw()
        
        if len(clicks) == 3:
            arc_px = calculate_arc_length(clicks[0], clicks[1], clicks[2])
            arc_um = arc_px * microns_per_pixel
            
            print(f"\n--- CURVED MEASUREMENT SUCCESS ---")
            print(f"True Curved Arc Length: {arc_um:.1f} um")
            print("==================================")
            clicks.clear()

# Load and display
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError("Could not find your image file.")

print("INSTRUCTIONS FOR SPIRAL ARC MEASUREMENT:")
print("For each bubble, click 3 points along its curved center path:")
print("1st Click: Bubble START (Red)")
print("2nd Click: Bubble MIDDLE CURVE (Yellow)")
print("3rd Click: Bubble END (Green) -> Instantly prints true arc length!")

fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Click 3 Points Along the Bubble Curve (Start -> Mid -> End)")
cid = fig.canvas.mpl_connect('button_press_event', on_click)
plt.show()