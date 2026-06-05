import cv2
import numpy as np

def create_synthetic_spiral_flow(filename="test_spiral_reactor.png"):
    # Create a large blank canvas (1000x1000 pixels)
    img = np.zeros((1000, 1000), dtype=np.uint8) + 40
    center_x, center_y = 500, 500
    
    # NEW PARAMETERS FOR A TIGHTLY PACKED 20-WINDING SPIRAL
    R_in = 40            # Starting inner radius from center
    pitch = 21           # Distances between consecutive rings (down from 70)
    channel_width = 10   # Narrower channel to prevent overlap (down from 18)
    
    # 20 full rotations (20 * 2 * pi)
    num_windings = 20
    theta_max = num_windings * 2 * np.pi
    
    # Increased resolution (25,000 steps) to keep the line smooth over a much longer path
    theta_steps = np.linspace(0, theta_max, 25000)
    
    # Scaled down the periods to match the narrower channel
    bubble_period = 50  # Number of steps a gas bubble lasts
    slug_period = 30    # Number of steps a liquid slug lasts
    
    # First pass: Draw the continuous channel tracks (light gray background)
    for theta in theta_steps:
        r = R_in + (pitch * theta / (2 * np.pi))
        x = int(center_x + r * np.cos(theta))
        y = int(center_y + r * np.sin(theta))
        if 0 <= x < 1000 and 0 <= y < 1000:
            # Outer wall definition
            cv2.circle(img, (x, y), channel_width // 2 + 1, 120, -1)

    # Second pass: Overlay the alternating Taylor bubbles (bright white)
    for idx, theta in enumerate(theta_steps):
        cycle_pos = idx % (bubble_period + slug_period)
        if cycle_pos < bubble_period:  # It's a gas bubble segment
            r = R_in + (pitch * theta / (2 * np.pi))
            x = int(center_x + r * np.cos(theta))
            y = int(center_y + r * np.sin(theta))
            if 0 <= x < 1000 and 0 <= y < 1000:
                # Inner gas phase definition
                cv2.circle(img, (x, y), channel_width // 2 - 1, 255, -1)
                
    cv2.imwrite(filename, img)
    print(f"Success! Tightly packed spiral reactor ({num_windings} windings) saved as '{filename}'")

if __name__ == "__main__":
    create_synthetic_spiral_flow()