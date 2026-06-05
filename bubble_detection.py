import cv2
import numpy as np
import matplotlib.pyplot as plt

def measure_bubble_length(image_path, true_channel_width_um):
    # 1. Load image and convert to grayscale
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Apply thresholding to binarize the image (makes bubbles white, liquid/walls black)
    # Adjust the threshold values based on your video lighting!
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 3. Find contours (boundaries) of the bubbles
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bubble_lengths_px = []
    
    for cnt in contours:
        # Filter out tiny noise particles by area
        if cv2.contourArea(cnt) > 500: 
            # Get bounding rectangle around the bubble
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Assuming flow is horizontal, 'w' is the bubble length in pixels
            bubble_lengths_px.append(w)
            
            # Draw rectangle on original image for visual verification
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
    # 4. Calibration: Use channel height/width in pixels to get scale
    # Let's say you manually know the channel spans 150 pixels vertically in the image
    pixel_width_of_channel = 150 
    scale_um_per_px = true_channel_width_um / pixel_width_of_channel
    
    # Convert pixel lengths to micrometers
    bubble_lengths_um = [l * scale_um_per_px for l in bubble_lengths_px]
    
    # Display verified image
    plt.figure(figsize=(10, 4))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Detected Bubble Lengths: {np.round(bubble_lengths_um, 1)} um")
    plt.axis('off')
    plt.show()
    
    return bubble_lengths_um

# Example usage:
# lengths = measure_bubble_length('mid_reactor_frame.png', true_channel_width_um=400)