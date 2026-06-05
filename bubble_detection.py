import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_spiral_reactor_images(image_path, true_width_um):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    center_x, center_y = 500, 500  # Must match the physical center of your spiral chip
    R_in = 60                      # Inner radius in pixels
    pitch = 70                     # Ring pitch in pixels
    
    # Map pixel size to microns based on known channel cross section
    pixel_width_of_channel = 18
    microns_per_pixel = true_width_um / pixel_width_of_channel
    
    # Trace along the mathematical spiral trajectory
    theta_steps = np.linspace(0, 5 * 2 * np.pi, 8000)
    spiral_profile = []
    z_distances_um = []
    
    current_z = 0.0
    last_x, last_y = None, None
    
    for theta in theta_steps:
        r = R_in + (pitch * theta / (2 * np.pi))
        x = int(center_x + r * np.cos(theta))
        y = int(center_y + r * np.sin(theta))
        
        if 0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]:
            # Record the pixel brightness at this exact point along the spiral path
            spiral_profile.append(gray[y, x])
            
            # Calculate cumulative physical length traveled (z)
            if last_x is not None:
                step_px = np.sqrt((x - last_x)**2 + (y - last_y)**2)
                current_z += step_px * microns_per_pixel
            z_distances_um.append(current_z)
            
            last_x, last_y = x, y
            
            # Draw a tiny dot on your image for diagnostic verification
            if int(theta * 10) % 5 == 0:
                cv2.circle(img, (x, y), 1, (0, 255, 0), -1)

    # Threshold the unrolled 1D profile to isolate gas (1) from liquid (0)
    spiral_profile = np.array(spiral_profile)
    _, binary_profile = cv2.threshold(spiral_profile, 200, 1, cv2.THRESH_BINARY)
    
    # Track phase switches to determine individual segment lengths
    phase_changes = np.where(np.diff(binary_profile) != 0)[0]
    
    bubble_lengths_um = []
    z_positions_um = []
    
    for i in range(len(phase_changes) - 1):
        idx_start = phase_changes[i]
        idx_end = phase_changes[i+1]
        
        # If the phase inside this segment is gas (high brightness)
        if binary_profile[idx_start + 1] == 1:
            length_um = z_distances_um[idx_end] - z_distances_um[idx_start]
            bubble_lengths_um.append(length_um)
            z_positions_um.append(z_distances_um[idx_start]) # Track WHERE this bubble is located
            
    # Display the tracked path overlay
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Tracked Spiral Path (Green)")
    plt.axis('off')
    
    # Plot the resulting data points showing bubble lengths over reactor position
    plt.subplot(1, 2, 2)
    plt.scatter(np.array(z_positions_um) / 10000, bubble_lengths_um, color='blue', alpha=0.7)
    plt.xlabel("Reactor Axial Length z (cm)")
    plt.ylabel("Measured Bubble Length L_b (um)")
    plt.title("Extracted Axial Data Profile")
    plt.tight_layout()
    plt.savefig("spiral_analysis_results.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return z_positions_um, bubble_lengths_um


# Run data extraction
# z_pos, b_lens = analyze_spiral_reactor_images('test_spiral_reactor.png', true_width_um=400)

