import cv2
import numpy as np

def create_synthetic_taylor_flow(filename="test_taylor_flow.png"):
    # 1. Create a dark canvas (representing the background/chip bulk)
    # 200 pixels high, 1200 pixels long
    img = np.zeros((200, 1200), dtype=np.uint8) + 50 
    
    # 2. Draw a clean, bright horizontal channel through the center
    # The channel is exactly 100 pixels tall (from y=50 to y=150)
    channel_top = 50
    channel_bottom = 150
    img[channel_top:channel_bottom, :] = 180 
    
    # 3. Inject "Gas Bubbles" (Capsules/Rounded Rectangles)
    # In Taylor flow, bubbles have dark borders or bright centers. 
    # Let's draw regular white blocks to simulate the gas phase segments.
    bubble_length_px = 160
    slug_length_px = 100
    start_x = 40
    
    while start_x + bubble_length_px < 1200:
        end_x = start_x + bubble_length_px
        
        # Draw the bubble body (white fill)
        img[channel_top+5:channel_bottom-5, start_x:end_x] = 255
        
        # Draw rounded caps on the bubble edges to mimic a real meniscus
        cv2.circle(img, (start_x, 100), 45, 255, -1)
        cv2.circle(img, (end_x, 100), 45, 255, -1)
        
        # Move forward by one complete unit cell (Bubble + Liquid Slug)
        start_x = end_x + slug_length_px

    # Clean up channel boundary lines outside bubbles
    img[0:channel_top, :] = 30
    img[channel_bottom:, :] = 30

    # Save the synthetic image to your current workspace directory
    cv2.imwrite(filename, img)
    print(f"Success! Synthetic Taylor Flow image saved as '{filename}'")

if __name__ == "__main__":
    create_synthetic_taylor_flow()