import matplotlib.pyplot as plt

# Your extracted bubble lengths (in microns)
slug_lengths_um_2 = [
    2944.6, 2833.5, 2647.8, 3297.9, 2514.6, 3222.3, 3332.1, 3201.4, 
    3009.9, 3258.8, 2898.6, 3197.3, 3312.5, 2904.7, 3092.4, 3157.9, 
    3184.1, 3453.9, 2922.2, 3331.3, 3795.9, 2953.8, 2757.4, 3281.3, 
    3100.5, 3194.5, 2777.1, 5203.6, 3821.9, 2580.9, 3221.3, 4274.8, 
    2889.7, 2829.9, 3156.8, 3120.2, 3688.0, 6548.9, 5796.3, 2865.6, 
    3787.6, 3190.1, 3246.9, 4269.5, 2844.4
]

# Generate an index (1, 2, 3...) for the x-axis
measurement_numbers = list(range(1, len(slug_lengths_um_2) + 1))

# Set up the plot size and style
plt.figure(figsize=(10, 6))

# Plot the points as a clear scatter plot
plt.scatter(measurement_numbers, slug_lengths_um_2, color='royalblue', edgecolor='darkblue', s=60, alpha=0.8, zorder=3)

# Connect them with a faint line to track sequencing trends
plt.plot(measurement_numbers, slug_lengths_um_2, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=2)

# Labels and Styling
plt.title("Experimental Slug Length Profile", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Measurement Sequence Number", fontsize=12)
plt.ylabel("Measured Slug Length L_s (µm)", fontsize=12)

# Add a clean grid
plt.grid(True, linestyle=':', alpha=0.6, zorder=1)

# Clean up layout margins
plt.tight_layout()

# Save the plot automatically as a high-res image for your report
plt.savefig("experimental_slug_lengths.png", dpi=300, bbox_inches='tight')

# Display the graph on screen
plt.show()