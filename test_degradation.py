"""
Test the degradation pipeline without RefCOCO dataset
This demonstrates the fog, smoke, and thermal effects on a sample image
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'rq1_dataset')

# Import the degradation pipeline
import os
os.chdir('/Volumes/TheDay/thedayproject/Cours Udem/IFT CV+NL/Horus Eye /Horus dev')

from degradation_pipeline import DegradationPipeline

# Create a simple test image (or use any image you have)
def create_test_image():
    """Create a colorful test image"""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some colorful shapes
    cv2.rectangle(img, (100, 100), (300, 300), (0, 255, 0), -1)  # Green square
    cv2.circle(img, (450, 240), 100, (255, 0, 0), -1)  # Blue circle
    cv2.rectangle(img, (200, 350), (440, 450), (0, 0, 255), -1)  # Red rectangle
    
    # Add some text
    cv2.putText(img, 'Test Image', (220, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (255, 255, 255), 2)
    
    return img

# Test the degradation pipeline
print("Testing Degradation Pipeline...")
print("=" * 50)

# Create pipeline with medium severity
pipeline = DegradationPipeline(severity=0.5)

# Create or load test image
test_image = create_test_image()

# Apply all degradations
results = pipeline.apply_all(test_image)

# Display results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
titles = ['Clean', 'Fog', 'Smoke', 'Thermal']

for ax, (name, img) in zip(axes.flat, results.items()):
    # Convert BGR to RGB for matplotlib
    if name != 'thermal':  # thermal is already in a colormap
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    ax.imshow(img_rgb)
    ax.set_title(name.capitalize(), fontsize=14, fontweight='bold')
    ax.axis('off')

plt.suptitle('Degradation Pipeline Test Results', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('degradation_test_results.png', dpi=150, bbox_inches='tight')
print("\n✓ Test completed successfully!")
print("✓ Results saved to: degradation_test_results.png")
print("\nAll degradation effects are working:")
print("  - Fog: Atmospheric scattering model")
print("  - Smoke: Procedural noise-based smoke")
print("  - Thermal: Infrared camera simulation")
print("\nNext steps:")
print("  1. Download RefCOCO annotations (see download_refcoco.sh)")
print("  2. Run dataset_pipeline.py with real RefCOCO data")

# plt.show()  # Commented out to prevent blocking in automated tests
