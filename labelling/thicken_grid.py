import os
import cv2
import numpy as np

'''
This script is used to thicken and blacken the grid highlighting the grid more explicitly
Main purpose is to apply enhanced separation between labels to be sent to Flash Vision API
Using CV2, techniques applied include:
- Gray scaling
- Contrast enhancement for better detection (a=0.8 b=0).
- Adaptive thresholding with fine-tuned parameters (15,3)
- Stronger dilation kernel (2,2) for more pronounced lines
- Enhanced morphological operations by horizontal ((MORPH_RECT, (100, 1)) and vertical (MORPH_RECT, (1, 100)) to capture more grid details
- Thicker grid drawing
'''

def process_image(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    # Increase contrast for better grid detection
    contrasted = cv2.convertScaleAbs(image, alpha=0.8, beta=0)  # Increase contrast (dark to darker and bright to brighter) by alpha factor, by opposite, beta decrease it

    # Convert to grayscale
    gray = cv2.cvtColor(contrasted, cv2.COLOR_BGR2GRAY)

    # Apply adaptive threshold to emphasize grid lines
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 3
    )

    # Use morphological operations to isolate grid lines
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=3)  # Increased iterations for thicker lines

    # Find horizontal and vertical lines separately
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 1)) # Larger kernel
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 150))   # Larger kernel

    horizontal_lines = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, vertical_kernel)

    # Combine the lines
    grid = cv2.bitwise_or(horizontal_lines, vertical_lines)

    # Dilate the combined grid further for intensity
    grid = cv2.dilate(grid, kernel, iterations=2)

    # Invert the grid to make lines black on a white background
    grid_inverted = cv2.bitwise_not(grid)

    # Overlay the black grid on the original image
    grid_bgr = cv2.cvtColor(grid_inverted, cv2.COLOR_GRAY2BGR)
    result = cv2.addWeighted(image, 0.8, grid_bgr, 0.3, 0)  # Increased blending for visibility

    # Show intermediate results for visualization
    # cv2.imshow("Contrast Enhanced", contrasted)
    # cv2.imshow("Binary Threshold", binary)
    # cv2.imshow("Horizontal Lines", horizontal_lines)
    # cv2.imshow("Vertical Lines", vertical_lines)
    # cv2.imshow("Combined Grid", grid)
    cv2.imshow("Final Result", result)
    cv2.waitKey(0)  # Wait for a key press in each visualization
    cv2.destroyAllWindows()

    # Save the result
    cv2.imwrite(output_path, result)
    print(f"Processed image saved to: {output_path}")


if __name__ == "__main__":
    # Directory setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(script_dir, "../data/train/images")
    output_dir = os.path.join(script_dir, "../data/train/processed_images")

    os.makedirs(output_dir, exist_ok=True)

    for image_file in os.listdir(image_dir):
        # Only process supported image files
        if image_file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(image_dir, image_file)
            output_path = os.path.join(output_dir, f"processed_{image_file}")
            process_image(image_path, output_path)
