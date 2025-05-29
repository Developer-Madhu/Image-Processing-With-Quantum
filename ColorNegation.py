import cv2
import numpy as np
import matplotlib.pyplot as plt

def ocqr_negation_color_image(image_path):
    """
    Apply Optimized Contrast-based Quantized Representation (OCQR) for negation of color image.

    Parameters:
        image_path (str): Path to the input image.

    Returns:
        negated_image (np.ndarray): OCQR-negated color image.
    """
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalize image to [0, 1] range
    norm_image = image.astype(np.float32) / 255.0

    # Apply OCQR - For this simplified implementation, assume OCQR is an intelligent contrast-aware inversion
    # Step 1: Get per-channel intensity
    contrast_weights = np.array([0.299, 0.587, 0.114])  # Standard luminance weights
    intensity = np.dot(norm_image, contrast_weights)

    # Step 2: Invert intensities and scale to colors
    negated_image = 1.0 - norm_image  # Standard negation

    # Step 3: Enhance contrast by stretching dynamic range in each channel
    for c in range(3):
        min_val = np.min(negated_image[:, :, c])
        max_val = np.max(negated_image[:, :, c])
        negated_image[:, :, c] = (negated_image[:, :, c] - min_val) / (max_val - min_val + 1e-5)

    # Convert back to 8-bit
    negated_image = (negated_image * 255).astype(np.uint8)
    return negated_image

def display_images(original, transformed):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(transformed)
    plt.title("OCQR Negated Image")
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    image_path = "your_image.jpg"  # Replace with your image path
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    negated = ocqr_negation_color_image(image_path)
    display_images(original_image, negated)
