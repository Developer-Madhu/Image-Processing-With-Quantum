# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from qiskit import QuantumCircuit

# def pixel_to_quantum_circuit(pixel_rgb):
#     """
#     Convert an RGB pixel to a quantum circuit representation.

#     Parameters:
#         pixel_rgb (list of int): RGB values [R, G, B] (0-255)

#     Returns:
#         QuantumCircuit: A quantum circuit representing the pixel's RGB.
#     """
#     # Normalize RGB values to range [0, 1]
#     r, g, b = [val / 255.0 for val in pixel_rgb]

#     # Create a 3-qubit circuit, one for each color channel
#     qc = QuantumCircuit(3, name=f"RGB {pixel_rgb}")

#     # Encode each channel into its qubit using Ry (rotation around Y-axis)
#     qc.ry(r * np.pi, 0)  # R to qubit 0
#     qc.ry(g * np.pi, 1)  # G to qubit 1
#     qc.ry(b * np.pi, 2)  # B to qubit 2

#     return qc

# def ocqr_negation_color_image(image_path):
#     """
#     Apply Optimized Contrast-based Quantized Representation (OCQR) for negation of color image.

#     Parameters:
#         image_path (str): Path to the input image.

#     Returns:
#         negated_image (np.ndarray): OCQR-negated color image.
#     """
#     # Load the image
#     image = cv2.imread(image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # Normalize image to [0, 1] range
#     norm_image = image.astype(np.float32) / 255.0

#     # Apply OCQR - For this simplified implementation, assume OCQR is an intelligent contrast-aware inversion
#     # Step 1: Get per-channel intensity
#     contrast_weights = np.array([0.299, 0.587, 0.114])  # Standard luminance weights
#     intensity = np.dot(norm_image, contrast_weights)

#     # Step 2: Invert intensities and scale to colors
#     negated_image = 1.0 - norm_image  # Standard negation

#     # Step 3: Enhance contrast by stretching dynamic range in each channel
#     for c in range(3):
#         min_val = np.min(negated_image[:, :, c])
#         max_val = np.max(negated_image[:, :, c])
#         negated_image[:, :, c] = (negated_image[:, :, c] - min_val) / (max_val - min_val + 1e-5)

#     # Convert back to 8-bit
#     negated_image = (negated_image * 255).astype(np.uint8)

#     # Print circuits for first 5 pixels
#     print("\nFirst 5 pixels transformation (original -> negated):")
#     h, w, _ = image.shape
#     count = 0
#     for i in range(h):
#         for j in range(w):
#             if count >= 5:
#                 break
#             original_pixel = image[i, j]
#             negated_pixel = negated_image[i, j]
#             print(f"Pixel ({i},{j}): {original_pixel.tolist()} -> {negated_pixel.tolist()}")
#             print("Quantum circuit for original pixel:")
#             qc = pixel_to_quantum_circuit(original_pixel.tolist())
#             print(qc.draw())
#             count += 1
#         if count >= 5:
#             break

#     return negated_image

# def display_images(original, transformed):
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.imshow(original)
#     plt.title("Original Image")
#     plt.axis('off')

#     plt.subplot(1, 2, 2)
#     plt.imshow(transformed)
#     plt.title("OCQR Negated Image")
#     plt.axis('off')

#     plt.show()

# if __name__ == "__main__":
#     image_path = "lena.jpeg"  # Replace with your image path
#     original_image = cv2.imread(image_path)
#     original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
#     negated = ocqr_negation_color_image(image_path)
#     display_images(original_image, negated)
import cv2
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
import time
from sklearn.metrics import mean_squared_error

def pixel_to_quantum_circuit(pixel_rgb):
    """
    Convert an RGB pixel to a quantum circuit representation.

    Parameters:
        pixel_rgb (list of int): RGB values [R, G, B] (0-255)

    Returns:
        QuantumCircuit: A quantum circuit representing the pixel's RGB.
    """
    # Normalize RGB values to range [0, 1]
    r, g, b = [val / 255.0 for val in pixel_rgb]

    # Create a 3-qubit circuit, one for each color channel
    qc = QuantumCircuit(3, name=f"RGB {pixel_rgb}")

    # Encode each channel into its qubit using Ry (rotation around Y-axis)
    qc.ry(r * np.pi, 0)  # R to qubit 0
    qc.ry(g * np.pi, 1)  # G to qubit 1
    qc.ry(b * np.pi, 2)  # B to qubit 2

    return qc

def calculate_mse(image1, image2):
    """
    Compute Mean Squared Error between two images.

    Parameters:
        image1 (np.ndarray): First image array.
        image2 (np.ndarray): Second image array.

    Returns:
        float: MSE value.
    """
    return mean_squared_error(image1.flatten(), image2.flatten())

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

    # Print circuits for first 5 pixels
    print("\nFirst 5 pixels transformation (original -> negated):")
    h, w, _ = image.shape
    count = 0
    for i in range(h):
        for j in range(w):
            if count >= 5:
                break
            original_pixel = image[i, j]
            negated_pixel = negated_image[i, j]
            print(f"Pixel ({i},{j}): {original_pixel.tolist()} -> {negated_pixel.tolist()}")
            print("Quantum circuit for original pixel:")
            qc = pixel_to_quantum_circuit(original_pixel.tolist())
            print(qc.draw())
            count += 1
        if count >= 5:
            break

    # Calculate MSE between original and negated images
    classical_negated = 255 - image
    classical_mse = calculate_mse(image, classical_negated)
    quantum_mse = calculate_mse(image, negated_image)

    print(f"\nClassical Negation MSE: {classical_mse:.4f}")
    print(f"Quantum (OCQR) Negation MSE: {quantum_mse:.4f}")

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
    start_time = time.time()

    image_path = "lena.jpeg"  # Replace with your image path
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    negated = ocqr_negation_color_image(image_path)
    display_images(original_image, negated)

    end_time = time.time()
    print(f"\nExecution Time: {end_time - start_time:.4f} seconds")
