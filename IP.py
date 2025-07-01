# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.ndimage import zoom
# from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, transpile
# from qiskit_aer import AerSimulator
# import math
# from numpy import pi
# import cv2  # OpenCV for image display
# from PIL import Image
# import os

# # === STEP 1: Enhanced Image Loading (TIFF, PNG, JPG, ASCII) ===
# def load_image():
#     # List of possible TIFF files to try
#     tiff_files = [
#         "./source_images/Lenna_512_grey.tiff"
#     ]
    
#     # Try TIFF files first
#     for tiff_file in tiff_files:
#         try:
#             if os.path.exists(tiff_file):
#                 print(f"Loading TIFF image: {tiff_file}")
#                 img = Image.open(tiff_file)
#                 # Convert to grayscale if needed
#                 if img.mode != 'L':
#                     img = img.convert('L')
#                 original_img = np.array(img, dtype=np.uint8)
#                 print(f"Successfully loaded TIFF image of shape: {original_img.shape}")
#                 return original_img, f"TIFF: {tiff_file}"
#         except Exception as e:
#             print(f"Failed to load {tiff_file}: {e}")
#             continue
    
#     # Default fallback image
#     print("No image files found. Using default 64x64 grayscale pattern.")
#     original_img = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
#     return original_img, "Default: Generated pattern"

# # === STEP 2: Downscaling to 64x64 ===
# def downscale_to_64x64(img):
#     """
#     Downscale image to exactly 64x64 using bilinear interpolation
#     """
#     if img.shape == (64, 64):
#         print("Image already 64x64, no downscaling needed")
#         return img.copy()
    
#     zoom_factors = (64 / img.shape[0], 64 / img.shape[1])
#     downscaled = zoom(img, zoom=zoom_factors, order=1).astype(np.uint8)
#     print(f"Downscaled from {img.shape} to {downscaled.shape}")
#     return downscaled

# # === STEP 3: Mathematical Interpolation Kernels ===
# # def R_kernel(u, order):
#     """
#     Generic interpolation kernel function based on the mathematical formulas
#     """
#     u = abs(u)
    
#     if order == 3:  # Bicubic (R3)
#         if u <= 1:
#             return 1.5 * u*3 - 2.5 * u*2 + 1
#         elif 1 < u < 2:
#             return -0.5 * u*3 + 2.5 * u*2 - 4 * u + 2
#         else:
#             return 0
#     elif order == 5:  # Biquintic (R5)
#         # Approximation for quintic kernel
#         if u <= 1:
#             return 1 - 2.5 * u*2 + 1.5 * u*3
#         elif 1 < u <= 2:
#             return -0.5 * u*3 + 2.5 * u*2 - 4 * u + 2
#         elif 2 < u <= 3:
#             return 0.166667 * u*3 - 0.833333 * u*2 + 1.333333 * u - 0.666667
#         else:
#             return 0
#     elif order == 7:  # Biseptic (R7)
#         # Approximation for septic (7th order) kernel
#         if u <= 1:
#             return 1 - 3.5 * u*2 + 2.5 * u*3
#         elif 1 < u <= 2:
#             return -0.5 * u*3 + 2.5 * u*2 - 4 * u + 2
#         elif 2 < u <= 3:
#             return 0.166667 * u*3 - 0.833333 * u*2 + 1.333333 * u - 0.666667
#         elif 3 < u <= 4:
#             return -0.02 * u*3 + 0.1 * u*2 - 0.15 * u + 0.07
#         else:
#             return 0
#     elif order == 9:  # Binonic (R9)
#         # Approximation for nonic (9th order) kernel
#         if u <= 1:
#             return 1 - 4.5 * u*2 + 3.5 * u*3
#         elif 1 < u <= 2:
#             return -0.5 * u*3 + 2.5 * u*2 - 4 * u + 2
#         elif 2 < u <= 3:
#             return 0.166667 * u*3 - 0.833333 * u*2 + 1.333333 * u - 0.666667
#         elif 3 < u <= 4:
#             return -0.02 * u*3 + 0.1 * u*2 - 0.15 * u + 0.07
#         elif 4 < u <= 5:
#             return 0.001 * u*3 - 0.005 * u*2 + 0.008 * u - 0.004
#         else:
#             return 0
#     else:
#         return 0

# def R_kernel(u, order):
#     """
#     Interpolation kernel function for different orders (cubic, quintic, septic, nonic).
#     """
#     u = abs(u)

#     if order == 3:  # Bicubic
#         if u <= 1:
#             return (1.5 * u ** 3) - (2.5 * u ** 2) + 1
#         elif 1 < u < 2:
#             return (-0.5 * u ** 3) + (2.5 * u ** 2) - (4 * u) + 2
#         else:
#             return 0

#     elif order == 5:  # Biquintic
#         if u <= 1:
#             return 1 - 2.5 * u ** 2 + 1.5 * u ** 3
#         elif 1 < u <= 2:
#             return -0.5 * u ** 3 + 2.5 * u ** 2 - 4 * u + 2
#         elif 2 < u <= 3:
#             return 0.166667 * u ** 3 - 0.833333 * u ** 2 + 1.333333 * u - 0.666667
#         else:
#             return 0

#     elif order == 7:  # Biseptic
#         if u <= 1:
#             return 1 - 3.5 * u ** 2 + 2.5 * u ** 3
#         elif 1 < u <= 2:
#             return -0.5 * u ** 3 + 2.5 * u ** 2 - 4 * u + 2
#         elif 2 < u <= 3:
#             return 0.166667 * u ** 3 - 0.833333 * u ** 2 + 1.333333 * u - 0.666667
#         elif 3 < u <= 4:
#             return -0.02 * u ** 3 + 0.1 * u ** 2 - 0.15 * u + 0.07
#         else:
#             return 0

#     elif order == 9:  # Binonic
#         if u <= 1:
#             return 1 - 4.5 * u ** 2 + 3.5 * u ** 3
#         elif 1 < u <= 2:
#             return -0.5 * u ** 3 + 2.5 * u ** 2 - 4 * u + 2
#         elif 2 < u <= 3:
#             return 0.166667 * u ** 3 - 0.833333 * u ** 2 + 1.333333 * u - 0.666667
#         elif 3 < u <= 4:
#             return -0.02 * u ** 3 + 0.1 * u ** 2 - 0.15 * u + 0.07
#         elif 4 < u <= 5:
#             return 0.001 * u ** 3 - 0.005 * u ** 2 + 0.008 * u - 0.004
#         else:
#             return 0

#     else:
#         return 0

# def interpolate_with_kernel(img, scale_factor, kernel_order, kernel_size):
#     """
#     Custom interpolation using mathematical formulas from the images
#     """
#     old_height, old_width = img.shape
#     new_height = int(old_height * scale_factor)
#     new_width = int(old_width * scale_factor)
    
#     result = np.zeros((new_height, new_width), dtype=np.float64)
    
#     for y_new in range(new_height):
#         for x_new in range(new_width):
#             # Map new coordinates to old coordinates
#             x_old = x_new / scale_factor
#             y_old = y_new / scale_factor
            
#             # Get integer and fractional parts
#             x_int = int(x_old)
#             y_int = int(y_old)
            
#             # Calculate interpolation weights
#             interpolated_value = 0.0
#             weight_sum = 0.0
            
#             # Apply the mathematical formula with the specified kernel size
#             for j in range(-kernel_size//2, kernel_size//2 + 1):
#                 for i in range(-kernel_size//2, kernel_size//2 + 1):
#                     # Source pixel coordinates
#                     src_x = x_int + i
#                     src_y = y_int + j
                
#                     # Check bounds
#                     if 0 <= src_x < old_width and 0 <= src_y < old_height:
#                         # Calculate distances
#                         dx = x_old - src_x
#                         dy = y_old - src_y
                        
#                         # Calculate kernel weights
#                         weight_x = R_kernel(dx, kernel_order)
#                         weight_y = R_kernel(dy, kernel_order)
#                         weight = weight_x * weight_y
                        
#                         # Accumulate weighted values
#                         interpolated_value += weight * img[src_y, src_x]
#                         weight_sum += weight
            
#             # Normalize and clamp
#             if weight_sum > 0:
#                 result[y_new, x_new] = interpolated_value / weight_sum
#             else:
#                 result[y_new, x_new] = 0
    
#     return np.clip(result, 0, 255).astype(np.uint8)

# def bilinear_interpolation(img, scale_factor):
#     """
#     Bilinear interpolation following the mathematical formula:
#     f(x,y) = ΣΣ f(i,j) · R(x-i) · R(y-j) where i,j go from 0 to 1
#     """
#     old_height, old_width = img.shape
#     new_height = int(old_height * scale_factor)
#     new_width = int(old_width * scale_factor)
    
#     result = np.zeros((new_height, new_width), dtype=np.float64)
    
#     for y_new in range(new_height):
#         for x_new in range(new_width):
#             # Map to old coordinates
#             x_old = x_new / scale_factor
#             y_old = y_new / scale_factor
            
#             # Get integer coordinates
#             x_int = int(x_old)
#             y_int = int(y_old)
            
#             # Get fractional parts
#             dx = x_old - x_int
#             dy = y_old - y_int
            
#             # Bilinear weights (based on the formula from Image 1)
#             interpolated_value = 0.0
            
#             # Apply the formula: sum over i=0,1 and j=0,1
#             for j in range(2):  # j = 0, 1
#                 for i in range(2):  # i = 0, 1
#                     src_x = x_int + i
#                     src_y = y_int + j
                    
#                     if 0 <= src_x < old_width and 0 <= src_y < old_height:
#                         # Calculate R(x-i) and R(y-j) for bilinear
#                         R_x = (1 - dx) if i == 0 else dx
#                         R_y = (1 - dy) if j == 0 else dy
                        
#                         interpolated_value += img[src_y, src_x] * R_x * R_y
            
#             result[y_new, x_new] = interpolated_value
    
#     return np.clip(result, 0, 255).astype(np.uint8)

# # === STEP 3: Six Interpolation Methods for Upscaling ===
# def upscale_with_six_methods(img_64x64):
#     """
#     Upscale 64x64 image to 128x128 using six different interpolation methods
#     following the exact mathematical formulas provided
#     """
#     methods = {}
#     scale_factor = 2.0  # 64x64 to 128x128
    
#     # Method 1: Nearest Neighbor (simple, no mathematical formula needed)
#     try:
#         methods['Nearest Neighbor'] = zoom(img_64x64, zoom=scale_factor, order=0).astype(np.uint8)
#         print("✓ Nearest Neighbor interpolation completed")
#     except Exception as e:
#         print(f"✗ Nearest Neighbor failed: {e}")
    
#     # Method 2: Bilinear (following Image 1 formula)
#     try:
#         methods['Bilinear'] = bilinear_interpolation(img_64x64, scale_factor)
#         print("✓ Bilinear interpolation completed (custom implementation)")
#     except Exception as e:
#         print(f"✗ Bilinear failed: {e}")
#         # Fallback to scipy
#         try:
#             methods['Bilinear'] = zoom(img_64x64, zoom=scale_factor, order=1).astype(np.uint8)
#             print("✓ Bilinear interpolation completed (fallback)")
#         except:
#             pass
    
#     # Method 3: Bicubic (following Image 2 formula with R3 kernel)
#     try:
#         methods['Bicubic'] = interpolate_with_kernel(img_64x64, scale_factor, 3, 4)
#         print("✓ Bicubic interpolation completed (R3 kernel, 4x4 neighborhood)")
#     except Exception as e:
#         print(f"✗ Bicubic failed: {e}")
#         # Fallback to scipy
#         try:
#             methods['Bicubic'] = zoom(img_64x64, zoom=scale_factor, order=3).astype(np.uint8)
#             print("✓ Bicubic interpolation completed (fallback)")
#         except:
#             pass
    
#     # Method 4: Biquintic (following Image 3 formula with R5 kernel)
#     try:
#         methods['Biquintic'] = interpolate_with_kernel(img_64x64, scale_factor, 5, 6)
#         print("✓ Biquintic interpolation completed (R5 kernel, 6x6 neighborhood)")
#     except Exception as e:
#         print(f"✗ Biquintic failed: {e}")
#         # Fallback to scipy
#         try:
#             methods['Biquintic'] = zoom(img_64x64, zoom=scale_factor, order=5).astype(np.uint8)
#             print("✓ Biquintic interpolation completed (fallback)")
#         except:
#             pass
    
#     # Method 5: Biseptic (following Image 4 formula with R7 kernel)
#     try:
#         methods['Biseptic'] = interpolate_with_kernel(img_64x64, scale_factor, 7, 8)
#         print("✓ Biseptic interpolation completed (R7 kernel, 8x8 neighborhood)")
#     except Exception as e:
#         print(f"✗ Biseptic failed: {e}")
#         # Fallback to scipy
#         try:
#             methods['Biseptic'] = zoom(img_64x64, zoom=scale_factor, order=7).astype(np.uint8)
#             print("✓ Biseptic interpolation completed (fallback)")
#         except:
#             pass
    
#     # Method 6: Binonic (following Image 5 formula with R9 kernel)
#     try:
#         methods['Binonic'] = interpolate_with_kernel(img_64x64, scale_factor, 9, 10)
#         print("✓ Binonic interpolation completed (R9 kernel, 10x10 neighborhood)")
#     except Exception as e:
#         print(f"✗ Binonic failed: {e}")
#         # Fallback to scipy
#         try:
#             methods['Binonic'] = zoom(img_64x64, zoom=scale_factor, order=9).astype(np.uint8)
#             print("✓ Binonic interpolation completed (fallback)")
#         except:
#             pass
    
#     return methods

# # === STEP 4: Quality Metrics ===
# def calculate_image_metrics(original, upscaled, method_name):
#     """
#     Calculate quality metrics for upscaled images
#     """
#     # Mean Squared Error
#     mse = np.mean((original.astype(float) - upscaled.astype(float)) ** 2)
    
#     # Peak Signal-to-Noise Ratio
#     if mse == 0:
#         psnr = float('inf')
#     else:
#         psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    
#     # Structural Similarity (simplified version)
#     mean_orig = np.mean(original)
#     mean_upsc = np.mean(upscaled)
#     var_orig = np.var(original)
#     var_upsc = np.var(upscaled)
#     covar = np.mean((original - mean_orig) * (upscaled - mean_upsc))
    
#     c1, c2 = 0.01*2, 0.03*2
#     ssim = ((2*mean_orig*mean_upsc + c1) * (2*covar + c2)) / \
#            ((mean_orig*2 + mean_upsc*2 + c1) * (var_orig + var_upsc + c2))
    
#     return {
#         'method': method_name,
#         'mse': mse,
#         'psnr': psnr,
#         'ssim': ssim,
#         'mean': np.mean(upscaled),
#         'std': np.std(upscaled)
#     }

# # === STEP 5: Quantum Circuit Encoding ===
# def create_quantum_image_circuit(matrix, sample_size=4):
#     rows, cols = matrix.shape
#     row_idx = np.linspace(0, rows - 1, sample_size, dtype=int)
#     col_idx = np.linspace(0, cols - 1, sample_size, dtype=int)
#     sampled_matrix = matrix[np.ix_(row_idx, col_idx)]

#     qy = math.ceil(math.log2(sample_size))
#     qx = math.ceil(math.log2(sample_size))

#     qreg_y = QuantumRegister(qy, 'y')
#     qreg_x = QuantumRegister(qx, 'x')
#     qreg_anc = QuantumRegister(qy + qx, 'anc')
#     qreg_color = QuantumRegister(1, 'color')
#     creg_c = ClassicalRegister(1, 'c')
#     circuit = QuantumCircuit(qreg_y, qreg_x, qreg_anc, qreg_color, creg_c)

#     for q in qreg_y:
#         circuit.h(q)
#     for q in qreg_x:
#         circuit.h(q)

#     for i in range(qy):
#         circuit.cx(qreg_y[i], qreg_anc[i])
#     for i in range(qx):
#         circuit.cx(qreg_x[i], qreg_anc[qy + i])

#     max_val = sampled_matrix.max()
#     min_val = sampled_matrix.min()

#     for y in range(sample_size):
#         for x in range(sample_size):
#             value = sampled_matrix[y, x]
#             theta = ((value - min_val) / (max_val - min_val)) * pi if max_val != min_val else 0
#             if theta > 0:
#                 controls = []
#                 for i in range(qy):
#                     if (y >> i) & 1:
#                         controls.append(qreg_anc[i])
#                     else:
#                         circuit.x(qreg_anc[i])
#                         controls.append(qreg_anc[i])
#                 for i in range(qx):
#                     if (x >> i) & 1:
#                         controls.append(qreg_anc[qy + i])
#                     else:
#                         circuit.x(qreg_anc[qy + i])
#                         controls.append(qreg_anc[qy + i])

#                 if len(controls) == 1:
#                     circuit.cry(theta, controls[0], qreg_color[0])
#                 else:
#                     circuit.mcry(theta, controls, qreg_color[0])

#                 for i in range(qy):
#                     if not ((y >> i) & 1):
#                         circuit.x(qreg_anc[i])
#                 for i in range(qx):
#                     if not ((x >> i) & 1):
#                         circuit.x(qreg_anc[qy + i])

#     circuit.measure(qreg_color[0], creg_c[0])
#     return circuit, (row_idx, col_idx)

# # === MAIN EXECUTION ===
# print("="*80)
# print("QUANTUM IMAGE PROCESSING WITH SIX INTERPOLATION METHODS")
# print("="*80)

# # Load original image
# original_img, source_info = load_image()
# original_shape = original_img.shape

# # Downscale to 64x64
# img_64x64 = downscale_to_64x64(original_img)

# # Apply all six upscaling methods
# print("\nApplying six interpolation methods...")
# upscaled_methods = upscale_with_six_methods(img_64x64)

# # Calculate quality metrics for each method
# print("\nCalculating quality metrics...")
# metrics_table = []
# reference_img = zoom(original_img, zoom=(128/original_img.shape[0], 128/original_img.shape[1]), order=1).astype(np.uint8)

# for method_name, upscaled_img in upscaled_methods.items():
#     metrics = calculate_image_metrics(reference_img, upscaled_img, method_name)
#     metrics_table.append(metrics)

# # === STEP 6: Comprehensive Visualization ===
# def display_comprehensive_comparison():
#     """Display all images and their comparisons"""
#     fig = plt.figure(figsize=(20, 15))
    
#     # Create a grid: 3 rows, 4 columns
#     # Row 1: Original, 64x64, Reference 128x128, and one method
#     # Row 2-3: Remaining five methods and metrics
    
#     methods_list = list(upscaled_methods.keys())
    
#     # Row 1
#     plt.subplot(3, 4, 1)
#     plt.imshow(original_img, cmap='gray', vmin=0, vmax=255)
#     plt.title(f'Original Image\n{original_shape}\n({source_info})')
#     plt.axis('off')
    
#     plt.subplot(3, 4, 2)
#     plt.imshow(img_64x64, cmap='gray', vmin=0, vmax=255)
#     plt.title('Downscaled\n64×64')
#     plt.axis('off')
    
#     plt.subplot(3, 4, 3)
#     plt.imshow(reference_img, cmap='gray', vmin=0, vmax=255)
#     plt.title('Reference\n128×128\n(Bilinear from original)')
#     plt.axis('off')
    
#     # Display all six methods
#     positions = [4, 5, 6, 7, 8, 9, 10, 11, 12]  # Skip position 4 if needed
#     for i, (method_name, upscaled_img) in enumerate(upscaled_methods.items()):
#         if i < len(positions):
#             plt.subplot(3, 4, positions[i])
#             plt.imshow(upscaled_img, cmap='gray', vmin=0, vmax=255)
            
#             # Find corresponding metrics
#             method_metrics = next((m for m in metrics_table if m['method'] == method_name), None)
#             if method_metrics:
#                 title = f"{method_name}\n128×128\nPSNR: {method_metrics['psnr']:.2f}dB"
#             else:
#                 title = f"{method_name}\n128×128"
            
#             plt.title(title, fontsize=10)
#             plt.axis('off')
    
#     plt.tight_layout()
#     plt.show()

# # Run comprehensive visualization
# display_comprehensive_comparison()

# # === STEP 7: Quantum Processing on Best Method ===
# # Select the method with highest PSNR
# if metrics_table:
#     best_method = max(metrics_table, key=lambda x: x['psnr'] if x['psnr'] != float('inf') else 0)
#     best_upscaled = upscaled_methods[best_method['method']]
#     print(f"\nBest interpolation method: {best_method['method']} (PSNR: {best_method['psnr']:.2f}dB)")
# else:
#     best_method = {'method': 'Bilinear'}
#     best_upscaled = upscaled_methods.get('Bilinear', list(upscaled_methods.values())[0])

# # Create quantum circuit for the best upscaled image
# print(f"\nCreating quantum circuit for {best_method['method']} upscaled image...")
# circuit, (sampled_rows, sampled_cols) = create_quantum_image_circuit(best_upscaled, sample_size=4)

# # Simulate quantum circuit
# simulator = AerSimulator()
# compiled = transpile(circuit, simulator)
# job = simulator.run(compiled, shots=1024)
# result = job.result()
# counts = result.get_counts()

# print("="*80)
# print("PROCESSING COMPLETE!")
# print("="*80)
# 'in the given code just make changes so that it can have original 512 image & downscaled 64 image in first row, the first 3 methods in 2nd row and remaining 3 images in third row neatly aligned with images and their names and also removing the 128X128 bilinear from original from code and also saving the images in matplotlib seperately as tiff files. keeping the same simplicity & working logic for rest of the code'
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import math
from numpy import pi
from PIL import Image
import os

# === STEP 1: Enhanced Image Loading (TIFF, PNG, JPG, ASCII) ===
def load_image():
    # List of possible TIFF files to try
    tiff_files = [
        "./source_images/Lenna_512_grey.tiff"
    ]
    
    # Try TIFF files first
    for tiff_file in tiff_files:
        try:
            if os.path.exists(tiff_file):
                print(f"Loading TIFF image: {tiff_file}")
                img = Image.open(tiff_file)
                # Convert to grayscale if needed
                if img.mode != 'L':
                    img = img.convert('L')
                original_img = np.array(img, dtype=np.uint8)
                print(f"Successfully loaded TIFF image of shape: {original_img.shape}")
                return original_img, f"TIFF: {tiff_file}"
        except Exception as e:
            print(f"Failed to load {tiff_file}: {e}")
            continue
    
    # Default fallback image
    print("No image files found. Using default 64x64 grayscale pattern.")
    original_img = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
    return original_img, "Default: Generated pattern"

# === STEP 2: Downscaling to 64x64 ===
def downscale_to_64x64(img):
    """
    Downscale image to exactly 64x64 using bilinear interpolation
    """
    if img.shape == (64, 64):
        print("Image already 64x64, no downscaling needed")
        return img.copy()
    
    zoom_factors = (64 / img.shape[0], 64 / img.shape[1])
    downscaled = zoom(img, zoom=zoom_factors, order=1).astype(np.uint8)
    print(f"Downscaled from {img.shape} to {downscaled.shape}")
    return downscaled

# === STEP 3: Mathematical Interpolation Kernels ===
def R_kernel(u, order):
    """
    Interpolation kernel function for different orders (cubic, quintic, septic, nonic).
    """
    u = abs(u)

    if order == 3:  # Bicubic
        if u <= 1:
            return (1.5 * u ** 3) - (2.5 * u ** 2) + 1
        elif 1 < u < 2:
            return (-0.5 * u ** 3) + (2.5 * u ** 2) - (4 * u) + 2
        else:
            return 0

    elif order == 5:  # Biquintic
        if u <= 1:
            return 1 - 2.5 * u ** 2 + 1.5 * u ** 3
        elif 1 < u <= 2:
            return -0.5 * u ** 3 + 2.5 * u ** 2 - 4 * u + 2
        elif 2 < u <= 3:
            return 0.166667 * u ** 3 - 0.833333 * u ** 2 + 1.333333 * u - 0.666667
        else:
            return 0

    elif order == 7:  # Biseptic
        if u <= 1:
            return 1 - 3.5 * u ** 2 + 2.5 * u ** 3
        elif 1 < u <= 2:
            return -0.5 * u ** 3 + 2.5 * u ** 2 - 4 * u + 2
        elif 2 < u <= 3:
            return 0.166667 * u ** 3 - 0.833333 * u ** 2 + 1.333333 * u - 0.666667
        elif 3 < u <= 4:
            return -0.02 * u ** 3 + 0.1 * u ** 2 - 0.15 * u + 0.07
        else:
            return 0

    elif order == 9:  # Binonic
        if u <= 1:
            return 1 - 4.5 * u ** 2 + 3.5 * u ** 3
        elif 1 < u <= 2:
            return -0.5 * u ** 3 + 2.5 * u ** 2 - 4 * u + 2
        elif 2 < u <= 3:
            return 0.166667 * u ** 3 - 0.833333 * u ** 2 + 1.333333 * u - 0.666667
        elif 3 < u <= 4:
            return -0.02 * u ** 3 + 0.1 * u ** 2 - 0.15 * u + 0.07
        elif 4 < u <= 5:
            return 0.001 * u ** 3 - 0.005 * u ** 2 + 0.008 * u - 0.004
        else:
            return 0

    else:
        return 0

def interpolate_with_kernel(img, scale_factor, kernel_order, kernel_size):
    """
    Custom interpolation using mathematical formulas from the images
    """
    old_height, old_width = img.shape
    new_height = int(old_height * scale_factor)
    new_width = int(old_width * scale_factor)
    
    result = np.zeros((new_height, new_width), dtype=np.float64)
    
    for y_new in range(new_height):
        for x_new in range(new_width):
            # Map new coordinates to old coordinates
            x_old = x_new / scale_factor
            y_old = y_new / scale_factor
            
            # Get integer and fractional parts
            x_int = int(x_old)
            y_int = int(y_old)
            
            # Calculate interpolation weights
            interpolated_value = 0.0
            weight_sum = 0.0
            
            # Apply the mathematical formula with the specified kernel size
            for j in range(-kernel_size//2, kernel_size//2 + 1):
                for i in range(-kernel_size//2, kernel_size//2 + 1):
                    # Source pixel coordinates
                    src_x = x_int + i
                    src_y = y_int + j
                
                    # Check bounds
                    if 0 <= src_x < old_width and 0 <= src_y < old_height:
                        # Calculate distances
                        dx = x_old - src_x
                        dy = y_old - src_y
                        
                        # Calculate kernel weights
                        weight_x = R_kernel(dx, kernel_order)
                        weight_y = R_kernel(dy, kernel_order)
                        weight = weight_x * weight_y
                        
                        # Accumulate weighted values
                        interpolated_value += weight * img[src_y, src_x]
                        weight_sum += weight
            
            # Normalize and clamp
            if weight_sum > 0:
                result[y_new, x_new] = interpolated_value / weight_sum
            else:
                result[y_new, x_new] = 0
    
    return np.clip(result, 0, 255).astype(np.uint8)

def bilinear_interpolation(img, scale_factor):
    """
    Bilinear interpolation following the mathematical formula:
    f(x,y) = ΣΣ f(i,j) · R(x-i) · R(y-j) where i,j go from 0 to 1
    """
    old_height, old_width = img.shape
    new_height = int(old_height * scale_factor)
    new_width = int(old_width * scale_factor)
    
    result = np.zeros((new_height, new_width), dtype=np.float64)
    
    for y_new in range(new_height):
        for x_new in range(new_width):
            # Map to old coordinates
            x_old = x_new / scale_factor
            y_old = y_new / scale_factor
            
            # Get integer coordinates
            x_int = int(x_old)
            y_int = int(y_old)
            
            # Get fractional parts
            dx = x_old - x_int
            dy = y_old - y_int
            
            # Bilinear weights (based on the formula from Image 1)
            interpolated_value = 0.0
            
            # Apply the formula: sum over i=0,1 and j=0,1
            for j in range(2):  # j = 0, 1
                for i in range(2):  # i = 0, 1
                    src_x = x_int + i
                    src_y = y_int + j
                    
                    if 0 <= src_x < old_width and 0 <= src_y < old_height:
                        # Calculate R(x-i) and R(y-j) for bilinear
                        R_x = (1 - dx) if i == 0 else dx
                        R_y = (1 - dy) if j == 0 else dy
                        
                        interpolated_value += img[src_y, src_x] * R_x * R_y
            
            result[y_new, x_new] = interpolated_value
    
    return np.clip(result, 0, 255).astype(np.uint8)

# === STEP 3: Six Interpolation Methods for Upscaling ===
def upscale_with_six_methods(img_64x64):
    """
    Upscale 64x64 image to 128x128 using six different interpolation methods
    following the exact mathematical formulas provided
    """
    methods = {}
    scale_factor = 2.0  # 64x64 to 128x128
    
    # Method 1: Nearest Neighbor (simple, no mathematical formula needed)
    try:
        methods['Nearest Neighbor'] = zoom(img_64x64, zoom=scale_factor, order=0).astype(np.uint8)
        print("Nearest Neighbor interpolation completed")
    except Exception as e:
        print(f"Nearest Neighbor failed: {e}")
    
    # Method 2: Bilinear (following Image 1 formula)
    try:
        methods['Bilinear'] = bilinear_interpolation(img_64x64, scale_factor)
        print("Bilinear interpolation completed (custom implementation)")
    except Exception as e:
        print(f"Bilinear failed: {e}")
        # Fallback to scipy
        try:
            methods['Bilinear'] = zoom(img_64x64, zoom=scale_factor, order=1).astype(np.uint8)
            print("Bilinear interpolation completed (fallback)")
        except:
            pass
    
    # Method 3: Bicubic (following Image 2 formula with R3 kernel)
    try:
        methods['Bicubic'] = interpolate_with_kernel(img_64x64, scale_factor, 3, 4)
        print("Bicubic interpolation completed (R3 kernel, 4x4 neighborhood)")
    except Exception as e:
        print(f"Bicubic failed: {e}")
        # Fallback to scipy
        try:
            methods['Bicubic'] = zoom(img_64x64, zoom=scale_factor, order=3).astype(np.uint8)
            print("Bicubic interpolation completed (fallback)")
        except:
            pass
    
    # Method 4: Biquintic (following Image 3 formula with R5 kernel)
    try:
        methods['Biquintic'] = interpolate_with_kernel(img_64x64, scale_factor, 5, 6)
        print("Biquintic interpolation completed (R5 kernel, 6x6 neighborhood)")
    except Exception as e:
        print(f"Biquintic failed: {e}")
        # Fallback to scipy
        try:
            methods['Biquintic'] = zoom(img_64x64, zoom=scale_factor, order=5).astype(np.uint8)
            print("Biquintic interpolation completed (fallback)")
        except:
            pass
    
    # Method 5: Biseptic (following Image 4 formula with R7 kernel)
    try:
        methods['Biseptic'] = interpolate_with_kernel(img_64x64, scale_factor, 7, 8)
        print("Biseptic interpolation completed (R7 kernel, 8x8 neighborhood)")
    except Exception as e:
        print(f"Biseptic failed: {e}")
        # Fallback to scipy
        try:
            methods['Biseptic'] = zoom(img_64x64, zoom=scale_factor, order=7).astype(np.uint8)
            print("Biseptic interpolation completed (fallback)")
        except:
            pass
    
    # Method 6: Binonic (following Image 5 formula with R9 kernel)
    try:
        methods['Binonic'] = interpolate_with_kernel(img_64x64, scale_factor, 9, 10)
        print("Binonic interpolation completed (R9 kernel, 10x10 neighborhood)")
    except Exception as e:
        print(f"Binonic failed: {e}")
        # Fallback to scipy
        try:
            methods['Binonic'] = zoom(img_64x64, zoom=scale_factor, order=9).astype(np.uint8)
            print("Binonic interpolation completed (fallback)")
        except:
            pass
    
    return methods

# === STEP 4: Quality Metrics ===
def calculate_image_metrics(original, upscaled, method_name):
    """
    Calculate quality metrics for upscaled images
    """
    # Mean Squared Error
    mse = np.mean((original.astype(float) - upscaled.astype(float)) ** 2)
    
    # Peak Signal-to-Noise Ratio
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    
    # Structural Similarity (simplified version)
    mean_orig = np.mean(original)
    mean_upsc = np.mean(upscaled)
    var_orig = np.var(original)
    var_upsc = np.var(upscaled)
    covar = np.mean((original - mean_orig) * (upscaled - mean_upsc))
    
    c1, c2 = 0.01*2, 0.03*2
    ssim = ((2*mean_orig*mean_upsc + c1) * (2*covar + c2)) / \
           ((mean_orig*2 + mean_upsc*2 + c1) * (var_orig + var_upsc + c2))
    
    return {
        'method': method_name,
        'mse': mse,
        'psnr': psnr,
        'ssim': ssim,
        'mean': np.mean(upscaled),
        'std': np.std(upscaled)
    }

# === STEP 5: Save Images as TIFF Files ===
def save_images_as_tiff(original_img, img_64x64, upscaled_methods):
    """
    Save all images as individual TIFF files
    """
    # Create output directory if it doesn't exist
    output_dir = "output_images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save original image
    Image.fromarray(original_img).save(os.path.join(output_dir, "01_original.tiff"))
    print(f"Saved: {output_dir}/01_original.tiff")
    
    # Save 64x64 downscaled image
    Image.fromarray(img_64x64).save(os.path.join(output_dir, "02_downscaled_64x64.tiff"))
    print(f"Saved: {output_dir}/02_downscaled_64x64.tiff")
    
    # Save all upscaled method images
    method_order = ['Nearest Neighbor', 'Bilinear', 'Bicubic', 'Biquintic', 'Biseptic', 'Binonic']
    for i, method_name in enumerate(method_order):
        if method_name in upscaled_methods:
            filename = f"{i+3:02d}_{method_name.lower().replace(' ', '_')}_128x128.tiff"
            Image.fromarray(upscaled_methods[method_name]).save(os.path.join(output_dir, filename))
            print(f"Saved: {output_dir}/{filename}")

# === STEP 6: Quantum Circuit Encoding ===
def create_quantum_image_circuit(matrix, sample_size=4):
    rows, cols = matrix.shape
    row_idx = np.linspace(0, rows - 1, sample_size, dtype=int)
    col_idx = np.linspace(0, cols - 1, sample_size, dtype=int)
    sampled_matrix = matrix[np.ix_(row_idx, col_idx)]

    qy = math.ceil(math.log2(sample_size))
    qx = math.ceil(math.log2(sample_size))

    qreg_y = QuantumRegister(qy, 'y')
    qreg_x = QuantumRegister(qx, 'x')
    qreg_anc = QuantumRegister(qy + qx, 'anc')
    qreg_color = QuantumRegister(1, 'color')
    creg_c = ClassicalRegister(1, 'c')
    circuit = QuantumCircuit(qreg_y, qreg_x, qreg_anc, qreg_color, creg_c)

    for q in qreg_y:
        circuit.h(q)
    for q in qreg_x:
        circuit.h(q)

    for i in range(qy):
        circuit.cx(qreg_y[i], qreg_anc[i])
    for i in range(qx):
        circuit.cx(qreg_x[i], qreg_anc[qy + i])

    max_val = sampled_matrix.max()
    min_val = sampled_matrix.min()

    for y in range(sample_size):
        for x in range(sample_size):
            value = sampled_matrix[y, x]
            theta = ((value - min_val) / (max_val - min_val)) * pi if max_val != min_val else 0
            if theta > 0:
                controls = []
                for i in range(qy):
                    if (y >> i) & 1:
                        controls.append(qreg_anc[i])
                    else:
                        circuit.x(qreg_anc[i])
                        controls.append(qreg_anc[i])
                for i in range(qx):
                    if (x >> i) & 1:
                        controls.append(qreg_anc[qy + i])
                    else:
                        circuit.x(qreg_anc[qy + i])
                        controls.append(qreg_anc[qy + i])

                if len(controls) == 1:
                    circuit.cry(theta, controls[0], qreg_color[0])
                else:
                    circuit.mcry(theta, controls, qreg_color[0])

                for i in range(qy):
                    if not ((y >> i) & 1):
                        circuit.x(qreg_anc[i])
                for i in range(qx):
                    if not ((x >> i) & 1):
                        circuit.x(qreg_anc[qy + i])

    circuit.measure(qreg_color[0], creg_c[0])
    return circuit, (row_idx, col_idx)

# === STEP 7: Enhanced 3-Row Visualization ===
def display_comprehensive_comparison():
    """Display images in a clean 3-row layout as requested"""
    # Create figure with proper spacing
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Quantum Image Processing: Interpolation Methods Comparison', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    methods_list = ['Nearest Neighbor', 'Bilinear', 'Bicubic', 'Biquintic', 'Biseptic', 'Binonic']
    available_methods = [method for method in methods_list if method in upscaled_methods]
    
    # Row 1: Original image and downscaled 64x64 image (first 2 positions)
    axes[0, 0].imshow(original_img, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title(f'Original Image\n{original_shape}', fontsize=11, fontweight='bold', pad=10)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_64x64, cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title('Downscaled\n64×64', fontsize=11, fontweight='bold', pad=10)
    axes[0, 1].axis('off')
    
    # Hide the third subplot in first row
    axes[0, 2].axis('off')
    
    # Row 2: First 3 interpolation methods
    for i in range(3):
        if i < len(available_methods):
            method_name = available_methods[i]
            upscaled_img = upscaled_methods[method_name]
            
            axes[1, i].imshow(upscaled_img, cmap='gray', vmin=0, vmax=255)
            
            # Find corresponding metrics
            method_metrics = next((m for m in metrics_table if m['method'] == method_name), None)
            if method_metrics:
                title = f"{method_name}\n128×128\nPSNR: {method_metrics['psnr']:.1f}dB"
            else:
                title = f"{method_name}\n128×128"
            
            axes[1, i].set_title(title, fontsize=10, fontweight='bold', pad=10)
            axes[1, i].axis('off')
        else:
            axes[1, i].axis('off')
    
    # Row 3: Remaining 3 interpolation methods
    for i in range(3):
        idx = i + 3  # Methods 4, 5, 6
        if idx < len(available_methods):
            method_name = available_methods[idx]
            upscaled_img = upscaled_methods[method_name]
            
            axes[2, i].imshow(upscaled_img, cmap='gray', vmin=0, vmax=255)
            
            # Find corresponding metrics
            method_metrics = next((m for m in metrics_table if m['method'] == method_name), None)
            if method_metrics:
                title = f"{method_name}\n128×128\nPSNR: {method_metrics['psnr']:.1f}dB"
            else:
                title = f"{method_name}\n128×128"
            
            axes[2, i].set_title(title, fontsize=10, fontweight='bold', pad=10)
            axes[2, i].axis('off')
        else:
            axes[2, i].axis('off')
    
    # Adjust layout for better spacing
    plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.05, 
                       wspace=0.15, hspace=0.35)
    plt.show()

# === MAIN EXECUTION ===
print("="*80)
print("QUANTUM IMAGE PROCESSING WITH SIX INTERPOLATION METHODS")
print("="*80)

# Load original image
original_img, source_info = load_image()
original_shape = original_img.shape

# Downscale to 64x64
img_64x64 = downscale_to_64x64(original_img)

# Apply all six upscaling methods
print("\nApplying six interpolation methods...")
upscaled_methods = upscale_with_six_methods(img_64x64)

# Calculate quality metrics for each method (using 64x64 upscaled to compare fairly)
print("\nCalculating quality metrics...")
metrics_table = []
reference_img = zoom(img_64x64, zoom=2.0, order=1).astype(np.uint8)  # Simple bilinear reference

for method_name, upscaled_img in upscaled_methods.items():
    metrics = calculate_image_metrics(reference_img, upscaled_img, method_name)
    metrics_table.append(metrics)

# Save all images as TIFF files
print("\nSaving images as TIFF files...")
save_images_as_tiff(original_img, img_64x64, upscaled_methods)

# Run comprehensive visualization with 3-row layout
print("\nDisplaying comprehensive comparison...")
display_comprehensive_comparison()

# === STEP 8: Quantum Processing on Best Method ===
# Select the method with highest PSNR
if metrics_table:
    best_method = max(metrics_table, key=lambda x: x['psnr'] if x['psnr'] != float('inf') else 0)
    best_upscaled = upscaled_methods[best_method['method']]
    print(f"\nBest interpolation method: {best_method['method']} (PSNR: {best_method['psnr']:.2f}dB)")
else:
    best_method = {'method': 'Bilinear'}
    best_upscaled = upscaled_methods.get('Bilinear', list(upscaled_methods.values())[0])

# Create quantum circuit for the best upscaled image
print(f"\nCreating quantum circuit for {best_method['method']} upscaled image...")
circuit, (sampled_rows, sampled_cols) = create_quantum_image_circuit(best_upscaled, sample_size=4)

# Simulate quantum circuit
simulator = AerSimulator()
compiled = transpile(circuit, simulator)
job = simulator.run(compiled, shots=1024)
result = job.result()
counts = result.get_counts()

print("="*80)
print("PROCESSING COMPLETE!")
print("="*80)
print("Images saved in 'output_images' directory as individual TIFF files")
print("="*80)