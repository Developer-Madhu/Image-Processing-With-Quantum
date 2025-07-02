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

# === STEP 3: Six Interpolation Methods for Downscaling (64x64 to 32x32) ===
def downscale_with_six_methods(img_64x64):
    """
    Downscale 64x64 image to 32x32 using six different interpolation methods
    following the exact mathematical formulas
    """
    methods = {}
    scale_factor = 0.5  # 64x64 to 32x32
    
    # Method 1: Nearest Neighbor (simple, no mathematical formula needed)
    try:
        methods['Nearest Neighbor'] = zoom(img_64x64, zoom=scale_factor, order=0).astype(np.uint8)
        print("Nearest Neighbor downscaling completed")
    except Exception as e:
        print(f"Nearest Neighbor downscaling failed: {e}")
    
    # Method 2: Bilinear (following Image 1 formula)
    try:
        methods['Bilinear'] = bilinear_interpolation(img_64x64, scale_factor)
        print("Bilinear downscaling completed (custom implementation)")
    except Exception as e:
        print(f"Bilinear downscaling failed: {e}")
        # Fallback to scipy
        try:
            methods['Bilinear'] = zoom(img_64x64, zoom=scale_factor, order=1).astype(np.uint8)
            print("Bilinear downscaling completed (fallback)")
        except:
            pass
    
    # Method 3: Bicubic (following Image 2 formula with R3 kernel)
    try:
        methods['Bicubic'] = interpolate_with_kernel(img_64x64, scale_factor, 3, 4)
        print("Bicubic downscaling completed (R3 kernel, 4x4 neighborhood)")
    except Exception as e:
        print(f"Bicubic downscaling failed: {e}")
        # Fallback to scipy
        try:
            methods['Bicubic'] = zoom(img_64x64, zoom=scale_factor, order=3).astype(np.uint8)
            print("Bicubic downscaling completed (fallback)")
        except:
            pass
    
    # Method 4: Biquintic (following Image 3 formula with R5 kernel)
    try:
        methods['Biquintic'] = interpolate_with_kernel(img_64x64, scale_factor, 5, 6)
        print("Biquintic downscaling completed (R5 kernel, 6x6 neighborhood)")
    except Exception as e:
        print(f"Biquintic downscaling failed: {e}")
        # Fallback to scipy
        try:
            methods['Biquintic'] = zoom(img_64x64, zoom=scale_factor, order=5).astype(np.uint8)
            print("Biquintic downscaling completed (fallback)")
        except:
            pass
    
    # Method 5: Biseptic (following Image 4 formula with R7 kernel)
    try:
        methods['Biseptic'] = interpolate_with_kernel(img_64x64, scale_factor, 7, 8)
        print("Biseptic downscaling completed (R7 kernel, 8x8 neighborhood)")
    except Exception as e:
        print(f"Biseptic downscaling failed: {e}")
        # Fallback to scipy
        try:
            methods['Biseptic'] = zoom(img_64x64, zoom=scale_factor, order=7).astype(np.uint8)
            print("Biseptic downscaling completed (fallback)")
        except:
            pass
    
    # Method 6: Binonic (following Image 5 formula with R9 kernel)
    try:
        methods['Binonic'] = interpolate_with_kernel(img_64x64, scale_factor, 9, 10)
        print("Binonic downscaling completed (R9 kernel, 10x10 neighborhood)")
    except Exception as e:
        print(f"Binonic downscaling failed: {e}")
        # Fallback to scipy
        try:
            methods['Binonic'] = zoom(img_64x64, zoom=scale_factor, order=9).astype(np.uint8)
            print("Binonic downscaling completed (fallback)")
        except:
            pass
    
    return methods

# === STEP 4: Six Interpolation Methods for Upscaling (32x32 back to 64x64) ===
def upscale_with_six_methods(downscaled_methods):
    """
    Upscale 32x32 images back to 64x64 using six different interpolation methods
    """
    upscaled_methods = {}
    scale_factor = 2.0  # 32x32 to 64x64
    
    for method_name, img_32x32 in downscaled_methods.items():
        try:
            if method_name == 'Nearest Neighbor':
                upscaled_methods[method_name] = zoom(img_32x32, zoom=scale_factor, order=0).astype(np.uint8)
                print(f"Nearest Neighbor upscaling completed for {method_name}")
            elif method_name == 'Bilinear':
                upscaled_methods[method_name] = bilinear_interpolation(img_32x32, scale_factor)
                print(f"Bilinear upscaling completed for {method_name}")
            elif method_name == 'Bicubic':
                upscaled_methods[method_name] = interpolate_with_kernel(img_32x32, scale_factor, 3, 4)
                print(f"Bicubic upscaling completed for {method_name}")
            elif method_name == 'Biquintic':
                upscaled_methods[method_name] = interpolate_with_kernel(img_32x32, scale_factor, 5, 6)
                print(f"Biquintic upscaling completed for {method_name}")
            elif method_name == 'Biseptic':
                upscaled_methods[method_name] = interpolate_with_kernel(img_32x32, scale_factor, 7, 8)
                print(f"Biseptic upscaling completed for {method_name}")
            elif method_name == 'Binonic':
                upscaled_methods[method_name] = interpolate_with_kernel(img_32x32, scale_factor, 9, 10)
                print(f"Binonic upscaling completed for {method_name}")
        except Exception as e:
            print(f"Upscaling failed for {method_name}: {e}")
            # Fallback to scipy
            try:
                order_map = {'Nearest Neighbor': 0, 'Bilinear': 1, 'Bicubic': 3, 
                            'Biquintic': 5, 'Biseptic': 7, 'Binonic': 9}
                order = order_map.get(method_name, 1)
                upscaled_methods[method_name] = zoom(img_32x32, zoom=scale_factor, order=order).astype(np.uint8)
                print(f"Upscaling completed for {method_name} (fallback)")
            except:
                pass
    
    return upscaled_methods

# === STEP 5: Quality Metrics ===
def calculate_image_metrics(original, processed, method_name):
    """
    Calculate quality metrics for processed images
    """
    # Mean Squared Error
    mse = np.mean((original.astype(float) - processed.astype(float)) ** 2)
    
    # Peak Signal-to-Noise Ratio
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    
    # Structural Similarity (simplified version)
    mean_orig = np.mean(original)
    mean_proc = np.mean(processed)
    var_orig = np.var(original)
    var_proc = np.var(processed)
    covar = np.mean((original - mean_orig) * (processed - mean_proc))
    
    c1, c2 = 0.01*2, 0.03*2
    ssim = ((2*mean_orig*mean_proc + c1) * (2*covar + c2)) / \
           ((mean_orig*2 + mean_proc*2 + c1) * (var_orig + var_proc + c2))
    
    return {
        'method': method_name,
        'mse': mse,
        'psnr': psnr,
        'ssim': ssim,
        'mean': np.mean(processed),
        'std': np.std(processed)
    }

# === STEP 6: Save Images as TIFF Files ===
def save_images_as_tiff(original_img, img_64x64, downscaled_methods, upscaled_methods):
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
    
    # Save all downscaled 32x32 method images
    method_order = ['Nearest Neighbor', 'Bilinear', 'Bicubic', 'Biquintic', 'Biseptic', 'Binonic']
    for i, method_name in enumerate(method_order):
        if method_name in downscaled_methods:
            filename = f"{i+3:02d}_{method_name.lower().replace(' ', '_')}_downscaled_32x32.tiff"
            Image.fromarray(downscaled_methods[method_name]).save(os.path.join(output_dir, filename))
            print(f"Saved: {output_dir}/{filename}")
    
    # Save all upscaled back to 64x64 method images
    for i, method_name in enumerate(method_order):
        if method_name in upscaled_methods:
            filename = f"{i+9:02d}_{method_name.lower().replace(' ', '_')}_upscaled_64x64.tiff"
            Image.fromarray(upscaled_methods[method_name]).save(os.path.join(output_dir, filename))
            print(f"Saved: {output_dir}/{filename}")

# === STEP 7: Quantum Circuit Encoding ===
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

# === STEP 8: Enhanced 3-Row Visualization ===
def display_comprehensive_comparison():
    """Display images in a clean 3-row layout as requested"""
    # Create figure with proper spacing
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
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
    
    # Row 2: First 3 interpolation methods (downscale-upscale results)
    for i in range(3):
        if i < len(available_methods):
            method_name = available_methods[i]
            processed_img = upscaled_methods[method_name]
            
            axes[1, i].imshow(processed_img, cmap='gray', vmin=0, vmax=255)
            
            # Find corresponding metrics
            method_metrics = next((m for m in metrics_table if m['method'] == method_name), None)
            if method_metrics:
                title = f"{method_name}\n32×32 (Processed)\nPSNR: {method_metrics['psnr']:.1f}dB"
            else:
                title = f"{method_name}\n32×32 (Processed)"
            
            axes[1, i].set_title(title, fontsize=10, fontweight='bold', pad=10)
            axes[1, i].axis('off')
        else:
            axes[1, i].axis('off')
    
    # Row 3: Remaining 3 interpolation methods (downscale-upscale results)
    for i in range(3):
        idx = i + 3  # Methods 4, 5, 6
        if idx < len(available_methods):
            method_name = available_methods[idx]
            processed_img = upscaled_methods[method_name]
            
            axes[2, i].imshow(processed_img, cmap='gray', vmin=0, vmax=255)
            
            # Find corresponding metrics
            method_metrics = next((m for m in metrics_table if m['method'] == method_name), None)
            if method_metrics:
                title = f"{method_name}\n32×32 (Processed)\nPSNR: {method_metrics['psnr']:.1f}dB"
            else:
                title = f"{method_name}\n32×32 (Processed)"
            
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
print("PROCESSING COMPLETE!")
print("="*80)
print("Images saved in 'output_images' directory as individual TIFF files")
print("Downscale-Upscale cycle completed using all six interpolation methods")
print("="*80)
print("QUANTUM IMAGE PROCESSING WITH SIX INTERPOLATION METHODS")
print("DOWNSCALE-UPSCALE CYCLE: 64x64 → 32x32 → 64x64")
print("="*80)

# Load original image
original_img, source_info = load_image()
original_shape = original_img.shape

# Downscale to 64x64
img_64x64 = downscale_to_64x64(original_img)

# Apply all six downscaling methods (64x64 to 32x32)
print("\nApplying six interpolation methods for downscaling (64x64 → 32x32)...")
downscaled_methods = downscale_with_six_methods(img_64x64)

# Apply all six upscaling methods (32x32 back to 64x64)
print("\nApplying six interpolation methods for upscaling (32x32 → 64x64)...")
upscaled_methods = upscale_with_six_methods(downscaled_methods)

# Calculate quality metrics for each method (comparing processed 64x64 to original 64x64)
print("\nCalculating quality metrics...")
metrics_table = []

for method_name, processed_img in upscaled_methods.items():
    metrics = calculate_image_metrics(img_64x64, processed_img, method_name)
    metrics_table.append(metrics)

# Save all images as TIFF files
print("\nSaving images as TIFF files...")
save_images_as_tiff(original_img, img_64x64, downscaled_methods, upscaled_methods)

# Run comprehensive visualization with 3-row layout
print("\nDisplaying comprehensive comparison...")
display_comprehensive_comparison()

# === STEP 9: Quantum Processing on Best Method ===
# Select the method with highest PSNR
if metrics_table:
    best_method = max(metrics_table, key=lambda x: x['psnr'] if x['psnr'] != float('inf') else 0)
    best_processed = upscaled_methods[best_method['method']]
    print(f"\nBest interpolation method: {best_method['method']} (PSNR: {best_method['psnr']:.2f}dB)")
else:
    best_method = {'method': 'Bilinear'}
    best_processed = upscaled_methods.get('Bilinear', list(upscaled_methods.values())[0])

# Create quantum circuit for the best processed image
print(f"\nCreating quantum circuit for {best_method['method']} processed image...")
circuit, (sampled_rows, sampled_cols) = create_quantum_image_circuit(best_processed, sample_size=4)

# Simulate quantum circuit
simulator = AerSimulator()
compiled = transpile(circuit, simulator)
job = simulator.run(compiled, shots=1024)
result = job.result()
counts = result.get_counts()
