import numpy as np
import cv2
import os
from PIL import Image
import time # To measure execution time

# --- Configuration ---
# Use data number
num_data = 5

# File paths
path_imagein = f'image/Original/data{num_data}.png'

# Fixed parameters for simulation
# We are optimizing ONLY alpha in this script
FIXED_num_bitplanes = 50 # Keep num_bitplanes fixed for this sweep
FIXED_q = 1              # Keep q fixed
FIXED_DC_rate = 0        # Keep DC_rate fixed

# Parameters to sweep
# Define a range of alpha values to test
alpha_values_to_test = [1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9] # Example alpha values

# --- PSNR Calculation Function ---
def calculate_psnr(img1, img2):
    """
    Calculates PSNR between two grayscale images (0-255).
    Args:
        img1 (np.ndarray): First grayscale image (H, W), uint8.
        img2 (np.ndarray): Second grayscale image (H, W), uint8.
    Returns:
        float: PSNR value.
    """
    # Ensure images are the same shape and data type
    if img1.shape != img2.shape:
         print(f"Warning: Image shapes differ: {img1.shape} vs {img2.shape}. Cannot calculate PSNR.")
         return float('-inf') # Return a very low PSNR

    # Ensure uint8 type for PSNR calculation
    if img1.dtype != np.uint8: img1 = img1.astype(np.uint8)
    if img2.dtype != np.uint8: img2 = img2.astype(np.uint8)


    # Handle potential case where one image is completely zero (e.g., black) which can cause issues
    # with MSE being zero and PSNR being infinite. Check if MSE is 0.
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf') # Images are identical

    # Calculate PSNR using OpenCV
    return cv2.PSNR(img1, img2)

# --- Bitplane Generation Function (from your original code) ---
# Note: Adjusted img_input expected type based on typical usage
def Function_BitplaneGen(img_input_gray_255, output_subframe_number, q, alpha, DC_rate):
    """
    Generates bitplanes using Poisson distribution and dark count simulation.

    Args:
        img_input_gray_255 (np.ndarray): Input grayscale image (H, W), uint8 (0-255).
                                         This is the ORIGINAL image.
        output_subframe_number (int): Number of bitplanes to generate (T).
        q (int): Threshold for pixel value (>= q photons -> 1).
        alpha (float): Luminance adjustment parameter.
        DC_rate (float): Dark count probability.

    Returns:
        np.ndarray: Generated bitplanes (H, W, T), float64 (0.0 or 1.0).
    """
    # Normalize input image to 0.0-1.0 for intensity
    img_DR = img_input_gray_255.astype(np.float64) / 255.0

    # As per your original function logic: multiply normalized intensity by T
    # This seems to imply alpha is average photons / bitplane / max_intensity (1.0)
    img_normalized = img_DR * output_subframe_number

    # Initializing arrays
    bitplane = np.zeros((img_input_gray_255.shape[0], img_input_gray_255.shape[1], output_subframe_number), dtype=np.float64)
    incident_photons = np.zeros_like(bitplane, dtype=np.float64) # incident_photons is not used after return, could remove


    # Poisson-based photon counting simulation
    for t in range(output_subframe_number):
        # 入射光子の平均値
        # Using the calculation structure from your original Function_BitplaneGen
        incident_photon_average = alpha * (img_normalized / output_subframe_number)

        # ポアソン乱数で入射光子数を生成
        # Poisson expects non-negative rates. Clamp just in case.
        incident_photon_average_clamped = np.maximum(incident_photon_average, 0.0)
        incident_photon = np.random.poisson(incident_photon_average_clamped)
        incident_photons[:, :, t] = incident_photon # Store if needed, otherwise remove this line

        # ダークカウントノイズの生成
        DC = (np.random.rand(*bitplane[:, :, t].shape) < DC_rate).astype(np.float64)

        # ビットプレーンの生成
        bitplane[:, :, t] = ((incident_photon + DC) >= q).astype(np.float64) # Result is 0.0 or 1.0

    # Return only bitplanes as incident_photons is not used outside
    return bitplane # Returning only bitplane array


# --- Main Optimization Logic ---

# Load the original grayscale image once as the ground truth
# Ensure it's loaded as grayscale (cv2.IMREAD_GRAYSCALE) and is uint8 (0-255)
original_image_gray = cv2.imread(path_imagein, cv2.IMREAD_GRAYSCALE)

if original_image_gray is None:
    print(f"Error: Could not load original image from {path_imagein}")
    exit()

# Ensure the original image is uint8 (0-255)
if original_image_gray.dtype != np.uint8:
     original_image_gray = original_image_gray.astype(np.uint8)

# Get image dimensions (from the loaded image)
height, width = original_image_gray.shape

print(f"Original image loaded: {path_imagein} ({width}x{height})")
print(f"Fixed parameters: num_bitplanes={FIXED_num_bitplanes}, q={FIXED_q}, DC_rate={FIXED_DC_rate}")
print(f"Alpha values to test: {alpha_values_to_test}")


best_psnr = float('-inf') # Initialize with a very low value
best_alpha = None

print("Starting alpha sweep optimization...")
start_time = time.time()

# Iterate through the alpha values
total_alphas = len(alpha_values_to_test)
for i, current_alpha in enumerate(alpha_values_to_test):
    print(f"Testing ({i+1}/{total_alphas}): alpha={current_alpha}...")

    # Generate bitplanes using the current alpha and fixed parameters
    # Pass the SINGLE grayscale image to the function
    bitplanes_simulated = Function_BitplaneGen(
        original_image_gray,
        FIXED_num_bitplanes,
        FIXED_q,
        current_alpha,
        FIXED_DC_rate
    )

    # --- Simple Reconstruction Method ---
    # Sum the bitplanes (which are 0.0 or 1.0) along the bitplane dimension (axis=2)
    # Then divide by the number of bitplanes to get an average value per pixel (0.0 to 1.0)
    reconstructed_image_01 = np.sum(bitplanes_simulated, axis=2) / FIXED_num_bitplanes

    # Scale the reconstructed image to 0-255 and convert to uint8 for PSNR calculation
    reconstructed_image_255 = (reconstructed_image_01 * 255).astype(np.uint8)

    # --- Calculate PSNR ---
    # Calculate PSNR between the original grayscale image and the reconstructed image
    current_psnr = calculate_psnr(original_image_gray, reconstructed_image_255)

    # Print PSNR result
    if current_psnr == float('inf'):
        print(f"  PSNR: Infinity (Images are identical)")
    else:
        print(f"  PSNR: {current_psnr:.4f} dB")


    # --- Check if this is the best PSNR found so far ---
    # Use > for regular numbers, handle infinity correctly
    if current_psnr > best_psnr or (current_psnr == float('inf') and best_psnr != float('inf')):
        best_psnr = current_psnr
        best_alpha = current_alpha
        print(f"  --> New best PSNR found!")

end_time = time.time()
elapsed_time = end_time - start_time

print("\n--- Optimization Complete ---")
print(f"Tested {total_alphas} alpha values.")
print(f"Total time taken: {elapsed_time:.2f} seconds")
print("\nBest parameter found:")
print(f"  Best Alpha: {best_alpha}")
# Handle infinite PSNR case for final print
if best_psnr == float('inf'):
     print(f"  Corresponding Best PSNR: Infinity (Images are identical)")
else:
    print(f"  Corresponding Best PSNR: {best_psnr:.4f} dB")
