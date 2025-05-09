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
# Note: We won't save all generated bitplanes during optimization
# path_imageout_folder = f'dataset/val/bitplane_data{num_data}/'

# Parameters to sweep
# Define ranges for alpha, num_bitplanes, and DC_rate to test
# Adjust these ranges and steps based on your expected values and computational resources

alpha_values_to_test = [1.3, 1.4, 1.5, 1.6] # Example alpha values
num_bitplanes_values_to_test = [100, 150] # Example num_bitplanes values
dc_rate_values_to_test = [0.0, 0.0001, 0.001, 0.01, 0.05] # Example DC_rate values (e.g., 0% to 5% noise)


# Fixed parameters for simulation
q = 1  # Incident photons >= q become 1 in bit-plane


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
    if img1.shape != img2.shape or img1.dtype != np.uint8 or img2.dtype != np.uint8:
         print("Warning: Images must be same shape and uint8 for PSNR calculation.")
         # Attempt to convert if possible, or handle error
         if img1.shape != img2.shape:
             print("Image shapes differ.")
             return float('-inf') # Return a very low PSNR
         # Basic type conversion attempt if shape matches
         try:
             if img1.dtype != np.uint8: img1 = img1.astype(np.uint8)
             if img2.dtype != np.uint8: img2 = img2.astype(np.uint8)
         except ValueError:
             print("Could not convert image types for PSNR.")
             return float('-inf')


    # Handle potential case where one image is completely zero (e.g., black) which can cause issues
    # with MSE being zero and PSNR being infinite. Check if MSE is 0.
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf') # Images are identical

    # Calculate PSNR using OpenCV
    return cv2.PSNR(img1, img2)

# --- Bitplane Generation Function (from your original code) ---
def Function_BitplaneGen(img_input_gray_255, output_subframe_number, q, alpha, DC_rate):
    """
    Generates bitplanes using Poisson distribution and dark count simulation.

    Args:
        img_input_gray_255 (np.ndarray): Input grayscale image (H, W), uint8 (0-255).
        output_subframe_number (int): Number of bitplanes to generate (T).
        q (int): Threshold for pixel value (>= q photons -> 1).
        alpha (float): Luminance adjustment parameter.
        DC_rate (float): Dark count probability (probability of a pixel being 1 due to dark count).

    Returns:
        tuple: (bitplanes, incident_photons)
            bitplanes (np.ndarray): Generated bitplanes (H, W, T), float64 (0.0 or 1.0).
            incident_photons (np.ndarray): Simulated incident photons (H, W, T), float64.
    """
    # Normalize input image to 0.0-1.0 for intensity
    img_DR = img_input_gray_255.astype(np.float64) / 255.0

    # Initializing arrays
    bitplane = np.zeros((img_input_gray_255.shape[0], img_input_gray_255.shape[1], output_subframe_number), dtype=np.float64)
    incident_photons = np.zeros_like(bitplane, dtype=np.float64)

    # Poisson-based photon counting simulation
    for t in range(output_subframe_number):
        # Average incident photons based on normalized intensity and alpha
        incident_photon_average = alpha * img_DR

        # Generate incident photons using Poisson distribution
        incident_photon = np.random.poisson(incident_photon_average)
        incident_photons[:, :, t] = incident_photon

        # Generate dark count noise
        # DC is 1 where a random number [0,1) is less than DC_rate
        DC = (np.random.rand(*bitplane[:, :, t].shape) < DC_rate).astype(np.float64)

        # Bitplane value is 1 if (incident photons + dark counts) >= q
        bitplane[:, :, t] = ((incident_photon + DC) >= q).astype(np.float64) # Result is 0.0 or 1.0

    return bitplane, incident_photons

# --- Main Optimization Logic ---

# Load the original grayscale image once
original_image_gray = cv2.imread(path_imagein, cv2.IMREAD_GRAYSCALE)

if original_image_gray is None:
    print(f"Error: Could not load original image from {path_imagein}")
    exit()

# Ensure the original image is uint8 (0-255)
if original_image_gray.dtype != np.uint8:
     original_image_gray = original_image_gray.astype(np.uint8)

# Get image dimensions
height, width = original_image_gray.shape

print(f"Original image loaded: {path_imagein} ({width}x{height})")
print(f"Parameter ranges: Alpha={alpha_values_to_test}, Bitplanes={num_bitplanes_values_to_test}, DC_rate={dc_rate_values_to_test}")

best_psnr = float('-inf') # Initialize with a very low value
best_alpha = None
best_num_bitplanes = None
best_dc_rate = None

print("Starting parameter sweep...")
start_time = time.time()

# Iterate through all parameter combinations
total_combinations = len(num_bitplanes_values_to_test) * len(alpha_values_to_test) * len(dc_rate_values_to_test)
current_combination_count = 0

for current_num_bitplanes in num_bitplanes_values_to_test:
    for current_alpha in alpha_values_to_test:
        for current_dc_rate in dc_rate_values_to_test:
            current_combination_count += 1
            print(f"Testing ({current_combination_count}/{total_combinations}): num_bitplanes={current_num_bitplanes}, alpha={current_alpha}, DC_rate={current_dc_rate}...")

            # Generate bitplanes for the current parameters
            bitplanes_simulated, _ = Function_BitplaneGen(
                original_image_gray,
                current_num_bitplanes,
                q,
                current_alpha,
                current_dc_rate # Pass the current DC_rate
            )

            # --- Simple Reconstruction Method ---
            # Sum the bitplanes (0.0 or 1.0) and divide by the number of bitplanes
            reconstructed_image_01 = np.sum(bitplanes_simulated, axis=2) / current_num_bitplanes

            # Scale to 0-255 and convert to uint8 for PSNR
            reconstructed_image_255 = (reconstructed_image_01 * 255).astype(np.uint8)

            # --- Calculate PSNR ---
            current_psnr = calculate_psnr(original_image_gray, reconstructed_image_255)

            # Handle infinite PSNR case if images are identical
            if current_psnr == float('inf'):
                print(f"  PSNR: Infinity (Images are identical)")
            else:
                print(f"  PSNR: {current_psnr:.4f} dB")


            # --- Check if this is the best PSNR found so far ---
            # Handle infinite PSNR as the absolute best
            if current_psnr > best_psnr or (current_psnr == float('inf') and best_psnr != float('inf')):
                best_psnr = current_psnr
                best_alpha = current_alpha
                best_num_bitplanes = current_num_bitplanes
                best_dc_rate = current_dc_rate
                print(f"  --> New best PSNR found!")

end_time = time.time()
elapsed_time = end_time - start_time

print("\n--- Optimization Complete ---")
print(f"Tested {total_combinations} combinations.")
print(f"Total time taken: {elapsed_time:.2f} seconds")
print("\nBest parameters found:")
print(f"  Best Alpha: {best_alpha}")
print(f"  Best Number of Bitplanes: {best_num_bitplanes}")
print(f"  Best DC Rate: {best_dc_rate}")
# Handle infinite PSNR case for final print
if best_psnr == float('inf'):
     print(f"  Corresponding Best PSNR: Infinity (Images are identical)")
else:
    print(f"  Corresponding Best PSNR: {best_psnr:.4f} dB")

# Optional: Add code here to re-run the simulation with the best parameters
# and save those specific bitplanes if needed.
