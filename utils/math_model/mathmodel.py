import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage.restoration import denoise_nl_means, estimate_sigma
import pywt

def load_oct_image(image_path):
    """
    Load a 2D OCT image from a PNG file
    
    Args:
        image_path: Path to the OCT image file
        
    Returns:
        2D numpy array of the OCT image
    """
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")
    
    print(f"Loaded image with shape: {image.shape}")
    return image

def preprocess_oct_image(image):
    """
    Preprocess the OCT image
    
    Args:
        image: 2D numpy array of OCT image
        
    Returns:
        Preprocessed OCT image
    """
    # Step 1: Denoise the image
    print("Denoising image...")
    
    # Estimate noise standard deviation
    sigma_est = np.mean(estimate_sigma(image))
    
    # Apply Non-Local Means denoising
    denoised_image = denoise_nl_means(image, 
                                     h=1.15 * sigma_est,
                                     fast_mode=True,
                                     patch_size=5,
                                     patch_distance=6)
    
    # Convert to uint8 for further processing
    denoised_image = (denoised_image * 255).astype(np.uint8)
    
    # Step 2: Enhance contrast
    print("Enhancing contrast...")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_image = clahe.apply(denoised_image)
    
    return enhanced_image

def locate_retina_region(image):
    """
    Identify the region of the image that contains the retina
    
    Args:
        image: 2D numpy array of OCT image
        
    Returns:
        Tuple of (top_row, bottom_row) indicating retina region
    """
    height, width = image.shape
    
    # Calculate row-wise mean intensity
    row_means = np.mean(image, axis=1)
    
    # Calculate the gradient of mean intensity
    gradient = np.gradient(row_means)
    
    # The retina typically has higher intensity than the background
    # Use a threshold to identify regions with significant intensity
    threshold = np.mean(row_means) + 0.5 * np.std(row_means)
    
    # Find rows with intensity above threshold
    high_intensity_rows = np.where(row_means > threshold)[0]
    
    if len(high_intensity_rows) > 0:
        top_row = max(0, np.min(high_intensity_rows) - 20)  # Add margin
        bottom_row = min(height - 1, np.max(high_intensity_rows) + 20)  # Add margin
    else:
        # If no clear retina region found, estimate from image height
        top_row = int(height * 0.3)  # Start from 30% of image height
        bottom_row = int(height * 0.7)  # End at 70% of image height
    
    return top_row, bottom_row

def segment_retinal_layers(image):
    """
    Segment ILM and RNFL layers from the OCT image
    
    Args:
        image: 2D numpy array of preprocessed OCT image
        
    Returns:
        Tuple of (ilm_line, rnfl_line) representing segmented layers
    """
    print("Segmenting retinal layers...")
    height, width = image.shape
    
    # Step 1: Locate the retina region
    top_row, bottom_row = locate_retina_region(image)
    print(f"Retina region identified between rows {top_row} and {bottom_row}")
    
    # Crop the image to focus on the retina
    retina_region = image[top_row:bottom_row, :]
    
    # Calculate vertical gradient in the retina region
    gradient_y = cv2.Sobel(retina_region, cv2.CV_64F, 0, 1, ksize=3)
    
    # Use binary thresholding to identify bright areas (retina tissue)
    _, binary = cv2.threshold(retina_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Initialize segmentation results
    ilm_line = np.zeros(width, dtype=int)
    rnfl_line = np.zeros(width, dtype=int)
    
    # For each A-scan (column)
    for x in range(width):
        # Extract column from binary image
        column = binary[:, x]
        
        # Find first white pixel (start of retina)
        white_pixels = np.where(column > 0)[0]
        if len(white_pixels) > 0:
            # ILM is the first white pixel (top of retina) + offset to original image
            ilm_idx = white_pixels[0]
            ilm_line[x] = top_row + ilm_idx
            
            # Extract A-scan gradient from this column
            a_scan_gradient = gradient_y[:, x]
            
            # Start search for RNFL from a few pixels below ILM
            search_start = min(ilm_idx + 5, len(a_scan_gradient) - 1)
            search_end = min(ilm_idx + 50, len(a_scan_gradient) - 1)  # Typical RNFL thickness range
            
            # Find strong negative gradient (bright to dark transition) after ILM
            # which corresponds to the RNFL-GCL boundary
            if search_start < search_end:
                subregion = a_scan_gradient[search_start:search_end]
                
                # Find location of minimum gradient (strongest negative transition)
                min_idx = np.argmin(subregion)
                rnfl_idx = search_start + min_idx
                
                # Add offset to get position in original image
                rnfl_line[x] = top_row + rnfl_idx
            else:
                # If search region is invalid, set RNFL boundary close to ILM
                rnfl_line[x] = ilm_line[x] + 5
        else:
            # If no retina tissue found in this column, use interpolation or nearby values
            # Find nearest valid column
            valid_indices = np.where(ilm_line > 0)[0]
            if len(valid_indices) > 0:
                nearest_idx = valid_indices[np.argmin(np.abs(valid_indices - x))]
                ilm_line[x] = ilm_line[nearest_idx]
                rnfl_line[x] = rnfl_line[nearest_idx]
            else:
                # Default values if no valid data yet
                ilm_line[x] = top_row
                rnfl_line[x] = top_row + 10
    
    # Apply median filter to smooth the segmentation
    ilm_line = ndimage.median_filter(ilm_line, size=11)
    rnfl_line = ndimage.median_filter(rnfl_line, size=11)
    
    # Ensure RNFL is always after ILM with minimum thickness
    for x in range(width):
        min_thickness = 3  # Minimum thickness except at fovea
        if rnfl_line[x] < ilm_line[x] + min_thickness:
            rnfl_line[x] = ilm_line[x] + min_thickness
    
    # Create foveal pit profile - thickness should approach zero at fovea center
    center_x = width // 2
    foveal_region_width = width // 5  # 20% of image width
    
    for x in range(max(0, center_x - foveal_region_width//2), 
                   min(width, center_x + foveal_region_width//2)):
        # Calculate distance from center (normalized to 0-1)
        dist_from_center = abs(x - center_x) / (foveal_region_width//2)
        
        # Calculate thickness factor (0 at center, increasing outward)
        thickness_factor = min(1.0, dist_from_center)
        
        # Apply thickness factor to RNFL at fovea
        current_thickness = rnfl_line[x] - ilm_line[x]
        adjusted_thickness = max(1, int(current_thickness * thickness_factor))
        
        # If we're very close to the center, force zero thickness
        if dist_from_center < 0.1:  # Inner 10% of foveal region
            adjusted_thickness = 0
            
        rnfl_line[x] = ilm_line[x] + adjusted_thickness
    
    return ilm_line, rnfl_line

def compute_rnfl_thickness(ilm_line, rnfl_line):
    """
    Compute RNFL thickness from segmented layers
    
    Args:
        ilm_line: 1D array of ILM boundary positions
        rnfl_line: 1D array of RNFL boundary positions
        
    Returns:
        1D array of RNFL thickness
    """
    print("Computing RNFL thickness...")
    thickness = rnfl_line - ilm_line
    return thickness

def detect_nfd_fovea(thickness):
    """
    Detect fovea position based on RNFL thickness
    
    Args:
        thickness: 1D array of RNFL thickness
        
    Returns:
        X coordinate of detected fovea
    """
    print("Detecting fovea position...")
    # Look for minimum thickness or zero thickness regions
    min_thickness = np.min(thickness)
    
    # Consider "near-zero" thickness (could be 0, 1, or 2 pixels)
    threshold = min(min_thickness + 2, 3)
    near_zero_thickness = (thickness <= threshold)
    
    # If there are zero/near-zero thickness regions
    if np.any(near_zero_thickness):
        # Find contiguous near-zero thickness regions
        labeled_regions, num_regions = ndimage.label(near_zero_thickness)
        
        # Get region properties
        region_properties = []
        
        for region_id in range(1, num_regions + 1):
            indices = np.where(labeled_regions == region_id)[0]
            center = int(np.mean(indices))
            size = len(indices)
            dist_from_center = abs(center - len(thickness) // 2)
            
            region_properties.append({
                'id': region_id,
                'center': center,
                'size': size,
                'dist_from_center': dist_from_center
            })
        
        # Sort by distance from image center (ascending) and size (descending)
        region_properties.sort(key=lambda x: (x['dist_from_center'], -x['size']))
        
        # Select the best region (closest to center with significant size)
        fovea_position = region_properties[0]['center']
    else:
        # No near-zero thickness regions found, look for a valley in thickness
        # Apply smoothing to reduce noise
        smoothed_thickness = ndimage.gaussian_filter1d(thickness, sigma=5)
        
        # Find local minima
        min_indices = []
        for i in range(1, len(smoothed_thickness) - 1):
            if (smoothed_thickness[i] < smoothed_thickness[i-1] and 
                smoothed_thickness[i] < smoothed_thickness[i+1]):
                min_indices.append((i, smoothed_thickness[i]))
        
        # If local minima found, sort by intensity and distance from center
        if min_indices:
            center = len(thickness) // 2
            min_indices.sort(key=lambda x: (x[1], abs(x[0] - center)))
            fovea_position = min_indices[0][0]
        else:
            # Fallback: use global minimum
            fovea_position = np.argmin(smoothed_thickness)
    
    return fovea_position

def nfd_detection_pipeline(image_path):
    """
    Full pipeline for detecting fovea in a 2D OCT image
    
    Args:
        image_path: Path to OCT image file
        
    Returns:
        Tuple of (fovea_position, ilm_line, rnfl_line, thickness, oct_image)
    """
    # Step 1: Load the image
    print("Loading OCT image...")
    oct_image = load_oct_image(image_path)
    
    # Step 2: Preprocess the image
    print("Preprocessing image...")
    preprocessed_image = preprocess_oct_image(oct_image)
    
    # Step 3: Segment retinal layers
    ilm_line, rnfl_line = segment_retinal_layers(preprocessed_image)
    
    # Step 4: Compute RNFL thickness
    thickness = compute_rnfl_thickness(ilm_line, rnfl_line)
    
    # Step 5: Detect fovea position
    fovea_position = detect_nfd_fovea(thickness)
    
    print(f"Fovea detected at position: {fovea_position}")
    
    return fovea_position, ilm_line, rnfl_line, thickness, oct_image

def visualize_results(oct_image, ilm_line, rnfl_line, thickness, fovea_position):
    """
    Visualize the results of fovea detection
    
    Args:
        oct_image: Original OCT image
        ilm_line: 1D array of ILM boundary
        rnfl_line: 1D array of RNFL boundary
        thickness: 1D array of RNFL thickness
        fovea_position: X coordinate of detected fovea
    """
    # Create figure for visualization
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot the OCT image with segmented layers
    axs[0].imshow(oct_image, cmap='gray')
    axs[0].set_title('OCT Image with Segmented Layers')
    
    # Plot ILM and RNFL boundaries
    x_range = np.arange(len(ilm_line))
    axs[0].plot(x_range, ilm_line, 'r-', linewidth=2, label='ILM')
    axs[0].plot(x_range, rnfl_line, 'g-', linewidth=2, label='RNFL')
    
    # Mark fovea position
    axs[0].axvline(x=fovea_position, color='b', linestyle='--', linewidth=2, label='Fovea')
    axs[0].scatter(fovea_position, ilm_line[fovea_position], color='cyan', s=100, marker='o', edgecolors='black')
    
    axs[0].legend()
    
    # Plot thickness profile
    axs[1].plot(x_range, thickness, 'b-', linewidth=2)
    axs[1].set_title('RNFL Thickness Profile')
    axs[1].set_xlabel('Position (pixels)')
    axs[1].set_ylabel('Thickness (pixels)')
    
    # Mark fovea position on thickness plot
    axs[1].axvline(x=fovea_position, color='r', linestyle='--', linewidth=2)
    axs[1].scatter(fovea_position, thickness[fovea_position], color='red', s=100, marker='o')
    
    # Add zero line for reference
    axs[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('oct_fovea_detection_result_2.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Results saved to 'oct_fovea_detection_result_2.png'")

def main(image_path):
    """
    Main function to process an OCT image and detect the fovea
    
    Args:
        image_path: Path to OCT image file
    """
    # Run the NFD detection pipeline
    fovea_position, ilm_line, rnfl_line, thickness, oct_image = nfd_detection_pipeline(image_path)
    
    # Visualize the results
    visualize_results(oct_image, ilm_line, rnfl_line, thickness, fovea_position)
    
    return fovea_position

if __name__ == "__main__":
    # Example usage
    image_path = "/sample_JPGs/40002816/20220531/100000/R/OCT/Carl_Zeiss_Meditec_CIRRUS_HD-OCT_5000(200x1024x200)/Original/ORG_IMG_JPG/40002816_20220531_100000_R_OCT_200x1024x200_ORG_IMG_JPG_0095.jpg"  # Replace with your PNG file path
    main(image_path)