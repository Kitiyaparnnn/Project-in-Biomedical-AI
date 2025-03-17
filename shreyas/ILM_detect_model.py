import os
import pickle
import random
import time
from glob import glob

import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage.restoration import denoise_nl_means, estimate_sigma
import pywt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sympy.core.random import shuffle


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
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
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
    Segment ILM layer from the OCT image

    Args:
        image: 2D numpy array of preprocessed OCT image

    Returns:
        1D array representing the segmented ILM layer
    """
    print("Segmenting ILM layer...")
    height, width = image.shape

    # Step 1: Locate the retina region
    top_row, bottom_row = locate_retina_region(image)
    print(f"Retina region identified between rows {top_row} and {bottom_row}")

    # Crop the image to focus on the retina
    retina_region = image[top_row:bottom_row, :]

    # Use binary thresholding to identify bright areas (retina tissue)
    _, binary = cv2.threshold(retina_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Initialize segmentation result
    ilm_line = np.zeros(width, dtype=int)

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
        else:
            # If no retina tissue found in this column, use interpolation or nearby values
            valid_indices = np.where(ilm_line > 0)[0]
            if len(valid_indices) > 0:
                nearest_idx = valid_indices[np.argmin(np.abs(valid_indices - x))]
                ilm_line[x] = ilm_line[nearest_idx]
            else:
                # Default value if no valid data yet
                ilm_line[x] = top_row

    # Apply median filter to smooth the segmentation
    ilm_line = ndimage.median_filter(ilm_line, size=11)

    return ilm_line


def visualize_results(oct_image, ilm_line, save_path=None):
    """
    Visualize the results of ILM detection

    Args:
        oct_image: Original OCT image
        ilm_line: 1D array of ILM boundary
        save_path: Path to save the visualization (optional)
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the OCT image with segmented ILM layer
    ax.imshow(oct_image, cmap='gray')
    ax.set_title('OCT Image with Segmented ILM Layer')

    # Plot ILM boundary
    x_range = np.arange(len(ilm_line))
    ax.plot(x_range, ilm_line, 'r-', linewidth=2, label='ILM')

    ax.legend()
    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Results saved to '{save_path}'")

    plt.close(fig)


def process_multiple_images(image_paths, output_dir, classifications):
    """
    Process multiple OCT images and detect ILM in each

    Args:
        image_paths: List of paths to OCT image files
        output_dir: Directory to save processed results
        classifications: Dictionary with image paths as keys and labels as values

    Returns:
        List of dictionaries with results for each image
    """
    results = []

    print(f"Processing {len(image_paths)} images...")

    for i, image_path in enumerate(image_paths):
        print(f"\nProcessing image {i + 1}/{len(image_paths)}: {os.path.basename(image_path)}")
        try:
            # Load and preprocess the image
            oct_image = load_oct_image(image_path)
            preprocessed_image = preprocess_oct_image(oct_image)

            # Segment ILM layer
            ilm_line = segment_retinal_layers(preprocessed_image)

            # Get the classification label
            label = classifications[image_path]

            # Store results
            results.append({
                'file_path': image_path,
                'ilm_line': ilm_line.tolist(),  # Convert numpy array to list for easier serialization
                'label': label
            })

            # Generate a unique filename for saving results
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_filename = os.path.join(output_dir, f'oct_ilm_detection_{base_name}.png')

            # Visualize and save individual results
            visualize_results(oct_image, ilm_line, save_path=output_filename)

            print(f"Processed {os.path.basename(image_path)}")

        except Exception as e:
            print(f"Error processing {os.path.basename(image_path)}: {str(e)}")
            results.append({
                'file_path': image_path,
                'error': str(e),
                'label': classifications[image_path]
            })

    return results


def get_all_file_names(path):
    try:
        # Get the list of all files in the directory
        files = os.listdir(path)

        # Filter out directories, keeping only files
        file_paths = [os.path.join(path, f) for f in files if os.path.isfile(os.path.join(path, f))]

        return file_paths
    except FileNotFoundError:
        print(f"Error: The path '{path}' does not exist.")
        return []
    except PermissionError:
        print(f"Error: Permission denied to access '{path}'.")
        return []


def prepare_data(results, test_size=0.22):
    X = []
    y = []
    for result in results:
        if 'error' not in result:
            X.append(result['ilm_line'])
            y.append(result['label'])

    X = np.array(X)
    y = np.array(y)

    # Shuffle and split the data
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split_idx = int(len(X) * (1 - test_size))

    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test


def train_model(X_train, X_test, y_train, y_test):
    # Create and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Generate classification report
    report = classification_report(y_test, y_pred)

    return model, accuracy, report


def save_model(model, filename):
    """
    Save the trained model to a file

    Args:
        model: Trained model object
        filename: Name of the file to save the model
    """
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {filename}")
def predict_image(model, image_path, output_dir):
    """
    Predict the class of an OCT image using the trained model.

    Args:
        model: Trained model object.
        image_path: Path to the OCT image file.
        output_dir: Directory to save processed results.
    """
    try:
        # Load and preprocess the image
        oct_image = load_oct_image(image_path)  # Function from your script
        preprocessed_image = preprocess_oct_image(oct_image)  # Function from your script

        # Segment ILM layer (or other features)
        ilm_line = segment_retinal_layers(preprocessed_image)  # Function from your script

        # Predict class using the trained model
        ilm_line_reshaped = np.expand_dims(ilm_line, axis=0)  # Reshape if necessary
        predicted_class = model.predict(ilm_line_reshaped)[0]
        if(predicted_class == 1):
            print(f"*** Predicted class for {os.path.basename(image_path)}: FOVEA")
        else:
            print(f"*** Predicted class for {os.path.basename(image_path)}: NO FOVEA")

        # Save visualization of results
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_filename = os.path.join(output_dir, f'oct_ilm_detection_{base_name}.png')
        visualize_results(oct_image, ilm_line, save_path=output_filename)  # Function from your script

    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")

def main(folder1, folder2):
    """
    Main function to process OCT images from two folders and detect the fovea

    Args:
        folder1: First directory containing OCT images
        folder2: Second directory containing OCT images
        pattern: File pattern to match in directories (default: "*.jpg")

    Returns:
        Array of fovea positions for each image and a dictionary with classifications
    """
    # Create processed directory in the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "processed")
    folder1_files = get_all_file_names(folder1)
    folder2_files = get_all_file_names(folder2)

    image_paths = folder1_files + folder2_files
    random.shuffle(image_paths)
    random.shuffle(image_paths)
    random.shuffle(image_paths)
    # Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    else:
        print(f"Using existing output directory: {output_dir}")

    classifications = {}

    for image in image_paths:

        if image in folder1_files:
            classifications[image] = 1
        else:
            classifications[image] = 0

    if not image_paths:
        raise ValueError("No image paths found in the provided folders.")

    # Process all images
    start_time = time.time()
    results = process_multiple_images(image_paths, output_dir, classifications)
    end_time = time.time()

    X_train, X_test, y_train, y_test = prepare_data(results)

    # Train the modelA
    model, accuracy, report = train_model(X_train, X_test, y_train, y_test)
    save_model(model, 'ILM_detect_model.pkl')
    print("\nModel Training Results:")
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)
    # Return the array of fovea positions and classifications
    return results, model, accuracy, report


if __name__ == "__main__":
    folder_yes = "../dataset/fovea_yes/"
    folder_no = "../dataset/fovea_no/"



    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "predicted")
    model_save_path = os.path.join(script_dir, "ILM_detect_model.pkl")
    if not os.path.exists(model_save_path):
        _, model_save, _, _ = main(folder1=folder_yes, folder2=folder_no)
    else:
        print("Loading existing model from", model_save_path)
        with open(model_save_path, 'rb') as f:
            model_save = pickle.load(f)
    predict_image(model_save, "../dataset/healthy_yes/2002000093_20240529_93900_OS_Carl_Zeiss_Meditec_5000_512x1024x128_ORG_IMG_JPG_064.jpg",output_dir)