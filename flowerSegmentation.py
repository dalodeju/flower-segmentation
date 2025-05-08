# Flower Segmentation
# Group 3

import cv2
import os
from skimage.metrics import structural_similarity as ssim

# --- Constants ---
NOISE_REDUCTION_KERNEL_SIZE = 5
MORPH_KERNEL_SIZE = (5, 5)

# --- Image Processing Functions ---
def convert_to_lab(image):
    """Convert a BGR image to LAB color space."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

def otsu_thresholding(image):
    """Apply OTSU thresholding to an image."""
    _, segmented_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return segmented_image

def binary_processing(image):
    """Apply morphological closing to a binary image."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL_SIZE)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def noise_reduction(image, kernel_size=NOISE_REDUCTION_KERNEL_SIZE):
    """Apply median blur for noise reduction."""
    return cv2.medianBlur(image, kernel_size)

def post_processing(segmented_image, ground_truth_image):
    """Compute SSIM similarity between segmented and ground truth images."""
    ground_truth_grayscale = cv2.cvtColor(ground_truth_image, cv2.COLOR_BGR2GRAY)
    ground_truth_resized = cv2.resize(ground_truth_grayscale, (segmented_image.shape[1], segmented_image.shape[0]))
    similarity_score, _ = ssim(segmented_image, ground_truth_resized, full=True)
    return similarity_score

def flower_segmentation(image):
    """Run the segmentation pipeline on an image and return processed LAB channels."""
    lab_image = convert_to_lab(image)
    L, A, B = cv2.split(lab_image)
    results = []
    for channel in (L, A, B):
        median = noise_reduction(channel)
        segmented = otsu_thresholding(median)
        processed = binary_processing(segmented)
        results.append(processed)
    return results  # [L_processed, A_processed, B_processed]


def main():
    """Main function to process all images by difficulty and save results."""
    input_folder = "dataset/input_images/"
    output_folder = "output/"
    ground_truth_folder = "dataset/ground_truths/"
    difficulties = ["easy", "medium", "hard"]

    for difficulty in difficulties:
        input_subfolder = os.path.join(input_folder, difficulty)
        output_subfolder = os.path.join(output_folder, difficulty)
        ground_truth_subfolder = os.path.join(ground_truth_folder, difficulty)
        os.makedirs(output_subfolder, exist_ok=True)

        # Get input image paths
        input_paths = [os.path.join(input_subfolder, f) for f in os.listdir(input_subfolder)
                       if f.lower().endswith((".jpg", ".jpeg"))]
        # Get ground truth image paths (assume .png, same base name)
        ground_truth_paths = [os.path.join(ground_truth_subfolder, os.path.splitext(os.path.basename(f))[0] + ".png")
                              for f in input_paths]

        for input_path, ground_truth_path in zip(input_paths, ground_truth_paths):
            filename = os.path.splitext(os.path.basename(input_path))[0]
            output_image_folder = os.path.join(output_subfolder, filename)
            os.makedirs(output_image_folder, exist_ok=True)

            img = cv2.imread(input_path)
            if img is None:
                print(f"Error: Unable to load input image: {input_path}")
                continue
            ground_truth_img = cv2.imread(ground_truth_path)
            if ground_truth_img is None:
                print(f"Error: Unable to load ground truth image: {ground_truth_path}")
                continue

            # Run segmentation pipeline
            segmented_channels = flower_segmentation(img)
            channel_names = ["L", "A", "B"]
            # Save all segmented channel images
            for name, seg in zip(channel_names, segmented_channels):
                cv2.imwrite(os.path.join(output_image_folder, f"{name}_segmented.jpg"), seg)

            # Evaluate similarity
            scores = [post_processing(seg, ground_truth_img) for seg in segmented_channels]
            best_idx = scores.index(min(scores))
            best_segmented_image = segmented_channels[best_idx]
            # Save best segmented image
            cv2.imwrite(os.path.join(output_image_folder, "best_segmented_image.jpg"), best_segmented_image)
            print(f"{filename}: L={scores[0]:.4f}, A={scores[1]:.4f}, B={scores[2]:.4f} | Best: {channel_names[best_idx]}")


if __name__ == "__main__":
    main()
