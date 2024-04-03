import cv2
import os


# Function for binary image processing (e.g., morphological operations)

def convert_to_lab(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# Function for OTSU thresholding
def otsu_thresholding(image, threshold=150):
    _, segmented_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return segmented_image

def binary_processing(image):
    # Apply morphological operations (e.g., dilation, erosion)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    processed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return processed_image


# Function for noise reduction
def noise_reduction(image, kernel_size=5):
    return cv2.medianBlur(image, kernel_size)


# Function for post-processing to select best segmented image based on ground truth
def post_processing(segmented_image, ground_truth_image):
    # Ensure both images are not None


    # Resize images to have the same dimensions
    ground_truth_grayscale = cv2.cvtColor(ground_truth_image, cv2.COLOR_BGR2GRAY)

    ground_truth_resized = cv2.resize(ground_truth_grayscale, (segmented_image.shape[1], segmented_image.shape[0]))

    # Calculate structural similarity
    from skimage.metrics import structural_similarity as ssim
    similarity_score, _ = ssim(segmented_image, ground_truth_resized, full=True)
    return similarity_score

# Modify flower_segmentation function to incorporate binary processing and noise reduction
def flower_segmentation(image):
    lab_image = convert_to_lab(image)
    L, A, B = cv2.split(lab_image)

    L_median = noise_reduction(L)
    A_median = noise_reduction(A)
    B_median = noise_reduction(B)

    L_segmented = otsu_thresholding(L_median)
    A_segmented = otsu_thresholding(A_median)
    B_segmented = otsu_thresholding(B_median)

    # Binary image processing
    L_processed = binary_processing(L_segmented)
    A_processed = binary_processing(A_segmented)
    B_processed = binary_processing(B_segmented)

    return L_processed, A_processed, B_processed


# Modify main function to include post-processing

def main():
    input_folder = "dataset/input_images/"
    output_folder = "output/"
    ground_truth_folder = "dataset/ground_truths/"  # Updated path for ground truth images

    for difficulty in ["easy", "medium", "hard"]:
        input_subfolder = os.path.join(input_folder, difficulty)
        output_subfolder = os.path.join(output_folder, difficulty)
        ground_truth_subfolder = os.path.join(ground_truth_folder, difficulty)

        for filename in os.listdir(input_subfolder):
            if filename.endswith(".jpg") or filename.endswith(".jpeg"):
                input_image_path = os.path.join(input_subfolder, filename)
                output_image_folder = os.path.join(output_subfolder, filename.split('.')[0])
                os.makedirs(output_image_folder, exist_ok=True)

                # Read input image
                img = cv2.imread(input_image_path)

                # Apply flower segmentation pipeline
                L_segmented, A_segmented, B_segmented = flower_segmentation(img)

                # Save segmented images
                cv2.imwrite(os.path.join(output_image_folder, "L_segmented.jpg"), L_segmented)
                cv2.imwrite(os.path.join(output_image_folder, "A_segmented.jpg"), A_segmented)
                cv2.imwrite(os.path.join(output_image_folder, "B_segmented.jpg"), B_segmented)

                # Load corresponding ground truth image
                ground_truth_image_path = os.path.join(ground_truth_subfolder, filename.replace(".jpg", ".png"))

                # Ensure ground truth image is loaded successfully
                ground_truth_img = cv2.imread(ground_truth_image_path)
                if ground_truth_img is None:
                    print(f"Error: Unable to load ground truth image: {ground_truth_image_path}")
                    continue  # Skip to the next image

                # Post-processing to select the best segmented image
                value1 = post_processing(L_segmented, ground_truth_img)
                value2 = post_processing(A_segmented, ground_truth_img)
                value3 = post_processing(B_segmented, ground_truth_img)

                # Choose the best segmented image based on similarity scores
                if value3 < value2:
                    if value3 < value1:
                        best_segmented_image = B_segmented
                    else:
                        best_segmented_image = L_segmented
                else:
                    if value2 < value1:
                        best_segmented_image = A_segmented
                    else:
                        best_segmented_image = L_segmented

                # Save the best segmented image
                print(value1,value2,value3)
                cv2.imwrite(os.path.join(output_image_folder, "best_segmented_image.jpg"), best_segmented_image)

if __name__ == "__main__":
    main()

