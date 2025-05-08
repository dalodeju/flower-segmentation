# Flower Segmentation

## Student Details
**Group 3**

| Name                        | ID        |
|-----------------------------|-----------|
| Darrel Loh De Jun           | 20414780  |
| Samuel Joshua Anand         | 20497938  |
| Keosha Vadhnii              | 20415775  |
| Valencia Ann Raj Davidraj   | 20410418  |

---

## Project Overview
This project implements an automated Python pipeline for segmenting flower regions in images. The pipeline processes all images in the dataset with no user intervention, producing binary images that highlight flower material. The approach is designed for batch processing and is suitable for applications in botanical research, plant phenotyping, and crop analysis.

---

## Setup Instructions

### 1. Install Python
Download and install Python from [python.org](https://www.python.org/).

### 2. Install Required Libraries
Install the necessary Python libraries using pip:
```bash
pip install opencv-contrib-python numpy scikit-image
```

---

## How to Run
1. **Clone or download** the project repository:
   ```
   git clone https://github.com/dalodeju/flower-segmentation.git
   ```
2. **Navigate** to the project directory:
   ```
   cd flowerSegmentation
   ```
3. **Ensure the dataset is unzipped** and the directory structure matches:
   - `dataset/input_images/<difficulty>/<image>.jpg`
   - `dataset/ground_truths/<difficulty>/<image>.png`
4. **Run the segmentation script:**
   ```
   python flowerSegmentation.py
   ```
5. **Results:**
   - Segmented images and the best binary segmentation for each input will be saved in the `output/` directory, organized by difficulty and image name.

---

## Pipeline Description
- **Color Space Conversion:** Converts images to LAB color space for better separation of flower and background.
- **Noise Reduction:** Applies median filtering to reduce noise.
- **Otsu's Thresholding:** Segments each LAB channel using adaptive thresholding.
- **Morphological Processing:** Refines segmentation with morphological closing.
- **Evaluation:** Compares each channel's segmentation to ground truth using Structural Similarity Index (SSIM) and selects the best result.

---

## Evaluation
The pipeline is fully automated and processes all images in the dataset. For each image, the best segmented result is chosen based on similarity to the ground truth. All results are saved for review.

---

## Conclusion
This flower segmentation pipeline provides a robust, automated solution for extracting flower regions from images. Its batch-processing capability and evaluation against ground truth make it suitable for research and practical applications requiring high-throughput, consistent segmentation. 