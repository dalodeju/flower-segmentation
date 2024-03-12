# Flower Segmentation
# Group 3

import cv2
from matplotlib import pyplot as plt

### INPUT
# read input image
img = cv2.imread('dataset/input_images/easy/easy_1.jpg')

# converting colorspace RGB to LAB
lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
L, A, B = cv2.split(lab_img)

# defining folder locations for easy1.jpg
L_out = 'imageprocessing-pipeline/easy/easy1/L'
A_out = 'imageprocessing-pipeline/easy/easy1/A/'
B_out = 'imageprocessing-pipeline/easy/easy1/B/'

# writing the initial L.A.B images to the image processing pipeline
cv2.imwrite(L_out + '1_L_out.jpg', L)
cv2.imwrite(A_out + '1_A_out.jpg', A)
cv2.imwrite(B_out + '1_B_out.jpg', B)


### PREPROCESSING
# 5x5 median filtering
L_median = cv2.medianBlur(L, 5)
A_median = cv2.medianBlur(A, 5)
B_median = cv2.medianBlur(B, 5)

# writing the median L.A.B images to the image processing pipeline
cv2.imwrite(L_out + '2_L_median.jpg', L_median)
cv2.imwrite(A_out + '2_A_median.jpg', A_median)
cv2.imwrite(B_out + '2_B_median.jpg', B_median)

# equalising the histogram
L_equalized = cv2.equalizeHist(L_median)
A_equalized = cv2.equalizeHist(A_median)
B_equalized = cv2.equalizeHist(B_median)

# writing the equalised L.A.B images to the image processing pipeline
cv2.imwrite(L_out + '3_L_equalized.jpg', L_equalized)
cv2.imwrite(A_out + '3_A_equalized.jpg', A_equalized)
cv2.imwrite(B_out + '3_B_equalized.jpg', B_equalized)


### OTSU
# Segmenting the foreground from background (to be adjusted)
_,L_otsu = cv2.threshold(L_equalized, 125, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_,A_otsu = cv2.threshold(A_equalized, 125, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_,B_otsu = cv2.threshold(B_equalized, 125, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# writing the segmented OTSU L.A.B images to the image processing pipeline
cv2.imwrite(L_out + '4_L_otsu.jpg', L_otsu)
cv2.imwrite(A_out + '4_A_otsu.jpg', A_otsu)
cv2.imwrite(B_out + '4_B_otsu.jpg', B_otsu)

### POSTPROCESSING
# insert postprocessing pipeline here
#
#

# writing the postprocessed L.A.B images to the image processing pipeline
# cv2.imwrite(L_out + '5_L_postprocessed.jpg', L_postprocessed)
# cv2.imwrite(A_out + '5_A_postprocessed.jpg', A_postprocessed)
# cv2.imwrite(B_out + '5_B_postprocessed.jpg', B_postprocessed)

### FINAL OUTPUT
# show otsu images (for testing threshold)
cv2.imshow('L', L_otsu)
cv2.imshow('A', A_otsu)
cv2.imshow('B', B_otsu)

# keep windows open
cv2.waitKey(0)
cv2.destroyAllWindows()