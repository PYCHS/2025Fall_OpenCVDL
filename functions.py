import cv2
import numpy as np

# ========== Question 1 ========== #
def color_separation1_1(img_path):
    # 1. read image
    image = cv2. imread(img_path)
    if image is None:
        print("Error: Cannot load image.")
        return None, None, None
    # 2. separate 3 channel
    b,g,r = cv2.split(image)
    # 3. create a whole black image, same size as single channel
    zeros = np.zeros_like(b)
    # 4. merge channels to be 3 colored images
    b_image = cv2.merge([b,zeros,zeros]) # keep blue
    g_image = cv2.merge([zeros,g,zeros]) # keep blue
    r_image = cv2.merge([zeros,zeros,r]) # keep blue

    return b_image, g_image, r_image

def color_transformation1_2(img_path):
    # 1. Read the image
    image = cv2.imread(img_path)
    if image is None:
        print("Error: Cannot load image.")
        return None, None
    # 2. Built-in grayscale conversion
    cv_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 3. average grayscale conversion
    b, g, r = cv2.split(image)
    avg_gray = ((b / 3 + g / 3 + r / 3)).astype(np.uint8)

    return cv_gray, avg_gray

def color_extraction1_3(img_path):
    # 1. Read image
    image = cv2.imread(img_path)
    if image is None:
        print("Error: Cannot load image.")
        return None, None, None
    # 2. Convert from BGR to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 3. Define yellow-green HSV range
    lower_bound = np.array([18, 0, 25])
    upper_bound = np.array([85, 255, 255])
    # 4. Create mask for yellow-green range
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    # 5. Invert the mask
    mask_inverse = cv2.bitwise_not(mask)
    # 6. Remove yellow-green parts using mask_inverse
    extracted_image = cv2.bitwise_and(image, image, mask=mask_inverse)

    return mask, mask_inverse, extracted_image

# ========== Question 2 ========== #
def gaussian_blur2_1(img_path):
    # 1. Read image
    image = cv2.imread(img_path)
    if image is None:
        print("Error: Cannot load image.")
        return
    # 2. Create window for trackbar
    window_name = "Gaussian Blur"
    cv2.namedWindow(window_name)
    # 3. Callback function for trackbar
    def on_trackbar(val):
        m = max(1, val)  # kernel radius m must be >= 1
        ksize = 2 * m + 1  # kernel size must be odd
        blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
        cv2.imshow(window_name, blurred)
    # 4. Create a trackbar to control kernel size (m = 1~5)
    cv2.createTrackbar("m", window_name, 1, 5, on_trackbar)
    # 5. Initial display
    on_trackbar(1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def bilateral_filter2_2(img_path):
    # 1. Read image
    image = cv2.imread(img_path)
    if image is None:
        print("Error: Cannot load image.")
        return
    # 2. Create a named window
    window_name = "Bilateral Filter"
    cv2.namedWindow(window_name)
    # 3. Callback function for the trackbar
    def on_trackbar(val):
        m = max(1, val)
        d = 2 * m + 1  # Kernel diameter
        sigmaColor = 90
        sigmaSpace = 90
        filtered = cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)
        cv2.imshow(window_name, filtered)
    # 4. Create a trackbar to control m (1¡V5)
    cv2.createTrackbar("m", window_name, 1, 5, on_trackbar)
    # 5. Show initial result
    on_trackbar(1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def median_filter2_3(img_path):
    # 1. Read image
    image = cv2.imread(img_path)
    if image is None:
        print("Error: Cannot load image.")
        return
    # 2. Create named window
    window_name = "Median Filter"
    cv2.namedWindow(window_name)
    # 3. Callback function for trackbar
    def on_trackbar(val):
        m = max(1, val)
        ksize = 2 * m + 1  # Ensure odd kernel size
        median = cv2.medianBlur(image, ksize)
        cv2.imshow(window_name, median)
    # 4. Create trackbar (m = 1~5)
    cv2.createTrackbar("m", window_name, 1, 5, on_trackbar)
    # 5. Initial display
    on_trackbar(1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()