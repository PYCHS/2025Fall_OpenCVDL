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

# ========== Question 3 ========== #
def sobel_x3_1(img_path):
    # 1. Read and convert to grayscale
    image = cv2.imread(img_path)
    if image is None:
        print("Error: Cannot load image.")
        return
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 2. Apply Gaussian smoothing to reduce noise
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # 3. Define Sobel X kernel
    sobel_x_kernel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    # 4. Get image dimensions for loops
    rows, cols = blur.shape
    sobel_x_img = np.zeros_like(blur, dtype=np.float32)
    # 5. Manual convolution
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            region = blur[i - 1:i + 2, j - 1:j + 2]
            gx = np.sum(region * sobel_x_kernel)
            sobel_x_img[i, j] = gx
    # 6. Normalize to 0¡V255 for display
    sobel_x_img = np.abs(sobel_x_img)
    sobel_x_img = (sobel_x_img / np.max(sobel_x_img) * 255).astype(np.uint8)
    # 7. Display result
    cv2.imshow("Sobel X", sobel_x_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def sobel_y3_2(img_path):
    # 1. Read and convert to grayscale
    image = cv2.imread(img_path)
    if image is None:
        print("Error: Cannot load image.")
        return
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 2. Gaussian smoothing to reduce noise
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # 3. Define Sobel Y kernel
    sobel_y_kernel = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ])
    # 4. Initialize output
    rows, cols = blur.shape
    sobel_y_img = np.zeros_like(blur, dtype=np.float32)
    # 5. Manual convolution
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            region = blur[i - 1:i + 2, j - 1:j + 2]
            gy = np.sum(region * sobel_y_kernel)
            sobel_y_img[i, j] = gy
    # 6. Normalize to 0¡V255 for display
    sobel_y_img = np.abs(sobel_y_img)
    sobel_y_img = (sobel_y_img / np.max(sobel_y_img) * 255).astype(np.uint8)
    # 7. Display result
    cv2.imshow("Sobel Y", sobel_y_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def sobel_combination_and_threshold3_3(img_path):
    # Step 1: Read image and convert to grayscale
    image = cv2.imread(img_path)
    if image is None:
        print("Error: Cannot load image.")
        return
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Step 2: Light Gaussian smoothing (same as friend's m=1 ¡÷ 3x3)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # Step 3: Sobel X/Y in float (keep full precision)
    sobel_x_f = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y_f = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    # Step 4: Gradient magnitude (friend-style, not weighted sum)
    magnitude = np.sqrt(sobel_x_f * sobel_x_f + sobel_y_f * sobel_y_f)
    # Step 5: Manual 0..255 stretch (first stretch, prevents dim edges)
    mag_max = magnitude.max()
    if mag_max > 0:
        combination = (magnitude / (mag_max + 1e-8)) * 255.0
    else:
        combination = magnitude.copy()
    combination = combination.astype(np.uint8)
    # Step 6: Normalize again to 0..255 (second stretch, matches friend's pipeline)
    normalized = cv2.normalize(combination, None, 0, 255, cv2.NORM_MINMAX)
    normalized = normalized.astype(np.uint8)
    # Step 7: Two thresholds (128 and 28) as in the PPT/demo
    _, result_128 = cv2.threshold(normalized, 128, 255, cv2.THRESH_BINARY)
    _, result_28  = cv2.threshold(normalized,  28, 255, cv2.THRESH_BINARY)
    # Step 8: Display all results
    cv2.imshow("Combination (Sobel X + Y)", normalized)
    cv2.imshow("Threshold = 128", result_128)
    cv2.imshow("Threshold = 28",  result_28)
    # Step 9: Wait for key press and close
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def gradient_angle3_4(img_path):
    # Step 1: Read image and convert to grayscale
    image = cv2.imread(img_path)
    if image is None:
        print("Error: Cannot load image.")
        return
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Step 2: Sobel X and Y
    sobel_x_kernel = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]], dtype=np.float32)
    sobel_y_kernel = np.array([[-1, -2, -1],
                               [0, 0, 0],
                               [1, 2, 1]], dtype=np.float32)

    sobel_x = cv2.filter2D(blur, cv2.CV_32F, sobel_x_kernel)
    sobel_y = cv2.filter2D(blur, cv2.CV_32F, sobel_y_kernel)

    # Step 3: Compute gradient magnitude
    combination = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    combination = cv2.normalize(combination, None, 0, 255, cv2.NORM_MINMAX)
    combination = combination.astype(np.uint8)

    # Step 4: Compute gradient angle
    angle = np.arctan2(sobel_y, sobel_x)
    angle_deg = np.degrees(angle)
    angle_deg = np.mod(angle_deg, 360)  # 0¡V360¢X

    # Step 5: Create masks for given ranges
    mask1 = np.zeros_like(angle_deg, dtype=np.uint8)
    mask2 = np.zeros_like(angle_deg, dtype=np.uint8)
    mask1[(angle_deg >= 170) & (angle_deg <= 190)] = 255
    mask2[(angle_deg >= 260) & (angle_deg <= 280)] = 255

    # Step 6: Apply bitwise AND to highlight areas
    result1 = cv2.bitwise_and(combination, mask1)
    result2 = cv2.bitwise_and(combination, mask2)

    # Step 7: Display
    cv2.imshow("Gradient Angle 170-190", result1)
    cv2.imshow("Gradient Angle 260-280", result2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ========== Question 4 ========== #
def affine_transform4(
    img_path,
    angle=30,         # degrees, positive = counter-clockwise
    scale=0.9,        # 0.9 as in the slide
    Tx=535, Ty=335,   # translation in pixels
    cx=240, cy=200    # object center C (burger center in the slide)
):
    image = cv2.imread(img_path)
    if image is None:
        print("Error: Cannot load image.")
        return

    h, w = image.shape[:2]

    # 1) Rotation + Scale about the OBJECT center C (cx, cy)
    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)

    # 2) Then translate (move the object center)
    M[0, 2] += Tx
    M[1, 2] += Ty

    # 3) Apply affine transform (keep canvas same size as input)
    result = cv2.warpAffine(image, M, (w, h))

    # 4) Show
    cv2.imshow("Original", image)
    cv2.imshow("Transformed", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ========== Question 5 ========== #
def global_threshold5_1(img_path):
    # Step 1: Read and convert to grayscale
    image = cv2.imread(img_path)
    if image is None:
        print("Error: Cannot load image.")
        return
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Apply global threshold
    _, threshold_image = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)

    # Step 3: Show results
    cv2.imshow("Original image", image)
    cv2.imshow("Global Threshold", threshold_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def local_threshold5_2(img_path):
    # Step 1: Read and convert to grayscale
    image = cv2.imread(img_path)
    if image is None:
        print("Error: Cannot load image.")
        return
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Apply local (adaptive) threshold
    threshold_image = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        19,
        -1
    )

    # Step 3: Show results
    cv2.imshow("Original image", image)
    cv2.imshow("Adaptive threshold image", threshold_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()