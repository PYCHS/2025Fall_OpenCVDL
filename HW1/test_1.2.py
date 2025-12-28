from functions import color_transformation1_2
import cv2

img_path = r"D:\2025FallOpengCVDL\Q1_image\rgb.jpg"

cv_gray, avg_gray = color_transformation1_2(img_path)

cv2.imshow("Perceptual Grayscale (cv_gray)", cv_gray)
cv2.imshow("Average Grayscale (avg_gray)", avg_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()