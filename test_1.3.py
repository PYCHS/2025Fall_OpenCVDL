from functions import color_extraction1_3
import cv2

img_path = r"D:\2025FallOpengCVDL\Q1_image\rgb.jpg"

mask, mask_inverse, extracted_image = color_extraction1_3(img_path)

cv2.imshow("Mask (Yellow-Green)", mask)
cv2.imshow("Inversed Mask", mask_inverse)
cv2.imshow("Extracted Image", extracted_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
