from functions import color_separation1_1
import cv2


img_path = "D:\\2025FallOpengCVDL\\Q1_image\\rgb.jpg"

b_img, g_img, r_img = color_separation1_1(img_path)


cv2.imshow("Blue Channel", b_img)
cv2.imshow("Green Channel", g_img)
cv2.imshow("Red Channel", r_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
