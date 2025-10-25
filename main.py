import sys
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog, QLabel, QLineEdit,
    QWidget, QVBoxLayout, QGridLayout, QGroupBox, QHBoxLayout
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from functions import (
    color_separation1_1, color_transformation1_2, color_extraction1_3,
    gaussian_blur2_1, bilateral_filter2_2, median_filter2_3,
    sobel_x3_1, sobel_y3_2, sobel_combination_and_threshold3_3, gradient_angle3_4,
    affine_transform4,
    global_threshold5_1, local_threshold5_2
)

img_path = None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenCV + DIP Homework GUI")
        self.setGeometry(200, 150, 1200, 650)

        # === Main layout ===
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # === Left column (buttons) ===
        left_layout = QVBoxLayout()

        # Load buttons
        self.btn_load1 = QPushButton("Load Image 1")
        self.btn_load1.clicked.connect(self.load_image1)
        self.btn_load2 = QPushButton("Load Image 2")
        self.btn_load2.clicked.connect(self.load_image2)

        # Group 1: Image Processing
        group1 = QGroupBox("1. Image Processing")
        g1_layout = QVBoxLayout()
        self.btn_q1_1 = QPushButton("1.1 Color Separation")
        self.btn_q1_2 = QPushButton("1.2 Color Transformation")
        self.btn_q1_3 = QPushButton("1.3 Color Extraction")
        for b in [self.btn_q1_1, self.btn_q1_2, self.btn_q1_3]:
            g1_layout.addWidget(b)
        group1.setLayout(g1_layout)

        # Group 2: Image Smoothing
        group2 = QGroupBox("2. Image Smoothing")
        g2_layout = QVBoxLayout()
        self.btn_q2_1 = QPushButton("2.1 Gaussian Blur")
        self.btn_q2_2 = QPushButton("2.2 Bilateral Filter")
        self.btn_q2_3 = QPushButton("2.3 Median Filter")
        for b in [self.btn_q2_1, self.btn_q2_2, self.btn_q2_3]:
            g2_layout.addWidget(b)
        group2.setLayout(g2_layout)

        # Group 3: Edge Detection
        group3 = QGroupBox("3. Edge Detection")
        g3_layout = QVBoxLayout()
        self.btn_q3_1 = QPushButton("3.1 Sobel X")
        self.btn_q3_2 = QPushButton("3.2 Sobel Y")
        self.btn_q3_3 = QPushButton("3.3 Combination and Threshold")
        self.btn_q3_4 = QPushButton("3.4 Gradient Angle")
        for b in [self.btn_q3_1, self.btn_q3_2, self.btn_q3_3, self.btn_q3_4]:
            g3_layout.addWidget(b)
        group3.setLayout(g3_layout)

        # Add to left layout
        left_layout.addWidget(self.btn_load1)
        left_layout.addWidget(group1)
        left_layout.addWidget(group2)
        left_layout.addWidget(group3)
        left_layout.addWidget(self.btn_load2)
        left_layout.addStretch(1)

        # === Right column ===
        right_layout = QVBoxLayout()

        # Group 4: Transform
        group4 = QGroupBox("4. Transform")
        g4_layout = QGridLayout()
        self.rotation_input = QLineEdit()
        self.scale_input = QLineEdit()
        self.tx_input = QLineEdit()
        self.ty_input = QLineEdit()
        g4_layout.addWidget(QLabel("Rotation:"), 0, 0)
        g4_layout.addWidget(self.rotation_input, 0, 1)
        g4_layout.addWidget(QLabel("deg"), 0, 2)
        g4_layout.addWidget(QLabel("Scaling:"), 1, 0)
        g4_layout.addWidget(self.scale_input, 1, 1)
        g4_layout.addWidget(QLabel("Tx:"), 2, 0)
        g4_layout.addWidget(self.tx_input, 2, 1)
        g4_layout.addWidget(QLabel("pixel"), 2, 2)
        g4_layout.addWidget(QLabel("Ty:"), 3, 0)
        g4_layout.addWidget(self.ty_input, 3, 1)
        g4_layout.addWidget(QLabel("pixel"), 3, 2)
        self.btn_q4 = QPushButton("4. Transform")
        g4_layout.addWidget(self.btn_q4, 4, 0, 1, 3)
        group4.setLayout(g4_layout)

        # Group 5: Threshold
        group5 = QGroupBox("5. Thresholding")
        g5_layout = QVBoxLayout()
        self.btn_q5_1 = QPushButton("5.1 Global Threshold")
        self.btn_q5_2 = QPushButton("5.2 Local Threshold")
        for b in [self.btn_q5_1, self.btn_q5_2]:
            g5_layout.addWidget(b)
        group5.setLayout(g5_layout)

        # Thumbnail preview
        self.preview_label = QLabel()
        self.preview_label.setFixedSize(200, 200)
        self.preview_label.setStyleSheet("border: 1px solid gray; background-color: #f8f8f8;")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setText("No Image")

        # Add right-side widgets
        right_layout.addWidget(group4)
        right_layout.addWidget(group5)
        right_layout.addWidget(QLabel("Current Image Preview:"))
        right_layout.addWidget(self.preview_label)
        right_layout.addStretch(1)

        # Add both layouts
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        # Connect all functions
        self.connect_buttons()

    # =========================================================
    #  Button Connections
    # =========================================================
    def connect_buttons(self):
        self.btn_q1_1.clicked.connect(self.run_q1_1)
        self.btn_q1_2.clicked.connect(self.run_q1_2)
        self.btn_q1_3.clicked.connect(self.run_q1_3)
        self.btn_q2_1.clicked.connect(lambda: self.safe_run(gaussian_blur2_1))
        self.btn_q2_2.clicked.connect(lambda: self.safe_run(bilateral_filter2_2))
        self.btn_q2_3.clicked.connect(lambda: self.safe_run(median_filter2_3))
        self.btn_q3_1.clicked.connect(lambda: self.safe_run(sobel_x3_1))
        self.btn_q3_2.clicked.connect(lambda: self.safe_run(sobel_y3_2))
        self.btn_q3_3.clicked.connect(lambda: self.safe_run(sobel_combination_and_threshold3_3))
        self.btn_q3_4.clicked.connect(lambda: self.safe_run(gradient_angle3_4))
        self.btn_q4.clicked.connect(self.run_transform)
        self.btn_q5_1.clicked.connect(lambda: self.safe_run(global_threshold5_1))
        self.btn_q5_2.clicked.connect(lambda: self.safe_run(local_threshold5_2))

    # =========================================================
    #  Q1: show images directly
    # =========================================================
    def run_q1_1(self):
        if img_path:
            b_img, g_img, r_img = color_separation1_1(img_path)
            cv2.imshow("Blue Channel", b_img)
            cv2.imshow("Green Channel", g_img)
            cv2.imshow("Red Channel", r_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("[ERROR] Please load an image first.")

    def run_q1_2(self):
        if img_path:
            cv_gray, avg_gray = color_transformation1_2(img_path)
            cv2.imshow("OpenCV Grayscale", cv_gray)
            cv2.imshow("Average Grayscale", avg_gray)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("[ERROR] Please load an image first.")

    def run_q1_3(self):
        if img_path:
            mask, mask_inverse, extracted = color_extraction1_3(img_path)
            cv2.imshow("Yellow Green Mask", mask)
            # cv2.imshow("Mask Inverse", mask_inverse)
            cv2.imshow("Extracted Image", extracted)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("[ERROR] Please load an image first.")

    # =========================================================
    #  General helpers
    # =========================================================
    def safe_run(self, func):
        global img_path
        if img_path:
            func(img_path)
        else:
            print("[ERROR] Please load an image first.")

    def run_transform(self):
        global img_path
        if img_path:
            try:
                angle = float(self.rotation_input.text() or 30)
                scale = float(self.scale_input.text() or 0.9)
                tx = float(self.tx_input.text() or 535)
                ty = float(self.ty_input.text() or 335)
                affine_transform4(img_path, angle, scale, tx, ty)
            except ValueError:
                print("[ERROR] Invalid input: please enter numeric values.")
        else:
            print("[ERROR] Please load an image first.")

    # =========================================================
    #  Load and display preview
    # =========================================================
    def load_image1(self):
        global img_path
        fname, _ = QFileDialog.getOpenFileName(self, "Open Image 1", "", "Images (*.jpg *.png)")
        if fname:
            img_path = fname
            print(f"[INFO] Image 1 loaded: {img_path}")
            self.show_thumbnail(fname)

    def load_image2(self):
        global img_path
        fname, _ = QFileDialog.getOpenFileName(self, "Open Image 2", "", "Images (*.jpg *.png)")
        if fname:
            img_path = fname
            print(f"[INFO] Image 2 loaded (replaced Image 1): {img_path}")
            self.show_thumbnail(fname)

    def show_thumbnail(self, path):
        img = cv2.imread(path)
        if img is None:
            return
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape
        scale = min(180 / w, 180 / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img_rgb, (new_w, new_h))
        qimg = QImage(resized.data, new_w, new_h, 3 * new_w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.preview_label.setPixmap(pixmap)


# =========================================================
#  Main
# =========================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
