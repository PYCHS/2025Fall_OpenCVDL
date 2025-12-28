import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                             QHBoxLayout, QWidget, QFileDialog, QLabel, QFrame, QSpacerItem, QSizePolicy)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from torchsummary import summary
from model import LeNet5  # Import our model class

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hw2_MNIST_Classifier")
        self.setGeometry(100, 100, 800, 500)
        
        # Variable to store the loaded image path
        self.current_image_path = None
        
        # --- Main Layout (Horizontal split) ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()     # Left and Right panels
        central_widget.setLayout(main_layout)
        
        # --- Left Panel (Buttons) ---
        left_layout = QVBoxLayout()
        
        # Button: Load Image (New Requirement)
        self.btn_load = QPushButton("Load Image")
        self.btn_load.clicked.connect(self.load_image)
        left_layout.addWidget(self.btn_load)
        
        # Button: 1.1 Show Architecture
        self.btn_1_1 = QPushButton("1.1 Show Architecture")
        self.btn_1_1.clicked.connect(self.show_architecture)
        left_layout.addWidget(self.btn_1_1)
        
        # Button: 1.2 Show Acc Loss
        self.btn_1_2 = QPushButton("1.2 Show Acc Loss")
        self.btn_1_2.clicked.connect(self.show_acc_loss)
        left_layout.addWidget(self.btn_1_2)
        
        # Button: 1.3 Predict
        self.btn_1_3 = QPushButton("1.3 Predict")
        self.btn_1_3.clicked.connect(self.predict_image)
        left_layout.addWidget(self.btn_1_3)
        
        # Spacer to push the label to the bottom (optional, purely visual)
        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        left_layout.addItem(spacer)
        
        # Label: Predict Result
        self.result_label = QLabel("Predict: ")
        left_layout.addWidget(self.result_label)
        
        # Add Left Layout to Main Layout
        main_layout.addLayout(left_layout, 1) # Ratio 1
        
        # --- Right Panel (Image Display) ---
        self.image_display = QLabel()
        self.image_display.setFrameShape(QFrame.Box) # Add a border like the screenshot
        self.image_display.setStyleSheet("background-color: white;")
        self.image_display.setAlignment(Qt.AlignCenter)
        
        # Add Right Widget to Main Layout
        main_layout.addWidget(self.image_display, 3) # Ratio 3 (Wider than left)

    def load_image(self):
        # Open file dialog
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Image", "", "Image Files (*.png *.jpg *.bmp)", options=options)
        
        if file_path:
            self.current_image_path = file_path # Save path for prediction step
            
            # Display image on the Right Panel using QPixmap
            pixmap = QPixmap(file_path)
            # Scale it to fit the box nicely
            self.image_display.setPixmap(pixmap.scaled(self.image_display.width(), self.image_display.height(), Qt.KeepAspectRatio))

    def show_architecture(self):
        # Task 1.1: Use torchsummary
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LeNet5().to(device)
        print("-" * 30)
        print("Model Architecture:")
        summary(model, (1, 32, 32))
        print("-" * 30)

    def show_acc_loss(self):
        # Task 1.2: Display the saved training plots
        img_sigmoid = cv2.imread("Loss&Acc_Sigmoid.jpg")
        img_relu = cv2.imread("Loss&Acc_Relu.jpg")
        
        if img_sigmoid is None or img_relu is None:
            print("Error: Images not found. Did you run train.py?")
            return

        cv2.imshow("Sigmoid Training Result", img_sigmoid)
        cv2.imshow("ReLU Training Result", img_relu)

    def predict_image(self):
        # Task 1.3: Inference using the Loaded Image
        if self.current_image_path is None:
            self.result_label.setText("Predict: Please load an image first!")
            return
            
        # 1. Preprocessing (Snippet 257-259 in PDF)
        img = cv2.imread(self.current_image_path, cv2.IMREAD_GRAYSCALE)
        
        # Invert colors (Black text on White bg -> White text on Black bg)
        # Assuming input is like a standard drawing on white paper
        img_inverted = cv2.bitwise_not(img)
        img_resized = cv2.resize(img_inverted, (32, 32))
        
        # Prepare for PyTorch
        img_tensor = torch.tensor(img_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        img_tensor = (img_tensor / 255.0 - 0.5) / 0.5  # Normalize
        
        # 2. Load Model (Using ReLU version as standard)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LeNet5(activation='relu').to(device)
        try:
            model.load_state_dict(torch.load("Weight_Relu.pth"))
        except:
            print("Error: Weight file not found. Run train.py first.")
            return
        
        model.eval()
        img_tensor = img_tensor.to(device)
        
        # 3. Inference
        with torch.no_grad():
            output = model(img_tensor)
            probs = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probs).item()
        
        # 4. Update GUI Label
        self.result_label.setText(f"Predict: {predicted_class}")
        
        # 5. Show Histogram
        probs_np = probs.cpu().numpy()[0]
        plt.figure()
        plt.bar(range(10), probs_np)
        plt.title("Probability of each class")
        plt.xlabel("Class")
        plt.ylabel("Probability")
        plt.xticks(range(10))
        plt.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())