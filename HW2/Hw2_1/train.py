import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import LeNet5  # Import the class from model.py

# 1. Configuration
BATCH_SIZE = 64
EPOCHS = 20          # Requirement: At least 15 epochs 
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_process(activation_name):
    print(f"Start Training with {activation_name}...")
    
    # 2. Data Preparation
    # CORRECTION: Removed Horizontal/Vertical Flips because they break digits (6 becomes 9)
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),           # Resize to 32x32
        transforms.RandomRotation(degrees=15), # Augmentation: Rotation is safe!
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Validation data should NOT be augmented
    val_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Download Datasets 
    train_ds = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
    val_ds = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=val_transform)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Model Setup
    model = LeNet5(activation=activation_name.lower()).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    # Lists to store metrics
    loss_train_hist, loss_val_hist = [], []
    acc_train_hist, acc_val_hist = [], []
    
    best_acc = 0.0

    # 4. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        valid_loss = val_loss / len(val_loader)
        valid_acc = 100 * correct_val / total_val
        
        loss_train_hist.append(train_loss)
        loss_val_hist.append(valid_loss)
        acc_train_hist.append(train_acc)
        acc_val_hist.append(valid_acc)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | Val Acc: {valid_acc:.2f}%")

        # Save Best Model 
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), f"Weight_{activation_name}.pth")

    # 5. Plotting and Saving Images 
    plt.figure(figsize=(12, 5))
    
    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(loss_train_hist, label='Training Loss')
    plt.plot(loss_val_hist, label='Validation Loss')
    plt.title(f'Loss Curve ({activation_name})')
    plt.legend()
    
    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(acc_train_hist, label='Training Acc')
    plt.plot(acc_val_hist, label='Validation Acc')
    plt.title(f'Accuracy Curve ({activation_name})')
    plt.legend()
    
    # Save as .jpg
    plt.savefig(f"Loss&Acc_{activation_name}.jpg")
    plt.close()
    print(f"Finished {activation_name}. Saved weights and plots.\n")

if __name__ == "__main__":
    # Train both versions
    train_process("Sigmoid")
    train_process("Relu")