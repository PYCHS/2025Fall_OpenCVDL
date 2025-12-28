import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self, activation='sigmoid'):
        super(LeNet5, self).__init__()
        self.activation_type = activation
        
        # 1. Convolution: Input (1,32,32) -> Output (6,28,28)
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        
        # 2. Subsampling (Average Pooling): Output (6,14,14)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # 3. Convolution: Output (16,10,10)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Choose activation function based on initialization
        if self.activation_type == 'sigmoid':
            act = torch.sigmoid
        else:
            act = F.relu
            
        x = self.pool(act(self.conv1(x)))
        x = self.pool(act(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5) # Flatten
        x = act(self.fc1(x))
        x = act(self.fc2(x))
        x = self.fc3(x)
        return x