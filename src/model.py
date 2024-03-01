from torch import nn

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2) # 28*28->32*32-->28*28
        # self.batchnorm1 = nn.BatchNorm2d(6)
        self.tanh1 = nn.Tanh()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  # 14*14
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)  # 10*10
        #self.batchnorm2 = nn.BatchNorm2d(16)
        self.tanh2 = nn.Tanh()
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2),  # 5*5
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.tanh3 = nn.Tanh()
        self.fc2 = nn.Linear(120, 84)
        self.tanh4 = nn.Tanh()
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        y = self.conv1(x)
        y = self.tanh1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.tanh2(y)
        y = self.pool2(y)
        y = self.flatten(y)
        y = self.fc1(y)
        y = self.tanh3(y)
        y = self.fc2(y)
        y = self.tanh4(y)
        y = self.fc3(y)
        return y