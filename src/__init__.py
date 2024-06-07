#things to note down 
#number of epoch i use num_epoch=10(which means a single epoch pass through the entire dataset and it ensure model has seen data atleast once)
# iterations---epoch consist of  10 iterations  so training 10 epoch means 100 iteration /parameter updates within epoch


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Define a linear regression model
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

#forward pass

    def forward(self, x):
        return self.linear(x)

# Load Fashion MNIST dataset

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# Define model, loss function, and optimizer
#transform into single dimenstion(dimentional reduction)

model = LinearRegression(784, 10) # 784 input features (28x28 pixels), 10 output classes
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01) #optimizer -- stochastic gradient decent

# Training loop

num_epochs = 10
for epoch in range(num_epochs):
    for images, labels in train_loader:
        images = images.view(-1, 28*28) # flatten images
        optimizer.zero_grad()                 # should be zero or else the gradient would accumulate across iteration
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()                       #back progagation from output to input
        optimizer.step()         
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

#finished indication
print('Training finished')
<<<<<<< HEAD
>>>>>>> 6f48284 (added __init__.py)
=======
>>>>>>> 5056bc67f52a7c9c88fa20cdbb28439c06fe7f33
