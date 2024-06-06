# convolution neural network using fashionMnist
# so In this CNN we need input ,convolutional,maxpooling,dense,output --layers

#importing the libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
#define a transform to normalize the data 
#normalize data is much more important 
# it making everything normal (technicall: put all the features on the same scale ',works better when the data is normalized)

transform = transforms.Compose([transforms.Totensor(),transforms.Normalize((0.5,), (0.5,))])

#above code in simple terms : it takes your data and converting in into a tensor , then normalizing it
#Totensor:this transform the data into tensor it changes data type and shape into multidimensional array so that pytorch can compute
#compose:allows multiple transformation: like creating a pipeline that ensures every data sample goes same preprocessing
#normalize: normalize is normal why the numbers ; by subtracting the mean and diving the standard deviation 

# downloading the dataset
trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform) #why here we use transforms: this applies the normalization
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

#torch.utils.data.dataloader this is pytorch utility that creates an iterable over the dataset it simplifies the process of fetching etc
#batch size: number of sample will be passed through the model at a time(efficiency, memory management,allows sgd variants like mini batch)
#shuffle= true: the data must be shuffled at starting of each epoch  (avoid overfitting, even distribution)

# Download and load the test data
testset = datasets.MNIST('MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

#when we put all these parameters together it comes with more efficient more precised ensuring better model  
#---------------------------------
# DEFINING THE CNN MODEL
# TWO CONVOLUTIONAL LAYER, TWO FULLY CONNECTED(DENSE) LAYER
class SimpleCNN(nn.Module):
    def __init__(self):   #method sets up the layer
        super(SimpleCNN, self).__init__()

        # Define the first convolutional layer   (1--number of input channel,32-number of output which means produce 32 feature maps)
        self.conv1 = nn.Conv2d(1, 32, 3, 1)     #(3-kernel size, 1-stride the step size filter moves across image)

        # Define the second convolutional layer  (32 -number of input its feature map from the first convolution layer ,64 number of output or filter)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        # Define the first fully connected layer  (9216input feature but based on output from convolutional layer after flattening , 128 is the number neurons)
        self.fc1 = nn.Linear(9216, 128)

        # Define the second fully connected layer  (128- input feature from first layer's output 10- number of output feature)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x) # Apply the first convolutional layer
        x = F.relu(x)     # Apply ReLU activation function
        x = self.conv2(x) # Apply the second convolutional layer
        x = F.relu(x)     # Apply ReLU activation function
        x = F.max_pool2d(x, 2) # Apply max pooling with a 2*2 window to reduce spatial dimensions of the feature maps
        x = torch.flatten(x, 1) # Flatten the output (1- indicating the flatten should start from first dim flatten the feature map into a single word)
        x = self.fc1(x)   # Apply the first fully connected layer 
        x = F.relu(x)     # Apply ReLU activation function
        x = self.fc2(x)   # Apply the second fully connected layer
        return F.log_softmax(x, dim=1) # Apply log-softmax to get probabilities

# Create an instance of the CNN model
model = SimpleCNN()


criterion = nn.CrossEntropyLoss() # used for multiclass classification and its a loss function)
optimizer = optim.Adam(model.parameters(), lr=0.001) #adam is the adaptive learning rate optimization algorithm that combines ideas from RMSprop
#model parameter weights and bias
#lr is learning rate we should define which controls the step size taken by the optimizer during parameter updates

# these two working together during training process to optimize the model and minimize the loss

#training a loop

num_epochs = 5
#initiating for loop 
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in trainloader:
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        # Print statistics
        running_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}')

#finished indication
print('Finished Training') 

