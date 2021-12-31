import torch
import torch.nn as nn
import torchvision.datasets as dataset
import torchvision.transforms as transform
import matplotlib.pyplot as plt

mnist_train = dataset.MNIST(root='.\\dataset\\', train=True, transform=transform.ToTensor(), download=True)
mnist_test = dataset.MNIST(root='.\\dataset\\', train=False, transform=transform.ToTensor(), download=True)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=300)
        self.fc2 = nn.Linear(in_features=300, out_features=100)
        self.fc3 = nn.Linear(in_features=100, out_features=10)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out

network = MLP()
network.load_state_dict(torch.load("basic_MLP.pth"))

with torch.no_grad():
    img_test = mnist_test.data.view(-1, 28 * 28).float()
    label_test = mnist_test.targets

    prediction = network(img_test)
    correct_prediction = torch.argmax(prediction, 1) == label_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())

    temp_data = mnist_test.data[1]
    prediction = network(temp_data.view(-1,28*28).float())
    print(prediction)
    prediction_num = torch.argmax(prediction)

    print("예측 값은 %d 입니다."%(prediction_num))
    plt.imshow(temp_data, cmap='gray')
    plt.show()