import torch
import torch.nn as nn
import torchvision.datasets as dataset
from torchvision.datasets.cifar import CIFAR10
import torchvision.transforms as transform
from torch.utils.data import DataLoader
import numpy as np

cifar10_train = dataset.CIFAR10(root='.\\dataset\\', train=True, transform=transform.ToTensor(), download=True)
cifar10_test = dataset.CIFAR10(root='.\\dataset\\', train=False, transform=transform.ToTensor(), download=True)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')#결과인 숫자는 알아보기 힘들기 때문에 치환

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(in_features=400, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)
        self.relu = nn.ReLU()
        self.maxPool2d = nn.MaxPool2d(kernel_size=2, stride=2)#렐루와 맥스풀은 파라메터가 동일하기 때문에 한번만 정의해도 된다.
    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.maxPool2d(out)
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.maxPool2d(out)   
        out = out.view(-1, 400)#평탄화
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out

batch_size = 100
learning_rate = 0.1
training_epochs = 15#전체 data에서 몇 번 반복할 것인지
loss_function = nn.CrossEntropyLoss()#loss function
network = LeNet5()
optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)#어떤 wp를 업데이트 할 것인가
data_loader = DataLoader(dataset=cifar10_train, batch_size=batch_size, shuffle=True, drop_last=True)#data set, batch size, training_epochs 번 data를 섞겠다, 남은건 버린다.

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = len(data_loader)

    for img, label in data_loader:
        
        pred = network(img)#실행, 결과받기

        loss = loss_function(pred, label)#loss function으로 loss계산
        optimizer.zero_grad()#loss에 대한 기울기 계산
        loss.backward()#weight update
        optimizer.step()
        avg_cost += loss / total_batch #epoch 끝날때마다 출력
    print('Epoch: %d Loss = %f'%(epoch+1, avg_cost))
torch.save(network.state_dict(), "leNet5_cifar10.pth")#wp 저장
print('Learning finished')

with torch.no_grad():#기울기 계산을 제외하고 계산, 실사용일때는 w계산이 필요가 없으니..
    img_test = torch.tensor(np.transpose(cifar10_test.data,(0,3,1,2)))/255. #입력된 테스트 이미지의 10000*32*32*3를 10000*3*32*32로 변환한다.
    #img_test = mnist_test.data.unsqueeze(1).float()#평탄화 제거, 차원(채널) 추가 : 10000*28*28 에서 10000*1*28*28 로 만들어준다.
    label_test = torch.tensor(cifar10_test.targets)#정답

    prediction = network(img_test)#실행
    correct_prediction = torch.argmax(prediction, 1) == label_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())

    #클래스별 정답률
    correct_counts = np.zeros(10)
    for idx in range(cifar10_test.__len__()):
        if correct_prediction[idx]:
            correct_counts[label_test[idx]] += 1
    accuracy_each_class = correct_counts / (cifar10_test.__len__()/10)

    for idx in range(10):
        print("Accuracy for %s\t: %f"%(classes[idx], accuracy_each_class[idx]))