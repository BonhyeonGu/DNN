import torch
import torch.nn as nn
import torchvision.datasets as dataset
import torchvision.transforms as transform
from torch.utils.data import DataLoader

mnist_train = dataset.MNIST(root='.\\dataset\\', train=True, transform=transform.ToTensor(), download=True)
mnist_test = dataset.MNIST(root='.\\dataset\\', train=False, transform=transform.ToTensor(), download=True)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=10)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        return out

batch_size = 100
learning_rate = 0.1
training_epochs = 15#전체 data에서 몇 번 반복할 것인지
loss_function = nn.CrossEntropyLoss()#loss function
network = MLP()
optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)#어떤 wp를 업데이트 할 것인가
data_loader = DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)#data set, batch size, training_epochs 번 data를 섞겠다, 남은건 버린다.

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = len(data_loader)

    for img, label in data_loader:
        img = img.view(-1, 28*28)#이미지 평탄화
        pred = network(img)#실행, 결과받기

        loss = loss_function(pred, label)#loss function으로 loss계산
        optimizer.zero_grad()#loss에 대한 기울기 계산
        loss.backward()#weight update
        optimizer.step()
        avg_cost += loss / total_batch #epoch 끝날때마다 출력
    print('Epoch: %d Loss = %f'%(epoch+1, avg_cost))

torch.save(network.state_dict(), "basic_MLP.pth")#wp 저장
print('Learning finished')

with torch.no_grad():#기울기 계산을 제외하고 계산, 실사용일때는 w계산이 필요가 없으니..
    img_test = mnist_test.data.view(-1, 28*28).float()#평탄화
    label_test = mnist_test.targets#정답

    prediction = network(img_test)#실행
    correct_prediction = torch.argmax(prediction, 1) == label_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())