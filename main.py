import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import ImageEnhance
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image # 导入PIL库

def loadtraindata():
    path = "F:/pythonProject/pythonProject1/Dateset"
    trainset = torchvision.datasets.ImageFolder(path, transform=transforms.Compose([
        # 将图片缩放到指定大小（h,w）或者保持长宽比并缩放最短的边到int大小
        transforms.Resize((64, 64)),
        transforms.CenterCrop(64),
        transforms.ToTensor()]))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    return trainloader


# 网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(3, 8, 6)
        nn.ReLU()
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        # 卷积层
        self.conv2 = nn.Conv2d(8, 16, 5)
        #激活函数
        nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 32, 3)
        # 激活函数
        nn.ReLU()
        nn.BatchNorm2d(32)
        nn.Dropout(p=0.5)
        #nn.Flatten()
        # 全连接层
        self.fc1 = nn.Linear(2304, 400)
        self.fc2 = nn.Linear(400, 120)
        # 2个输出
        self.fc3 = nn.Linear(120, 2)

    # 前向传播
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # view()函数用来改变tensor的形状，
        # 例如将2行3列的tensor变为1行6列，其中-1表示会自适应的调整剩余的维度
        # 在CNN中卷积或者池化之后需要连接全连接层，所以需要把多维度的tensor展平成一维
        x = x.view(x.size(0), -1)
        # 从卷基层到全连接层的维度转换
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train():
    trainloader = loadtraindata()
    # 神经网络结构
    net = Net()
    # 优化器，学习率为0.001
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # 损失函数也可以自己定义，我们这里用的交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 训练部分，训练的数据量为5个epoch，每个epoch为一个循环
    for epoch in range(50):
        # 每个epoch要训练所有的图片，每训练完成200张便打印一下训练的效果（loss值）
        # 定义一个变量方便我们对loss进行输出
        running_loss = 0.0
        # 这里我们遇到了第一步中出现的trailoader，代码传入数据
        for i, data in enumerate(trainloader, 0):
            # enumerate是python的内置函数，既获得索引也获得数据
            # data是从enumerate返回的data，包含数据和标签信息，分别赋值给inputs和labels
            inputs, labels = data

            # 转换数据格式用Variable
            inputs, labels = Variable(inputs), Variable(labels)
            # 梯度置零，因为反向传播过程中梯度会累加上一次循环的梯度
            optimizer.zero_grad()

            # forward + backward + optimize，把数据输进CNN网络net
            outputs = net(inputs)
            # 计算损失值
            loss = criterion(outputs, labels)
            # loss反向传播
            loss.backward()
            # 反向传播后参数更新
            optimizer.step()
            # loss累加
            running_loss += loss.item()
            if i % 100 == 99:
                # 然后再除以100，就得到这一百次的平均损失值
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                # 这一个100次结束后，就把running_loss归零，下一个100次继续使用
                running_loss = 0.0

    print('Finished Training')

	# 保存神经网络
    netScript = torch.jit.script(net)
    # 保存整个神经网络的结构和模型参数
    torch.jit.save(netScript, 'gender_classfication.pt')

classes = ('女','男')
mbatch_size = 50

def loadtestdata():
    path = "F:/pythonProject/pythonProject1/Dateset"
    testset = torchvision.datasets.ImageFolder(path,
                                                transform=transforms.Compose([
                                                    transforms.Resize((64, 64)),
                                                    transforms.ToTensor()])
                                                )

    testloader = torch.utils.data.DataLoader(testset, batch_size=mbatch_size,
                                             shuffle=True, num_workers=2)

    return testloader

def reload_net():
    trainednet = torch.jit.load('gender_classfication.pt')
    return trainednet

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
def test():
    testloader = loadtestdata()
    net = reload_net()
    dataiter = iter(testloader)
    images, labels = dataiter.__next__()
    # nrow是每行显示的图片数量，缺省值为8
    imshow(torchvision.utils.make_grid(images,nrow=10))
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    # 预测值
    print('test: ', " ".join('%5s' % classes[predicted[j]] for j in range(10)))
    print('test: ', " ".join('%5s' % classes[predicted[j]] for j in range(10,20)))
    print('test: ', " ".join('%5s' % classes[predicted[j]] for j in range(20,30)))
    print('test: ', " ".join('%5s' % classes[predicted[j]] for j in range(30,40)))
    print('test: ', " ".join('%5s' % classes[predicted[j]] for j in range(40, 50)))

    image_path = "F:/pythonProject/pythonProject1/testimages/4083.jpg"
    img_data = Image.open(image_path)
    # 调整图像的饱和度
    random_factor1 = np.random.randint(5, 20) / 10.  # 随机因子
    color_image = ImageEnhance.Color(img_data).enhance(random_factor1)
    # 调整图像的亮度
    random_factor2 = np.random.randint(5, 21) / 10.
    brightness_image = ImageEnhance.Brightness(img_data).enhance(random_factor2)
    # 调整图像对比度
    random_factor3 = np.random.randint(5, 20) / 10.
    contrast_image = ImageEnhance.Contrast(img_data).enhance(random_factor3)
    # 调整图像的锐度
    random_factor4 = np.random.randint(5, 20) / 10.
    sharp_image = ImageEnhance.Sharpness(img_data).enhance(random_factor4)
    plt.subplot(2, 2, 1)
    plt.title("饱和度")
    plt.imshow(color_image)

    plt.subplot(2, 2, 2)
    plt.title("亮度")
    plt.imshow(brightness_image)

    plt.subplot(2, 2, 3)
    plt.title("对比度")
    plt.imshow(contrast_image)

    plt.subplot(2, 2, 4)
    plt.title("锐度")
    plt.imshow(sharp_image)
    spec_outputs = net(color_image)

if __name__ == "__main__":
    IMG_DIR = "Images/photo_1"  ### 原始数据集图像的路径
    train()
    test()
    """### Predict and Write to csv file"""

