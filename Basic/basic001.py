import torch
import torchvision
import numpy as np
from itertools import chain
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class BaseTorchOne:

    @staticmethod
    def run():

        # 获取指定设备
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")

        # 获取正态分布随机数，并存储到指定设备
        input_x = torch.randn(64,32,dtype = torch.float32,device = device)
        input_y = torch.randn(64,1,dtype = torch.float32,device = device)

        # Linear为单层神经元
        model = torch.nn.Sequential(
            torch.nn.Linear(32,16),
            torch.nn.ReLU(),
            torch.nn.Linear(16,1)
        )
        model.to(device=device)

        # 定义损失函数MSE - reduction(sum,mean,none)
        # mean表示对loss平均
        # sum表示对loss不平均
        mse_loss = torch.nn.MSELoss(reduction='mean')

        for i in range(500):
            # 清空梯度空间(防止出现梯度累加)
            model.zero_grad()

            y_hat = model(input_x)
            train_loss = mse_loss(y_hat,input_y)
            if i % 10 == 0:
                # item()-将包含单个元素向量的值取出
                print(i,train_loss.item())

            # 反向传播-计算梯度
            train_loss.backward()
            # no_grad下计算不进入计算图
            with torch.no_grad():
                # 更新参数
                for param in model.parameters():
                    param -= 0.1 * param.grad


        return

    """
    注： loss(无激活函数): 0.4339706301689148
        loss(添加激活函数): 1.818804662434559e-08
    """


class BaseTorchTwo:

    # 自定义模块
    @staticmethod
    class Net(torch.nn.Module):

        def __init__(self):
            # 调用父类构造
            super(BaseTorchTwo.Net,self).__init__()
            self.layer_1 = torch.nn.Linear(32,16)
            self.active_layer = torch.nn.ReLU()
            self.layer_2 = torch.nn.Linear(16,1)
            return

        # 定义前向计算
        def forward(self, input):
            l1_output = self.layer_1(input)
            l1_active = self.active_layer(l1_output)
            output = self.layer_2(l1_active)
            return output

    @staticmethod
    def run():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")
        input_x = torch.randn(64,32,dtype = torch.float32,device = device)
        input_y = torch.randn(64,1,dtype = torch.float32,device = device)

        net = BaseTorchTwo.Net()
        mse_loss = torch.nn.MSELoss(reduction='mean')

        # 定义优化器
        optimizer = torch.optim.Adam(net.parameters(),lr=0.1)

        net.to(device)
        for i in range(500):
            # 清空梯度空间
            optimizer.zero_grad()
            y_hat = net(input_x)
            train_loss = mse_loss(y_hat,input_y)
            if i % 10 == 0:
                print(i,train_loss.item())
            # 反向传播
            train_loss.backward()
            # 更新参数
            optimizer.step()
        return


class BaseTorchThree:

    @staticmethod
    class Net(torch.nn.Module):
        def __init__(self):
            super(BaseTorchThree.Net,self).__init__()

            self.l1 = torch.nn.Sequential(
                torch.nn.Linear(784, 300),
                # 添加BN、Dropout层
                torch.nn.BatchNorm1d(300),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.5)
            )
            self.l2 = torch.nn.Sequential(
                torch.nn.Linear(300,100),
                torch.nn.BatchNorm1d(100),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.5)
            )
            self.l3 = torch.nn.Sequential(
                torch.nn.Linear(100, 10),
                torch.nn.Softmax(dim=1)
            )
            return

        def forward(self, input):
            l1_output = self.l1(input)
            l2_output = self.l2(l1_output)
            l3_output = self.l3(l2_output)
            return l3_output

    @staticmethod
    def run():

        # 获取PyTorch中MNIST数据集
        train_data = torchvision.datasets.MNIST(root = "../data/mnist/train/",
                                                train = True,
                                                download = True,
                                                # 对数据进行transform操作
                                                transform=torchvision.transforms.Compose([
                                                    # 将图像转化成Tensor
                                                    torchvision.transforms.ToTensor(),
                                                    # 对Tensor进行lambda自定义转化
                                                    torchvision.transforms.Lambda(lambda image:image.reshape(-1))
                                                ]))

        val_data = torchvision.datasets.MNIST(root="../data/mnist/test/",
                                              train = False,
                                              download = True,
                                              transform=torchvision.transforms.Compose([
                                                   torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Lambda(lambda image: image.reshape(-1))
                                               ]))
        # 将数据放入数据批次加载器
        train_data_loader =  DataLoader(dataset = train_data,
                                        batch_size = 32,
                                        shuffle = True)

        val_data_loader = DataLoader(dataset = val_data,
                                     batch_size = 32,
                                     shuffle = True)

        # 获取可用显卡数量
        gpu_num = torch.cuda.device_count()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")

        net = BaseTorchThree.Net()
        cross_entropy_loss = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

        # 模型训练状态,开启BN、Dropout，同时将模型放入指定设备
        net.train()
        net.to(device)

        for epoch in range(1):
            for index,(data,label) in enumerate(train_data_loader):
                # 将数据放入指定设备
                data,label = data.to(device),label.to(device)

                optimizer.zero_grad()
                y_hat = net(data)
                train_loss = cross_entropy_loss(y_hat, label)
                train_loss.backward()
                optimizer.step()

                if index % 10 == 0:
                    print(index,train_loss.item())
        # 保存模型
        torch.save(net,"../model/mnist.pkl")
        """

        1、加载模型
        2、模型预测

        """
        net = torch.load("../model/mnist.pkl")

        # 模型预测状态,关闭BN、Dropout
        net.eval()
        predict_result = []
        with torch.no_grad():
            for index,(data,label) in enumerate(val_data_loader):
                data, label = data.to(device), label.to(device)
                # Tensor转numpy
                y_hat = net(data).numpy()
                predict_result.append(np.argmax(y_hat,axis=1))
        # flatten List
        print(list(chain(*predict_result)))


        return


class BaseTorchFour:

    @staticmethod
    class TestDataSet(Dataset):
        def __init__(self):
            super(BaseTorchFour.TestDataSet,self).__init__()
            self.train_data = torch.randn(size=(32, 10, 64))
            self.train_label = torch.randint(low=0, high=2, size=(1, 32)).squeeze(0)

        def __getitem__(self, index):
            return self.train_data[index],self.train_label[index]

        def __len__(self):
            return len(self.train_label)




    @staticmethod
    class Net(torch.nn.Module):

        def __init__(self):
            super(BaseTorchFour.Net,self).__init__()
            # 定义RNN层
            #   input_size：     表示序列中单个元素大小
            #   hidden_size：    表示隐藏层输出大小
            #   num_layer：      表示层数
            #   bidirectional：  表示循环网络是否为双向
            self.rnn = torch.nn.RNN(input_size=64,
                                   hidden_size=128,
                                   num_layers=1,
                                   bidirectional=False)


            # self.lstm = torch.nn.LSTM(input_size=64,
            #                           hidden_size=128,
            #                           num_layers=1,
            #                           bidirectional=False)

            # self.gru = torch.nn.GRU(input_size=64,
            #                        hidden_size=128,
            #                        num_layers=1,
            #                        bidirectional=False)

            self.l2 = torch.nn.Sequential(
                torch.nn.Linear(128,64),
                torch.nn.Dropout(p = 0.5),
                torch.nn.ReLU()
            )

            self.l3 = torch.nn.Sequential(
                torch.nn.Linear(64,2),
                torch.nn.Softmax(dim=1)
            )

            return

        def forward(self, input):
            # RNN输入：
            #    input： (序列长度seq_len, 文本批量batch, 序列中单个元素大小input_size)
            #   hidden： (1, batch, hidden_size)，默认为0，1 = num_layers * num_directions
            # RNN输出：
            #   output： (seq_len, batch, hidden_size)   hidden_size = num_directions * hidden_size
            #   hidden： (1, batch, hidden_size)         1 = num_layers * num_directions
            rnn_output, last_hidden = self.rnn(input)

            # LSTM输入：
            #    input： 同上
            #   hidden： 同上
            # LSTM输出：
            #   output： 同上
            #   hidden： 同行
            #     cell： 同hidden
            # rnn_output, (last_hidden, last_cell) = self.lstm(input)

            # GRU输入：
            #    input： 同上
            #   hidden： 同上
            # GRU输出：
            #   output： 同上
            #   hidden： 同上
            # rnn_output, last_hidden = self.gru(input)


            l2_output = self.l2(last_hidden.squeeze(0))
            l3_output = self.l3(l2_output.squeeze(0))
            return l3_output


    @staticmethod
    def run():
        # get device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")

        # prepare data
        data = BaseTorchFour.TestDataSet()
        train_iter = DataLoader(dataset = data,
                                batch_size = 2,
                                shuffle=True)


        # build model
        net = BaseTorchFour.Net()
        net.train()
        net.to(device)

        # build loss and optimizer
        cross_entropy_loss = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

        for epoch in range(100):
            epoch_loss = []
            for data,label in train_iter:
                optimizer.zero_grad()
                data = data.permute(1,0,2).to(device)
                y_hat = net(data)
                train_loss = cross_entropy_loss(y_hat,label)
                train_loss.backward()
                optimizer.step()
                epoch_loss.append(train_loss.item())
            print("the epoch %d,the loss %f" % (epoch,np.mean(epoch_loss)))


        return


def main():
    # BaseTorchOne.run()
    # BaseTorchTwo.run()
    # BaseTorchThree.run()
    BaseTorchFour.run()
    return

if __name__ == '__main__':
    main()
