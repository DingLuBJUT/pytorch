import time
import torch
import torchtext
import torchvision
import numpy as np
import pandas as pd
from nltk import word_tokenize
from datetime import timedelta
import torch.nn.functional as F
from nltk.corpus import stopwords
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from nltk.stem import WordNetLemmatizer
from itertools import chain,repeat,islice
from torchnlp.word_to_vector import GloVe
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence




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

    # PyTorch自定义数据
    @staticmethod
    class TestDataSet(Dataset):
        # 加载数据
        def __init__(self):
            super(BaseTorchFour.TestDataSet,self).__init__()
            self.train_data = torch.randn(size=(32, 10, 64))
            self.train_label = torch.randint(low=0, high=2, size=(1, 32)).squeeze(0)

        # 根据下标返回数据元素和标签
        def __getitem__(self, index):
            return self.train_data[index],self.train_label[index]

        # 过去数据集大小
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


class BaseTorchFive:

    @staticmethod
    class TextProcess:
        def __init__(self):
            self.stop_words = stopwords.words("english")
            self.glove_vectors = GloVe(name='6B')
            self.lemma = WordNetLemmatizer()
            self.token_index = {}
            return

        def tokenize(self, sentence):
            return [self.lemma.lemmatize(token.lower()) for token in word_tokenize(sentence) if token not in self.stop_words]

        def padding(self, sentence, max_length, padding_value=0):
            return list(islice(chain(sentence, repeat(padding_value)), max_length))

        def encoder_corpus(self, corpus):
            corpus_token = list(map(self.tokenize, corpus))

            for index, token in enumerate(set(chain(*corpus_token)), start=1):
                self.token_index[token] = index

            token_to_index = lambda tokens: list(map(self.token_index.get, tokens))

            corpus_index = list(map(token_to_index, corpus_token))

            max_length = max([len(token_indexes) for token_indexes in corpus_index])

            return list(map(lambda tokens: self.padding(tokens, max_length), corpus_index))

        def get_embedding(self):
            self.token_index = dict(sorted(self.token_index.items(), key=lambda x: x[1]))
            token_embedding = map(lambda token: self.glove_vectors[token], list(self.token_index.keys()))
            pre_trained_embedding = np.array([embedding_tensor.numpy() for embedding_tensor in token_embedding])
            return pre_trained_embedding

    @staticmethod
    class CustomDataSet(Dataset):
        def __init__(self,data_path):
            super(BaseTorchFive.CustomDataSet,self).__init__()
            # 加载数据
            data = pd.read_csv(data_path)
            # 预处理数据
            text_process = BaseTorchFive.TextProcess()
            # *数据包装成tensor*
            self.text = torch.from_numpy(np.array(text_process.encoder_corpus(data['text'].tolist())))
            # encoder label
            label_encoder = LabelEncoder()
            #  *数据包装成tensor*
            self.label = torch.from_numpy(label_encoder.fit_transform(data['author']))
            # 预训练词向量
            self.pre_trained_embedding = text_process.get_embedding()

        def __getitem__(self, index):
            return self.text[index],self.label[index]

        def __len__(self):
            return len(self.label)



    @staticmethod
    class Net(torch.nn.Module):
        def __init__(self,num_embeddings,embedding_dim,pre_trained_embeddings):
            super(BaseTorchFive.Net,self).__init__()

            self.num_embeddings = num_embeddings
            self.embedding_dim = embedding_dim

            # padding_idx使得embedding weight中的第一行向量为0
            self.embedding = torch.nn.Embedding(self.num_embeddings + 1,
                                                self.embedding_dim,
                                                padding_idx=0)
            # 替换预训练向量
            self.embedding.weight.data[1:] = torch.from_numpy(pre_trained_embeddings)
            # 禁止嵌入层参数更新
            self.embedding.weight.requires_grad = False

            self.gru = torch.nn.GRU(input_size=300,
                                    hidden_size=128,
                                    num_layers=1,
                                    bidirectional=False)

            self.l2 = torch.nn.Sequential(
                torch.nn.Linear(128, 64),
                torch.nn.Dropout(p = 0.5),
                torch.nn.ReLU()
            )

            self.l3 = torch.nn.Sequential(
                torch.nn.Linear(64, 3),
                torch.nn.Softmax(dim = 1)
            )

            return

        def forward(self, input):
            # 对输入对index进行嵌入
            embedding_input = self.embedding(input)
            rnn_output, last_hidden = self.gru(embedding_input)
            l2_output = self.l2(last_hidden.squeeze(0))
            l3_output = self.l3(l2_output.squeeze(0))
            return l3_output


    @staticmethod
    def run():
        # 创建数据
        data = BaseTorchFive.CustomDataSet("../data/spooky-author-identification/train.csv")
        pre_trained_embeddings = data.pre_trained_embedding
        num_embeddings,embedding_dim = pre_trained_embeddings.shape

        # 定义设备
        device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")

        # 创建数据迭代器
        data_iter = DataLoader(dataset = data,
                               batch_size = 64,
                               shuffle = True)

        net = BaseTorchFive.Net(num_embeddings = num_embeddings,
                                embedding_dim = embedding_dim,
                                pre_trained_embeddings = pre_trained_embeddings)
        net.train()
        net.to(device)

        cross_entropy_loss = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(),lr = 0.05)

        for epoch in range(10):
            epoch_loss = []
            for text,label in data_iter:
                optimizer.zero_grad()
                y_hat = net(text.T)
                train_loss = cross_entropy_loss(y_hat,label)
                train_loss.backward()
                optimizer.step()
                epoch_loss.append(train_loss.item())
            print("the epoch %d,the loss %f" % (epoch, np.mean(epoch_loss)))
        return


class BaseTorchSix:

    def __init__(self):
        super(BaseTorchSix,self).__init__()
        self.data_train_path = "../data/Quora Insincere Questions Classification/train.csv"
        self.data_test_path = "../data/Quora Insincere Questions Classification/test.csv"
        self.pre_embedding_path = "../data/Quora Insincere Questions Classification/glove.6B.100d.txt"
        self.model_path = "../model/attention.pkl"


    def data_pre_process(self):
        # 声明数据字段处理流
        text = torchtext.data.Field(
            # 当前字段是否为文本序列
            sequential = True,
            # 当前字段是否为label
            is_target = False,
            # 是否创建词典映射关系
            use_vocab = True,
            # 数据batch size维度是否在第一位
            batch_first = True,
            # 是否进行文本小写转化
            lower = True,
            # 是否返回文本长度信息
            include_lengths = True,
            # 指定分词器
            tokenize = word_tokenize,
            # 指定填充token
            pad_token = '<pad>',
            # 指定not know token
            unk_token = '<unk>',
            # token index类型
            dtype = torch.int64
        )

        target = torchtext.data.Field(
            sequential = False,
            is_target = True,
            use_vocab = False,
            batch_first = True
        )

        # 创建Tabular对象
        data_train = torchtext.data.TabularDataset(
                    # 指定数据文件路径
                    path = self.data_train_path,
                    # 指定数据文件类型
                    format = 'CSV',
                    #
                    fields = {
                                'question_text':('text',text),
                                'target':('target',target)
                            }
        )

        # 加载待预测数据集
        data_test = torchtext.data.TabularDataset(
                    # 指定数据文件路径
                    path = self.data_test_path,
                    # 指定数据文件类型
                    format = 'CSV',
                    #
                    fields = {
                                'question_text':('text',text)
                            }
        )

        # build vocab(创建词典映射关系)
        text.build_vocab(data_train, data_test, min_freq=3)
        # 加载预训练词向量
        text.vocab.load_vectors(torchtext.vocab.Vectors(self.pre_embedding_path))

        # 分割数据集
        train, val = data_train.split(split_ratio=0.9)

        # 获取当前语料库中所有不同词个数
        vocab_size = len(text.vocab.itos)
        # 获取padding token在词典映射中的下标
        padding_index = text.vocab.stoi[text.pad_token]

        # 获取分割之后的训练集的预训练词向量,对之后对Embedding层进行替换
        pre_trained_embedding = data_train.fields['text'].vocab.vectors

        print("the pad token: ",text.pad_token)
        print("the unk token ",text.unk_token)
        print("the data set size: ",len(train) + len(val))
        print("the train set size: ",len(train))
        print("the val set size: ",len(val))
        print("the vocab size: ",len(text.vocab.itos))
        print("the embedding shape: ",text.vocab.vectors.shape)


        process_result = {
            "data_train": train,
            "data_val": val,
            "data_test": data_test,
            "vocab_size": vocab_size,
            "padding_index": padding_index,
            "pre_trained_embedding": pre_trained_embedding
        }

        return process_result


    class Model(torch.nn.Module):
        def __init__(self,**kwargs):
            super(BaseTorchSix.Model,self).__init__()
            self.input_size = kwargs["input_size"]
            self.hidden_size = kwargs["hidden_size"]
            self.num_layers = kwargs["num_layers"]
            self.drop_rate = kwargs["drop_rate"]
            self.bidirectional = kwargs["bidirectional"]
            self.bias = kwargs["bias"]
            self.batch_first = kwargs["batch_first"]

            self.embedding_dim = kwargs["embedding_dim"]
            self.vocab_size = kwargs["vocab_size"]
            self.pre_trained_embedding = kwargs["pre_trained_embedding"]
            self.padding_index = kwargs["padding_index"]
            self.NEG_INF = -1000000.0
            self.TINY_FLOAT = 1e-6


        def build(self):
            if self.pre_trained_embedding is None:
                # 预测阶段直接加载模型
                self.embedding_layer = torch.nn.Embedding(self.vocab_size,
                                                          self.embedding_dim)
            else:
                # 根据预训练向量构建嵌入层
                self.embedding_layer = torch.nn.Embedding.from_pretrained(self.pre_trained_embedding,
                                                                          freeze=False)

            # 指定嵌入层中padding token词向量对下标
            self.embedding_layer.padding_idx = self.padding_index

            self.lstm = torch.nn.LSTM(self.input_size,
                                      self.hidden_size,
                                      self.num_layers,
                                      self.bias,
                                      self.batch_first,
                                      dropout = self.drop_rate,
                                      bidirectional = self.bidirectional)

            self.drop_layer = torch.nn.Dropout(p = self.drop_rate)

            self.attention_fc_layer = torch.nn.Linear(self.hidden_size * 2, 1)

            self.sequential_layer = torch.nn.Sequential(
                torch.nn.Linear(self.hidden_size * 6, self.hidden_size),
                torch.nn.ReLU(),
                torch.nn.Dropout(p = self.drop_rate),
                torch.nn.Linear(self.hidden_size,1)
            )

            # 二分类损失函数
            # 与交叉熵不同之处在于,其没有soft max层
            # 直接通过sigmoid转化之后再进行交叉熵计算
            self.loss = torch.nn.BCEWithLogitsLoss()

            return


        def attention(self,encoder_output,mask):
            """

            :param encoder_output: rnn output with padding
            :param mask: sequence length mask
            :return: shape is batch size
            """
            # attention project
            # prepare for attention, output shape: batch_size * input_size
            encoder_output_project = self.attention_fc_layer(encoder_output).squeeze(-1)

            # 对mask进行修改,使得padding部分不参与attention计算,原理：soft max层计算过小部分为0
            encoder_output_masked = encoder_output_project + ((1 - mask) * self.NEG_INF)
            # 这里使用torch.nn.functional意味着,只是计算,没有参数更新
            attention_weight = F.softmax(encoder_output_masked, dim = -1)
            # 对encoder输出进行加权求和
            attention_output = torch.sum(attention_weight.unsqueeze(-1) * encoder_output, dim = 1)
            return attention_output


        def forward(self, sequences,lengths):

            # 对sequence进行词嵌入
            sequences_embedding = self.embedding_layer(sequences)
            # 对词向量进行dropout,比如8维词向量部分被drop
            sequences_embedding_drop = self.drop_layer(sequences_embedding)

            # 将batch中的句子按照句子长度大小降序排序,同时连带句子长度信息排序
            _,sorted_index = torch.sort(lengths,dim=0,descending=True)
            sorted_sequences = torch.index_select(sequences_embedding_drop,dim = 0,index = sorted_index)
            sorted_lengths = torch.index_select(lengths,dim = 0,index = sorted_index)

            # 对已经padding对sequence进行压缩,使得padding的部分不参与计算. 注:要对输入进行排序
            sequences_packed = pack_padded_sequence(sorted_sequences,sorted_lengths,batch_first=True)

            rnn_output, (last_hidden, last_cell) = self.lstm(sequences_packed)

            # 对压缩后的sequence结果进行解压，即对输出结果添加padding的部分
            rnn_output_unpacked,_  = pad_packed_sequence(rnn_output, batch_first=True)

            # 对排序之后对输出结果进行还原，即还原原始的batch中的句子输入顺序
            _,unsorted_index = torch.sort(sorted_index,dim = 0)
            encoder_output = torch.index_select(rnn_output_unpacked,dim = 0,index = unsorted_index)

            """
            sequence mask
            """

            # mask
            # repeat按照某个维度数据重复若干次
            sequence_length_range = torch.arange(torch.max(lengths)).repeat(len(sequences),1)
            # 构建sequence length mask
            # [1,1,1,1,...,0,0,0]
            mask = torch.gt(lengths.unsqueeze(1),sequence_length_range).to(torch.float32)


            """
            Attention
            """
            sequence_attention = self.attention(encoder_output,mask)


            """
            Pooling
            """

            # mean Pooling
            sequence_sum = torch.sum(encoder_output * mask.unsqueeze(-1).float(),dim = 1)
            sequence_mean = sequence_sum / (torch.sum(mask,dim = -1).unsqueeze(-1) + self.TINY_FLOAT)

            # max Pooling
            min_mask = (1 - mask.unsqueeze(-1)) * self.NEG_INF
            sequence_max,_ = torch.max(encoder_output + min_mask, dim = 1)

            # 联合attention、Pooling特征
            concat_features = torch.cat([sequence_attention,sequence_mean,sequence_max],dim = -1)
            # 将输出结果变为单一维度
            output = self.sequential_layer(concat_features).squeeze(-1)

            return output

    class Trainer:
        def __init__(self,**kwargs):
            self.num_epochs = kwargs["num_epochs"]
            self.batch_size = kwargs["batch_size"]
            self.model_path = kwargs["model_path"]
            self.train_data = kwargs["train_data"]
            self.val_data = kwargs["val_data"]
            self.model = kwargs["model"]
            self.device = kwargs["device"]
            self.optimizer = kwargs["optimizer"]
            self.early_stop = kwargs["early_stop"]
            self.best_val_loss = 10000.0
            self.best_train_loss = 10000.0
            self.train_info = "the epoch {},the train loss {},the cost time {}"
            self.val_info = "the val loss {}best val loss {},the cost time {}"

            return

        def train(self):
            self.model.to(self.device)

            train_iter = torchtext.data.Iterator(
                dataset = self.train_data,
                batch_size = self.batch_size,
                device = self.device,
                train = True,
                shuffle = True,
                sort = False
            )
            val_iter = torchtext.data.Iterator(
                dataset = self.val_data,
                batch_size = self.batch_size,
                device = self.device,
                train = False,
                shuffle = False,
                sort = False
            )

            for epoch in range(self.num_epochs):
                self.model.train()
                train_iter.init_epoch()
                train_loss_list = []
                star_time = time.time()
                for batch_data in train_iter:
                    (seq, len), label = batch_data.text, batch_data.target
                    self.optimizer.zero_grad()
                    y_hat = self.model(seq,len)
                    train_loss = self.model.loss(y_hat,label.float())
                    train_loss.backward()
                    self.optimizer.step()
                    train_loss_list.append(train_loss.item())
                    break
                end_time = time.time()
                cost_time = timedelta(seconds = round(end_time - star_time))
                print(self.train_info.format(epoch,np.mean(train_loss_list),cost_time))

                if self.val_data is not None:
                    if self.evaluate(self.model,val_iter) < 0:
                        return 0
            return 0

        def save_model(self,model,model_path):
            # model_path 要具体到某个.pkl文件
            # state_dict 返回
            torch.save(model.state_dict(),model_path)
            return

        def evaluate(self,model,val_iter):
            # 模型开始预测模式
            model.eval()
            val_loss_list =[]
            star_time = time.time()
            for batch_data in val_iter:
                (seq, len), label = batch_data.text, batch_data.target
                with torch.no_grad():
                    y_hat = self.model(seq, len)
                    val_loss = self.model.loss(y_hat, label.float())
                    val_loss_list.append(val_loss.item())
            val_loss = np.mean(val_loss_list)
            end_time = time.time()
            cost_time = timedelta(seconds = round(end_time - star_time))

            if val_loss > self.best_val_loss:
                self.early_stop -= 1
            else:
                self.best_val_loss = val_loss
                print(self.val_info.format(val_loss, self.best_val_loss, cost_time))
                self.save_model(self.model,self.model_path)

            if self.early_stop <= 0:
                print("the early stop best val loss {}".format(self.best_val_loss))
                return -1
            else:
                return 0

    class Predictor:
        def __init__(self, **kwargs):
            self.batch_size = kwargs["batch_size"]
            self.model = kwargs["model"]
            self.model_path = kwargs["model_path"]
            self.predict_data = kwargs["predict_data"]
            self.device = kwargs["device"]

        def load_model(self):
            # 加载训练好的模型参数
            self.model.load_state_dict(torch.load(self.model_path))
            return

        def predict(self):
            predict_iter = torchtext.data.Iterator(
                dataset=self.predict_data,
                batch_size=self.batch_size,
                device=self.device,
                train=True,
                shuffle=False,
                sort=False
            )
            # 加载模型
            self.load_model()
            # 将模型放入到指定设备中
            self.model.to(self.device)
            # 模型开启预测模型
            self.model.eval()

            predict_list = []
            for batch_data in predict_iter:
                seq, len = batch_data.text
                y_hat = self.model(seq, len)
                """
                detach: 作用是将变量从计算图中抽离出来,不进行参数更新,即不进行梯度下降

                # y=A(x), z=B(y) 求B网络中参数的梯度，不求A网络中参数的梯度
                y = A(x)
                z = B(y.detach())
                z.backward()

                """
                predict_list.append(y_hat.detach())
                break

            print("the predict is done!", predict_list)
            return


    def run(self):
        # 数据处理阶段
        process_result = self.data_pre_process()

        # 模型构建阶段
        model_kwargs = {
            "input_size": 100,
            "hidden_size": 50,
            "num_layers": 2,
            "drop_rate": 0.5,
            "bidirectional": True,
            "bias": True,
            "batch_first": True,
            "embedding_dim": 100,
            "vocab_size": process_result["vocab_size"],
            "pre_trained_embedding": process_result["pre_trained_embedding"],
            "padding_index": process_result["padding_index"],
            "NEG_INF": -1000000.0,
            "TINY_FLOAT": 1e-6
        }

        model = self.Model(**model_kwargs)
        model.build()

        # 训练阶段
        train_kwargs = {
            "num_epochs": 1,
            "batch_size": 128,
            "model_path": self.model_path,
            "train_data": process_result["data_train"],
            "val_data": process_result["data_val"],
            "model": model,
            "device": "cuda:0" if torch.cuda.is_available() else "cpu:0",
            "optimizer": torch.optim.Adam(model.parameters(), lr=0.05),
            "early_stop": 5,
            "best_val_loss": 10000.0,
            "best_train_loss": 10000.0,
            "train_info": "the epoch {},the train loss {},the cost time {}",
            "val_info": "the val loss {}best val loss {},the cost time {}"
        }
        trainer = self.Trainer(**train_kwargs)
        trainer.train()

        # 预测阶段
        model = self.Model(**model_kwargs)
        model.build()
        test_kwargs = {
            "batch_size": 32,
            "model": model,
            "model_path": self.model_path,
            "predict_data": process_result["data_test"],
            "device": "cuda:0" if torch.cuda.is_available() else "cpu:0"
        }

        predictor = self.Predictor(**test_kwargs)
        predictor.predict()
        return


def main():
    # BaseTorchOne.run()
    # BaseTorchTwo.run()
    # BaseTorchThree.run()
    # BaseTorchFour.run()
    # BaseTorchFive.run()
    BaseTorchSix().run()


    return

if __name__ == '__main__':
    main()
