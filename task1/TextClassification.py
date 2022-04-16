#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
实现文本分类
'''


# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# ## 读取数据

# In[3]:


def read_data(file_path):
    data_all = pd.read_csv(file_path,sep='\t')
    print("data_all shape is :",data_all.shape)
    x = data_all['Phrase']
    y = data_all['Sentiment']
    debug = 1
    if debug == 1:
        # index = np.arange(len(X_data))
        # np.random.shuffle(index)
        # X_data = X_data[index[:2000]]
        # y_data = y_data[index[:2000]]
        x = x[:1000]
        y = y[:1000]
        y = np.array(y).reshape(len(y), 1)
    
    return x,y


# ## 文本特征表示

# In[4]:


train_path = 'data/train.tsv'
X, y = read_data(train_path)

test_path = 'data/test.tsv'


# In[5]:


# BagOfWord词袋表示
class BagOfWord:
    def __init__(self, do_lower_case = False):
        # 元组形式
        self.vocab = {}
        self.do_lower_case = do_lower_case

    # 构建文档词汇字典
    # CountVectorizer.fit
    def fit(self, data_list):
        for sentence in data_list:
            if self.do_lower_case:
                sentence = sentence.lower()
            words = sentence.strip().split(" ")
            for word in words:
                if word not in self.vocab:
                    # 元素值为下标索引值
                    self.vocab[word] = len(self.vocab)
    
    # document-term matrix, count the frequency
    def transform(self, data_list):
        vocab_size = len(self.vocab)
        document_term_matrix = np.zeros((len(data_list),vocab_size))

        for idx, sentence in enumerate(data_list):
            if self.do_lower_case:
                sentence = sentence.lower()
            words = sentence.strip().split(" ")
            for word in words:
                vocab_idex = self.vocab[word]
                document_term_matrix[idx][vocab_idex] += 1
        return document_term_matrix

    def fit_transform(self, data_list):
        self.fit(data_list)
        return self.transform(data_list)


# In[6]:


# N-gram表示
class NGram:
    def __init__(self, ngram_range, do_lower_case = False):
        # 元组形式
        self.ngram_vocab = {}
        self.ngram_range = ngram_range
        self.do_lower_case = do_lower_case

    # 构建文档词汇字典
    # CountVectorizer.fit
    def fit(self, data_list):
        for gram in self.ngram_range:
            for sentence in data_list:
                if self.do_lower_case:
                    sentence = sentence.lower()
                words = sentence.split(" ")
                for i in range(len(words) - gram + 1):
                    n_gram_word = "_".join(words[i : i + gram])
                    if n_gram_word not in self.ngram_vocab:
                        # 元素值为下标索引值
                        self.ngram_vocab[n_gram_word] = len(self.ngram_vocab)
    
    # document-term matrix, count the frequency
    def transform(self, data_list):
        vocab_size = len(self.ngram_vocab)
        print("n:{},m:{}".format(len(data_list), vocab_size))
        document_term_matrix = np.zeros((len(data_list),vocab_size))

        for idx, sentence in enumerate(data_list):
            if self.do_lower_case:
                sentence = sentence.lower()
            words = sentence.strip().split(" ")
            for gram in self.ngram_range:
                for i in range(len(words) - gram + 1):
                    n_gram_word = "_".join(words[i : i + gram])
                    vocab_idex = self.ngram_vocab[n_gram_word]
                    document_term_matrix[idx][vocab_idex] += 1
        return document_term_matrix

    def fit_transform(self, data_list):
        self.fit(data_list)
        return self.transform(data_list)


# ## 分类器：softmax  regression

# In[7]:


def softmax(x):
    x -= np.max(x, axis = 1, keepdims = True)
    exp_x = np.exp(x)
    z = exp_x / np.sum(exp_x, axis = 1, keepdims = True)
    return z 


# In[8]:


class softmaxRegression:
    def __init__(self):
        # 变量初始化
        self.weight = None              # 模型权重
        self.learning_rate = None       # 模型学习率
        self.class_num = None           # 类别数量
        self.m = 0                      # 样本数量
        self.n = 0                      # 特征维数

    def fit(self, X, y, learning_rate, epoch_num, class_num, print_steps, update_strategy):
        self.m, self.n = X.shape
        self.class_num = class_num
        self.weight = np.random.randn(self.n, self.class_num)
        self.learning_rate = learning_rate

        # 类别y转换为独热编码
        y_one_hot = np.zeros((self.m, self.class_num))
        for i in range(self.m):
            y_one_hot[i][y[i]] = 1

        # 记录损失函数值
        loss_history = []
        for epoch in range(epoch_num):
            loss = 0

            if update_strategy == "stochastic":
                random_index = np.arange(len(X))
                np.random.shuffle(random_index)
                for index in list(random_index):
                    X_i = X[index]
                    z = np.dot(X_i,self.weight)
                    predict = softmax(z)
                    loss -= np.log(prediect[i][y[i]])
                    grad = X_i.reshape(1, self.n).T.dot((y_one_hot[i] - predict[i]).reshape(1, self.class_num)).T
                    self.weight += grad

                
            
            if update_strategy == "batch":
                # X--(m,n), weight--(n,class_num)
                z = np.dot(X, self.weight)
                predict = softmax(z)

                # 构造同维度的梯度矩阵
                grad = np.zeros_like(self.weight)

                # 损失函数 & 梯度下降更新
                for i in range(self.m):
                    loss -= np.log(prediect[i][y[i]])
                    grad += X[i].reshape(1, self.n).T.dot((y_one_hot[i] - predict[i]).reshape(1, self.class_num)).T
                
                # grad = np.dot(X.T, (y_one_hot - predict))
                self.weight += self.learning_rate * grad / self.m

            loss /= self.m
            loss_history.append(loss)
            if print_steps == 1:
                print("epoch {} loss {}".format(epoch, loss))

        return loss_history

    def predict(self, X):
        z = np.dot(X, self.weight)
        pred = softmax(z)
        return pred.argmax(axis = 1)

    def score(self, X, y):
        pred = self.predict(X)
        return np.sum(pred == y) / self.m


# ## 建模训练过程

# In[9]:


bow_model = BagOfWord(do_lower_case = True)
ngram_model = NGram(ngram_range = (1,2), do_lower_case = True)
X_bow = bow_model.fit_transform(X)
X_ngram = ngram_model.fit_transform(X)

X_bow_train, X_bow_val, y_bow_train, y_bow_val = train_test_split(X_bow, y, test_size = 0.2, stratify = y)
X_ngram_train, X_ngram_val, y_ngram_train, y_ngram_val = train_test_split(X_ngram, y, test_size = 0.2, stratify = y)

epoch_num = 10
bow_learning_rate = 0.1
ngram_learning_rate = 0.1

model1 = softmaxRegression()
history = model1.fit(X_bow_train, y_bow_train, epoch_num = epoch_num, class_num = 5, learning_rate = bow_learning_rate, print_steps = True, update_strategy="stochastic")
plt.plot(np.arange(len(history)), np.array(history))
plt.show()
train_score1 = model1.score(X_bow_train, y_bow_train)
val_score1 = model1.score(X_bow_train, y_bow_train)
print("Bow train {} test {}".format(train_score1,val_score1))

model2 = softmaxRegression()
history = model2.fit(X_ngram_train, y_ngram_bow, epoch_num = epoch_num, class_num = 5, learning_rate = ngram_learning_rate, print_steps = True, update_strategy="stochastic")
plt.plot(np.arange(len(history)), np.array(history))
plt.show()
train_score2 = model2.score(X_ngram_train, y_ngram_train)
val_score2 = model2.score(X_ngram_train, y_ngram_val)
print("Bow train {} test {}".format(train_score2,val_score2))

