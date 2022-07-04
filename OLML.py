import numpy as np
import random,time
import copy,random
import parameters as p
import preprocess
from sklearn import preprocessing
from tqdm import tqdm
from miscMethods import *
import math

class olml:
    def __init__(self, data, C, Lambda, B, option, sparse,remove):
        self.C = C
        self.Lambda = Lambda
        self.B = B
        self.option = option
        self.data = data
        self.rounds = p.rounds
        self.sparse = sparse
        self.remove=remove

        # self.mode="trapezoidal"

    def parameter_set(self, X,loss):
        # inner_product = np.sum([X[k] * X[k] for k in X.keys()])
        inner_product=dotDict(X, X)
        if inner_product == 0:
            inner_product = 1

        if self.option == 0: return loss / inner_product
        if self.option == 1: return np.minimum(self.C, 2*loss / inner_product)
        # if self.option == 2: return loss*2 / ((1 / (2 * self.C)) + inner_product)

    def set_classifier(self):
        self.weights = {key: 0 for key in self.X[0].keys()}
        self.u_count={key: 1 for key in self.X[0].keys()}
        self.stability = self.X[0]
        self.count = dict()
        self.A_ =  dict()
        self.A  =  dict()
        self.keyCount=dict()
        self.n_num = 0
        self.e_num = 0
        self.s_num = 0
        self.sum_loss=0
    def update_stability(self,X):
        e_stability=self.stability
        for key in X.keys():
            if key not in self.count.keys():
                self.count[key] = 1
                self.keyCount[key]=1#最开始出现，权重应该大，信息量大
                self.A_[key]=X[key]
                self.A[key]=X[key]
                self.stability[key]=0.000001#最开始出现时，初始权重
            else:
                # self.keyCount[key]+=1
                self.count[key] +=1
                self.A_[key]=self.A[key]
                self.A[key]=self.A[key] + (X[key] - self.A[key]) / self.count[key]
                self.stability[key]=(self.count[key]-1)/self.count[key]**2*(X[key]-self.A_[key])**2+(self.count[key]-1)/self.count[key]*self.stability[key]
        return e_stability
    def upKeyCount(self,X):  # 方差比例越大信息量越大,出现为1，不出现为零，统计离散度
        # sum = 0
        sum1 = 0
        # e_w=self.e_keys/len(X)
        # s_w=self.s_keys/len(X)
        # n_w=self.n_keys/len(X)
        e_KeyCount=self.keyCount
        for key in X.keys():
            sum1 += self.stability[key]
        for key in X.keys():
            # self.keyCount[key]=sum/self.u_count[key]
            # self.keyCount[key] = self.stability[key] / sum1*self.keyCount[key]
            self.keyCount[key] = self.stability[key] / sum1
        return e_KeyCount

    def predict(self,X):
        y_pre=np.sum([X[k] * self.keyCount[k] * self.weights[k] for k in X.keys()])
        return y_pre



    def expand_space(self,X):#补全特征空间
        self.n_keys=0
        self.e_keys=0
        self.s_keys=0
        e_weights=self.weights
        for key in findDifferentKeys(X, self.weights):
            self.weights[key] = 0
            self.u_count[key] = 1
            self.n_keys+=1
        for key in findDifferentKeys(self.weights,X):
            X[key] = 0
            self.e_keys+=1
        for key in findCommonKeys(X,self.weights):
            self.u_count[key] += 1
            self.s_keys=+1
        return X,e_weights





    def fit(self):#
        np.random.seed(p.random_seed)  # set seed to make consistent experiments
        mean_error_vector = []
        mean_loss_vector = []
        mean_acc_vector = []
        mean_std = []
        mean_F1 = []
        mean_runtime = []
        for i in tqdm(range(self.rounds), desc="OLML training"):  # 进度条
            start = time.time()
            self.getShuffledData(self.data)
            # start = time.time()
            train_error, train_loss, train_acc = 0, 0, 0
            train_error_vector, train_loss_vector, train_acc_vector = [], [], []
            F1 = []
            TN = 0
            self.set_classifier()
            u=1
            l=0
            for t in range(0, len(self.y)):
                l+=1
                row,e_weights=self.expand_space(self.X[t])
                if len(row) == 0:
                    train_error_vector.append(train_error / (l + 1))
                    train_loss_vector.append(train_loss / (l + 1))
                    train_acc_vector.append(1 - train_error / (l + 1))
                    continue
                sta_=self.update_stability(row)
                KC_=self.upKeyCount(row)
                y_value = self.predict(row)
                y_hat = np.sign(y_value)
                y_up = self.y[t]
                if y_up==3:
                    l -= 1
                    u += 1
                    loss = math.exp(-3 * np.square(y_value))
                    if u==1:
                        self.sum_loss = loss
                    if loss>self.sum_loss/u:
                        self.keyCount=KC_
                        self.stability=sta_
                        self.weights=e_weights
                        continue
                    y_up=y_value
                    self.sum_loss += loss
                if y_hat != self.y[t]:
                    train_error += 1
                    if self.y[i] == 1:
                        TN += 1
                loss = (np.maximum(0, (1 - y_up * y_value)))
                tao = self.parameter_set(row, loss)
                self.weights=self.upweight(tao,row,y_up)
                train_error_vector.append(train_error / (l + 1))
                train_loss += loss
                # if (i+1)%100==0:
                #     print("t:{},loss:{},parameter:{},pre:{},error:{}".format(i+1, train_loss / (i + 1), tao, y_pre, train_error / (i + 1)))
                train_loss_vector.append(train_loss / (l + 1))
                train_acc_vector.append(1 - (train_error / (l + 1)))
            mean_error_vector.append(train_error_vector)
            mean_loss_vector.append(train_loss_vector)
            mean_acc_vector.append(train_acc_vector)
            mean_std.append(train_error)
            mean_F1.append(F1)
            mean_runtime.append(time.time() - start)
        mean_error_vector = np.array(mean_error_vector).mean(axis=0)
        mean_loss_vector = np.array(mean_loss_vector).mean(axis=0)
        acc_mean = np.array([i[-1] for i in mean_acc_vector]).mean(axis=0)
        acc_std = np.array([i[-1] for i in mean_acc_vector]).std(axis=0)
        mean_acc_vector = np.array(mean_acc_vector).mean(axis=0)
        mean_F1 = np.array(mean_F1).mean(axis=0)
        error_mean = np.array(mean_std).mean(axis=0)
        error_std = np.array(mean_std).std(axis=0)
        runtime_mean = np.array(mean_runtime).mean(axis=0)
        runtime_std = np.array(mean_runtime).std(axis=0)
        # print("OLML  # (\n C:{:.7f}\n B:{:.1f}\n loss:{:.7f}\n error:{:.1f}±{:.1f}\n acc:{:.3f}±{:.5f}\n runtime:{:.3f}±{:.3f})".format(self.C, self.B, mean_loss_vector[-1], error_mean, error_std, acc_mean, acc_std, runtime_mean,runtime_std))
        return mean_error_vector, mean_loss_vector, mean_acc_vector, mean_F1, [error_mean, error_std], [acc_mean,acc_std], [runtime_mean, runtime_std]


    def upweight(self,tao,X,y):#更新出现问题，等于把分类器重复部分重复叠加,分成出现的和旧的，旧的不参与更新？


        return {key: self.weights[key] + tao * y *X[key]*self.keyCount[key] for key in X.keys()}#稳定度与特征空间同步更新 所以报错
        # return {key: self.weights[key] + tao * y *X[key]for key in X.keys()}#稳定度与特征空间同步更新 所以报错

    def getShuffledData(self,data):  # generate data for cross validation
        copydata = copy.deepcopy(data)
        np.random.shuffle(copydata)
        # random.shuffle(copydata)
        # print("getShuffleData:",np.random.random((1,5)))
        # dataset = preprocess.removeRandomData(copydata)
        X, y = [], []
        for row in copydata:
            y.append(row['class_label'])
            del row['class_label']
            X.append(row)
        y = preprocess.removeRandomlabel(y,self.remove)
        #
        self.X, self.y = X, y