import csv
import random
from sklearn import preprocessing
from sklearn.datasets import load_svmlight_file
import numpy as np
import parameters as p
import copy
from scipy.io import loadmat
import pickle, pandas as pd
import os
import math
from tqdm import tqdm

spambase = './otherDatasets/spambase.csv'
isolet = './otherDatasets/isolet.csv'
german = './otherDatasets/german.csv'
ionosphere = './otherDatasets/ionosphere.csv'
magic04 = './otherDatasets/magic04.csv'
wdbc = './otherDatasets/wdbc.csv'
wpbc = './otherDatasets/wpbc.csv'
wbc = './otherDatasets/wbc.csv'
a8a_training = './otherDatasets/a8a.txt'
svmguide3 = './otherDatasets/svmguide3.txt'
url = './otherDatasets/url_svmlight/'

australian = './otherDatasets/australian.csv'
credit_a = './otherDatasets/credit-a.csv'
credit_g = './otherDatasets/credit-g.csv'
diabetes = './otherDatasets/diabetes.csv'
dna12 = './otherDatasets/dna12.csv'
kr_vs_kp = './otherDatasets/kr-vs-kp.csv'
splice = './otherDatasets/splice.csv'
letter = './otherDatasets/letter.csv'
satimage = './otherDatasets/satimage.csv'
HAPT = './otherDatasets/HAPT.csv'
IMDB = './otherDatasets/IMDB'
EN_FR = './otherDatasets/EN_FR'
EN_GR = './otherDatasets/EN_GR.txt'
EN_IT = './otherDatasets/EN_IT.txt'
EN_SP = './otherDatasets/EN_SP.txt'

moa1 = "../../data/moa1.csv"
moa2 = "../../data/moa2.csv"
moa3 = "../../data/moa3.csv"
agr_g = "../../data//agr_g.csv"
agr_a = "../../data/agr_a.csv"
amazon = "../../data/amazon.csv"
hyper_f = "../../data/hyper_f.csv"
twitter = "../../data/twitter.csv"
from collections import Counter
# from imblearn.datasets import fetch_datasets


# satimage (6435, 36), [(-1, 5809), (1, 626)]
# car_eval_4 (1728, 21), [(-1, 1663), (1, 65)]
# wine_quality (4898, 11), [(-1, 4715), (1, 183)]
# letter_img (20000, 16), [(-1, 19266), (1, 734)]
# yeast_me2 (1484, 8), [(-1, 1433), (1, 51)]
# ozone_level (2536, 72), [(-1, 2463), (1, 73)]

def preRead(file):
    dataset = []
    # return_dataset = []
    with open(file) as csvfile:
        next(csvfile)
        reader = csv.reader(csvfile)
        for row in tqdm(reader,desc="reading"):
            # if row!=0:
            dataset.append(row)
    dataset = np.array(dataset)
    numpy_dataset = dataset.astype(np.float)
    numpy_dataset[:, :-1] = preprocessing.scale(numpy_dataset[:, :-1])
    return numpy_dataset

def readMoa():
    dataset = []
    return_dataset = []
    with open(moa3) as csvfile:
        next(csvfile)
        reader = csv.reader(csvfile)
        for row in tqdm(reader,desc="reading"):
            # next()
            dataset.append(row)
    # print(type(np.float((dataset[0][4]))))
    numpy_dataset = np.array(dataset)
    float_dataset = numpy_dataset.astype(np.float)
    # numpy_dataset = float(numpy_dataset)
    float_dataset[:, :-1] = preprocessing.scale(float_dataset[:, :-1])
    for row in tqdm(float_dataset,desc="reading by line"):
        mydict = {v: k for v, k in enumerate(row)}
        last_colunm = len(mydict.keys()) - 1
        mydict[last_colunm] = 1 if mydict[last_colunm] == 1 else -1
        mydict['class_label'] = mydict.pop(last_colunm)
        return_dataset.append(mydict)
    return return_dataset

def readAgr_g():
    return_dataset = []
    numpy_dataset = preRead(agr_g)
    for row in numpy_dataset:
        mydict = {v: k for v, k in enumerate(row)}
    if mydict[9] == 0:
        mydict[9] = -1
        mydict['class_label'] = mydict.pop(9)
        return_dataset.append(mydict)
    return return_dataset
def readAgr_a():
    return_dataset = []
    numpy_dataset = preRead(agr_a)
    for row in numpy_dataset:
        mydict = {v: k for v, k in enumerate(row)}
        if mydict[9] == 0:
            mydict[9] = -1
        mydict['class_label'] = mydict.pop(9)
        return_dataset.append(mydict)
    return return_dataset

def readTwitter():
    return_dataset = []
    numpy_dataset = preRead(twitter)
    for row in numpy_dataset:
        mydict = {v: k for v, k in enumerate(row)}
        if mydict[30] == 0:
            mydict[30] = -1
        mydict['class_label'] = mydict.pop(30)
        return_dataset.append(mydict)
    return return_dataset


def readAmazon():
    return_dataset = []
    numpy_dataset = preRead(amazon)
    for row in numpy_dataset:
        mydict = {v: k for v, k in enumerate(row)}
        if mydict[30] == 0:
            mydict[30] = -1
        mydict['class_label'] = mydict.pop(30)
        return_dataset.append(mydict)
    return return_dataset
def readHyper_f():
    return_dataset = []
    numpy_dataset = preRead(hyper_f)
    for row in numpy_dataset:
        mydict = {v: k for v, k in enumerate(row)}
        if mydict[10] == 0:
            mydict[10] = -1
        mydict['class_label'] = mydict.pop(10)
        return_dataset.append(mydict)
    return return_dataset






def readWpbcNormalized1():
    dataset = []
    return_dataset = []
    with open(wpbc) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            dataset.append(row)
    dataset = np.array(dataset)
    dataset = dataset[:, 1:]
    numpy_dataset = dataset.astype(np.float)
    featureNumber = numpy_dataset.shape[1]
    NumpyArraryFirstHalf = numpy_dataset[:, 0:int(featureNumber / 2)]
    NumpyArraryLastHalf = numpy_dataset[:, int(featureNumber / 2):]
    randomNumpyArrary = np.random.randint(0, 100, (NumpyArraryLastHalf.shape[0], NumpyArraryLastHalf.shape[1]))
    lastHalf100 = np.multiply(NumpyArraryLastHalf, 100)
    randomNumpyArrary1 = randomNumpyArrary - lastHalf100
    numpy_dataset1 = np.hstack([NumpyArraryFirstHalf, randomNumpyArrary1])
    numpy_dataset[:, 1:] = preprocessing.scale(numpy_dataset1[:, 1:])
    for row in numpy_dataset:
        mydict = {v: k for v, k in enumerate(row)}
        mydict['class_label'] = mydict.pop(0)
        return_dataset.append(mydict)
    return return_dataset

def readWdbcNormalized1():
    dataset = []
    return_dataset = []
    with open(wdbc) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            dataset.append(row)
    dataset = np.array(dataset)
    dataset = dataset[:, 1:]
    numpy_dataset = dataset.astype(np.float)
    featureNumber = numpy_dataset.shape[1]
    NumpyArraryFirstHalf = numpy_dataset[:, 0:int(featureNumber / 2)]
    NumpyArraryFirstHalf[:, 1:] = preprocessing.scale(NumpyArraryFirstHalf[:, 1:])

    NumpyArraryLastHalf = numpy_dataset[:, int(featureNumber / 2):]
    randomNumpyArrary = np.random.randint(0, 10, (NumpyArraryLastHalf.shape[0], NumpyArraryLastHalf.shape[1]))
    lastHalf100 = np.multiply(NumpyArraryLastHalf, 10)
    randomNumpyArrary1 = randomNumpyArrary-lastHalf100
    randomNumpyArrary2 = preprocessing.scale(randomNumpyArrary1,with_mean=False,with_std=False)
    numpy_dataset = np.hstack([NumpyArraryFirstHalf,randomNumpyArrary2])
    # numpy_dataset[:, 1:] = preprocessing.scale(numpy_dataset1[:, 1:])
    for row in numpy_dataset:
        mydict = {v: k for v, k in enumerate(row)}
        mydict['class_label'] = mydict.pop(0)
        return_dataset.append(mydict)
    return return_dataset

def readWdbcNormalized2():
    dataset = []
    return_dataset = []
    with open(wdbc) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            dataset.append(row)
    dataset = np.array(dataset)
    dataset = dataset[:, 1:]
    numpy_dataset = dataset.astype(np.float)
    featureNumber = numpy_dataset.shape[1]
    numpy_dataset[:, 1:] = preprocessing.scale(numpy_dataset[:, 1:])
    NumpyArraryFirstHalf = numpy_dataset[:, 0:int(featureNumber / 2)]
    NumpyArraryLastHalf = numpy_dataset[:, int(featureNumber / 2):]
    randomNumpyArrary = np.random.randint(0, 10, (NumpyArraryLastHalf.shape[0], NumpyArraryLastHalf.shape[1]))
    lastHalf100 = np.multiply(NumpyArraryLastHalf, np.random.rand())
    randomNumpyArrary1 = randomNumpyArrary-lastHalf100
    numpy_dataset = np.hstack([NumpyArraryFirstHalf,randomNumpyArrary1])
    for row in numpy_dataset:
        mydict = {v: k for v, k in enumerate(row)}
        mydict['class_label'] = mydict.pop(0)
        return_dataset.append(mydict)
    return return_dataset

def readSplice1():
    dataset = []
    return_dataset = []
    with open(splice) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            dataset.append(row)
    dataset = np.array(dataset)
    numpy_dataset = dataset.astype(np.float)
    featureNumber = numpy_dataset.shape[1]
    NumpyArraryFirstHalf = numpy_dataset[:, 0:int(featureNumber/2)]
    NumpyArraryLastHalf = numpy_dataset[:, int(featureNumber / 2):]
    randomNumpyArrary = np.random.randint(0, 100, (NumpyArraryFirstHalf.shape[0], NumpyArraryFirstHalf.shape[1]))
    randomNumpyArrary = np.multiply(NumpyArraryFirstHalf,100)-randomNumpyArrary
    numpy_dataset = np.hstack([randomNumpyArrary, NumpyArraryLastHalf])
    numpy_dataset[:, :-1] = preprocessing.scale(numpy_dataset[:, :-1])
    for row in numpy_dataset:
        mydict = {v: k for v, k in enumerate(row)}
        mydict['class_label'] = mydict.pop(len(mydict.keys()) - 1)
        return_dataset.append(mydict)
    return return_dataset

from collections import Counter
from imblearn.datasets import fetch_datasets


# satimage (6435, 36), [(-1, 5809), (1, 626)]
# car_eval_4 (1728, 21), [(-1, 1663), (1, 65)]
# wine_quality (4898, 11), [(-1, 4715), (1, 183)]
# letter_img (20000, 16), [(-1, 19266), (1, 734)]
# yeast_me2 (1484, 8), [(-1, 1433), (1, 51)]
# ozone_level (2536, 72), [(-1, 2463), (1, 73)]
def readSatimage():
    dataset = []
    return_dataset = []
    with open(satimage) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            row = np.array(row[0].split()).astype(np.float)
            dataset.append(row)
    dataset = np.array(dataset)
    numpy_dataset = dataset.astype(np.float)
    numpy_dataset[:, :-1] = preprocessing.scale(numpy_dataset[:, :-1])
    for row in numpy_dataset:
        mydict = {v: k for v, k in enumerate(row)}
        mydict['class_label'] = 1 if mydict.pop(36) == 4 else -1
        return_dataset.append(mydict)
    return return_dataset


def readLetter():
    dataset = []
    return_dataset = []
    with open(letter) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            row[0] = 1 if row[0] == "A" else -1
            dataset.append(row)
    dataset = np.array(dataset)
    numpy_dataset = dataset.astype(np.float)
    numpy_dataset[:, 1:] = preprocessing.scale(numpy_dataset[:, 1:])
    for row in numpy_dataset:
        mydict = {v: k for v, k in enumerate(row)}
        mydict['class_label'] = mydict.pop(0)
        return_dataset.append(mydict)
    return return_dataset


def read_standard_file(file):
    return_dataset = []
    with open(file, "r") as f:
        for line in tqdm(f, desc="read standard file"):
            arrays = line.split()
            mydict = {"class_label": int(arrays[0])}
            for item in arrays[1:]:
                tmp_item = item.split(":")
                mydict.update({int(tmp_item[0]):np.float(tmp_item[1])})
            return_dataset.append(mydict)
            del mydict
    f.close()
    return return_dataset


def readEN_FR():
    return_dataset = []
    if os.path.exists(EN_FR + "_standard.txt"):
        return_dataset = read_standard_file(EN_FR + "_standard.txt")
    else:
        X, y = load_svmlight_file(EN_FR + ".txt")
        numpy_dataset = X.toarray()
        standard_dataset = preprocessing.scale(numpy_dataset)
        with open(EN_FR + "_standard.txt", "w", encoding="utf-8") as f:
            for i, row in enumerate(tqdm(standard_dataset, desc="read EN_FR")):
                mydict = {v: k for v, k in enumerate(row)}
                for index in range(len(numpy_dataset[i])):
                    if numpy_dataset[i][index] == 0:
                        del mydict[index]
                line = str(y[i])
                for key, value in mydict.items():
                    line += " " + str(key) + ":" + str(value)
                f.write(line + "\n")
                mydict['class_label'] = y[i]
                return_dataset.append(mydict)
            f.close()

    return return_dataset


def readIMDB():
    return_dataset = []
    if os.path.exists(IMDB + "_standard.txt"):
        return_dataset = read_standard_file(IMDB + "_standard.txt")
    else:
        X, y = load_svmlight_file(IMDB + ".txt")
        numpy_dataset = X.toarray()
        standard_dataset = preprocessing.scale(numpy_dataset)
        with open(IMDB + "_standard.txt", "w", encoding="utf-8") as f:
            for i, row in enumerate(tqdm(standard_dataset, desc="read IMDB")):
                mydict = {v: k for v, k in enumerate(row)}
                for index in range(len(numpy_dataset[i])):
                    if numpy_dataset[i][index] == 0:
                        del mydict[index]
                label = 1 if y[i] > 5 else -1
                line = str(label)
                for key, value in mydict.items():
                    line += " " + str(key) + ":" + str(value)
                f.write(line + "\n")
                mydict['class_label'] = label
                return_dataset.append(mydict)
            f.close()

    return return_dataset


def readHAPT():
    dataset = []
    return_dataset = []
    with open(HAPT) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            dataset.append(row)
    dataset = np.array(dataset)
    numpy_dataset = dataset.astype(np.float)
    # numpy_dataset[:,1:]=preprocessing.scale(numpy_dataset[:,1:])
    for row in numpy_dataset:
        mydict = {v: k for v, k in enumerate(row)}
        mydict[0] = 1 if mydict[0] <= 6 else -1
        mydict['class_label'] = mydict.pop(0)
        return_dataset.append(mydict)
    return return_dataset


def readIsolet():
    dataset = []
    return_dataset = []
    with open(isolet) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            dataset.append(row)
    dataset = np.array(dataset)
    numpy_dataset = dataset.astype(np.float)
    numpy_dataset[:, :-1] = preprocessing.scale(numpy_dataset[:, :-1])
    for row in numpy_dataset:
        mydict = {v: k for v, k in enumerate(row)}
        last_colunm = len(mydict.keys()) - 1
        mydict[last_colunm] = 1 if mydict[last_colunm] == 1 else -1
        mydict['class_label'] = mydict.pop(last_colunm)
        return_dataset.append(mydict)
    return return_dataset


def readAustralian():
    dataset = []
    return_dataset = []
    with open(australian) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            dataset.append(row)
    dataset = np.array(dataset)
    numpy_dataset = dataset.astype(np.float)
    numpy_dataset[:, :-1] = preprocessing.scale(numpy_dataset[:, :-1])
    for row in numpy_dataset:
        mydict = {v: k for v, k in enumerate(row)}
        last_colunm = len(mydict.keys()) - 1
        mydict[last_colunm] = 1 if mydict[last_colunm] == 1 else -1
        mydict['class_label'] = mydict.pop(last_colunm)
        return_dataset.append(mydict)
    return return_dataset


def readCreditA():
    dataset = []
    return_dataset = []
    with open(credit_a) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            dataset.append(row)
    dataset = np.array(dataset)
    numpy_dataset = dataset.astype(np.float)
    numpy_dataset[:, :-1] = preprocessing.scale(numpy_dataset[:, :-1])
    for row in numpy_dataset:
        mydict = {v: k for v, k in enumerate(row)}
        mydict['class_label'] = mydict.pop(len(mydict.keys()) - 1)
        return_dataset.append(mydict)
    return return_dataset


def readCreditG():
    dataset = []
    return_dataset = []
    with open(credit_g) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            dataset.append(row)
    dataset = np.array(dataset)
    numpy_dataset = dataset.astype(np.float)
    numpy_dataset[:, :-1] = preprocessing.scale(numpy_dataset[:, :-1])
    for row in numpy_dataset:
        mydict = {v: k for v, k in enumerate(row)}
        mydict['class_label'] = mydict.pop(len(mydict.keys()) - 1)
        return_dataset.append(mydict)
    return return_dataset


def readDiabetes():
    dataset = []
    return_dataset = []
    with open(diabetes) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            dataset.append(row)
    dataset = np.array(dataset)
    numpy_dataset = dataset.astype(np.float)
    numpy_dataset[:, :-1] = preprocessing.scale(numpy_dataset[:, :-1])
    for row in numpy_dataset:
        mydict = {v: k for v, k in enumerate(row)}
        last_colunm = len(mydict.keys()) - 1
        mydict[last_colunm] = 1 if mydict[last_colunm] == 1 else -1
        mydict['class_label'] = mydict.pop(last_colunm)
        return_dataset.append(mydict)
    return return_dataset


def readDna12():
    dataset = []
    return_dataset = []
    with open(dna12) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            dataset.append(row)
    dataset = np.array(dataset)
    numpy_dataset = dataset.astype(np.float)
    numpy_dataset[:, :-1] = preprocessing.scale(numpy_dataset[:, :-1])
    for row in numpy_dataset:
        mydict = {v: k for v, k in enumerate(row)}
        mydict['class_label'] = mydict.pop(len(mydict.keys()) - 1)
        return_dataset.append(mydict)
    return return_dataset


def readKrVsKp():
    dataset = []
    return_dataset = []
    with open(kr_vs_kp) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            dataset.append(row)
    dataset = np.array(dataset)
    numpy_dataset = dataset.astype(np.float)
    numpy_dataset[:, :-1] = preprocessing.scale(numpy_dataset[:, :-1])
    for row in numpy_dataset:
        mydict = {v: k for v, k in enumerate(row)}
        mydict['class_label'] = mydict.pop(len(mydict.keys()) - 1)
        return_dataset.append(mydict)
    return return_dataset


def readSplice():
    dataset = []
    return_dataset = []
    with open(splice) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            dataset.append(row)
    dataset = np.array(dataset)
    numpy_dataset = dataset.astype(np.float)
    numpy_dataset[:, :-1] = preprocessing.scale(numpy_dataset[:, :-1])
    for row in numpy_dataset:
        mydict = {v: k for v, k in enumerate(row)}
        mydict['class_label'] = mydict.pop(len(mydict.keys()) - 1)
        return_dataset.append(mydict)
    return return_dataset


def readUrlNormalized(file_number):
    filename = url + "Day" + str(file_number) + ".svm"
    dataset = []
    with open(filename) as f:
        for line in f:
            line_dict = {}
            x = (line.rstrip()[3:]).split()
            y = int(line[:3])
            for elem in x:
                elem_list = elem.split(":")
                line_dict[int(elem_list[0])] = float(elem_list[1])
            line_dict['class_label'] = int(y)
            dataset.append(line_dict)
    return dataset


def readSvmguide3Normalized():  # already normalized
    return_dataset = []
    X, y = load_svmlight_file(svmguide3)
    numpy_dataset = X.toarray()
    # numpy_dataset=dataset.astype(np.float)
    numpy_dataset = preprocessing.scale(numpy_dataset)
    i = 0
    for row in numpy_dataset:
        mydict = {v: k for v, k in enumerate(row)}
        mydict['class_label'] = y[i]
        return_dataset.append(mydict)
        i += 1
    return return_dataset


def readA8ANormalized():  # already normalized
    return_dataset = []
    X, y = load_svmlight_file(a8a_training)
    numpy_dataset = X.toarray()
    # numpy_dataset=dataset.astype(np.float)
    numpy_dataset = preprocessing.scale(numpy_dataset)
    i = 0
    for row in numpy_dataset:
        mydict = {v: k for v, k in enumerate(row)}
        mydict['class_label'] = y[i]
        return_dataset.append(mydict)
        i += 1
    return return_dataset


def readSpambaseNormalized():
    dataset = []
    return_dataset = []
    with open(spambase) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            dataset.append(row)
    dataset = np.array(dataset)
    numpy_dataset = dataset.astype(np.float)
    numpy_dataset[:, :-1] = preprocessing.scale(numpy_dataset[:, :-1])
    for row in numpy_dataset:
        mydict = {v: k for v, k in enumerate(row)}
        if mydict[57] == 0:
            mydict[57] = -1
        mydict['class_label'] = mydict.pop(57)
        return_dataset.append(mydict)
    return return_dataset


def readGermanNormalized():
    dataset = []
    return_dataset = []
    with open(german) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            row = row[0].split(" ")
            row = list(filter(None, row))  # fastest
            dataset.append(row)
    dataset = np.array(dataset)
    numpy_dataset = dataset.astype(np.float)
    numpy_dataset[:, :-1] = preprocessing.scale(numpy_dataset[:, :-1])
    for row in numpy_dataset:
        mydict = {v: k for v, k in enumerate(row)}
        if mydict[24] == 2:
            mydict[24] = -1
        mydict['class_label'] = mydict.pop(24)
        return_dataset.append(mydict)
    return return_dataset


def readIonosphereNormalized():
    dataset = []
    return_dataset = []
    with open(ionosphere) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            dataset.append(row)
    dataset = np.array(dataset)
    for row in dataset:
        if row[34] == 'b':
            row[34] = -1
        else:
            row[34] = 1
    numpy_dataset = dataset.astype(np.float)
    numpy_dataset[:, :-1] = preprocessing.scale(numpy_dataset[:, :-1])
    for row in numpy_dataset:
        mydict = {v: k for v, k in enumerate(row)}
        mydict['class_label'] = mydict.pop(34)
        return_dataset.append(mydict)
    return return_dataset


def readMagicNormalized():
    dataset = []
    return_dataset = []
    with open(magic04) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            dataset.append(row)
    dataset = np.array(dataset)
    for row in dataset:
        if row[10] == 'h':
            row[10] = 1
        else:
            row[10] = -1
    numpy_dataset = dataset.astype(np.float)
    numpy_dataset[:, :-1] = preprocessing.scale(numpy_dataset[:, :-1])
    for row in numpy_dataset:
        mydict = {v: k for v, k in enumerate(row)}
        mydict['class_label'] = mydict.pop(10)
        return_dataset.append(mydict)
    return return_dataset


def readWdbcNormalized():
    dataset = []
    return_dataset = []
    with open(wdbc) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            dataset.append(row)
    dataset = np.array(dataset)
    dataset = dataset[:, 1:]
    numpy_dataset = dataset.astype(np.float)
    numpy_dataset[:, 1:] = preprocessing.scale(numpy_dataset[:, 1:])
    for row in numpy_dataset:
        mydict = {v: k for v, k in enumerate(row)}
        mydict['class_label'] = mydict.pop(0)
        return_dataset.append(mydict)
    return return_dataset


def readWpbcNormalized():
    dataset = []
    return_dataset = []
    with open(wpbc) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            dataset.append(row)
    dataset = np.array(dataset)
    dataset = dataset[:, 1:]
    numpy_dataset = dataset.astype(np.float)
    numpy_dataset[:, 1:] = preprocessing.scale(numpy_dataset[:, 1:])
    # for row in numpy_dataset:
    #     mydict = {v: k for v, k in enumerate(row)}
    #     mydict['class_label'] = mydict.pop(0)
    #     return_dataset.append(mydict)
    return_dataset = NumpyArrary2Dict(numpy_dataset, 1, 1)
    return return_dataset


def readWbcNormalized():
    dataset = []
    return_dataset = []
    with open(wbc) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            dataset.append(row)
    dataset = np.array(dataset)
    dataset = dataset[:, 1:]
    dataset[dataset=="?"]=1
    numpy_dataset = dataset.astype(np.float)
    numpy_dataset[:, :-1] = preprocessing.scale(numpy_dataset[:, :-1])
    # for row in numpy_dataset:
    #     mydict = {v: k for v, k in enumerate(row)}
    #     label = mydict.pop(len(row) - 1)
    #     if int(label) == 2:
    #         mydict['class_label'] = -1
    #     elif int(label) == 4:
    #         mydict['class_label'] = 1
    #     return_dataset.append(mydict)
    return_dataset = NumpyArrary2Dict(numpy_dataset, -1, 4)
    return return_dataset

def NumpyArrary2Dict(dataset, index, classes):
    data = []
    for row in dataset:
        mydict = {k:v for k,v in enumerate(row)}
        label = mydict.pop(0 if index is 1 else len(row)-1)
        mydict["class_label"] = 1.0 if label==classes or label==np.float(classes) else -1.0
        data.append(mydict)
    return data




# def removeRandomData(dataset): #remove based on removing percentage
# #     for row in dataset:
# #         for i in list(row):
# #             if random.random()<p.remove_percent:
# #                 if i!='class_label':
# #                     row.pop(i)
# #     return dataset

# def removeRandomData(dataset):  # remove based on removing percentage
#     for row in dataset:
#         keys = list(row.keys())
#         keys.remove("class_label")
#         for key in random.sample(keys, int(p.remove_percent * len(keys))):
#             row.pop(key)
#     return dataset

def removeRandomData(data):
    maxFeatureNum = len(data[0])
    featureKept = math.ceil(len(data[0]) * p.remove_percent)
    chunkSize = math.ceil(len(data) * 0.5)
    howSparseLength = featureKept
    for (i, vec) in enumerate(data):
        if (i+1) % chunkSize == 0:
            howSparseLength = min(howSparseLength + featureKept, maxFeatureNum)
        rDelSamples = random.sample(range(maxFeatureNum), howSparseLength)
        for k, v in vec.copy().items():
            if k not in rDelSamples:
                if k!="class_label":
                    del vec[k]
    return data

def removeDataTrapezoidal(original_dataset):  # trapezoidal
    dataset = original_dataset[:]
    features = len(dataset[0])
    rows = len(dataset)
    for i in range(0, len(dataset)):
        multiplier = int(i / (rows / 10)) + 1
        increment = int(features / 10)
        features_left = multiplier * increment
        if (i == len(dataset) - 1):
            features_left = features - 2
        for key, value in dataset[i].copy().items():
            if key != 'class_label' and key > features_left:
                dataset[i].pop(key)
    return dataset



def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

crta = './otherDatasets/crt-a.csv'
def readAustralian3():
    dataset=[]
    with open(crta) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            dataset.append(row)
    dataset = np.array(dataset)
    label = np.array([1 if i=="+" else -1 for i in dataset[:,-1]])
    dataset[dataset=="t"] = 1
    dataset[dataset == "f"] = 0
    for colIndex in range(dataset.shape[1]-1):
        col = dataset[:,colIndex]
        unique_elements, counts_elements = np.unique(col, return_counts=True)
        unique_elements, counts_elements = unique_elements.tolist(), counts_elements.tolist()
        isContinue = [is_number(element) for element in unique_elements]
        if any(isContinue)==True:   # 是否是连续值
            if False in isContinue: # 是否有非连续值，如：？
                copyCol = copy.deepcopy(col.tolist())
                while "?" in copyCol: copyCol.remove("?")
                copyCol = np.array(copyCol).astype(np.float)
                mean = np.mean(copyCol)
                col[col=="?"]=mean
        else:
            if "?" in unique_elements:
                counts_elements.pop(unique_elements.index("?"))
                unique_elements_copy = copy.deepcopy(unique_elements)
                unique_elements_copy.pop(unique_elements_copy.index("?"))
                unique_elements_num = [i for i in range(len(unique_elements_copy))]
                maxElementIndex = counts_elements.index(max(counts_elements))
                maxElement = unique_elements_num[maxElementIndex]
                col[col=="?"] = maxElement
                for ele,arr in zip(unique_elements_copy,unique_elements_num):
                    col[col==ele] = arr
            else:
                unique_elements_num = [i for i in range(len(unique_elements))]
                for ele,arr in zip(unique_elements,unique_elements_num):
                    col[col==ele] = arr
        col = np.array(col).astype(np.float)
    dataset = np.array(np.hstack((dataset[:,:-1],label.reshape(-1,1)))).astype(np.float)
    from sklearn import preprocessing
    dataset[:,:-1] = preprocessing.scale(dataset[:,:-1])
    return dataset

def removeDataEvolvable(dataset):
    from math import floor
    from copy import deepcopy
    rows, features = dataset.shape
    # np.random.seed(4)
    gaussianRandomOperator = np.random.standard_normal((features-1, int((features - 1) * 0.7)))#返回指定形状的标准正态分布数组
    newDataset = dataset[:, :-1].dot(gaussianRandomOperator)    # 生成映射数据
    datasetx5 = dataset
    newDatasetx5 = newDataset
    # for i in range(p.fesl_epoch-1):
    #     permutation = np.random.permutation(rows)
    #     shuffled_dataset = dataset[permutation, :]
    #     shuffled_newdataset = newDataset[permutation, :]
    #     datasetx5 = np.vstack((datasetx5,shuffled_dataset))
    #     newDatasetx5 = np.vstack((newDatasetx5, shuffled_newdataset))
    # T1 = int(np.floor(rows*p.fesl_epoch / 10))
    # T2 = rows*p.fesl_epoch - T1
    T1 = int(np.floor(rows / 10))
    T2 = rows - T1
    label = np.vstack([datasetx5[:T1,-1].reshape(-1,1), datasetx5[T1:,-1].reshape(-1,1)]).reshape(-1,)
    X1, X2, T1_B, T2_B = datasetx5[:T1,:-1], newDatasetx5[T1:,:], datasetx5[T1 - p.fesl_B:T1,:-1], newDatasetx5[T1 - p.fesl_B:T1,:]
    return X1, X2, T1_B, T2_B, label

from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore", message="Numerical issues were encountered ")

def scale(X1, X2, overlap):
    returnDataset = []
    featureLength = [data.shape[1] for data in [X1, X2, overlap]]
    maxFeature = max(featureLength)
    rows, features = X1.shape
    zeroMatrix = np.zeros((rows,maxFeature))
    zeroMatrix[:,:features] = zeroMatrix[:,:features]+X1
    rowsB, featuresB = overlap.shape
    overlap = preprocessing.scale(overlap)
    numpyData = np.vstack((zeroMatrix,overlap))
    rows2, features2 = X2.shape
    zeroMatrix = np.zeros((rows2,maxFeature))
    zeroMatrix[:,maxFeature-features2:] = zeroMatrix[:,maxFeature-features2:]+X2
    numpyData = np.vstack((numpyData, zeroMatrix))
    numpyData = preprocessing.scale(numpyData)
    # for row in scaleData[:rows, :features]:
    #     mydict = {k: v for k, v in enumerate(row)}
    #     returnDataset.append(mydict)
    # for row in scaleData[rows+1:rows+rowsB,:]:
    #     mydict = {k:v for k,v in enumerate(row)}
    #     returnDataset.append(mydict)
    # for row in scaleData[rows+rowsB+1:,maxlength-features2,:]:
    #     mydict = {features+k:v for k,v in enumerate(row)}
    #     returnDataset.append(mydict)
    for index, row in enumerate(numpyData):
        if index<rows:
            mydict = {k:v for k,v in enumerate(row[:features])}
        elif index>=rows and  index<rows+rowsB:
            mydict = {k:v for k,v in enumerate(row)}
        else:
            mydict = {features+k:v for k,v in enumerate(row[maxFeature-features2:])}
        returnDataset.append(mydict)
    return returnDataset

def scale1(X1,X2,T1_B,T2_B):
    returnDataset = []
    scale_T2 = preprocessing.scale(np.vstack((T2_B, X2)))
    T2_B = scale_T2[:len(T2_B), :]
    X2 = scale_T2[len(T2_B):, :]
    overlap = np.hstack((T1_B, T2_B))
    featureLength = [data.shape[1] for data in [X1, X2, overlap]]
    maxFeature = max(featureLength)
    rows, features = X1.shape
    zeroMatrix = np.zeros((rows, maxFeature))
    zeroMatrix[:, :features] = zeroMatrix[:, :features] + X1
    rowsB, featuresB = overlap.shape
    numpyData = np.vstack((zeroMatrix, overlap))
    rows2, features2 = X2.shape
    zeroMatrix = np.zeros((rows2, maxFeature))
    zeroMatrix[:, maxFeature - features2:] = zeroMatrix[:, maxFeature - features2:] + X2
    numpyData = np.vstack((numpyData, zeroMatrix))
    # numpyData = preprocessing.scale(numpyData)
    for index, row in enumerate(numpyData):
        if index < rows:
            mydict = {k: v for k, v in enumerate(row[:features])}
        elif index >= rows and index < rows + rowsB:
            mydict = {k: v for k, v in enumerate(row)}
        else:
            mydict = {features + k: v for k, v in enumerate(row[maxFeature - features2:])}
        returnDataset.append(mydict)
    return returnDataset











