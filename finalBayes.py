# coding:utf-8
import csv
import random
import math

def loadCsv(filename):
    lines = csv.reader(open(filename,'r'))
    dataset = list(lines)
    for i in range(dataset):
        dataset[i] = [float(x) for x in dataset[i]]
    return  dataset

def splitDataset(dataset, splitRatio):
    trainSize = int (len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize :
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet,copy]

def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        