# coding:utf-8
import csv
import random
import math
#读取文件
def loadCsv(filename):
    lines = csv.reader(open(filename,'r'))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return  dataset
#将数据分为训练集 和 测试集
def splitDataset(dataset, splitRatio):
    trainSize = int (len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize :
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet,copy]
#按类分类
def seperateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1]  not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated
#平均数
def mean(numbers):
    return sum(numbers)/float(len(numbers))
#方差
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers) - 1)
    return  math.sqrt(variance)
#求特征值
def summarize(dataset):
    summaries = [(mean(attribute),stdev(attribute) )for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries
#按类求特征值
def summarizeByClass(dataset):
    seperated = seperateByClass(dataset)
    summaries = {}
    for classValue,instance in seperated.items():
        summaries[classValue] = summarize(instance)
    return summaries
#计算测试数据在该类的可能性
def calculateProbability(x,mean,stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return  (1/(math.sqrt(2*math.pi)*stdev))*exponent
#按类计算可能性
def calculateClassProbabilities(summaries,inputVector):
    probabilities = {}
    for classValue , classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean,stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x,mean,stdev)
    return probabilities
#单一属性预测
def predic(summaries ,inputVector):
    probabilities = calculateClassProbabilities(summaries,inputVector)
    bestLabel ,bestProb = None, -1
    for classValue, probabilities in probabilities.items():
        if bestProb is None or probabilities > bestProb:
            bestProb = probabilities
            bestLabel = classValue
    return bestLabel
#综合预测
def getPredictions(summaries,testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predic(summaries,testSet[i])
        predictions.append(result)
    return predictions
#测试精度
def getAccurancy(testSet,predictions):
    correct = 0
    for i in range(len(testSet)) :
        if testSet[i][-1] == predictions[i]:
            correct+= 1
    return (correct/float(len(testSet))) * 100.0
#主函数
def main():
    filename = 'pima-indians-diabetes.data.csv'
    splitRatio = 0.7
    dataset = loadCsv(filename)
    trainingSet , testSet = splitDataset(dataset,splitRatio)
    summaries = summarizeByClass(trainingSet)
    predictions = getPredictions(summaries,testSet)
    accuracy = getAccurancy(testSet,predictions)
    print(accuracy)
main()