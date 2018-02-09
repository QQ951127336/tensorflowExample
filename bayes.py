# coding:utf-8
import csv
def loadCsv(filename):
    lines = csv.reader(open(filename,"r",encoding="utf-8"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset
filename = 'pima-indians-diabetes.data.csv'
dataset = loadCsv(filename)
print(dataset)
print("Loaded data file "+filename+" with "+str(len(dataset))+" rows")
import random

def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset)* splitRatio)
    trainSet = []
    copy  = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet,copy]
dataset = [[1],[2],[3],[4],[5]]
splitRatio = 0.8
train , test = splitDataset(dataset,splitRatio)
print(str(len(train))+ "    "+str(len(test)))
def seperateByClass (dataset):
    seperated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in seperated):
            seperated[vector[-1]] = []
        seperated[vector[-1]].append(vector)
    return seperated

dataset = [[1,20,1],[2,21,0],[3,22,1]]
seperated = seperateByClass(dataset)
print(seperated)
import math
def mean(numbers):
    return sum(numbers)/float(len(numbers))
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers) -1)
    return math.sqrt(variance)
numbers =[1,2,3,4,5]
print(" "+str(mean(numbers))+" "+str(stdev(numbers)))

def summarize(dataset):
    summaries = [(mean(attribute),stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries

dataset = [[1,20,0],[2,21,1],[3,22,0]]

summary = summarize(dataset)
print(summary)


def summarizeByClass(dataset):
    seperated = seperateByClass(dataset)
    summaries = {}
    for classValue, instances in seperated.items():
        summaries[classValue] = summarize(instances)
    return summaries

dataset = [[1,20,1],[2,21,0],[3,22,1],[4,22,0]]
summary = summarizeByClass(dataset)
print(summary)

def calculateProbability(x,mean,stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1/(math.sqrt(2*math.pi)*stdev))*exponent
x = 71.5
mean = 73
stdev = 6.2
probability = calculateProbability(x,mean,stdev)
print(probability)

def calculateClassProbabilities(summaries,inputVector):
    probabilities = {}
    for classValue ,classSummeries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummeries)):
            mean,stdev = classSummeries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x,mean,stdev)
    return probabilities
summaries = {0:[(1,0.5)],1:[(20,5.0)]}
inputVector = [1.1,'?']
probabilities = calculateClassProbabilities(summaries,inputVector)
print(probabilities)

def predict(summaries , inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel,bestProb = None,-1
    for classValue,probability in probabilities.items():
        if bestLabel is None or probability> bestProb:
            bestProb = probability
            bestLabel = classValue
    return  bestLabel

summaries = {'A':[(1,0.5)] , 'B':[(20 , 5.0)]}
inputVector = [1.1 , '?']
result = predict(summaries,inputVector)
print(result)

def getPredictions(summaries,testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries,testSet[i])
        predictions.append(result)
    return predictions

summaries = {'A':[(1,0.5)], 'B':[(20,5.0)]}
testSet = [[1.1,'?'],[19.1,'?']]
predictions = getPredictions(summaries,testSet)
print(predictions)

def getAccurancy(testSet,predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return(correct/float(len(testSet))) * 100.0

testSet = [[1,1,1,'a'],[2,2,2,'a'],[3,3,3,'b']]
predictions = ['a','a','a']
accurany = getAccurancy(testSet,predictions)
print(accurany)
print([1,'?'])