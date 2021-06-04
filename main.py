import pandas as pd
import numpy as np
import random
import math
import time
import copy

def euclideanDistance(x,y):
    distance = (x-y)**2
    return distance

def nearestNeighbor(instance, data, currentSetOfFeatures, feature):
    # number of correct instances calculate
    #instance is the row we are using to calculate nearest neighbor
    #data[instance][feature] = the row and feature to be used in calculating nearest neighbor
    nearest = 9999
    correct = 0
    indexPrediction = -1
    # print(len(data)) <-- prints 100
    # print(data[instance][feature])
    for i in range(len(data)):
        # print(i) <-- 0...99
        if(i == instance):
            continue
        distance = 0.0
        # check to see if there are any other features to use for distance
        if(len(currentSetOfFeatures) > 0):
            for j in currentSetOfFeatures:
                distance += euclideanDistance(data[instance][j],data[i][j])
        # adding new feature distance
        distance += euclideanDistance(data[instance][feature],data[i][feature])
        if(distance < nearest):
            nearest = distance
            indexPrediction = i
    # print(data[instance][1],data[i][1])
    if(data[instance][0] == data[indexPrediction][0]):
        correct = 1
    return correct

def leaveOneOutValidator(data, currentSetOfFeatures, feature):
    # Train our data without instance 0 and by getting the accuracy of each feature set at the start 
    correct = 0
    for i in range(len(data)):
        correct += nearestNeighbor(i, data, currentSetOfFeatures, feature)
    accuracy = correct/len(data)
    return accuracy

def featureSearchForward(dataSet, normal):
    # We initialize an empty set to keep track of features
    start1 = time.perf_counter()
    timeFlag = 1
    currentSetOfFeatures = []
    bestFeatures = []
    globalAccuracy = 0
    
    # Using pandas and numpy we convert and normalize the data set
    print(pd.read_csv(dataSet, delim_whitespace=True, header=None))
    if(normal):
        data = normalizeData(dataSet)
    else:
        data = pd.read_csv(dataSet, delim_whitespace=True, header=None).to_numpy()
        # print(data)

    # Traverse through the search tree
    print(len(data[0]))
    for i in range(1,len(data[0])):
        print('On the %dth level' % (i))
        featureToAdd= []
        currentBestAccuracy = 0
        accuracy = 0

        for j in range(1,len(data[0])):
            if(j not in currentSetOfFeatures):
                accuracy = leaveOneOutValidator(data, currentSetOfFeatures, j)
                print(' -Current Features %s Considering adding the %d feature - %%%s'%(currentSetOfFeatures,j,accuracy))
            if accuracy > currentBestAccuracy:
                currentBestAccuracy = accuracy
                featureToAdd = j
            if accuracy > globalAccuracy:
                globalAccuracy = accuracy
        print('On level %d, I added feature %d to current set with accuracy %%%s' % (i,featureToAdd,currentBestAccuracy))
        currentSetOfFeatures.append(featureToAdd)
        if(currentBestAccuracy >= globalAccuracy):
            bestFeatures.append(featureToAdd)
        if(currentBestAccuracy < globalAccuracy):
            print('---WARNING, Accuracy has decreased! Continuing search in case of local maxima---')
            if(timeFlag):
                finish1 = time.perf_counter()
                print(f"Time of completion: {finish1-start1:0.4f}s")
                timeFlag = 0
    print('Best Accuracy %%%s using features %s' % (globalAccuracy,bestFeatures))
    return

def featureSearchBackward(dataSet, normal):
    # We initialize an empty set to keep track of features
    start1 = time.perf_counter()
    timeFlag = 1
    bestFeatures = []
    globalAccuracy = 0
    featuresRemoved = []
    

    # Using pandas and numpy we convert and normalize the data set
    print(pd.read_csv(dataSet, delim_whitespace=True, header=None))
    if(normal):
        data = normalizeData(dataSet)
    else:
        data = pd.read_csv(dataSet, delim_whitespace=True, header=None).to_numpy()
        # print(data)

    currentSetOfFeatures = [i for i in range(1,len(data[0]))]
    # Traverse through the search tree
    for i in range(1,len(data[0])):
        print('On the %dth level' % (i))
        featureToRemove = 0
        currentWorstAccuracy = 9999
        accuracy = 0
        
        for j in range(1,len(data[0])):
            if j in featuresRemoved:
                continue
            if(len(currentSetOfFeatures) > 0):
                accuracy = leaveOneOutValidator(data, currentSetOfFeatures, j)
                print(' -Current Features %s Considering removing the %d feature - %%%s'%(currentSetOfFeatures,j,accuracy))
            else:
                continue
            if accuracy < currentWorstAccuracy:
                currentWorstAccuracy = accuracy
                featureToRemove = j
            if accuracy > globalAccuracy:
                globalAccuracy = accuracy
        finish1 = time.perf_counter()
        bestFeatures.append((copy.deepcopy(currentSetOfFeatures),currentWorstAccuracy,time.perf_counter()))
        if featureToRemove:
            print('On level %d, I removed feature %s to current set with accuracy %%%s' %(i,featureToRemove,currentWorstAccuracy))
            currentSetOfFeatures.remove(featureToRemove)
            featuresRemoved.append(featureToRemove)
            # print(bestFeatures)
        else:
            print('No feature removed')
        # if(currentWorstAccuracy >= globalAccuracy):
        #     bestFeatures.append(currentSetOfFeatures)
        # if(currentWorstAccuracy < globalAccuracy):
        #     print('---WARNING, Accuracy has decreased! Continuing search in case of local maxima---')
        #     if(timeFlag):
        #         finish1 = time.perf_counter()
        #         print(f"Time of completion: {finish1-start1:0.4f}s")
        #         timeFlag = 0
    bestFeatures.sort(key=lambda tup: tup[1])
    print('Best Accuracy %%%s using features %s' % (globalAccuracy,bestFeatures[-1][0]))
    print(f"Time of completion: {bestFeatures[-1][2]-start1:0.4f}s")
    return

def normalizeData(dateSet):
    #normalize data with z-score
    data = pd.read_csv(dataSet, delim_whitespace=True, header=None)
    for i in range(1, len(data.columns)):
        data[i] = (data[i]-data[i].mean())/data[i].std()
    # print(data)
    return data.to_numpy()

if __name__ == '__main__':
    dataSet = input("select data set: \n 1.)small\n 2.)large\n")
    if dataSet == '1':
        dataSet = 'CS205_small_testdata__14.txt'
    else:
        dataSet = 'CS205_large_testdata__30.txt'
    normal = input("Do you want to normalize: \n 1.)yes\n 2.)no\n")
    if normal == '1':
        normal = 1
    else:
        normal = 2
    search = input("Please select your Search: \n1) Forward Selection \n2) Back Elimination\n")
    start = time.perf_counter()
    if search == '1':
        featureSearchForward(dataSet, normal)
    else:
        featureSearchBackward(dataSet, normal)
    finish = time.perf_counter()
    print(f"Time of completion: {finish-start:0.4f}s")
    # featureSearchBackward(dataSet)
