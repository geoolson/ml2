import pandas as pd
import numpy as np
import sys

learningRates = (0.01,0.1,1.0) # the three different learning rates
correct = 0.0 # for keeping track the number of times a perceptron identifies its target
testCorrect = 0.0 # same but for the test data
nums = 0 # keeping track of the number of targets the perceptron is supposed to recognize
testNums = 0 # same for the test data

TARGET = 0 # A constant labeling which element of the data tuple contains the targets

def setWeights():
    weights = np.random.rand(785) #creates array of floats from 0-1
    f = lambda x: x * .1 - 0.05 # lambda function to shift the range from -.05 to .05
    weights = f(weights) # apply the lambda to the array
    return weights

# returns a tuple of (targets, mxn array of processed images)
def preProcessData(data):
    data = data.sample(frac=1).reset_index(drop=True) #shuffle data
    processedData = data.drop(0, axis=1).applymap(lambda x: x/255.0)
    rowC = len(processedData.index)
    ones = np.ones(rowC)
    return ( data[data.columns[0]], np.column_stack( (ones, processedData.values) ) )

# output function for the perceptron
def out(weights, inputData):
    y = np.dot(weights, inputData)
    if y > 0 :
        return 1
    else:
        return 0

# checks first if the perceptron mis-identified the image then adjusts the weights
def adjustWeights(weights, target, out, inputData, rate):
    global correct
    global nums
    if target == perceptNum and out == 1 :
        correct += 1
        nums += 1
        return weights
    elif target != perceptNum and out == 0 :
        return weights
    if target == perceptNum :
        nums += 1
        t = 1
    else :
        t = 0
    lhs = rate * (t - out)
    deltaWeight = np.multiply(lhs, inputData)
    return np.add( weights, deltaWeight )

if __name__ == '__main__':
    # arguments to feed the script 1st argument will be the number of epochs
    # and the second argument is which learning rate
    epochs = int(sys.argv[1])
    rateIndex = int(sys.argv[2])
    
    #setting up variables for training data 
    dataFile = pd.read_csv("../data/mnist_train.csv", header=None)
    trainingData = preProcessData(dataFile)
    confusionMatrix = np.int_(np.zeros((10,10)))
    accuracies = np.zeros((10,epochs))
    
    #setting up variables for test data
    testFile = pd.read_csv("../data/mnist_test.csv", header=None)
    testData = preProcessData(testFile)
    testConfusionMatrix = np.int_(np.zeros((10,10)))
    testAccuracies = np.zeros((10,epochs))

    #initializing a list of 10 sets of weights for each perceptron
    weightsk = []
    for i in range(10) : 
        weightsk.append(setWeights())

    for perceptNum in range(10) :
        weights = weightsk[perceptNum]
        for x in range(epochs) :
            # training the perceptrons through the training data for user decided number of epochs
            if x == epochs-1 : 
                # this loop adds data to the confusion matrix on the last epoch
                for i in range(60000) :
                    perceptOut = out(weights, trainingData[1][i,:])
                    if perceptOut == 1 :
                        confusionMatrix[perceptNum][trainingData[TARGET][i]] += 1
                    weights = adjustWeights(weights, trainingData[TARGET][i], perceptOut, trainingData[1][i,:], learningRates[rateIndex])
            else :
                # iterating through the training data and adjusting weights
                for i in range(60000) :
                    perceptOut = out(weights, trainingData[1][i,:])
                    weights = adjustWeights(weights, trainingData[TARGET][i], perceptOut, trainingData[1][i,:], learningRates[rateIndex])
            accuracies[perceptNum][x] = (correct/nums)
            # resetting the variables keeping track of the accuracy for the next epoch
            correct = 0
            nums = 0
            
            #running the adjusted weights through the test data 
            if x == epochs-1 : 
                # last epoch add results to confusion matrix
                for i in range(10000) :
                    perceptOut = out(weights, testData[1][i,:])
                    if perceptOut == 1 :
                        testConfusionMatrix[perceptNum][testData[TARGET][i]] += 1
                    if perceptNum == testData[TARGET][i] :
                        testNums += 1
                        if perceptOut == 1 :
                            testCorrect += 1
            else :
                # checking the accuracy of the perceptron through the test data
                for i in range(10000) :
                    perceptOut = out(weights, testData[1][i,:])
                    if perceptNum == testData[TARGET][i] :
                        testNums += 1
                        if perceptOut == 1 :
                            testCorrect += 1
            testAccuracies[perceptNum][x] = (testCorrect/testNums)
            # resetting the variables keeping track of the accuracy for the next epoch
            testCorrect = 0
            testNums = 0
    
    # saving the accuracy data and confusion matrix to two separate csv files for the training data
    pd.DataFrame(accuracies).to_csv("../data/accuracy_learningRate_" + str(learningRates[rateIndex]) + ".csv", header=False, index=False)
    pd.DataFrame(confusionMatrix).to_csv("../data/confusionMatrix_learningRate_" + str(learningRates[rateIndex]) + ".csv", header=False, index=False)

    # saving the accuracy data and confusion matrix to two separate csv files for the test data
    pd.DataFrame(testAccuracies).to_csv("../data/test_accuracy_learningRate_" + str(learningRates[rateIndex]) + ".csv", header=False, index=False)
    pd.DataFrame(testConfusionMatrix).to_csv("../data/test_confusionMatrix_learningRate_" + str(learningRates[rateIndex]) + ".csv", header=False, index=False)
