from scipy.special import expit # sigmoid function 
import pandas as pd
import numpy as np
import sys

learningRates = (0.01,0.1,1.0) # the three different learning rates

targets = [ 
        np.array([0.9,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]),
        np.array([0.1,0.9,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]),
        np.array([0.1,0.1,0.9,0.1,0.1,0.1,0.1,0.1,0.1,0.1]),
        np.array([0.1,0.1,0.1,0.9,0.1,0.1,0.1,0.1,0.1,0.1]),
        np.array([0.1,0.1,0.1,0.1,0.9,0.1,0.1,0.1,0.1,0.1]),
        np.array([0.1,0.1,0.1,0.1,0.1,0.9,0.1,0.1,0.1,0.1]),
        np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.9,0.1,0.1,0.1]),
        np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.9,0.1,0.1]),
        np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.9,0.1]),
        np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.9])
        ]

def setWeights(weightC):
    weights = np.random.rand(weightC) #creates array of floats from 0-1
    f = lambda x: x * .1 - 0.05 # lambda function to shift the range from -.05 to .05
    weights = f(weights) # apply the lambda to the array
    return weights

def setMatrixWeights(rows, columns):
    weights = []
    for i in range(rows):
        weights.append(setWeights(columns))
    return np.array(weights)

# returns a tuple of (targets, mxn array of processed images)
def preProcessData(data):
    data = data.sample(frac=1).reset_index(drop=True) #shuffle data
    processedData = data.drop(0, axis=1).applymap(lambda x: x/255.0)
    rowC = len(processedData.index)
    ones = np.ones(rowC)
    return ( data[data.columns[0]], np.column_stack( (ones, processedData.values) ) )

def out(w, x):
    return expit(np.dot(w,x))

def errors(targets, outputs):
    derivative = np.multiply( outputs, np.subtract(1, outputs) )
    return np.multiply( np.subtract(targets, outputs), derivative )

def deltaWeights(learnRate, errors, inputs):
    return np.multiply( learnRate, np.outer(errors, inputs) )


if __name__ == '__main__':
    # arguments to feed the script 1st argument will be the number of epochs
    # and the second argument is which learning rate
    epochs = int(sys.argv[1])
    rateIndex = int(sys.argv[2])
    hiddenNodes = int(sys.argv[3])
    outputNodes = 10

    hiddenWeights = setMatrixWeights(hiddenNodes, 785)
    outerWeights = setMatrixWeights(outputNodes, hiddenNodes)

    #setting up variables for training data 
    dataFile = pd.read_csv("../data/mnist_test.csv", header=None)
    trainingData = preProcessData(dataFile)
    confusionMatrix = np.int_(np.zeros((10,10)))
    accuracies = np.zeros((10,epochs))

    hiddenLayer = out(hiddenWeights, trainingData[1][0])
    outputLayer = out( outerWeights, hiddenLayer ) 
    print( outputLayer )
    print("errors")
    outputErrors = errors(targets[trainingData[0][0]], outputLayer)
    print(outputErrors)
    print("delta weights")
    delta = deltaWeights( .1, outputErrors, outputLayer)
    print(delta)
    
"""
    #setting up variables for test data
    testFile = pd.read_csv("../data/mnist_test.csv", header=None)
    testData = preProcessData(testFile)
    testConfusionMatrix = np.int_(np.zeros((10,10)))
    testAccuracies = np.zeros((10,epochs))

    
    # saving the accuracy data and confusion matrix to two separate csv files for the training data
    pd.DataFrame(accuracies).to_csv("../data/accuracy_learningRate_" + str(learningRates[rateIndex]) + ".csv", header=False, index=False)
    pd.DataFrame(confusionMatrix).to_csv("../data/confusionMatrix_learningRate_" + str(learningRates[rateIndex]) + ".csv", header=False, index=False)

    # saving the accuracy data and confusion matrix to two separate csv files for the test data
    pd.DataFrame(testAccuracies).to_csv("../data/test_accuracy_learningRate_" + str(learningRates[rateIndex]) + ".csv", header=False, index=False)
    pd.DataFrame(testConfusionMatrix).to_csv("../data/test_confusionMatrix_learningRate_" + str(learningRates[rateIndex]) + ".csv", header=False, index=False)
"""
