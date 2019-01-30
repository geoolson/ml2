import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import sys

if __name__ == "__main__" :
    if len(sys.argv) == 1 :
        exit()
    rate = int(sys.argv[1])
    trainingAccuracy = []
    trainingAccuracy.append(pd.read_csv("../data/accuracy_learningRate_0.01.csv", header=None).values)
    trainingAccuracy.append(pd.read_csv("../data/accuracy_learningRate_0.1.csv", header=None).values)
    trainingAccuracy.append(pd.read_csv("../data/accuracy_learningRate_1.0.csv", header=None).values)

    testAccuracy = []
    testAccuracy.append(pd.read_csv("../data/test_confusionMatrix_learningRate_0.01.csv", header=None).values)
    testAccuracy.append(pd.read_csv("../data/test_confusionMatrix_learningRate_0.1.csv", header=None).values)
    testAccuracy.append(pd.read_csv("../data/test_confusionMatrix_learningRate_1.0.csv", header=None).values)

    for i in range(3) :
        trainingAccuracy[i] = trainingAccuracy[i].sum(axis=0)
        trainingAccuracy[i] = np.divide(trainingAccuracy[i], 10)
    
    sumNumerator = 0.0
    sumTotal = 0.0
    for i in range(10) :
        sumNumerator += testAccuracy[0][i][i]
        for j in range(10) :
            sumTotal += testAccuracy[0][i][j]
    print ( sumNumerator / sumTotal ) 
    if rate >= 0 :
        plt.plot(trainingAccuracy[rate], label="training data")
        # plt.plot(testAccuracy[rate], label="test data")
        plt.ylabel("Accuracy")
        plt.xlabel("Epochs")
        plt.legend()
        plt.show()

