import perceptron as percept
import pandas as pd
import numpy as np
import unittest

data = pd.read_csv("../data/test.csv", header=None)
goldFile = pd.read_csv("../data/goldFile.csv", header=None)

class testPercept(unittest.TestCase):
    def testLearningRates(self):
        self.assertEqual(0.01, percept.learningRates[0])
        self.assertEqual(0.1, percept.learningRates[1])
        self.assertEqual(1.0, percept.learningRates[2])
    def testPreprocessor(self):
        preProcessedData = pd.DataFrame(percept.preProcessData(data)[1]).to_string()
        goldData = goldFile.to_string()
        self.assertEqual(goldData, preProcessedData)
    def testWeights(self):
        weights = percept.setWeights()
        for i in np.nditer(weights):
            self.assertAlmostEqual(i, 0, delta=.0501)
    def testOut(self):
        testfile = pd.read_csv("../data/mnist_train.csv", header=None)
        testData = percept.preProcessData(testfile)[1]
        val = percept.out( percept.setWeights(), testData )
        if val == 1 or val == 0 :
            pass
        else :
            self.assertEqual(1, 0)

def generateGoldFile(testData):
    preprocessedData = percept.preProcessData(testData)[1]
    pd.DataFrame(preprocessedData).to_csv("../data/goldFile.csv", header=False, index=False)
    return preprocessedData

if __name__ == '__main__':
    unittest.main()
