from math import exp
from numpy.linalg import inv
import numpy as np


def sigmoid(x):
    return 1 / (1 + exp(-x))


class MyLogisticRegression1:
    def __init__(self):
        self.intercept_ = 0.0
        self.coef_ = []

    # use the gradient descent method
    # simple stochastic GD
    def fit(self, x, y, learningRate=0.001, noEpochs=1000):
        self.coef_ = [0.0 for _ in range(1 + len(x[0]))]  # beta or w coefficients y = w0 + w1 * x1 + w2 * x2 + ...
        # self.coef_ = [random.random() for _ in range(len(x[0]) + 1)]    #beta or w coefficients
        for epoch in range(noEpochs):
            # TBA: shuffle the trainind examples in order to prevent cycles
            for i in range(len(x)):  # for each sample from the training data
                ycomputed = sigmoid(self.eval(x[i], self.coef_))  # estimate the output
                crtError = ycomputed - y[i]  # compute the error for the current sample
                for j in range(0, len(x[0])):  # update the coefficients
                    self.coef_[j + 1] = self.coef_[j + 1] - learningRate * crtError * x[i][j]
                self.coef_[0] = self.coef_[0] - learningRate * crtError * 1

        self.intercept_ = self.coef_[-1]
        self.coef_ = self.coef_[:-1]

    def eval(self, xi, coef):
        yi = coef[0]
        for j in range(len(xi)):
            yi += coef[j + 1] * xi[j]
        return yi

    def predictOneSample(self, sampleFeatures):
        threshold = 0.5
        coefficients = [self.intercept_] + [c for c in self.coef_]
        computedFloatValue = self.eval(sampleFeatures, coefficients)
        computed01Value = sigmoid(computedFloatValue)
        # computedLabel = 0 if computed01Value < threshold else 1

        return computed01Value

    def predict(self, inTest):
        computedLabels = [self.predictOneSample(sample) for sample in inTest]
        return computedLabels

    class MyLogisticRegression:
        def __init__(self, num_classes):
            self.num_classes = num_classes
            self.classifiers = []

        def fit(self, x, y, learningRate=0.001, noEpochs=1000):
            for class_idx in range(self.num_classes):
                y_one_vs_all = [1 if el == class_idx else 0 for el in y]
                clf = MyLogisticRegression()
                clf.coef_ = [0.0 for _ in range(1 + len(x[0]))]
                for epoch in range(noEpochs):
                    for i in range(len(x)):
                        ycomputed = sigmoid(self.eval(x[i], clf.coef_))
                        crtError = ycomputed - y_one_vs_all[i]
                        for j in range(0, len(x[0])):
                            clf.coef_[j + 1] = clf.coef_[j + 1] - learningRate * crtError * x[i][j]
                        clf.coef_[0] = clf.coef_[0] - learningRate * crtError * 1
                self.classifiers.append(clf)

        def predictOneSample(self, sampleFeatures):
            class_scores = []
            for clf in self.classifiers:
                class_scores.append(self.eval(sampleFeatures, clf.coef_))
            return np.argmax(class_scores)

        def predict(self, inTest):
            computedLabels = [self.predictOneSample(sample) for sample in inTest]
            return computedLabels

