from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

data = load_breast_cancer()
inputs = data['data']
outputs = data['target']
outputNames = data['target_names']
featureNames = list(data['feature_names'])
feature1 = [feat[featureNames.index('mean radius')] for feat in inputs]
feature2 = [feat[featureNames.index('mean texture')] for feat in inputs]
inputs = [[feat[featureNames.index('mean radius')], feat[featureNames.index('mean texture')]] for feat in inputs]

#data iris
dataIris=load_iris()
inputsIris=dataIris['data']
outputsIris=dataIris['target']
outputNamesIris=dataIris['target_names']
featureNamesIris=list(dataIris['feature_names'])
lengthSepal=[feat[featureNamesIris.index('sepal length (cm)')] for feat in inputsIris]
widthSepal=[feat[featureNamesIris.index('sepal width (cm)')] for feat in inputsIris]
lengthPetal=[feat[featureNamesIris.index('petal length (cm)')] for feat in inputsIris]
widthPetal=[feat[featureNamesIris.index('petal length (cm)')] for feat in inputsIris]

inputsIris=[[feat[featureNamesIris.index('sepal length (cm)')],feat[featureNamesIris.index('sepal width (cm)')],feat[featureNamesIris.index('petal length (cm)')],feat[featureNamesIris.index('petal length (cm)')]] for feat in inputsIris]
import matplotlib.pyplot as plt
labelsIris=set(outputsIris)
lenDtaIris=len(inputsIris)
for crt1 in labelsIris:
    x=[lengthSepal[i] for i in range(lenDtaIris)if outputsIris[i]==crt1]
    y =[widthSepal[i] for i in range(lenDtaIris) if outputsIris[i] == crt1]
    # axisI[0].scatter(x,y,label=outputNamesIris[crt1])
    # axisI[0].set_title("Sepal")
    plt.scatter(x,y,label=outputNamesIris[crt1])

plt.show()
for crt1 in labelsIris:
    x=[lengthPetal[i] for i in range(lenDtaIris)if outputsIris[i]==crt1]
    y =[widthPetal[i] for i in range(lenDtaIris) if outputsIris[i] == crt1]
    # axisI[0].set_title("Petal")
    # axisI[1].scatter(x,y,label=outputNamesIris[crt1])
    plt.scatter(x,y,label=outputNamesIris[crt1])

plt.show()

labels = set(outputs)
noData = len(inputs)
# for crtLabel in labels:
#     x = [feature1[i] for i in range(noData) if outputs[i] == crtLabel ]
#     y = [feature2[i] for i in range(noData) if outputs[i] == crtLabel ]
#     plt.scatter(x, y, label = outputNames[crtLabel])
# plt.xlabel('mean radius')
# plt.ylabel('mean texture')
# plt.legend()
# plt.show()

# fig, ax = plt.subplots(1, 3,  figsize=(4 * 3, 4))
# ax[0].hist(feature1, 10)
# ax[0].title.set_text('Histogram of mean radius')
# ax[1].hist(feature2, 10)
# ax[1].title.set_text('Histogram of mean texture')
# ax[2].hist(outputs, 10)
# ax[2].title.set_text('Histogram of cancer class')
# plt.show()


def normalisation(trainData, testData):
    scaler = StandardScaler()
    if not isinstance(trainData[0], list):
        # encode each sample into a list
        trainData = [[d] for d in trainData]
        testData = [[d] for d in testData]

        scaler.fit(trainData)  # fit only on training data
        normalisedTrainData = scaler.transform(trainData)  # apply same transformation to train data
        normalisedTestData = scaler.transform(testData)  # apply same transformation to test data

        # decode from list to raw values
        normalisedTrainData = [el[0] for el in normalisedTrainData]
        normalisedTestData = [el[0] for el in normalisedTestData]
    else:
        scaler.fit(trainData)  # fit only on training data
        normalisedTrainData = scaler.transform(trainData)  # apply same transformation to train data
        normalisedTestData = scaler.transform(testData)  # apply same transformation to test data
    return normalisedTrainData, normalisedTestData

def normalisationIris(traindata,testdata):
    scaler=StandardScaler()
    if not isinstance(traindata[0],list):
        traindata=[[d] for d in traindata ]
        testdata=[[d] for d in testdata]
        scaler.fit(traindata)
        noralisedtrainData=scaler.transform(traindata)
        normalisedTestData=scaler.transform(testdata)

        normalisedTestData=[el[0] for el in normalisedTestData]
        noralisedtrainData=[el[0] for el in noralisedtrainData]
    else:
        scaler.fit(traindata)
        noralisedtrainData=scaler.transform(traindata)
        normalisedTestData = scaler.transform(testdata)
    return noralisedtrainData,normalisedTestData

def plotIris(lengthSepal,widthSepal,lengthPetal,widthPetal,outputsIris):
    labelsIris=set(outputsIris)
    lenDtaIris=len(lengthSepal)
    for crt1 in labelsIris:
        x = [lengthSepal[i] for i in range(lenDtaIris) if outputsIris[i] == crt1]
        y = [widthSepal[i] for i in range(lenDtaIris) if outputsIris[i] == crt1]
        # axisI[0].scatter(x,y,label=outputNamesIris[crt1])
        # axisI[0].set_title("Sepal")
        plt.scatter(x, y, label=outputNamesIris[crt1])

    plt.show()
    for crt1 in labelsIris:
        x = [lengthPetal[i] for i in range(lenDtaIris) if outputsIris[i] == crt1]
        y = [widthPetal[i] for i in range(lenDtaIris) if outputsIris[i] == crt1]
        # axisI[0].set_title("Petal")
        # axisI[1].scatter(x,y,label=outputNamesIris[crt1])
        plt.scatter(x, y, label=outputNamesIris[crt1])

    plt.show()
def plotClassificationData(feature1, feature2, outputs, title=None):
    labels = set(outputs)
    noData = len(feature1)
    for crtLabel in labels:
        x = [feature1[i] for i in range(noData) if outputs[i] == crtLabel]
        y = [feature2[i] for i in range(noData) if outputs[i] == crtLabel]
        plt.scatter(x, y, label=outputNames[crtLabel])
    plt.xlabel('mean radius')
    plt.ylabel('mean texture')
    plt.legend()
    plt.title(title)
    plt.show()

# step2: impartire pe train si test
# step2': normalizare
import numpy as np

# split data into train and test subsets
# np.random.seed(5)
# indexes = [i for i in range(len(inputs))]
# trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace = False)
# testSample = [i for i in indexes  if not i in trainSample]
#
# trainInputs = [inputs[i] for i in trainSample]
# trainOutputs = [outputs[i] for i in trainSample]
# testInputs = [inputs[i] for i in testSample]
# testOutputs = [outputs[i] for i in testSample]
#
# #normalise the features
# trainInputs, testInputs = normalisation(trainInputs, testInputs)
#
# #plot the normalised data
# feature1train = [ex[0] for ex in trainInputs]
# feature2train = [ex[1] for ex in trainInputs]
# feature1test = [ex[0] for ex in testInputs]
# feature2test = [ex[1] for ex in testInputs]
#
# plotClassificationData(feature1train, feature2train, trainOutputs, 'normalised train data')

#impartire pe train si input

np.random.seed(5)
indexesIris = [i for i in range(len(inputsIris))]
trainSampleIris = np.random.choice(indexesIris, int(0.8 * len(inputsIris)), replace = False)
testSampleIris = [i for i in indexesIris  if not i in trainSampleIris]
trainInputsIris = [inputsIris[i] for i in trainSampleIris]
trainOutputsIris = [outputsIris[i] for i in trainSampleIris]
testInputsIris = [inputsIris[i] for i in testSampleIris]
testOutputsIris = [outputsIris[i] for i in testSampleIris]


trainInputsIris, testInputsIris = normalisation(trainInputsIris, testInputsIris)

lengthSepal=[ex[0] for ex in trainInputsIris]
widthSepal=[ex[1] for ex in trainInputsIris]
lengthPetal=[ex[2] for ex in trainInputsIris]
widthPetal=[ex[3] for ex in trainInputsIris]

lengthSepalTest=[ex[0] for ex in testInputsIris]
widthSepalTest=[ex[1] for ex in testInputsIris]
lengthPetalTest=[ex[2] for ex in testInputsIris]
widthPetalTest=[ex[3] for ex in testInputsIris]

plotIris(lengthSepal,widthSepal,lengthPetal,widthPetal,trainOutputsIris)
# step3: invatare model (cu tool linear_model.LogisticRegression() -- [link](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) -- si cu implementare proprie)


#identify (by training) the classifier

# using sklearn
from sklearn import linear_model
classifier = linear_model.LogisticRegression()
classifier.fit(trainInputsIris,trainOutputsIris)
w0,w1,w2,w3=classifier.intercept_,classifier.coef_[0],classifier.coef_[1],classifier.coef_[2]
print('classification model: y(feat1, feat2) = ', w0, ' + ', w1, ' * feat1 + ', w2, ' * feat2',w3, ' * feat3')
computedTestData=classifier.predict(testInputsIris)
def plotPredIris(lengthSepal,widthSepal,lengthPetal,widthPetal,realoutputsIris,computedOutputs):
    labelsIris=list(set(outputsIris))
    lenDtaIris=len(lengthSepal)
    for crt1 in labelsIris:
        x = [lengthSepal[i] for i in range(lenDtaIris) if realoutputsIris[i] == crt1 and computedOutputs[i]==crt1]
        y = [widthSepal[i] for i in range(lenDtaIris) if realoutputsIris[i] == crt1 and computedOutputs[i]==crt1]
        # axisI[0].scatter(x,y,label=outputNamesIris[crt1])
        # axisI[0].set_title("Sepal")
        plt.scatter(x, y, label=outputNamesIris[crt1]+'(correct)')
    for crt1 in labelsIris:
        x = [lengthSepal[i] for i in range(lenDtaIris) if realoutputsIris[i] == crt1 and computedOutputs[i]!=crt1]
        y = [widthSepal[i] for i in range(lenDtaIris) if realoutputsIris[i] == crt1 and computedOutputs[i]!=crt1]
        # axisI[0].scatter(x,y,label=outputNamesIris[crt1])
        # axisI[0].set_title("Sepal")
        plt.scatter(x, y, label=outputNamesIris[crt1]+'(incorrect)')
    plt.xlabel('length of sepal')
    plt.ylabel('width of sepal')
    plt.legend()
    plt.show()
    for crt1 in labelsIris:
        x = [lengthPetal[i] for i in range(lenDtaIris) if realoutputsIris[i] == crt1 and computedOutputs[i]==crt1]
        y = [widthPetal[i] for i in range(lenDtaIris) if realoutputsIris[i] == crt1 and computedOutputs[i]==crt1]
        # axisI[0].set_title("Petal")
        # axisI[1].scatter(x,y,label=outputNamesIris[crt1])
        plt.scatter(x, y, label=outputNamesIris[crt1]+'(correct)')

    for crt1 in labelsIris:
        x = [lengthPetal[i] for i in range(lenDtaIris) if realoutputsIris[i] == crt1 and computedOutputs[i]!=crt1]
        y = [widthPetal[i] for i in range(lenDtaIris) if realoutputsIris[i] == crt1 and computedOutputs[i]!=crt1]
        # axisI[0].set_title("Petal")
        # axisI[1].scatter(x,y,label=outputNamesIris[crt1])
        plt.scatter(x, y, label=outputNamesIris[crt1]+'(incorrect)')

    plt.xlabel('length of petal')
    plt.ylabel('width of petal')
    plt.legend()
    plt.show()
# plotPredIris(lengthSepal,widthSepal,lengthPetal,widthPetal,trainOutputsIris,computedTestData)
plotPredIris(lengthSepalTest,widthSepalTest,lengthPetalTest,widthPetalTest,testOutputsIris,computedTestData)
# using developed code


from LogisticRegression import MyLogisticRegression1


# #impartim in 3 probleme de regresie
# trainOutputsIris1=[1 if el==0 else 0 for el in trainOutputsIris]
# trainOutputsIris2=[1 if el==1 else 0 for el in trainOutputsIris]
# trainOutputsIris3=[1 if el==2 else 0 for el in trainOutputsIris]
# # w0,w1,w2,w3,w4=myClassifier.intercept_,myClassifier.coef_[0],myClassifier.coef_[1],myClassifier.coef_[2],myClassifier.coef_[3]
# # myComputedVals=[w0 + w1 * el[0] + w2 * el[1]+w3*el[2]+w4*el[3] for el in testInputsIris]
# myClassifier1=MyLogisticRegression()
# myClassifier1.fit(trainInputsIris,trainOutputsIris1)
# myComputedVals1=myClassifier1.predict(testInputsIris)
#
# myClassifier2=MyLogisticRegression()
# myClassifier2.fit(trainInputsIris,trainOutputsIris2)
# myComputedVals2=myClassifier2.predict(testInputsIris)
#
# myClassifier3=MyLogisticRegression()
# myClassifier3.fit(trainInputsIris,trainOutputsIris3)
# myComputedVals3=myClassifier3.predict(testInputsIris)
# myComputedVals=[]
myClassifier=MyLogisticRegression1(3)
myClassifier.fit(trainInputsIris,trainOutputsIris)
myComputedVals=myClassifier.predict(testInputsIris)
# for i in range(len(myComputedVals3)):
#     maxComp=max(myComputedVals1[i],myComputedVals2[i],myComputedVals3[i])
#     if(maxComp==myComputedVals1[i]):
#         myComputedVals.append(0)
#     else:
#         if (maxComp == myComputedVals2[i]):
#             myComputedVals.append(1)
#         else:
#             myComputedVals.append(2)
plotPredIris(lengthSepalTest,widthSepalTest,lengthPetalTest,widthPetalTest,testOutputsIris,myComputedVals)
# model initialisation
# classifier = MyLogisticRegression()
#
# # train the classifier (fit in on the training data)
# classifier.fit(trainInputs, trainOutputs)
# # parameters of the liniar regressor
# w0, w1, w2 = classifier.intercept_, classifier.coef_[0], classifier.coef_[1]
# print('classification model: y(feat1, feat2) = ', w0, ' + ', w1, ' * feat1 + ', w2, ' * feat2')
#
#
# # step4: testare model, plot rezultate, forma outputului si interpretarea lui
#
# # makes predictions for test data
# # computedTestOutputs = [w0 + w1 * el[0] + w2 * el[1] for el in testInputs]
#
# # makes predictions for test data (by tool)
# computedTestOutputs = classifier.predict(testInputs)
# def plotPredictions(feature1, feature2, realOutputs, computedOutputs, title, labelNames):
#     labels = list(set(outputs))
#     noData = len(feature1)
#     for crtLabel in labels:
#         x = [feature1[i] for i in range(noData) if realOutputs[i] == crtLabel and computedOutputs[i] == crtLabel ]
#         y = [feature2[i] for i in range(noData) if realOutputs[i] == crtLabel and computedOutputs[i] == crtLabel]
#         plt.scatter(x, y, label = labelNames[crtLabel] + ' (correct)')
#     for crtLabel in labels:
#         x = [feature1[i] for i in range(noData) if realOutputs[i] == crtLabel and computedOutputs[i] != crtLabel ]
#         y = [feature2[i] for i in range(noData) if realOutputs[i] == crtLabel and computedOutputs[i] != crtLabel]
#         plt.scatter(x, y, label = labelNames[crtLabel] + ' (incorrect)')
#     plt.xlabel('mean radius')
#     plt.ylabel('mean texture')
#     plt.legend()
#     plt.title(title)
#     plt.show()
#
# plotPredictions(feature1test, feature2test, testOutputs, computedTestOutputs, "real test data", outputNames)
#
#
# # step5: calcul metrici de performanta (acc)
#
# # evalaute the classifier performance
# # compute the differences between the predictions and real outputs
# # print("acc score: ", classifier.score(testInputs, testOutputs))
# error = 0.0
# for t1, t2 in zip(computedTestOutputs, testOutputs):
#     if (t1 != t2):
#         error += 1
# error = error / len(testOutputs)
# print("classification error (manual): ", error)
#
# from sklearn.metrics import accuracy_score
# error = 1 - accuracy_score(testOutputs, computedTestOutputs)
# print("classification error (tool): ", error)
