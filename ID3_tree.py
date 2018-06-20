#from math import log
import operator
import matplotlib.pyplot as plt
from sklearn import datasets
from pylab import *
import numpy as np
import random
import pandas as pd

iris = datasets.load_iris()

def mergeData(data,classes):
    dataSet = np.concatenate([data,classes.reshape(-1,1)],1)
    return dataSet.tolist()


def calcShannonEnt(dataSet):
    numEntries = len(dataSet) #nrows
    #为所有分类类目创建字典
    labelCounts = {}
    for featVec in dataSet:
        currentLable = featVec[-1] #取最后一列数据
        if currentLable not in labelCounts.keys():
            labelCounts[currentLable] = 0
        labelCounts[currentLable] += 1
    #print(labelCounts)
    #计算熵
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= (prob * math.log(prob,2))
    return shannonEnt

#定义按照某个特征进行划分的函数splitDataSet
#输入上个变量（待划分的数据集，特征，分类值）
def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet #返回不含划分特征的子集

#定义按照最大信息增益划分数据的函数
def chooseBestFeatureToSplit(dataSet):
    numFeature = len(dataSet[0])-1
    baseEntropy = calcShannonEnt(dataSet) #计算熵
    bestInforGrain = 0
    bestFeature = -1
    for i in range(numFeature):
        featList = [number[i] for number in dataSet] #得到某个特征下所有值(某列)
        uniqualVals = set(featList) #set无重复的属性特征值
        newEntropy = 0
        for value in uniqualVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet) / float(len(dataSet)) #p(t)
            newEntropy += prob * calcShannonEnt(subDataSet) #对各子集香农熵求和
        infoGain = baseEntropy-newEntropy #计算信息增益
        #最大信息增益
        if (infoGain>bestInforGrain):
            bestInforGrain = infoGain
            bestFeature = i
    return bestFeature #返回特征值

#投票表决
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount

def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    #print('classlist:{0}'.format(classList))
    #类别相同，停止划分
    if classList.count(classList[-1]) == len(classList):
        return classList[-1]
    #长度为1，返回出现次数最多的类别
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    #按照信息增益最高选取分类特征属性
    bestFeat = chooseBestFeatureToSplit(dataSet) #返回分类的特征序号
    bestFeatLable = labels[bestFeat] #该特征的label
    myTree = {bestFeatLable:{}} #构建树的字典
    del(labels[bestFeat]) #从labels的list中删除该label
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLables = labels[:] #子集合
        #构建数据的子集合，并进行递归
        myTree[bestFeatLable][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLables)
    return myTree

#输入三个变量(决策树，属性特征标签，测试的shuju)
def classify(inputTree,featLables,testVec):
    classLabel = ''
    firstStr = list(inputTree.keys())[0] #获取树的第一个属性特征
    secondDict = inputTree[firstStr] #树的分支，子集合Dict
    featIndex = featLables.index(firstStr) #获取决策树第一层在featLables中的位置
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key],featLables,testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

def storeTress(inputTree,filename):
    import pickle
    fw = open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename,'rb')
    return pickle.load(fr)

mpl.rcParams['font.sans-serif'] = ['SimHei'] #否则中文无法正常显示

decisionNode=dict(boxstyle='sawtooth',fc='0.8') #决策点样式
leafNode=dict(boxstyle='round4',fc='0.8') #叶节点样式
arrow_args=dict(arrowstyle='<-') #箭头样式

def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',
                            xytext=centerPt,textcoords='axes fraction',
                            va='center',ha='center',bbox=nodeType,arrowprops=arrow_args)

def createPlot():
    fig=plt.figure(1,facecolor='white')
    fig.clf()
    createPlot.ax1=plt.subplot(111,frameon=False)
    plotNode('决策节点',(0.5,0.1),(0.1,0.5),decisionNode)
    plotNode('叶节点',(0.8,0.1),(0.3,0.8),leafNode)
    plt.show()

#测试
#获取叶节点数量（广度）
def getNumLeafs(myTree):
    numLeafs=0
    firstStr=list(myTree.keys())[0]#'dict_keys' object does not support indexing
    secondDict=myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            numLeafs+=getNumLeafs(secondDict[key])
        else:numLeafs+=1
    return numLeafs

#获取树的深度的函数（深度）
def getTreeDepth(myTree):
    maxDepth=0
    firstStr=list(myTree.keys())[0]
    secondDict=myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth=1+getTreeDepth(secondDict[key])
        else: thisDepth=1
        if thisDepth > maxDepth:
            maxDepth=thisDepth
    return maxDepth
#定义一个预先创建树的函数
def retrieveTree(i):
    listOfTrees=[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                 {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head':{0:'no', 1: 'yes'}},1:'no'}}}}
                 ]
    return listOfTrees[i]

#定义在父子节点之间填充文本信息的函数
def plotMidText(cntrPt,parentPt,txtString):
    xMid=(parentPt[0]-cntrPt[0])/2+cntrPt[0]
    yMid=(parentPt[1]-cntrPt[1])/2+cntrPt[1]
    createPlot.ax1.text(xMid,yMid,txtString)

#定义树绘制的函数
def plotTree(myTree,parentPt,nodeTxt):
    numLeafs=getNumLeafs(myTree)
    depth=getTreeDepth(myTree)
    firstStr=list(myTree.keys())[0]
    cntrPt=(plotTree.xOff+(1.0+float(numLeafs))/2/plotTree.totalW,plotTree.yOff)
    plotMidText(cntrPt,parentPt,nodeTxt)
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    secondDict=myTree[firstStr]
    plotTree.yOff=plotTree.yOff -1/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            plotTree.xOff=plotTree.xOff+1.0/plotTree.totalW
            plotNode(secondDict[key],(plotTree.xOff,plotTree.yOff),cntrPt,leafNode)
            plotMidText((plotTree.xOff,plotTree.yOff),cntrPt,str(key))
    plotTree.yOff=plotTree.yOff+1/plotTree.totalD

 #定义主函数，来调用其它函数
def createPlot(inTree):
    fig=plt.figure(1,facecolor='white')
    fig.clf()
    axprops=dict(xticks=[],yticks=[])
    createPlot.ax1=plt.subplot(111,frameon=False,**axprops)
    plotTree.totalW=float(getNumLeafs(inTree))
    plotTree.totalD=float(getTreeDepth(inTree))
    plotTree.xOff=-0.5/plotTree.totalW;plotTree.yOff=1.0
    plotTree(inTree,(0.5,1.0),'')
    plt.show()

if __name__ == '__main__':
    #print(iris)
    data = mergeData(iris.data,iris.target)
    print(data)
    random.shuffle(data)
    dataSet = data[:100]
    testSet = data[100:]
    labels = ['label1','label2','label3','label4']
    print(dataSet)
    myTree = createTree(dataSet,labels)
    print('Tree:{0}'.format(myTree))
    featurelabels = ['label1','label2','label3','label4']
    correct_num = 0
    for i in range(len(testSet)):
        classify_result = classify(myTree,featurelabels,testSet[i])
        print('real_class:{0}<--------->predict:{1}'.format(testSet[i][-1],classify_result))
        if testSet[i][-1] == classify_result:
            correct_num += 1
    print('Accuracy: {0:.2f}%'.format(float(correct_num/len(testSet))*100))
    createPlot(myTree)