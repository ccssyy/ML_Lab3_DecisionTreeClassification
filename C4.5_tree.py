from pylab import *
from scipy import *
from math import log
import operator
from sklearn import datasets
import numpy as np

def mergeData(data,classes):
    dataSet = np.concatenate([data,classes.reshape(-1,1)],1)
    return dataSet.tolist()

#计算给定数据的香浓熵：
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}  #类别字典（类别的名称为键，该类别的个数为值）
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():  #还没添加到字典里的类型
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:  #求出每种类型的熵
        prob = float(labelCounts[key])/numEntries  #每种类型个数占所有的比值
        shannonEnt -= prob * math.log(prob, 2)
    return shannonEnt  #返回熵

#按照给定的特征划分数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:  #按dataSet矩阵中的第axis列的值等于value的分数据集
        if featVec[axis] == value:      #值等于value的，每一行为新的列表（去除第axis个数据）
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet  #返回分类后的新矩阵

#选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0])-1  #求属性的个数
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):  #求所有属性的信息增益
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)  #第i列属性的取值（不同值）数集合
        newEntropy = 0.0
        splitInfo = 0.0
        for value in uniqueVals:  #求第i列属性每个不同值的熵*他们的概率
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))  #求出该值在i列属性中的概率
            newEntropy += prob * calcShannonEnt(subDataSet)  #求i列属性各值对于的熵求和
            splitInfo -= prob * math.log(prob, 2)
        infoGain = (baseEntropy - newEntropy) / splitInfo  #求出第i列属性的信息增益率
        #print('infoGain:{0}'.format(infoGain))
        if(infoGain > bestInfoGain):  #保存信息增益率最大的信息增益率值以及所在的下表（列值i）
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

#找出出现次数最多的分类名称
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

#创建树
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]    #创建需要创建树的训练数据的结果列表（例如最外层的列表是[N, N, Y, Y, Y, N, Y]）
    if classList.count(classList[0]) == len(classList):  #如果所有的训练数据都是属于一个类别，则返回该类别
        return classList[0]
    if (len(dataSet[0]) == 1):  #训练数据只给出类别数据（没给任何属性值数据），返回出现次数最多的分类名称
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet)   #选择信息增益最大的属性进行分（返回值是属性类型列表的下标）
    bestFeatLabel = labels[bestFeat]  #根据下表找属性名称当树的根节点
    myTree = {bestFeatLabel:{}}  #以bestFeatLabel为根节点建一个空树
    del(labels[bestFeat])  #从属性列表中删掉已经被选出来当根节点的属性
    featValues = [example[bestFeat] for example in dataSet]  #找出该属性所有训练数据的值（创建列表）
    uniqueVals = set(featValues)  #求出该属性的所有值得集合（集合的元素不能重复）
    for value in uniqueVals:  #根据该属性的值求树的各个分支
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)  #根据各个分支递归创建树
    return myTree  #生成的树

#实用决策树进行分类
def classify(inputTree, featLabels, testVec):
    classLabel = ''
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else: classLabel = secondDict[key]
    return classLabel

#读取数据文档中的训练数据（生成二维列表）
def createTrainData():
    lines_set = open('../data/ID3/Dataset.txt').readlines()
    labelLine = lines_set[2]
    labels = labelLine.strip().split()
    lines_set = lines_set[4:11]
    dataSet = []
    for line in lines_set:
        data = line.split()
        dataSet.append(data)
    return dataSet, labels


#读取数据文档中的测试数据（生成二维列表）
def createTestData():
    lines_set = open('../data/ID3/Dataset.txt').readlines()
    lines_set = lines_set[15:22]
    dataSet = []
    for line in lines_set:
        data = line.strip().split()
        dataSet.append(data)
    return dataSet

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
    iris = datasets.load_iris()
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
    print('Accuracy: {0:.2f}%'.format(float(correct_num / len(testSet)) * 100))
    createPlot(myTree)