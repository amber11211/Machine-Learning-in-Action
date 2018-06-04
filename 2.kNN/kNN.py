
# coding: utf-8

# k近邻算法；
# 2.1.1准备：使用python导入数据。导入模块：科学计算包Numpy，运算符模块；createDataSet()函数：创建数据集和标签

# In[4]:


from numpy import *
import operator
from os import listdir#手写识别算法


def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels


# > import kNN
# > 
# > group, labels = kNN.createDataSet()

# 2.1.2实施kNN分类算法。程序清单2-1 k近邻算法。
# classify0()函数有4个输入参数：用于分类的输入变量inX，输入的训练样本集dataSet，标签向量labels，参数k表示用语选择最近邻居的数目。

# In[2]:


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1))-dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(),
                             key=operator.itemgetter(1),
                             reverse=True)
    return sortedClassCount[0][0]


# > kNN.classify0([0.0], group, labels, 3)

# 2.2.1准备数据：从文本文件中解析数据。程序清单2-2 将文本记录(datingTestSet2.txt)转换为NumPy的解析程序。file2matrix函数处理输入格式问题：输入为文件名字符串，输出为训练样本矩阵和类标签向量。

# In[3]:


def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector


# > reload(kNN)
# >
# > datingDataMat,datingLabels = kNN.file2matrix('datingTestSet2,txt')

# 2.2.2分析数据：使用Matplotlib创建散点图。
# 首先使用Matplotlib制作原始数据的散点图：
# > import matplotlib
# >
# > import matplotlib.pyplot as plt
# >
# > fig = plt.figure()
# >
# > ax=fig.add_subplot(111)
# >
# > ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
# >
# > plt.show()
# 
# 改变样本分类的特征值：
# > ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15.0\*array(datingLabels), 15.0\*array(datingLabels))
# 
# 若错误提示未定义array：
# > from numpy import *

# 2.2.3准备数据：归一化数值［newValue=(oldValue-min)/(max-min)］
# 程序清单2-3 归一化特征值：函数autoNorm()自动将数字特征值转化为0到1的区间；tile()函数将变量内容复制成输入矩阵同样大小的矩阵。

# In[4]:


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet- tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet, ranges, minVals


# >reload(kNN)
# 
# >normMat, ranges, minVals = kNN.autoNorm(datingDataMat)
# 
# >normMat
# 
# >ranges
# 
# >minVals

# 2.2.4测试算法：作为完整程序验证分类器
# 程序清单2-4 分类器针对约会网站的测试代码：创建函数datingClassTest，使用file2matrix和autoNorm函数读取数据并归一化特征值

# In[17]:


def datingClassTest():
    hoRatio = 0.01
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print "the classifier came back with: %d, the real answer is: %d"% (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 10
    print "the total error rate is: %f" % (errorCount/float(numTestVecs))


# >kNN.datingClassTest()

# 2.2.5使用算法：构建完整可用系统
# 程序清单2-5 约会网站预测函数：raw_input（）函数允许用户输入文本行命令并返回用户所输入的命令。

# In[18]:


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(raw_input("percentage of time spend playing video games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print "You will probably like this person: ", resultList[classifierResult - 1]


# >kNN.classifyPerson()

# 2.3手写识别系统

# 2.3.1准备数据：将图像转换为测试向量
# 函数img2vector将图像转换为向量

# In[1]:


def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect


# >testVector = kNN.img2vector('testDigits/0_13.txt')
# >
# >testVector[0,0:31]
# >
# >testVector[0,32:63]

# 2.3.2测试算法：使用k-近邻算法识别手写数字
# 程序清单2-6:函数handwritingClassTest()测试分类器

# In[5]:


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('_')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" %(classifierResult, classNumStr)
        if (classifierResult !=classNumStr): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))


# >kNN.handwritingClassTest()
