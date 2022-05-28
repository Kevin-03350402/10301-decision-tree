import numpy as np
import sys
import math

class Node:
    '''
    Here is an arbitrary Node class that will form the basis of your decision
    tree. 
    Note:
        - the attributes provided are not exhaustive: you may add and remove
        attributes as needed, and you may allow the Node to take in initial
        arguments as well
        - you may add any methods to the Node class if desired 
    '''
    def __init__(self):
        self.left = None
        self.right = None
        self.attr = None
        self.vote = None
        self.df = None

    
train= np.genfromtxt(sys.argv[1], delimiter="\t", dtype=None, encoding=None)
test= np.genfromtxt(sys.argv[2], delimiter="\t", dtype=None, encoding=None)
depth = int(sys.argv[3])

def binarymode(array):
    
    zeroNum = np.count_nonzero(array=='0')
    oneNum =  np.count_nonzero(array=='1')
    if (zeroNum <= oneNum):
        return '1'
    else:
        return '0'

def getlabel(df):
    res = []
    label = df [1:,len(df[0])-1]
    for i in label:
        res.append(i)
    return (res)


def major_vote(df):
    return (binarymode(getlabel(df)))


# test the error rate
def testr (df):
    totalD = len(df)-1
    error = 0
    labelp = len(df[0])-1
    lb = df[1:,labelp]
    for i in lb:
        if i != major_vote(input):
            error += 1
    return (error/totalD)


def binaryentropy (array):
    if array == []:
        return -1
    c0 = 0
    c1 = 0
    length = len(array)
    for i in array :
        if i == '0': c0 += 1
        else: c1 += 1
    p0 = c0/length
    p1 = c1/length
    if p0==0 or p1==0:
        return (0)
    entropy = -(p0*math.log2(p0) + p1*math.log2(p1) )
    return (entropy)

def mi (attr, label):
    hy = binaryentropy(label)
    lenx = len(attr)
    cx0 = 0
    cx1 = 0
    for i in attr:
        if i == '0': cx0 += 1
        else: cx1 += 1
    px0 = cx0/ lenx
    px1 = cx1/lenx
    y_x0 = []
    y_x1 = []
    for i in range(0, lenx):
        if attr[i] == '0':
            y_x0.append(label[i])
        else:
            y_x1.append(label[i])
    res = hy-px0*binaryentropy(y_x0)-px1*binaryentropy(y_x1)
    return res

def allzero(l):
    for ele in l:
        if ele != 0:
            return False
    return True




def allequal(l):
    if l == []: return True
    fir = l[0]
    for ele in l:
        if ele != fir:
            return False
    return True

def pickunique(l1,l2):
    res = []
    for i in l1:
        if i not in l2: res. append(i)
    return res

def findfirstmax(l):
    if l == []:
        return 0
    maxele = max(l)
    for i in range(0,len(l)):
        if l[i] == maxele: return i

def pickrows(df,xfeature,strn):
    index = []
    measurecol = df[:,xfeature]
    for i in range(0, len(measurecol)):
        if measurecol[i] == strn:
            index.append(i)
    return (df[index,:])



# note: the df must be removed of the first row (label names)

level = 0 
used_label = []

def buildtree(df,mdepth,level):


    label = df[:,-1]
    # get the first row excpt the label
    fs = df[0,:-1]
    features = []
    milist = []

    length = len(fs)
    # convert features into numbers
    for i in range(0,length):
        features.append(i)
    # calcuate the mi of each feature, put results into milist
    for attr in features:
        x_attr = df[:,attr]
        
        minfo = mi (x_attr, label)

        milist.append(minfo)
    #reach base case
    if level >= mdepth or max(milist) == 0 or allequal(df[:,-1]): 
        node = Node()
        node.df= df
        node.vote = binarymode(df[:,-1])
        return node

    else:    
        node = Node() 
        node.df = df
        # choose the feature with highest mi
        highmi = findfirstmax(milist)
        
        used_label.append(highmi)
        # split based on the selected feature
        node.attr = highmi
        # create left and right children

        # select the subset of dataframes
        leftc = pickrows(df,highmi,'1')
        rightc = pickrows(df,highmi,'0')
        # depth increased by 1, use level to compare with the maximum depth
        level+=1
        # recursive call
        node.left = buildtree(leftc,mdepth,level)
        node.right= buildtree(rightc,mdepth,level)
        return node




train_set = train[1:,:]
test_set = test[1:,:]

treepredict = []

tree = buildtree(train_set,depth,level)

def predict(data, tree):

    if tree.left == None and tree.right == None:

        return tree.vote
    else:


        if data[tree.attr] == '1':
            return predict(data, tree.left)

        else:
            return predict(data, tree.right)
      
# note: makeprediction fucntion should remove the feature names
def makeprediction(df):
    res = []

    for i in range(0,len(df)):
        dp = df[i,:-1]
        pre = predict(dp, tree)
        res.append(pre)
    return res

def treeerror(df,res):
    truey = getlabel(df)

    error = 0
    total = len(res)
    for i in range (0, len(res)):
        if res[i]!= truey[i]: error += 1

    return (error/total)

train_tree_res = makeprediction(train_set)
test_tree_res = makeprediction(test_set)


trainout= open(sys.argv[4],"w+")

testout = open(sys.argv[5],"w+")

metric = open(sys.argv[6],"w+")

ltrain = len(train_set)
for i in range (0,ltrain):
    trainout.write(f'{train_tree_res[i]}\n')

ltest = len(test_set)

for i in range (0, ltest):
    testout.write(f'{test_tree_res[i]}\n')


metric.write(f'error(train): {treeerror(train,train_tree_res):.6f}\n')
metric.write(f'error(test): {treeerror(test,test_tree_res):.6f}\n')


metric.close()
trainout.close()
testout.close()

featurenames = test[0,:-1]

def findzero(array):
    count = 0
    for i in array: 
        if i == '0':count += 1
    return (count)

def findone(array):
    count = 0
    for i in array: 
        if i == '1':count += 1
    return (count)

dfy = getlabel(train)
y0 = findzero(dfy)
y1 = findone(dfy)

print('[',y0,'0','/',y1,'1',']')
def printtree(tree, branch):
    if tree.attr == None:
        return
    else:
        branch+=1
        x_n = tree.attr
        yl = tree.left.df[:,-1]
        yr = tree.right.df[:,-1]
        yl0 = findzero(yl)
        yl1 = findone(yl)
        yr0 = findzero(yr)
        yr1 = findone(yr)
        
        print('| '*branch, featurenames[x_n], "= 0:",'[',yr0,'0','/',yr1,'1',']')
        
        printtree(tree.right, branch)


        print('| '*branch, featurenames[x_n], "= 1:",'[',yl0,'0','/',yl1,'1',']')


        printtree(tree.left, branch)



print(printtree(tree, 0))




    




            
    
  







if __name__ == '__main__':
    pass
