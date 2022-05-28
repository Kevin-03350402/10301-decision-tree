import numpy as np
import sys
import math


#read in input/output



input= np.genfromtxt(sys.argv[1], delimiter="\t", dtype=None, encoding=None)

output = open(sys.argv[2],"w+")


def binarymode(array):
    zeroNum = np.count_nonzero(array=='0')
    oneNum =  np.count_nonzero(array=='1')
    if (zeroNum <= oneNum):
        return '1'
    else:
        return '0'

def getlabel(df):
    label = df [1:,len(df[0])-1]
    return (label)


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


def binaryentropy (df):
    label = getlabel(df)
    c0 = 0
    c1 = 0
    length = len(label)
    for i in label :
        if i == '0': c0 += 1
        else: c1 += 1
    p0 = c0/length
    p1 = c1/length
    entropy = -(p0*math.log2(p0) + p1*math.log2(p1) )
    return (entropy)

def writem(m):
    
    m.write(f'entropy: {binaryentropy (input)}\n')
    m.write(f'error: {testr(input)}\n')

writem(output)
output.close()

