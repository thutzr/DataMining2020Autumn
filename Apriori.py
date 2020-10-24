import numpy as np
import pandas as pd
from collections import defaultdict
import sys
from optparse import OptionParser
from itertools import chain, combinations

def readDataFromFile(filename):
    '''
    parameter(s)::
        filename: the file name that is going to be read.
    return::
        a generator contains all transactions
    '''
    with open(filename,'r') as file:
        for line in file:
            line = line.rstrip(',\n')
            record = frozenset(line.split(',')[1:])
            yield record

def getItemSetTransactionList(transactionGenerator):
    '''
    parameter(s)::
        transactionGenerator: a generator contains all transactions
    return::
        itemSet: the set of items exist in all transactions
        transactionList: a python list contains all transactions
    '''
    itemSet = set()
    transactionList = list()
    for record in transactionGenerator:
        transaction = frozenset(record)
        transactionList.append(transaction)
        for item in transaction:
            itemSet.add(frozenset([item]))

    return itemSet, transactionList

def minSupFilter(itemSet,transactionList,minSup,freqDict):
    '''
    parameter(s)::
        itemSet: the set of items exist in all transactions
        transactionList: a python list contains all transactions
        minSup: the specified minimum support
        freqDict: a python dictionary consists of the frequency of some patterns
    return::
        localItemSet: a python dictionary contains patterns that satisfy the minimum support constraint.
    '''
    localItemSet = set()
    if itemSet == set():
        return localItemSet
    localFreqDict = defaultdict(int)
    for item in itemSet:
        for transaction in transactionList:
            if item.issubset(transaction):
                freqDict[item] += 1
                localFreqDict[item] += 1
    for item,count in localFreqDict.items():
        if count >= minSup:
            localItemSet.add(item)
    return localItemSet

# def generateLkSetWithPrune(LSet,length,transactionList,minSup,freqDict):
#     LkSet = set([i.union(j) for i in LSet for j in LSet if len(i.union(j)) == length])
#     return minSupFilter(LkSet,transactionList,minSup,freqDict)

def generateLkSet(LSet,length):
    '''
    parameter(s)::
        LSet: patterns that has length of length-1
        length: the length of target patterns
    return::
        the set of patterns that have length length
    '''
    return set([i.union(j) for i in LSet for j in LSet if len(i.union(j))==length])

def properSubset(itemSet):
    '''
    parameter(s)::
        itemSet: a set of items
    return::
        all proper subsets of itemSet
    '''
    return chain(*[combinations(itemSet, i + 1) for i, a in enumerate(itemSet) if i<len(itemSet)-1])

def Apriori(transactionGenerator, minSup, minConf):
    '''
    parameter(s)::
        transactionGenerator: a generator contains all transaction
        minSup: minimum support
        minConf: minimum confidence
    return::
        LSetDict: a dictionary consists of patterns of all length that satisfy the minimum support constraint
        freqPattern: a python list contains all frequent pattern
        associationRule: a python list contains all association rule satisfy minimum support and minimum confidence constraints
    '''
    itemSet, transactionList = getItemSetTransactionList(transactionGenerator)
    freqDict = defaultdict(int)
    LOneCandidate = minSupFilter(itemSet,transactionList,minSup,freqDict)
    k = 2
    LSetDict = dict()
    currentLSet = LOneCandidate
    while (currentLSet != set()):
        LSetDict[k-1] = currentLSet
        tempSet = currentLSet
        currentLSet = generateLkSet(tempSet,k)
        currentCSet = minSupFilter(currentLSet,transactionList,minSup,freqDict)
        currentLSet = currentCSet
        k += 1

    freqPattern = list()
    for key, lset in LSetDict.items():
        freqPattern.extend([(set(item),freqDict[item]) for item in lset])
    associationRule = list()
    for key, lset in list(LSetDict.items())[1:]:
        for lseq in lset:
            properSubsets = [frozenset(x) for x in properSubset(lseq)]
            for item in properSubsets:
                if (len(lseq.difference(item))>0):
                    conf = freqDict[lseq]/freqDict[item]
                    if conf >= minConf:
                        associationRule.append([set(item),set(lseq.difference(item)),conf])
    return LSetDict, freqPattern,associationRule

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-f','--file',dest = 'file',help = 'input file',default = None)
    parser.add_option('-s','--minsup',dest = 'minsup',help = 'min support',default = 10)
    parser.add_option('-c','--minconf',dest = 'minconf',help = 'min confidence',default = 0.3)
    parser.add_option('-of','--optFP',dest = 'outputfreq',help = 'path to save frequent patterns',default = 'freqPatterns.csv')
    parser.add_option('-oa','--optAR',dest = 'outputasso',help = 'path to save association rules',default = 'associationRules.csv')
    (options,args) = parser.parse_args()
    file = None
    if options.file is None:
        file = sys.stdin
    else:
        file = readDataFromFile(options.file)
    minSup = int(options.minsup)
    minConf = float(options.minconf)
    LSetDict, freqPattern, associationRule = Apriori(file,minSup,minConf)
    freqPatternDF = pd.DataFrame(freqPattern)
    freqPatternDF.columns = ['pattern','frequency']
    freqPath = options.optFP
    freqPatternDF.to_csv(freqPath)
    associationRuleDF = pd.DataFrame(associationRule)
    associationRuleDF.columns = ['X','Y','confidence']
    associationPath = options.optAR
    associationRuleDF.to_csv(associationPath)
