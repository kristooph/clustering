# -*- coding: utf-8 -*-
import copy
import math
import os
import random
import re
import sys
import decimal
from fnmatch import fnmatch

# defines
DATA_PATH     = 'data'
FILES_PATTERN = '*.txt'

class Document():
    # initialize document words
    def __init__(self, words):
        self.__words = words

    # check if document contains word
    def contains(self, word):
        if word in self.__words:
            return True
        return False

    # return list of words in document
    def words(self):
        return self.__words

    # return number of words
    def numberOfWords(self):
        return len(self.__words)

class Clustering():
    def __init__(self, bagOfWords, documents):
        self.__bagOfWords = bagOfWords
        self.__documents = documents

    def run(self, clusters, groups):
        decimal.getcontext().prec = 320

        probCluster = {}       # probability of cluster
        probDocInCluster = {}  # probability of belonging document to the cluster
        probWordInCluster = {} # probability of belonging term to the clusterster

        # initialize parameters
        for cluster in clusters:
            probCluster[cluster] = 1.0 / len(clusters)

        for document in self.__documents:
            probDocInCluster[document] = {}
            for cluster in clusters:
                if document in groups[cluster]:
                    probDocInCluster[document][cluster] = 1.0
                    continue
                probDocInCluster[document][cluster] = 0.0

        for word in self.__bagOfWords:
            probWordInCluster[word] = {}
            for cluster in clusters:
                probWordInCluster[word][cluster] = 0.0

        # EM algorithm
        iterations = 0
        errorEpsilon = 1.0e-6
        errorValue = len(self.__documents) * errorEpsilon * len(clusters)
        errorMax = errorValue
        while errorValue >= errorMax:
            probDocInClusterPrev = copy.deepcopy(probDocInCluster)

            # maximization step
            for word in self.__bagOfWords:
                for cluster in clusters:
                    nominator = decimal.Decimal(0.0001)
                    denominator = decimal.Decimal(0.0)
                    for document in self.__documents:
                        if self.__documents[document].contains(word):
                            nominator += decimal.Decimal(probDocInCluster[document][cluster])
                        denominator += decimal.Decimal(probDocInCluster[document][cluster])
                    probWordInCluster[word][cluster] = decimal.Decimal(nominator) / decimal.Decimal(denominator)

            for cluster in clusters:
                probCluster[cluster] = 0
                for document in self.__documents:
                    probCluster[cluster] += decimal.Decimal(probDocInCluster[document][cluster])

            # normalize
            norm = decimal.Decimal(0.0)
            for cluster in clusters:
                norm += decimal.Decimal(probCluster[cluster])
            for cluster in clusters:
                probCluster[cluster] /= decimal.Decimal(norm)

            # expectation step
            denominators = {}
            for document in self.__documents:
                denominators[document] = decimal.Decimal(0.0)
                for cluster in clusters:
                    nominator = decimal.Decimal(probCluster[cluster])
                    for term in self.__bagOfWords:
                        if self.__documents[document].contains(term):
                            if probWordInCluster[term][cluster] > 0:
                                nominator *= decimal.Decimal(probWordInCluster[term][cluster])
                            # if probWordInCluster[term][cluster] > 0:
                            #     nominator *= probWordInCluster[term][cluster]
                            #     nominator += math.log(probWordInCluster[term][cluster])
                        else:
                            nominator *= decimal.Decimal(1 - probWordInCluster[term][cluster])
                            # if (1 - probWordInCluster[term][cluster]) > 0:
                            #     # nominator *= (1 - probWordInCluster[term][cluster])
                            #     nominator += math.log(1 - probWordInCluster[term][cluster])
                    probDocInCluster[document][cluster] = decimal.Decimal(nominator)
                    denominators[document] += decimal.Decimal(nominator)
            for document in self.__documents:
                for cluster in clusters:
                    probDocInCluster[document][cluster] /= decimal.Decimal(denominators[document])

            errorValue = 0
            for document in self.__documents:
                for cluster in clusters:
                    errorValue += abs(decimal.Decimal(probDocInCluster[document][cluster]) - decimal.Decimal(probDocInClusterPrev[document][cluster]))
            iterations += 1

        return probDocInCluster, iterations

class Main():
    def __init__(self):
        self.__filesPaths = self.findAllFilesPaths()
        self.__bagOfWords = set()
        self.__documents = {}

    def run(self):
        global DATA_PATH

        # initialize bagOfWords & document
        for filePath in self.__filesPaths:
            words = self.readWordsFromFile(filePath)
            self.__bagOfWords.update(words)
            fileName = filePath[(len(DATA_PATH) + 1):]
            self.__documents[fileName] = Document(words)

        # domains groups
        self.domainClustering()

    # domain clustering
    def domainClustering(self):
        # clusters set & groups
        clusters, groups = list(), {}
        for filePath in self.__filesPaths:
            fileName = filePath[(len(DATA_PATH) + 1):]
            cluster = fileName.split('/')[0]
            if cluster not in groups:
                groups[cluster] = []
                groups[cluster].append(fileName)
                clusters.append(cluster)
        # clusters = [1, 2]
        groups = {'c1': 'c1/6.txt', 'c2': 'c2/7.txt'}

        clustering = Clustering(self.__bagOfWords, self.__documents)
        probDocInCluster, iterations = clustering.run(clusters, groups)

        print('# Iterations:', iterations)
        clustersDistribution = {}
        for cluster in clusters:
            clustersDistribution[cluster] = 0

        notClusterized = 0
        for probDoc in probDocInCluster:
            maxValue = 0.0
            for cluster in clusters:
                if probDocInCluster[probDoc][cluster] > maxValue:
                    maxValue = probDocInCluster[probDoc][cluster]
            # sumaaa = 0
            for cluster in clusters:
                # sumaaa += probDocInCluster[probDoc][cluster]
                if maxValue == 0.0:
                    notClusterized += 1
                    print('# NOT CLUSTERIZED: ' + probDoc)
                    # print('# ' + probDoc + ':')
                    # print('  >', probDocInCluster[probDoc])
                    break
                if probDocInCluster[probDoc][cluster] == maxValue:
                    clustersDistribution[cluster] += 1
            # if sumaaa != 1.0:
                # print('NOT 1.0:', probDoc, sumaaa)
        
        print('# Not clusterized:', notClusterized)
        print('# Clusters distribution:')
        for cluster in clusters:
            print('  > ' + str(cluster) + ': ' + str(clustersDistribution[cluster]))

        # for document in probDocInCluster:
        #     for cluster in probDocInCluster[document]:
        #         value = float(probDocInCluster[document][cluster])
        #         print('{} -> {} = {}'.format(document, cluster, value))
        #     print(' ')

    # read all words (terms) from specified filePath
    def readWordsFromFile(self, filePath):
        with open(filePath, 'r', encoding='utf-8') as input:
            content = input.read()

            words = re.sub('[^a-zA-ZÀ-ž \']+', ' ', content) # leave chars, spaces and '
            words = re.sub('\s+', ' ', words)                # remove multiple whitespaces
        return set(words.split())

    # find all files paths in DATA_PATH
    def findAllFilesPaths(self):
        global DATA_PATH
        global FILES_PATTERN

        filesPaths = []
        for path, _, files in os.walk(DATA_PATH):
            for name in files:
                if fnmatch(name, FILES_PATTERN):
                    filePath = os.path.join(path, name)
                    filesPaths.append(filePath)
        return filesPaths

if __name__ == '__main__':
    Main().run()
