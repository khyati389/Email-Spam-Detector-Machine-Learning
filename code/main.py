import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import re
from Processor import TextProcessor

TRAIN_DOCUMENTS = "dataset/train/"
TEST_DOCUMEMENTS = "dataset/test/"

textProcessor = TextProcessor()

'''
FileProcessor class which reads and processes files
'''
class FileProcessor:

    def __init__(self):
        pass    
    
    '''
    function to get list of files in a directory
    '''
    def loadDataFiles(self, path):
        return os.listdir(path)
    
    '''
    Indentifies the class type (spam | ham) from the file name
    '''
    def getClassType(self, file):
        docBase = os.path.basename(file.__getattribute__('name'))
        docName = os.path.splitext(docBase)[0]

        if re.search('(.*)-(ham)-(.*)', docName):
            return 'ham'
        else:
            return 'spam'

    
    '''
    read files line by line and processes it
    '''
    def processFiles(self, files, path):
        for file in files:
            try:
                with open(str(path+file), "r", encoding="utf8", errors='ignore') as f:
                    classType = self.getClassType(f)

                    for line in f:
                        line = line.strip()
                        words = textProcessor.tokenize(line)
                        textProcessor.recordWordCount(words)
                        textProcessor.updatefrequencyCountInClass(classType, words)

            finally:
                f.close()

'''
main method to execute scripts
'''
def main():
    fileProcessor = FileProcessor()
    
    trainFiles = fileProcessor.loadDataFiles(TRAIN_DOCUMENTS)
    testFiles = fileProcessor.loadDataFiles(TEST_DOCUMEMENTS)
    
    fileProcessor.processFiles(trainFiles, TRAIN_DOCUMENTS)

if __name__ == "__main__":
    main()



