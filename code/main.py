import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import re
from Processor import TextProcessor
from Processor import FileProcessor

TRAIN_DOCUMENTS = "dataset/train/"
TEST_DOCUMEMENTS = "dataset/test/"
VOCABULARY_DOCUMENT = "results/model.txt"
RESULT_DOCUMENT = "results/result.txt"

'''
main method to execute scripts
'''
def main():
    textProcessor = TextProcessor()
    fileProcessor = FileProcessor()
    
    trainFiles = fileProcessor.loadDataFiles(TRAIN_DOCUMENTS)
    testFiles = fileProcessor.loadDataFiles(TEST_DOCUMEMENTS)
    
    fileProcessor.processFiles(trainFiles, TRAIN_DOCUMENTS, textProcessor)
    textProcessor.buildVocabulary()
    fileProcessor.storeVocabulary(VOCABULARY_DOCUMENT, textProcessor.getVocabulary())

if __name__ == "__main__":
    main()



