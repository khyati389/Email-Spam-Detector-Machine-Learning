from Processor import TextProcessor
from Processor import FileProcessor
from Model import NaiveBayesClassifier

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
    
    # load train and test files
    trainFiles = fileProcessor.loadDataFiles(TRAIN_DOCUMENTS)
    testFiles = fileProcessor.loadDataFiles(TEST_DOCUMEMENTS)
    
    # Process each train documents
    # 1. read a file content into string
    # 2. tokenize the string into words
    # 3. lower each word and counts it frequency
    #    in all class document, in spam class and in ham class document
    # 4. Calculate smoothed conditional probability of each word in class spam and ham
    fileProcessor.processFiles(trainFiles, TRAIN_DOCUMENTS, textProcessor)

    # Build the Vocabulary of words from training documents
    textProcessor.buildVocabulary()
    # Get the Vocabulary and Store it in a file
    fileProcessor.storeVocabulary(VOCABULARY_DOCUMENT, textProcessor.getVocabulary())

    totalTestDocs, totalHamDocs, totalSpamDocs = fileProcessor.getNumOfDocuments(testFiles)

    # Train Classifier on Vocabulary
    naiveBayesClassifier = NaiveBayesClassifier()
    naiveBayesClassifier.fit(textProcessor.getVocabulary())
    naiveBayesClassifier.setPriorHam(totalTestDocs, totalHamDocs)
    naiveBayesClassifier.setPriorSpam(totalTestDocs, totalSpamDocs)

    # Run Classifier on Test documents
    for file in testFiles:
            try:
                with open(str(TEST_DOCUMEMENTS+file), "r", encoding="utf8", errors='ignore') as f:
                    classType = fileProcessor.getClassType(f)
                    wordsDict = {}

                    for line in f:
                        line = line.strip()
                        wordsDict = textProcessor.getWordsFromDocument(textProcessor.tokenize(line), wordsDict)
                    
                    naiveBayesClassifier.predict(file, classType, wordsDict.keys())

            finally:
                f.close()
    
    # Get the Classification result and store it in a file
    fileProcessor.storeClassificationResult(RESULT_DOCUMENT, naiveBayesClassifier.getClassificationResult())

if __name__ == "__main__":
    main()
