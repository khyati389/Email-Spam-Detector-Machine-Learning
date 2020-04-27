from Processor import TextProcessor
from Processor import FileProcessor
from Model import NaiveBayesClassifier

TRAIN_DOCUMENTS = "dataset/train/"
TEST_DOCUMEMENTS = "dataset/test/"
VOCABULARY_DOCUMENT = "results/model.txt"
RESULT_DOCUMENT = "results/result.txt"

class Console:
    def log(self, text):
        print(str(text)+"...")
'''
main method to execute scripts
'''
def main():
    console = Console()
    textProcessor = TextProcessor()
    fileProcessor = FileProcessor()
    
    # load train and test files
    console.log("loading train files")
    trainFiles = fileProcessor.loadDataFiles(TRAIN_DOCUMENTS)
    console.log("loading test files")
    testFiles = fileProcessor.loadDataFiles(TEST_DOCUMEMENTS)
    
    # Process each train documents
    # 1. read a file content into string
    # 2. tokenize the string into words
    # 3. lower each word and counts it frequency
    #    in all class document, in spam class and in ham class document
    # 4. Calculate smoothed conditional probability of each word in class spam and ham
    console.log("processing train documents")
    fileProcessor.processFiles(trainFiles, TRAIN_DOCUMENTS, textProcessor)

    # Build the Vocabulary of words from training documents
    console.log("building vocabulary")
    textProcessor.buildVocabulary()
    # Get the Vocabulary and Store it in a file
    console.log("storing the vocabulary in "+VOCABULARY_DOCUMENT)
    fileProcessor.storeVocabulary(VOCABULARY_DOCUMENT, textProcessor.getVocabulary())

    totalTrainDocs, totalHamDocs, totalSpamDocs = fileProcessor.getNumOfDocuments(trainFiles)

    # Train Classifier on Vocabulary
    console.log("\ncreating NaiveBayesClassifier Model")
    naiveBayesClassifier = NaiveBayesClassifier()
    console.log("feeding vocabulary to classifier")
    naiveBayesClassifier.fit(textProcessor.getVocabulary())
    naiveBayesClassifier.setPriorHam(totalTrainDocs, totalHamDocs)
    naiveBayesClassifier.setPriorSpam(totalTrainDocs, totalSpamDocs)

    # Run Classifier on Test documents
    console.log("running the classifier on test documents")
    for file in testFiles:
            try:
                with open(str(TEST_DOCUMEMENTS+file), "r", encoding="utf8", errors='ignore') as f:
                    classType = fileProcessor.getClassType(f)
                    wordsList = []

                    for line in f:
                        line = line.strip()
                        wordsList.extend(textProcessor.getWordsFromDocument(textProcessor.tokenize(line)))
                    
                    naiveBayesClassifier.predict(file, classType, wordsList)

            finally:
                f.close()

    # Get the Classification result and store it in a file
    fileProcessor.storeClassificationResult(RESULT_DOCUMENT, naiveBayesClassifier.getClassificationResult())
    console.log("\nclassification done, result stored at "+RESULT_DOCUMENT)

    # Print confusion matrix and other Perfomance Measures
    console.log("printing the perfomance measures")
    naiveBayesClassifier.printConfusionMatrix()
    print("Accuracy measure:  "+ str(naiveBayesClassifier.getAccuracy())+ "\n" )
    print("Precision measure: "+ str(naiveBayesClassifier.getPrecision())+ "\n"  )
    print("recall measure:    "+ str(naiveBayesClassifier.getRecall())+ "\n"  )
    print("f1-measure:        "+ str(naiveBayesClassifier.getF1Measure())+ "\n"  )

if __name__ == "__main__":
    main()
