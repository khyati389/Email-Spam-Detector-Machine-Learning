# Spam-Detector
A Python-based spam detector using the Naive Bayes approach.

The spam detection process goes through a series of following steps:

**1. Common Aspects of Text Mining:**
  * Preparation of Corpus.
  * Corpus Inspection.
  * Cleansing of Corpus.
  * Tokenize the Corpus and count the word frequency.
  * Compute probabilities.

**2. Naive Bayes Classifier Approach:**
  * Build the Vocabulary of words by separating SPAM and HAM from training data.
  * Store vocabulary of words in a file.
  * Training the classifier on vocabulary.
  * Evaluate the performance of model on test data.
  * Generate Confusion and Evaluation Matrix.

#### File Structure:
	.
	├── Processor.py	# File Processor and Text Processor
	├── Model.py		# Naive Bayes Classifier
	├── main.py		# Entry point of script
	├── dataset
		├── train		# Training dataset
		├── test	        # Test dataset
	├── report			# Documentation for the project
  	├── results      		
		├── model.txt		# Saved Model
		├── result.txt  	# Classification Results
	└── README.md
  
## Processor.py
This file contains code for reading and loading the training and test documents of spam and ham class. It also consists of logic necessary to implement the steps required for classification of email such as tokenization of corpus, calculating frequency of words, computing conditional probabilities, building and storing the vocabulary and classification results.
 
## Model.py
This python file consists of below functionalities:

1. Methods to implement the NaiveBayes Classifier as well as calculates the parameters such as Accuracy, Precision, Recall and F1 Measure required for analysis of Model. 

2. Logic that predicts class of Email as SPAM or HAM for given document.

3. Method for constructing the Confusion Matrix.

## main.py
It instantiates the objects to call methods for corpus preparation and training classifier. It also runs classifier on test data and displays the Performance Measures for the built model.

## Run Code
On the Command Prompt, run the command:

```python
python main.py
```

## Evaluation Matrix
* Accuracy: <img src="https://latex.codecogs.com/svg.latex?\frac{TP&plus;TN}{TP&plus;TN&plus;FP&plus;FN}" title="\frac{TP+TN}{TP+TN+FP+FN}" />
* Recall : <img src="https://latex.codecogs.com/svg.latex?\frac{TP}{TP&plus;FN}" title="\frac{TP}{TP+FN}" />
* Precision: <img src="https://latex.codecogs.com/svg.latex?\frac{TP}{TP&plus;FP}" title="\frac{TP}{TP+FP}" />
* F1-Measure: <img src="https://latex.codecogs.com/svg.latex?\frac{2*Recall*Precision}{Recall&plus;Precision}" title="\frac{2*Recall*Precision}{Recall+Precision}" />
## Confusion Matrix

Considering SPAM as a positive class and HAM as the negative class:

|                  | SPAM (Predicted)   | HAM (Predicted)   |
|------------------|-----------------|----------------|
| SPAM (Actual) |       TP = 336      |        FN = 64      |
| HAM (Actual)  |       FP =   6      |       TN = 394      |

## References

* https://www3.nd.edu/~steve/computing_with_data/20_text_mining/text_mining_example.html#/
* https://towardsdatascience.com/spam-filtering-using-naive-bayes-98a341224038
* https://medium.com/coinmonks/spam-detector-using-naive-bayes-c22cc740e257
* https://en.wikipedia.org/wiki/Naive_Bayes_spam_filtering
