# Spam-Detector
A Python-based spam detector using the Naive Bayes approach.

Steps:

1. Preparing the text data.
2. Creating word dictionary.
3. Feature extraction process.
4. Training the classifier.
5. Run classifier on the test set.

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
This file contains code for reading and loading the training and test documents of spam and ham class. It also consists of logic necessary to implement the steps required for classification of email such as tokenization of corpus, calculating frequency of words, computing conditional probabilities, building vocabulary, training the classifier and evaluating performance of model on test data.
 
## Model.py
This python file consists of below functionalities:

1. Methods to implement the NaiveBayes Classifier as well as calculates the parameters such as Accuracy, Precision, Recall and F1 Measure required for analysis of Model. 

2. Logic that predicts class of Email as SPAM or HAM for given document.

3. Method for constructing the Confusion Matrix.

## main.py
It instantiates the objects to call methods for corpus preparation and training classifier. It also runs classifier on test data and displays the Performance Measures for the built model.

## Run Code
Open Command Prompt

```python
python main.py
