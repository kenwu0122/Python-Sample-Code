import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *

import utils

# setup the randoms tate
RANDOM_STATE = 545510477

#input: X_train, Y_train
#output: Y_pred
def logistic_regression_pred(X_train, Y_train, X_test):   
	#Train a logistic regression classifier using X_train and Y_train. Use this to predict labels of X_test
	#use default params for the classifier
    np.random.seed(RANDOM_STATE)
    learner = LogisticRegression()
    learner.fit(X_train, Y_train)
    Y_pred = learner.predict(X_test) 	
    return Y_pred

#input: X_train, Y_train and X_test
#output: Y_pred
def svm_pred(X_train, Y_train, X_test):
	#Train a SVM classifier using X_train and Y_train. Use this to predict labels of X_test
	#use default params for the classifier
    np.random.seed(RANDOM_STATE)
    learner = LinearSVC()
    learner.fit(X_train, Y_train)
    Y_pred = learner.predict(X_test)
    return Y_pred


#input: X_train, Y_train and X_test
#output: Y_pred
def decisionTree_pred(X_train, Y_train, X_test):
	#Train a logistic regression classifier using X_train and Y_train. Use this to predict labels of X_test
	#IMPORTANT: use max_depth as 5. Else test cases might fail.
    np.random.seed(RANDOM_STATE)
    learner = DecisionTreeClassifier(max_depth = 5)
    learner.fit(X_train, Y_train)
    Y_pred = learner.predict(X_test)  	
    return Y_pred


#input: Y_pred,Y_true
#output: accuracy, auc, precision, recall, f1-score
def classification_metrics(Y_pred, Y_true):
	#Calculate the above mentioned metrics
	#NOTE: It is important to provide the output in the same order
    
    acc = accuracy_score(Y_pred,Y_true)
    auc_ = roc_auc_score(Y_pred,Y_true)
    precision = precision_score(Y_pred,Y_true)
    recall = recall_score(Y_pred,Y_true)
    f1score = f1_score(Y_pred,Y_true)
    return acc,auc_,precision,recall,f1score

#input: Name of classifier, predicted labels, actual labels
def display_metrics(classifierName,Y_pred,Y_true):
	print("______________________________________________")
	print("Classifier: "+classifierName)
	acc, auc_, precision, recall, f1score = classification_metrics(Y_pred,Y_true)
	print("Accuracy: "+str(acc))
	print("AUC: "+str(auc_))
	print("Precision: "+str(precision))
	print("Recall: "+str(recall))
	print("F1-score: "+str(f1score))
	print("______________________________________________")
	print("")

def main():
	X_train, Y_train = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
    
    #Check model performance on training set
	display_metrics("Logistic Regression",logistic_regression_pred(X_train,Y_train, X_train),Y_train)
	display_metrics("SVM",svm_pred(X_train,Y_train, X_train),Y_train)
	display_metrics("Decision Tree",decisionTree_pred(X_train,Y_train, X_train),Y_train)
    
if __name__ == "__main__":
	main()
	
