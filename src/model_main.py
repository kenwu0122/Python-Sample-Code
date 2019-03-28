from models import *

def main():
	X_train, Y_train = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
	X_test, Y_test = utils.get_data_from_svmlight("../data/features_svmlight.validate")
    
    #Check model performance on training set
	print("_________Training Set_____________________________")
	display_metrics("Logistic Regression",logistic_regression_pred(X_train,Y_train, X_train),Y_train)
	display_metrics("SVM",svm_pred(X_train,Y_train, X_train),Y_train)
	display_metrics("Decision Tree",decisionTree_pred(X_train,Y_train, X_train),Y_train)
   
    #Check model performance on test set    
	print("_________Test Set_____________________________")    
	display_metrics("Logistic Regression",logistic_regression_pred(X_train,Y_train,X_test),Y_test)
	display_metrics("SVM",svm_pred(X_train,Y_train,X_test),Y_test)
	display_metrics("Decision Tree",decisionTree_pred(X_train,Y_train,X_test),Y_test)
if __name__ == "__main__":
	main()