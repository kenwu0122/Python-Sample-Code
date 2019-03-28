import models
from sklearn.cross_validation import KFold, ShuffleSplit
from numpy import mean
from sklearn.metrics import *
import utils

# USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_kfold(X,Y,k=5):
	#TODO:First get the train indices and test indices for each iteration
	#Then train the classifier accordingly
	#Report the mean accuracy and mean auc of all the folds
    kf = KFold(len(Y),n_folds=k,random_state = 545510477)
    acc = []
    auc = []
    for train, test in kf:
        Y_pred = models.logistic_regression_pred(X[train], Y[train], X[test])
        acc.append(accuracy_score(Y[test], Y_pred))
        auc.append(roc_auc_score(Y[test], Y_pred))
    return mean(acc), mean(auc)

#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_randomisedCV(X,Y,iterNo=5,test_percent=0.2):
	#TODO: First get the train indices and test indices for each iteration
	#Then train the classifier accordingly
	#Report the mean accuracy and mean auc of all the iterations
    kf = ShuffleSplit(len(Y),n_iter=iterNo, test_size=test_percent, random_state = 545510477)
    acc = []
    auc = []
    for train, test in kf:
        Y_pred = models.logistic_regression_pred(X[train], Y[train], X[test])
        acc.append(accuracy_score(Y[test], Y_pred))
        auc.append(roc_auc_score(Y[test], Y_pred))
    return mean(acc), mean(auc)

def main():
	X,Y = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
	print("Classifier: Logistic Regression__________")
	acc_k,auc_k = get_acc_auc_kfold(X,Y)
	print(("Average Accuracy in KFold CV: "+str(acc_k)))
	print(("Average AUC in KFold CV: "+str(auc_k)))
	acc_r,auc_r = get_acc_auc_randomisedCV(X,Y)
	print(("Average Accuracy in Randomised CV: "+str(acc_r)))
	print(("Average AUC in Randomised CV: "+str(auc_r)))

if __name__ == "__main__":
	main()

