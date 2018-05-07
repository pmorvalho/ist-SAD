from sklearn.decomposition import PCA
from sklearn.metrics import *
from sklearn.tree import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
import pandas as pd
import numpy as np
import graphviz 
from math import *
import os
import matplotlib.pyplot as plt


data = pd.read_csv("data/base_crabs.csv")
X = np.array(data.drop("class",axis=1))
y = np.array(data["class"])
target_names = np.array(["B","O"])
feature_names = np.array(["sex","FL","RW","CL", "CW", "BD"])
# split dataset into training/test portions

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0, stratify=y)


def decisionTree(X, X_train, y_train, X_test, y_test, min_sample_leaf, min_sample_node):
	
	clf = DecisionTreeClassifier(min_samples_leaf=min_sample_leaf, min_samples_split=min_sample_node)

	clf = clf.fit(X_train,y_train)
	y_pred = clf.predict(X_test)
	# print("\nDecision Tree Calssififer:")
	# print(classification_report(y_test,y_pred,target_names=target_names))
	# print(confusion_matrix(y_test,y_pred, labels=range(2)))
	# print("Accuracy score: %f" % (accuracy_score(y_test,y_pred)))
	
	# print("ROC auc score: %f" % (roc_auc_score(y_test,y_pred)))
	graph = graphviz.Source(export_graphviz(clf, out_file=None,
						 feature_names=feature_names,
                         class_names=target_names,  
                         filled=True, rounded=True,  
                         special_characters=True))  
	graph.render("decision-trees-examples/crabs-dt-"+str(min_sample_leaf)+"samples_leaf-"+str(min_sample_node)+"samples_node")
	# clf = DecisionTreeClassifier()
	# clf.fit(X,y)
	# print("Cross-Validation (10-fold) score: %f" % (cross_val_score(clf, X, y, cv=10).mean()))
	os.system("rm decision-trees-examples/crabs-dt-"+str(min_sample_leaf)+"samples_leaf-"+str(min_sample_node)+"samples_node")
	
	# return str(accuracy_score(y_test,y_pred))
	treeObj = clf.tree_
	return str(treeObj.node_count) +","+ str(accuracy_score(y_test,y_pred))

# decisionTree(X, X_train, y_train, X_test, y_test, 21, 5)
# decisionTree(X, X_train, y_train, X_test, y_test, 23, 5)
# decisionTree(X, X_train, y_train, X_test, y_test, 25, 5)
# decisionTree(X, X_train, y_train, X_test, y_test, 26, 5)
# decisionTree(X, X_train, y_train, X_test, y_test, 40, 5)
# decisionTree(X, X_train, y_train, X_test, y_test, 42, 5)
# decisionTree(X, X_train, y_train, X_test, y_test, 28, 5)
# decisionTree(X, X_train, y_train, X_test, y_test, 20, 50)

# decisionTree(X, X_train, y_train, X_test, y_test, 2, 2)
# decisionTree(X, X_train, y_train, X_test, y_test, 12, 2)
# decisionTree(X, X_train, y_train, X_test, y_test, round(21/3), 21)
# decisionTree(X, X_train, y_train, X_test, y_test, round(22/3), 22)
# decisionTree(X, X_train, y_train, X_test, y_test, round(23/3), 23)
# decisionTree(X, X_train, y_train, X_test, y_test, round(24/3), 24)
# decisionTree(X, X_train, y_train, X_test, y_test, round(25/3), 25)


# print("\n=================================== Min Samples Leaf =============================================")	
# print("\n==================================================================================================")	

# for i in range(1,51,1):
# 	print(str(i)+","+decisionTree(X, X_train, y_train, X_test, y_test, i, 5))	

# print("\n=================================== Min Samples Node =============================================")	
# print("\n==================================================================================================")	

# for i in range(2,51,1):
	# print(str(i)+","+decisionTree(X, X_train, y_train, X_test, y_test, round(i/3), i))	

# print("\n=================================== Min Samples Node =============================================")	
# print("\n==================================================================================================")	

# for i in range(1,51,12):
# 	some =""
# 	for j in range(2, 51, 12):
# 		some+=str(j)+","+str(i)+","+decisionTree(X, X_train, y_train, X_test, y_test, i, j)+","
# 	print(some[:-1])


# for i in range(101,2,-1):
# 	print(decisionTree(X, X_train, y_train, X_test, y_test, 1, i))	

# print(decisionTree(X, X_train, y_train, X_test, y_test, 1, 90))
# print(decisionTree(X, X_train, y_train, X_test, y_test, 1, 50))

# clf = DecisionTreeClassifier()
# train_sizes,train_scores, test_scores = learning_curve(
#     clf, X, y, cv=10, n_jobs=1)
# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)


# plt.figure()
# plt.title("Learning Curve DT's")
# plt.xlabel("Training examples")
# plt.ylabel("Score")
# plt.grid()

# plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                  train_scores_mean + train_scores_std, alpha=0.1,
#                  color="r")
# plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
#                  test_scores_mean + test_scores_std, alpha=0.1, color="g")
# plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
#          label="Training score")
# plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
#          label="Cross-validation score")

# plt.legend(loc="best")
# plt.show()
