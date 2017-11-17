from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score
from sklearn.tree import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
import pandas as pd
import numpy as np
import graphviz 
import os
import matplotlib.pyplot as plt


data = pd.read_csv("data/base_noshows.csv")
X = np.array(data.drop("class",axis=1))
y = np.array(data["class"])
target_names = np.array(["No", "Yes"])
feature_names = np.array(["Gender", "ScheduledDay", "AppointmentDay", "Age", "Scholarship", "Hipertension", "Diabetes", "Alcoholism", "Handcap", "SMS_received"])
# split dataset into training/test portions
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0, stratify=y)



def return_metric_vectors(metric, k,X_train, y_train, X_test, y_test, X_train_pca, X_test_pca):
	metrics_functions = {
		"roc_auc" : roc_auc_score,
		"accuracy" : accuracy_score,
		"precision" : precision_score,
		"recall" : recall_score	
	}
	metric_score = metrics_functions[metric]
	k_values, non_pca_metrics, with_pca_metrics = [], [], []
	for n in range(1,k,4):
		k_values.append(n)
		clf = KNeighborsClassifier(n_neighbors=n, n_jobs=8)
		clf.fit(X_train,y_train)
		y_pred = clf.predict(X_test)
		non_pca_metrics.append(metric_score(y_test,y_pred))

		clf = KNeighborsClassifier(n_neighbors=n, n_jobs=8)
		clf.fit(X_train_pca,y_train)
		y_pred = clf.predict(X_test_pca)
		with_pca_metrics.append(metric_score(y_test,y_pred))

	return [k_values,non_pca_metrics,with_pca_metrics]




def decisionTree(X, X_train, y_train, X_test, y_test, min_sample_leaf, min_sample_node, balanced=None):
	
	clf = DecisionTreeClassifier(min_samples_leaf=min_sample_leaf, min_samples_split=min_sample_node, class_weight=balanced)

	clf = clf.fit(X_train,y_train)
	y_pred = clf.predict(X_test)


	# print("\nDecision Tree Calssififer:")
	# print(classification_report(y_test,y_pred,target_names=target_names))
	# print(confusion_matrix(y_test,y_pred, labels=range(2)))
	# print("Accuracy score: %f" % (accuracy_score(y_test,y_pred)))
	
	# print("ROC auc score: %f" % (roc_auc_score(y_test,y_pred)))
	# graph = graphviz.Source(export_graphviz(clf, out_file=None,
	# 					 feature_names=feature_names,
 #                         class_names=target_names,  
 #                         filled=True, rounded=True,
 #                         special_characters=True))  
	# graph.render("decision-trees-examples/noshows-dt-"+str(min_sample_leaf)+"samples_leaf-"+str(min_sample_node)+"samples_node")
	# clf = DecisionTreeClassifier()
	# clf.fit(X,y)
	# print("Cross-Validation (10-fold) score: %f" % (cross_val_score(clf, X, y, cv=10).mean()))
	# os.system("rm decision-trees-examples/noshows-dt-"+str(min_sample_leaf)+"samples_leaf-"+str(min_sample_node)+"samples_node")
	
	return str(accuracy_score(y_test,y_pred))
	# treeObj = clf.tree_
	# return str(treeObj.node_count) +","+ str(accuracy_score(y_test,y_pred))
	# return [treeObj.node_count, accuracy_score(y_test,y_pred),recall_score(y_test,y_pred)]

def draw_precisionrecall_graph(node_vectors,node_vectors_balanced):
	node_counts, accuracy_values, recall_values = node_vectors
	node_counts_bal, accuracy_values_bal, recall_values_bal = node_vectors_balanced

	f = plt.figure()
	plt.title("Accuracy + Sensibility by Number of Nodes")
	plt.xlabel("Number of nodes")
	plt.ylabel("Accuracy/Sensibility average value")
	plt.gca().set_ylim([0,1])
	plt.grid()

	plt.plot(node_counts, accuracy_values, '.-', color="r", label="Accuracy Default")
	plt.plot(node_counts_bal, accuracy_values_bal, '.-', color="b", label="Accuracy Balanced")
	plt.plot(node_counts, recall_values, '.-', color="g", label="Sensibility Default")
	plt.plot(node_counts_bal, recall_values_bal, '.-', color="y", label="Sensibility Balanced")

	plt.legend(loc="center left", bbox_to_anchor=(1.04, 0.5))	
	#plt.show()
	f.savefig("precisionrecall_dts.png",bbox_inches="tight")
	f.savefig("precisionrecall_dts.pdf",bbox_inches="tight")

# print("\n================================== Min Samples Leaf =============================================")	
# print("\n==================================================================================================")	

# for i in range(1,50001,1000):
# 	print(str(i)+","+decisionTree(X, X_train, y_train, X_test, y_test, i, 5))	

# print("\n=================================== Min Samples Node =============================================")	
# print("\n==================================================================================================")	

# for i in range(2,50001,1000):
# 	print(str(i)+","+decisionTree(X, X_train, y_train, X_test, y_test, round(i/3), i))	

# print("\n=================================== Min Samples Node =============================================")	
# print("\n==================================================================================================")	

# for i in range(1,62501,12500):
# 	some =""
# 	for j in range(2, 62501, 12500):
# 		some+=str(j)+","+str(i)+","+decisionTree(X, X_train, y_train, X_test, y_test, i, j)+","
# 	print(some[:-1])

# decisionTree(X, X_train, y_train, X_test, y_test, 37501, 37502)
# decisionTree(X, X_train, y_train, X_test, y_test, 50001, 50002)
# decisionTree(X, X_train, y_train, X_test, y_test, 50001, 2)
# decisionTree(X, X_train, y_train, X_test, y_test, 37501, 2)
# decisionTree(X, X_train, y_train, X_test, y_test, 12501, 2)
# decisionTree(X, X_train, y_train, X_test, y_test, 25001, 2)

# node_counts, accuracy_values, recall_values = [], [], []
# for i in range(50001,2,-63):
# 	nc, av, rv = decisionTree(X, X_train, y_train, X_test, y_test, round(i/3), i)
# 	node_counts.append(nc)
# 	accuracy_values.append(av)
# 	recall_values.append(rv)

# node_counts_bal, accuracy_values_bal, recall_values_bal = [], [], []
# for i in range(50001,2,-63):
# 	nc, av, rv = decisionTree(X, X_train, y_train, X_test, y_test, round(i/3), i, balanced='balanced')
# 	node_counts_bal.append(nc)
# 	accuracy_values_bal.append(av)
# 	recall_values_bal.append(rv)

# node_vectors = [node_counts, accuracy_values, recall_values]
# node_vectors_balanced = [node_counts_bal, accuracy_values_bal, recall_values_bal]
# draw_precisionrecall_graph(node_vectors,node_vectors_balanced)

# for i in range(50001,2,-63):
	# print(decisionTree(X, X_train, y_train, X_test, y_test, round(i/3), i))	



# clf = DecisionTreeClassifier(class_weight='balanced')
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
