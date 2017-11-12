from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.tree import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
import graphviz 

data = pd.read_csv("data/base_crabs.csv")
X = np.array(data.drop("class",axis=1))
y = np.array(data["class"])
target_names = np.array(["B","O"])
feature_names = np.array(["sex","index","FL","RW","CL", "CW", "BD"])
# split dataset into training/test portions
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

# PCA part
pca = PCA(n_components=3).fit(X)
X_pca = pca.transform(X)

pca = PCA(n_components=3).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

def decisionTree(X, X_train, y_train, X_test, y_test, f, wpca):
	clf = DecisionTreeClassifier()
	clf = clf.fit(X_train,y_train)
	y_pred = clf.predict(X_test)
	print("\nDecision Tree Calssififer:")
	print(classification_report(y_test,y_pred,target_names=target_names))
	print(confusion_matrix(y_test,y_pred, labels=range(2)))
	print("Accuracy score: %f" % (accuracy_score(y_test,y_pred)))
	print("ROC auc score: %f" % (roc_auc_score(y_test,y_pred)))
	if wpca == "-non_pca":
		dot_data = export_graphviz(clf, out_file=None,
							 feature_names=f,
	                         class_names=target_names,  
	                         filled=True, rounded=True,  
	                         special_characters=True)  
		graph = graphviz.Source(dot_data)  
		graph.render("crabs"+wpca)
	clf = DecisionTreeClassifier()
	clf.fit(X,y)
	print("Cross-Validation (10-fold) score: %f" % (cross_val_score(clf, X, y, cv=10).mean()))
	
print("\n================= Non-PCA executions ========================")
decisionTree(X, X_train, y_train, X_test, y_test, feature_names, "-non_pca")
print("===============================================================")

print("\n================= With-PCA executions =======================")
decisionTree(X_pca, X_train_pca,y_train,X_test_pca,y_test, feature_names, "-pca")
print("===============================================================")

