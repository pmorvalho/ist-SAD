from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np

data = pd.read_csv("data/base_crabs.csv")
X = np.array(data.drop("class",axis=1))
y = np.array(data["class"])
target_names = np.array(["B","O"])

# split dataset into training/test portions
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

# PCA part
pca = PCA(n_components=2).fit(X)
X_pca = pca.transform(X)

pca = PCA(n_components=2).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

def run_all_knn(X, y, X_train, y_train, X_test, y_test):
	for n in range(1,12,2):
		clf = KNeighborsClassifier(n_neighbors=n)
		clf.fit(X_train,y_train)
		y_pred = clf.predict(X_test)
		print("\nKNN classifier with %d neighbors" % (n))
		print(classification_report(y_test,y_pred,target_names=target_names))
		print(confusion_matrix(y_test,y_pred, labels=range(2)))
		print("Accuracy score: %f" % (accuracy_score(y_test,y_pred)))
		print("ROC auc score: %f" % (roc_auc_score(y_test,y_pred)))
		clf = KNeighborsClassifier(n_neighbors=n)
		clf.fit(X,y)
		print("Cross-Validation (10-fold) score: %f" % (cross_val_score(clf, X, y, cv=10).mean()))


print("\n================= Non-PCA executions ========================")
run_all_knn(X, y, X_train, y_train, X_test, y_test)
print("===============================================================")

print("\n================= With-PCA executions =======================")
run_all_knn(X_pca, y, X_train_pca,y_train,X_test_pca,y_test)
print("===============================================================")
