from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import ClusterCentroids
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("data/base_noshows.csv")
X = np.array(data.drop("class",axis=1))
y = np.array(data["class"])
target_names = np.array(["No","Yes"])

# split dataset into training/test portions
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0, stratify=y)

# PCA part
pca = PCA(n_components=2).fit(X)
X_pca = pca.transform(X)

pca = PCA(n_components=2).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# Over-sampling techniques
sm = SMOTE(random_state=1)
X_sm, y_sm = sm.fit_sample(X,y)
X_train_sm, y_train_sm = sm.fit_sample(X_train,y_train)

pca = PCA(n_components=2).fit(X_sm)
X_sm_pca = pca.transform(X_sm)

pca = PCA(n_components=2).fit(X_train_sm)
X_train_sm_pca = pca.transform(X_train_sm)


ada = ADASYN(random_state=2)
X_ada, y_ada = ada.fit_sample(X,y)
X_train_ada, y_train_ada = ada.fit_sample(X_train,y_train)

pca = PCA(n_components=2).fit(X_ada)
X_ada_pca = pca.transform(X_ada)

pca = PCA(n_components=2).fit(X_train_ada)
X_train_ada_pca = pca.transform(X_train_ada)


# n_jobs set to 4 to try and paralelize computation when possible!

def run_all_knn(X, y, X_train, y_train, X_test, y_test):
	for n in range(1,12,2):
		clf = KNeighborsClassifier(n_neighbors=n, n_jobs=8)
		clf.fit(X_train,y_train)
		y_pred = clf.predict(X_test)
		print("\nKNN classifier with %d neighbors" % (n))
		print(classification_report(y_test,y_pred,target_names=target_names))
		print(confusion_matrix(y_test,y_pred, labels=range(2)))
		print("Accuracy score: %f" % (accuracy_score(y_test,y_pred)))
		print("ROC auc score: %f" % (roc_auc_score(y_test,y_pred)))
		clf = KNeighborsClassifier(n_neighbors=n, n_jobs=4)
		clf.fit(X,y)
		print("Cross-Validation (10-fold) score: %f" % (cross_val_score(clf, X, y, cv=10).mean()))

def draw_learning_curve(X, y, X_pca, filename):
	clf = KNeighborsClassifier(n_jobs=8)
	train_sizes,train_scores, test_scores = learning_curve(
	    clf, X, y, cv=10, n_jobs=8)
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)

	clf_pca = KNeighborsClassifier(n_jobs=8)
	train_sizes_pca,train_scores_pca, test_scores_pca = learning_curve(
	    clf, X_pca, y, cv=10, n_jobs=8)
	train_scores_mean_pca = np.mean(train_scores_pca, axis=1)
	train_scores_std_pca = np.std(train_scores_pca, axis=1)
	test_scores_mean_pca = np.mean(test_scores_pca, axis=1)
	test_scores_std_pca = np.std(test_scores_pca, axis=1)

	f = plt.figure()
	plt.title("Learning Curve KNN - " + filename)
	plt.xlabel("Training examples")
	plt.ylabel("Score")
	plt.grid()

	plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
	         label="Training score Non-PCA")
	plt.plot(train_sizes_pca, train_scores_mean_pca, 'o-', color="b",
	         label="Training score with PCA")
	plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
	         label="Cross-validation score Non-PCA")
	plt.plot(train_sizes_pca, test_scores_mean_pca, 'o-', color="y",
	         label="Cross-validation score with PCA")

	plt.legend(loc="right")
	#plt.show()
	f.savefig("lc_knn_"+filename+".pdf")

def run_non_pca_knn():	
	print("\n================= Basic Non-PCA =============================")
	run_all_knn(X, y, X_train, y_train, X_test, y_test)
	print("===============================================================")

	print("\n================= (OS) SMOTE Non-PCA ========================")
	run_all_knn(X, y, X_train_sm, y_train_sm, X_test, y_test)
	print("===============================================================")

	print("\n================= (OS) ADASYN Non-PCA =======================")
	run_all_knn(X, y, X_train_ada, y_train_ada, X_test, y_test)
	print("===============================================================")


def run_pca_knn():
	print("\n================ Basic PCA executions =======================")
	run_all_knn(X_pca, y, X_train_pca,y_train,X_test_pca,y_test)
	print("===============================================================")

	print("\n================= (OS) SMOTE PCA ============================")
	run_all_knn(X_sm_pca, y_sm, X_train_sm_pca,y_train_sm,X_test_pca,y_test)
	print("===============================================================")

	print("\n================= (OS) ADASYN PCA ===========================")
	run_all_knn(X_ada_pca, y_ada, X_train_ada_pca,y_train_ada,X_test_pca,y_test)
	print("===============================================================")

def draw_all_learning_curves():
	draw_learning_curve(X,y,X_pca,"default")
	draw_learning_curve(X_sm,y_sm,X_sm_pca, "SMOTE")
	draw_learning_curve(X_ada,y_ada, X_ada_pca, "ADASYN")


# Under-sampling techniques
# DANGER: RAM MEMORY EXPLOSION. run in sigma for lols
# cc = ClusterCentroids(random_state=42)
# X_train_cc, y_train_cc = cc.fit_sample(X_train, y_train)



