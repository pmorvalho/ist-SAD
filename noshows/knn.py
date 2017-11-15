from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
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
pca = PCA(n_components=2).fit(X_test)
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


# using n_jobs to try and paralelize computation when possible!

def run_all_knn(X, y, X_train, y_train, X_test, y_test):
	for n in range(1,12,2):
		clf = KNeighborsClassifier(n_neighbors=n, n_jobs=8)
		clf.fit(X_train,y_train)
		y_pred = clf.predict(X_test)
		print("\nKNN classifier with %d neighbors" % (n))
		print(classification_report(y_test,y_pred,target_names=target_names))
		tn, fp, fn, tp = confusion_matrix(y_test,y_pred, labels=range(2)).ravel()
		print("TN: %d \tFP: %d \nFN: %d \tTP: %d" % (tn, fp, fn, tp))
		print("Accuracy score: %f" % (accuracy_score(y_test,y_pred)))
		print("ROC auc score: %f" % (roc_auc_score(y_test,y_pred)))
		clf = KNeighborsClassifier(n_neighbors=n, n_jobs=4)
		clf.fit(X,y)
		print("Cross-Validation (10-fold) score: %f" % (cross_val_score(clf, X, y, cv=10).mean()))


def draw_roc_graph(k,X_train, y_train, X_test, y_test, X_train_pca, X_test_pca, filename):
	non_pca_ks,non_pca_rocs,with_pca_ks,with_pca_rocs = return_metric_vectors("roc_auc",k,X_train, y_train, X_test, y_test, X_train_pca, X_test_pca)

	f = plt.figure()
	plt.title("AUC ROC Score by k - " + filename)
	plt.xlabel("k-neighbors")
	plt.ylabel("AUC ROC Score")
	plt.gca().set_ylim([0.49,0.62])
	plt.grid()

	plt.plot(non_pca_ks, non_pca_rocs, '.-', color="r", label="Non-PCA")
	plt.plot(with_pca_ks, with_pca_rocs, '.-', color="b", label="with PCA")

	plt.legend(loc="center left", bbox_to_anchor=(1.04, 0.5))	
	#plt.show()
	f.savefig("roc_auc_knn_"+filename+".png",bbox_inches="tight")

def draw_oversample_roc_graph(smote_roc_vectors,adasyn_roc_vectors):
	f = plt.figure()
	plt.title("AUC ROC Score by k - Over-sampling")
	plt.xlabel("k-neighbors")
	plt.ylabel("AUC ROC Score")
	plt.gca().set_ylim([0.49,0.62])
	plt.grid()

	non_pca_ks,non_pca_rocs,with_pca_ks,with_pca_rocs = smote_roc_vectors
	non_pca_ks_ada,non_pca_rocs_ada,with_pca_ks_ada,with_pca_rocs_ada = adasyn_roc_vectors

	plt.plot(non_pca_ks, non_pca_rocs, '.-', color="r", label="SMOTE Non-PCA")
	plt.plot(non_pca_ks_ada, non_pca_rocs_ada, '.-', color="b", label="ADASYN Non-PCA")
	plt.plot(with_pca_ks, with_pca_rocs, '.-', color="g", label="SMOTE with PCA")
	plt.plot(with_pca_ks_ada, with_pca_rocs_ada, '.-', color="y", label="ADASYN with PCA")

	plt.legend(loc="center left", bbox_to_anchor=(1.04, 0.5))
	f.savefig("roc_auc_knn_oversampled.png",bbox_inches="tight")

def return_metric_vectors(metric, k,X_train, y_train, X_test, y_test, X_train_pca, X_test_pca):
	metrics_functions = {
		"roc_auc" : roc_auc_score,
		"accuracy" : accuracy_score,
		"precision" : precision_score,
		"recall" : recall_score	
	}
	metric_score = metrics_functions[metric]
	# non-pca
	non_pca_ks = []
	non_pca_metrics = []

	# with pca
	with_pca_ks = []
	with_pca_metrics = []
	for n in range(1,k,4):
		clf = KNeighborsClassifier(n_neighbors=n, n_jobs=8)
		clf.fit(X_train,y_train)
		y_pred = clf.predict(X_test)
		non_pca_ks.append(n)
		non_pca_metrics.append(metric_score(y_test,y_pred))

		clf = KNeighborsClassifier(n_neighbors=n, n_jobs=8)
		clf.fit(X_train_pca,y_train)
		y_pred = clf.predict(X_test_pca)
		with_pca_ks.append(n)
		with_pca_metrics.append(metric_score(y_test,y_pred))

	return [non_pca_ks,non_pca_metrics,with_pca_ks,with_pca_metrics]


def draw_accuracy_graph(k,X_train, y_train, X_test, y_test, X_train_pca, X_test_pca, filename):
	non_pca_ks,non_pca_accuracys,with_pca_ks,with_pca_accuracys = return_metric_vectors("accuracy",k,X_train, y_train, X_test, y_test, X_train_pca, X_test_pca)

	f = plt.figure()
	plt.title("Accuracy score by k - " + filename)
	plt.xlabel("k-neighbors")
	plt.ylabel("Accuracy score")
	# plt.gca().set_ylim([0.49,0.62])
	plt.grid()

	plt.plot(non_pca_ks, non_pca_accuracys, '.-', color="r", label="Non-PCA")
	plt.plot(with_pca_ks, with_pca_accuracys, '.-', color="b", label="with PCA")

	plt.legend(loc="center left", bbox_to_anchor=(1.04, 0.5))	
	#plt.show()
	f.savefig("accuracy_knn_"+filename+".png",bbox_inches="tight")
	f.savefig("accuracy_knn_"+filename+".pdf",bbox_inches="tight")

def draw_oversample_accuracy_graph(smote_accuracy_vectors,adasyn_accuracy_vectors):
	f = plt.figure()
	plt.title("Accuracy score by k - Over-sampling")
	plt.xlabel("k-neighbors")
	plt.ylabel("Accuracy score")
	# plt.gca().set_ylim([0.49,0.62])
	plt.grid()

	non_pca_ks,non_pca_accuracys,with_pca_ks,with_pca_accuracys = smote_accuracy_vectors
	non_pca_ks_ada,non_pca_accuracys_ada,with_pca_ks_ada,with_pca_accuracys_ada = adasyn_accuracy_vectors

	plt.plot(non_pca_ks, non_pca_accuracys, '.-', color="r", label="SMOTE Non-PCA")
	plt.plot(non_pca_ks_ada, non_pca_accuracys_ada, '.-', color="b", label="ADASYN Non-PCA")
	plt.plot(with_pca_ks, with_pca_accuracys, '.-', color="g", label="SMOTE with PCA")
	plt.plot(with_pca_ks_ada, with_pca_accuracys_ada, '.-', color="y", label="ADASYN with PCA")

	plt.legend(loc="center left", bbox_to_anchor=(1.04, 0.5))
	f.savefig("accuracy_knn_oversampled.png",bbox_inches="tight")
	f.savefig("accuracy_knn_oversampled.pdf",bbox_inches="tight")

def draw_precisionrecall_graph(k,X_train, y_train, X_test, y_test, X_train_pca, X_test_pca, filename):
	non_pca_ks,non_pca_precisions,with_pca_ks,with_pca_precisions = return_metric_vectors("precision",k,X_train, y_train, X_test, y_test, X_train_pca, X_test_pca)
	non_pca_ks,non_pca_recalls,with_pca_ks,with_pca_recalls = return_metric_vectors("recall",k,X_train, y_train, X_test, y_test, X_train_pca, X_test_pca)

	f = plt.figure()
	plt.title("Precision + Recall by k - " + filename)
	plt.xlabel("k-neighbors")
	plt.ylabel("Precsion/Recall average value")
	# plt.gca().set_ylim([0.49,0.62])
	plt.grid()

	plt.plot(non_pca_ks, non_pca_precisions, '.-', color="r", label="Precision Non-PCA")
	plt.plot(with_pca_ks, with_pca_precisions, '.-', color="g", label="Precision with PCA")
	plt.plot(non_pca_ks, non_pca_recalls, '.-', color="b", label="Recall Non-PCA")
	plt.plot(with_pca_ks, with_pca_recalls, '.-', color="y", label="Recall with PCA")

	plt.legend(loc="center left", bbox_to_anchor=(1.04, 0.5))	
	#plt.show()
	f.savefig("precisionrecall_knn_"+filename+".png",bbox_inches="tight")
	f.savefig("precisionrecall_knn_"+filename+".pdf",bbox_inches="tight")

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
	plt.gca().set_ylim([0.46,0.86])
	plt.grid()

	plt.plot(train_sizes, train_scores_mean, '.-', color="r",
	         label="Training score Non-PCA")
	plt.plot(train_sizes_pca, train_scores_mean_pca, '.-', color="b",
	         label="Training score with PCA")
	plt.plot(train_sizes, test_scores_mean, '.-', color="g",
	         label="Cross-validation score Non-PCA")
	plt.plot(train_sizes_pca, test_scores_mean_pca, '.-', color="y",
	         label="Cross-validation score with PCA")

	plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1))	
	#plt.show()
	f.savefig("lc_knn_"+filename+".png",bbox_inches="tight")

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

def draw_all_roc_graphs(k):
	draw_roc_graph(k, X_train, y_train, X_test, y_test,  X_train_pca, X_test_pca, "default")
	smote_roc_vectors = return_metric_vectors("roc_auc",k, X_train_sm, y_train_sm, X_test, y_test,  X_train_sm_pca, X_test_pca)
	adasyn_roc_vectors = return_metric_vectors("roc_auc",k, X_train_ada, y_train_ada, X_test, y_test,  X_train_ada_pca, X_test_pca)
	draw_oversample_roc_graph(smote_roc_vectors,adasyn_roc_vectors)

def draw_all_accuracy_graphs(k):
	draw_accuracy_graph(k, X_train, y_train, X_test, y_test,  X_train_pca, X_test_pca, "default")
	smote_accuracy_vectors = return_metric_vectors("accuracy", k, X_train_sm, y_train_sm, X_test, y_test,  X_train_sm_pca, X_test_pca)
	adasyn_accuracy_vectors = return_metric_vectors("accuracy", k, X_train_ada, y_train_ada, X_test, y_test,  X_train_ada_pca, X_test_pca)
	draw_oversample_accuracy_graph(smote_accuracy_vectors,adasyn_accuracy_vectors)

def draw_all_precisionrecall_graphs(k):
	draw_precisionrecall_graph(k, X_train, y_train, X_test, y_test,  X_train_pca, X_test_pca, "default")