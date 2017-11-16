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

GRAPHS_FOLDER = "knn_graphs/"
data = pd.read_csv("data/base_crabs.csv")
X = np.array(data.drop("class",axis=1))
y = np.array(data["class"])
target_names = np.array(["B","O"])

# split dataset into training/test portions
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0, stratify=y)

# PCA part
pca = PCA(n_components=2).fit(X)
X_pca = pca.transform(X)

pca = PCA(n_components=2).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# using n_jobs to try and paralelize computation when possible!

def run_all_knn(X, y, X_train, y_train, X_test, y_test):
	for n in range(1,51,2):
		clf = KNeighborsClassifier(n_neighbors=n, n_jobs=8)
		clf.fit(X_train,y_train)
		y_pred = clf.predict(X_test)
		print("\nKNN classifier with %d neighbors" % (n))
		# print(classification_report(y_test,y_pred,target_names=target_names))
		tn, fp, fn, tp = confusion_matrix(y_test,y_pred, labels=range(2)).ravel()
		print("TN: %d \tFP: %d \nFN: %d \tTP: %d" % (tn, fp, fn, tp))
		# print("Accuracy score: %f" % (accuracy_score(y_test,y_pred)))
		# print("ROC auc score: %f" % (roc_auc_score(y_test,y_pred)))
		# clf = KNeighborsClassifier(n_neighbors=n, n_jobs=4)
		# clf.fit(X,y)
		# print("Cross-Validation (10-fold) score: %f" % (cross_val_score(clf, X, y, cv=10).mean()))

def return_metric_vectors(metric, k,X_train, y_train, X_test, y_test, X_train_pca, X_test_pca):
	metrics_functions = {
		"roc_auc" : roc_auc_score,
		"accuracy" : accuracy_score,
		"precision" : precision_score,
		"recall" : recall_score	
	}
	metric_score = metrics_functions[metric]
	k_values, non_pca_metrics, with_pca_metrics = [], [], []
	for n in range(1,k,2):
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


metrics_titles = {
	"roc_auc" : "AUC ROC score",
	"accuracy" : "Accuracy score",
	"precision" : "Precision score",
	"recall" : "Recall score"	
}

def draw_single_metric_graph(metric,k,X_train, y_train, X_test, y_test, X_train_pca, X_test_pca,filename,y_lim=None):
	k_values,non_pca_metrics,with_pca_metrics = return_metric_vectors(metric,k,X_train, y_train, X_test, y_test, X_train_pca, X_test_pca)
	f = plt.figure()
	plt.title("%s by %s - %s" % (metrics_titles[metric],"k-neighbors",filename))
	plt.xlabel("k-neighbors")
	plt.ylabel(metrics_titles[metric])
	if y_lim is not None:
		plt.gca().set_ylim(y_lim)
	plt.grid()

	plt.plot(k_values, non_pca_metrics, '.-', color="r", label="Non-PCA")
	plt.plot(k_values, with_pca_metrics, '.-', color="b", label="with PCA")

	plt.legend(loc="center left", bbox_to_anchor=(1.04, 0.5))	

	f.savefig("%s%s_knn_%s.png" % (GRAPHS_FOLDER,metric,filename),bbox_inches="tight")
	f.savefig("%s%s_knn_%s.pdf" % (GRAPHS_FOLDER,metric,filename),bbox_inches="tight")

def draw_precisionrecall_graph(k,X_train, y_train, X_test, y_test, X_train_pca, X_test_pca, filename):
	k_values,non_pca_precisions,with_pca_precisions = return_metric_vectors("precision",k,X_train, y_train, X_test, y_test, X_train_pca, X_test_pca)
	k_values,non_pca_recalls,with_pca_recalls = return_metric_vectors("recall",k,X_train, y_train, X_test, y_test, X_train_pca, X_test_pca)

	f = plt.figure()
	plt.title("Precision + Recall by k - " + filename)
	plt.xlabel("k-neighbors")
	plt.ylabel("Precsion/Recall average value")
	plt.gca().set_ylim([0,1])
	# plt.gca().set_ylim([0.49,0.62])
	plt.grid()

	plt.plot(k_values, non_pca_precisions, '.-', color="r", label="Precision Non-PCA")
	plt.plot(k_values, with_pca_precisions, '.-', color="g", label="Precision with PCA")
	plt.plot(k_values, non_pca_recalls, '.-', color="b", label="Recall Non-PCA")
	plt.plot(k_values, with_pca_recalls, '.-', color="y", label="Recall with PCA")

	plt.legend(loc="center left", bbox_to_anchor=(1.04, 0.5))	
	#plt.show()
	f.savefig(GRAPHS_FOLDER+"precisionrecall_knn_"+filename.lower()+".png",bbox_inches="tight")
	f.savefig(GRAPHS_FOLDER+"precisionrecall_knn_"+filename.lower()+".pdf",bbox_inches="tight")


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
	plt.gca().set_ylim([0,1])
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
	f.savefig(GRAPHS_FOLDER+"lc_knn_"+filename+".png",bbox_inches="tight")
	f.savefig(GRAPHS_FOLDER+"lc_knn_"+filename+".pdf",bbox_inches="tight")

def run_non_pca_knn():	
	print("\n================= Basic Non-PCA =============================")
	run_all_knn(X, y, X_train, y_train, X_test, y_test)
	print("===============================================================")


def run_pca_knn():
	print("\n================ Basic PCA executions =======================")
	run_all_knn(X_pca, y, X_train_pca,y_train,X_test_pca,y_test)
	print("===============================================================")


def draw_all_learning_curves():
	draw_learning_curve(X,y,X_pca,"default")

def draw_all_roc_graphs(k):
	draw_single_metric_graph("roc_auc", k, X_train, y_train, X_test, y_test,  X_train_pca, X_test_pca, "default", y_lim=[0.5,1])

def draw_all_accuracy_graphs(k):
	draw_single_metric_graph("accuracy", k, X_train, y_train, X_test, y_test,  X_train_pca, X_test_pca, "default",y_lim=[0.5,1])

def draw_all_precisionrecall_graphs(k):
	draw_precisionrecall_graph(k, X_train, y_train, X_test, y_test,  X_train_pca, X_test_pca, "default")

if __name__ == '__main__':
	draw_all_learning_curves()
	draw_all_roc_graphs(51)
	draw_all_accuracy_graphs(51)
	draw_all_precisionrecall_graphs(51)
	run_non_pca_knn()
	run_pca_knn()
