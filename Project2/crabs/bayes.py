from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GRAPHS_FOLDER = "bayes_graphs/"
data = pd.read_csv("data/base_crabs.csv")
# data = pd.read_csv("data/base_noshows.csv")
#print(data)
X = np.array(data.drop("class",axis=1))
y = np.array(data["class"])
target_names = np.array(["No","Yes"])

# split dataset into training/test portions
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

# PCA part
pca = PCA(n_components=2).fit(X)
X_pca = pca.transform(X)

pca = PCA(n_components=2).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

def run_all_bayes(X, y, X_train, y_train, X_test, y_test, option):
    if option == "GaussianNB":
        gnb = GaussianNB()
    elif option == "MultinomialNB":
        gnb = MultinomialNB()
    elif option == "BernoulliNB":
        gnb = BernoulliNB()
    gnb.fit(X_train,y_train)
    y_pred = gnb.predict(X_test)
    print("\nBayes classifier")
    print(classification_report(y_test,y_pred,target_names=target_names))
    print(confusion_matrix(y_test,y_pred, labels=range(2)))
    print("Accuracy score: %f" % (accuracy_score(y_test,y_pred)))
    print("ROC auc score: %f" % (roc_auc_score(y_test,y_pred)))
    res = 0
    # Nao funciona no PCA
    # for el in range(0,len(X_test)):
    #     sol = X_test[el,0]
    #     if y_pred[el] != sol:
    #         res = res + 1 
    # print("Number of mislabeled points out of a total %d points : %d" % (len(X_test),res))
    if option == "GaussianNB":
        gnb = GaussianNB()
    elif option == "MultinomialNB":
        gnb = MultinomialNB()
    elif option == "BernoulliNB":
        gnb = BernoulliNB()
    gnb.fit(X,y)
    print("Cross-Validation (10-fold) score: %f" % (cross_val_score(gnb, X, y, cv=10).mean()))

def draw_learning_curve(X, y, X_pca, filename):
    clf = GaussianNB()
    train_sizes,train_scores, test_scores = learning_curve(
        clf, X, y, cv=10, n_jobs=8)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    clf_pca = GaussianNB()
    train_sizes_pca,train_scores_pca, test_scores_pca = learning_curve(
        clf, X_pca, y, cv=10, n_jobs=8)
    train_scores_mean_pca = np.mean(train_scores_pca, axis=1)
    train_scores_std_pca = np.std(train_scores_pca, axis=1)
    test_scores_mean_pca = np.mean(test_scores_pca, axis=1)
    test_scores_std_pca = np.std(test_scores_pca, axis=1)

    f = plt.figure()
    plt.title("Learning Curve Naive Bayes - " + filename)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.gca().set_ylim([0.46,1.06])
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
    f.savefig(GRAPHS_FOLDER+"lc_bayes_"+filename+".png",bbox_inches="tight")
    f.savefig(GRAPHS_FOLDER+"lc_bayes_"+filename+".pdf",bbox_inches="tight")

def run_non_pca_knn():  
    print("\n================= Basic Non-PCA =============================")
    run_all_bayes(X, y, X_train, y_train, X_test, y_test, "GaussianNB")
    print("===============================================================")


def run_pca_knn():
    print("\n================ Basic PCA executions =======================")
    run_all_bayes(X_pca, y, X_train_pca,y_train,X_test_pca,y_test,"GaussianNB")
    print("===============================================================")

def draw_all_learning_curves():
    draw_learning_curve(X,y,X_pca,"default")

run_non_pca_knn()
run_pca_knn()
draw_all_learning_curves()