from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np

data = pd.read_csv("data/base_crabs.csv")
#print(data)
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


print("\n================= Non-PCA executions ========================")
print("\nGaussinan: \n")
run_all_bayes(X, y, X_train, y_train, X_test, y_test, "GaussianNB")
# print("\nMultinomial: \n")
# run_all_bayes(X, y, X_train, y_train, X_test, y_test, "MultinomialNB")
# print("\nBernoulli: \n")
# run_all_bayes(X, y, X_train, y_train, X_test, y_test, "BernoulliNB")
print("===============================================================")

print("\n================= With-PCA executions =======================")
print("\nGaussinan: \n")
run_all_bayes(X_pca, y, X_train_pca,y_train,X_test_pca,y_test, "GaussianNB")

# Doens't work
# print("\nMultinomial: \n")
# run_all_bayes(X_pca, y, X_train_pca,y_train,X_test_pca,y_test, "MultinomialNB")

# print("\nBernoulli: \n")
# run_all_bayes(X_pca, y, X_train_pca,y_train,X_test_pca,y_test, "BernoulliNB")
print("===============================================================")