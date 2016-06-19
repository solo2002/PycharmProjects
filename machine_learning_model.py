from sklearn import datasets
from sklearn import svm
import numpy as np

iris = datasets.load_iris()
# SVM
clf = svm.LinearSVC()

print 'iris.data', type(iris.data)
print 'iris.target', type(iris.target)
target = [1,0,0,0,1,0,0]
data =[[ 5.1,  3.5,  1.4,  0.2],
 [ 4.9,  3.,   1.4,  0.2],
 [ 4.7,  3.2,  1.3,  0.2],
 [ 4.6 , 3.1 , 1.5,  0.2],
 [ 5.   ,3.6  ,1.4 , 0.2],
 [ 5.4  ,3.9 , 1.7 , 0.4],
 [ 4.6  ,3.4,  1.4 , 0.3],
]
clf.fit(data, target)
print clf.predict([[ 5.0,  3.6,  1.3,  0.25]])
# linear SVC
svc = svm.SVC(kernel = 'linear')
svc.fit(iris.data, iris.target)

# KNN
from sklearn import neighbors

knn = neighbors.KNeighborsClassifier()
knn.fit(iris.data, iris.target)
print knn.predict([[0.1, 0.2, 0.3, 0.4]])

perm = np.random.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]
knn.fit(iris.data[:100], iris.target[:100])
print knn.score(iris.data[100:], iris.target[100:])

# unsupervised classification
# k means
from sklearn import cluster, datasets
from sklearn import cluster, datasets
k_means = cluster.KMeans(n_clusters=3)
k_means.fit(iris.data)

print(k_means.labels_[::10])
print(iris.target[::10])

# PCA
from skearn import pca
pca = decomposition.PCA(n_components = 2)
pca.fit(iris.data)
X = pca.transform(iris.data)

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
print("Number of mislabeled points out of a total %d points : %d"
    % (iris.data.shape[0],(iris.target != y_pred).sum()))

# Multinomial NB, implements the naive Bayes algorithm for multinomially
# distributed data, and is one of the two classic naive Bayes variants used in
# text classification