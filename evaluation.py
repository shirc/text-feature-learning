__author__ = 'SRC'

import dataset
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.decomposition import PCA
import featurelearning

def evaluate(dataset_name, fl, ratio):
    print dataset_name, fl.__name__, ratio
    d = dataset.load_dataset(dataset_name)
    fea = d.data
    label = d.target
    fea = fl(fea)
    ss = StratifiedShuffleSplit(label, 3, test_size=(1-ratio), random_state=0)
    svc = LinearSVC()
    for train, test in ss:
        svc.fit(fea[train,:], label[train,:])
        predict = svc.predict(fea[test, :])
        acc = accuracy_score(label[test, :], predict)
        print acc

if __name__ == '__main__':
    pca = PCA()
    train = fetch_20newsgroups_vectorized('train')
    test = fetch_20newsgroups_vectorized('test')
    svc = LinearSVC()
    train_data = pca.fit_transform(train.data.toarray())
    svc.fit(train_data, train.target)
    test_data = pca.transform(test.data.toarray())
    predict = svc.predict(test_data)
    acc = accuracy_score(test.target, predict)
    print acc
    # evaluate('20newsgroups', featurelearning.TF_IDF, 0.1)
    # evaluate('20newsgroups', featurelearning.LDA, 0.1)