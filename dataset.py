__author__ = 'SRC'

from sklearn.datasets import fetch_20newsgroups
import pickle

def load_dataset(name):
    dataset = pickle.load(open('dataset/'+name,'rb'))
    return dataset

if __name__ == '__main__':
    dataset = fetch_20newsgroups(subset='all')
    pickle.dump(dataset, open('dataset/20newsgroups','wb'))