__author__ = 'SRC'

from sklearn.feature_extraction.text import TfidfTransformer
from gensim.models import LdaModel
from gensim import matutils

def TF_IDF(fea):
    return TfidfTransformer().fit_transform(fea)

def PCA(fea):
    pass

def LDA(fea):
    corpus = matutils.Scipy2Corpus(fea)
    lda = LdaModel(corpus, num_topics=100)
    return matutils.corpus2csc(lda[corpus], 100).transpose()

def tf_idf_LDA(fea):
    pass