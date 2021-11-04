
#comapare pickle loaders
from time import time
import pickle
import os
import pickle as cPickle
from sklearn.externals import joblib

file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'database.clf')
t1 = time()
lis = []
d = pickle.load(open(file,"rb"))
print("time for loading file size with pickle", os.path.getsize(file),"KB =>", time()-t1)

t1 = time()
cPickle.load(open(file,"rb"))
print("time for loading file size with cpickle", os.path.getsize(file),"KB =>", time()-t1)

t1 = time()
joblib.load(file)
print("time for loading file size joblib", os.path.getsize(file),"KB =>", time()-t1)
