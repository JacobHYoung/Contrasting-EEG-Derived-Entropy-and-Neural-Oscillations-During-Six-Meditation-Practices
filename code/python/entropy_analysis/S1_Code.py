from numpy import *
from numpy.linalg import * 
from scipy import signal
from scipy.signal import hilbert
from scipy.stats import ranksums
from scipy.io import savemat
from scipy.io import loadmat
from random import *
from itertools import combinations
from pylab import *

'''
Python code to compute complexity measures LZc, ACE and SCE as described in "Complexity of multi-dimensional spontaneous EEG decreases during propofol induced general anaesthesia"

Author: m.schartner@sussex.ac.uk
Date: 09.12.14

To compute the complexity meaures LZc, ACE, SCE for continuous multidimensional time series X, where rows are time series (minimum 2), and columns are observations, type the following in ipython: 
 
execfile('CompMeasures.py')
LZc(X)
ACE(X)
SCE(X)


Some functions are shared between the measures.
'''

def Pre(X):
 '''
 Detrend and normalize input data, X a multidimensional time series
 '''
 ro,co=shape(X)
 Z=zeros((ro,co))
 for i in range(ro):
  Z[i,:]=signal.detrend(X[i,:]-mean(X[i,:]), axis=0)
 return Z


##########
'''
LZc - Lempel-Ziv Complexity, column-by-column concatenation
'''
##########

def cpr(string):
 '''
 Lempel-Ziv-Welch compression of binary input string, e.g. string='0010101'. It outputs the size of the dictionary of binary words.
 '''
 d={} 
 w = ''
 i=1
 for c in string: 
  wc = w + c
  if wc in d:
   w = wc
  else:
   d[wc]=wc
   w = c
  i+=1
 return len(d)

def str_col(X):
 '''
 Input: Continuous multidimensional time series
 Output: One string being the binarized input matrix concatenated comlumn-by-column
 '''
 ro,co=shape(X)
 TH=zeros(ro)
 M=zeros((ro,co))
 for i in range(ro):
  M[i,:]=abs(hilbert(X[i,:]))
  TH[i]=mean(M[i,:])

 s=''
 for j in xrange(co):
  for i in xrange(ro):
   if M[i,j]>TH[i]:
    s+='1'
   else:
    s+='0'

 return s

def LZc(X):
 '''
 Compute LZc and use shuffled result as normalization
 '''
 X=Pre(X)
 SC=str_col(X)
 M=list(SC)
 shuffle(M)
 w=''
 for i in range(len(M)):
  w+=M[i]
 return cpr(SC)/float(cpr(w))

##########
'''
ACE - Amplitude Coalition Entropy
'''
##########

def map2(psi):
 '''
 Bijection, mapping each binary column of binary matrix psi onto an integer.
 '''
 ro,co=shape(psi) 
 c=zeros(co)
 for t in range(co):
  for j in range(ro):
   c[t]=c[t]+psi[j,t]*(2**j)
 return c

def binTrowH(M):
 '''
 Input: Multidimensional time series M
 Output: Binarized multidimensional time series
 '''
 ro,co=shape(M)
 M2=zeros((ro,co))
 for i in range(ro):
  M2[i,:]=signal.detrend(M[i,:],axis=0)
  M2[i,:]=M2[i,:]-mean(M2[i,:])
 M3=zeros((ro,co))
 for i in range(ro):
  M2[i,:]=abs(hilbert(M2[i,:]))
  th=mean(M2[i,:]) 
  for j in range(co):
   if M2[i,j] >= th :
    M3[i,j]=1
   else:
    M3[i,j]=0 
 return M3

def entropy(string):
 '''
 Calculates the Shannon entropy of a string
 '''
 string=list(string)
 prob = [ float(string.count(c)) / len(string) for c in dict.fromkeys(list(string)) ]
 entropy = - sum([ p * log(p) / log(2.0) for p in prob ])

 return entropy


def ACE(X):
 '''
 Measure ACE, using shuffled reslut as normalization.
 '''
 X=Pre(X)
 ro,co=shape(X)
 M=binTrowH(X)
 E=entropy(map2(M))
 for i in range(ro):
  shuffle(M[i])
 Es=entropy(map2(M))
 return E/float(Es)


##########
'''
SCE - Synchrony Coalition Entropy
'''
##########

def diff2(p1,p2):
 '''
 Input: two series of phases 
 Output: synchrony time series thereof 
 '''
 d=array(abs(p1-p2))
 d2=zeros(len(d))
 for i in range(len(d)):
  if d[i]>pi:
   d[i]=2*pi-d[i]
  if d[i]<0.8:
   d2[i]=1

 return d2


def Psi(X):
 '''
 Input: Multi-dimensional time series X
 Output: Binary matrices of synchrony for each series
 '''
 X=angle(hilbert(X))
 ro,co=shape(X)
 M=zeros((ro, ro-1, co))
 
 #An array containing 'ro' arrays of shape 'ro' x 'co', i.e. being the array of synchrony series for each channel. 
 
 for i in range(ro):
  l=0
  for j in range(ro):
   if i!=j:
    M[i,l]=diff2(X[i],X[j])
    l+=1
 
 return M

def BinRan(ro,co):
 '''
 Create random binary matrix for normalization
 '''

 y=rand(ro,co)
 for i in range(ro):
  for j in range(co):
   if y[i,j]>0.5:
    y[i,j]=1
   else:
    y[i,j]=0
 return y
 
def SCE(X): 
 X=Pre(X)    
 ro,co=shape(X)
 M=Psi(X)
 ce=zeros(ro)
 norm=entropy(map2(BinRan(ro-1,co)))
 for i in range(ro):
  c=map2(M[i])
  ce[i]=entropy(c)

 return mean(ce)/norm,ce/norm

# START our work

# load necessary packages
import os 
import pandas as pd
import numpy as np
import sys

print("PYTHON VERSION: ",sys.version)
# initialize dictionary mapping file name to lzc and ace value
file_name_to_lzc_value_dictionary = {}
file_name_to_ace_value_dictionary = {}
# specify the directory we are reading the files from
participant_type = "ze"
data_input_path = "../../../data/Interp/" + participant_type + "/"
#     read in data
# (1) get a list of files in the directory of the interp data
list_of_files = []
for f in os.listdir(data_input_path):
  if ".csv" in f:
    list_of_files.append(f)

# now iteratre through each file
# f means file
for file_name in list_of_files:

  # now use pandas to extract the matrix from the csv files
  time_series_matrix = (pd.read_csv(data_input_path + file_name)).to_numpy() # we get a nice data frame now! 2D array
  # now Lmplz complexity metric, ace
  lzc_metric = LZc(time_series_matrix)
  ace_metric = ACE(time_series_matrix)
  # add new key value paris to the dictionaries by the update function 
  file_name_to_lzc_value_dictionary.update({file_name: [lzc_metric]})
  file_name_to_ace_value_dictionary.update({file_name: [ace_metric]})


# we created a dictionary with key value paris mapping files to their respective complexity meaures, now
# output that data to two csv files, for the ace and lzc respectively
data_output_path = "../../../data/complexity/"
lzc_df = pd.DataFrame(file_name_to_lzc_value_dictionary, columns = list_of_files)
lzc_df.to_csv(data_output_path + participant_type + "_lzc_for_each_participant.csv", header = True, index = False)
ace_df = pd.DataFrame(file_name_to_ace_value_dictionary, columns = list_of_files)
ace_df.to_csv(data_output_path + participant_type + "_ace_for_each_participant.csv", header = True, index = False)

















