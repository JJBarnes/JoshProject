#!/usr/bin/env python
# coding: utf-8

# In[51]:


#Import Modules
import uproot
import awkward
import numpy
import matplotlib.pyplot as plt
import matplotlib.figure as figure
import keras
import keras.layers as layers
from Sum import Sum
from sklearn.model_selection import train_test_split
import pandas as pd
from numpy.lib.recfunctions import structured_to_unstructured
from tensorflow.keras import callbacks
import tensorflow as tf
import time

# In[52]:

#Set hyperparameters
MASKVAL = -999
MAXTRACKS = 32
BATCHSIZE = 64
EPOCHS = 5000
MAXEVENTS = 99999999999999999
# VALFACTOR = 10
LR = 1e-3


# In[53]:


# Define Callbacks

# Define Early Stopping
early_stopping = callbacks.EarlyStopping(
    min_delta=0.001,
    patience = 50,
    restore_best_weights = True,
)

#Save weights throughout
save_weights = callbacks.ModelCheckpoint('/home/physics/phuspv/.ssh/Project/Weights/Inbox.ckpt', save_weights_only=True, monitor='loss', mode='min', save_best_only=True)

#Define ReducedLR
reduce_lr = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.9, patience=5, min_lr=1e-99)

#Define timehistory class to track average epoch times
class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

time_callback = TimeHistory()


# In[54]:


#Open the root file
tree = uproot.open("hffrag.root:CharmAnalysis")


# In[55]:


# Decide which branches of the tree we actually want to look at
# Not currently used!
branches = \
  [ \

  # true jet information
   "AnalysisAntiKt4TruthJets_pt"
   , "AnalysisAntiKt4TruthJets_eta"
   , "AnalysisAntiKt4TruthJets_phi"
   , "AnalysisAntiKt4TruthJets_m"


  # true b-hadron information
  # these b-hadrons are inside the truth jets
   , "AnalysisAntiKt4TruthJets_ghostB_pdgId"
    , "AnalysisAntiKt4TruthJets_ghostB_pt"
   , "AnalysisAntiKt4TruthJets_ghostB_eta"
   , "AnalysisAntiKt4TruthJets_ghostB_phi"
   , "AnalysisAntiKt4TruthJets_ghostB_m"
  

  # reconstructed jet information
   , "AnalysisJets_pt_NOSYS"
   , "AnalysisJets_eta"
   , "AnalysisJets_phi"
   , "AnalysisJets_m"


  # reconstructed track information
  , "AnalysisTracks_pt"
  , "AnalysisTracks_eta"
  , "AnalysisTracks_phi"
  , "AnalysisTracks_z0sinTheta"
  , "AnalysisTracks_d0sig"
  , "AnalysisTracks_d0"
  , "AnalysisTracks_d0sigPV"
  , "AnalysisTracks_d0PV"
  ]


  # True jet information
jetfeatures = \
  [ "AnalysisAntiKt4TruthJets_pt"
  , "AnalysisAntiKt4TruthJets_eta"
  , "AnalysisAntiKt4TruthJets_phi"
  , "AnalysisAntiKt4TruthJets_ghostB_pt"
  , "AnalysisAntiKt4TruthJets_ghostB_eta"
  , "AnalysisAntiKt4TruthJets_ghostB_phi"
  ]

# true b-hadron information
# these b-hadrons are inside the truth jets
bhadfeatures = \
   [ "AnalysisAntiKt4TruthJets_ghostB_pt"
   , "AnalysisAntiKt4TruthJets_ghostB_eta"
   , "AnalysisAntiKt4TruthJets_ghostB_phi"
   , "AnalysisAntiKt4TruthJets_ghostB_m"
   ]
  

# reconstructed track information
trackfeatures = \
  [ "AnalysisTracks_pt"
  , "AnalysisTracks_eta"
  , "AnalysisTracks_phi"
  , "AnalysisTracks_z0sinTheta"
  , "AnalysisTracks_d0sig"
  , "AnalysisTracks_d0"
  , "AnalysisTracks_d0sigPV"
  , "AnalysisTracks_d0PV"
  ]


# In[56]:


# Read in the requested branches from the file
features = tree.arrays(jetfeatures + trackfeatures, entry_stop=MAXEVENTS)


# In[57]:


#Find where angular distance is small
def matchTracks(jets, trks):
  jeteta = jets["AnalysisAntiKt4TruthJets_eta"] 
  jetphi = jets["AnalysisAntiKt4TruthJets_phi"]

  trketas = trks["AnalysisTracks_eta"]
  trkphis = trks["AnalysisTracks_phi"]

  detas = jeteta - trketas
  dphis = numpy.abs(jetphi - trkphis)

  # deal with delta phis being annoying
  awkward.where(dphis > numpy.pi, dphis - numpy.pi, dphis)

  return numpy.sqrt(dphis**2 + detas**2) < 0.4


# In[58]:


#Converting from polar to cartesian

#Used for jets
def ptetaphi2pxpypz(ptetaphi):
  pts = ptetaphi[:,0:1]
  etas = ptetaphi[:,1:2]
  phis = ptetaphi[:,2:3]

  pxs = pts * numpy.cos(phis)
  pys = pts * numpy.sin(phis)
  pzs = pts * numpy.sinh(etas)

  isinf = numpy.isinf(pzs)

  if numpy.any(isinf):
    print("inf from eta:")
    print(etas[isinf])
    raise ValueError("infinity from sinh(eta)")

  return numpy.concatenate([pxs, pys, pzs], axis=1)

#Used for tracks
def ptetaphi2pxpypz2(ptetaphi):
  pts = ptetaphi[:,:,0:1]
  etas = ptetaphi[:,:,1:2]
  phis = ptetaphi[:,:,2:3]

  mask = pts == MASKVAL
  #Looking in array and testing a condition - if finds mask, replaces mask with pt value
  pxs = numpy.where(mask, pts, pts * numpy.cos(phis)) # Apply transformation only to actual pT
  pys = numpy.where(mask, pts, pts * numpy.sin(phis))
  pzs = numpy.where(mask, pts, pts * numpy.sinh(etas))

  isinf = numpy.isinf(pzs)

  if numpy.any(isinf):
    print("inf from eta:")
    print(etas[isinf])
    raise ValueError("infinity from sinh(eta)")

  return numpy.concatenate([pxs, pys, pzs], axis=2)


# In[59]:


# Pads inputs with nans up to the given maxsize
def pad(xs, maxsize):
  #Find 'none' values in array and replace with MASKVAL (= fill_none)
  ys = \
    awkward.fill_none \
  ( awkward.pad_none(xs, maxsize, axis=1, clip=True) #Adding 'none' values to make sure it is correct size
  , MASKVAL
  )[:,:maxsize]

  return awkward.to_regular(ys, axis=1)


# In[60]:


def flatten1(xs, maxsize=-1):
  ys = {}
  for field in xs.fields:
    zs = xs[field]
    if maxsize > 0:
      zs = pad(zs, maxsize)
    ys[field] = zs

  return awkward.zip(ys)


# In[61]:


#Define histogram plotting functions
# returns a fixed set of bin edges
def fixedbinning(xmin, xmax, nbins):
  return numpy.mgrid[xmin:xmax:nbins*1j]


# define two functions to aid in plotting
def hist(xs, binning, normalized=False):
  ys = numpy.histogram(xs, bins=binning)[0]

  yerrs = numpy.sqrt(ys)

  if normalized:
    s = numpy.sum(ys)
    ys = ys / s
    yerrs = yerrs / s

  return ys, yerrs


def binneddensity(xs, binning, label=None, xlabel=None, ylabel="binned probability density"):
  fig = figure.Figure(figsize=(8, 8))
  plt = fig.add_subplot(111)

  ys , yerrs = hist(xs, binning, normalized=True)

  # determine the central value of each histogram bin
  # as well as the width of each bin
  # this assumes a fixed bin size.
  xs = (binning[1:]+binning[:-1]) / 2.0
  xerrs = ((binning[1:]-binning[:-1]) / 2.0)

  plt.errorbar \
    ( xs
    , ys
    , xerr=xerrs
    , yerr=yerrs
    , label=label
    , linewidth=0
    , elinewidth=2
    )

  plt.set_xlabel(xlabel)
  plt.set_ylabel(ylabel)

  return fig


# In[62]:


events = \
  features[awkward.sum(features["AnalysisAntiKt4TruthJets_pt"] > 25000, axis=1) > 0]

jets1 = events[jetfeatures][:,0] #First jet
tracks = events[trackfeatures]


# In[63]:


matchedtracks = tracks[matchTracks(jets1, tracks)] 
matchedtracks = flatten1(matchedtracks, MAXTRACKS) #Turn into regular np array


# In[64]:


bjets = awkward.sum(jets1["AnalysisAntiKt4TruthJets_ghostB_pt"] > 5000, axis=1) > 0 #Find b hadron jets with certain momentum
jets2 = jets1[bjets] #Jets identified as b jets are only jets considered
bhadspt= jets2["AnalysisAntiKt4TruthJets_ghostB_pt"][:,0] #np Stack here - Each sub array contains all the features of the jet (axis -1)
bhadseta = jets2["AnalysisAntiKt4TruthJets_ghostB_eta"][:, 0]
bhadsphi = jets2["AnalysisAntiKt4TruthJets_ghostB_phi"][:,0]
matchedtracks = matchedtracks[bjets]


# In[65]:


jets3 = structured_to_unstructured(jets2[jetfeatures[:-3]]) #number of features
matchedtracks = structured_to_unstructured(matchedtracks)


# In[66]:


jets4 = ptetaphi2pxpypz(jets3).to_numpy()
tracks = ptetaphi2pxpypz2(matchedtracks.to_numpy())
tracks = numpy.concatenate([tracks, matchedtracks[:,:,3:].to_numpy()], axis = 2)
bhadspt = bhadspt.to_numpy()
bhadseta = bhadseta.to_numpy()
bhads = numpy.stack([bhadspt, bhadseta], axis=-1)


# In[67]:


# Creating the training model

tracklayers = [ 32 , 32 , 32 , 32 , 32 ]
jetlayers = [ 64 , 64 , 64 , 64 , 64 ]


def buildModel(tlayers, jlayers, ntargets):  
  inputs = layers.Input(shape=(None, tlayers[0]))

  outputs = inputs
  outputs = layers.Masking(mask_value=MASKVAL)(outputs)

  for nodes in tlayers[:-1]:
    outputs = layers.Dropout(0.3)(outputs)
    outputs = layers.TimeDistributed(layers.Dense(nodes, activation='relu'))(outputs)
    outputs = layers.BatchNormalization()(outputs)

  outputs = layers.TimeDistributed(layers.Dense(tlayers[-1], activation='softmax'))(outputs)
  outputs = Sum()(outputs)

  for nodes in jlayers:
    outputs = layers.Dropout(0.3)(outputs)
    outputs = layers.Dense(nodes, activation='relu')(outputs)
    outputs = layers.BatchNormalization()(outputs)

  outputs = layers.Dense(ntargets + ntargets*(ntargets+1)//2)(outputs)

  return \
    keras.Model \
    ( inputs = inputs
    , outputs = outputs
    )


# In[68]:


## Generalise loss to n targets
##Convert b hadron features to cartesian


# In[69]:


# Creating the loss function
# this ignores any dimension beyond the first!
def LogNormal1D(true, meanscovs):
  ntargets = true.shape[1] #Number of variables predicting
  means = meanscovs[:,:ntargets] #First n targets are the means
  # ensure diagonal is positive
  logsigma = meanscovs[:,ntargets:2*ntargets]
  rest = meanscovs[:,2*ntargets:]

  # TODO
  # build matrix
  loss = 0
  for x in range(ntargets):
    loss = loss + ((means[:,x] - true[:,x])**2 / (2*keras.backend.exp(logsigma[:,x])**2)) + logsigma[:,x]
  return loss


# In[70]:


model = buildModel([len(trackfeatures)] + tracklayers, jetlayers, 2)

model.summary()

model.compile \
  ( loss = LogNormal1D
  , optimizer = keras.optimizers.Adam(learning_rate=LR)
  )

#Saves predictions after every epoch
predpts = []
errPredpT = []

def test(epochs):
  pred = model.predict(tracks, use_multiprocessing=True)
  predpts.append(pred[:,0])
  errPredpT.append(pred[:,2])

from keras.callbacks import LambdaCallback
call = LambdaCallback(on_epoch_end= lambda epochs,
        logs: test(epochs))


# In[71]:
# Loads the training and validation data sets
#X_train = numpy.load("/home/physics/phuspv/.ssh/Project/TrainingAndValidationData/X_train_data.npy")
#X_valid = numpy.load("/home/physics/phuspv/.ssh/Project/TrainingAndValidationData/X_valid_data.npy")
#y_train = numpy.load("/home/physics/phuspv/.ssh/Project/TrainingAndValidationData/y_train_data.npy")
#y_valid = numpy.load("/home/physics/phuspv/.ssh/Project/TrainingAndValidationData/y_valid_data.npy")

X_train, X_valid, y_train, y_valid = train_test_split(tracks, bhads, train_size=0.5, random_state = 42)

# In[72]:


# Trains the data
history = model.fit(tracks, bhads, batch_size=BATCHSIZE, callbacks = [reduce_lr, save_weights, time_callback, call], epochs=EPOCHS, use_multiprocessing=True)
numpy.save('/home/physics/phuspv/.ssh/Project/Epoch Times/Inbox', time_callback.times)
#Plots the loss curve and saves the data
history_df = numpy.log(pd.DataFrame(history.history))
LossFigure = history_df.loc[:, ['loss', 'val_loss']].plot().get_figure()
history_df.to_pickle("/home/physics/phuspv/.ssh/Project/Loss Data/Inbox.pkl")

#Saves the predictions
numpy.save('/home/physics/phuspv/.ssh/Project/Predictions')
numpy.save('/home/physics/phuspv/.ssh/Project/Prediction uncertainty')


# In[ ]:


#Saves the model
#model.save('Model')
model.save_weights('/home/physics/phuspv/.ssh/Project/Weights/Inbox.h5')


# In[ ]:



# In[ ]:
pred = model.predict(tracks)
pTDiff=pred[:,0] - bhads[:,0]
pTErr= numpy.exp(pred[:,2])
pTPull = pTDiff/pTErr

etaDiff = pred[:,1] - bhads[:,1]
etaErr = numpy.exp(pred[:, 3])
etaPull = etaDiff/etaErr

errors = numpy.exp(history_df)

#Print any results we want

print('Minimum validation loss is', numpy.min(errors.loc[:,'val_loss']))
print(errors.loc[errors['val_loss']==numpy.min(errors['val_loss'])])
print('pT Diff Mean = ' + str(numpy.mean(pTDiff)))
print('pT Diff Median = ' + str(numpy.median(pTDiff)))
print('pT Diff Std = ' + str(numpy.std(pTDiff)))
print('pT Diff IQR = ' + str(numpy.percentile(pTDiff, 75)-numpy.percentile(pTDiff, 25)))
print('Pull Mean = ' + str(numpy.mean(pTPull)))
print('Pull Median = ' + str(numpy.median(pTPull)))
print('Pull Std = ' + str(numpy.std(pTPull)))
print('Pull IQR = ' + str(numpy.percentile(pTPull, 75)-numpy.percentile(pTPull, 25)))
