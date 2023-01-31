#!/usr/bin/env python
# coding: utf-8

# In[434]:


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


# In[564]:


#Set hyperparameters
MASKVAL = -999
MAXTRACKS = 32
BATCHSIZE = 64
EPOCHS = 200
MAXEVENTS = 99999999999999999
# VALFACTOR = 10
LR = 1e-2


# In[565]:


# Define Callbacks

# Define Early Stopping
early_stopping = callbacks.EarlyStopping(
    min_delta=0.001,
    patience = 50,
    restore_best_weights = True, monitor = 'val_loss'
)

#Define ReducedLR
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9,
                              patience=20, min_lr=1e-8)


# In[437]:


#Open the root file
tree = uproot.open("/storage/epp2/phswmv/data/hffrag/hffrag.root:CharmAnalysis")


# In[438]:


# Decide which branches of the tree we actually want to look at
# Not currently used!
branches =   [ 
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
jetfeatures =   [ "AnalysisAntiKt4TruthJets_pt"
  , "AnalysisAntiKt4TruthJets_eta"
  , "AnalysisAntiKt4TruthJets_phi"
  , "AnalysisAntiKt4TruthJets_m"
  , "AnalysisAntiKt4TruthJets_ghostB_pt"
  , "AnalysisAntiKt4TruthJets_ghostB_eta"
  , "AnalysisAntiKt4TruthJets_ghostB_phi"
  ,"AnalysisAntiKt4TruthJets_ghostB_m"
  ]


# true b-hadron information
# these b-hadrons are inside the truth jets
bhadfeatures =    [ "AnalysisAntiKt4TruthJets_ghostB_pt"
   , "AnalysisAntiKt4TruthJets_ghostB_eta"
   , "AnalysisAntiKt4TruthJets_ghostB_phi"
   , "AnalysisAntiKt4TruthJets_ghostB_m"
   ]
  

# reconstructed track information
trackfeatures =   [ "AnalysisTracks_pt"
  , "AnalysisTracks_eta"
  , "AnalysisTracks_phi"
  , "AnalysisTracks_z0sinTheta"
  , "AnalysisTracks_d0sig"
  , "AnalysisTracks_d0"
  , "AnalysisTracks_d0sigPV"
  , "AnalysisTracks_d0PV"
  ]


# In[439]:


# Read in the requested branches from the file
features = tree.arrays(jetfeatures + trackfeatures, entry_stop=MAXEVENTS)


# In[440]:


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


# In[441]:


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


# In[442]:


# Pads inputs with nans up to the given maxsize
def pad(xs, maxsize):
  #Find 'none' values in array and replace with MASKVAL (= fill_none)
  ys =     awkward.fill_none   ( awkward.pad_none(xs, maxsize, axis=1, clip=True) #Adding 'none' values to make sure it is correct size
  , MASKVAL
  )[:,:maxsize]

  return awkward.to_regular(ys, axis=1)


# In[443]:


def flatten1(xs, maxsize=-1):
  ys = {}
  for field in xs.fields:
    zs = xs[field]
    if maxsize > 0:
      zs = pad(zs, maxsize)
    ys[field] = zs

  return awkward.zip(ys)


# In[444]:


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

  plt.errorbar     ( xs
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


# In[445]:


events =   features[awkward.sum(features["AnalysisAntiKt4TruthJets_pt"] > 25000, axis=1) > 0]

jets1 = events[jetfeatures][:,0] #First jet
tracks = events[trackfeatures]


# In[446]:


matchedtracks = tracks[matchTracks(jets1, tracks)] 
matchedtracks = flatten1(matchedtracks, MAXTRACKS) #Turn into regular np array


# In[447]:


bjets = awkward.sum(jets1["AnalysisAntiKt4TruthJets_ghostB_pt"] > 5000, axis=1) > 0 #Find b hadron jets with certain momentum
jets2 = jets1[bjets] #Jets identified as b jets are only jets considered
bhadspt= jets2["AnalysisAntiKt4TruthJets_ghostB_pt"][:,0] #np Stack here - Each sub array contains all the features of the jet (axis -1)
bhadseta = jets2["AnalysisAntiKt4TruthJets_ghostB_eta"][:, 0]
bhadsphi = jets2["AnalysisAntiKt4TruthJets_ghostB_phi"][:,0]
bmass = jets2["AnalysisAntiKt4TruthJets_ghostB_m"][:,0].to_numpy()
matchedtracks = matchedtracks[bjets]


# In[448]:


jets3 = structured_to_unstructured(jets2[jetfeatures[:-3]]) #number of features
matchedtracks = structured_to_unstructured(matchedtracks)


# In[449]:


jets4 = ptetaphi2pxpypz(jets3).to_numpy()
tracks = ptetaphi2pxpypz2(matchedtracks.to_numpy())
tracks = numpy.concatenate([tracks, matchedtracks[:,:,3:].to_numpy()], axis = 2)
bhadspt = bhadspt.to_numpy()
bhadseta = bhadseta.to_numpy()
bhads = numpy.stack([bhadspt, bhadseta], axis=-1)
bhads2 = numpy.stack([bhadspt, bhadseta, bhadsphi], axis=-1)
bhadscart = ptetaphi2pxpypz(bhads2).to_numpy()
matchedtracks = matchedtracks.to_numpy()

jetpT = jets3[:,0]
jetEta = jets3[:,1]
jetphi = jets3[:,2]
jets5 = numpy.stack([jetpT, jetEta, jetphi], axis = -1).to_numpy()


# In[ ]:


bhadspx = bhadscart[:,0] 
bhadspy = bhadscart[:,1]
bhadspz = bhadscart[:,2]
bhadsmom = (bhadspx**2+bhadspy**2+bhadspz**2)
bhadenergy = (bhadsmom+bmass**2)
truejetpx = jets4[:,0]
truejetpy = jets4[:,1]
truejetpz = jets4[:,2]
truejetpT = jets5[:,0]
truejeteta = jets5[:,1]
truejetmass = jets2["AnalysisAntiKt4TruthJets_m"].to_numpy()

bfeatures = numpy.stack([bhadspx, bhadspy, bhadspz, bhadspt, bhadseta, bmass], axis = -1)
truejetfeatures = numpy.stack([truejetpx, truejetpy, truejetpz, truejetpT, truejeteta, truejetmass], axis = -1)

scaler = numpy.stack([truejetpx, truejetpy, truejetpz, truejetpT, truejeteta, truejetmass], axis=-1)
targets = bfeatures/scaler
targets = targets.to_numpy()


# In[566]:


# Creating the training model

tracklayers = [ 32 , 32 , 32 , 32 , 32 ]
jetlayers = [ 64 , 64, 64, 64, 64 ]

def buildModel(tlayers, jlayers, ntargets):

  inputs = layers.Input(shape=(None, tlayers[0]))

  outputs = inputs
  outputs = layers.Masking(mask_value=MASKVAL)(outputs)
  outputs = layers.Normalization()(outputs)

  for nodes in tlayers[:-1]:
    outputs = layers.Dropout(0.30)(outputs)
    outputs = layers.TimeDistributed(layers.Dense(nodes, activation='leaky_relu', kernel_initializer='he_normal', kernel_regularizer='l1_l2'))(outputs)
    outputs = layers.BatchNormalization()(outputs)

  outputs = layers.TimeDistributed(layers.Dense(tlayers[-1], activation='softmax'))(outputs)
  outputs = Sum()(outputs)


  for nodes in jlayers:
    outputs = layers.Dropout(0.3)(outputs)
    outputs = layers.Dense(nodes, activation='leaky_relu', kernel_initializer='he_normal', kernel_regularizer='l1_l2')(outputs)
    outputs = layers.BatchNormalization()(outputs)

  outputs = layers.Dense(ntargets + ntargets*(ntargets+1)//2)(outputs)


  
  return     keras.Model     ( inputs = inputs
    , outputs = outputs
    )


# In[567]:


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


# In[568]:


model = buildModel([len(trackfeatures)] + tracklayers, jetlayers, 6)


# In[569]:


model.summary()


# In[570]:


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer._decayed_lr(tf.float32)
    return lr

import tensorflow_addons as tfa
steps_per_epoch = len(matchedtracks)
clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate = 1e-8,maximal_learning_rate=0.01, scale_fn = lambda x: 1/(2**(x-1)), step_size = 2*steps_per_epoch)
steps = numpy.arange(0,100 * steps_per_epoch)
lr = clr(steps)

optimizer = keras.optimizers.Adam(learning_rate = lr, clipnorm = 6)
lr_metric = get_lr_metric(optimizer)

model.compile   ( loss = LogNormal1D
  , optimizer = optimizer,
  metrics = [lr_metric]
  )


# In[571]:


class PerformancePlotCallback(keras.callbacks.Callback):
    def __init__(self, x_test, y_test, model_name):
        self.x_test = x_test
        self.y_test = y_test
        self.model_name = model_name
        
    def on_epoch_end(self, epoch, logs={}):
        pred = self.model.predict([matchedtracks, truejetmass], use_multiprocessing=True)
        pred[:,:6] = pred[:,:6]*scaler
        pred[:,6:12] = numpy.exp(pred[:,6:12])
        pred[:,6:12] = pred[:,6:12]*scaler


        fig, axes = plt.subplots(2,3, figsize = (15,12))
        plt.suptitle("Correlations between true and predicted, epoch: " + str(epoch) + " LR: "+ str(lr_metric))
        axes[0,0].scatter(bfeatures[:,0], pred[:,0], color='red', ec='black', alpha = 0.6)
        axes[0,0].set(xlabel='True px (MeV)', ylabel='Predicted px (MeV)', xlim=(-500000,500000), ylim=(-500000,500000))
        axes[0,1].scatter(bfeatures[:,1], pred[:,1], color = 'blue', ec='black', alpha = 0.6)
        axes[0,1].set(xlabel='True py (MeV)', ylabel='Predicted py (MeV)', xlim=(-500000,500000), ylim=(-500000,500000))
        axes[0,2].scatter(bfeatures[:,2], pred[:,2], color = 'green', ec='black', alpha = 0.6)
        axes[0,2].set(xlabel='True pz (MeV)', ylabel='Predicted pz (MeV)', xlim=(-700000,700000), ylim=(-700000,700000))
        axes[1,0].scatter(bfeatures[:,3], pred[:,3], color = 'yellow', ec='black', alpha = 0.6)
        axes[1,0].set(xlabel='True pT (MeV)', ylabel='Predicted pT (MeV)', xlim=(0,500000), ylim=(0,500000))
        axes[1,1].scatter(bfeatures[:,4], pred[:,4], color = 'cyan', ec='black', alpha = 0.6)
        axes[1,1].set(xlabel='True eta', ylabel='Predicted eta', xlim=(-5,5), ylim=(-5,5))
        axes[1,2].scatter(bfeatures[:,5], pred[:,5], color = 'magenta', ec='black', alpha = 0.6)
        axes[1,2].set(xlabel='True mass (MeV)', ylabel='Predicted mass (MeV)', xlim=(0,10000), ylim=(0,10000))
        plt.savefig('/home/physics/phuspv/.ssh/Project/Plot2/long_run_lognormal/'+self.model_name+"_"+str(epoch))
        plt.close()

        massdiff = pred[:,5] - bmass
        masserr = pred[:,11]
        massPull = massdiff / masserr
        fig = binneddensity(massPull, fixedbinning(-10,10,100), xlabel = 'Mass Pull', label = ("Mass Pull - Epoch: "+str(epoch)))
        fig.savefig('/home/physics/phuspv/.ssh/Project/Plot2/Mass_Pull_Plots/'+self.model_name+"_"+str(epoch))
        plt.close()
        fig = binneddensity(massPull, fixedbinning(numpy.min(massPull),numpy.max(massPull),100), xlabel = 'Mass Pull', label = ("Mass Pull - Epoch: "+str(epoch)))
        fig.savefig('/home/physics/phuspv/.ssh/Project/Plot2/Mass_Pull_Plots_Scaled/'+self.model_name+"_"+str(epoch))
        plt.close()
        fig = binneddensity(massdiff, fixedbinning(numpy.min(massdiff),numpy.max(massdiff),100), xlabel = 'Mass Difference', label = ("Mass Difference - Epoch: "+str(epoch)))
        fig.savefig('/home/physics/phuspv/.ssh/Project/Plot2/Mass_Diff_Plots_Scaled/'+self.model_name+"_"+str(epoch))
        plt.close()
        fig = binneddensity(massdiff, fixedbinning(-5000,5000,100), xlabel = 'Mass Difference', label = ("Mass Difference - Epoch: "+str(epoch)))
        fig.savefig('/home/physics/phuspv/.ssh/Project/Plot2/Mass_Diff_Plots/'+self.model_name+"_"+str(epoch))
        plt.close()


# In[572]:


# Splits the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(matchedtracks, bhads, train_size = 0.5)

performance = PerformancePlotCallback(X_valid, y_valid, "Long Test")


# In[573]:


# Trains the data
history = model.fit(X_train, y_train, validation_data = (X_valid, y_valid), batch_size=BATCHSIZE, callbacks = reduce_lr, epochs=EPOCHS)

#Plots and saves the loss curve
history_df = numpy.log(pd.DataFrame(history.history))
LossFigure = history_df.loc[:, ['loss', 'val_loss']].plot().get_figure()
LossFigure.savefig('Long Loss.png')


# In[ ]:


#Saves the model
#model.save('Model')
#model.save_weights('Model Weights 5000 epoch.h5')


# In[ ]:


# Uses the model to predict validation set
pred = model.predict(matchedtracks)


# In[ ]:


pred[:, :6] = (pred[:,:6]* scaler)
pred[:,6:12] = numpy.exp(pred[:,6:12])
pred[:,6:12] = (pred[:,6:12]* scaler)


# In[ ]:


fig, axes = plt.subplots(2,3, figsize = (18,12))
plt.suptitle("Correlations between true and predicted")
axes[0,0].scatter(bfeatures[:,0], pred[:,0], color='red', ec='black', alpha = 0.6)
axes[0,0].set(xlabel='True px (MeV)', ylabel='Predicted px (MeV)', xlim=(-500000,500000), ylim=(-500000,500000))
axes[0,1].scatter(bfeatures[:,1], pred[:,1], color = 'blue', ec='black', alpha = 0.6)
axes[0,1].set(xlabel='True py (MeV)', ylabel='Predicted py (MeV)', xlim=(-500000,500000), ylim=(-500000,500000))
axes[0,2].scatter(bfeatures[:,2], pred[:,2], color = 'green', ec='black', alpha = 0.6)
axes[0,2].set(xlabel='True pz (MeV)', ylabel='Predicted pz (MeV)', xlim=(-2000000,2000000), ylim=(-2000000,2000000))
axes[1,0].scatter(bfeatures[:,3], pred[:,3], color = 'yellow', ec='black', alpha = 0.6)
axes[1,0].set(xlabel='True pT (MeV)', ylabel='Predicted pT (MeV)', xlim=(0,500000), ylim=(0,500000))
axes[1,1].scatter(bfeatures[:,4], pred[:,4], color = 'cyan', ec='black', alpha = 0.6)
axes[1,1].set(xlabel='True eta', ylabel='Predicted eta')
axes[1,2].scatter(bfeatures[:,5], pred[:,5], color = 'magenta', ec='black', alpha = 0.6)
axes[1,2].set(xlabel='True Mass (MeV)', ylabel='Predicted Mass (MeV)')
plt.tight_layout()
plt.show()
plt.savefig("Long Correlations")
plt.close()


# In[ ]:


xs = numpy.linspace(0, len(bfeatures), len(bfeatures))
print(numpy.shape(xs))
fig, axes = plt.subplots(1,3, figsize = (16,8))
axes[0].scatter(xs, pred[:,5], label = 'Pred', color = 'red', ec = 'black', alpha = 0.6)
axes[0].set_ylim(5000,6000)
axes[0].legend()
axes[1].scatter(xs, bmass, label = 'True', color='green', ec = 'black', alpha = 0.6)
axes[1].set_ylim(5000,6000)
axes[1].legend()
axes[2].scatter(xs, bmass - pred[:,5], label = 'True - Pred', color = 'blue', ec = 'black', alpha = 0.6)
axes[2].legend()
plt.show()
plt.savefig('Long Mass Diff')
plt.close()

