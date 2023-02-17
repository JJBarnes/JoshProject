#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Modules
import uproot
import awkward
import numpy
import matplotlib.pyplot as plt
import matplotlib.figure as figure
import tensorflow as tf
import keras
import keras.layers as layers
from Sum import Sum
from sklearn.model_selection import train_test_split
import pandas as pd
from numpy.lib.recfunctions import structured_to_unstructured
from tensorflow.keras import callbacks
import time


# In[2]:


#Set hyperparameters
MASKVAL = -999
MAXTRACKS = 32
BATCHSIZE = 64
EPOCHS = 1000
MAXEVENTS = 99999999999999999
# VALFACTOR = 10
LR = 1e-2


# In[3]:


# Define Callbacks

# Define Early Stopping
early_stopping = callbacks.EarlyStopping(
    min_delta=0.1,
    patience = 20,
    restore_best_weights = True, monitor = 'loss'
)

#Define ReducedLR
reduce_lr = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.75,
                              patience=10, min_lr=0)


#Define learning schedule

def scheduler(epoch, lr):
  if epoch == (EPOCHS/2):
    return LR
  else:
    return lr

lrscheduler = callbacks.LearningRateScheduler(scheduler)



#Save weights throughout
save_weights = callbacks.ModelCheckpoint('/storage/physics/phuspv/Project/Weights/LongInbox.ckpt', save_weights_only=True, monitor='val_loss', mode='min', save_best_only=True)


# In[4]:


#Open the root file
tree = uproot.open("hffrag.root:CharmAnalysis")


# In[5]:


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


# In[6]:


# Read in the requested branches from the file
features = tree.arrays(jetfeatures + trackfeatures, entry_stop=MAXEVENTS)


# In[7]:


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


# In[8]:


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


# In[9]:


# Pads inputs with nans up to the given maxsize
def pad(xs, maxsize):
  #Find 'none' values in array and replace with MASKVAL (= fill_none)
  ys =     awkward.fill_none   ( awkward.pad_none(xs, maxsize, axis=1, clip=True) #Adding 'none' values to make sure it is correct size
  , MASKVAL
  )[:,:maxsize]

  return awkward.to_regular(ys, axis=1)


# In[10]:


def flatten1(xs, maxsize=-1):
  ys = {}
  for field in xs.fields:
    zs = xs[field]
    if maxsize > 0:
      zs = pad(zs, maxsize)
    ys[field] = zs

  return awkward.zip(ys)


# In[11]:


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


# In[12]:


events =   features[awkward.sum(features["AnalysisAntiKt4TruthJets_pt"] > 25000, axis=1) > 0]

jets1 = events[jetfeatures][:,0] #First jet
tracks = events[trackfeatures]


# In[13]:


matchedtracks = tracks[matchTracks(jets1, tracks)] 
matchedtracks = flatten1(matchedtracks, MAXTRACKS) #Turn into regular np array


# In[14]:


bjets = awkward.sum(jets1["AnalysisAntiKt4TruthJets_ghostB_pt"] > 5000, axis=1) > 0 #Find b hadron jets with certain momentum
jets2 = jets1[bjets] #Jets identified as b jets are only jets considered
bhadspt= jets2["AnalysisAntiKt4TruthJets_ghostB_pt"][:,0] #np Stack here - Each sub array contains all the features of the jet (axis -1)
bhadseta = jets2["AnalysisAntiKt4TruthJets_ghostB_eta"][:, 0]
bhadsphi = jets2["AnalysisAntiKt4TruthJets_ghostB_phi"][:,0]
bmass = jets2["AnalysisAntiKt4TruthJets_ghostB_m"][:,0].to_numpy()
matchedtracks = matchedtracks[bjets]


# In[15]:


jets3 = structured_to_unstructured(jets2[jetfeatures[:-5]]) #number of features
matchedtracks = structured_to_unstructured(matchedtracks)
matchedtracks = matchedtracks.to_numpy()


# In[16]:


jets4 = ptetaphi2pxpypz(jets3).to_numpy()
tracks = ptetaphi2pxpypz2(matchedtracks)
tracks = numpy.concatenate([tracks, matchedtracks[:,:,3:]], axis = 2)
bhadspt = bhadspt.to_numpy()
bhadseta = bhadseta.to_numpy()
bhads = numpy.stack([bhadspt, bhadseta], axis=-1)
bhads2 = numpy.stack([bhadspt, bhadseta, bhadsphi], axis=-1)
bhadscart = ptetaphi2pxpypz(bhads2).to_numpy()

jetpT = jets3[:,0]
jetEta = jets3[:,1]
jetphi = jets3[:,2]
jets5 = numpy.stack([jetpT, jetEta, jetphi], axis = -1).to_numpy()


# In[17]:


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
truejetmom = numpy.sqrt(numpy.square(truejetpz) +numpy.square(truejetpz) + numpy.square(truejetpz))
truejetenergy = numpy.sqrt(numpy.square(truejetmom)+numpy.square(truejetmass))
truejetmomfraction = truejetmom / truejetenergy
truejetfeatures = numpy.stack([truejetpx, truejetpy, truejetpz, truejetpT, truejeteta, truejetmass, truejetenergy, truejetmomfraction, truejetmom], axis = -1)
bmomfraction = bhadspt/truejetpT
bfeatures = numpy.stack([bhadspx, bhadspy, bhadspz, bhadspt, bhadseta, bmomfraction], axis = -1)


# In[18]:


#Stack using numpy.where, mask using numpy.ma, normalise, input
inputs1 = numpy.concatenate([tracks[:,:,:3], matchedtracks], axis = 2)
trackpx = tracks[:,:,0]
trackpy = tracks[:,:,1]
trackpz = tracks[:,:,2]
trackp = numpy.where(trackpx!=-999, numpy.sqrt(numpy.square(trackpx) + numpy.square(trackpy) + numpy.square(trackpz)), -999)

trackpfrac = numpy.zeros(shape=numpy.shape(trackp))
trackpfrac = numpy.where(trackp!=-999, numpy.divide(trackp,truejetmom[:,numpy.newaxis]), -999)
#logtrackpfrac = (trackpfrac)
logtrackpfrac = numpy.where(trackpfrac!=-999, numpy.log(trackpfrac), -999)
logtrackpfrac = logtrackpfrac[:,:, numpy.newaxis]
unmasked_inputs = numpy.concatenate([inputs1, logtrackpfrac], axis = 2)

masked_inputs = numpy.ma.masked_values(unmasked_inputs, -999)
def scalert(array):
    scaled = numpy.zeros(shape = numpy.shape(array))
    for i in range(array.shape[-1]):
        scaled[:,:,i] = (array[:,:,i] - numpy.mean(array[:,:,i]))/numpy.std(array[:,:,i])
    return scaled

def scalerj(array):
    scaled = numpy.zeros(shape = numpy.shape(array))
    for i in range(array.shape[-1]):
        scaled[:,i] = (array[:,i] - numpy.mean(array[:,i]))/numpy.std(array[:,i])
    return scaled

def scalerb(array):
    scaled = numpy.zeros(shape = numpy.shape(array))
    scaled = (array - numpy.mean(array))/numpy.std(array)
    return scaled

def unscaler(array, fac):
    unscaled = array*numpy.std(fac) + numpy.mean(fac)
    return unscaled

trackinputs = scalert(masked_inputs)
jetinputs = scalerj(truejetfeatures)

targets = scalerj(bfeatures)


#Tracks = px, py, pz, pT, eta, phi, 5* IP, logTrack/Jet
#Sum Tracks = sum px, sum py, sum pz, sum pT, sqrt sum px^2, sqrt sum py^2, sqrt sum pz^2, sqrt sum pT^2, sqrt sum px^2/ sum px, sqrt sum py^2 / sum py, sqrt sum pz^2 / sum pz, sqrt sum pT^2 / sum pT,
#jets = px, py, pz, pT, eta, mass, energy, mom/energy, mom
#Bhads = px, py, pz, pT, Eta, Mass


# In[19]:


masked_tracks = numpy.ma.masked_values(masked_inputs, -999)
sumtracks = numpy.sum(masked_tracks, axis = 1)

sum_square_tracks = numpy.sqrt(numpy.sum(numpy.square(masked_tracks), axis = 1))
scaled_sum_square_tracks = sum_square_tracks/sumtracks


inputs2 = numpy.concatenate([sumtracks, sum_square_tracks, scaled_sum_square_tracks], axis = 1)
summed_inputs = scalerj(inputs2)
print(numpy.min((summed_inputs)))

testing = numpy.where(inputs2==None, -99999999999, inputs2)
print(numpy.min(testing))


# In[20]:


print(numpy.shape(masked_inputs[:,0]))
print(numpy.shape(summed_inputs))


# In[21]:


# Creating the training model

tracklayers = [256 , 128, 64, 32 ]
jetlayers = [ 32, 64 , 128, 256 ]

def buildModel(tlayers, jlayers, ntargets):

  inputs = layers.Input(shape=(None, tlayers[0]))
  inputs2 = layers.Input(shape = ( 9,))
  inputs3 = layers.Input(shape=(36,))

  outputs = inputs
  outputs = layers.Masking(mask_value=MASKVAL)(outputs)

  
  outputs2 = inputs2
  outputs3 = inputs3


  for nodes in tlayers[:-1]:
    outputs = layers.TimeDistributed(layers.Dense(nodes, activation='leaky_relu', kernel_initializer = 'he_normal'))(outputs)

  outputs = layers.TimeDistributed(layers.Dense(tlayers[-1], activation='softmax'))(outputs)
  outputs = Sum()(outputs)

  outputs = keras.layers.Concatenate()([outputs, outputs2, outputs3])

  for nodes in jlayers:
    outputs = layers.Dense(nodes, activation='leaky_relu', kernel_initializer='he_normal')(outputs)#, kernel_regularizer='l1_l2' 

  #outputs  = layers.Dense(ntargets + ntargets*(ntargets+1)//2)(outputs)
  outputs  = layers.Dense(ntargets)(outputs)


  
  return     keras.Model     ( inputs = [inputs, inputs2, inputs3]
    , outputs = outputs
    )


# In[22]:


# Creating the loss function
def LogNormal1D(true, meanscovs):
  ntargets = true.shape[1] #Number of variables predicting
  means = meanscovs[:,:ntargets] #First n targets are the means
  # ensure diagonal is positive
  logsigma = meanscovs[:,ntargets:2*ntargets]
  covs = meanscovs[:,2*ntargets:]

  # TODO
  # build matrix
  loss = 0
  for x in range(ntargets):
    loss = loss + ((means[:,x] - true[:,x])**2 / (2*keras.backend.exp(logsigma[:,x])**2)) + logsigma[:,x]
  return loss


# In[23]:


import tensorflow as tf

def LogNormalCov(true, meanscovs):
    ntargets = true.shape[1] #Number of variables predicting
    means = meanscovs[:,:ntargets] #First n targets are the means
    # ensure diagonal is positive
    logsigma = meanscovs[:,ntargets:2*ntargets]
    covs = meanscovs[:,2*ntargets:]

    # build covariance matrix
    cov_matrix = tf.eye(ntargets)  # initialize with identity matrix
    for i in range(ntargets):
        for j in range(i,ntargets):
            cov_matrix = tf.tensor_scatter_nd_update(cov_matrix, [[i,j]], covs[i*ntargets+j])
            cov_matrix = tf.tensor_scatter_nd_update(cov_matrix, [[j,i]], covs[i*ntargets+j])

    # calculate the log normal distribution
    diff = true - means
    deviation = tf.linalg.diag(tf.exp(logsigma))
    log_normal = -0.5 * tf.reduce_sum(tf.linalg.logdet(2 * numpy.pi * deviation) + tf.matmul(tf.matmul(diff,tf.linalg.inv(deviation)),diff),axis=1)

    # return the negative log-likelihood
    return -tf.reduce_mean(log_normal)


# In[24]:


model = buildModel([len(trackinputs[0,0,:])] + tracklayers, jetlayers, 6)


# In[25]:


model.summary()


# In[37]:


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer._decayed_lr(tf.float32)
    return lr

optimizer = keras.optimizers.Nadam(learning_rate = 0.02)
lr_metric = get_lr_metric(optimizer)


def custom_accuracy(y_true, y_pred, leeway=0.1):
    numtargets = y_true.shape[1]
    diff = keras.backend.abs(y_true - y_pred[:, :numtargets])
    within_leeway = keras.backend.less_equal(diff, leeway)
    acc = keras.backend.mean(within_leeway)
    return acc

def custom_MAE(y_true, y_pred):
    npreds = y_pred.shape[0]
    C_MAE = keras.backend.mean(keras.backend.abs(y_pred - y_true))
    return C_MAE




model.compile   ( loss = keras.losses.Huber(delta=1)
  , optimizer = optimizer
  , metrics = [lr_metric, custom_accuracy, keras.metrics.MeanAbsoluteError()]
  )


# In[27]:


# Splits the data into training and validation sets
X1_train, X1_valid, X2_train, X2_valid, X3_train, X3_valid, y_train, y_valid = train_test_split(trackinputs, jetinputs, summed_inputs, targets, train_size = 0.9, random_state=42)


# In[38]:


# Trains the data
history = model.fit([X1_train, X2_train, X3_train], y_train, validation_data = ([X1_valid, X2_valid, X3_valid], y_valid), callbacks = [reduce_lr, save_weights], batch_size=BATCHSIZE, epochs=1000)


# In[29]:


#Plots and saves the loss curve
history_df = numpy.log(pd.DataFrame(history.history))
LossFigure = history_df.loc[:, ['loss', 'val_loss']].plot().get_figure()
LossFigure.savefig('EverythingLossUnscaled.png')


# In[30]:


#Saves the model
#model.save('Model')
model.save_weights('Everything working.h5')


# In[31]:


# Uses the model to predict validation set
pred = model.predict([trackinputs, jetinputs, summed_inputs])


# In[32]:


fig, axes = plt.subplots(1,3, figsize = (18,6))
plt.suptitle("Correlations between True and Predicted")

axes[0].scatter(bfeatures[:,3], unscaler(pred[:,0],bhads[:,0]) , color='red', ec='black', alpha = 0.6, label = 'B-Had')
axes[2].scatter(truejetfeatures[:,3], unscaler(pred[:,0],bhads[:,0]), color='green', ec='black', alpha = 0.6, label = 'Jet')
axes[1].scatter(bhads[:,0],truejetfeatures[:,3], color = 'blue', ec = 'black', alpha = 0.6, label = 'True')

axes[0].set(xlabel='True pT (MeV)', ylabel='Predicted pT(MeV)', ylim=(-100,1e6))
axes[2].set(xlabel='True pT (MeV)', ylabel='Predicted pT (MeV)', ylim=(0,1e6))
axes[1].set(xlabel = 'Bhads pT (MeV)', ylabel = 'Jets pT (MeV)', ylim=(0,1e6))

axes[0].legend()
axes[1].legend()
axes[2].legend()

plt.tight_layout()
plt.show()
plt.savefig("Test")
plt.close()


# In[39]:


fig, axes = plt.subplots(2,3, figsize = (18,12))
plt.suptitle("Correlations between Targets and Outputs")

axes[0,0].scatter(targets[:,0], pred[:,0], color='red', ec='black', alpha = 0.6, label = 'B-Had')
axes[0,1].scatter(targets[:,1], pred[:,1], color = 'blue', ec='black', alpha = 0.6, label = 'B-Had')
axes[0,2].scatter(targets[:,2], pred[:,2], color = 'green', ec='black', alpha = 0.6, label = 'B-Had')
axes[1,0].scatter(targets[:,3], pred[:,3], color = 'yellow', ec='black', alpha = 0.6, label = 'B-Had')
axes[1,1].scatter(targets[:,4], pred[:,4], color = 'cyan', ec='black', alpha = 0.6, label = 'B-Had')
axes[1,2].scatter(targets[:,5], pred[:,5], color = 'magenta', ec='black', alpha = 0.6, label = 'B-Had')


#axes[0,0].scatter(scaled_bfeatures[:,0], scaledpred[:,0], color='yellow', ec='black', alpha = 0.6, label = 'Jet')
#axes[0,1].scatter(scaled_bfeatures[:,1], scaledpred[:,1], color = 'cyan', ec='black', alpha = 0.6, label = 'Jet')
#axes[0,2].scatter(scaled_bfeatures[:,2], scaledpred[:,2], color = 'magenta', ec='black', alpha = 0.6, label = 'Jet')
#axes[1,0].scatter(scaled_bfeatures[:,3], scaledpred[:,3], color = 'red', ec='black', alpha = 0.6, label = 'Jet')
#axes[1,1].scatter(scaled_bfeatures[:,4], scaledpred[:,4], color = 'blue', ec='black', alpha = 0.6, label = 'Jet')
#axes[1,2].scatter(scaled_bfeatures[:,5], scaledpred[:,5], color = 'green', ec='black', alpha = 0.6, label = 'Jet')

axes[0,0].set(xlabel='Scaled true py', ylabel='Predicted scaled px ',xlim = (-10,10), ylim = (-10,10))
axes[0,1].set(xlabel='Scaled true py', ylabel='Predicted scaled py ', xlim = (-10,10), ylim = (-10,10))
axes[0,2].set(xlabel='Scaled true pz', ylabel='Predicted scaled pz ', xlim = (-10,10), ylim = (-10,10))
axes[1,0].set(xlabel='Scaled true pT', ylabel='Predicted scaled pT ', xlim = (0,10), ylim = (0,10))
axes[1,1].set(xlabel='Scaled true eta', ylabel='Predicted scaled eta', xlim = (-5,5), ylim = (-5,5))
axes[1,2].set(xlabel='Scaled true momentum fraction', ylabel='Predicted scaled momentum fraction', xlim = (-3,10), ylim = (-3,10))

axes[0,0].legend()
axes[0,1].legend()
axes[0,2].legend()
axes[1,0].legend()
axes[1,1].legend()
axes[1,2].legend()

plt.tight_layout()
plt.show()
plt.savefig("Everything Correlations Unscaled")
plt.close()


# In[34]:


fig, axes = plt.subplots(2,3, figsize = (18,12))
plt.suptitle("Correlations between True and Predicted")

axes[0,0].scatter(bfeatures[:,0], unscaler(pred[:,0],bfeatures[:,0]), color='red', ec='black', alpha = 0.6, label = 'B-Had')
axes[0,1].scatter(bfeatures[:,1], unscaler(pred[:,1],bfeatures[:,1]), color = 'blue', ec='black', alpha = 0.6, label = 'B-Had')
axes[0,2].scatter(bfeatures[:,2], unscaler(pred[:,2],bfeatures[:,2]), color = 'green', ec='black', alpha = 0.6, label = 'B-Had')
axes[1,0].scatter(bfeatures[:,3], unscaler(pred[:,3],bfeatures[:,3]), color = 'yellow', ec='black', alpha = 0.6, label = 'B-Had')
axes[1,1].scatter(bfeatures[:,4], unscaler(pred[:,4],bfeatures[:,4]), color = 'cyan', ec='black', alpha = 0.6, label = 'B-Had')
axes[1,2].scatter(bfeatures[:,5], unscaler(pred[:,5],bfeatures[:,5]), color = 'magenta', ec='black', alpha = 0.6, label = 'B-Had')

'''axes[0,0].scatter(truejetfeatures[:,0], pred[:,0], color='yellow', ec='black', alpha = 0.6, label = 'Jet')
axes[0,1].scatter(truejetfeatures[:,1], pred[:,1], color = 'cyan', ec='black', alpha = 0.6, label = 'Jet')
axes[0,2].scatter(truejetfeatures[:,2], pred[:,2], color = 'magenta', ec='black', alpha = 0.6, label = 'Jet')
axes[1,0].scatter(truejetfeatures[:,3], pred[:,3], color = 'red', ec='black', alpha = 0.6, label = 'Jet')
axes[1,1].scatter(truejetfeatures[:,4], pred[:,4], color = 'blue', ec='black', alpha = 0.6, label = 'Jet')
axes[1,2].scatter(truejetfeatures[:,5], pred[:,5], color = 'green', ec='black', alpha = 0.6, label = 'Jet')
'''
axes[0,0].set(xlabel='True px (MeV)', ylabel='Predicted px (MeV)', xlim=(-500000,500000), ylim=(-500000,500000))
axes[0,1].set(xlabel='True py (MeV)', ylabel='Predicted py (MeV)', xlim=(-500000,500000), ylim=(-500000,500000))
axes[0,2].set(xlabel='True pz (MeV)', ylabel='Predicted pz (MeV)', xlim=(-2000000,2000000), ylim=(-2000000,2000000))
axes[1,0].set(xlabel='True pT (MeV)', ylabel='Predicted pT (MeV)', xlim=(0,500000), ylim=(0,500000))
axes[1,1].set(xlabel='True eta', ylabel='Predicted eta', xlim=(-5,5), ylim=(-5,5))
axes[1,2].set(xlabel='True Momentum Fraction', ylabel='Predicted Momentum Fraction', xlim = (0,3), ylim = (0,3))

axes[0,0].legend()
axes[0,1].legend()
axes[0,2].legend()
axes[1,0].legend()
axes[1,1].legend()
axes[1,2].legend()

plt.tight_layout()
plt.show()
plt.savefig("Everything Correlations 2 Unscaled")
plt.close()


# In[35]:


diffs = unscaler(pred[:,:6], bfeatures) - bfeatures
uncertainty = numpy.exp(unscaler(pred[:,6:12], bfeatures))
pull = diffs / uncertainty


# In[ ]:


fig = binneddensity(diffs[:,0], fixedbinning(-100000,100000, 100), xlabel = 'diffs')
fig


# In[ ]:


fig = binneddensity(pull[:,0], fixedbinning(-100000,100000, 100), xlabel = 'diffs')
fig


# In[ ]:


fig, axes = plt.subplots(2,3, figsize = (18,12))
plt.suptitle("Correlations between Jet and B-Hadron")
axes[0,0].scatter(bfeatures[:,0], scaler[:,0], color='red', ec='black', alpha = 0.6)
axes[0,0].set(xlabel='B-Had px (MeV)', ylabel='Jet px (MeV)', xlim=(-500000,500000), ylim=(-500000,500000))
axes[0,1].scatter(bfeatures[:,1], scaler[:,1], color = 'blue', ec='black', alpha = 0.6)
axes[0,1].set(xlabel='B-Had py (TeV)', ylabel='Jet py (MeV)', xlim=(-500000,500000), ylim=(-500000,500000))
axes[0,2].scatter(bfeatures[:,2], scaler[:,2], color = 'green', ec='black', alpha = 0.6)
axes[0,2].set(xlabel='B-Had pz (TeV)', ylabel='Jet pz (MeV)', xlim=(-2000000,2000000), ylim=(-2000000,2000000))
axes[1,0].scatter(bfeatures[:,3], scaler[:,3], color = 'yellow', ec='black', alpha = 0.6)
axes[1,0].set(xlabel='B-Had pT (TeV)', ylabel='Jet pT (MeV)', xlim=(0,500000), ylim=(0,500000))
axes[1,1].scatter(bfeatures[:,4], scaler[:,4], color = 'cyan', ec='black', alpha = 0.6)
axes[1,1].set(xlabel='B-Had eta', ylabel='Jet eta')
axes[1,2].scatter(bfeatures[:,5], scaler[:,5], color = 'magenta', ec='black', alpha = 0.6)
axes[1,2].set(xlabel='B-Had Mass (MeV)', ylabel='Jet Mass (MeV)')
plt.tight_layout()
plt.show()
plt.close()


# In[ ]:


bfeatures = bfeatures*plotscaler
scaler = scaler*plotscaler
pred = pred*plotscaler


# In[ ]:


err = (pred[:,6:12])
bdiffs = pred[:,:6] - bfeatures[:,:6]
bPull = bdiffs / err
jdiffs = pred[:,:6] - truejetfeatures[:,:6]
jPull = jdiffs / err
fig = binneddensity(bdiffs[:,0], fixedbinning(-100000,100000,100), xlabel = 'pT Difference (MeV)')
fig


# In[ ]:


fig = binneddensity(bPull[:,0], fixedbinning(-5,5, 100), xlabel = ' Mass Pull (MeV)')
fig


# In[ ]:


c = 3e8
predpx = numpy.square(pred[:,0])
predpy = numpy.square(pred[:,1])
predpz = numpy.square(pred[:,2])
predpt = numpy.square(pred[:,3])
predp = (predpz + predpx + predpz)
invMass = pred[:,5] - predp
predmass = numpy.sqrt(invMass)
print(numpy.where(invMass < 0)[0])
print(pred[1,5])
print(predp[1])


#predmomtot = numpy.sqrt(pred[:,0]**2+pred[:,1]**2+pred[:,2]**2)
#predmass = numpy.sqrt(pred[:,5]**2-predmomtot**2)


# In[ ]:


xs = numpy.linspace(0, len(bfeatures), len(bfeatures))
print(numpy.shape(xs))
fig, axes = plt.subplots(1,3, figsize = (16,8))
axes[0].scatter(xs, pred[:,0], label = 'Pred', color = 'red', ec = 'black', alpha = 0.6)
axes[0].set_ylim(5000,6000)
axes[0].legend()
axes[1].scatter(xs, bmass, label = 'True', color='green', ec = 'black', alpha = 0.6)
axes[1].set_ylim(5000,6000)
axes[1].legend()
axes[2].scatter(xs, bmass - pred[:,0], label = 'True - Pred', color = 'blue', ec = 'black', alpha = 0.6)
axes[2].legend()
plt.show()
plt.close()


# In[ ]:


fig = binneddensity(bmass, fixedbinning(5200,5800,100))
fig


# In[ ]:


fig, axes = plt.subplots(2,3, figsize = (18,12))
plt.suptitle("Correlations between True Jet and Predicted")
axes[0,0].scatter(scaler[:,0], pred[:,0], color='red', ec='black', alpha = 0.6)
axes[0,0].set(xlabel='True Jet px (MeV)', ylabel='Predicted px (MeV)', xlim=(-500000,500000), ylim=(-500000,500000))
axes[0,1].scatter(scaler[:,1], pred[:,1], color = 'blue', ec='black', alpha = 0.6)
axes[0,1].set(xlabel='True Jet py (MeV)', ylabel='Predicted py (MeV)', xlim=(-500000,500000), ylim=(-500000,500000))
axes[0,2].scatter(scaler[:,2], pred[:,2], color = 'green', ec='black', alpha = 0.6)
axes[0,2].set(xlabel='True Jet pz (MeV)', ylabel='Predicted pz (MeV)', xlim=(-2000000,2000000), ylim=(-2000000,2000000))
axes[1,0].scatter(scaler[:,3], pred[:,3], color = 'yellow', ec='black', alpha = 0.6)
axes[1,0].set(xlabel='True Jet pT (MeV)', ylabel='Predicted pT (MeV)', xlim=(0,500000), ylim=(0,500000))
axes[1,1].scatter(scaler[:,4], pred[:,4], color = 'cyan', ec='black', alpha = 0.6)
axes[1,1].set(xlabel='True Jet eta', ylabel='Predicted eta')
axes[1,2].scatter(scaler[:,5], pred[:,5], color = 'magenta', ec='black', alpha = 0.6)
axes[1,2].set(xlabel='True Jet Mass (MeV)', ylabel='Predicted Mass (MeV)')
plt.tight_layout()
plt.show()
plt.close()


# In[ ]:


fig = binneddensity(truejetmass, fixedbinning(0,100000, 100), xlabel = 'True Jet Mass Distribution (MeV)')
fig


# In[ ]:


from sklearn import metrics
print(numpy.shape(bfeatures))
print(numpy.shape(bfeatures[:1,5]))
print(bfeatures[:2,5])
mi_scores = metrics.mutual_info_score(bfeatures[:8,5], matchedtracks[0,0,:8])
print(mi_scores)


# In[ ]:


unmasked_inputs = numpy.concatenate([inputs1, logtrackpfrac], axis = 2)
inputs1 = numpy.concatenate([tracks[:,:,:3], matchedtracks], axis = 2)
inputs2 = numpy.stack([sumtracks, sum_square_tracks, scaled_sum_square_tracks], axis = -1)
truejetfeatures = numpy.stack([truejetpx, truejetpy, truejetpz, truejetpT, truejeteta, truejetmass, truejetenergy, truejetmomfraction, truejetmom], axis = -1)
bfeatures = numpy.stack([bhadspx, bhadspy, bhadspz, bhadspt, bhadseta, bmass], axis = -1)


# In[ ]:


print(numpy.max(bmass))
bmass_class = keras.utils.to_categorical(bmass, num_classes = 10000)
print(bmass_class)
print(numpy.shape(bmass_class))
print(bmass[:2])
print(bmass_class[0:10])


# In[ ]:


mi_scores = mutual_info_classif(numpy.mean(matchedtracks, axis = 1), bmass_class)


# In[ ]:


#Tracks = px, py, pz, pT, eta, phi, 5* IP, logTrack/Jet
#jets = px, py, pz, pT, eta, mass, energy, mom/energy, mom
#Sum Tracks = sum px, sum py, sum pz, sum pT, sqrt sum px^2, sqrt sum py^2, sqrt sum pz^2, sqrt sum pT^2, sqrt sum px^2/ sum px, sqrt sum py^2 / sum py, sqrt sum pz^2 / sum pz, sqrt sum pT^2 / sum pT,
#Bhads = px, py, pz, pT, Eta, Mass

#Sum of etas are good for predicting pz!


# In[48]:


import pandas
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
numpy.set_printoptions(formatter={'float_kind':'{:f}'.format})
mi_scores = mutual_info_regression((trackinputs[:,0]), bfeatures[:,0])
print(mi_scores)
mi_scores = mutual_info_regression(jetinputs, bfeatures[:,0])
print(mi_scores)
mi_scores = mutual_info_regression(summed_inputs, bfeatures[:,0])
print(mi_scores)


# In[41]:


mi_scores = mutual_info_regression((trackinputs[:,0]), bfeatures[:,1])
print(mi_scores)
mi_scores = mutual_info_regression(jetinputs, bfeatures[:,1])
print(mi_scores)
mi_scores = mutual_info_regression(summed_inputs, bfeatures[:,1])
print(mi_scores)


# In[55]:


mi_scores = mutual_info_regression((trackinputs[:,0]), bfeatures[:,2])
print(mi_scores)
mi_scores = mutual_info_regression(jetinputs, bfeatures[:,2])
print(mi_scores)
mi_scores = mutual_info_regression(summed_inputs, bfeatures[:,2])
print(mi_scores)


# In[43]:


mi_scores = mutual_info_regression((trackinputs[:,0]), bfeatures[:,3])
print(mi_scores)
mi_scores = mutual_info_regression(jetinputs, bfeatures[:,3])
print(mi_scores)
mi_scores = mutual_info_regression(summed_inputs, bfeatures[:,3])
print(mi_scores)


# In[44]:


mi_scores = mutual_info_regression((trackinputs[:,0]), bfeatures[:,4])
print(mi_scores)
mi_scores = mutual_info_regression(jetinputs, bfeatures[:,4])
print(mi_scores)
mi_scores = mutual_info_regression(summed_inputs, bfeatures[:,4])
print(mi_scores)


# In[45]:


mi_scores = mutual_info_regression((trackinputs[:,0]), bfeatures[:,5])
print(mi_scores)
mi_scores = mutual_info_regression(jetinputs, bfeatures[:,5])
print(mi_scores)
mi_scores = mutual_info_regression(summed_inputs, bfeatures[:,5])
print(mi_scores)


# In[53]:


mi_scores = mutual_info_regression(bfeatures, bfeatures[:,0])
print(mi_scores)
mi_scores = mutual_info_regression(bfeatures, bfeatures[:,1])
print(mi_scores)
mi_scores = mutual_info_regression(bfeatures, bfeatures[:,2])
print(mi_scores)
mi_scores = mutual_info_regression(bfeatures, bfeatures[:,3])
print(mi_scores)
mi_scores = mutual_info_regression(bfeatures, bfeatures[:,4])
print(mi_scores)
mi_scores = mutual_info_regression(bfeatures, bfeatures[:,5])
print(mi_scores)


# In[52]:


mi_scores = mutual_info_regression(targets, targets[:,0])
print(mi_scores)
mi_scores = mutual_info_regression(targets, targets[:,1])
print(mi_scores)
mi_scores = mutual_info_regression(targets, targets[:,2])
print(mi_scores)
mi_scores = mutual_info_regression(targets, targets[:,3])
print(mi_scores)
mi_scores = mutual_info_regression(targets, targets[:,4])
print(mi_scores)
mi_scores = mutual_info_regression(targets, targets[:,5])
print(mi_scores)


# In[54]:


mi_scores = mutual_info_regression(targets, bfeatures[:,0])
print(mi_scores)
mi_scores = mutual_info_regression(targets, bfeatures[:,1])
print(mi_scores)
mi_scores = mutual_info_regression(targets, bfeatures[:,2])
print(mi_scores)
mi_scores = mutual_info_regression(targets, bfeatures[:,3])
print(mi_scores)
mi_scores = mutual_info_regression(targets, bfeatures[:,4])
print(mi_scores)
mi_scores = mutual_info_regression(targets, bfeatures[:,5])
print(mi_scores)

