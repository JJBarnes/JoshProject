#!/usr/bin/env python
# coding: utf-8

# In[3]:


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
from tensorflow.keras.constraints import max_norm


# In[4]:


#Set hyperparameters
MASKVAL = -999
MAXTRACKS = 32
BATCHSIZE = 64
EPOCHS = 1000
MAXEVENTS = 99999999999999999
# VALFACTOR = 10
LR = 1e-2


# In[5]:


# Define Callbacks

# Define Early Stopping
early_stopping = callbacks.EarlyStopping(min_delta=0.001, patience = 20, restore_best_weights = True, monitor = 'val_loss')

#Define ReducedLR
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=10, min_lr=0)


#Define learning schedule

def scheduler(epoch, lr):
  if epoch == (EPOCHS/2):
    return LR
  else:
    return lr

lrscheduler = callbacks.LearningRateScheduler(scheduler)



# In[6]:


#Open the root file
tree = uproot.open("hffrag.root:CharmAnalysis")


# In[7]:


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
  , "AnalysisAntiKt4TruthJets_ghostB_pdgId"
  ]

# true b-hadron information
# these b-hadrons are inside the truth jets
bhadfeatures =    [ "AnalysisAntiKt4TruthJets_ghostB_pt"
   , "AnalysisAntiKt4TruthJets_ghostB_eta"
   , "AnalysisAntiKt4TruthJets_ghostB_phi"
   , "AnalysisAntiKt4TruthJets_ghostB_m"
   ]
  
svertex = ["TruthParticles_Selected_LxyT"]

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


# In[8]:


# Read in the requested branches from the file
features = tree.arrays(jetfeatures + trackfeatures + svertex, entry_stop=MAXEVENTS)


# In[9]:


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


# In[10]:


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


# In[11]:


# Pads inputs with nans up to the given maxsize
def pad(xs, maxsize):
  #Find 'none' values in array and replace with MASKVAL (= fill_none)
  ys =     awkward.fill_none   ( awkward.pad_none(xs, maxsize, axis=1, clip=True) #Adding 'none' values to make sure it is correct size
  , MASKVAL
  )[:,:maxsize]

  return awkward.to_regular(ys, axis=1)


# In[12]:


def flatten1(xs, maxsize=-1):
  ys = {}
  for field in xs.fields:
    zs = xs[field]
    if maxsize > 0:
      zs = pad(zs, maxsize)
    ys[field] = zs

  return awkward.zip(ys)


# In[13]:


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


def binneddensitysub(xs, binning, ax, label=None, xlabel=None, ylabel="binned probability density"):
  #fig = figure.Figure(figsize=(8, 8))
  #plt = fig.add_subplot(111)

  ys , yerrs = hist(xs, binning, normalized=True)

  # determine the central value of each histogram bin
  # as well as the width of each bin
  # this assumes a fixed bin size.
  xs = (binning[1:]+binning[:-1]) / 2.0
  xerrs = ((binning[1:]-binning[:-1]) / 2.0)

  ax.errorbar     ( xs
    , ys
    , xerr=xerrs
    , yerr=yerrs
    , label=label
    , linewidth=0
    , elinewidth=2
    )

  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)

  #return fig

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


# In[158]:


events =   features[awkward.sum(features["AnalysisAntiKt4TruthJets_pt"] > 25000, axis=1) > 0]

jets1 = events[jetfeatures][:,0] #First jet
tracks = events[trackfeatures]


# In[15]:


matchedtracks = tracks[matchTracks(jets1, tracks)] 
matchedtracks = flatten1(matchedtracks, MAXTRACKS) #Turn into regular np array


# In[16]:


bjets = awkward.sum(jets1["AnalysisAntiKt4TruthJets_ghostB_pt"] > 5000, axis=1) > 0 #Find b hadron jets with certain momentum
jets2 = jets1[bjets] #Jets identified as b jets are only jets considered
bhadspt= jets2["AnalysisAntiKt4TruthJets_ghostB_pt"][:,0] #np Stack here - Each sub array contains all the features of the jet (axis -1)
bhadseta = jets2["AnalysisAntiKt4TruthJets_ghostB_eta"][:, 0]
bhadsphi = jets2["AnalysisAntiKt4TruthJets_ghostB_phi"][:,0]
bmass = jets2["AnalysisAntiKt4TruthJets_ghostB_m"][:,0].to_numpy()
#secondaryVertex = jets2["TruthParticles_SelectedLxyT"].to_numpy()
bhadid = jets2["AnalysisAntiKt4TruthJets_ghostB_pdgId"][:,0].to_numpy()
matchedtracks = matchedtracks[bjets]
#svertex = jets2["TruthParticles_Selected_LxyT"]


# In[17]:


jets3 = structured_to_unstructured(jets2[jetfeatures[:3]]) #number of features
matchedtracks = structured_to_unstructured(matchedtracks)
matchedtracks = matchedtracks.to_numpy()


# In[18]:


unique_id, idcounts = numpy.unique(bhadid, return_counts=True)
print(numpy.stack([unique_id, idcounts], axis = -1 ))
other = awkward.from_numpy(numpy.array([-5332,-5232,-5132, -5122, -541, 541, 553, 5122, 5132, 5232, 5332]))
classTargets = numpy.zeros(len(bhadid))
for x in range(len(bhadid)):
    if bhadid[x] in other:
        classTargets[x] = 0
    elif numpy.abs(bhadid[x]) == 531:
        classTargets[x] = 1
    elif numpy.abs(bhadid[x]) == 521:
        classTargets[x] = 2
    elif numpy.abs(bhadid[x]) == 511:
        classTargets[x] = 3
    else:
        classTargets[x] = bhadid[x]
        print([x])

classTargets = classTargets.astype(int)


# In[19]:


unique_ID_values, IDcounts = numpy.unique((classTargets), return_counts=True)
# Get the indices of the unique masses in the sorted unique masses array
idindices = numpy.searchsorted(unique_ID_values, classTargets)
classTargets = idindices


# In[20]:


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


# In[21]:


bhadspx = bhadscart[:,0] 
bhadspy = bhadscart[:,1]
bhadspz = bhadscart[:,2]
bhadsmom = numpy.sqrt((bhadspx**2+bhadspy**2+bhadspz**2))
bhadenergy = numpy.sqrt(bhadsmom**2+bmass**2)
truejetpx = jets4[:,0]
truejetpy = jets4[:,1]
truejetpz = jets4[:,2]
truejetpT = jets5[:,0]
truejeteta = jets5[:,1]
truejetmass = jets2["AnalysisAntiKt4TruthJets_m"].to_numpy()
truejetmom = numpy.sqrt(numpy.square(truejetpz) +numpy.square(truejetpz) + numpy.square(truejetpz))
truejetenergy = numpy.sqrt(numpy.square(truejetmom)+numpy.square(truejetmass))
truejetmomfraction = truejetpT / truejetenergy
truejet_transversemass = numpy.sqrt(numpy.square(truejetenergy) - numpy.square(truejetpz))
truejetfeatures = numpy.stack([truejetpx, truejetpy, truejetpz, truejetpT, truejeteta, truejetmass, truejetenergy, truejetmomfraction, truejetmom], axis = -1)
bmomfraction = bhadspt/truejetpT
bdotproduct = numpy.ones(len(bhadscart))
for i in range(len(bdotproduct)):
    bdotproduct[i] = numpy.dot(bhadscart[i], jets4[i])
jetnorm = numpy.linalg.norm(jets4, axis=1)
bprojection = bdotproduct/(jetnorm**2)
secondary_vertex = events[svertex]
secondary_vertex = flatten1(secondary_vertex, MAXTRACKS)[:,0]
secondary_vertex = secondary_vertex[bjets].to_numpy()
bfeatures = numpy.stack([bhadspx*10**-6, bhadspy*10**-6, bhadspz*10**-6, bhadspt*10**-6, bhadsmom*10**-6, bhadenergy*10**-6, bmomfraction, bprojection], axis = -1)


# In[22]:


#Stack using numpy.where, mask using numpy.ma, normalise, input
inputs1 = numpy.concatenate([tracks[:,:,:3], matchedtracks], axis = 2)
trackpx = tracks[:,:,0]
trackpy = tracks[:,:,1]
trackpz = tracks[:,:,2]
trackp = numpy.where(trackpx!=-999, numpy.sqrt(numpy.square(trackpx) + numpy.square(trackpy) + numpy.square(trackpz)), -999)

trackpfrac = numpy.zeros(shape=numpy.shape(trackp))
#trackpxfrac = numpy.where(trackp!=-999, numpy.divide(inputs1[:,:,0], truejetfeatures[:,0, numpy.newaxis]), -999)
#trackpyfrac = numpy.where(trackp!=-999, numpy.divide(inputs1[:,:,1], truejetfeatures[:,1, numpy.newaxis]), -999)
trackpzfrac = numpy.where(trackp!=-999, numpy.divide(inputs1[:,:,2], truejetfeatures[:,2, numpy.newaxis]), -999)
trackptfrac = numpy.where(trackp!=-999, numpy.divide(inputs1[:,:,3], truejetfeatures[:,3, numpy.newaxis]), -999)
trackpfrac = numpy.where(trackp!=-999, numpy.divide(trackp, truejetmom[:,numpy.newaxis]),-999)
trackpfrac = numpy.stack([trackptfrac, trackpfrac], axis = -1)
logtrackpfrac = numpy.where(trackpfrac!=-999, numpy.log(trackpfrac), -999)

unmasked_inputs = numpy.concatenate([inputs1, logtrackpfrac], axis = -1)

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

targets = bfeatures

TargetShape = (len(numpy.shape(targets)))
if TargetShape == 1:
    numtargets = 1
else:
    numtargets = len(targets[-1])


print(numpy.shape(unique_ID_values))
numtargetsClass = len(unique_ID_values)

#Tracks = px, py, pz, pT, eta, phi, 5* IP, logTrack/Jet
#Sum Tracks = sum px, sum py, sum pz, sum pT, sqrt sum px^2, sqrt sum py^2, sqrt sum pz^2, sqrt sum pT^2, sqrt sum px^2/ sum px, sqrt sum py^2 / sum py, sqrt sum pz^2 / sum pz, sqrt sum pT^2 / sum pT,
#jets = px, py, pz, pT, eta, mass, energy, mom/energy, mom
#Bhads = px, py, pz, pT, Eta, Mass


# In[23]:


masked_tracks = numpy.ma.masked_values(masked_inputs, -999)
sumtracks = numpy.sum(masked_tracks, axis = 1)

sum_square_tracks = numpy.sqrt(numpy.sum(numpy.square(masked_tracks), axis = 1))
scaled_sum_square_tracks = sum_square_tracks/sumtracks


inputs2 = numpy.concatenate([sumtracks, sum_square_tracks, scaled_sum_square_tracks], axis = 1)
summed_inputs = scalerj(inputs2)
print(numpy.min((summed_inputs)))

testing = numpy.where(inputs2==None, -99999999999, inputs2)
print(numpy.min(testing))


# In[24]:


print("Track Input Shape = ", str(numpy.shape(trackinputs)))
print("Jet Input Shape = ", str(numpy.shape(jetinputs)))
print("SumTrack Input Shape = ", str(numpy.shape(summed_inputs)))


# In[25]:


#Define MC Dropout
class MCDropout(keras.layers.GaussianDropout):
	def call(self, inputs):
		return super().call(inputs, training = True)


# In[26]:


# Creating a Deep ResNet training model

# Creating the training model

tracklayers = [256 , 128, 64, 32 ]
jetlayers = [ 32, 64 , 128, 256 ]

def buildModel(tlayers, jlayers, ntargets, ntargetsclass):

  inputs = layers.Input(shape=(None, tlayers[0]))
  inputs2 = layers.Input(shape = ( 9,))
  inputs3 = layers.Input(shape=(39,))

  outputs = inputs
  outputs = layers.Masking(mask_value=MASKVAL)(outputs)
  #outputs = (layers.GaussianNoise(1))(outputs)

  
  outputs2 = inputs2
  #outputs2 = layers.GaussianNoise(1)(outputs2)
  outputs3 = inputs3
  #outputs3 = layers.GaussianNoise(1)(outputs2)

  
  outputs = layers.TimeDistributed(MCDropout(0.3))(outputs)
  outputs = layers.TimeDistributed(layers.Dense(64, activation='gelu', kernel_initializer = 'he_normal', kernel_constraint = max_norm(1.), kernel_regularizer='l1_l2'))(outputs)
  outputsa = layers.TimeDistributed(layers.LayerNormalization())(outputs)

  outputs = layers.TimeDistributed(MCDropout(0.3))(outputsa)
  outputs = layers.TimeDistributed(layers.Dense(64, activation='gelu', kernel_initializer = 'he_normal', kernel_constraint = max_norm(1.), kernel_regularizer='l1_l2'))(outputs)
  outputsb = layers.TimeDistributed(layers.LayerNormalization())(outputs)

  outputs = layers.Add()([outputsa, outputsb])

  outputs = layers.TimeDistributed(MCDropout(0.3))(outputs)
  outputs = layers.TimeDistributed(layers.Dense(64, activation='gelu', kernel_initializer = 'he_normal', kernel_constraint = max_norm(1.), kernel_regularizer='l1_l2'))(outputs)
  outputsc = layers.TimeDistributed(layers.LayerNormalization())(outputs)

  outputs = layers.Add()([outputsa, outputsb, outputsc])

  outputs = layers.TimeDistributed(MCDropout(0.3))(outputs)
  outputs = layers.TimeDistributed(layers.Dense(64, activation='gelu', kernel_initializer = 'he_normal', kernel_constraint = max_norm(1.), kernel_regularizer='l1_l2'))(outputs)
  outputsd = layers.TimeDistributed(layers.LayerNormalization())(outputs)

  outputs = layers.Add()([outputsa, outputsb, outputsc, outputsd])

  outputs = layers.TimeDistributed(layers.Dense(tlayers[-1], activation='softmax'))(outputs)
  outputs = Sum()(outputs)

  outputs = keras.layers.Concatenate()([outputs, outputs2, outputs3])

  outputs = MCDropout(0.3)(outputs)
  outputs = layers.Dense(64, activation='gelu', kernel_initializer='he_normal', kernel_constraint = max_norm(1.), kernel_regularizer='l1_l2')(outputs)#, kernel_regularizer='l1_l2' 
  outputsa = layers.LayerNormalization()(outputs)

  outputs = MCDropout(0.3)(outputsa)
  outputs = layers.Dense(64, activation='gelu', kernel_initializer='he_normal', kernel_constraint = max_norm(1.), kernel_regularizer='l1_l2')(outputsa)#, kernel_regularizer='l1_l2' 
  outputsb = layers.LayerNormalization()(outputs)

  outputs = layers.Add()([outputsa, outputsb])

  outputs = MCDropout(0.3)(outputs)
  outputs = layers.Dense(64, activation='gelu', kernel_initializer='he_normal', kernel_constraint = max_norm(1.), kernel_regularizer='l1_l2')(outputs)#, kernel_regularizer='l1_l2' 
  outputsc = layers.LayerNormalization()(outputs)

  outputs = layers.Add()([outputsa, outputsb, outputsc])

  outputs = MCDropout(0.3)(outputs)
  outputs = layers.Dense(64, activation='gelu', kernel_initializer='he_normal', kernel_constraint = max_norm(1.), kernel_regularizer='l1_l2')(outputs)#, kernel_regularizer='l1_l2' 
  outputsd = layers.LayerNormalization()(outputs)

  outputs = layers.Add()([outputsa, outputsb, outputsc, outputsd])

  outputsReg  = layers.Dense(ntargets + ntargets*(ntargets+1)//2, name = 'RegOuts')(outputs)
  #outputs  = layers.Dense(ntargets)(outputs)
  #outsputs = layers.Dense(3*ntargets)(outputs)

  classOuts = layers.Dense((ntargetsclass), name = 'ClassOuts', activation='softmax')(outputs)
  
  return     keras.Model     ( inputs = [inputs, inputs2, inputs3]
    , outputs = [outputsReg, classOuts]
    )


# In[27]:


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


# In[28]:


model = buildModel([len(trackinputs[0,0,:])] + tracklayers, jetlayers, numtargets, numtargetsClass)


# In[29]:


model.summary()


# In[30]:


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer._decayed_lr(tf.float32)
    return lr

optimizer = keras.optimizers.Nadam(learning_rate = 0.002)
lr_metric = get_lr_metric(optimizer)


def custom_accuracy(y_true, y_pred, leeway=0.1):
    numtargets = y_true.shape[1]
    diff = keras.backend.abs(y_true - y_pred[:, :numtargets])
    within_leeway = keras.backend.less_equal(diff, leeway)
    acc = keras.backend.mean(within_leeway)
    return acc

def custom_MAE(y_true, y_pred):
    numtargets = y_true.shape[1]
    C_MAE = keras.backend.mean(keras.backend.abs(y_pred[:,:numtargets] - y_true))
    return C_MAE



#model.compile(loss = tf.keras.losses.sparse_categorical_crossentropy, optimizer = optimizer, metrics = 'accuracy')

model.compile   ( loss = {'RegOuts': LogNormal1D, 'ClassOuts': tf.keras.losses.sparse_categorical_crossentropy}
  , optimizer = optimizer
  , metrics = {'RegOuts': [lr_metric, custom_accuracy, custom_MAE], 'ClassOuts': [keras.metrics.sparse_categorical_accuracy]}
  )


# In[31]:


from keras.layers import Layer

class LoopBack(Layer):
    def __init__(self, layer, **kwargs):
        self.layer = layer
        super(LoopBack, self).__init__(**kwargs)

    def call(self, inputs):
        return self.layer(inputs)

#inputs = Input(shape=(input_shape,))
#x1 = Dense(units)(inputs)
#x2 = Dense(units)(x1)

#x1_loop = LoopBack(x1)(x2)
#x2_loop = LoopBack(x2)(x1_loop)


# In[32]:


# Splits the data into training and validation sets
X1_train, X1_valid, X2_train, X2_valid, X3_train, X3_valid, y_train, y_valid, y2_train, y2_valid = train_test_split(trackinputs, jetinputs, summed_inputs, targets, classTargets, train_size = 0.5, random_state=42)


# In[33]:


'''df = pd.DataFrame(y2_train_OS)
df2 = pd.DataFrame(y2_train)
df3 = pd.DataFrame(y2_train_US)
unique_df, dfCounts = numpy.unique(y2_train_OS, return_counts=True)
indices = numpy.searchsorted(unique_df, y2_train_OS)
iddf = pd.DataFrame(indices)
unique_df2, df2Counts = numpy.unique(y2_train, return_counts=True)
indices2 = numpy.searchsorted(unique_df2, y2_train)
iddf2 = pd.DataFrame(indices2)
unique_df3, df3Counts = numpy.unique(y2_train_US, return_counts=True)
indices3 = numpy.searchsorted(unique_df3, y2_train_US)
iddf3 = pd.DataFrame(indices3)

#fig, axes = plt.subplots(2,2, figsize=(16,16))
#axes[0,0] = df.groupby(0).size().plot(kind='pie', y = "0", label = "Type", autopct='%1.1f%%', subplots=True, ax = [0,0])
#axes[1,1] = df2.groupby(0).size().plot(kind='pie', y = "0", label = "Type", autopct='%1.1f%%', subplots=True, ax = [1,1])

fig, axes = plt.subplots(1,3, figsize = (36,12))

#df.groupby(0).size().plot.pie(subplots=True, ax = axes[1],autopct='%1.1f%%', label = "Oversampled Sample", pctdistance=0.8, textprops={'fontsize':12})
#df2.groupby(0).size().plot.pie(subplots=True, ax=axes[0], autopct='%1.1f%%', label = "Data Sample", pctdistance = 0.8, textprops={'fontsize':12})
#df3.groupby(0).size().plot.pie(subplots=True, ax=axes[2], autopct='%1.1f%%', label = "Rescaled Sample", pctdistance = 0.8, textprops={'fontsize':12})

iddf.groupby(0).size().plot.bar(subplots=True, ax = axes[1], label = "Oversampled Sample")
iddf2.groupby(0).size().plot.bar(subplots=True, ax=axes[0], label = "Data Sample")
iddf3.groupby(0).size().plot.bar(subplots=True, ax=axes[2], label = "Rescaled Sample")
#axes[0].legend(loc='upper left', bbox_to_anchor=(1,1), prop={'size': 14})
axes[0].set(xlabel = "PDG Class")
axes[1].set(xlabel = "PDG Class")
axes[2].set(xlabel = "PDG Class")
plt.show()
'''


# In[34]:


numpy.save("X1_train", X1_train)
numpy.save("X2_train", X2_train)
numpy.save("X3_train", X3_train)
numpy.save("y_train", y_train)
numpy.save("y2_train", y2_train)
numpy.save("X1_valid", X1_valid)
numpy.save("X2_valid", X2_valid)
numpy.save("X3_valid", X3_valid)
numpy.save("y_valid", y_valid)
numpy.save("y2_valid", y2_valid)


# In[35]:


# Trains the data
history = model.fit([X1_train, X2_train, X3_train], [y_train, y2_train], validation_data = ([X1_valid, X2_valid, X3_valid], [y_valid, y2_valid]), callbacks = [reduce_lr], batch_size=BATCHSIZE, epochs=500, use_multiprocessing=True)
#class_weights={'classification_output': class_weights_dict}
# class_weight=class_weights


# In[36]:


#Plots and saves the loss curve
history_df = (pd.DataFrame(history.history))
LossFigure = history_df.loc[:, ['loss', 'val_loss']].plot().get_figure()
LossFigure.savefig('LossOfAll.png')


# In[37]:


#Saves the model
#model.save('Model')
model.save_weights('PDGID Weights.h5')


# In[38]:


# Uses the model to predict validation set
preds = numpy.stack([model([X1_valid, X2_valid, X3_valid])[0] for sample in range(100)])
pred = preds.mean(axis=0)
preds2 = numpy.stack([model([X1_valid, X2_valid, X3_valid])[1] for sample in range(100)])
pred2 = preds2.mean(axis=0)

numpy.save('All Reg Preds Val', pred)
numpy.save('Class Preds Val', pred2)

# In[39]:
preds = numpy.stack([model([X1_train, X2_train, X3_train])[0] for sample in range(100)])
pred = preds.mean(axis=0)
preds2 = numpy.stack([model([X1_train, X2_train, X3_train])[1] for sample in range(100)])
pred2 = preds2.mean(axis=0)

numpy.save('All Reg Preds Train', pred)
numpy.save('Class Preds Train', pred2)

pred3arg = pred2[:,:, numpy.newaxis].argmax(axis = 1) # Find Index of maximum


# In[40]:


from sklearn.metrics import confusion_matrix

fig, axes = plt.subplots(figsize = (8,8))
conf_mat = confusion_matrix(y2_valid, pred3arg)
axes.matshow(conf_mat, cmap = plt.cm.hot)
axes.set(xlabel = 'Predicted Mass Category', ylabel = 'True Mass Category')
axes.xaxis.set_label_position('top') 
plt.show()


# In[45]:


pred3arg = numpy.reshape(pred3arg, len(pred3arg))
print(numpy.shape(pred3arg))
print(numpy.shape(y2_valid))


# In[47]:


from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(18):
    fpr[i], tpr[i], _ = roc_curve(y2_valid, pred3arg, pos_label=1)
    roc_auc[i] = auc(fpr[i], tpr[i])

fig, axes = plt.subplots(figsize = (8,8))
colors = cycle(['blue', 'red', 'green'])
for i, color in zip(range(18), colors):
    axes.plot(fpr[i], tpr[i], color=color,
             label='Class {0} (Area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
axes.plot([0, 1], [0, 1], 'k--')
axes.set(xlim = ([-0.05, 1.0]), ylim = ([0.0, 1.05]), xlabel = ('False Positive Rate'), ylabel = ('True Positive Rate'), title = ('ROC Curve for PDGID'))
plt.legend(loc="upper right", bbox_to_anchor=(1.75,1.1))

plt.show()


# In[ ]:


# In[ ]:


from matplotlib import colors
from matplotlib.ticker import PercentFormatter


# In[ ]:


pred[:,:6] = pred[:,:6]* 10**6
pred[:,9:15] = numpy.exp(pred[:,9:17])
pred[:,9:15] = pred[:,9:15]*10**6


# In[ ]:


print(numpy.shape(pred))
diffs = y_train[:,:9] - pred[:,:9]
err = pred[:,9:18]
pull = diffs/err


# In[ ]:


fig, axes = plt.subplots(2,2, figsize = (12,12))
plt.suptitle("Correlations between True and Predicted")
lims = (-150000, 150000)

axes[0,0].hist2d(y_train[:,0], pred[:,0], bins = [500,500], norm = colors.LogNorm(), cmap ="YlOrRd", label = 'B-Had')
binneddensitysub(pred[:,0], fixedbinning(lims[0], lims[1], nbins = 100), ax = axes[0,1], xlabel= 'Predicted px (MeV)')
binneddensitysub(diffs[:,0], fixedbinning(lims[0], lims[1], nbins = 100), ax = axes[1,0], xlabel= 'px Difference')
binneddensitysub(pull[:,0], fixedbinning(-5,5, nbins = 100), ax = axes[1,1], xlabel= 'px Pull ')


axes[0,0].set(xlabel='True px (MeV)', ylabel='Predicted px (MeV)', xlim = lims, ylim = lims)

axes[0,0].plot(lims,lims)

axes[0,0].legend()

plt.tight_layout()
plt.show()
plt.close()
print("True Mean and Standard Deviation = " + str(numpy.mean(bmomfraction)) + u" \u00B1 " + str(numpy.std(bmomfraction)))
print("Prediction Mean and Uncertainty = " + str(numpy.mean(pred[:,0])) + u" \u00B1 " + str(numpy.std(pred[:,0])))
print("Differences Mean and Standard Deviation = " + str(numpy.mean(diffs)) + u" \u00B1 " + str(numpy.std(diffs)))
print("Pull Mean and Standard Deviation = " + str(numpy.mean(pull)) + u" \u00B1 " + str(numpy.std(pull)))


# In[ ]:


fig, axes = plt.subplots(2,2, figsize = (12,12))
plt.suptitle("Correlations between True and Predicted")
lims = (-150000, 150000)

axes[0,0].hist2d(y_train[:,1], pred[:,1], bins = [500,500], norm = colors.LogNorm(), cmap ="YlOrRd", label = 'B-Had')
binneddensitysub(pred[:,1], fixedbinning(lims[0], lims[1], nbins = 100), ax = axes[0,1], xlabel= 'Predicted py (MeV)')
binneddensitysub(diffs[:,1], fixedbinning(lims[0], lims[1], nbins = 100), ax = axes[1,0], xlabel= 'py Difference')
binneddensitysub(pull[:,1], fixedbinning(-5,5, nbins = 100), ax = axes[1,1], xlabel= 'py Pull ')


axes[0,0].set(xlabel='True py (MeV)', ylabel='Predicted py (MeV)', xlim = lims, ylim = lims)

axes[0,0].plot(lims,lims)

axes[0,0].legend()

plt.tight_layout()
plt.show()
plt.close()
print("True Mean and Standard Deviation = " + str(numpy.mean(bmomfraction)) + u" \u00B1 " + str(numpy.std(bmomfraction)))
print("Prediction Mean and Uncertainty = " + str(numpy.mean(pred[:,0])) + u" \u00B1 " + str(numpy.std(pred[:,0])))
print("Differences Mean and Standard Deviation = " + str(numpy.mean(diffs)) + u" \u00B1 " + str(numpy.std(diffs)))
print("Pull Mean and Standard Deviation = " + str(numpy.mean(pull)) + u" \u00B1 " + str(numpy.std(pull)))


# In[ ]:


fig, axes = plt.subplots(2,2, figsize = (12,12))
plt.suptitle("Correlations between True and Predicted")
lims = (-250000, 250000)

axes[0,0].hist2d(y_train[:,2], pred[:,2], bins = [500,500], norm = colors.LogNorm(), cmap ="YlOrRd", label = 'B-Had')
binneddensitysub(pred[:,2], fixedbinning(lims[0], lims[1], nbins = 100), ax = axes[0,1], xlabel= 'Predicted pz (MeV)')
binneddensitysub(diffs[:,2], fixedbinning(lims[0], lims[1], nbins = 100), ax = axes[1,0], xlabel= 'pz Difference')
binneddensitysub(pull[:,2], fixedbinning(-5,5, nbins = 100), ax = axes[1,1], xlabel= 'pz Pull ')


axes[0,0].set(xlabel='True pz (MeV)', ylabel='Predicted pz (MeV)', xlim = lims, ylim = lims)

axes[0,0].plot(lims,lims)

axes[0,0].legend()

plt.tight_layout()
plt.show()
plt.close()
print("True Mean and Standard Deviation = " + str(numpy.mean(bmomfraction)) + u" \u00B1 " + str(numpy.std(bmomfraction)))
print("Prediction Mean and Uncertainty = " + str(numpy.mean(pred[:,0])) + u" \u00B1 " + str(numpy.std(pred[:,0])))
print("Differences Mean and Standard Deviation = " + str(numpy.mean(diffs)) + u" \u00B1 " + str(numpy.std(diffs)))
print("Pull Mean and Standard Deviation = " + str(numpy.mean(pull)) + u" \u00B1 " + str(numpy.std(pull)))


# In[ ]:


fig, axes = plt.subplots(2,2, figsize = (12,12))
plt.suptitle("Correlations between True and Predicted")
lims = (0, 150000)

axes[0,0].hist2d(y_train[:,3], pred[:,3], bins = [500,500], norm = colors.LogNorm(), cmap ="YlOrRd", label = 'B-Had')
binneddensitysub(pred[:,3], fixedbinning(lims[0], lims[1], nbins = 100), ax = axes[0,1], xlabel= 'Predicted pt (MeV)')
binneddensitysub(diffs[:,3], fixedbinning(-lims[1], lims[1], nbins = 100), ax = axes[1,0], xlabel= 'pT Difference')
binneddensitysub(pull[:,3], fixedbinning(-5,5, nbins = 100), ax = axes[1,1], xlabel= 'pT Pull ')


axes[0,0].set(xlabel='True pT (MeV)', ylabel='Predicted pT (MeV)', xlim = lims, ylim = lims)

axes[0,0].plot(lims,lims)

axes[0,0].legend()

plt.tight_layout()
plt.show()
plt.close()
print("True Mean and Standard Deviation = " + str(numpy.mean(bmomfraction)) + u" \u00B1 " + str(numpy.std(bmomfraction)))
print("Prediction Mean and Uncertainty = " + str(numpy.mean(pred[:,0])) + u" \u00B1 " + str(numpy.std(pred[:,0])))
print("Differences Mean and Standard Deviation = " + str(numpy.mean(diffs)) + u" \u00B1 " + str(numpy.std(diffs)))
print("Pull Mean and Standard Deviation = " + str(numpy.mean(pull)) + u" \u00B1 " + str(numpy.std(pull)))


# In[ ]:


fig, axes = plt.subplots(2,2, figsize = (12,12))
plt.suptitle("Correlations between True and Predicted")
lims = (0, 250000)

axes[0,0].hist2d(y_train[:,4], pred[:,4], bins = [500,500], norm = colors.LogNorm(), cmap ="YlOrRd", label = 'B-Had')
binneddensitysub(pred[:,4], fixedbinning(lims[0], lims[1], nbins = 100), ax = axes[0,1], xlabel= 'Predicted Momentum (MeV)')
binneddensitysub(diffs[:,4], fixedbinning(-lims[1], lims[1], nbins = 100), ax = axes[1,0], xlabel= 'Momentum Difference')
binneddensitysub(pull[:,4], fixedbinning(-5,5, nbins = 100), ax = axes[1,1], xlabel= 'Momentum Pull ')


axes[0,0].set(xlabel='True Momentum (MeV)', ylabel='Predicted Momentum (MeV)', xlim = lims, ylim = lims)

axes[0,0].plot(lims,lims)

axes[0,0].legend()

plt.tight_layout()
plt.show()
plt.close()
print("True Mean and Standard Deviation = " + str(numpy.mean(bmomfraction)) + u" \u00B1 " + str(numpy.std(bmomfraction)))
print("Prediction Mean and Uncertainty = " + str(numpy.mean(pred[:,0])) + u" \u00B1 " + str(numpy.std(pred[:,0])))
print("Differences Mean and Standard Deviation = " + str(numpy.mean(diffs)) + u" \u00B1 " + str(numpy.std(diffs)))
print("Pull Mean and Standard Deviation = " + str(numpy.mean(pull)) + u" \u00B1 " + str(numpy.std(pull)))


# In[ ]:


fig, axes = plt.subplots(2,2, figsize = (12,12))
plt.suptitle("Correlations between True and Predicted")
lims = (0, 250000)

axes[0,0].hist2d(y_train[:,5], pred[:,5], bins = [500,500], norm = colors.LogNorm(), cmap ="YlOrRd", label = 'B-Had')
binneddensitysub(pred[:,5], fixedbinning(lims[0], lims[1], nbins = 100), ax = axes[0,1], xlabel= 'Predicted Energy (MeV)')
binneddensitysub(diffs[:,5], fixedbinning(-lims[1], lims[1], nbins = 100), ax = axes[1,0], xlabel= 'Energy Difference')
binneddensitysub(pull[:,5], fixedbinning(-5,5, nbins = 100), ax = axes[1,1], xlabel= 'Energy Pull ')


axes[0,0].set(xlabel='True Energy (MeV)', ylabel='Predicted Energy (MeV)', xlim = lims, ylim = lims)

axes[0,0].plot(lims,lims)

axes[0,0].legend()

plt.tight_layout()
plt.show()
plt.close()
print("True Mean and Standard Deviation = " + str(numpy.mean(bmomfraction)) + u" \u00B1 " + str(numpy.std(bmomfraction)))
print("Prediction Mean and Uncertainty = " + str(numpy.mean(pred[:,0])) + u" \u00B1 " + str(numpy.std(pred[:,0])))
print("Differences Mean and Standard Deviation = " + str(numpy.mean(diffs)) + u" \u00B1 " + str(numpy.std(diffs)))
print("Pull Mean and Standard Deviation = " + str(numpy.mean(pull)) + u" \u00B1 " + str(numpy.std(pull)))


# In[ ]:


fig, axes = plt.subplots(2,2, figsize = (12,12))
plt.suptitle("Correlations between True and Predicted")
lims = (0, 5)

axes[0,0].hist2d(y_train[:,6], pred[:,6], bins = [500,500], norm = colors.LogNorm(), cmap ="YlOrRd", label = 'B-Had')
binneddensitysub(pred[:,6], fixedbinning(lims[0], lims[1], nbins = 100), ax = axes[0,1], xlabel= 'Predicted Momentum Fraction')
binneddensitysub(diffs[:,6], fixedbinning(-lims[1], lims[1], nbins = 100), ax = axes[1,0], xlabel= 'Momentum Fraction Difference')
binneddensitysub(pull[:,6], fixedbinning(-5,5, nbins = 100), ax = axes[1,1], xlabel= 'Momentum Fraction Pull ')


axes[0,0].set(xlabel='True Momentum Fraction', ylabel='Predicted Momentum Fraction', xlim = lims, ylim = lims)

axes[0,0].plot(lims,lims)

axes[0,0].legend()

plt.tight_layout()
plt.show()
plt.close()
print("True Mean and Standard Deviation = " + str(numpy.mean(bmomfraction)) + u" \u00B1 " + str(numpy.std(bmomfraction)))
print("Prediction Mean and Uncertainty = " + str(numpy.mean(pred[:,0])) + u" \u00B1 " + str(numpy.std(pred[:,0])))
print("Differences Mean and Standard Deviation = " + str(numpy.mean(diffs)) + u" \u00B1 " + str(numpy.std(diffs)))
print("Pull Mean and Standard Deviation = " + str(numpy.mean(pull)) + u" \u00B1 " + str(numpy.std(pull)))


# In[ ]:


fig, axes = plt.subplots(2,2, figsize = (12,12))
plt.suptitle("Correlations between True and Predicted")
lims = (0, 5)

axes[0,0].hist2d(y_train[:,7], pred[:,7], bins = [500,500], norm = colors.LogNorm(), cmap ="YlOrRd", label = 'B-Had')
binneddensitysub(pred[:,7], fixedbinning(lims[0], lims[1], nbins = 100), ax = axes[0,1], xlabel= 'Predicted Momentum Projection')
binneddensitysub(diffs[:,7], fixedbinning(-lims[1], lims[1], nbins = 100), ax = axes[1,0], xlabel= 'Momentum Projection Difference')
binneddensitysub(pull[:,7], fixedbinning(-5,5, nbins = 100), ax = axes[1,1], xlabel= 'Momentum Projection Pull ')


axes[0,0].set(xlabel='True Momentum Projection', ylabel='Predicted Momentum Projection', xlim = lims, ylim = lims)

axes[0,0].plot(lims,lims)

axes[0,0].legend()

plt.tight_layout()
plt.show()
plt.close()
print("True Mean and Standard Deviation = " + str(numpy.mean(bmomfraction)) + u" \u00B1 " + str(numpy.std(bmomfraction)))
print("Prediction Mean and Uncertainty = " + str(numpy.mean(pred[:,0])) + u" \u00B1 " + str(numpy.std(pred[:,0])))
print("Differences Mean and Standard Deviation = " + str(numpy.mean(diffs)) + u" \u00B1 " + str(numpy.std(diffs)))
print("Pull Mean and Standard Deviation = " + str(numpy.mean(pull)) + u" \u00B1 " + str(numpy.std(pull)))


# In[ ]:


fig, axes = plt.subplots(2,2, figsize = (12,12))
plt.suptitle("Correlations between True and Predicted")
lims = (0, 5)

axes[0,0].hist2d(y_train[:,8], pred[:,8], bins = [500,500], norm = colors.LogNorm(), cmap ="YlOrRd", label = 'B-Had')
binneddensitysub(pred[:,8], fixedbinning(lims[0], lims[1], nbins = 100), ax = axes[0,1], xlabel= 'Predicted SV')
binneddensitysub(diffs[:,8], fixedbinning(-lims[1], lims[1], nbins = 100), ax = axes[1,0], xlabel= 'SV Difference')
binneddensitysub(pull[:,8], fixedbinning(-5,5, nbins = 100), ax = axes[1,1], xlabel= 'SV Pull ')


axes[0,0].set(xlabel='True SV', ylabel='Predicted SV', xlim = lims, ylim = lims)

axes[0,0].plot(lims,lims)

axes[0,0].legend()

plt.tight_layout()
plt.show()
plt.close()
print("True Mean and Standard Deviation = " + str(numpy.mean(bmomfraction)) + u" \u00B1 " + str(numpy.std(bmomfraction)))
print("Prediction Mean and Uncertainty = " + str(numpy.mean(pred[:,0])) + u" \u00B1 " + str(numpy.std(pred[:,0])))
print("Differences Mean and Standard Deviation = " + str(numpy.mean(diffs)) + u" \u00B1 " + str(numpy.std(diffs)))
print("Pull Mean and Standard Deviation = " + str(numpy.mean(pull)) + u" \u00B1 " + str(numpy.std(pull)))

