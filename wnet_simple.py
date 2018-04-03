import theano.tensor as T
import numpy as np
import random
import theano
import lasagne
import _pickle as pickle
import matplotlib.pyplot as plt
import sys
sys.setrecursionlimit(100000)
import pandas as pd

# Helper function to compute receptive field
def compute_receptive_field(nStacks, dilation, filterWidth):
    if filterWidth > 1: 
        receptiveField = nStacks*(dilation*filterWidth) - (nStacks-1)
    else:
        receptiveField = 1
    return receptiveField

#############################################################
#     DEFINE THE LAYERS
#############################################################

# A PReLU activation
class PReLU(object):
    def __init__(self, X):
        iAlpha = 0
        self.alpha = theano.shared(value = iAlpha, borrow = True)
        self.result = T.switch(X < 0, self.alpha * X, X)
        self.params = [self.alpha]

class DilatedConv1D(object):
    #   Task:
    #       creates a dilated convolutional layer
    #   Args:
    #       rng: a random number generator used to initialize weights
    #       dilation: The dilation factor for each layer
    #       filterWidth: The samples that are included in each convolution, after dilating
    #       nFilters: How many filters to learn for the dilated convolution
    #       nChannels: Channels in input data
    #       batchSize: Size of training set used per iteration
    #       learningRate: Learning rate 
    def __init__(self, input, rng, dilation, filterHeight, filterWidth, nFilters, nChannels, applyBias, activation = 'linear'):
        self.input = input
        self.dilation = dilation
        self.filterWidth = filterWidth
        self.nFilters = nFilters
        self.nChannels = nChannels
        self.filterHeight = filterHeight
        # Initialization of filter for each layer of size (nFilters, nChannels in input, filterHeight, filterWidth)
        if activation == 'tanh':
            iFilters = rng.uniform(-np.sqrt(6)/np.sqrt(2*filterWidth*nFilters), np.sqrt(6)/np.sqrt(2*filterWidth*nFilters), [nFilters, nChannels, filterHeight, filterWidth]).astype(theano.config.floatX)
        elif activation == 'sigmoid':
            iFilters = rng.uniform(-4*np.sqrt(6)/np.sqrt(2*filterWidth*nFilters), 4*np.sqrt(6)/np.sqrt(2*filterWidth*nFilters), [nFilters, nChannels, filterHeight, filterWidth]).astype(theano.config.floatX)
        elif activation == 'relu':
            iFilters = rng.normal(0, np.sqrt(2)/np.sqrt(filterWidth*nFilters), [nFilters, nChannels, filterHeight, filterWidth]).astype(theano.config.floatX)
        else:
            iFilters = rng.uniform(-np.sqrt(6)/np.sqrt(filterWidth*nFilters), np.sqrt(6)/np.sqrt(filterWidth*nFilters), [nFilters, nChannels, filterHeight, filterWidth]).astype(theano.config.floatX)
        self.filters = theano.shared(value = iFilters, borrow = True)
        # Convolve input feature map with filters
        result = T.nnet.conv2d(self.input, self.filters, border_mode = 'valid', filter_dilation = (1,self.dilation))
        # Check for bias
        if applyBias:
            # Define bias
            iBias = np.zeros([nFilters], dtype = theano.config.floatX)
            self.bias = theano.shared(value = iBias, borrow = True)
            # Store parameters of this layer
            self.params = [self.filters, self.bias]
            # Apply bias
            result += self.bias[None, :, None, None]
        else:
            self.params = [self.filters]
        self.output = result
        
#############################################################
#                     BUILD THE MODEL
#############################################################
        
class cWaveNet(object):
    def __init__(self, input, nCond, rng, nStacks, dilations, nFilters, filterWidth, nChannels):
        recField = compute_receptive_field(nStacks, dilations[-1], filterWidth)
        # Input shape is (nBatches = 1, nChannels, 1, N)
        self.result = input
        self.params = []
        self.L2 = 0
        
        # Define applyBias and activation used in DilatedConv1D layer
        applyBias = True
        activation = 'relu'
        
        for s in range(nStacks):
            for i in range(len(dilations)):
                print('Stack ' + str(s) + ' Layer ' + str(i))
                
                # Input will have nChannels channels, output will have nFilters channels
                originalX = self.result
                output = DilatedConv1D(self.result, rng, dilations[i], 1, filterWidth, nFilters, nChannels, applyBias, activation)
                self.params += output.params
                # Use regularization, here L2
                self.L2 += 0.5*T.sum(T.sqr(output.params[0]))
                outputPrelu = PReLU(output.output)
                self.result = outputPrelu.result
                
                # Add a residual connection from originalX to output
                output = DilatedConv1D(originalX, rng, 1, 1, 1, nFilters, nChannels, applyBias, activation)
                self.params += output.params
                self.L2 += 0.5*T.sum(T.sqr(output.params[0]))
                originalX = output.output
                nChannels = nFilters
                    
                if filterWidth == 1:
                    self.result += originalX[:,:,:,:]
                else:
                    self.result += originalX[:,:,:,dilations[i]:]
        
        # End with a 1x1 convolution, to reduce nChannels back to nCond
        print('Final layer')
        output = DilatedConv1D(self.result, rng, 1, 1, 1, nCond, nChannels, applyBias)
        self.resultFull = output.output
        self.params += output.params
        self.L2 += 0.5*T.sum(T.sqr(output.params[0]))
        self.result = self.resultFull[:,:,:,0:-1]
                    
#############################################################
#     TRAIN AND EVALUATE THE MODEL
#############################################################

# CONDITIONAL WAVENET
# Takes as input the dataset with nCond inputs of size [N]
# Outputs the forecast 
def trainCWN(dataset, nCond, rng, nStacks, dilations, nFilters, filterWidth, nChannels, regRate, trainIter, learningRate, nTest):
    # Define the inputs and the functions
    recField = compute_receptive_field(nStacks, dilations[-1], filterWidth)
    input = T.tensor4('input')
    testInput = input
    model = cWaveNet(testInput, nCond, rng, nStacks, dilations, nFilters, filterWidth, nChannels)
    # The cost function, e.g. absolute error 
    cost = T.sum(T.abs_(testInput[:,:,:,recField:]-model.result)) + regRate*model.L2

    print('Building the gradients')
    grads = T.grad(cost, model.params)
    updates = lasagne.updates.adam(grads, model.params, learning_rate=learningRate)

    # Define the test and train functions
    train_fn = theano.function(
        [input],
        cost,
        updates=updates,
        on_unused_input='warn'
    )    

    sample_fn = theano.function(
        [input],
        model.resultFull, 
        updates = updates,
        on_unused_input = 'warn'
    )
    
    # Define the data: split the datasets into a train and test set
    datasetTrain = dataset[:,:,:,:dataset.shape[3]-nTest]
    N = datasetTrain.shape[3]

    # Modify the input data to fit the model by appending recField zeros, in order to not have any look-ahead bias, i.e. the 'causal convolution'
    trainData = np.append(np.zeros([dataset.shape[0],dataset.shape[1],dataset.shape[2],recField]),    datasetTrain, axis = 3)

    print('Training!')  
    totalIters = 0
    costs = []
    for j in range(0,trainIter):
        cost = train_fn(trainData[:,:,:,:])
        if j%1000==0:
            print(totalIters, cost)
        totalIters += 1
        costs.append(cost)
    
    print('Sampling!')
    testData = np.append(np.zeros([dataset.shape[0],dataset.shape[1],dataset.shape[2],recField]),    dataset, axis = 3) # Shape is 1, nCond, 1, N+nTest+recField
    # One day ahead sampling
    output = sample_fn(testData)[:,:,:,:-1]
    
    return N, costs, output

#############################################################
#                          GET DATA
#############################################################

def getDataLorenz(stepCnt, dt = 0.01, initx = 0., inity = 1., initz = 1., s = 5, r = 20, b = 2):
    xs = np.zeros(stepCnt+1)
    ys = np.zeros(stepCnt+1)
    zs = np.zeros(stepCnt+1)
    xs[0], ys[0], zs[0] = (initx, inity, initz)
    for i in range(stepCnt):
        x_dot = s*(ys[i] - xs[i])
        y_dot = r*xs[i] - ys[i] - xs[i]*zs[i]
        z_dot = xs[i]*ys[i] - b*zs[i]
        xs[i+1] = xs[i] + (x_dot * dt)
        ys[i+1] = ys[i] + (y_dot * dt)
        zs[i+1] = zs[i] + (z_dot * dt)
    # Rescale data to [-0.5, 0.5] range
    xs = (xs - np.amax(xs))/(np.amax(xs)-np.amin(xs)) + 0.5
    ys = (ys - np.amax(ys))/(np.amax(ys)-np.amin(ys)) + 0.5
    zs = (zs - np.amax(zs))/(np.amax(zs)-np.amin(zs)) + 0.5
    return xs, ys, zs
    
#############################################################
#                     ERROR METRICS
#############################################################

# Usually we are interested in error over test set, i.e. dataset[N:N+nTest] - output[0,0,0,N:N+nTest]

def RMSE(dataset, output):
    nTest = dataset.shape[0]
    error = dataset - output
    MSE = np.sum(error**2)/nTest
    RMSE = np.sqrt(MSE)
    return RMSE

#############################################################
#                          RESULTS
#############################################################

#Define the data; here we use the Lorenz curve
datax, datay, dataz = getDataLorenz(1500)
L = datax.shape[0]
data = np.concatenate((datax.reshape(1,L), datay.reshape(1,L), dataz.reshape(1,L)), axis = 0)
nCond = 3
# Reshape the data into a 4d tensor
dataset = data.reshape(1,nCond,1,L)

# The dilations array defines the number of layers and corresponding dilations, which we always set to be powers of 2, 2^0, 2^1,...
dilations = [1,2,4]
# nStacks is always set to 1 for now
nStacks = 1

# Note that each condition is defined as a channel in the input!
nChannels = nCond
nFilters = nCond

#Other parameters
filterWidth = 2
trainIter = 20000
learningRate = 0.001
regRate = 0.1

recField = compute_receptive_field(nStacks, dilations[-1], filterWidth)

nTest = 500

RMSE_con = np.zeros([1, nCond])

# Conditional results
rng = np.random.RandomState(1234) # Set random state
N1, costs1, out = trainCWN(dataset, nCond, rng, nStacks, dilations, nFilters, filterWidth, nChannels, regRate, trainIter, learningRate, nTest)
for j in range(0,nCond):
    RMSE_con[0,j] += [RMSE(dataset[0,j,0,N1:N1+nTest], out[0,j,0,N1:N1+nTest])]
    

print(RMSE_con)

nPlot = 0

f, (ax1, ax2, ax3) = plt.subplots(3)
ax1.plot(datax[-nPlot:], 'g')
ax1.plot(out[0,0,0,-nPlot:],'r')
ax2.plot(datay[-nPlot:], 'g')
ax2.plot(out[0,1,0,-nPlot:],'r')
ax3.plot(dataz[-nPlot:], 'g')
ax3.plot(out[0,2,0,-nPlot:],'r')
plt.show()
