import numpy as np
from numpy.random import randint
# Lasagne (& friends) imports
import theano
from lasagne.objectives import aggregate, binary_crossentropy
from lasagne.layers import (InputLayer, DropoutLayer, DenseLayer, FeaturePoolLayer, 
                            Conv1DLayer, MaxPool1DLayer, ReshapeLayer, NINLayer, 
                            DimshuffleLayer, ConcatLayer, SliceLayer, flatten)
from lasagne.updates import nesterov_momentum
from theano.tensor.nnet import sigmoid
from theano import tensor as T
from lasagne.nonlinearities import very_leaky_rectify
# Local imports
from nnet import AdjustVariable, EarlyStopping, LayerFactory
from index_batch_iterator import IndexNeuralNet, IndexTrainSplit
from grasp_batch_iterator import GraspBatchIterator

# This is the total number of EEG channels. The raw data is N x N_CHANNELS 
N_CHANNELS = 32
# This is the number of events we need to predict. The training labels are N x N_EVENTS
N_EVENTS = 6

    
def create_net(train_source, test_source, ch=None, 
               sample_size=4096, n_features=6,
               filter0_width=5, filter1_num=32, filter2_num=64,
               dense=512, maxout=4, 
               batch_size=32, max_epochs=100, rate=0.04, patience=20, stop_window=1): 
    """create a nolearn.lasagne based neural net: STF7
    
    Standard Arguments:
    train_source -- `Source` for training data, see grasp.py
    test_source -- `Source` for the test data.   
    ch -- which channels to use. None means use all channels.
    sample_size -- the number of time points to feed into the net
    n_features -- N_CHANNELS is reduced to n_features in the first layer of the net
    batch_size -- mini batch size to use for training
    max_epochs -- maximum number of epochs to train for
    rate -- the learning rate
    patience -- number of epochs to wait before quiting if no improvement
    
    Net Specific Arguments:
    filter0_width -- width of the, initial, spatio-temporal filter
    filter1_num -- number of filters used in first convolutional layer
    filter2_num -- number of filters used in second convolutional layer
    
    The net specific arguments were varied to provide diversity. However, the default
    arguments gave the best private leaderboard results of those tried.    
    """
    
    learning_rate = theano.shared(np.float32(rate))
    momentum = theano.shared(np.float32(0.9))

    batch_iter_train = GraspBatchIterator(train_source, ch, sample_size, batch_size=batch_size)
    batch_iter_test  = GraspBatchIterator(test_source, ch, sample_size, batch_size=batch_size)
    
    LF = LayerFactory()
                
    layers = [
        LF(InputLayer, shape=(None, batch_iter_train.n_channels, sample_size)), 
        #
        # This reduces the number of spatial from N_CHANNELS to n_features while at
        # same time performing (linear) time-domain filtering. This is the 
        # spatial-temporal filtering stage for which this particular net was named.
        LF(Conv1DLayer, num_filters=n_features, filter_size=filter0_width, 
           nonlinearity=None, pad="same"),
        #
        # This downsamples the data by a factor of 16 in the time dimension. The code below
        # is equivalent to performing convolution with size-16 filter with a stride of
        # 16. This allows the net to fit an appropriate downsampling scheme to the problem.
        # The convoluted version below turns out to be much faster than performing this 
        # using a convolutional layer. 
        LF(DimshuffleLayer, pattern=(0,2,1)),
        LF(ReshapeLayer, shape=([0], -1, 16*n_features)),
        LF(DimshuffleLayer, pattern=(0,2,1)),
        LF(NINLayer, num_units=n_features, nonlinearity=None),
        #
        # The next 4 layers comprise a fairly standard convolutional pipeline with
        # alternating convolutional and overlapping maxpooling layers. 
        #
        # very_leaky_rectify == max(0.3*a, a)
        LF(Conv1DLayer, num_filters=filter1_num, filter_size=7, pad="same",
            nonlinearity=very_leaky_rectify, untie_biases=False),
        #
        LF(MaxPool1DLayer, pool_size=3, stride=2, ignore_border=False),
        #
        # This combination of no nonlinearity and a feature pool layer (which by default
        # uses `max` to pool the result) is referred to as maxout. This is different 
        # from maxpooling only in that maxout is pooled across features, while maxpooling
        # is pooled across, in this case, the time dimension.
        LF(Conv1DLayer, num_filters=filter2_num, filter_size=5, pad="same",
            nonlinearity=None, untie_biases=True, layer_name="last_conv"),
        LF(FeaturePoolLayer, pool_size=2),
        #
        # The drastic maxpooling here helps reduce overfitting quite a bit at the
        # expense of making the location of features quite fuzzy.
        LF(MaxPool1DLayer, pool_size=12, stride=8),  
        LF(flatten, layer_name="all_time"),
        #
        # We reach back up to before the previous maxpooling layer to grab
        # the most recent 8 time-points worth of data before it gets fuzzed.
        # This allows us better precision for these last, important points
        # without increasing overfitting much.
        LF(SliceLayer, incoming="last_conv", indices=slice(-8,None,None)),
        LF(FeaturePoolLayer, pool_size=2),
        LF(flatten, layer_name="recent"),
        #
        # Merge the (accurate) recent time points with fuzzy time points.
        LF(ConcatLayer, incomings=["all_time", "recent"]),
        #   
        # These last six layers are a nearly completely standard dense
        # net section. The only out of the ordinary feature is the use
        # of maxout. In this case, the maxout is 4 wide. Maxout gives 
        # the net somewhat more expressiveness for the same size layer 
        # and can be helpful to reduce overfitting. 
        LF(DropoutLayer, p=0.5),  
        #
        LF(DenseLayer, nonlinearity=None, num_units=maxout*dense),
        LF(FeaturePoolLayer, pool_size=maxout),
        #
        LF(DropoutLayer, p=0.5),
        #
        LF(DenseLayer, nonlinearity=None, num_units=maxout*dense),
        LF(FeaturePoolLayer, pool_size=maxout),
        #
        LF(DropoutLayer, p=0.5),
        #
        LF(DenseLayer, layer_name="output", num_units=N_EVENTS, nonlinearity=sigmoid)
    ]


    def binary_crossentropy_loss(x,t):
        return aggregate(binary_crossentropy(x, t))
    
    # We setup the learning rate to decay with a half life of 20 epochs.
    on_epoch_finished = [AdjustVariable(learning_rate, target=0, half_life=20)]
    on_training_finished = []
    on_training_started = []
    if patience:
        # If `patience` is set, then training will stop if the validation doesn't 
        # improve in patience epochs. At that point the best weights up to that point
        # (according to validation error) will be loaded into the net. 
        # Similarly when the net finishes without running out of `patience`, 
        # the best weights till that time will be loaded.
        earlyStopper = EarlyStopping(patience=patience, window=stop_window)
        on_epoch_finished.append(earlyStopper)
        on_training_finished.append(earlyStopper.finished)
        on_training_started.append(earlyStopper.started)
    
        
    nnet =  IndexNeuralNet(
        y_tensor_type = T.matrix,
        train_split = IndexTrainSplit(),
        layers = layers,
        batch_iterator_train = batch_iter_train,
        batch_iterator_test = batch_iter_test,
        max_epochs = max_epochs,
        verbose=1,
        update = nesterov_momentum, 
        update_learning_rate = learning_rate,
        update_momentum = 0.9,
        objective_loss_function = binary_crossentropy_loss,
        regression = True,
        on_epoch_finished = on_epoch_finished,
        on_training_started = on_training_started,
        on_training_finished = on_training_finished,
        **LF.kwargs
        )


    return nnet
    
 