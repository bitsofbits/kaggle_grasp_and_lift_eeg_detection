import numpy as np
from nolearn.lasagne import BatchIterator


# This is the total number of EEG channels. The raw data is N x N_CHANNELS 
N_CHANNELS = 32
# This is the number of events we need to predict. The training labels are N x N_EVENTS
N_EVENTS = 6


class GraspBatchIterator(BatchIterator):
    """supply blocks of time points and labels for training based on their INDEX"""

    def __init__(self, source, channels, sample_size, *args, **kwargs):
        super(GraspBatchIterator, self).__init__(*args, **kwargs)
        if channels is None:
            channels = np.arange(N_CHANNELS)
        self.channels = channels
        self.n_channels = len(channels)
        self.sample_size = sample_size
        self.set_source(source)
        self.Xbuf = np.zeros([self.batch_size, self.n_channels, sample_size], np.float32) 
        self.Ybuf = np.zeros([self.batch_size, N_EVENTS], np.float32) 
    
    def set_source(self, source):
        self.source = source
        if source is None:
            self.augmented = None
        else:
            self.augmented = self.augment(source)
    
    def augment(self, source):
        # This function pads the data at the beginning be `sample_size-1` points.
        # That way when we examine the point located at zero, we still have access
        # to SAMPLE_SIZE points at or before time zero.
        offset = self.sample_size-1
        augmented = np.zeros([self.n_channels, len(source)+offset], dtype=np.float32)
        augmented[:,offset:] = source.data.transpose()[self.channels]
        # This line is actually the result of a cut and paste error. This is what we
        # used for the contest however, so I'm leaving it in place. Also, fixing it
        # results in slightly better scores, weirdly enough! The commented out line
        # below is what was intended.
        augmented[:,:offset] = augmented[:,offset][:,None][::-1]
#         augmented[:,:offset] = augmented[:,offset][:,None]
        return augmented
    
    def transform(self, X_indices, y_indices):
        X_indices, y_indices = super(GraspBatchIterator, self).transform(X_indices, y_indices)
        [count] = X_indices.shape
        # Use preallocated space
        X = self.Xbuf[:count]
        Y = self.Ybuf[:count]
        for i, ndx in enumerate(X_indices):
            if ndx == -1:
                # If an index of -1 is passed in we pick a random index and train on that.
                ndx = np.random.randint(len(self.source.events))
            # Since we've padded the data by sample_size-1 points, the following starts
            # at nxd-(sample_size-1) and ends at ndx on the original data.
            augmented = self.augmented[:,ndx:ndx+self.sample_size]
            X[i] = augmented
            if y_indices is not None:
                Y[i] = self.source.events[ndx]
        Y = None if (y_indices is None) else Y
        return X, Y
    
