from __future__ import print_function
import numpy as np
import importlib
from sklearn.metrics import roc_auc_score
from numpy import fft
import pandas
import pickle
from scipy.signal import lfilter, butter
from copy import copy
import warnings
import json

with open("SETTINGS.json") as file:
    config = json.load(file)

warnings.filterwarnings('ignore', module='.*nolearn.lasagne.*')

# Constants that are fixed for the competition

N_EVENTS = 6
SAMPLE_RATE = 500
SUBJECTS = list(range(1,13))
TRAIN_SERIES = list(range(1,9))
TEST_SERIES = [9,10]


# By default we train on DEFAULT_TRAIN_SIZE randomly selected location each "epoch"
# (yes, that's not really an epoch). Similarly, we validate each "epoch" on
# VALID_SIZE randomly chosen points. The validation points are chosen with the
# random seed set to VALID_SEED. Note that the training points are different
# each epoch while the validation points are the same.
DEFAULT_TRAIN_SIZE = 16*1024
DEFAULT_VALID_SIZE = 8*1024
VALID_SEED = 199


# Utility Functions


def path(subject, series, kind):
    prefix = config["TRAIN_DATA_PATH"] if (series in TRAIN_SERIES) else config["TEST_DATA_PATH"]
    return "{0}/subj{1}_series{2}_{3}.csv".format(prefix, subject, series, kind)
    
def read_csv(path):
    return pandas.read_csv(path, index_col=0).values

def splice_at(x, n, size=32):
    # Make a smooth splice at a junction created by earlier concatenation.
    # n-1 is one side of junction, n is the other
    mean = 0.5 * (x[n-1] + x[n])
    region = x[n-size:n+size] - mean
    region[:size] *= np.linspace(1,0,size)[:,None]
    region[size:] *= np.linspace(0,1,size)[:,None]
    x[n-size:n+size] = region + mean

FILTER_N = 4  # Order of the filters to use

def butter_lowpass(highcut, fs, order):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype="lowpass")
    return b, a

def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    cutoff = [lowcut / nyq, highcut / nyq]
    b, a = butter(order, cutoff, btype="bandpass")
    return b, a

def butter_highpass(highcut, fs, order):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype="highpass")
    return b, a


# Sources for Batch Iterators
#
# These classes load training and test data and perform some basic preprocessing on it.
# They are then passed to factory functions that create the net. There they are used
# as data sources for the batch iterators that feed data to the net.
# All classes band pass or low pass filter their data based on min / max freq using
# a causal filter (lfilter) when the data is first loaded.
# * TrainSource loads a several series of EEG data and events, splices them together into
#   one long stream, then normalizes the EEG data to zero mean and unit standard deviation.
# * TestSource is like TrainSource except that it uses the mean and standard deviation
#   computed for the associated training source to normalize the EEG data.
# * SubmitSource is like TestSource except that it does not load and event data.

class Source:

    mean = None
    std = None
        
    _series_cache = {}
    # Big enough to cache first to subjects for interactive trial and error
    MAX_CACHE_SIZE = 20
    
    def load_series(self, subject, series):
        min_freq = self.min_freq
        max_freq = self.max_freq
        key = (subject, series, min_freq, max_freq) 
        if key not in self._series_cache:
            while len(self._series_cache) > self.MAX_CACHE_SIZE:
                # Randomly throw away an item
                self._series_cache.popitem()    
            print("Loading", subject, series)
            data = read_csv(path(subject, series, "data"))
            # Filter here since it's slow and we don't want to filter multiple
            # times. `lfilter` is CAUSAL and thus doesn't violate the ban on future data.
            if (self.min_freq is None) or (self.min_freq == 0):
                print("Low pass filtering, f_h =", max_freq)
                b, a = butter_lowpass(max_freq, SAMPLE_RATE, FILTER_N)
            else:
                print("Band pass filtering, f_l =", min_freq, "f_h =", max_freq)
                b, a = butter_bandpass(min_freq, max_freq, SAMPLE_RATE, FILTER_N)                
            self._series_cache[key] = lfilter(b, a, data, axis=0)
        return self._series_cache[key]
        
    def load_raw_data(self, subject, series_list):
        self.raw_data = [self.load_series(subject, i) for i in series_list]
            
    def assemble_data(self):
        if self.raw_data :
            self.data = np.concatenate(self.raw_data, axis=0)
        else:
            self.data = np.zeros([0,32])
        n = 0
        for x in self.raw_data[:-1]:
            n += len(x)
            splice_at(self.data, n)

    def load_events(self, subject, series):
        self.raw_events = [read_csv(path(subject, i, "events")) for i in series]
        
    def assemble_events(self):
        if self.raw_events:
            self.events = np.concatenate(self.raw_events, axis=0)
        else:
            self.events = np.zeros([0])
    
    
    def normalize(self):
        self.data -= self.mean
        self.data /= self.std
            
    def __len__(self):
        return len(self.data)
    
    
class TrainSource(Source):
    def __init__(self, subject, series_list, min_freq, max_freq):
        self.subject = subject
        self.series_list = series_list
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.load_raw_data(subject, series_list)
        self.load_events(subject, series_list)
        self._init()
        
    def _init(self):
        self.assemble_data()
        self.assemble_events()
        self.mean = self.data.mean(axis=0)
        self.std = self.data.std(axis=0)
        self.normalize()


class TestSource(Source):
    def __init__(self, train_source):
        self.min_freq = train_source.min_freq
        self.max_freq = train_source.max_freq
        vseries = sorted(set(TRAIN_SERIES) - set(train_source.series_list))
        self.series_list = vseries
        self.load_raw_data(train_source.subject, vseries)
        self.load_events(train_source.subject, vseries)
        self._init(train_source)
        
    def _init(self, train_source):
        self.assemble_data()
        self.assemble_events()
        self.mean = train_source.mean
        self.std = train_source.std
        self.normalize()
        
        
class SubmitSource(Source):
    def __init__(self, subj, series, train_source):
        self.series_list = series
        self.min_freq = train_source.min_freq
        self.max_freq = train_source.max_freq
        self.load_raw_data(subj, series)
        self._init(train_source)
        
    def _init(self, train_source):
        self.assemble_data()
        self.mean = train_source.mean
        self.std = train_source.std
        self.normalize()


# These two function support validation using the last several trials in each series.
# This is specified in train/train_all with validation=<integer>

def find_split_index(events, count):
    in_event = False
    events_seen = 0
    ndx = len(events) - 1
    while ndx > 0:
        if in_event:
            if events[ndx].sum() == 0:
                in_event = False
                events_seen += 1
                if events_seen >= count:
                    start = ndx
        else:
            if events_seen >= count and events[ndx].sum():
                return int((ndx + start) / 2)
            if events[ndx,0] == 1:
                in_event = True
        ndx -= 1
    else:
        raise ValueError("couldn't find a good split point")


def split_source(train_source, validation_count=1):
    """Split a training source into a training and a test Source
    
    This takes `validation_count` trials from each series and uses them in returned
    test series. The remaining trials are used for the train source.
    
    Arguments:
    validation_count -- number of trial from each series to place into the test source
    
    """
    raw_events = train_source.raw_events
    raw_data = train_source.raw_data
    #
    new_train = copy(train_source)
    new_train.raw_data = []
    new_train.raw_events = []
    new_test = TestSource(train_source)
    new_test.raw_data = []
    new_test.raw_events = []
    for events, data in zip(raw_events, raw_data):
        split_index = find_split_index(events, validation_count)
        new_train.raw_events.append(events[:split_index])
        new_test.raw_events.append( events[split_index:])
        new_train.raw_data.append(data[:split_index])
        new_test.raw_data.append( data[split_index:])
    new_train._init()    
    new_test._init(new_train)
    #
    return new_train, new_test



# These are utility functions associated with computin the ROC AUC score



def make_valid_indices(source, count):
    """Make a set of `count` indices to use for validaton""" 
    test_indices = np.arange(len(source.data))
    np.random.seed(VALID_SEED)
    np.random.shuffle(test_indices)
    return test_indices[:count]

def score(net, samples=4096):
    """Compute the area under the curve, ROC score from a trained net
    
    We take `samples` random samples and compute the ROC AUC
    score on those samples. 
    """
    source = net.batch_iterator_test.source
    test_indices = make_valid_indices(source, samples)
    predicted = net.predict_proba(test_indices)
    if predicted.shape[-1] != N_EVENTS:
        predicted = decode(predicted)
    actual = source.events[test_indices]
    try:
        return roc_auc_score(actual.reshape(-1), predicted.reshape(-1))
    except:
        return 0
    
def score_for(train_info, subj, series, samples=1024, **kwargs):
    """Compute the roc_auc score from train_info"""
    factory, info = train_info
    weights, train_source = info[subj]
    if isinstance(series, int):
        min_freq = train_source.min_freq
        max_freq = train_source.max_freq
        base_source = TrainSource(subj, TRAIN_SERIES, min_freq, max_freq)
        _, test_source = split_source(base_source, series)
    else:
        test_source = TestSource(train_source)  
    indices = np.arange(len(test_source.data))
    net = factory(train_source=None, test_source=test_source, **kwargs)
    net.load_weights_from(weights)
    #
    return score(net, samples)   
       

# *******************************************************************    
# These are the core functions meant to be used from outside of the 
# module: train, train_all, submit, load, dump

def train(factory, subject, max_epochs=100, validation=[3,6], 
            min_freq=0.2, max_freq=50, params=None, 
            train_size=DEFAULT_TRAIN_SIZE, valid_size=DEFAULT_VALID_SIZE,
            **kwargs):
    """Train a net created by `factory` for `subject`
    
    Arguments:
    factory -- function that returns a net. Arguments vary and can be passed in `kwargs`
    max_epochs -- maximum number of epochs to train for
    validation -- type of validation to use. This can be either:
       - a list: series specified in the list are used for validation
       - an integer: last `validation` trials of each series are used for validation
    min_freq -- lower frequency to band pass filter at. If None low pass filter instead
    max_freq -- upper frequency to band pass or low pass filter at
    train_size -- the number of points to train with each epoch
    valid_size -- the number of points to validate with
    **kwargs -- extra arguments to be passed to `factory`
    
    """
    # by passing in -1s to the train source, we get a random set of points
    # to train at each time.
    train_indices = np.zeros([train_size], dtype=int) - 1
    #
    if isinstance(validation, int):
        base_source = TrainSource(subject, TRAIN_SERIES, min_freq, max_freq)
        train_source, test_source = split_source(base_source, validation)
        valid_indices = make_valid_indices(test_source, valid_size)
    elif validation:
        tseries = sorted(set([1,2,3,4,5,6,7,8]) - set(validation))
        train_source = TrainSource(subject, tseries, min_freq, max_freq)
        test_source = TestSource(train_source)
        valid_indices = make_valid_indices(test_source, valid_size)
    else:
        test_source = None
        valid_indices = []
        kwargs['patience'] = 0
    for k, v in list(kwargs.items()):
        if isinstance(v, dict):
            if subject in v:
                kwargs[k] = v[subject]
            else:
                del kwargs[k]
    net = factory(train_source, test_source, max_epochs=max_epochs, **kwargs)
    if params is not None:
        net.load_params_from(params)
    while True:
        try:
            net.fit(train_indices, valid_indices)
        except MemoryError:
            input("Memory Error press any key to retry (^C) to stop")
        else:
            break
    params = net.get_all_params_values()
    if validation:
        score = score_for( (factory, {subject : (params, train_source)}), subject, validation, **kwargs)
        print("Score:", score)
    else:
        score = None
    return (params, train_source, score)

 
def train_all(factory, max_epochs=20, epoch_boost=20, **kwargs):
    """Train a net created by `factory` for all subjects
    
    We train the net for the first subject for a maximum of `max_epochs`+`epoch_boost`
    epochs. Subsequent subjects are trained for only `max_epochs`, but we use a warm
    start, initializing their weights based on the weights computed for the previous
    subject. This greatly speeds up the fit.
    
    This is the primary function for training nets. Typical usage would be:
    
    >>> import grasp
    >>> import net_stf7
    >>> info = grasp.train_all(net_stf7.create_net, max_epochs=50)
    wait for several hours .....
    >>> grasp.make_submission(info, "path_to_write_output_to.csv")
    
    Arguments:
    factory -- a function that returns a net. Arguments vary and can be passed in `kwargs`
    max_epochs -- the maximum number of epochs to train all but the first subject for
    epoch_boost --  extra epochs to train for on first subject
    **kwargs -- args to forward on to `train`
    
    """
    info = {}
    net = None
    params = None
    scores = []
    for subj in SUBJECTS:
        print("Subject:", subj)
        epochs = max_epochs + epoch_boost
        params, source, score = train(factory, subj, epochs, params=params, **kwargs)
        scores.append(score)
        info[subj] = (params, source, score)
        epoch_boost = 0
    print("Overall score:", np.mean(scores))
    kwargs.update({'max_epochs' : max_epochs, 'epoch_boost' : epoch_boost})
    return (factory, kwargs, info)   
  
def submit_only_kwargs(kwargs):
    """Strip out kwargs that are not used in submit"""
    kwargs = kwargs.copy()
    for key in ['patience', 'min_freq', 'max_freq', 'validation',
                "max_epochs", "epoch_boost", "train_size", "valid_size"]:
        _ =  kwargs.pop(key, None)
    return kwargs
 
def make_submission(train_info, path):
    """create a submission file based on `train_info` at `path`"""
    factory, kwargs, info = train_info
    all_probs = []
    for subj in SUBJECTS:
        weights, train_source, score = info[subj]
        for series in [9,10]:
            print("Subject:", subj, ", series:", series)
            submit_source = SubmitSource(subj, [series], train_source)  
            indices = np.arange(len(submit_source.data))
            net = factory(train_source=None, test_source=submit_source, **submit_only_kwargs(kwargs))
            net.load_weights_from(weights)
            all_probs.append((subj, series, net.predict_proba(indices)))
    #           
    with open(path, 'w') as file:
        file.write("id,HandStart,FirstDigitTouch,BothStartLoadPhase,LiftOff,Replace,BothReleased\n")
        for subj, series, probs in all_probs:
            for i, p in enumerate(probs):
                id = "subj{0}_series{1}_{2},".format(subj, series, i)
                file.write(id + ",".join(str(x) for x in p)+"\n")    
                
def dump(train_info, path):
    factory, kwargs, info = train_info
    factory_tuple = (factory.__module__, factory.__name__)
    series = set([1,2,4,5,7,8])
    stripped_info = {k : {'params':params, 'series_list':source.series_list, 'score' : score}
                             for (k, (params, source, score)) in info.items()}
    dumpable_info = {'factory' : factory_tuple, 'kwargs' : kwargs, 'subject_info' : stripped_info}
    with open(path, 'wb') as f:
        pickle.dump(dumpable_info, f)
        
def load(path):
    with open(path, 'rb') as f:
        try:
            # This works if data was saved under py2 and this is py3
            loaded = pickle.load(f, encoding="latin-1")
        except:
            loaded = pickle.load(f)    
    mod_name, func_name = loaded["factory"]
    try:
        mod = importlib.import_module(mod_name)
    except:
        # Some of the dumps were created before nets moved to `nets` directory
        mod = importlib.import_module("nets."+mod_name)        
    factory = getattr(mod, func_name)
    info = {}
    # Some of the dumps were computed before kwargs were dumped.
    kwargs = loaded.get('kwargs', {})
    min_freq = kwargs.get("min_freq", 0.2)
    max_freq = kwargs.get("max_freq", 50)
    for subj, value in loaded['subject_info'].items():
        source = TrainSource(subj, value['series_list'], min_freq, max_freq)
        score = value.get("score", 0)
        info[subj] = (value['params'], source, score)
    return factory, kwargs, info         

                
