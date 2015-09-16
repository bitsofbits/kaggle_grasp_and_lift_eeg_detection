"""Some helper functions for nolearn.lasagne

"""
import numpy as np
import pickle


class LayerFactory:

    def __init__(self, pretty_default_names=False):
        self.layer_cnt = 0
        self.kwargs = {}
        self.pretty_default_names = pretty_default_names

    def __call__(self, layer, layer_name=None, **kwargs):
        self.layer_cnt += 1
        if layer_name is None:
            if self.pretty_default_names:
                base = layer.__name__ if hasattr(layer, "__name__") else "Layer"
            else:
                base = "layer"
            layer_name = "{0}{1}".format(base, self.layer_cnt)
        for k, v in kwargs.items():
            self.kwargs["{0}_{1}".format(layer_name, k)] = v
        return (layer_name, layer)  
        
        

class AdjustVariable(object):
    def __init__(self, variable, target, half_life=20):
        self.variable = variable
        self.target = target
        self.half_life = half_life
    def __call__(self, nn, train_history):
        delta = self.variable.get_value() - self.target
        delta /= 2**(1.0/self.half_life)
        self.variable.set_value(np.float32(self.target + delta))


class EarlyStopping(object):

    def __init__(self, patience=100, window=1):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = -1
        self.best_weights = None
        self.window = window

    def __call__(self, nn, train_history):
        # This isn't a full implementation of windowing; it would probably be better
        # to take the best single loss over the range with minimum mean loss, or the 
        # center value, rather than the most recent value. That would require storing
        # `window` previous weights however.
        current_valid = np.mean([x['valid_loss'] for x in train_history[-self.window:]])
        current_epoch = train_history[-1]['epoch']
        if np.isnan(current_valid):
            print("NAN: Early stopping")
            if self.best_weights is not None:
                nn.load_params_from(self.best_weights)
            raise StopIteration()
        elif current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            raise StopIteration()

    def started(self, nn, history):
        self.best_weights = nn.get_all_params_values()

    def finished(self, nn, history):
        print("Best valid accuracy was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
        if self.best_weights is not None:
            nn.load_params_from(self.best_weights)


class WeightDumper(object):

    def __init__(self, name, interval=20):
        self.name = name
        self.interval = interval

    def __call__(self, nn, train_history):
        current_epoch = train_history[-1]['epoch']
        if (current_epoch % self.interval) == 0:
            nn.save_weights_to("{0}-{1}.pickle".format(self.name, current_epoch))




