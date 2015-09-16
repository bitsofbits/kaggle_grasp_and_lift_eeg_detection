"""IndexBatchIterator

TODO: add basic example usage


"""
from nolearn.lasagne import NeuralNet, BatchIterator, TrainSplit

class IndexTrainSplit(TrainSplit):
    
    def __init__(self, eval_size=None, stratify_using=None):
        """
        Note: will stratify on non-regression problems if a non-null values of 
        `stratify_using` is passed in.
        """
        self.eval_size = eval_size
        self.stratify_using = stratify_using        

    def __call__(self, train, valid, net):
        if valid is None:
            if self.eval_size:
                if net.regression or (self.stratify_using is None):
                    kf = KFold(len(train), round(1. / self.eval_size))
                else:
                    kf = StratifiedKFold(self.stratify_using[train], round(1. / self.eval_size))
                train_indices, valid_indices = next(iter(kf))
                train, valid = train[train_indices], train[valid_indices]
            else:
                valid = train[len(train):]
        return train, valid, train, valid


class IndexNeuralNet(NeuralNet):

    def __init__(self, use_label_encoder=False, *args, **kwargs):
        if use_label_encoder:
            raise ValueError("cannot use label encoder with IndexNeuralNet")
        super(IndexNeuralNet, self).__init__(*args, **kwargs)
        
    def _check_good_input(self, train, valid=None):
        # Haven't thought about what would make sense here, so just return as is for now.
        return train, valid

    
class IndexBatchIterator(BatchIterator):
    """Base class / bare bones example for IndexBatchIterator
    """

    def __init__(self, source, *args, **kwargs):
        super(IndexBatchIterator, self).__init__(*args, **kwargs)
        self.source = source
    
    def transform(self, X_indices, y_indices):
        X_indices, y_indices = super(IndexBatchIterator, self).transform(X_indices, y_indices)
        X = self.source.X[X_indices]
        y = None if (y_indices is None) else self.source.y[y_indices]
        return X, Y
    
