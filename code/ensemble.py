from __future__ import print_function
import pandas
import os
import numpy as np

    
def naive_ensemble(output_path, input_paths, weights):
    """load and average csv at `input_paths` using `weights`
    
    Arguments:
    output_path -- location to write resultant csv file to
    input_paths -- paths of csv file to average
    weights -- a dictionary of weights keyed to the files basename
    
    """
    print(input_paths)
    data = 0
    numerator = 0
    weights = weights.copy()
    for path in input_paths:
        name = os.path.basename(path)
        x = pandas.read_csv(path, index_col=0)
        wt = weights.pop(name, 1)
        print("Adding", name, "with weight",  wt)
        data += wt * x
        numerator += wt
    if weights:
        print("Warning: not all weights used", weights)
    data = data / numerator
    data.to_csv(output_path)
    




   
