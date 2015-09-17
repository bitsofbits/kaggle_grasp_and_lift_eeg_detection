from __future__ import print_function
import grasp
import ensemble
import importlib
import sys
import os
import subprocess
from multiprocessing import Process
import yaml
import json
import argparse

with open("SETTINGS.json") as file:
    config = json.load(file)


with open("final_nets.yml") as file:
    all_net_kwargs = yaml.load(file)
    
for x in all_net_kwargs:
    if ('dropch' in x) and x.pop('dropch'):
        x['ch'] = range(2,32)
    
def check_output_names(net_kwargs):
    names = set()
    for kwargs in net_kwargs:
        name = kwargs["output_name"]
        if name in names:
            raise ValueError("duplicate output name", name)
        names.add(name)
        
check_output_names(all_net_kwargs)


# We weighted know good results (0.97+ on public leaderboard by 2 relative to the 
# the other results).

ensemble_weights = { 'net_stf7.csv':2,
                     "net_stf7b.csv":2,
                     "net_stf7i.csv":2,
                     'net_stf7m.csv':2,
                     'net_stf7_fea6_150e20_LPF_00_100_dense1024_val6_allfreqdata.csv':2, 
                     # We intended to weight this net by two, but another net that
                     # was supposed to be in here net_stf7m_v3i somehow ended up
                     # being replaced by a duplicate of net_stf7b_v3i so it was
                     # effectively weighted by 4.
                     'net_stf7b_v3i.csv' : 4, 
}




def run_only_kwargs(kwargs):
    kwargs = kwargs.copy()
    for key in ['min_freq', 'max_freq', 'validation', "train_size", "valid_size"]:
        _ =  kwargs.pop(key, None)
    return kwargs

    
def run_net(i, run_type="run"):
    kwargs = all_net_kwargs[i].copy()
    print("*"*64)
    mod_name = kwargs.pop("net")
    output_name = kwargs.pop("output_name")
    dump_path = os.path.join(config["MODEL_PATH"], output_name) + ".dump"
    csv_path = os.path.join(config["SUBMISSION_PATH"], output_name) + ".csv"
    print("Loading module", mod_name, "for net", output_name)
    mod = importlib.import_module("nets." + mod_name)
    factory = getattr(mod, 'create_net')
    if run_type == "dry":
        kwargs = run_only_kwargs(kwargs)
        items = ["{0}={1}".format(k,v) for (k,v) in sorted(kwargs.items())]
        argstring = ", ".join(['None', 'None'] + items)
        print("Instantiating:", "{0}.create_net({1})".format(mod_name, argstring)) 
        net = factory(None, None, **kwargs)
        print("Would normally dump results to:", csv_path)
    else:
        if os.path.exists(dump_path):
            print(dump_path, "already exists; skipping training")
            print("Executing:", "info = load({0})".format(dump_path)) 
            info = grasp.load(dump_path)
        else:
            if run_type in  ("test_dump", "test_csv"):
                kwargs["max_epochs"] = 1
                kwargs["epoch_boost"] = 0
                dump_path += ".test"
                csv_path += ".test"
            items = ["{0}={1}".format(k,v) for (k,v) in sorted(kwargs.items())]
            argstring = ", ".join(["{0}.create_net".format(mod_name)] + items)
            print("Executing:", "info = train_all({1})".format(mod_name, argstring)) 
            info = grasp.train_all(factory, **kwargs)
            print("Executing:", "dump(info, '{0}')".format(dump_path)) 
            grasp.dump(info, dump_path)
        if run_type != "test_dump":
            if os.path.exists(csv_path) or os.path.exists(csv_path + ".gz"):
                print(csv_path, "already exists; skipping")
                return
            print("Executing: make_submission(info)") 
        grasp.make_submission(info, csv_path)
    
    
def submitted_net_names():
    return [x['output_name'] for x in all_net_kwargs]
        
    


def worker(offset, run_type):
#     flags = THEANO_FLAGS[offset % len(THEANO_FLAGS)]
    env = os.environ.copy()
    env["THEANO_FLAGS"] = config["theano_flags"][offset%len(config["theano_flags"])]
    n_workers = config["submission_workers"]
    for i in range(offset, len(all_net_kwargs), n_workers):
        print("processing", all_net_kwargs[i]["output_name"])
        print(env["THEANO_FLAGS"] )
        output_name = all_net_kwargs[i]["output_name"]
        csv_path = os.path.join(config["SUBMISSION_PATH"], output_name) + ".csv"
        with open(csv_path + ".log", 'w') as log:
            subprocess.check_call(["python", "submission.py", '-r', run_type, '-n', str(i)], stdout=log, env=env
            )



if __name__ == "__main__":

    help = """
    `python submission.py -h` -- list this help message.
    `python submission.py -r run -n <N>` -- train net #N.
    `python submission.py -r run` -- train all nets. This will take a LONG time.
    `python submission.py -r ensemble` -- compute the weighted average used in final submission.
    
    The directories for train, test, dumped models and csv output files are
    set in SETTINGS.json.
    
    When running all nets, the programs spreads the load out over `submission_workers`
    processes. Mulitple GPUs can be used by specifying an appropriate set of flags in
    `theano_flags`. Both of these can be found in SETTINGS.json.
    
    Note that this first checks if the dump file for a given net exists, if so it uses
    that, if not, it retrains the net (slow).  Then it checks if the csv file exists for
    this net, creating it if it doesn't exist.
    
    The submitted nets that are availble to run are:

"""

    
    net_names = "\n".join("    {0}: {1}".format(i, x) for (i, x) in 
                            enumerate(submitted_net_names()))

    brief = ("This is the main program to train the neural nets,\n"
             "save the nets into model files and generate predictions\n"
              "to testing set,and save it to csv output submission files.\n\n")
    
    parser = argparse.ArgumentParser(prog="python submission.py", 
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=brief+help+net_names)
    
    parser.add_argument('-r','--run_type', help='The run type' , required=True) 
    parser.add_argument('-n','--net', type=int, help='Neural Net ID', default=-1) 
    args = parser.parse_args()
    
    run_type = args.run_type
    which = args.net
        
    ## show values ##
    print ("The run type is: %s" % run_type )
    print ("The network to train: %s" % which )
    
    if which != -1:
        assert run_type in ["run", "test_csv", "test_dump", "dry"], run_type
        print("Running net: " + str(which) + " with runtype of: " + str(run_type))
        run_net(which, run_type)
    else:
        if run_type == "ensemble":
            output_path = os.path.join(config["SUBMISSION_PATH"], "ensemble.csv")
            input_paths = [os.path.join(config["SUBMISSION_PATH"], x["output_name"]) + '.csv' for x in all_net_kwargs]
            ensemble.naive_ensemble(output_path, input_paths, ensemble_weights)
            print("Running ensemble")
        else:
            print("Running nets in parallel")
            jobs = [Process(target=worker, args=(i, run_type)) for i in range(config["submission_workers"])]
            for p in jobs:
                p.start()
            for p in jobs:
                p.join()
