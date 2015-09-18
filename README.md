README
=======

**Table of Contents**

- [1. How To Generate the Solution](#1-how-to-generate-the-solution)
	- [1.1 Hardware / OS](#11-hardware--os)
	- [1.2 Installing Dependencies](#12-installing-dependencies)
	- [1.3 How to recreate test data.](#13-how-to-recreate-test-data)
	- [1.4 Making predictions on new data](#14-making-predictions-on-new-data)

# 1. How To Generate the Solution

## 1.1 Hardware / OS

Here is the hardware configuration of the main box we used for this competition:

	# CPU
    6 core / 12 virtual with hyperthreding - Intel(R) Xeon(R) CPU E5-2667 0 @ 2.90GHz base clock + turbo+

    # GPU
    NVIDIA GM204 [GeForce GTX 980]
	NVIDIA GM204 [GeForce GTX 980]

    # Memory
    DDR3 DIMM Mem: 32G  Swap: 225G

    # Hard Drive Space
    root: / 	100GB
    home: /home 1.8TB

    # OS
    Ubuntu 14.04 LTS

## 1.2 Installing Dependencies

The follwing setup assumes you have Ubuntu 14.04 LTS

Installing python & sklearn:
-   Install build-essential, which is a package in Ubuntu which includes gcc and other build tools

	- `$ sudo apt-get install python-dev`

	- `$ pip install --user -e scikit-learn`

Installing CUDA:
-   Install Cuda 7.0 (preferably 7.5 if you have GTX 980) by downloading the deb/run file from NVIDIA's website.
    Do not install it from the ubuntu ppa repos. The latest version there is always lagging behind.

-   Mostly follow: [The cuda getting started guide](http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-linux/)
	- If for some reason (such as interaction with nouveau driver, etc ...) 
	  the driver doesn't load (which you can check by running `$ cat /proc/driver/nvidia/version`,
      run `$ sudo ldconfig`

-   Run `$ nvidia-smi` to check for proper installation

-   Preferably run `$ cuda-install-samples-6.5.sh ~` and run some of the examples.

Installing Theano:
-   Install Theano dependencies first
    -   numpy
    -   scipy
    -   pip
    -   Just use the following command: `$ suodo apt-get install numpy scipy`
-   `$ git clone git://github.com/Theano/Theano.git`
-   `$ pip install --user -e Theano`
    - This installs Theano from the git repository you just cloned. The `--user` is needed
      if you don't want to install it system wide. You should do this so you don't pollute
      the system wide python packages. Virtualenv is also an option.

-   Follow [The using_gpu tutorial](http://deeplearning.net/software/theano/tutorial/using_gpu.html#using-gpu) to test Theano with the GPU

-   Place the following in ~/.theanorc

        [global]
        device = gpux # where x can be 0,1,2, ... GPU ID you want to make the default
        floatX = float32

Installing Lasagne and Nolearn:
- After making sure that Theano setup works, you can always use the following command to install the latest Theano, Lasagne, & Nolearn packages:

	- `pip install --no-deps git+git://github.com/Theano/Theano`

	- `pip install --no-deps git+git://github.com/benanne/Lasagne`

	- `pip install --no-deps git+git://github.com/dnouri/nolearn`

## 1.3 How to recreate test data.

The file *submission.py* is designed to be used to recreate the ensemble that
was used to generate our third place submission. To generate our result, folow
these four steps:

1. Install all depenencies as descriped below.

2. Edit SETTINGS.json: 
    - Set `train_dir` and `test_dir` to point to the correct training and
    testing data.

    - Set `dump_dir` and `submission_dir` to point to directories in which to
    write the trained models and the csv files respectively.

    - Optionally set `submission_workers` and `theano_flags`.
    `submission_workers` selects how many worker processes *submission.py*
    should spawn when training. `theano_flags` is a list of flags to pass to the
    theano. The primary use for this is to spread training across multiple GPUs.
    For instance `"theano_flags" : ["device=gpu0", "device=gpu1"] will run half
    the workers on *gpu0* and half on *gpu1*.

3. Run `python submission.py run` -- this will run all the models specified in
*final_nets.yml*, write the train models to `dump_dir` and the csv output files
to `submission_dir`. If a dump file with the corresponding name already exists
in *dump_dir*, then the training step will be skipped. If a csv file with the
corresponding name already exists in *submission_dir* then the submission
creation step will also be skipped. If all nets are being rerun from scratch,
this may take several days, depending on your hardware.

4. Run `python submission.py ensemble`. This will comput a weighted average of
all of the nets created in step 3 using the weighting scheme we used in our best
submission. The results will be written to `submission_dir` with the name
*ensemble.csv*.

In addition to the above procedure for generating *ensemble.csv*,
*submission.py* has a few other potentially useful commands.

- `python submission.py dry` -- performs a dry run; instantiating all nets but
not training

- `python submission.py` -- show a numbered list of the nets in
*final_nets.yml*

- `python submission.py run <N>` -- train and dump only net
number N, where N is the number given py the previous command.

Note that `python submission.py run` **will not** overwrite already existing
files, so if you wish to replace files you will have to remove them manually
before starting the run.

## 1.4 Making predictions on new data

To make a prediction on a new test set, one would simply make a new test
directory containing the new data and point SETTINGS.json to that directory.