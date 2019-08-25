# Goal Inference GAN (GIGAN)
> GIGAN is an implementation of Cusumano's Inference strategy that samples using trajectories from SGAN.
This project is part of the "Intent Prediction" 2019 INTEL Labs project in collaboration with CIMAT.

#### Table of contents
1. Installation guide
2. Running samples
3. Library structure
4. Bibliography

------------------------
 
## <a name="chapter1"></a> 1. Installation guide

First make a clone of the project running:

    $ git clone https://github.com/cimat-ris/GoalInferenceGAN.git
    
And before opening it up we, should run the following scripts:

    $ cd GoalInferenceGAN
    $ ./scripts/install_repositories.sh
    $ ./scripts/install_dataset.sh    

> - The first script will clone our *cimat-ris/sgan* fork that has a couple tweaks for being called as a python library.
> - The second script will download datasets to the *samples/datasets* folder

GIGAN is a PyCharm project, it has the ".idea" folder so it can be opened with it by default. The *sgan* clone should be
at the same folder level as the *GoalInferenceGAN* for PyCharm to find it correctly.

## <a name="chapter2"></a> 2. Library structure

GoalInferenceGAN sources consists on the following folders:
- _**gigan**_ : The python library itself, hence can be used and imported into any other python project using the 
_import_ command. It's modules are as follows:
    - _data_ : Common data structures for entities such as trajectories, obstacles, agents, etc...
    - _extensions_ : Special code that is meant to extend functionality of dependant python libraries. Related helper 
    functions for such libraries will be here too.
    - _models_ : Prebuilt inference models will be stored here. Though they might be only used for particular datasets,
    since models depend on the problem, this will be helpful reference for further implementations.
    - _utils_ : Operations too common in the python world, probably copy-pasted from the internet. Might have some actual
     useful code for mathematical evaluations.
    - _visualization_ : Code that will depend mostly on OpenCV for displaying results and annotated video playback.
- _**scripts**_ : Shell scripts for UNIX systems (tested on Ubuntu at least) for automatic download and installation of
what might be needed in order to run the samples.
- _**samples**_ : Actual example usage of the GIGAN library. Should contain many executables for real-life problems using
the framework we created and give a good idea on how to improve and expand functionality of existing models. 
- _**practice**_ : Mostly useless code, personal tests on third-party code in order to evaluate what we can use for 
GIGAN in future releases. 
- _**tests**_ : Contains test programs for testing out accuracy and correct results of individual components.
- _**docs**_ : Automated document-generation using sphynx. Hopefully most of the functions in GIGAN will have proper
 documentation for it to be generated here.   
      
## <a name="chapter3"></a> 3. Running samples

#### Cusumano's Goal Inference model
Inspired by paper [2], the basic inference framework using Metropolis-Hastings runs the model: 
START, GOAL, PATH, NOISY_PATH] but instead of sampling the PATH on a Rapidly-growing Random Tree it uses SGAN's 
trained neuron for path generation. Execution is simply by:
    
    $ python samples/gigan_cusumano.py

It should be noted you need to install datasets as stated on [1](#chapter1). Running gigan_cusumano.py should begin a 
video playback of the frames, performing inference each single frame and drawing results (inferred goal for each 
pedestrian) on the image. Lines on targets trajectory represent ground-truth and observed data using different colors.

## <a name="chapter4"></a> 4. Bibliography

* [1] **Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks**. 
    Agrim Gupta, Justin Johnson, Fei-Fei Li, Silvio Savarese, Alexandre Alahi. Presented at CVPR 2018.
    [GitHub Project page.](https://github.com/agrimgupta92/sgan)
    
* [2] **Probabilistic programs for inferring the goals of autonomous agents**. Marco F. Cusumano-Towner, Alexey Radul, 
    David Wingate, Vikash K. Mansinghka
    [arXiv page](https://arxiv.org/abs/1704.04977)