# LanguageRefer: Spatial-Language Model for 3D Visual Grounding

Anonymous submission to CoRL 2021.

![LR Figure](/resources/lr.png)

* pdf: https://openreview.net/pdf?id=dgQdvPZnH-t
* project: https://sites.google.com/view/language-refer

We added a video, examples of ReferIt3D datasets, qualitative results of the model, 
and a link to the orientation annotation webpage on the project page.


## Instruction to run the code
For running the code, we have to install the prerequisites, setup an environment, and then run the code.
You would be able to run the evaluation code if you follow the instruction step by step.

***WARNING: one of scripts contains the code that modifies your `~/.bashrc` file. 
Please make a copy of your `~/.bashrc` file.***

### Install prerequisites and environment
We recommend using `miniconda` to setup the environment for the code.
Here's a list important libraries that are used in the code:
* python==3.8
* pytorch==1.9
These libraries will be installed if you follow the guide below.
  
#### Install guide
1. Install `anaconda3` by following commands below:
```shell
export CONTREPO=https://repo.continuum.io/archive/
export ANACONDAURL=$(wget -q -O - $CONTREPO index.html | grep "Anaconda3-" | grep "Linux" | grep "86_64" | head -n 1 | cut -d \" -f 2)
wget -O ~/anaconda.sh $CONTREPO$ANACONDAURL
bash ~/anaconda.sh -b && rm ~/anaconda.sh && echo '# added by Anaconda3 installer' >> ~/.bashrc && echo '. $HOME/anaconda3/etc/profile.d/conda.sh' >> ~/.bashrc
```
Note that this will modify your `~/.bashrc` file.
1. Setup an environment by running `conda env create -f env.yml`.
1. Activate the environment by running `conda activate lr`.
1. Download pre-trained model files
```shell
mkdir -p ./resources/models/nr3d/
wget -N http://54.201.45.51:5000/lr/nr3d/model.pt -o ./resources/models/nr3d/model.pt
mkdir -p ./resources/models/sr3d/
wget -N http://54.201.45.51:5000/lr/sr3d/model.pt -o ./resources/models/sr3d/model.pt
```
1. With the conda environment `lr` activated,
  1. For `nr3d`, run `python eval.py --dataset-name nr3d --pretrain-path (PROJECT_PATH)/resources/models/nr3d`.
  1. For `sr3d`, run `python eval.py --dataset-name sr3d --pretrain-path (PROJECT_PATH)/resources/models/sr3d`.
