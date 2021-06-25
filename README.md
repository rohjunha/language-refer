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
We recommend using `anaconda3` to setup the environment for the code.
Here's a list important libraries that are used in the code:

* python==3.8
* pytorch==1.9

These libraries will be installed if you follow the guide below.
  
#### Install guide
1. Install `anaconda3` manually or by running `bash ./install_anaconda.sh`. (Note that this will modify your `~/.bashrc` file.)
1. Setup an environment by running `conda env create -f env.yml`.
1. Activate the environment by running `conda activate lr`.
1. Download pre-trained model files by running `bash ./download_models.sh`
1. With the conda environment `lr` activated,
  1. For `nr3d`, run `python eval.py --dataset-name nr3d --pretrain-path $(PROJECT_PATH)/resources/models/nr3d`.
  1. For `sr3d`, run `python eval.py --dataset-name sr3d --pretrain-path $(PROJECT_PATH)/resources/models/sr3d`.
