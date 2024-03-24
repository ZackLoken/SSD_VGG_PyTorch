# SSD_VGG_PyTorch
 Repo containing PyTorch code and data for training SSD w/ VGG16 backbone to detect ducks from UAV imagery.

 Repository Contents:
 
 * SSD_VGG16_PyTorch_CustomDataset.ipynb -- Jupyter notebook containing code for performing PyTorch object detection on a custom dataset. Specifically, this notebook contains code for pre-processing image and annotation data, tuning hyperparameters using Bayesian Optimization, gradient accumulation enabled fine-tuning of SSD w/ VGG16 pre-trained on COCO and ImageNet, respectively, final model inference on the test dataset, and visualizing model predictions on original images. 
 
 * coco_eval.py -- COCO Style Dataset Evaluation Tools
 
 * coco_utils.py -- COCO Style Dataset Data Utilities
 
 * engine_gradientAccumulation.py -- Gradient accumulation enabled PyTorch object detection model training and evaluation engines
 
 * environment.yml -- YAML for cloning the Conda environment for this repo
 
 * transforms.py -- PyTorch object detection transformation functions
 
 * utils.py -- PyTorch utils for object detection training and evaluation engines

Data:

* https://drive.google.com/file/d/16RubwKyxA-Tr5K6KbQo2KbJto4I1p9o-/view?usp=sharing
