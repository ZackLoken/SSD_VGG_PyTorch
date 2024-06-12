![alt text](https://github.com/ZackLoken/SSD_VGG_PyTorch/blob/main/test_results.png)  

Sample predictions for four images in the test dataset. Predictions have been post-processed using non-maximum suppression and filtered to remove any predictions with confidence scores below 0.5. For each detection, the model predicts the class, location, and confidence score. The model correctly predicted the locations and classes of one American coot and three northern shoveler (top left), 11 mallard (top right), six green-winged teal (bottom left), and 19 gadwall and one ring-necked duck (bottom right).

# SSD_VGG_PyTorch
Repo containing PyTorch code and data for training SSD w/ VGG16 backbone to detect ducks from UAV imagery. Users can clone this repository, load the corresponding conda environment using `environment.yml`, download the training images and annotation data (link below), and then change the folder paths in `SSD_VGG16_PyTorch_CustomDataset.ipynb` to the directory where the training data was downloaded. If using a GPU, CUDA Toolkit version 12.3 must be downloaded following the instructions here: (https://developer.nvidia.com/cuda-12-3-0-download-archive)

Repository Contents:
 
 * SSD_VGG16_PyTorch_CustomDataset.ipynb -- Jupyter notebook containing code for performing PyTorch object detection on a custom dataset. Specifically, this notebook contains code for pre-processing image and annotation data, tuning hyperparameters using Bayesian Optimization, gradient accumulation enabled fine-tuning of SSD w/ VGG16 pre-trained on COCO and ImageNet, respectively, final model inference on the test dataset, and visualizing model predictions on original images. 
 
 * coco_eval.py -- COCO style dataset evaluation tools
 
 * coco_utils.py -- COCO style dataset utilities
 
 * engine_gradientAccumulation.py -- Gradient accumulation enabled PyTorch object detection model training and evaluation engines
 
 * environment.yml -- YAML for cloning the Conda environment for this repo
 
 * transforms.py -- PyTorch object detection transformation functions
 
 * utils.py -- PyTorch utils for object detection training and evaluation engines

Data:

* https://drive.google.com/file/d/16RubwKyxA-Tr5K6KbQo2KbJto4I1p9o-/view?usp=sharing
