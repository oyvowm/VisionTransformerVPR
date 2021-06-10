# VisionTransformerVPR
 
 This project explored the viability of Vision Transformers for VPR tasks. Two training schemes formulating the problem as classification- and ranking tasks were proposed, which then later used to train seven different models based DeiT, PiT and CaiT. 
 
 ## Prerequisites 
 
 ### For evaluating:
 
 - Create an environment with the required dependencies:
 
   `conda env create --file environment_eval.yml`
 
 - Pytorch might have to be installed as per: https://pytorch.org/get-started/locally/
 
 - Download the required datasets from:
 
   https://surfdrive.surf.nl/files/index.php/s/sbZRXzYe3l0v67W
 
   These should then be extracted into the "Datasets" folder.
 
 - For comparisons to other VPR-techniques, precomputed data can be downloaded from:
 
   https://surfdrive.surf.nl/files/index.php/s/ThIgFycwwhRCVZv
 
 - The filenames of the .npy files must then be renamed, following the naming convention used by evaluate.py, that is:
 
   `<name_of_dataset>_precomputed_data_<name_of_VPR_technique>`
 
   After being renamed, the matching data can be added to the "precomputed" folder.
 
 ### For training new models:
 
 - The notebook "training.ipynb" can be uploaded to google colab, or alternatively hosted on a local machine.
 - To train on the SPED dataset, this currently available partition can be downloaded at: https://www.dropbox.com/s/aklu4tz3hurycj0/SPED_900.zip?dl=0
 - To train on the Mapillary SLS dataset, access needs to be requested from: https://www.mapillary.com/dataset/places 
 - If using the PiT architecture, its pre-trained models can be downloaded from: https://github.com/naver-ai/pit
 - DeiT and CaiT will be automatically downloaded from torch hub when using the notebook.
 

 
 ## Evaluation
 
 To run evaluations:
 
   `evaluation.py -d <name_of_dataset> -m all -pre True`
 
   Where -m can alternatively take in the name/s of the models/VPR-techniques to evaluate, and -pre flags whether or not to include the precomputed data for external techniques.
 
 ## Training
 To train a new model:
 
  - The path to the extracted dataset need to added to the training_config dictionary. 
  -  Add the python scripts of this repository to the current working directory, on colab the repository can be cloned onto the current instance executing the following in a cell:

  `!git clone https://github.com/oyvowm/VisionTransformerVPR`

 - Other desired tweaks and adjustments to the training procedure can be defined through changing the initial dictionaries, in which much of the training and model configurations are defined.
 
