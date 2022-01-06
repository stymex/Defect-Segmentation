This project test four major segmentation networks on the BSData dataset.
To setup the project for execution:
Download the dataset in question
```
git clone https://github.com/2Obe/BSData.git
```
Install requirements
```
pip install -r requirements.txt
```
create necessary folders
```
mkdir IMAGES MASKS IMAGE_PATCHES IMAGE_PATCHES_AUGUMENTED
```
Convert json to png masks
```
python json2mask.py
```
Transform images into patches
```
python patching_and_saving.py
```
Train the network on default parameters, signing into Wandb will be necessary for progress logging
```
python Unet.py
```

