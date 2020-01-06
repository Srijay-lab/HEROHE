# HEROHE

The HEROHE challenge is about the classification of whole slide images into one of the classes - HER2+ and HER2-. This repo is having the implementation of baseline model
for the challenge. The idea is to apply resnet (with some variations) to annotated patches collected from the whole slide images. The annotated patches for training are collected using some preprocessing techniques like
tile scoring, while color filtering. The section below defines the procedure of data collection, model training and evaluating the results.

# Data Generation:

Aim is to collect the annotated patches for experimentation of resnet. Detailed steps are given below -

1. Annotated Patches Collection

The code is pushed here for this here - https://github.com/Srijay-lab/py-wsi . Using this, we extracted the annotated patches from WSIs of size 256*256 with the magnification 10X. We get .png images with the format -
[slide number]_x_y_[label]. The label is the integer that you used for the annotation while doing first step.

2. Tile scoring and filtering

Just run the command =>

python generate_data_for_experiment.py --annotated-image-folder [Path to directory containing annotated images of the format => [slide number]_x_y_[label]] --min-white-filter-threshold [Min white filter threshold] --max-white-filter-threshold [Max white filter threshold] --max-patches-per-slide [Max number of patches per slide] --labels [comma separated labels that you want to extract] --output-folder [path to directory where you want to store extracted patches]

Filter thresholds : Specify minimum and maximum white filter threshold in the above command. The code will filter out patches having average pixel value out of the range.
Max patches per slide : Specify maximum number of patches extracted per slide to balance the patch level distribution.
labels : Specify comma separated labels that you want to keep. e.g. if you want to keep only invasive regions, keep integer corresponding to it.

Now you have filtered dataset ready to train the model

# Model Training & Evaluation :

Configure the parameters you need for experiment in config.txt and run the command =>
                                                               
python main.py --config [config file]

* mode : It can be train, eval or collect_stats. If set to "train", it will train the model based on the given training parameters. It will save model every after 5 epochs.
At last, it will save model named best.pth of the epoch which gives the maximum validation accuracy. If mode is set to "eval", it will evaluate model based on the
given training model path and plot auc-roc curve. If mode is "collect_stats", it will plot the patch distribution of whole slide images.patches

* If augment is set to true, it will append 4 extra images per training image. Augmented images are formed by applying random rotation, cropping in random order to
original image.

* If one-label-smoothing is set to true, it will assign real labels to the samples instead of binary 0 and 1. For example, sample can be assigned to label 0.1 instead
of 0 and 0.9 instead of 1.

After running model for sufficient epochs, you will get decent trained model.

# Contact

In case of any difficulty in understanding, Kindly email me on srijay.deshpande@warwick.ac.uk or whatsapp me on +918830695728
