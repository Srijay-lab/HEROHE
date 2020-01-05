# HEROHE
Implementation of baseline code (using resnet) for HEROHE Challenge

#To Generate Data for experiment:

1. Just run the command =>

python generate_data_for_experiment.py --annotated-image-folder [Path to directory containing annotated images of the format => [slide number]_x_y_[label]] --min-white-filter-threshold [Min white filter threshold] --max-white-filter-threshold [Max white filter threshold] --max-patches-per-slide [Max number of patches per slide] --labels [comma separated labels that you want to extract] --output-folder [path to directory where you want to store extracted patches]

#To run the training code :

Configure the parameters you need for experiment in config.txt and run the command =>
                                                               
python main.py --config [config file]

In case of any difficulty in understanding, Kindly email me on srijay.deshpande@warwick.ac.uk or whatsapp me on +918830695728
