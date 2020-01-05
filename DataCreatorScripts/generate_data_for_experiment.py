import collections
import glob
import os
import argparse
import numpy as np
import imageio

from DataCreatorScripts.tile_scorer import score_tile
from DataCreatorScripts.tile_scorer import tissue_percent

def score_tiles(image_paths):
    image_tilescore_map = collections.OrderedDict()
    for impath in image_paths:
        image_name = os.path.split(impath)[1]
        image = imageio.imread(impath)
        tile_score = score_tile(image, tissue_percent=tissue_percent(image))
        image_tilescore_map[image_name] = tile_score
    image_tilescore_map = collections.OrderedDict(sorted(image_tilescore_map.items(),
                                      key=lambda kv: kv[1], reverse=True))
    return image_tilescore_map

def check_if_patches_not_exceed_threshold(case_image_map,wsi_number,k):
    if(wsi_number in case_image_map):
        if(case_image_map[wsi_number] <= k):
            return True
        else:
            return False
    return True

def extract_patches(args):
    annotated_image_folder = args.annotated_image_folder
    MIN_WHITE_FILTER_THRESHOLD = args.min_white_filter_threshold
    MAX_WHITE_FILTER_THRESHOLD = args.max_white_filter_threshold
    MAX_IMAGES_PER_SLIDE = args.max_patches_per_slide
    output_folder = args.output_folder + "ImagesForExperiment_" + str(MAX_WHITE_FILTER_THRESHOLD) + "_" + str(MAX_IMAGES_PER_SLIDE)

    image_paths = glob.glob(annotated_image_folder + "/*.png")
    image_tilescore_map = score_tiles(image_paths)
    slide_images_dict = {}

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read Images
    for image_path in image_paths:
        image_name = os.path.split(image_path)[1]
        splt = image_name.split("_")
        case_num = int(splt[0])
        case_label_ = splt[3]
        case_label = int(case_label_.split(".")[0])
        if ((case_label == 2) or (case_label == 4)):  # For invasive region
            image = imageio.imread(image_path)
            avg = np.mean(image)
            if ((avg <= MAX_WHITE_FILTER_THRESHOLD) and (avg >= MIN_WHITE_FILTER_THRESHOLD)):
                if (case_num in slide_images_dict):
                    slide_images_dict[case_num] += 1
                else:
                    slide_images_dict[case_num] = 1
                if (check_if_patches_not_exceed_threshold(slide_images_dict, case_num, MAX_IMAGES_PER_SLIDE)):
                    imageio.imsave(os.path.join(output_folder, image_name), image)
    return 0

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Tile scorer given a image')
  parser.add_argument('--annotated-image-folder', default='F:/Projects/HEROHE/Dataset/AnnotatedPatches',
                      help='Give the path to directory containing annotated images of the format => [slide number]_x_y_[label]')
  parser.add_argument('--min-white-filter-threshold', default=120, type=int,
                      help='Min white filter threshold')
  parser.add_argument('--max-white-filter-threshold', default=200, type=int,
                      help='Max white filter threshold')
  parser.add_argument('--max-patches-per-slide', default=500, type=int,
                      help='Max number of patches per slide')
  parser.add_argument('--labels', default="2,4",
                      help='comma separated labels that you want to extract')
  parser.add_argument('--output-folder', default='F:/Projects/HEROHE/Dataset',
                      help='Give the path to directory where you want to store extracted patches')
  args = parser.parse_args()
  extract_patches(args)