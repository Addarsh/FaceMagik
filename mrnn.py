"""
Mask R-CNN
Train on the face dataset to get segmentation.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Addarsh

"""

import os
import sys
import json
import datetime
import numpy as np
import argparse
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("Mask_RCNN")
DATASET_DIR = os.path.abspath("resized")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join("checkpoints")

class_dict = {'eye': 1, 'eyebrow': 2, 'nose': 3, 'upper lip': 4, 'lower lip': 5,
'face': 6, 'neck': 7, 'hair': 8, 'glasses': 9, 'ear': 10, 'teeth': 11,
'beard': 12, 'hat': 13, 'french beard': 14, 'sunglasses': 15, 'tongue': 16,
'goatee': 17, 'hoodie': 18, 'moustache': 19, 'stubble': 20, 'closed eye': 21}

############################################################
#  Configurations
############################################################


class FaceConfig(Config):
    """
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "face"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 21 + 1  # Background + class_dict

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class FaceDataset(utils.Dataset):

    def load_faces(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        source = "face"
        # Add classes.
        for c in class_dict:
          self.add_class(source, class_dict[c], c)

        # Train or validation dataset?
        assert subset in ["train", "val"]
        img_dir = os.path.join(dataset_dir, subset)

        for f in os.listdir(img_dir):
          if not f.endswith(".jpg") and not f.endswith(".png"):
            continue
          label_dir = os.path.join(dataset_dir, "labels")
          an = json.load(open(os.path.join(label_dir, os.path.splitext(f)[0] + ".json")))
          polygons = []
          for v in an["shapes"]:
            all_x = [p[0] for p in v["points"]]
            all_y = [p[1] for p in v["points"]]
            polygons.append({"all_points_x":all_x, "all_points_y":all_y, "label": v["label"]})

          image_path = os.path.join(img_dir, f)
          self.add_image(
            source=source,
            image_id=f, # filename used for annotation.
            path=image_path,
            width=an["imageWidth"],
            height=an["imageHeight"],
            polygons=polygons,
          )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "face":
          raise Exception("source is not face in load_mask")

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        class_ids = np.ones(len(info["polygons"]), dtype=np.int32)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            rr = self.bounds(rr, info["height"])
            cc = self.bounds(cc, info["width"])
            mask[rr, cc, i] = 1
            class_ids[i] = class_dict[p["label"]]

        # Return mask, and array of class IDs of each instance.
        return mask.astype(np.bool), class_ids

    def bounds(self, indices, ub):
      return [min(max(0, idx), ub-1) for idx in indices]

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] != "face":
          raise Exception("source is not face in image_reference")
        return info["path"]


def train(model):
    """Train the model."""
    print ("Training starts")

    # Training dataset.
    dataset_train = FaceDataset()
    dataset_train.load_faces(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = FaceDataset()
    dataset_val.load_faces(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')

def color_splash(image, mask, class_ids):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    count = 0
    m = np.zeros((image.shape[0], image.shape[1], len(class_ids)))
    for i, c in enumerate(class_ids):
      if c == 3 or c == 1 or c== 4 or c == 5 or c== 2 or c == 11 or c==8 or c==12 or c==9:
        m[:, :, i] = mask[:, :, i]
    mask = m

    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash

def detect(model, image_path):
  print("Running on {}".format(args.image))
  # Read image
  image = skimage.io.imread(args.image)
  # Detect objects
  r = model.detect([image], verbose=1)[0]

  print ("len mask: ", len(r["masks"]))
  # Color splash
  splash = color_splash(image, r["masks"], r["class_ids"])
  # Save output
  file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
  skimage.io.imsave(file_name, splash)

if __name__ == '__main__':
  # Parse command line arguments
  parser = argparse.ArgumentParser(
      description='Train Mask R-CNN to detect balloons.')
  parser.add_argument("command",
                      metavar="<command>",
                      help="'train' or 'detect'")
  parser.add_argument('--dataset', required=False,
                      metavar="/path/to/balloon/dataset/",
                      help='Directory of the Balloon dataset')
  parser.add_argument('--weights', required=True,
                      metavar="/path/to/weights.h5",
                      help="Path to weights .h5 file or 'coco'")
  parser.add_argument('--logs', required=False,
                      default=DEFAULT_LOGS_DIR,
                      metavar="/path/to/logs/",
                      help='Logs and checkpoints directory (default=logs/)')
  parser.add_argument('--image', required=False,
                      metavar="path or URL to image",
                      help='Image to apply the color splash effect on')
  args = parser.parse_args()

  # Validate arguments
  if args.command == "train":
      assert args.dataset, "Argument --dataset is required for training"
  elif args.command == "detect":
      assert args.image,\
             "Provide --image or --video to apply color splash"

  print("Weights: ", args.weights)
  print("Dataset: ", args.dataset)
  print("Logs: ", args.logs)

  # Configurations
  if args.command == "train":
      config = FaceConfig()
  else:
      class InferenceConfig(FaceConfig):
          # Set batch size to 1 since we'll be running inference on
          # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
          GPU_COUNT = 1
          IMAGES_PER_GPU = 1
      config = InferenceConfig()
  config.display()

  # Create model
  if args.command == "train":
      model = modellib.MaskRCNN(mode="training", config=config,
                                model_dir=args.logs)
  else:
      model = modellib.MaskRCNN(mode="inference", config=config,
                                model_dir=args.logs)

  # Select weights file to load
  if args.weights.lower() == "coco":
      weights_path = COCO_WEIGHTS_PATH
      # Download weights file
      if not os.path.exists(weights_path):
          utils.download_trained_weights(weights_path)
  elif args.weights.lower() == "last":
      # Find last trained weights
      weights_path = model.find_last()
  elif args.weights.lower() == "imagenet":
      # Start from ImageNet trained weights
      weights_path = model.get_imagenet_weights()
  else:
      weights_path = args.weights

  # Load weights
  print("Loading weights ", weights_path)
  if args.weights.lower() == "coco":
      # Exclude the last layers because they require a matching
      # number of classes
      model.load_weights(weights_path, by_name=True, exclude=[
          "mrcnn_class_logits", "mrcnn_bbox_fc",
          "mrcnn_bbox", "mrcnn_mask"])
  else:
      model.load_weights(weights_path, by_name=True)

  # Train or evaluate
  if args.command == "train":
    train(model)
  elif args.command == "detect":
    detect(model, args.image)
