import os
import sys
import numpy as np
import argparse
import json
import random
import skimage.draw
from train import FaceConfig, DATASET_DIR, CHECKPOINT_DIR, modellib, label_id_map
from train import (
  EYE_OPEN,
  EYEBALL,
  EYEBROW,
  READING_GLASSES,
  SUNGLASSES,
  EYE_CLOSED,
  NOSE,
  NOSTRIL,
  UPPER_LIP,
  LOWER_LIP,
  TEETH,
  TONGUE,
  FACIAL_HAIR,
  FACE,
  HAIR_ON_HEAD,
  BALD_HEAD,
  EAR,
)
from imantics import Mask
CLASS = "class"
DATA = "data"

OUTPUT_DIR = "output"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
ANNOTATIONS_DIR = os.path.join(OUTPUT_DIR, "annotations")

id_label_map = {v: k for k, v in label_id_map.items()}

def color(image, mask, class_ids):
  m = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255

  r = lambda: random.randint(0,255)

  color_map = {}
  for class_id in class_ids:
    if class_id in color_map:
      continue
    color_map[class_id] = (r(),r(),r())

  ann = []
  for i, class_id in enumerate(class_ids):
    polygons = Mask(mask[:,:, i]).polygons()
    list_polygons = []
    for p in polygons.points:
      list_polygons.append(p.tolist())
    ann.append({CLASS: id_label_map[class_id], DATA: list_polygons})
    gmask = np.reshape(mask[:,:, i], (image.shape[0], image.shape[1], 1))
    m = np.where(gmask, color_map[class_id], m).astype(np.uint8)

  return m, ann


"""
detect runs detection algorithm for face masks.
"""
def detect(model, image_path):
  print("Running on {}".format(args.image))
  # Read image
  image = skimage.io.imread(args.image)
  # Detect objects
  r = model.detect([image], verbose=1)[0]

  print ("len mask: ", len(r["masks"]))
  # Color splash
  splash, ann = color(image, r["masks"], r["class_ids"])

  fname = os.path.splitext(os.path.split(image_path)[1])[0]

  # Save output
  skimage.io.imsave(os.path.join(IMAGE_DIR, fname+".jpg"), splash)

  # save annotation.
  with open(os.path.join(ANNOTATIONS_DIR, fname+".json"), 'w') as outfile:
    json.dump(ann, outfile)

if __name__ == '__main__':
  # Parse command line arguments
  parser = argparse.ArgumentParser(
      description='Train Mask R-CNN to detect balloons.')
  parser.add_argument('--image', required=True,
                      metavar="path or URL to image",
                      help='Image to apply the color splash effect on')
  args = parser.parse_args()

  print("Dataset: ", DATASET_DIR)
  print("Logs: ", CHECKPOINT_DIR)

  # Configurations
  class InferenceConfig(FaceConfig):
      # Set batch size to 1 since we'll be running inference on
      # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
      GPU_COUNT = 1
      IMAGES_PER_GPU = 1
  config = InferenceConfig()
  config.display()

  # Create model
  model = modellib.MaskRCNN(mode="inference", config=config,
                                model_dir=CHECKPOINT_DIR)

  # Select weights file to load
  weights_path = ""
  try:
    weights_path = model.find_last()
  except Exception as e:
    raise

  # Load weights
  print("Loading weights ", weights_path)
  model.load_weights(weights_path, by_name=True)

  # Run inference.
  detect(model, args.image)
