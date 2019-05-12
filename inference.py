import os
import sys
import datetime
import numpy as np
import argparse
import skimage.draw
from train import FaceConfig, DATASET_DIR, CHECKPOINT_DIR, modellib

"""Apply color splash effect.
image: RGB image [height, width, 3]
mask: instance segmentation mask [height, width, instance count]

Returns result image.
"""
def color_splash(image, mask, class_ids):
    count = 0
    m = np.zeros((image.shape[0], image.shape[1], len(class_ids)))
    for i, c in enumerate(class_ids):
      if c== 14 or c==15 or c==16 or c==17:
      #if c == 1 or c==7 or c==9 or c==10 or c==4 or c == 3:
      #if c == 13:
        m[:, :, i] = mask[:, :, i]
    mask = m

    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, (0, 255, 0), gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


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
  splash = color_splash(image, r["masks"], r["class_ids"])
  # Save output
  file_name = "splash_{:%Y%m%dT%H%M%S}.jpg".format(datetime.datetime.now())
  skimage.io.imsave(file_name, splash)

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
