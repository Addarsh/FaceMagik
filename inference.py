import os
import sys
import numpy as np
import argparse
import json
import time
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
from math_utils import MathUtils
from image_utils import ImageUtils

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

# maskPoints returns points in the mask that are True.
def maskPoints(mask):
  points = []
  for x in range(mask.shape[0]):
    for y in range(mask.shape[1]):
      if mask[x,y]:
        points.append((x,y))
  return points

# newMask returns a new boolean mask which is
# True for given set of points.
def newMask(dims, points):
  mask = np.zeros(dims, dtype=bool)
  for p in points:
    x, y = p
    mask[x, y] = True
  mask = np.reshape(mask, (dims[0], dims[1], 1))
  return mask

# shadwo_masks constructs shadows from given input points
# and applies them to input mask.
def shadow_masks(m, image, maskPts):
  gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255

  pts1, pts2 = MathUtils.segregate_points(image, maskPts)
  pts1_mean, pts2_mean = ImageUtils.avg_intensity(gray, pts1), ImageUtils.avg_intensity(gray, pts2)

  darker_pts = pts1 if pts1_mean[0] < pts2_mean[0] else pts2
  lighter_pts = pts1 if pts1_mean[0] >= pts2_mean[0] else pts2
  d = pts1_mean if pts1_mean[0] < pts2_mean[0] else pts2_mean
  l = pts1_mean if pts1_mean[0] >= pts2_mean[0] else pts2_mean

  darkMask = newMask(m.shape[:2], darker_pts)
  lightMask = newMask(m.shape[:2], lighter_pts)

  m = np.where(darkMask, d, m).astype(np.uint8)
  m = np.where(lightMask, l, m).astype(np.uint8)

  return m, darker_pts, lighter_pts

"""
detect face is responsible for finding the face segmentation pixels
and clustering them by similarity.
"""
def detect_face():
  model = construct_model(1)

  print("Running on {}".format(args.image))
  # Read image
  image = skimage.io.imread(args.image)
  # Detect objects
  preds = model.detect([image], verbose=1)[0]

  print ("len mask: ", len(preds["masks"]))

  m = np.zeros(image.shape)

  for i, class_id in enumerate(preds["class_ids"]):
    if id_label_map[class_id] != FACE:
      continue
    m, dpts, lpts = shadow_masks(m, image, maskPoints(preds["masks"][:, :, i]))
    m, _, _ = shadow_masks(m, image, lpts)
    m , _, _ = shadow_masks(m, image, dpts)
    break


  # Save output
  skimage.io.imsave(os.path.split(args.image)[1], m)

# save_images saves the images in the given list to directory.
def save_images(model, imgList):
  preds = model.detect([t[0] for t in imgList], verbose=1)
  for i, pred in enumerate(preds):
    image, iDir = imgList[i]
    m = np.zeros(image.shape)
    for j, class_id in enumerate(pred["class_ids"]):
      if id_label_map[class_id] != FACE:
        continue
      m, dpts, lpts = shadow_masks(m, image, maskPoints(pred["masks"][:, :, j]))
      m, _, _ = shadow_masks(m, image, lpts)
      m , _, _ = shadow_masks(m, image, dpts)
      break
    skimage.io.imsave(os.path.join(iDir, "shadow.jpg"), m)

# Run detection on AWS to get multiple faces for training.
def detect_on_aws():
  k = 3 # Number of images on which to perform detection.
  dirpath = "train"
  model = construct_model(k)
  imgList = []
  for d in os.listdir(dirpath):
    if not os.path.isdir(os.path.join(dirpath, d)):
      continue
    for sd in os.listdir(os.path.join(dirpath, d)):
      if not os.path.isdir(os.path.join(os.path.join(dirpath, d), sd)):
        continue
      imPath = os.path.join(os.path.join(os.path.join(dirpath, d), sd), "image.jpg")
      imgList.append((skimage.io.imread(imPath),os.path.join(os.path.join(dirpath, d), sd)))

      if len(imgList) == k:
        save_images(model, imgList)
        imgList = []

  if len(imgList) > 0:
    model = construct_model(len(imgList))
    save_images(model, imgList)
    imgList = []

# construct_model constructs a model and returns it.
def construct_model(imagesPerGPU):
  class InferenceConfig(FaceConfig):
      # Set batch size to 1 since we'll be running inference on
      # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
      GPU_COUNT = 1
      IMAGES_PER_GPU = imagesPerGPU

  config = InferenceConfig()

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
  return model

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
  start_time = time.time()

  # Parse command line arguments
  parser = argparse.ArgumentParser(
      description='Train Mask R-CNN to detect balloons.')
  parser.add_argument('--image', required=False,
                      metavar="path or URL to image",
                      help='Image to apply the color splash effect on')
  args = parser.parse_args()

  print("Dataset: ", DATASET_DIR)
  print("Logs: ", CHECKPOINT_DIR)

  detect_start_time = time.time()
  # Run inference.
  if args.image != "":
    detect_face()
  else:
    detect_on_aws()

  print ("Total detection time taken: ", time.time() - detect_start_time)
