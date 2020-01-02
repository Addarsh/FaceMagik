import os
import sys
import numpy as np
import argparse
import json
import cv2
import imageio
import time
import random
import h5py
import skimage
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
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
from collections import Counter

CLASS = "class"
DATA = "data"

IMAGE_DIR = "output"
ANNOTATIONS_DIR = os.path.join(IMAGE_DIR, "annotations")

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

# shadow_masks constructs shadows from given input points
# and applies them to input mask.
def shadow_masks(m, image, maskPts):
  pts1, pts2 = MathUtils.segregate_points(image, maskPts)
  pts1_mean, pts2_mean = ImageUtils.avg_intensity(image, pts1), ImageUtils.avg_intensity(image, pts2)

  darker_pts = pts1 if pts1_mean[0] < pts2_mean[0] else pts2
  lighter_pts = pts1 if pts1_mean[0] >= pts2_mean[0] else pts2
  d = pts1_mean if pts1_mean[0] < pts2_mean[0] else pts2_mean
  l = pts1_mean if pts1_mean[0] >= pts2_mean[0] else pts2_mean

  darkMask = newMask(m.shape[:2], darker_pts)
  lightMask = newMask(m.shape[:2], lighter_pts)

  m = np.where(darkMask, d, m).astype(np.uint8)
  m = np.where(lightMask, l, m).astype(np.uint8)

  return m, darker_pts, lighter_pts

""""
segregate_points_rgb segregates given mask points in given image
into two clusters of dominant colors. The two colors are applied
to the given mask at the calculated points as well. The image
is in RGB space.
"""
def segregate_points_rgb(m, image, maskPts):
  pts1, pts2 = MathUtils.segregate_points(image, maskPts)
  gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
  pts1_darkness, pts2_darkness = ImageUtils.avg_intensity(gray, pts1), ImageUtils.avg_intensity(gray, pts2)
  pts1_mean, pts2_mean = ImageUtils.avg_color(image, pts1), ImageUtils.avg_color(image, pts2)

  darker_pts = pts1 if pts1_darkness[0] < pts2_darkness[0] else pts2
  lighter_pts = pts1 if pts1_darkness[0] >= pts2_darkness[0] else pts2
  d = pts1_mean if pts1_darkness[0] < pts2_darkness[0] else pts2_mean
  l = pts1_mean if pts1_darkness[0] >= pts2_darkness[0] else pts2_mean

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

"""
detect_strictly_face is responsible for detecting face segmentation
pixels excluding eyes, eyebrows, lips and teeth.
"""
def detect_strictly_face():
  model = construct_model(1)

  print("Running on {}".format(args.image))
  # Read image
  image = skimage.io.imread(args.image)
  # Detect objects
  preds = model.detect([image], verbose=1)[0]

  print ("len mask: ", len(preds["masks"]))

  facePts = set()
  for i, class_id in enumerate(preds["class_ids"]):
    if id_label_map[class_id] != FACE:
      continue
    facePts = set(maskPoints(preds["masks"][:, :, i]))
    break

  if len(facePts) == 0:
    raise Exception("Face not found in image")

  for i, class_id in enumerate(preds["class_ids"]):
    if id_label_map[class_id] == EYE_OPEN or id_label_map[class_id] == EYE_CLOSED \
      or id_label_map[class_id] == NOSTRIL or id_label_map[class_id] == TEETH \
      or id_label_map[class_id] == TONGUE or id_label_map[class_id] == UPPER_LIP \
      or id_label_map[class_id] == LOWER_LIP or id_label_map[class_id] == SUNGLASSES \
      or id_label_map[class_id] == EYEBROW:
      facePts -= set(maskPoints(preds["masks"][:, :, i]))

  facePts = list(facePts)

  # Segregate points.
  m, dpts, lpts = segregate_points_rgb(np.copy(image), image, facePts)
  m, ldpts, _ = segregate_points_rgb(m, image, lpts)
  m , _, dlpts = segregate_points_rgb(m, image, dpts)

  print ("average color: ", ImageUtils.avg_color(image, ldpts+dlpts))


  # Save output
  skimage.io.imsave(os.path.join(IMAGE_DIR, os.path.split(args.image)[1]), m)

"""
get_face_mask is a helper function to retrieve face mask numpy array minus the eyes, nose,
ear and mouth points. It checks to see if the annotation already exists and returns
if it does. If not, it calculates the face mask and saves it.
"""
def get_face_mask():
  # Read image
  image = skimage.io.imread(args.image)

  dirpath = os.path.join(os.path.join(IMAGE_DIR, os.path.splitext(os.path.split(args.image)[1])[0]), "annotations")

  if os.path.exists(os.path.join(dirpath, "face_mask.hdf5")):
    with h5py.File(os.path.join(dirpath, "face_mask.hdf5"), 'r') as f:
      return f["face_mask"][:]

  model = construct_model(1)

  print("Running on {}".format(args.image))
  # Detect objects
  preds = model.detect([image], verbose=1)[0]

  faceMask = np.zeros(image.shape[:2], dtype=bool)
  faceFound = False
  for i, class_id in enumerate(preds["class_ids"]):
    if id_label_map[class_id] == EYE_OPEN or id_label_map[class_id] == EYE_CLOSED \
      or id_label_map[class_id] == NOSTRIL or id_label_map[class_id] == TEETH \
      or id_label_map[class_id] == TONGUE or id_label_map[class_id] == UPPER_LIP \
      or id_label_map[class_id] == LOWER_LIP or id_label_map[class_id] == SUNGLASSES \
      or id_label_map[class_id] == EYEBROW or id_label_map[class_id] == FACE:
      faceMask = np.bitwise_xor(faceMask, preds["masks"][:, :, i])
      if id_label_map[class_id] == FACE:
        faceFound = True

  if not faceFound:
    raise Exception("Face not found in image")

  if not os.path.exists(dirpath):
    os.makedirs(dirpath)

  with h5py.File(os.path.join(dirpath, "face_mask.hdf5"), 'w') as f:
    dset = f.create_dataset("face_mask", faceMask.shape, dtype=bool, chunks=True, compression="gzip")
    dset[...] = faceMask

  return faceMask


"""
detect dominant colors uses K means clustering to find the top K colors
in the face.
"""
def detect_dominant_colors(K=1):
  faceArr = get_face_arr()

  image = skimage.io.imread(args.image)

  clusterArray = np.zeros((len(facePts), 3))
  for i, p in enumerate(facePts):
    clusterArray[i] = image[p[0], p[1]]

  # applyu kmeans.
  clf = KMeans(n_clusters=K)
  labels = clf.fit_predict(clusterArray)

  counts = Counter(labels)
  center_colors = clf.cluster_centers_


  ordered_colors = [center_colors[i] for i in counts.keys()]
  hex_colors = [ImageUtils.RGB2HEX(ordered_colors[i]) for i in counts.keys()]
  rgb_colors = [ordered_colors[i] for i in counts.keys()]

  plt.figure(figsize=(8,6))
  plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)
  plt.savefig(os.path.join(IMAGE_DIR, os.path.splitext(os.path.split(args.image)[1])[0] + "_pie_" + str(K) + ".jpg"))

"""
mean_shift_clustering calculates the clusters in the face image based on mean shift
algorithm.
"""
def mean_shift_clustering():
  # Read image
  image = skimage.io.imread(args.image)

  faceMask = get_face_mask()

  flatfaceRGB = image[faceMask]

  bandwidth = estimate_bandwidth(flatfaceRGB, quantile=0.2, n_samples=500)
  ms = MeanShift(bandwidth, bin_seeding=True)
  ms.fit(flatfaceRGB)
  labels = ms.labels_
  cluster_centers = ms.cluster_centers_
  print ("Number of clusters: ", len(cluster_centers))

  bins = np.bincount(labels)
  randomColors = np.random.choice(256, size=(bins.shape[0], 3))
  image[faceMask] = randomColors[labels]

  print ("dominant color: ", ImageUtils.RGB2HEX(cluster_centers[np.argmax(bins)]))

  plt.imshow(image)
  plt.show()

"""
foundation_color uses mean shift algorithm to calculate the average foundation
color.
"""
def foundation_color():
  # Read image
  image = skimage.io.imread(args.image)

  flat_image= np.reshape(image, (-1, 3))

  bandwidth = estimate_bandwidth(flat_image, quantile=0.4, n_samples=500)
  ms = MeanShift(bandwidth, bin_seeding=True)
  ms.fit(flat_image)
  labels = ms.labels_
  cluster_centers = ms.cluster_centers_
  print ("Number of clusters: ", len(cluster_centers))

  bins = np.bincount(labels)
  randomColors = np.random.choice(256, size=(bins.shape[0], 3))
  flat_image = np.reshape(cluster_centers[labels], image.shape).astype(int)

  print ("dominant color: ", ImageUtils.RGB2HEX(cluster_centers[np.argmax(bins)]))

  plt.imshow(flat_image)
  plt.show()


"""
get_reflectance_matrix returns the reflectance matrix for given RGB numpy array.
Input is (n,3) and output is (n, 36).
"""
def get_reflectance_matrix(rgbArr):
  with h5py.File("reflectance.hdf5", "r") as f:
    flatRGBArr = rgbArr[:, 0]*256*256 + rgbArr[:, 1]*256 + rgbArr[:, 2]
    uniqueRGB = np.unique(flatRGBArr)
    R_unique = f["reflectance"][uniqueRGB]
    sorted_idx = np.searchsorted(uniqueRGB, flatRGBArr)
    return R_unique[sorted_idx]

"""
apply_foundation applies foundation on given image face with given
hex color and given blend ratio.
"""
def apply_foundation():
  image = skimage.io.imread(args.image)
  faceMask = get_face_mask()

  dirpath = os.path.join(os.path.join(IMAGE_DIR, os.path.splitext(os.path.split(args.image)[1])[0]), "annotations")
  if os.path.exists(os.path.join(dirpath, "R_skin.hdf5")):
    with h5py.File(os.path.join(dirpath, "R_skin.hdf5"), "r") as f:
      R_skin = f["reflectance"][:]
      R_foundation = get_reflectance_matrix(np.array([ImageUtils.HEX2RGB(args.color)]))
  else:
    R_matrix = get_reflectance_matrix(np.concatenate((image[faceMask], [ImageUtils.HEX2RGB(args.color)])))
    R_skin = R_matrix[:-1]
    R_foundation = R_matrix[-1:]
    # Save to file.
    with h5py.File(os.path.join(dirpath, "R_skin.hdf5"), "w") as f:
      dset = f.create_dataset("reflectance", R_skin.shape, dtype='f', chunks=True, compression="gzip")
      dset[...] = R_skin

  # Create video and save it.
  foundpath = os.path.join(os.path.join(IMAGE_DIR, os.path.splitext(os.path.split(args.image)[1])[0]), args.color)
  if not os.path.exists(foundpath):
    os.makedirs(foundpath)

  all_images = []
  for alpha in np.linspace(0.9, 0.5, 10).tolist():
    image_copy = image.copy()
    image_copy[faceMask] = ImageUtils.add_gamma_correction_matrix(np.transpose(np.matmul(ImageUtils.read_T_matrix(), np.transpose(np.power(R_skin, alpha)*np.power(R_foundation, 1-alpha)))))
    all_images.append(image_copy)

  imageio.mimwrite(os.path.join(foundpath, "output.mp4"), np.array(all_images) , fps = 1.0)

"""
apply_foundation_depth applies foundation on given image face with given
hex color using a mixing formula proposed in the following paper:
https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Li_Simulating_Makeup_Through_2015_CVPR_paper.pdf
"""
def apply_foundation_depth():
  image = skimage.io.imread(args.image)
  faceMask = get_face_mask()

  dirpath = os.path.join(os.path.join(IMAGE_DIR, os.path.splitext(os.path.split(args.image)[1])[0]), "annotations")
  if os.path.exists(os.path.join(dirpath, "R_skin.hdf5")):
    with h5py.File(os.path.join(dirpath, "R_skin.hdf5"), "r") as f:
      R_skin = f["reflectance"][:]
      R_foundation = get_reflectance_matrix(np.array([ImageUtils.HEX2RGB(args.color)]))
  else:
    R_matrix = get_reflectance_matrix(np.concatenate((image[faceMask], [ImageUtils.HEX2RGB(args.color)])))
    R_skin = R_matrix[:-1]
    R_foundation = R_matrix[-1:]
    # Save to file.
    with h5py.File(os.path.join(dirpath, "R_skin.hdf5"), "w") as f:
      dset = f.create_dataset("reflectance", R_skin.shape, dtype='f', chunks=True, compression="gzip")
      dset[...] = R_skin

  # Create video and save it.
  foundpath = os.path.join(os.path.join(IMAGE_DIR, os.path.splitext(os.path.split(args.image)[1])[0]), args.color)
  if not os.path.exists(foundpath):
    os.makedirs(foundpath)

  all_images = []
  for alpha in np.linspace(0.01, 0.1, 20).tolist():
    image_copy = image.copy()

    R_found_updated, T_found_updated = ImageUtils.R_foundation_depth(R_foundation, alpha)
    R_mixed = R_found_updated + (np.power(T_found_updated, 2)*R_skin)/(1-R_found_updated*R_skin)
    image_copy[faceMask] = ImageUtils.add_gamma_correction_matrix(np.transpose(np.matmul(ImageUtils.read_T_matrix(), np.transpose(R_mixed))))
    all_images.append(image_copy)

  imageio.mimwrite(os.path.join(foundpath, "output_paper.mp4"), np.array(all_images) , fps = 1.0)


"""
hsv is responsible for converting RGB image to HSV and writing it.
"""
def hsv():
  # Read image
  image = skimage.io.imread(args.image)
  hsv_img = skimage.color.rgb2hsv(image)

  hue_img = hsv_img[:, :, 1]

  # Save output
  skimage.io.imsave(os.path.join(IMAGE_DIR, os.path.split(args.image)[1]), hue_img)


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

def change_ref():
  with h5py.File("reflectance.hdf5", "r") as f:
    with h5py.File("c_reflectance.hdf5", "w") as g:
      dset = g.create_dataset("reflectance", (256*256*256, 36), dtype='f', chunks=True, compression="gzip")
      dset[...] = f["reflectance"][:]

if __name__ == '__main__':
  start_time = time.time()

  # Parse command line arguments
  parser = argparse.ArgumentParser(
      description='Train Mask R-CNN to detect balloons.')
  parser.add_argument('--image', required=False,
                      metavar="path or URL to image",
                      help='Image to apply the color splash effect on')
  parser.add_argument('--color', required=False,
                      metavar="path or URL to image",
                      help='Image to apply the color splash effect on')
  args = parser.parse_args()

  print("Dataset: ", DATASET_DIR)
  print("Logs: ", CHECKPOINT_DIR)

  detect_start_time = time.time()
  # Run inference.
  if args.image != "":
    #hsv()
    #detect_dominant_colors()
    #apply_foundation()
    #apply_foundation_depth()
    #foundation_color()
    #mean_shift_clustering()
    detect_strictly_face()
    #get_face_arr()
    #change_ref()
  else:
    detect_on_aws()

  print ("Total detection time taken: ", time.time() - detect_start_time)
