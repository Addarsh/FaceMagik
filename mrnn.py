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
import boto3
import socket
import keras
import requests
import shutil
from PIL import Image

if socket.gethostname() == "Addarshs-MacBook-Pro.local":
  volume_mount_dir = "/Users/addarsh/virtualenvs/aws_train/dltraining"
else:
  volume_mount_dir = '/dltraining/'

# Root directory of the project
ROOT_DIR = "Mask_RCNN"
DATASET_DIR = os.path.join(volume_mount_dir, "dataset")
CHECKPOINT_DIR = os.path.join(volume_mount_dir, "checkpoints")
SPLIT_FILE = os.path.join(DATASET_DIR, "data_split.json")
# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.abspath(os.path.join(volume_mount_dir, "mask_rcnn_coco.h5"))

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Dictionary of labels for training.
class_dict = {
  "Eye (Open)": 1, "Eyeball": 2, "Eyebrow": 3, "Reading glasses": 4, "Sunglasses": 5, "Eye (Closed)": 6,
  "Nose": 7, "Nostril": 8, "Upper Lip":9, "Lower Lip":10, "Teeth":11, "Tongue": 12, "Facial Hair":13,
  "Face": 14, "Hair (on head)": 15, "Head Gear": 16, "Bald Head": 17, "Ear": 18
}

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
    GPU_COUNT = 4
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 18 + 1  # Background + class_dict

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    # Remove mini mask to improve accuracy.
    USE_MINI_MASK = False

    # Use Resnet50 for faster training.
    BACKBONE = "resnet50"


############################################################
#  Dataset
############################################################

class FaceDataset(utils.Dataset):

    def load_faces(self, subset):
        """Load a subset of the Balloon dataset.
        subset: Subset to load: train or val
        """
        source = "face"
        # Add classes.
        for c in class_dict:
          self.add_class(source, class_dict[c], c)

        # Train or validation dataset?
        assert subset in ["train", "val"]

        # Load split annotations.
        if not os.path.exists(SPLIT_FILE):
          print ("Split file does not exist, downloading dataset...")
          download_dataset()
          print ("Download complete")
        split_ann = []
        with open(SPLIT_FILE, "r") as g:
          split_ann = json.load(g)
        want_set = set(split_ann[subset])

        img_count = 0
        annotation_count = 0
        for i in range(1,10):
          for j in range(1,9):
            base_dir = os.path.join("helen_r"+str(i),"50_"+str(j))
            if not os.path.exists(os.path.join(DATASET_DIR, base_dir)):
              continue
            for f in os.listdir(os.path.join(DATASET_DIR, base_dir)):
              if not f.endswith(".jpg"):
                continue
              if os.path.join(base_dir, f) not in want_set:
                continue
              img_path = os.path.join(DATASET_DIR, os.path.join(base_dir, f))
              print ("Adding image path: ",  img_path)
              img_count += 1

              # Get image annotations.
              polygons = []
              merged_dir = os.path.join(DATASET_DIR, os.path.join("merged", base_dir))
              for c in ["eye", "nose", "face"]:
                ann_path = os.path.join(os.path.join(merged_dir, c), os.path.splitext(f)[0]+".json" )
                ann = json.load(open(ann_path))
                for d in ann:
                  if d["class"] not in class_dict:
                    continue
                  polygons.append(d)
                  annotation_count += 1

              # Get image width and height.
              im = Image.open(img_path)
              width, height = im.size

              self.add_image(
                source=source,
                image_id=f, # filename used for annotation.
                path=img_path,
                width=width,
                height=height,
                polygons=polygons,
              )
        print ("subset: ", subset)
        print ("Image count: ", img_count)
        print ("Annotation count: ", annotation_count)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a face dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "face":
          raise Exception("source is not face in load_mask")

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        class_ids = np.ones(len(info["polygons"]), dtype=np.int32)
        for i, c in enumerate(info["polygons"]):
          # Get indices of pixels inside the polygon and set them to 1
          for points in c["data"]:
            x = [p[0] for p in points]
            y = [p[1] for p in points]

            rr, cc = skimage.draw.polygon(y, x, shape=(info["height"] , info["width"]))
            mask[rr, cc, i] = 1

          class_ids[i] = class_dict[c["class"]]

        # Return mask, and array of class IDs of each instance.
        return mask.astype(np.bool), class_ids

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
    dataset_train.load_faces("train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = FaceDataset()
    dataset_val.load_faces("val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    callback = None if socket.gethostname() == "Addarshs-MacBook-Pro.local" else SpotTermination()
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=60,
                layers='all', custom_callbacks=[callback])

def color_splash(image, mask, class_ids):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    count = 0
    m = np.zeros((image.shape[0], image.shape[1], len(class_ids)))
    for i, c in enumerate(class_ids):
      if c== 14 or c==15 or c==18 or c==17:
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
download_dataset will fetch the face annotation dataset from Amazon S3 bucket.
This is in the event that the new instance is assigned a different zone
than the elastic block.
"""
def download_dataset():
  s3 = boto3.resource('s3')
  bucket_name = "addarsh-face-segment"

  bucket = s3.Bucket(bucket_name)

  remote_merged_dir = "merged"
  if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

  # Download the split file.
  split_file = "data_split.json"
  for object_summary in bucket.objects.filter(Prefix=split_file):
    bucket.download_file(object_summary.key, os.path.join(DATASET_DIR, object_summary.key))

  for i in range(1, 10):
    for j in range(1, 9):
      download_img_and_annotation(os.path.join("helen_r"+str(i),"50_"+str(j)), remote_merged_dir, bucket)

"""
download_img_and_annotation downloads images in given directory and corresponding merged annotations.
"""
def download_img_and_annotation(remote_dir, remote_merged_dir, bucket):
  # Download image file.
  for object_summary in bucket.objects.filter(Prefix=remote_dir+"/"):
    if not os.path.split(object_summary.key)[1].endswith(".jpg"):
      continue
    if os.path.exists(os.path.join(DATASET_DIR,object_summary.key)):
      continue
    local_dir = os.path.join(DATASET_DIR, remote_dir)
    if not os.path.exists(local_dir):
      os.makedirs(local_dir)
    bucket.download_file(object_summary.key, os.path.join(DATASET_DIR,object_summary.key))

  # Download associated merged annotation files.
  for c in ["eye", "nose", "face"]:
    for object_summary in bucket.objects.filter(Prefix=os.path.join(os.path.join(remote_merged_dir, remote_dir), c)):
      if not os.path.split(object_summary.key)[1].endswith(".json"):
        continue
      if os.path.exists(os.path.join(DATASET_DIR,object_summary.key)):
        continue
      local_merged_dir = os.path.join(os.path.join(DATASET_DIR, remote_merged_dir), remote_dir)
      for c in ["eye", "nose", "face"]:
        if not os.path.exists(os.path.join(local_merged_dir,c)):
          os.makedirs(os.path.join(local_merged_dir,c))
      bucket.download_file(object_summary.key, os.path.join(DATASET_DIR,object_summary.key))


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

"""
epoch_path returns the latest epoch file.
"""
def epoch_path(list_of_checkpoint_files):
  max_epoch_number = 1
  max_epoch_file = ""
  for f in list_of_checkpoint_files:
    if not f.startswith("mask_rcnn_face"):
      continue
    a = f.split(".")[0]
    num = int(a.split("_")[-1])
    if num > max_epoch_number:
      max_epoch_number = num
      max_epoch_file = f
  if max_epoch_file == "":
    raise Exception("Max epoch file cannot be empty")
  return max_epoch_file

"""
Spot termination class checks if spot instance is going to be terminated
every 5 seconds and stops training if thats the case.
"""
class SpotTermination(keras.callbacks.Callback):
  def on_epoch_begin(self, epoch, logs=None):
    print ("Starting to train on epoch number: ", epoch)
    if epoch <= 1:
      return
    all_dirs = sorted([f for f in os.listdir(CHECKPOINT_DIR) if os.path.isdir(os.path.join(CHECKPOINT_DIR, f)) and f.startswith("face")])
    for i in range(len(all_dirs)-1):
      os.remove(os.path.join(CHECKPOINT_DIR, all_dirs[i]))
    latest_dir = all_dirs[-1]

    # Delete epoch -1 file to save space.
    path = "mask_rcnn_face_{:04d}.h5".format(epoch-1)
    if os.path.exists(os.path.join(os.path.join(os.path.join(CHECKPOINT_DIR, latest_dir), path))):
      os.remove(os.path.join(os.path.join(os.path.join(CHECKPOINT_DIR, latest_dir), path)))

  def on_batch_begin(self, batch, logs={}):
    status_code = requests.get("http://169.254.169.254/latest/meta-data/spot/instance-action").status_code
    if status_code != 404:
      # Stop training.
      self.model.stop_training = True

if __name__ == '__main__':
  # Parse command line arguments
  parser = argparse.ArgumentParser(
      description='Train Mask R-CNN to detect balloons.')
  parser.add_argument("--command",
                      metavar="<command>",
                      help="'train' or 'detect'")
  parser.add_argument('--image', required=False,
                      metavar="path or URL to image",
                      help='Image to apply the color splash effect on')
  args = parser.parse_args()

  # Validate arguments
  if args.command == "train":
    # Download dataset if it does not exist.
    if not os.path.exists(DATASET_DIR):
      print ("Dataset does not exist, downloading....")
      download_dataset()
      print ("Dataset download complete")
  elif args.command == "detect":
      assert args.image,\
             "Provide --image or --video to apply color splash"

  print("Dataset: ", DATASET_DIR)
  print("Logs: ", CHECKPOINT_DIR)

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
                                model_dir=CHECKPOINT_DIR)
  else:
      model = modellib.MaskRCNN(mode="inference", config=config,
                                model_dir=CHECKPOINT_DIR)

  # Select weights file to load
  weights_path = ""
  try:
    weights_path = model.find_last()
    print ("Using last iteration weights")
  except Exception as e:
    # Load Coco weights.
    weights_path = COCO_WEIGHTS_PATH
    # Download weights file
    if not os.path.exists(weights_path):
      print ("Downloading Coco weights...")
      utils.download_trained_weights(weights_path)
    print ("Using Coco weights")

  # Load weights
  print("Loading weights ", weights_path)
  if weights_path == COCO_WEIGHTS_PATH:
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
    if socket.gethostname() != "Addarshs-MacBook-Pro.local":
      # Backup terminal output once training is complete
      shutil.copy2('/var/log/cloud-init-output.log', os.path.join(volume_mount_dir,
                                                                'cloud-init-output-{}.log'.format(datetime.datetime.today().strftime('%Y-%m-%d'))))
  elif args.command == "detect":
    detect(model, args.image)
