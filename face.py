"""
This file implements a Face class that abstracts
some of the operations around face detection, light estimation
and albedo estimation of a given image.
"""
import os
import numpy as np
import cv2
import h5py
import argparse
from image_utils import ImageUtils
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

# Parse command line arguments
parser = argparse.ArgumentParser(description='Image path')
parser.add_argument('--image', required=False,metavar="path or URL to image")
args = parser.parse_args()

class Face:
  def __init__(self, imagePath):
    self.outputDir = "output"
    self.imagePath = imagePath
    self.faceMaskDirPath =  os.path.join(os.path.join(self.outputDir, os.path.splitext(os.path.split(imagePath)[1])[0]), "annotations")
    self.hdf5File = "face_mask.hdf5"
    self.CLASS_IDS_KEY = "class_ids"
    self.MASKS_KEY = "masks"
    self.id_label_map = {v: k for k, v in label_id_map.items()}

    self.image = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2RGB)
    self.preds = self.detect_face()

    self.windowName = imagePath
    self.windowSize = 900

  """
  show_mask will display given mask.
  """
  def show_mask(self, mask):
    clone = self.image.copy()
    dims = clone[mask].shape
    clone[mask] = np.array([0, 255, 0])
    self.show(clone)

  """
  show_orig_image shows the original RGB image without modifications.
  """
  def show_orig_image(self):
    return self.show(self.image)

  """
  show is an internal helper function to display given RGB image.
  """
  def show(self, image):
    cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(self.windowName, self.windowSize, self.windowSize)
    cv2.imshow(self.windowName, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)

  """
  detect_face will detect face in the given image and segment out eyes,
  eyebrows, nose, lips etc using a MaskRCNN network. It returns a predictions
  dictionary.
  """
  def detect_face(self):
    if os.path.exists(os.path.join(self.faceMaskDirPath, self.hdf5File)):
      preds = {self.CLASS_IDS_KEY: []}
      with h5py.File(os.path.join(self.faceMaskDirPath, self.hdf5File), 'r') as f:
        preds[self.MASKS_KEY] = np.zeros((self.image.shape[0], self.image.shape[1], len(f.keys())), dtype=bool)
        for i, class_id in enumerate(f.keys()):
          preds[self.CLASS_IDS_KEY].append(int(class_id.split("-")[0]))
          preds[self.MASKS_KEY][:, :, i] = f[class_id][:]
      return preds

    print("Running on {}".format(self.imagePath))

    # Detect face.
    model = self.construct_model()
    preds = model.detect([self.image], verbose=1)[0]

    if not os.path.exists(self.faceMaskDirPath):
      os.makedirs(self.faceMaskDirPath)

    with h5py.File(os.path.join(self.faceMaskDirPath, self.hdf5File), 'w') as f:
      for i, class_id in enumerate(preds[self.CLASS_IDS_KEY]):
        mask = preds[self.MASKS_KEY][:, :, i]
        dset = f.create_dataset(str(class_id)+ "-" + str(i), mask.shape, dtype=bool, chunks=True, compression="gzip")
        dset[...] = mask
    return preds

  """
  get_face_mask returns face mask numpy array from stored dictionary of face predictions.
  """
  def get_face_mask(self):
    faceMask = np.zeros(self.image.shape[:2], dtype=bool)
    faceFound = False
    for i, class_id in enumerate(self.preds[self.CLASS_IDS_KEY]):
      if self.id_label_map[class_id] == EYE_OPEN or self.id_label_map[class_id] == EYE_CLOSED \
        or self.id_label_map[class_id] == NOSTRIL or self.id_label_map[class_id] == TEETH \
        or self.id_label_map[class_id] == TONGUE or self.id_label_map[class_id] == UPPER_LIP \
        or self.id_label_map[class_id] == LOWER_LIP or self.id_label_map[class_id] == SUNGLASSES \
        or self.id_label_map[class_id] == EYEBROW or self.id_label_map[class_id] == FACE:
        faceMask = np.bitwise_xor(faceMask, self.preds[self.MASKS_KEY][:, :, i])
        if self.id_label_map[class_id] == FACE:
          if faceFound:
            raise Exception("Only 1 face allowed per image")
          faceFound = True
    if not faceFound:
      raise Exception("Face not found in image")
    return faceMask

  """
  get_face_keypoints returns keypoints of the face that are best representatives
  of skin tone.
  """
  def get_face_keypoints(self):
    keypoints = np.zeros(self.image.shape[:2], dtype=bool)
    for m in [self.get_forehead_keypoints(), self.get_left_cheek_keypoints(),  self.get_right_cheek_keypoints(), self.get_nose_keypoints()]:
      keypoints = np.bitwise_xor(keypoints, m)
    return keypoints

  """
  get_forehead_keypoints returns keypoints of the forehead that are best representatives
  of skin tone on the face.
  """
  def get_forehead_keypoints(self):
    eyebrow_masks = self.get_attr_masks(EYEBROW)
    face_mask = self.get_attr_masks(FACE)[0]
    assert len(eyebrow_masks) == 2, "Want 2 masks for eyebrow!"
    (left_eyebrow, right_eyebrow) = (eyebrow_masks[0], eyebrow_masks[1]) if self.bbox(eyebrow_masks[0])[1] <= self.bbox(eyebrow_masks[1])[1] else (eyebrow_masks[1], eyebrow_masks[0])
    left_row_min, left_col_min, left_width, left_height = self.bbox(left_eyebrow)
    right_row_min, right_col_min, right_width, right_height = self.bbox(right_eyebrow)

    left_col = left_col_min + int(left_width/2)
    right_col = right_col_min + int(right_width/2)
    row_max = min(left_row_min, right_row_min)
    row_min = max(np.nonzero(face_mask[:, left_col])[0][0], np.nonzero(face_mask[:, right_col])[0][0])

    keypoints = np.zeros(self.image.shape[:2], dtype=bool)
    keypoints[row_min:row_max, left_col:right_col] = 1
    return keypoints

  """
  get_left_cheek_keypoints returns keypoints from left cheek that are best representatives of skin
  tone on the face.
  """
  def get_left_cheek_keypoints(self):
    eye_masks = self.get_attr_masks(EYE_OPEN)
    face_mask = self.get_attr_masks(FACE)[0]
    nose_mask = self.get_attr_masks(NOSE)
    assert len(eye_masks) == 2, "Want 2 masks for eye open!"
    assert len(nose_mask) == 1, "Want 1 mask for nose!"

    nose_mask = nose_mask[0]
    left_eye_mask = eye_masks[0] if self.bbox(eye_masks[0])[1] <= self.bbox(eye_masks[1])[1] else eye_masks[1]
    nose_row_min, nose_col_min, nose_width, nose_height = self.bbox(nose_mask)
    left_eye_row_min, left_eye_col_min, left_eye_width, left_eye_height = self.bbox(left_eye_mask)

    row_min = left_eye_row_min+left_eye_height
    row_max = nose_row_min+nose_height
    col_min = max(np.nonzero(face_mask[row_min, :])[0][0], np.nonzero(face_mask[row_max, :])[0][0])
    col_max = min(nose_col_min, left_eye_col_min+left_eye_width)
    height = row_max - row_min + 1
    row_min = row_min + int(height/4)
    row_max = row_max - int(height/4)

    keypoints = np.zeros(self.image.shape[:2], dtype=bool)
    keypoints[row_min:row_max, col_min:col_max] = 1
    return keypoints

  """
  get_right_cheek_keypoints returns keypoints from right cheek that are best representatives of skin
  tone on the face.
  """
  def get_right_cheek_keypoints(self):
    eye_masks = self.get_attr_masks(EYE_OPEN)
    face_mask = self.get_attr_masks(FACE)[0]
    nose_mask = self.get_attr_masks(NOSE)
    assert len(eye_masks) == 2, "Want 2 masks for eye open!"
    assert len(nose_mask) == 1, "Want 1 mask for nose!"

    nose_mask = nose_mask[0]
    right_eye_mask = eye_masks[0] if self.bbox(eye_masks[0])[1] >= self.bbox(eye_masks[1])[1] else eye_masks[1]
    nose_row_min, nose_col_min, nose_width, nose_height = self.bbox(nose_mask)
    right_eye_row_min, right_eye_col_min, right_eye_width, right_eye_height = self.bbox(right_eye_mask)

    row_min = right_eye_row_min+right_eye_height
    row_max = nose_row_min+nose_height
    col_min = max(nose_col_min + nose_width, right_eye_col_min)
    col_max = min(np.nonzero(face_mask[row_min, :])[0][-1], np.nonzero(face_mask[row_max, :])[0][-1])
    height = row_max - row_min + 1
    row_min = row_min + int(height/4)
    row_max = row_max - int(height/4)

    keypoints = np.zeros(self.image.shape[:2], dtype=bool)
    keypoints[row_min:row_max, col_min:col_max] = 1
    return keypoints

  """
  get_nose_keypoints returns keypoints from nose that are best representatives of skin
  tone on the face.
  """
  def get_nose_keypoints(self):
    nose_mask = self.get_attr_masks(NOSE)
    nostril_masks = self.get_attr_masks(NOSTRIL)
    assert len(nose_mask) == 1, "Want 1 mask for nose!"
    row_max = -1
    for mask in nostril_masks:
      row, _, _, _ = self.bbox(mask)
      if row_max == -1 or row < row_max:
        row_max = row
    if row_max == -1:
      return nose_mask[0]
    nose_mask = nose_mask[0]
    nose_mask[row_max:, :] = 0
    return nose_mask

  """
  bbox returns bounding box (x1, y1, w, h) (top left point, width and height) of given mask.
  """
  def bbox(self, mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, cmin, cmax-cmin+1, rmax-rmin+1

  """
  get_custom_mask will return a mask of points with given attributes.
  """
  def get_custom_mask(self, attrs):
    allMasks = []
    for attr in attrs:
      allMasks += self.get_attr_masks(attr)
    if len(allMasks) == 0:
      raise Exception("No mask found for any attribute!")
    mask = np.zeros(self.image.shape[:2], dtype=bool)
    for m in allMasks:
      mask = np.bitwise_xor(mask, m)
    return mask

  """
  get_attr_masks returns a list of masks for given attribute. Attribute can be one of
  EYE_OPEN, NOSE, UPPER_LIP etc. It will return an empty list for any attribute
  that doesn't exist in the face.
  """
  def get_attr_masks(self, attr):
    masks = []
    for i, class_id in enumerate(self.preds[self.CLASS_IDS_KEY]):
      if label_id_map[attr] == class_id:
        masks.append(self.preds[self.MASKS_KEY][:, :, i])
    return masks

  """
  construct_model constructs a MaskRCNN model and returns it.
  """
  def construct_model(self, imagesPerGPU=1):
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

    print("Loading weights from: ", weights_path)
    model.load_weights(weights_path, by_name=True)

    return model

if __name__ == "__main__":
  f = Face(args.image)
  f.show_mask(f.get_face_keypoints())
