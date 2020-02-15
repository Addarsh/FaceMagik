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
import json
from scipy import ndimage
from scipy.optimize import minimize
from image_utils import ImageUtils
from sklearn.cluster import KMeans
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
    self.clf = KMeans(n_clusters=2)
    self.preds = self.detect_face()
    self.faceMask = self.get_face_mask()
    self.beta = self.specularity()
    if imagePath.startswith("server/data"):
      # Load lighting, normals and face vertices.
      dirPath, _ = os.path.split(imagePath)
      with open(os.path.join(dirPath, "face_vertices.json"), "r") as f:
        self.faceVertices = json.load(f)
      with open(os.path.join(dirPath, "normals.json"), "r") as f:
        self.normals = np.array(json.load(f))
      with open(os.path.join(dirPath, "lighting.json"), "r") as f:
        self.lighting = json.load(f)
      with open(os.path.join(dirPath, "triangle_indices.json"), "r") as f:
        self.triangleIndices = json.load(f)

      # Constants array from Ramamoorthi's paper.
      self.c = np.array([0.429043, 0.511664, 0.743125, 0.886227, 0.247708])

    self.windowName = imagePath
    self.windowSize = 900

  """
  show_mask will display given mask.
  """
  def show_mask(self, mask):
    clone = self.image.copy()
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
  specularity returns the 2D specularity array as shown in Shen et. al. 2009.
  """
  def specularity(self):
    Vmin = np.amin(self.image, axis=-1)
    mu = np.mean(Vmin)
    sigma = np.std(Vmin)
    eta = 0.5
    Tv = mu + eta*sigma
    tau = Vmin.copy()
    tau = np.where(tau > Tv, Tv, tau)
    return Vmin - tau

  """
  compute_specular_masks computes and stores all specular regions on the face  as the dominant
  specular region (based on maximum mean specularity).
  It is 3D array of shape (IMG_WIDTH, IMG_HEIGHT, num regions).
  """
  def compute_specular_masks(self):
    bImg = np.bitwise_and(self.beta != 0, self.faceMask)

    # Label all specular clusters.
    label_im, nb_labels = ndimage.label(bImg)
    sizes = ndimage.sum(bImg, label_im, range(nb_labels + 1))
    indices = np.transpose(np.argwhere(sizes > int(len(np.nonzero(self.faceMask)[0])/1000.0)))

    self.allMasks = np.repeat(label_im[:, :, np.newaxis], len(indices), axis=2) == indices
    domMask = self.allMasks[:, :, np.argmax(np.sum(np.where(self.allMasks, np.repeat(self.beta[:, :, np.newaxis], len(indices), axis=2), 0), axis=(0,1))/sizes[indices])]

    # Find surrounding region of given dominant mask.
    contours, _ = cv2.findContours(cv2.threshold((200*domMask).astype(np.uint8), 127, 255, 0)[1], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cimg = np.zeros(self.image.shape[:2])
    cv2.drawContours(cimg, contours, 0, color=255, thickness=3)

    self.surrMask = cimg == 255
    self.domMask = np.bitwise_and(np.bitwise_xor(domMask,self.surrMask), domMask)

  """
  evaluate_k calculates the value of constant k (Shen et al) using given dominant and specular masks.
  k is obtained via least squares minimization.
  """
  def evaluate_k(self, domMask, surrMask):
    domVals = np.sum(np.where(np.repeat(domMask[:, :, np.newaxis], 3, axis=2), self.image, 0), axis=(0,1))/np.count_nonzero(domMask)
    surrVals = np.sum(np.where(np.repeat(surrMask[:, :, np.newaxis], 3, axis=2), self.image, 0), axis=(0,1))/np.count_nonzero(surrMask)
    domBetaVals = np.sum(np.where(np.repeat(domMask[:, :, np.newaxis], 3, axis=2), np.repeat(self.beta[:, :, np.newaxis], 3, axis=2), 0), axis=(0,1))/np.count_nonzero(domMask)
    surrBetaVals = np.sum(np.where(np.repeat(surrMask[:, :, np.newaxis], 3, axis=2), np.repeat(self.beta[:, :, np.newaxis], 3, axis=2), 0), axis=(0,1))/np.count_nonzero(surrMask)

    fun = lambda k:((domVals - surrVals) -k*(domBetaVals - surrBetaVals)) @ ((domVals - surrVals) -k*(domBetaVals - surrBetaVals))
    res = minimize(fun, [0.6], method="SLSQP")
    return res.x

  """
  remove_specular_highlights removes specular highlights from given image using Shen et al. 2009.
  https://www.researchgate.net/publication/24410295_Simple_and_efficient_method_for_specularity_removal_in_a_image
  It returns the diffuse image as a result.
  """
  def remove_specular_highlights(self):
    self.compute_specular_masks()

    k = self.evaluate_k(self.domMask, self.surrMask)
    print ("k: ", k)

    return (self.image.copy() - k*np.repeat(self.beta[:, :, np.newaxis], 3, axis=2)).astype(np.uint8)

  """
  biclustering_Kmeans returns the dominant and surrounding masks of given mask.
  The mask clustering is done using Kmeans of the original mask.
  """
  def biclustering_Kmeans(self, mask):
    labels = self.clf.fit_predict(self.image[mask])

    flatMask = np.ndarray.flatten(mask)
    indices = np.argwhere(flatMask)

    # Compute masks of both clusters.
    aMask, bMask = flatMask.copy(), flatMask.copy()
    aMask[indices] = np.reshape(labels == 0, (len(labels == 0), 1))
    bMask[indices] = np.reshape(labels == 1,  (len(labels == 1), 1))
    aMask = np.reshape(aMask, mask.shape)
    bMask = np.reshape(bMask, mask.shape)

    domdomMask, domSurrMask = None, None
    if np.sum(self.beta[aMask])/np.count_nonzero(aMask) >= np.sum(self.beta[bMask])/np.count_nonzero(bMask):
      return aMask, bMask

    return bMask, aMask

  """
  remove_specular_highlights_modified is the modified version of remove specular highlights.
  This algorithm calculates the dffuse image using Shen et al but stores the specular
  region after further processing. The stored specular region can be reused in diffuse calculations.
  """
  def remove_specular_highlights_modified(self):
    self.compute_specular_masks()

    domdomMask, domSurrMask = self.biclustering_Kmeans(self.domMask)

    k = self.evaluate_k(domdomMask, domSurrMask)

    self.diffuse = (self.image.copy() - k*np.repeat(self.beta[:, :, np.newaxis], 3, axis=2)).astype(np.uint8)

    specularMask = np.zeros(self.image.shape[:2], dtype=bool)
    for i in range(self.allMasks.shape[2]):
      domdomMask, _ = self.biclustering_Kmeans(self.allMasks[:, :, i])
      specularMask = np.bitwise_or(specularMask, domdomMask)

    self.specularMask = specularMask
    return specularMask

  """
  compute_diffuse will compute the irradiance(E) and diffuse color using the
  normals of the face and spherical harmonic coefficients.
  """
  def compute_diffuse(self):
    if self.lighting == None or self.faceVertices == None:
      return
    self.adjust_SH_coeffs()

    Mred = self.compute_M_matrix(self.lighting["red"])
    Mgreen = self.compute_M_matrix(self.lighting["green"])
    Mblue = self.compute_M_matrix(self.lighting["blue"])

    # Convert to n by 4 array.
    normals = self.normals.copy()
    normals = np.append(normals, np.ones((normals.shape[0], 1)), 1)
    Ered = (normals @ Mred @ np.transpose(normals)).diagonal()
    Egreen = (normals @ Mgreen @ np.transpose(normals)).diagonal()
    Eblue = (normals @ Mblue @ np.transpose(normals)).diagonal()
    E = np.column_stack((Ered, Egreen, Eblue))

    # Calculate centroid vertices of each triangle in the mesh.
    faceVertices = np.array(self.faceVertices)
    tIndices = self.triangleIndices
    tVertices = []
    for i in range(0, len(tIndices), 3):
      t1, t2 , t3 = tIndices[i], tIndices[i+1], tIndices[i+2]
      tVertices.append([(faceVertices[t1, 0]+faceVertices[t2, 0]+faceVertices[t3, 0])/3.0, (faceVertices[t1,1]+ faceVertices[t2, 1]+faceVertices[t3, 1])/3.0])
    tVertices = np.array(tVertices).astype(int)

    # Show irrandiance mask.
    clone = self.image.copy()
    clone[:, :] = [0, 0, 0]
    clone[tVertices[:, 0], tVertices[:, 1]] = np.uint8(np.clip(E*100, None, 255))
    self.show(clone)

    # Calculate intensity at each centroid vertex and show.
    clone = self.image.copy()
    intensity = ImageUtils.add_gamma_correction_matrix(ImageUtils.remove_gamma_correction_matrix(clone[tVertices[:, 0], tVertices[:, 1]].astype(np.float))/E)
    clone[:, :] = [255, 255, 255]
    for i in range(intensity.shape[0]):
      cv2.circle(clone, (tVertices[i, 1], tVertices[i, 0]), 1, [int(intensity[i][0]), int(intensity[i][1]), int(intensity[i][2])], 20)

    self.show(clone)

  """
  adjust_SH_coeffs adjusts the signs of SH coefficients provided by ARKit depending
  on the sign x, y and z components of the face normals.
  """
  def adjust_SH_coeffs(self):
    normals = self.normals

    cntNormals = normals.shape[0]
    self.cntXpos = np.count_nonzero(normals[:, 0] >= 0)
    self.cntXneg = cntNormals - self.cntXpos
    self.cntYpos = np.count_nonzero(normals[:, 1] >= 0)
    self.cntYneg = cntNormals - self.cntYpos
    self.cntZpos = np.count_nonzero(normals[:, 2] >= 0)
    self.cntZneg = cntNormals - self.cntZpos
    print ("X pos cnt: ", self.cntXpos, " X neg cnt: ", self.cntXneg)
    print ("Y pos cnt: ", self.cntYpos, " Y neg cnt: ", self.cntYneg)
    print ("Z pos cnt: ", self.cntZpos, " Z neg cnt: ", self.cntZneg)

    self.adjust_SH_coeffs_helper(self.lighting["red"])
    self.adjust_SH_coeffs_helper(self.lighting["green"])
    self.adjust_SH_coeffs_helper(self.lighting["blue"])

  """
  adjust_SH_coeffs helper will adjust coefficients
  for given channel (red, blue or green).
  """
  def adjust_SH_coeffs_helper(self, L):
    # Always flip X (based on observation).
    # Check flip conditions for y and z directions.
    flipY = True if ((self.cntYpos >= self.cntYneg and L[1] < 0) or (self.cntYpos < self.cntYneg and L[1] >= 0)) else False
    flipZ = True if ((self.cntZpos >= self.cntZneg and L[2] < 0) or (self.cntZpos < self.cntZneg and L[2] >= 0)) else False

    L[1] = -L[1] if flipY else L[1]
    L[2] = -L[2] if flipZ else L[2]
    L[3] = -L[3]
    L[4] = L[4] if flipY else -L[4]
    L[5] = L[5] if flipY == flipZ else -L[5]
    L[7] = L[7] if flipZ else -L[7]

  """
  compute_M_matrix computes the M matrix specified in Ramamoorthi's paper.
  The M matrix is multiplied with normalized normal vector at each vertex
  to obtain the irradiance at the vertex. We need to flip spherical harmonic
  values of x and z directions based on observation.
  """
  def compute_M_matrix(self, L):
    c = self.c
    return np.array([
      [c[0]*L[8], c[0]*L[4], c[0]*L[7], c[1]*L[3]], [c[0]*L[4], -c[0]*L[8], c[0]*L[5], c[1]*L[1]],
      [c[0]*L[7], c[0]*L[5], c[2]*L[6], c[1]*L[2]], [c[1]*L[3], c[1]*L[1], c[1]*L[2], c[3]*L[0]-c[4]*L[6]],
    ])

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
  f.compute_diffuse()
  #f.show_mask(f.remove_specular_highlights_modified())
