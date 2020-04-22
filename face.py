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

class Face:
  def __init__(self, imagePath="", image=None, outputDir="output"):
    self.outputDir = outputDir
    self.imagePath = imagePath
    self.faceMaskDirPath =  os.path.join(os.path.join(self.outputDir, os.path.splitext(os.path.split(imagePath)[1])[0]), "annotations")
    self.hdf5File = "face_mask.hdf5"
    self.CLASS_IDS_KEY = "class_ids"
    self.MASKS_KEY = "masks"
    self.id_label_map = {v: k for k, v in label_id_map.items()}

    if image is None:
      self.image = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2RGB)
    else:
      self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    self.satImage = self.to_satImage()
    self.brightImage = self.to_brightImage()
    self.ratioImage = self.to_ratioImage()

    self.clf = KMeans(n_clusters=2)
    self.preds = self.detect_face()
    self.faceMask = self.get_face_mask()
    self.beta = self.specularity()
    if imagePath.startswith("server/data"):
      # Load lighting, normals and face vertices.
      dirPath, _ = os.path.split(imagePath)
      with open(os.path.join(dirPath, "face_vertices.json"), "r") as f:
        self.faceVertices = np.array(json.load(f))
      with open(os.path.join(dirPath, "lighting.json"), "r") as f:
        self.lighting = json.load(f)
      with open(os.path.join(dirPath, "triangle_indices.json"), "r") as f:
        self.triangleIndices = np.array(json.load(f))
      with open(os.path.join(dirPath, "vertex_normals.json"), "r") as f:
        self.vertexNormals = np.array(json.load(f))
        self.compute_vertex_normals()

      # Constants array from Ramamoorthi's paper.
      self.c = np.array([0.429043, 0.511664, 0.743125, 0.886227, 0.247708])

    self.windowName = imagePath
    self.windowSize = 900

  """
  show_masks will display multiple masks in the image.
  """
  def show_masks(self, masks, colorList):
    assert len(masks) == len(colorList), "Num masks != num colors"
    clone = self.image.copy()
    for i, mask in enumerate(masks):
      clone[mask] = np.array(colorList[i])
    return self.show(clone)

  """
  show_mask will display given mask.
  """
  def show_mask(self, mask, color=[0, 255, 0]):
    clone = self.image.copy()
    clone[mask] = np.array(color)
    return self.show(clone)

  """
  show_orig_image shows the original RGB image without modifications.
  """
  def show_orig_image(self):
    return self.show(self.image)

  """
  show_irrad_mask shows given irradiance mask.
  """
  def show_irrad_mask(self, vertices, E):
    clone = self.image.copy()
    clone[:, :] = [0, 0, 0]
    clone[vertices[:, 0], vertices[:, 1]] = np.clip(E*100, None, 255).astype(np.uint8)
    return self.show(clone)

  """
  show is an internal helper function to display given RGB image.
  """
  def show(self, image):
    cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(self.windowName, self.windowSize, self.windowSize)
    cv2.imshow(self.windowName, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return cv2.waitKey(0) & 0xFF

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
    eta = 0.1
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
    if self.allMasks.shape[2] == 0:
      # No masks found.
      print ("NO Specular Masks found")
      return

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

    if np.sum(self.beta[aMask])/np.count_nonzero(aMask) >= np.sum(self.beta[bMask])/np.count_nonzero(bMask):
      return aMask, bMask

    return bMask, aMask

  """
  biclustering_Kmeans returns the dominant and surrounding masks (in that order) of given mask.
  It differs from original implementation in that it calculates dominant based on greater average
  brightness of pixels.
  """
  def biclustering_Kmeans_mod(self, image, mask,
    func=lambda img, m1, m2: (m1, m2) if np.mean(img[m1][:, 0]) >= np.mean(img[m2][:, 0]) else (m2, m1)):
    labels = self.clf.fit_predict(image[mask])

    flatMask = np.ndarray.flatten(mask)
    indices = np.argwhere(flatMask)

    # Compute masks of both clusters.
    aMask, bMask = flatMask.copy(), flatMask.copy()
    aMask[indices] = np.reshape(labels == 0, (len(labels == 0), 1))
    bMask[indices] = np.reshape(labels == 1,  (len(labels == 1), 1))
    aMask = np.reshape(aMask, mask.shape)
    bMask = np.reshape(bMask, mask.shape)

    return func(image, aMask, bMask)

  """
  remove_specular_highlights_modified is the modified version of remove specular highlights.
  This algorithm calculates the dffuse image using Shen et al but stores the specular
  region for further processing. The stored specular region can be reused in diffuse calculations.
  """
  def remove_specular_highlights_modified(self):
    self.compute_specular_masks()

    domdomMask, domSurrMask = self.biclustering_Kmeans(self.domMask)

    k = self.evaluate_k(domdomMask, domSurrMask)

    self.diffuse = (self.image.copy() - k*np.repeat(self.beta[:, :, np.newaxis], 3, axis=2)).astype(np.uint8)

    specularMask = np.zeros(self.image.shape[:2], dtype=bool)
    for i in range(self.allMasks.shape[2]):
      #domdomMask, _ = self.biclustering_Kmeans(self.allMasks[:, :, i])
      specularMask = np.bitwise_or(specularMask, self.allMasks[:, :, i])

    self.specularMask = specularMask
    return specularMask

  """
  compute_diffuse will compute the irradiance(E) and diffuse color using the
  normals of the face and spherical harmonic coefficients.
  """
  def compute_diffuse(self):
    if self.lighting == None:
      return

    tVertices = self.allVerts.astype(int)
    self.adjust_SH_coeffs()

    Mred = self.compute_M_matrix(self.lighting["red"])
    Mgreen = self.compute_M_matrix(self.lighting["green"])
    Mblue = self.compute_M_matrix(self.lighting["blue"])

    # Convert to n by 4 array.
    normals = np.append(self.allVertNorms, np.ones((self.allVertNorms.shape[0], 1)), 1)
    Ered = np.einsum('ij,ij->i', normals @ Mred, normals)
    Egreen = np.einsum('ij,ij->i', normals @ Mgreen, normals)
    Eblue = np.einsum('ij,ij->i', normals @ Mblue, normals)
    E = np.column_stack((Ered, Egreen, Eblue))
    self.show_irrad_mask(tVertices, E)

    # Calculate intensity at each vertex point and show.
    clone = self.image.copy()
    intensity = ImageUtils.add_gamma_correction_matrix(ImageUtils.remove_gamma_correction_matrix(clone[tVertices[:, 0], tVertices[:, 1]])*((1.0)/E))
    clone[:, :] = [255, 255, 255]
    for i in range(intensity.shape[0]):
      cv2.circle(clone, (tVertices[i, 1], tVertices[i, 0]), 1, [int(intensity[i][0]), int(intensity[i][1]), int(intensity[i][2])], 1)
    self.show(clone)

  """
  compute_diffuse_new will compute the irradiance(E) and diffuse color using the
  normals of the face and spherical harmonic coefficients using an alternate
  formula to calculate the irradiance.
  """
  def compute_diffuse_new(self):
    if self.lighting == None:
      return

    tVertices = self.allVerts.astype(int)
    self.adjust_SH_coeffs()

    # Compute harmonic functions.
    normals = self.allVertNorms
    normMatrix = np.column_stack((np.ones((normals.shape[0],1)), normals[:, 1], normals[:, 2], normals[:, 0], normals[:, 1]*normals[:, 0], normals[:, 1]*normals[:, 2], (3*normals[:, 2]*normals[:, 2])-1, normals[:, 2]*normals[:, 0], (normals[:, 0]*normals[:, 0]) - (normals[:, 1]*normals[:, 1])))

    Ered = normMatrix @ self.lighting["red"]
    Egreen = normMatrix @ self.lighting["green"]
    Eblue = normMatrix @ self.lighting["blue"]
    E = np.column_stack((Ered, Egreen, Eblue))
    self.show_irrad_mask(tVertices, E)

    # Calculate intensity at each vertex point and show.
    print ("E, verts: ", E.shape[0], tVertices.shape[0])
    clone = self.image.copy()
    intensity = ImageUtils.add_gamma_correction_matrix(ImageUtils.remove_gamma_correction_matrix(clone[tVertices[:, 0], tVertices[:, 1]])*((1.0)/E))
    clone[:, :] = [255, 255, 255]
    for i in range(intensity.shape[0]):
      cv2.circle(clone, (tVertices[i, 1], tVertices[i, 0]), 1, [int(intensity[i][0]), int(intensity[i][1]), int(intensity[i][2])], 1)
    self.show(clone)


  """
  adjust_SH_coeffs adjusts the signs of SH coefficients provided by ARKit depending
  on the sign x, y and z components of the face normals.
  """
  def adjust_SH_coeffs(self):
    normals = self.allVertNorms

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

    self.reverseXZ = False
    if abs(self.cntYpos - self.cntYneg) > abs(self.cntZpos - self.cntZneg):
      self.reverseXZ = True
      print ("reversed X and Z")

    self.show_positive_normals()

    self.adjust_SH_coeffs_helper(self.lighting["red"])
    self.adjust_SH_coeffs_helper(self.lighting["green"])
    self.adjust_SH_coeffs_helper(self.lighting["blue"])

  """
  show_positive_normals displays positive normals mask in X, Y and Z directions.
  """
  def show_positive_normals(self):
    normals = self.allVertNorms
    verts = self.allVerts.astype(int)

    # X direction.
    clone = self.image.copy()
    vmask = verts[normals[:, 0] >= 0]
    clone[vmask[:, 0], vmask[:, 1]] = [255, 0, 0]
    self.show(clone)

    # Y direction.
    clone = self.image.copy()
    vmask = verts[normals[:, 1] >= 0]
    clone[vmask[:, 0], vmask[:, 1]] = [0, 255, 0]
    self.show(clone)

    # Z direction.
    clone = self.image.copy()
    vmask = verts[normals[:, 2] >= 0]
    clone[vmask[:, 0], vmask[:, 1]] = [0, 0, 255]
    self.show(clone)

  """
  adjust_SH_coeffs helper will adjust coefficients
  for given channel (red, blue or green) based on whether
  X,Y and Z are flipped or not.
  """
  def adjust_SH_coeffs_helper(self, L):
    if self.reverseXZ:
      flipZ = True
      flipY = True
      flipX = True if ((self.cntXpos >= self.cntXneg and L[3] < 0) or (self.cntXpos < self.cntXneg and L[3] >= 0)) else False
    else:
      flipZ = True if ((self.cntZpos >= self.cntZneg and L[2] < 0) or (self.cntZpos < self.cntZneg and L[2] >= 0)) else False
      flipZ = True
      flipY = True
      flipX = True

    print ("flipX: ", flipX, " flipY: ", flipY, " flipZ: ", flipZ)
    L[1] = -L[1] if flipY else L[1]
    L[2] = -L[2] if flipZ else L[2]
    L[3] = -L[3] if flipX else L[3]
    L[4] = L[4] if flipY == flipX else -L[4]
    L[5] = L[5] if flipY == flipZ else -L[5]
    L[7] = L[7] if flipX == flipZ else -L[7]

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

  def compute_vertex_normals(self):
    faceVertices = self.faceVertices
    triangleIndices = np.reshape(self.triangleIndices, (-1, 3))
    triangle2DArr = faceVertices[triangleIndices]
    vertexNormals = self.vertexNormals[triangleIndices]
    allAreas = self.compute_areas(triangle2DArr)

    minxarr = np.min(triangle2DArr[:, :, 0], axis=1)
    maxxarr = np.max(triangle2DArr[:, :, 0], axis=1)
    minyarr = np.min(triangle2DArr[:, :, 1], axis=1)
    maxyarr = np.max(triangle2DArr[:, :, 1], axis=1)

    self.allVerts = np.zeros((0,2))
    self.allVertNorms = np.zeros((0,3))
    for i in range(triangle2DArr.shape[0]):
      bbox = np.mgrid[minxarr[i]:maxxarr[i], minyarr[i]:maxyarr[i]].reshape(2, -1).T
      v0, v1, v2 = triangle2DArr[i][0], triangle2DArr[i][1], triangle2DArr[i][2]

      bLen = bbox.shape[0]
      areas0 = self.compute_areas(np.stack((np.repeat(v1[np.newaxis, :], bLen, axis=0), np.repeat(v2[np.newaxis, :], bLen, axis=0), bbox), axis=1))
      areas1 = self.compute_areas(np.stack((np.repeat(v2[np.newaxis, :], bLen, axis=0), np.repeat(v0[np.newaxis, :], bLen, axis=0), bbox), axis=1))
      areas2 = self.compute_areas(np.stack((np.repeat(v0[np.newaxis, :], bLen, axis=0), np.repeat(v1[np.newaxis, :], bLen, axis=0), bbox), axis=1))

      areas = np.stack((areas0, areas1, areas2), axis=1)
      tol = 0.001
      masks = np.stack((areas0 >= tol, areas1 >= tol, areas2 >= tol), axis=1)

      idxArr = np.where(masks.all(axis=1))[0]
      w = areas[idxArr]/allAreas[i]

      # n by 3 where n is len(idxArr)
      vNorms = w @ vertexNormals[i]
      verts = bbox[idxArr]
      if verts.shape[0] == 0:
        continue

      self.allVerts = np.concatenate((self.allVerts, verts), axis=0)
      self.allVertNorms = np.concatenate((self.allVertNorms, vNorms), axis=0)

  """
  compute_areas will compute deteminants of given array of triangle(each triangle is represented by 3 vertices).
  The input shape of triangles is (n, 3, 2) and the points are in counter clockwise order.
  The out is an array of areas of dimension (n, 1).
  """
  def compute_areas(self, a):
    # Get vector array from triangle vertex array.
    return np.linalg.det(np.stack((a[:, 1] - a[:, 0], a[:, 2] - a[:, 0]), axis=1))

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

    # Reduce facemask to remove beard.
    """
    nosemask = self.get_attr_masks(NOSE)[0]
    row, _, _, h = self.bbox(nosemask)
    faceMask[(row+h):, :] = False
    """

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
  get_chakra_keypoints returns keypoints between
  the two eyebrows.
  """
  def get_chakra_keypoints(self):
    eyebrow_masks = self.get_attr_masks(EYEBROW)
    eye_masks = self.get_attr_masks(EYE_OPEN)
    assert len(eyebrow_masks) == 2, "Want 2 masks for eyebrows!"
    assert len(eye_masks) == 2, "Want 2 masks for eyes!"

    left_eyebrow_mask = eyebrow_masks[0] if self.bbox(eyebrow_masks[0])[1] < self.bbox(eyebrow_masks[1])[1] else eyebrow_masks[1]
    right_eyebrow_mask = eyebrow_masks[1] if self.bbox(eyebrow_masks[0])[1] < self.bbox(eyebrow_masks[1])[1] else eyebrow_masks[0]

    left_eye_mask = eye_masks[0] if self.bbox(eye_masks[0])[1] < self.bbox(eye_masks[1])[1] else eye_masks[1]
    right_eye_mask = eye_masks[1] if self.bbox(eye_masks[0])[1] < self.bbox(eye_masks[1])[1] else eye_masks[0]

    lebrmin,lebcmin, lebw, _ = self.bbox(left_eyebrow_mask)
    rebrmin, rebcmin, rebw, _ = self.bbox(right_eyebrow_mask)

    lermin, _, _, leh = self.bbox(left_eye_mask)
    rermin, _, _, reh = self.bbox(right_eye_mask)

    rmin = int(max(lebrmin, rebrmin))
    rmax = int(min(lermin + leh, rermin + reh))
    ebWidth = rebcmin - lebcmin - lebw
    cmin = int(lebcmin + lebw + ebWidth/4)
    cmax = int(rebcmin - ebWidth/4)

    mask = np.zeros(left_eye_mask.shape, dtype=bool)
    mask[rmin:rmax+1, cmin:cmax+1] = True
    return mask

  """
  get_neck_points returns some points of the neck of the person.
  """
  def get_neck_points(self):
    faceMasks = self.get_attr_masks(FACE)
    hairMasks = self.get_attr_masks(HAIR_ON_HEAD)
    eyeMasks = self.get_attr_masks(EYE_OPEN)
    assert len(faceMasks) == 1, "Want 1 mask for face!"
    assert len(eyeMasks) == 2, "Want 2 masks for eye!"

    # Use facemask to determine maximum height of neck mask.
    # Use eyes to restrict width of neck mask.
    faceMask = faceMasks[0]
    rmin, _, _, h = self.bbox(faceMasks[0])
    _, c1, w1, _ = self.bbox(eyeMasks[0])
    _, c2, w2, _ = self.bbox(eyeMasks[1])
    w1, w2 = int(w1/2), int(w2/2)

    mask = np.zeros(faceMask.shape, dtype=bool)
    mask[rmin+int(0.9*h):rmin + int(1.3*h),  min(c1+w1, c2+w2):max(c1+w1, c2+w2)] = True
    mask = np.bitwise_xor(np.bitwise_and(mask, faceMask), mask)
    self.show_mask(mask)

    # Filter out any interactions with hair.
    for hMask in hairMasks:
      mask = np.bitwise_xor(np.bitwise_and(mask, hMask), mask)

    # Cluster on saturation.
    ratioImage = self.to_ratioImage()

    #g = lambda img, m1, m2: (m1, m2) if np.mean(img[m1][:, 0]) >= np.mean(img[m2][:, 0]) else (m2, m1)
    g = lambda img, m1, m2: (m1, m2) if np.mean(img[m1][:, 1]) >= np.mean(img[m2][:, 1]) else (m2, m1)

    #l1Mask, l2Mask = self.biclustering_Kmeans_mod(self.image, mask, g)
    l1Mask, l2Mask = self.divide_by_ratio(ratioImage, mask, g)
    bMask = mask
    while ImageUtils.delta_e_mask(self.image, l1Mask, l2Mask) > 20:
      self.show_masks([l1Mask, l2Mask], [[0, 255, 0], [255, 0, 0]])
      bMask = l1Mask
      #l1Mask, l2Mask = self.biclustering_Kmeans_mod(self.image, bMask, g)
      l1Mask, l2Mask = self.divide_by_ratio(ratioImage, bMask, g)

    neckTone = np.median(self.image[l1Mask], axis=0)
    print ("final mask tone: ", ImageUtils.RGB2HEX(neckTone))
    self.show_masks([l1Mask, l2Mask], [[0, 255, 0], [255, 0, 0]])
    self.show_masks([self.faceMask], [neckTone])
    return mask

  """
  get_forehead_points returns points on the forehead.
  """
  def get_forehead_points(self):
    faceMasks = self.get_attr_masks(FACE)
    assert len(faceMasks) == 1, "Want 1 mask for face!"
    faceMask = faceMasks[0]
    eyebrowMasks = self.get_attr_masks(EYEBROW)
    assert len(eyebrowMasks) > 0, "Want atleast 1 mask for eyebrows"

    mask = faceMask
    for ebMask in eyebrowMasks:
      rmin, _, _, _ = self.bbox(ebMask)
      mask[rmin:,:] = False
    return mask

  """
  good_face_points returns a mask containing points that are usually good to
  sample for skin tone. It excludes points near the eyes and lips.
  """
  def good_face_points(self):
    faceMasks = self.get_attr_masks(FACE)
    eyeMasks = self.get_attr_masks(EYE_OPEN)
    ulipMasks = self.get_attr_masks(UPPER_LIP)
    llipMasks = self.get_attr_masks(LOWER_LIP)
    assert len(faceMasks) == 1, "Want 1 mask for face!"
    assert len(ulipMasks) == 1, "Want 1 mask for upper lip!"
    assert len(llipMasks) == 1, "Want 1 mask for lower lip!"
    assert len(eyeMasks) == 2, "Want 2 masks for eye!"

    mask = faceMasks[0]

    rmin1, _, _, h1 = self.bbox(eyeMasks[0])
    rmin2, _, _, h2 = self.bbox(eyeMasks[0])
    tol = 1.0
    rmin = min(rmin1, rmin2) - tol*max(h1, h2)
    rmax = max(rmin1 + h1, rmin2 + h2) + tol*max(h1, h2)
    mask[int(rmin):int(rmax)+1, :] = False

    tol = 0.5
    tol_w = 0.1
    rmin, cmin, w, h = self.bbox(ulipMasks[0])
    mask[int(rmin-tol*h):int(rmin+ h + tol*h), int(cmin -tol_w*w):int(cmin+w+tol_w*w)] = False

    rmin, cmin, w, h = self.bbox(llipMasks[0])
    mask[int(rmin-tol*h):int(rmin+ h + tol*h), int(cmin -tol_w*w):int(cmin+w+tol_w*w)] = False

    return mask

  """
  to_ratioImage converts sRGB image to HSV like saturation image
  with H = 0, S = Brightness/saturation (float), V = 0.
  """
  def to_ratioImage(self):
    ratioImage = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV).astype(np.float)
    ratioImage[:, :, 0] = 0
    flatMask = ratioImage[ratioImage[:, :, 1] == 0]
    flatMask[:, 1] = 0.1
    ratioImage[ratioImage[:, :, 1] == 0] = flatMask
    ratioImage[:, :, 1] = ratioImage[:, :, 2]/(ratioImage[:, :, 1])
    ratioImage[:, :, 2] = 0
    return ratioImage

  """
  to_brightImage converts sRGB image to HSV image with
  H = 0, S = 0 and only brightness values.
  """
  def to_brightImage(self):
    hsvImage = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)
    hsvImage[:, :, 0] = 0
    hsvImage[:, :, 1] = 0
    return hsvImage

  """
  to_brightImage converts sRGB image to HSV image with
  H = 0, V = 0 and only saturation values.
  """
  def to_satImage(self):
    hsvImage = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)
    hsvImage[:, :, 0] = 0
    hsvImage[:, :, 2] = 0
    return hsvImage

  """
  check_if_good_image returns true if image is good to analyze for skin
  tone else returns false.
  """
  def check_if_good_image(self):
    ImageUtils.plot_histogram(self.image, self.faceMask, False)
    hist = cv2.calcHist([self.image],[0], self.faceMask.astype(np.uint8)*255,[256],[0,256])

    print ("Median Red: ", np.median(self.image[self.faceMask][:, 0]))
    if np.median(self.image[self.faceMask][:, 0]) <= 140:
      print ("Image too dark: ", np.median(self.image[self.faceMask][:, 0]) )
      self.show(self.image)
      return False

    if hist[-1]/max(hist) >= 0.2:
      # Red pixel == 255, too bright.
      print ("Image too bright: ", hist[-1]/max(hist))
      self.show(self.image)
      return False

    return True

  """
  detect_skin_tone detects skin tone for given face.
  """
  def detect_skin_tone(self):
    # self.check_if_good_image()
    higherSatMask, lowerSatMask = self.divide_by_saturation(self.faceMask)
    if np.mean(self.ratioImage[higherSatMask], axis=0)[1] <= 1.54:
      print ("Skin is darker")
      higherSatMask = np.bitwise_and(higherSatMask, self.good_face_points())
      self.show_mask(higherSatMask)
      skinTone = np.median(self.image[higherSatMask], axis=0)
      print ("Skin tone: ", ImageUtils.RGB2HEX(skinTone))
      self.show_masks([self.faceMask], [skinTone])
      self.show_orig_image()
      return

    print ("Skin is lighter")
    largerSatMask = higherSatMask if np.count_nonzero(higherSatMask) > np.count_nonzero(lowerSatMask) else lowerSatMask
    higherBriMask, _ = self.divide_by_brightness(self.faceMask)
    higherSatFinalMask, lowerSatFinalMask = self.divide_by_saturation(np.bitwise_and(higherBriMask, largerSatMask))

    finalMask = lowerSatFinalMask if np.count_nonzero(higherSatMask) == np.count_nonzero(largerSatMask) else higherSatFinalMask
    finalMask = np.bitwise_and(finalMask, self.good_face_points())
    self.show_mask(finalMask)
    skinTone = np.median(self.image[finalMask], axis=0)
    print ("Final Skin tone: ", ImageUtils.RGB2HEX(skinTone))
    self.show_masks([self.faceMask], [skinTone])
    self.show_orig_image()

  """
  divide_by_saturation divides given mask using saturation image
  into higher and lower saturation sub masks.
  It returns the mask with higher brightness first.
  """
  def divide_by_saturation(self, mask):
    higherSatMask, lowerSatMask = self.biclustering_Kmeans_mod(self.satImage, mask, func=lambda img, m1, m2: (m1, m2) if np.mean(img[m1][:, 1]) >= np.mean(img[m2][:, 1]) else (m2, m1))
    self.masks_info(higherSatMask, lowerSatMask, mask)
    return higherSatMask, lowerSatMask

  """
  divide_by_brightness divides given mask using brightness image
  into higher and lower brightness sub masks.
  It returns the mask with higher saturation first.
  """
  def divide_by_brightness(self, mask):
    higherBriMask, lowerBriMask = self.biclustering_Kmeans_mod(self.brightImage, mask, func=lambda img, m1, m2: (m1, m2) if np.mean(img[m1][:, 2]) >= np.mean(img[m2][:, 2]) else (m2, m1))
    self.masks_info(higherBriMask, lowerBriMask, mask)
    return higherBriMask, lowerBriMask

  """
  divide_by_ratio will divide given mask using brightness by brightness/saturation ratio.
  It returns the mask with higher brightness/saturation ratio first.
  """
  def divide_by_ratio(self, mask):
    higherRatioMask, lowerRatioMask = self.biclustering_Kmeans_mod(self.ratioImage, mask, func=lambda img, m1, m2: (m1, m2) if np.mean(img[m1][:, 1]) >= np.mean(img[m2][:, 1]) else (m2, m1))
    self.masks_info(higherRatioMask, lowerRatioMask, mask)
    return higherRatioMask, lowerRatioMask

  """
  masks_info will print various measurements about the input masks like brightness,
  saturation, delta_e_cie2000 diff etc. which can then be analyzed later.
  The function also displays the masks.
  """
  def masks_info(self, leftMask, rightMask, mask):
    print ("\tMask division: ", np.count_nonzero(leftMask)/np.count_nonzero(mask), np.count_nonzero(rightMask)/np.count_nonzero(mask))
    print ("\tSat values: ", np.median(self.satImage[leftMask], axis=0)[1]*100.0/255.0, np.median(self.satImage[rightMask], axis=0)[1]*100.0/255.0)
    print ("\tBrightness values: ", np.median(self.brightImage[leftMask], axis=0)[2]*100.0/255.0, np.median(self.brightImage[rightMask], axis=0)[2]*100.0/255.0)
    print ("\tRatio values: ", np.median(self.ratioImage[leftMask], axis=0)[1], np.median(self.ratioImage[rightMask], axis=0)[1])
    print ("\tColor diff: ", ImageUtils.delta_e_mask(self.image, leftMask, rightMask))
    print ("")
    self.show_masks([leftMask, rightMask], [[0, 255, 0], [255, 0, 0]])

  """
  divide_mask will divide given mask into clusters based on delta_e_cie2000 differences.
  """
  def divide_mask(self, mask):
    DELTA = 5

    domMask, surrMask = self.biclustering_Kmeans_mod(self.image, mask)
    if ImageUtils.delta_e_mask(self.image, domMask, surrMask) <= DELTA:
      return [mask]

    domSubMasks = self.divide_mask(domMask)
    surrSubMasks = self.divide_mask(surrMask)

    return domSubMasks + surrSubMasks

  """
  adjust_brightness will set the brightness of color to brightness v.
  """
  def adjust_brightness(self, color, v):
    hsv = ImageUtils.sRGBtoHSV(color)
    hsv[0,2] = v
    return ImageUtils.HSVtosRGB(hsv)

  def apply_brightness(self, mask, sRGB):
    # Modify image brightness to brightness of given color.
    clone = self.image.copy()
    clone = cv2.cvtColor(clone, cv2.COLOR_RGB2HSV)
    fmask = clone[mask]
    fmask[:, 2] = ImageUtils.sRGBtoHSV(sRGB)[0, 2]
    clone[mask] = fmask
    clone = cv2.cvtColor(clone, cv2.COLOR_HSV2RGB)
    return clone

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
  # Parse command line arguments
  parser = argparse.ArgumentParser(description='Image path')
  parser.add_argument('--image', required=False,metavar="path or URL to image")
  args = parser.parse_args()

  f = Face(args.image)
  #f.get_neck_points()
  f.detect_skin_tone()
  #f.compute_diffuse()
  #f.compute_diffuse_new()
  #f.show(f.remove_specular_highlights())
