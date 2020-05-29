import os
import cv2
import argparse
import json
import random
import numpy as np
from image_utils import ImageUtils

from PIL import Image
from face import Face

FACE_COLORS_FILENAME = "face_colors.json"

"""
mesh_colors returns an 3D array(n, k, 3) of colors of n mesh points
from k different faces.
"""
def mesh_colors(videoPath, faceList):
  filePath = os.path.join(videoPath, FACE_COLORS_FILENAME)
  if os.path.exists(filePath):
    with open(filePath, "r") as f:
      return np.array(json.load(f))

  cList = []
  for f in faceList:
    cList.append(f.image[f.faceVertices[:, 0], f.faceVertices[:, 1]])

  cArray = np.stack(cList, axis=1)
  with open(os.path.join(videoPath, FACE_COLORS_FILENAME), "w") as f:
    json.dump(cArray.tolist(), f)
  return cArray

"""
faces returns a list of face class instances in given video path.
"""
def faces(videoPath):
  faceList = []
  for d in os.listdir(videoPath):
    if not os.path.isdir(os.path.join(videoPath, d)):
      continue
    imPath = os.path.join(videoPath, os.path.join(d, d)) + ".png"
    f = Face(imPath)
    faceList.append(f)
  return faceList

"""
remove_non_face_points removes mesh points from the mask
that are not part of the facemask mask.
"""
def remove_non_face_points(mask, faceMask, meshVertices):
  meshMask = np.zeros(faceMask.shape, dtype=bool)
  meshMask[meshVertices] = True

  badMask = np.bitwise_xor(np.bitwise_and(meshMask, faceMask), meshMask)
  badMeshVertices = np.transpose(np.nonzero(badMask))
  rows = np.where((meshVertices==badMeshVertices[:,None]).all(-1))[1]
  mask[rows, :] = False

"""
analyze will process images within given video directory.
"""
def analyze(videoPath):
  faceList = faces(videoPath)
  meshColors = mesh_colors(videoPath, faceList).astype(np.uint8)

  # Only select mesh points that don't vary by much over pictures.
  mask = np.zeros(meshColors.shape[:2], dtype=bool)
  for r in range(meshColors.shape[0]):
    colors = meshColors[r, :, :]
    columns = colors.shape[0]
    for i in range(columns):
      for j in range(i+1, columns):
        delta = ImageUtils.delta_cie2000(colors[i], colors[j])
        if delta <= 3:
          mask[r, i] = True
          mask[r, j] = True

  # Remove points in mask that are not in faceMask.
  remove_non_face_points(mask, faceList[0].faceMask, faceList[0].faceVertices)

  # Set bad mask to white.
  meshColors[mask==False] = [255, 255, 255]
  ImageUtils.show_rgb(np.repeat(meshColors, 100, axis=1))

  # Prepare for Kmeoids clustering.
  goodColors = meshColors[mask]
  dMatrix = np.zeros((goodColors.shape[0],goodColors.shape[0]))
  for i in range(goodColors.shape[0]):
    dMatrix[i, :] = ImageUtils.delta_e_mask_matrix(meshColors, goodColors[i],mask)
  medMasks = Kmedoids(goodColors, dMatrix, k=5)

  # Show each cluster.
  maskIndices = np.transpose(np.nonzero(mask))
  for k in range(medMasks.shape[0]):
    clone = meshColors.copy()
    kMask = np.reshape(medMasks[k, :], (goodColors.shape[0],))
    badMask = kMask == False

    clone[maskIndices[badMask]] = [255, 255, 255]
    ImageUtils.show_rgb(np.repeat(clone, 100, axis=1))

"""
Kmedoids implements Kmedoids algorithm (using given
distance matrix) to form k clusters for given colors.
"""
def Kmedoids(colors, dMatrix, k):
  n = dMatrix.shape[0]

  pairs = [[i, i] for i in range(n)]
  medIndices = random.sample(range(n), k)

  costMap = {}
  print ("Starting Kmedoids")
  numIterations = 0
  while True:
    # Assign labels to non medoid data points.
    cMap = {}
    for i in range(n):
      if i in set(medIndices):
        continue
      id = np.argmin(dMatrix[i, medIndices])
      pairs[i][1] = medIndices[id]
      if medIndices[id] not in cMap:
        cMap[medIndices[id]] = []
      cMap[medIndices[id]].append(i)

    # Compute costs in each cluster and swap medoid if needed.
    newIndices = []
    for medIdx in medIndices:
      cost = np.mean(dMatrix[medIdx, cMap[medIdx]])
      bestIdx = -1
      for j in cMap[medIdx]:
        jCost = np.mean(dMatrix[j, cMap[medIdx]])
        if jCost < cost:
          cost = jCost
          bestIdx = j
      if bestIdx == -1:
        bestIdx = medIdx

      costMap[bestIdx] = cost
      newIndices.append(bestIdx)

    if set(medIndices) == set(newIndices):
      # complete.
      print ("Kmeoids Computation complete")
      break

    numIterations += 1
    if numIterations == 100:
      # Too many iterations, break.
      print ("Too many iterations, complete computation.")
      break

    medIndices = newIndices

  mask = np.zeros((k, n), dtype=bool)
  for i, medIdx in enumerate(medIndices):
    mask[i, cMap[medIdx]] = True
    print ("Medoid : ", ImageUtils.RGB2HEX(colors[medIdx]), " cost: ", costMap[medIdx], " percent: ", (len(cMap[medIdx])/n)*100.0)
  return mask


if __name__ == "__main__":
  # Parse command line arguments
  parser = argparse.ArgumentParser(description='Video file path')
  parser.add_argument('--video', required=False,metavar="path to video file")
  args = parser.parse_args()

  analyze(args.video)
