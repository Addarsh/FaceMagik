import os
import cv2
import argparse
import json
import random
import math
import numpy as np
from image_utils import ImageUtils
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch

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
analyze will process images within given video directory.
"""
def analyze(videoPath):
  faceList = faces(videoPath)
  k = 6

  # Set up figure to plot.
  fig = plt.figure()
  ax = fig.add_axes([0, 0, 1, 1])
  ax.set_xlim(0, len(faceList))
  ax.set_ylim(0, k)
  ax.axis('off')
  plt.ion()

  count = 0
  for f in faceList:
    f.windowName = "image"
    colors = f.ym_colors()

    n = colors.shape[0]

    dMatrix = np.zeros((n,n))
    for i in range(n):
      dMatrix[i, :] = ImageUtils.delta_e_mask_matrix(colors[i], colors)

    medoids = []
    bestMedoids = []
    minCost = 10000.0
    for iter in range(500):
      try:
        medoids, cost = Kmedoids(colors, dMatrix, k=k)
        if cost < minCost:
          minCost = cost
          bestMedoids = medoids.copy()
      except Exception as e:
        print (e)

    print ("Dividing image into K: ", k, " clusters, with MIN COST: ", minCost)
    medoids = bestMedoids.copy()

    # Divide ym mask into given medoid clusters.
    allColors = f.image[f.ym]
    n = allColors.shape[0]
    newClusters = np.zeros((k, n))
    for i in range(k):
      newClusters[i, :] = ImageUtils.delta_e_mask_matrix(medoids[i]["medoid"], allColors)
    clusterIndices = np.argmin(newClusters, axis=0)

    allMasks = []
    allCords = np.transpose(np.nonzero(f.ym))
    for i in range(k):
      # Points in mask closest to ith medoid.
      clusterMask = clusterIndices == i
      cm = np.zeros(f.faceMask.shape, dtype=bool)
      cm[allCords[clusterMask][:, 0], allCords[clusterMask][:, 1]] = True
      allMasks.append(cm)

    fhMask = np.zeros(f.faceMask.shape, dtype=np.bool)
    try:
      fhMask = f.get_forehead_points()
      print ("forehead mask")
      f.show_mask(fhMask)
    except Exception as e:
      print ("forehead points not found: ", e)

    # Show mask corresponding to each medoid.
    for i in range(k):
      #if np.count_nonzero(allMasks[i])/np.count_nonzero(f.ym) <= 0.05:
      #  continue
      print ("Medoid color: ", ImageUtils.RGB2HEX(medoids[i]["medoid"]))
      print ("Percent: ", (np.count_nonzero(allMasks[i])/np.count_nonzero(f.ym))*100.0)
      print ("Mean brightness: ", np.mean(f.brightImage[allMasks[i]], axis=0)[2]*(100.0/255.0))
      print ("Mean sat: ", np.mean(f.satImage[allMasks[i]], axis=0)[1]*(100.0/255.0))
      print ("Std brightness: ", np.std(f.brightImage[allMasks[i]], axis=0)[2]*(100.0/255.0))
      print ("Mean ratio: ", np.mean(f.brightImage[allMasks[i]], axis=0)[2]/np.mean(f.satImage[allMasks[i]], axis=0)[1])
      print ("Meoid cluster sum: ", np.sum(dMatrix[medoids[i]["idx"], medoids[i]["cIndices"]]), medoids[i]["cIndices"])
      print ("")

      # Add medoid to plot.
      ax.add_patch(mpatch.Rectangle((count, i), 1, 1, color=ImageUtils.RGB2HEX(medoids[i]["medoid"])))

      f.show_mask(allMasks[i])

    # Show specular mask.
    f.show_mask(np.bitwise_and(f.beta != 0, f.faceMask))

    # Plot medoids.
    plt.show(block=False)

    count += 1
    if f.show_orig_image() == ord("q"):
      break

"""
Kmedoids implements Kmedoids algorithm (using given
distance matrix) to form k clusters for given colors.
Returned cluster matrix is of size (k, n).
"""
def Kmedoids(colors, dMatrix, k):
  n = dMatrix.shape[0]

  medIndices = random.sample(range(n), k)
  costMap = {}
  totalCost = 0.0
  while True:
    # Assign labels to non medoid data points.
    #print ("medIndices: ", medIndices)
    clusterMap = {}
    odMap = {}
    for i in range(n):
      id = np.argmin(dMatrix[i, medIndices])
      if medIndices[id] not in clusterMap:
        clusterMap[medIndices[id]] = set()
      clusterMap[medIndices[id]].add(i)
      odMap[i] = medIndices[id]

    bestCost = 0.0
    for m in medIndices:
      bestCost += np.sum(dMatrix[m, list(clusterMap[m])])
    totalCost = bestCost

    # Compute total cost after swapping each non medoid with each medoid.
    bestPair = (-1, -1)
    medIndicesSet = set(medIndices)
    for m in medIndices:
      for i in range(n):
        if i in medIndicesSet:
          continue

        # Swap i with m and compute cost of configuration.
        odc = odMap[i]
        clusterMap[i] = clusterMap[m].copy()
        odMap[i] = i
        del clusterMap[m]
        if odc != m:
          odMap[m] = odc
          clusterMap[i].add(i)
          clusterMap[i].remove(m)
          clusterMap[odc].remove(i)
          clusterMap[odc].add(m)
        else:
          odMap[m] = i

        cost = 0.0
        for r in clusterMap:
          cost += np.sum(dMatrix[r, list(clusterMap[r])])

        if cost < bestCost:
          # Save current best pair.
          bestCost = cost
          bestPair = (m, i)

        # Swap i and m back.
        odMap[m] = m
        odMap[i] = odc
        clusterMap[m] = clusterMap[i].copy()
        del clusterMap[i]
        if odc != m:
          clusterMap[odc].remove(m)
          clusterMap[odc].add(i)
          clusterMap[m].add(m)
          clusterMap[m].remove(i)

    if bestPair == (-1, -1) or math.isclose(totalCost, bestCost, rel_tol=1e-3):
      #print ("Kmedoids complete")
      break

    # Update medIndices.
    medIndicesSet.remove(bestPair[0])
    medIndicesSet.add(bestPair[1])
    medIndices = list(medIndicesSet).copy()

  medoids = []
  for idx in medIndices:
    med = {}
    med["medoid"] = colors[idx]
    med["cluster"] = colors[list(clusterMap[idx])]
    med["idx"] = idx
    med["cIndices"] = list(clusterMap[idx])
    medoids.append(med)

  return medoids, totalCost


if __name__ == "__main__":
  # Parse command line arguments
  parser = argparse.ArgumentParser(description='Video file path')
  parser.add_argument('--video', required=False,metavar="path to video file")
  args = parser.parse_args()

  analyze(args.video)
