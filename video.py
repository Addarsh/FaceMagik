import os
import argparse
import numpy as np
from image_utils import ImageUtils
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import time
import re

from face import Face

"""
faces returns a list of face class instances in given video path.
"""
def faces(videoPath):
  faceList = []
  fnameList = []
  for fname in os.listdir(videoPath):
    if os.path.isdir(os.path.join(videoPath, fname)):
      continue
    if not fname.endswith(".png") and not fname.endswith(".PNG"):
      continue
    fnameList.append(fname)

  fnameList = sorted(fnameList)
  for fname in fnameList:
    imPath = os.path.join(videoPath, fname)
    f = Face(imPath, hdf5FileName=os.path.splitext(os.path.split(imPath)[1])[0] + ".hdf5")
    faceList.append(f)
  return faceList, fnameList

"""
analyze will process images within given video directory.
"""
def analyze(videoPath=None, imagePath=None, k=3, delta_tol=4, sat_tol=5):
  if videoPath is None and imagePath is None:
    raise Exception("One of videoPath/imagePath needs to be non-empty")

  startTime = time.time()
  if imagePath is None:
    faceList, fnameList = faces(videoPath)
  else:
    faceList = [Face(args.image, hdf5FileName=os.path.splitext(os.path.split(args.image)[1])[0] + ".hdf5")]
    fnameList = [os.path.split(args.image)[1]]

  for r, f in enumerate(faceList):
    f.windowName = "image"

    #mypts = f.get_forehead_points()
    #mypts = f.get_nose_keypoints()
    #mypts = f.get_left_cheek_keypoints()
    #mypts = f.get_right_cheek_keypoints()
    #mypts = f.get_face_mask()
    #mypts = f.get_face_keypoints()
    #mypts = f.get_face_until_nose_end()
    cmap = {"Blue": ["BGB", "BG", "GBG", "G"],"Green": ["GYG", "GY", "YGY"], "YellowOrange": ["Y", "YRY"], "Red": ["YR", "RYR", "R", "RPR"]}
    prevMap = {"B": "BGB","BG": "GBG", "G": "GYG", "GY": "YGY", "Y": "YRY", "YR": "RYR", "R": "RPR"}
    nextMap = {"B": "BPB", "BG": "BGB", "G": "GBG", "GY": "GYG", "Y": "YGY", "YR": "YRY", "R": "RYR"}
    finalCMasks = {"Blue": np.zeros(faceList[0].faceMask.shape, dtype=bool), "Green": np.zeros(faceList[0].faceMask.shape, dtype=bool), "YellowOrange": np.zeros(faceList[0].faceMask.shape, dtype=bool), "Red": np.zeros(faceList[0].faceMask.shape, dtype=bool)}
    allPointsCount = 0

    for mypts in [f.get_forehead_points(), f.get_nose_keypoints(), f.get_left_cheek_keypoints(), f.get_right_cheek_keypoints()]:

      f.to_YCrCb()

      medoids, allMasks, allIndices = best_clusters(f, mypts, delta_tol)

      print ("Dividing image into ", len(allMasks), " clusters")

      # Sort masks by decreasing brightness.
      tupleList = [(m, i) for m, i in zip(allMasks, allIndices)]
      stupleList = sorted(tupleList, key=lambda m:np.mean(f.brightImage[m[0]], axis=0)[2])
      resMasks = [m[0] for m in stupleList]
      iMasks = [m[1] for m in stupleList]

      munsellMasks = {"RPR": [], "R": [], "RYR": [], "YR": [], "YRY": [], "Y": [], "YGY": [], "GY": [], "GYG": [], "G": [], "GBG": [], "BG": [], "BGB": [], "B": [], "BPB": []}
      for i, m in enumerate(resMasks):
        meanColor = np.mean(f.image[m], axis=0)
        meanBrightness = np.mean(f.brightImage[m], axis=0)[2]*(100.0/255.0)
        meanSaturation = np.mean(f.satImage[m], axis=0)[1]*(100.0/255.0)
        meanHue = np.mean(f.hueImage[m], axis=0)[0]
        clusterPercent = (np.count_nonzero(m)/np.count_nonzero(mypts))*100.0

        print ("i: ", i)
        print ("Mean color: ", ImageUtils.RGB2HEX(meanColor))
        print ("Percent: ", clusterPercent)
        print ("Cluster Cost Mean: ", np.mean(ImageUtils.delta_e_mask_matrix(medoids[iMasks[i]], f.image[m])))
        print ("Mean Hue: ", meanHue)
        print ("Mean sat: ", meanSaturation)
        print ("Mean brightness: ", meanBrightness)
        print ("mHue: ", ImageUtils.sRGBtoMunsell(meanColor), "\n")
        print ("")

        # Add munsell color hue.
        mHue = ImageUtils.sRGBtoMunsell(meanColor).split(" ")[0]
        rx = re.compile("[A-Z]+")
        hVals = re.findall(re.compile("[A-Z]+"), mHue)
        if len(hVals) == 1:
          if hVals[0] in munsellMasks:
            hueNum = float(mHue.replace(hVals[0], ""))
            if hueNum >= 2.5 and hueNum <= 6:
              munsellMasks[hVals[0]].append(m)
            elif hueNum < 2.5:
              munsellMasks[prevMap[hVals[0]]].append(m)
            else:
              munsellMasks[nextMap[hVals[0]]].append(m)

        print ("")

        if args.show == "True":
          f.show_mask(m)

      print ("Total time taken: ", time.time() - startTime)

      if args.show == "True":
        f.show_orig_image()

      #f.yCrCb_to_sRGB()

      totalPoints = 0
      for mHue in munsellMasks:
        rMask = np.zeros(faceList[0].faceMask.shape, dtype=bool)
        for m in munsellMasks[mHue]:
          rMask = np.bitwise_or(rMask, m)
        totalPoints += np.count_nonzero(rMask)

      allPointsCount += totalPoints

      for mHue in ["BPB", "B", "BGB", "BG", "GBG", "G", "GYG", "GY", "YGY", "Y", "YRY","YR", "RYR","R", "RPR"]:
        if mHue not in munsellMasks:
          continue
        rMask = np.zeros(faceList[0].faceMask.shape, dtype=bool)
        for m in munsellMasks[mHue]:
          rMask = np.bitwise_or(rMask, m)
        pcent = round(((np.count_nonzero(rMask)/totalPoints)*100.0),2)
        if pcent == 0:
          continue

        print ("Munsell Hue: ",mHue, ", Percent: ", pcent, " Saturation: ", round(np.mean(f.satImage[rMask], axis=0)[1]*(100.0/255.0), 2), " Brightness: ", round(np.mean(f.brightImage[rMask], axis=0)[2]*(100.0/255.0), 2), " hue: ", round(np.mean(f.hueImage[rMask], axis=0)[0], 2))

        f.show_mask(rMask)

      print ("\n")

      # Show aggregated clusters.
      for ckey in cmap:
        rMask = np.zeros(faceList[0].faceMask.shape, dtype=bool)
        for mHue in cmap[ckey]:
          if mHue not in munsellMasks:
            continue
          for m in munsellMasks[mHue]:
            rMask = np.bitwise_or(rMask, m)

        pcent = round(((np.count_nonzero(rMask)/totalPoints)*100.0),2)
        if pcent == 0:
          continue

        finalCMasks[ckey] = np.bitwise_or(finalCMasks[ckey], rMask)

        print ("Result Color: ", ckey, ", Percent: ", pcent, " Saturation: ", round(np.mean(f.satImage[rMask], axis=0)[1]*(100.0/255.0), 2), " Brightness: ", round(np.mean(f.brightImage[rMask], axis=0)[2]*(100.0/255.0), 2), " hue: ", round(np.mean(f.hueImage[rMask], axis=0)[0], 2))

        f.show_mask(rMask)

      f.yCrCb_to_sRGB()

    print ("\n")
    for ckey in finalCMasks:
      rMask = finalCMasks[ckey]
      pcent = round(((np.count_nonzero(rMask)/allPointsCount)*100.0),2)
      if pcent == 0:
        continue
      print ("Final Color: ", ckey, ", Percent: ", pcent, " Saturation: ", round(np.mean(f.satImage[rMask], axis=0)[1]*(100.0/255.0), 2), " Brightness: ", round(np.mean(f.brightImage[rMask], axis=0)[2]*(100.0/255.0), 2), " hue: ", round(np.mean(f.hueImage[rMask], axis=0)[0], 2))

      f.show_mask(rMask)

    f.show_orig_image()



"""
best_clusters repeatedly divides each mask until the cluster mean (when calculating delta_e of cluster w.r.t medoid)
is less than the given tolerance.
"""
def best_clusters(f, faceMask, delta_tol):
  k = 2
  allMedoids, allMasks, allIndices, _ = ImageUtils.best_clusters(f.distinct_colors(faceMask, tol=0.0005), f.image, faceMask, k)

  while True:
    masks = []
    medoids = []
    indices = []

    for i, m in zip(allIndices, allMasks):
      if np.mean(ImageUtils.delta_e_mask_matrix(allMedoids[i], f.image[m])) <= delta_tol:
        # No need to sub divide mask.
        masks.append(m)
        indices.append(len(medoids))
        medoids.append(allMedoids[i])
        continue

      cmeds, cmasks, _ , _ = ImageUtils.best_clusters(f.distinct_colors(m, tol=0.0005), f.image, m, k, delta_tol)
      if len(cmasks) == 0:
        # Some kind of exception occrured, just use the mask and move on.
        masks.append(m)
        indices.append(len(medoids))
        medoids.append(allMedoids[i])
        continue

      # Append both sub masks.
      masks.append(cmasks[0])
      indices.append(len(medoids))
      medoids.append(cmeds[0])

      masks.append(cmasks[1])
      indices.append(len(medoids))
      medoids.append(cmeds[1])

    if len(masks) == len(allMasks):
      # Iterations complete, return masks.
      break

    allMasks = masks.copy()
    allIndices = indices.copy()
    allMedoids = medoids.copy()

  return allMedoids, allMasks, allIndices

"""
sub_masks returns masks in each part of face.
"""
def sub_masks(f, mask):
  lcmask = np.bitwise_and(f.get_left_cheek_keypoints(), mask)
  rcmask = np.bitwise_and(f.get_right_cheek_keypoints(), mask)
  fhmask = np.bitwise_and(f.get_forehead_points(), mask)
  nsmask = np.bitwise_and(f.get_nose_keypoints(), mask)
  return [lcmask, rcmask, fhmask, nsmask]

"""
centroids returns centroids associated with given mask. It returns
one centroid (if it exists) per left cheek, right cheek, nose and forehead.
"""
def centroids(f, mask, cdict):
  lcmask = np.bitwise_and(f.get_left_cheek_keypoints(), mask)
  rcmask = np.bitwise_and(f.get_right_cheek_keypoints(), mask)
  lfhmask = np.bitwise_and(f.get_left_forehead_points(), mask)
  rfhmask = np.bitwise_and(f.get_right_forehead_points(), mask)
  lnsmask = np.bitwise_and(f.get_left_nose_points(), mask)
  rnsmask = np.bitwise_and(f.get_right_nose_points(), mask)

  points = []
  for i, m in enumerate([lcmask, rcmask, lfhmask, rfhmask, lnsmask, rnsmask]):
    if (np.count_nonzero(m) == 0):
      continue
    pt = np.mean(np.argwhere(m),axis=0)
    points.append(pt)
    cdict[i].append(pt)

  return points

def skip_shadow_masks(resColor, masks, f):
  print ("")
  resMasks = []
  for i, m in enumerate(masks):
    de = abs(ImageUtils.sRGBtoHSV(resColor)[0, 1]*(100.0/255.0) - np.mean(f.satImage[m], axis=0)[1]*(100.0/255.0))
    print ("Sat diff with : ", i, " is: ", de)
    if de <= 5:
      resMasks.append(m)
      print ("Got index: ", i)

  return resMasks

def print_mask_info(m, resColor, faceMask, f):
  print ("")
  print ("FINAL MASK")
  print ("Final Mask with resColor: ", ImageUtils.RGB2HEX(resColor))
  print ("Percent: ", (np.count_nonzero(m)/np.count_nonzero(faceMask))*100.0)
  print ("Cluster Cost Mean: ", np.mean(ImageUtils.delta_e_mask_matrix(np.mean(f.image[m], axis=0), f.image[m])))
  print ("Mean sat: ", np.mean(f.satImage[m], axis=0)[1]*(100.0/255.0))
  print ("Mean brightness: ", np.mean(f.brightImage[m], axis=0)[2]*(100.0/255.0))
  print ("Mean ratio: ", np.mean(f.ratioImage[m], axis=0)[1])

def new_masks(resColor, masks, f):
  print ("")
  resMasks = []
  for i in range(len(masks)):
    de = np.mean(ImageUtils.delta_e_mask_matrix(resColor, f.image[masks[i]]))
    print ("Delta E with : ", i, " is: ", de)
    if de < 9:
      resMasks.append(masks[i])
      print ("Got index: ", i)

  return resMasks

def print_sats(masks, f):
  print ("")
  for i, m in enumerate(masks):
    print ("Sat mask ", i, " :", np.mean(f.satImage[m], axis=0)[1]*(100.0/255.0))

def res_color(masks, f):
  resColor = [0.0, 0.0, 0.0]
  rMask = np.zeros(f.faceMask.shape, dtype=bool)
  for i, m in enumerate(masks):
    rMask = np.bitwise_or(rMask, m)
    meanColor = np.mean(f.image[m], axis=0)
    resColor = [x[0] + np.count_nonzero(m) * x[1] for x in zip(resColor, meanColor)]

  return [int(x/np.count_nonzero(rMask)) for x in resColor]

def print_color_info(resColor):
  print ("")
  print ("RES COLOR")
  print ("Res Color: ", ImageUtils.RGB2HEX(resColor))
  print ("Res Hue: ", ImageUtils.sRGBtoHSV(resColor)[0, 0]*2)
  print ("Res Sat: ", ImageUtils.sRGBtoHSV(resColor)[0, 1]*(100.0/255.0))
  print ("Res val: ", ImageUtils.sRGBtoHSV(resColor)[0, 2]*(100.0/255.0))
  print ("Res ratio: ", ImageUtils.sRGBtoHSV(resColor)[0, 2]/ImageUtils.sRGBtoHSV(resColor)[0, 1])

if __name__ == "__main__":
  # Parse command line arguments
  parser = argparse.ArgumentParser(description='Video file path')
  parser.add_argument('--video', required=False,metavar="path to video file")
  parser.add_argument('--image', required=False,metavar="path to video file")
  parser.add_argument('--k', required=False,metavar="number of clusters")
  parser.add_argument('--tol', required=False,metavar="tol")
  parser.add_argument('--sat', required=False,metavar="sat")
  parser.add_argument('--bri', required=False,metavar="bri")
  parser.add_argument('--show', default="True", required=False,metavar="show")
  args = parser.parse_args()

  delta_tol = 4
  sat_tol = 5
  bri = 0
  if args.tol is not None:
    delta_tol = int(args.tol)
  if args.sat is not None:
    sat_tol = int(args.sat)

  analyze(args.video, args.image, int(args.k), delta_tol, sat_tol)
