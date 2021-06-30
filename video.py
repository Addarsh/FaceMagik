import os
import argparse
import numpy as np
from image_utils import ImageUtils
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import time
import re
import random
import string

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
Performs Kmeans clustering of given mask using diffImg and returns 2 submasks.
"""
def Kmeans_1d(diffImg, totalPoints, mask):
  from sklearn.cluster import KMeans

  maskCords = np.transpose(np.nonzero(mask))
  data = diffImg[mask].reshape((-1,1))

  kmeans = KMeans(n_clusters=2).fit(data)
  labels = kmeans.predict(data)
  c1 = kmeans.cluster_centers_[0][0]
  c2 = kmeans.cluster_centers_[1][0]

  # first mask.
  f1Mask = np.zeros(diffImg.shape, dtype=bool)
  xArr = maskCords[labels == 0][:, 0]
  yArr = maskCords[labels == 0][:, 1]
  f1Mask[xArr, yArr] = True

  # second mask.
  f2Mask = np.zeros(diffImg.shape, dtype=bool)
  xArr = maskCords[labels == 1][:, 0]
  yArr = maskCords[labels == 1][:, 1]
  f2Mask[xArr, yArr] = True

  #print ("centroids: ", round(c1,2), round(c2, 2), " percent: ", percentPoints(f1Mask, totalPoints), percentPoints(f2Mask, totalPoints))

  return ((f1Mask, c1), (f2Mask, c2)) if np.mean(diffImg[f1Mask]) > np.mean(diffImg[f2Mask]) else ((f2Mask, c2), (f1Mask, c1))

"""
Computes percent of points in childMask relative to parentMask.
"""
def percent(childMask, parentMask):
  return round((np.count_nonzero(childMask)/np.count_nonzero(parentMask))*100.0, 2)

"""
Computes percent of points in mask relative to given totalPoints.
"""
def percentPoints(mask, totalPoints):
  return round((np.count_nonzero(mask)/totalPoints)*100.0, 2)

"""
Returns the distance between munsell Hue values of given masks for given
YCrCb image. The distance is measured on munsell circle's circumference
with 10 being the distance of 1 Hue (0-10 units).
"""
def munsell_diff(f, mask1, mask2):
  hueIndexMap = {"R": 1, "YR": 2, "Y": 3, "GY": 4, "G": 5, "BG": 6, "B": 7}
  mHue1 = ImageUtils.sRGBtoMunsell(np.mean(f.image[mask1], axis=0)).split(" ")[0]
  hueLetter1 = munsell_hue_letter(mHue1)
  hueNumber1 = munsell_hue_number(mHue1)

  mHue2 = ImageUtils.sRGBtoMunsell(np.mean(f.image[mask2], axis=0)).split(" ")[0]
  hueLetter2 = munsell_hue_letter(mHue2)
  hueNumber2 = munsell_hue_number(mHue2)

  if hueLetter1 == hueLetter2:
    return max(hueNumber1, hueNumber2) - min(hueNumber1, hueNumber2)

  # if mask1 is darker (Y is darker than YR for example) than mask 2.
  mask1Darker = True if np.mean(f.image[mask1], axis=0)[0] < np.mean(f.image[mask2], axis=0)[0] else False
  baseDelta = 10*abs(hueIndexMap[hueLetter1] - hueIndexMap[hueLetter2])
  return hueNumber1 + baseDelta - hueNumber2 if mask1Darker else hueNumber2 + baseDelta - hueNumber1


"""
Helper function to return the munsell hue letter(e.g. R) from given munsell string.
"""
def munsell_hue_letter(s):
  match = re.compile("[^\W\d]").search(s)
  return s[match.start():match.end() + 1]

"""
Helper function to return the munsell hue number (like 7 in 7R) from given munsell string.
"""
def munsell_hue_number(s):
  match = re.compile("[^\W\d]").search(s)
  return float(s[:match.start()])

"""
Helper function to return munsell hue letter(e.g. R) from given YCrCb image and mask.
"""
def munsell_Hue_letter(f, mask):
  mHue = ImageUtils.sRGBtoMunsell(np.mean(f.image[mask], axis=0)).split(" ")[0]
  if mHue != "None":
    mHue = munsell_hue_letter(mHue)
  return mHue

"""
Repeatedly divides mask into clusters using kmeans until difference between
clusters is less than given tolerance. Returns the cluster with the largest
diff value.
"""
def brightest_cluster(diffImg, mask, totalPoints, tol=15, cutoffPercent=1):
  c1Tuple, c2Tuple = Kmeans_1d(diffImg, totalPoints, mask)
  c1Mask, centroid1 = c1Tuple
  c2Mask, centroid2 = c2Tuple

  if percentPoints(c1Mask, totalPoints) < cutoffPercent or percentPoints(c2Mask, totalPoints) < cutoffPercent:
    # end cluster division.
    return mask
  if abs(centroid1 - centroid2) <= tol:
    # end cluster division.
    return mask
  return brightest_cluster(diffImg, c1Mask, totalPoints, tol, cutoffPercent)

"""
Combines masks with the same Munsell Hue letter in given input allMasks dict
and returns a new dict.
"""
def combine_masks_with_same_Hue(f, allMasks):
  allCombMasks = {}
  for mHue in allMasks:
    combMask = np.zeros(f.image.shape[:2], dtype=bool)
    for gm in allMasks[mHue]:
      combMask = np.bitwise_or(combMask, gm)

    allCombMasks[mHue] = combMask
  return allCombMasks

"""
Plot "mean" points of given masks on a copy of given image.
"""
def plot_points(image, allMasks):
  allPoints = []
  for m in allMasks:
    meanPoint = np.mean(np.argwhere(m), axis=0)
    allPoints.append([meanPoint[0], meanPoint[1]])
  return ImageUtils.plot_points_new(image.copy(), allPoints)

"""
Breaks given mask periodically along x-axis to ensure that
when cv2.findContours is applied later, we don't get large concave boundaries in
in the result that then result in incorrect mean points.
"""
def break_mask_periodically(mask):
  coordinates = np.argwhere(mask)
  ymin, xmin = np.min(coordinates, axis=0)
  ymax, xmax = np.max(coordinates, axis=0)

  # If xmax-min < delta, then we don't break up the mask
  # as applying findContours will give us accurate mean point
  # representing the mask.
  delta = 20
  newMask = mask.copy()
  for x in range(xmin, xmax, delta):
    newMask[:, x] = False
  return newMask

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

    # wtfPoints = f.get_face_keypoints()
    #wtfPoints = f.get_face_until_nose_end()
    #wtfPoints = f.get_face_mask()
    #wtfPoints = f.get_face_mask_without_area_around_eyes()
    wtfPoints = f.get_forehead_points()
    #wtfPoints = f.get_left_cheek_keypoints()
    #wtfPoints = f.get_right_cheek_keypoints()
    #wtfPoints = f.get_nose_keypoints()

    diff = np.max(f.image,axis=2).astype(float)
    tol = 10
    cutoffPercent = 5
    count = 0
    currMask = wtfPoints.copy()
    totalPoints = np.count_nonzero(wtfPoints)

    allMasks = {}
    ycrcbImage = ImageUtils.to_YCrCb(f.image)
    hueMap = {}
    while True:
      bMask = brightest_cluster(diff, currMask, totalPoints, tol=tol, cutoffPercent=cutoffPercent)
      satBriProd = (np.mean(f.brightImage[bMask], axis=0)[2]/255.0) * (np.mean(f.satImage[bMask], axis=0)[1]/255.0) *100.0
      munsellColor = ImageUtils.sRGBtoMunsell(np.mean(ycrcbImage[bMask], axis=0))
      mHue = "None"
      if munsellColor != "None":
        mHue = munsell_hue_letter(munsellColor)
      if mHue not in hueMap:
        hueMap[mHue] = bMask
      else:
        hueMap[mHue] = np.bitwise_or(hueMap[mHue], bMask)

      print ("\npercent " + str(count) + ": ", percentPoints(bMask, totalPoints), " sat val: ", round(np.mean(f.satImage[bMask], axis=0)[1]*(100.0/255.0),2), " bright val: ", round(np.mean(f.brightImage[bMask], axis=0)[2]*(100.0/255.0),2), " hue val: ", round(ImageUtils.sRGBtoHSV(np.mean(f.image[bMask], axis=0))[0, 0]*2,2),  " sat bri prod: ", round(satBriProd, 2), " munsell: ", munsellColor, "\n")
      f.show_mask(bMask)
      brokenMask = break_mask_periodically(bMask)
      f.show_mask(brokenMask)

      # Find boundary masks.
      allBoundaryMasks = ImageUtils.find_boundaries(brokenMask)
      if len(allBoundaryMasks) > 0:
        f.show_masks(allBoundaryMasks)
        f.show(plot_points(f.image, allBoundaryMasks))
      else:
        print ("No boundary points!")

      currMask = np.bitwise_xor(currMask, bMask)
      if percentPoints(currMask, totalPoints) < 1:
        break
      count += 1

    f.show_orig_image()
    for mHue in hueMap:
      combMask = hueMap[mHue]
      satBriProd = (np.mean(f.brightImage[combMask], axis=0)[2]/255.0) * (np.mean(f.satImage[combMask], axis=0)[1]/255.0) *100.0
      munsellColor = ImageUtils.sRGBtoMunsell(np.mean(ycrcbImage[combMask], axis=0))
      print ("\npercent: ", percentPoints(combMask, totalPoints), " hue: ", mHue," sat val: ", round(np.mean(f.satImage[combMask], axis=0)[1]*(100.0/255.0),2),  " bright val: ", round(np.mean(f.brightImage[combMask], axis=0)[2]*(100.0/255.0),2), " hue val: ", round(ImageUtils.sRGBtoHSV(np.mean(f.image[combMask], axis=0))[0, 0]*2,2), " sat bri prod: ", round(satBriProd, 2), " munsell: ", munsellColor)
      f.show_mask(combMask)

    f.show_orig_image()


"""
Converts sRGB to YCrCb and then uses "diffImage" to come up with small clusters that
are close to each other. These small clusters are then combined based on either their
closeness to the corresponding Munsell Hue cluster or their preceeding cluster.
Closeness is calculated using delta_cie2000 and in YCrCb space. If the min delta_cie2000
value is greater than 4, we break that cluster into its own sub cluster and repeat
iterations until there are no changes. Usually can take up to 10 iterations
for the final clusters to converge. The smaller the value of "tol", the larger
the number of iterations until convergence.
Pros: Method is able to cluster out masks based on saturation differences that
humans can observe even if both masks have similar brightness values.
Cons: Takes longer to run and since humans are more sensitive to brightness values,
it's not clear if there is much benefit to the extra accuracy of saturation
differences between sub clusters. It might be useful once we find the primary
cluster within using the other brightness based algorithm.
"""
def cluster_wtfPoints_in_YCrCb(f, wtfPoints):
  f.to_YCrCb()
  diff = (f.image[:, :, 0].astype(float)*1 + f.image[:, :, 2].astype(float)*0) - (f.image[:, :, 1].astype(float))*0
  tol = 3
  count = 1
  currMask = wtfPoints.copy()
  totalPoints = np.count_nonzero(wtfPoints)

  allMasks = {}
  while True:
    bMask = brightest_cluster(diff, currMask, totalPoints, tol=tol)
    mHue = munsell_Hue_letter(f, bMask)
    if mHue not in allMasks:
      allMasks[mHue] = [bMask]
    else:
      allMasks[mHue].append(bMask)
    print ("\npercent " + str(count) + ": ", percentPoints(bMask, totalPoints), " redness val: ", round(np.mean(diff[bMask]),2), " sat val: ", round(np.mean(f.satImage[bMask], axis=0)[1]*(100.0/255.0),2), " munsell: ", ImageUtils.sRGBtoMunsell(np.mean(f.image[bMask], axis=0)), " bright val: ", round(np.mean(f.image[bMask], axis=0)[0]*(100.0/255.0),2), "\n")
    #f.show_mask(bMask)
    currMask = np.bitwise_xor(currMask, bMask)
    if percentPoints(currMask, totalPoints) < 1:
      break
    count += 1

  times = 0

  while True:
    # Create allCombMasks dictionary.
    allCombMasks = combine_masks_with_same_Hue(f, allMasks)
    for mHue in allCombMasks:
      combMask = allCombMasks[mHue]
      print ("\nmHue: ", mHue, " percent: ", percent(combMask, wtfPoints), " bright val: ", round(np.mean(f.image[combMask], axis=0)[0]*(100.0/255.0),2), " sat val: ", round(np.mean(f.satImage[combMask], axis=0)[1]*(100.0/255.0),2), "\n")
      #f.show_mask(combMask)

    # Use allCombMasks and prevMask info to decide which cluster a mask belongs to.
    allNewMasks = {}
    prevMask = np.zeros(f.image.shape[:2], dtype=bool)
    prevNewHue = ""
    numChanges = 0

    for mHue in allMasks:
      print ("\nWe are in Hue: ", mHue, "\n")
      for gm in allMasks[mHue]:
        prev_delta_cie = 0 if np.count_nonzero(prevMask) == 0 else ImageUtils.delta_cie2000(np.mean(f.image[gm], axis=0), np.mean(f.image[prevMask], axis=0))
        print ("\npercent: ", percent(gm, wtfPoints), " delta cie prev: ", round(prev_delta_cie, 2), " mask hue: ", ImageUtils.sRGBtoMunsell(np.mean(f.image[gm], axis=0)).split(" ")[0])

        bestHue = ""
        min_delta_cie = 100
        for cbHue in allCombMasks:
          delta_hue = ImageUtils.delta_cie2000(np.mean(f.image[gm], axis=0), np.mean(f.image[allCombMasks[cbHue]], axis=0))
          if delta_hue < min_delta_cie:
            min_delta_cie = delta_hue
            bestHue = cbHue
          print ("delta cie with ", cbHue, " is: ", round(delta_hue, 2))

        # Can be removed if not useful. Generally I'm seeing better
        # cluster differentiation (like IMG_5216/5217) because of this.
        mega_tol = 4
        if min_delta_cie >= mega_tol:
          # if prev cluster is not part of the bestHue cluster and has
          # good delta_cie with current cluster, merge with it.
          #if prevNewHue != "" and prevNewHue != bestHue and prev_delta_cie < mega_tol:
          #  bestHue = prevNewHue
          if prevNewHue != "" and prev_delta_cie < mega_tol:
            bestHue = prevNewHue
          else:
            # Should keep this as its own cluster perhaps.
            #bestHue = ImageUtils.sRGBtoMunsell(np.mean(f.image[gm], axis=0)).split(" ")[0]
            bestHue = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))

        if bestHue != mHue:
          numChanges += 1

        if bestHue not in allNewMasks:
          allNewMasks[bestHue] = [gm]
        else:
          allNewMasks[bestHue].append(gm)

        print ("\n")
        #f.show_mask(gm)
        prevMask = gm
        prevNewHue = bestHue


    allMasks = allNewMasks.copy()
    times += 1
    print ("\nNum changes: ", numChanges, " after num times: ", times, "\n")
    if numChanges == 0:
      break

  #f.yCrCb_to_sRGB()

  allCombMasks = combine_masks_with_same_Hue(f, allMasks)
  prevCombMask = np.zeros(f.image.shape[:2], dtype=bool)
  diff = (f.image[:, :, 0].astype(float)*1 + f.image[:, :, 2].astype(float)*1) - (f.image[:, :, 1].astype(float))*2
  for mHue in allCombMasks:
    combMask = allCombMasks[mHue]
    munsellDiff = 0 if np.count_nonzero(prevCombMask) == 0 else munsell_diff(f, combMask, prevCombMask)
    satBriProd = (np.mean(f.brightImage[combMask], axis=0)[2]/255.0) * (np.mean(f.satImage[combMask], axis=0)[1]/255.0) *100.0
    print ("\nmHue: ", mHue, " percent: ", percent(combMask, wtfPoints), " bright val: ", round(np.mean(f.image[combMask], axis=0)[0],2), " sat val: ", round(np.mean(f.satImage[combMask], axis=0)[1],2), " hue val: ", round(np.mean(f.hueImage[combMask], axis=0)[0],2), " redness: ", round(np.mean(diff[combMask]),2), " ratio: ", round(np.mean(f.image[combMask], axis=0)[0]/np.mean(f.satImage[combMask], axis=0)[1] , 2),  " munsell hue: ", ImageUtils.sRGBtoMunsell(np.mean(f.image[combMask], axis=0)).split(" ")[0], " munsell diff: ", round(munsellDiff,2), " std bri prod: ", round(satBriProd,2), "\n")
    f.show_mask(combMask)
    prevCombMask = combMask

  f.show_orig_image()

  f.yCrCb_to_sRGB()

  print ("\n\n")

  diff = (f.image[:, :, 0].astype(float)*1 + f.image[:, :, 2].astype(float)*1) - (f.image[:, :, 1].astype(float))*2
  prevCombMask = np.zeros(f.image.shape[:2], dtype=bool)
  for mHue in allCombMasks:
    combMask = allCombMasks[mHue]
    delta_ratio = 0 if np.count_nonzero(prevCombMask) == 0 else abs(np.mean(f.brightImage[combMask], axis=0)[2] - np.mean(f.brightImage[prevCombMask], axis=0)[2])/abs(np.mean(f.satImage[combMask], axis=0)[1] - np.mean(f.satImage[prevCombMask], axis=0)[1])
    delta_cie = 0 if np.count_nonzero(prevCombMask) == 0 else ImageUtils.delta_cie2000(np.mean(f.image[combMask], axis=0), np.mean(f.image[prevCombMask], axis=0))
    satBriProd = (np.mean(f.brightImage[combMask], axis=0)[2]/255.0) * (np.mean(f.satImage[combMask], axis=0)[1]/255.0) *100.0
    print ("\nmHue: ", mHue, " percent: ", percent(combMask, wtfPoints), " bright val: ", round(np.mean(f.brightImage[combMask], axis=0)[2]*(100.0/255.0),2), " sat val: ", round(np.mean(f.satImage[combMask], axis=0)[1]*(100.0/255.0),2), " hue val: ", round(np.mean(f.hueImage[combMask], axis=0)[0],2), " redness: ", round(np.mean(diff[combMask]),2), " ratio: ", round(np.mean(f.brightImage[combMask], axis=0)[2]/np.mean(f.satImage[combMask], axis=0)[1] , 2), " std bri prod: ", round(satBriProd,2), " delta cie2000: ", round(delta_cie, 2), "\n")
    f.show_mask(combMask)
    prevCombMask = combMask

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

  return resMasks

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
