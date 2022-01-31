import os
import argparse
import numpy as np
from image_utils import ImageUtils
import matplotlib.pyplot as plt
import time

from face import Face

"""
Repeatedly divides mask into clusters using kmeans until difference between
clusters is less than given tolerance. Returns the cluster with the largest
diff value. diffImg has dimensions (W, H) and contains the values to peform
clustering on. mask is boolean mask of same dimensions,
"""
def brightest_cluster(diffImg, mask, totalPoints, tol=2, cutoffPercent=2):
  c1Tuple, c2Tuple = ImageUtils.Kmeans_1d(diffImg, totalPoints, mask)
  c1Mask, centroid1 = c1Tuple
  c2Mask, centroid2 = c2Tuple

  if ImageUtils.percentPoints(c1Mask, totalPoints) < cutoffPercent or ImageUtils.percentPoints(c2Mask, totalPoints) < cutoffPercent:
    # end cluster division.
    return mask
  if abs(centroid1 - centroid2) <= tol:
    # end cluster division.
    return mask
  return brightest_cluster(diffImg, c1Mask, totalPoints, tol, cutoffPercent)

def plot_color(image, maskList, totalPoints):
  if len(maskList) > 1:
    plt.close()
  fig = plt.figure()
  ax = fig.add_subplot(111)

  x = [i for i in range(len(maskList))]
  percentList = [ImageUtils.percentPoints(mask, totalPoints) for mask in maskList]
  colorList = [np.mean(image[mask], axis=0)/255.0 for mask in maskList]
  munsellColorList = [ImageUtils.sRGBtoMunsell(np.mean(image[mask], axis=0)) for mask in maskList]
  ax.bar(x=x, height=percentList, color=colorList, align="edge")
  # Set munsell color values on top of value.
  for i in range(len(x)):
    plt.text(i, percentList[i], munsellColorList[i])
  plt.show(block=False)


  ax.bar(x=x, height=percentList, color=colorList, align="edge")
  #ax.set_xticklabels(x, fontsize=8)
  # Set munsell color values on top of value.
  for i in range(len(x)):
    plt.text(i, percentList[i], munsellColorList[i])
  plt.show(block=False)

"""
analyze will process image with the given path. It will augment the image
by given factor of brightness and saturation before processing.
"""
def analyze(imagePath=None, cut=1.0, sat=1.0):
  startTime = time.time()

  # Detect face and show class.
  f = Face(args.image, hdf5FileName=os.path.splitext(os.path.split(args.image)[1])[0] + ".hdf5")

  f.windowName = "image"

  newImg = ImageUtils.set_brightness(f.image, cut)
  newImg = ImageUtils.set_saturation(newImg, sat)
  f.image = newImg

  print ("teeth visible: ", f.is_teeth_visible())
  #f.show_mask(f.get_mouth_points())

  # maskToProcess = f.get_face_keypoints()
  #maskToProcess = f.get_face_until_nose_end()
  #maskToProcess = f.get_face_mask()
  #maskToProcess = f.get_face_mask_without_area_around_eyes()
  maskToProcess = f.get_face_until_nose_end_without_area_around_eyes()
  #maskToProcess = f.get_forehead_points()
  #maskToProcess = f.get_left_cheek_keypoints()
  #maskToProcess = f.get_mouth_points()
  #maskToProcess = f.get_right_cheek_keypoints()
  #maskToProcess = f.get_nose_keypoints()

  tol = 2
  cutoffPercent = 2
  #tol = 2
  #cutoffPercent = 1
  currMask = maskToProcess.copy()
  totalPoints = np.count_nonzero(maskToProcess)

  effectiveColorMap = {}
  allClusterMasks = []
  maskDirectionsList = []
  maskPercentList = []

  ycrcbImage = ImageUtils.to_YCrCb(f.image)
  diff = (ycrcbImage[:, :, 0]).astype(float)

  # Divide the image into smaller clusters.
  while True:
    # Find brightest cluster.
    bMask = brightest_cluster(diff, currMask, totalPoints, tol=tol, cutoffPercent=cutoffPercent)
    # Find least saturated cluster of brightest cluster. This provides more fine grained clusters
    # but is also more expensive. Comment it out if you want to plot "color of each cluster versus
    # the associated munsell hue" to iterate/improve effective color mapping.
    #bMask = brightest_cluster(255.0 -(ImageUtils.to_hsv(ycrcbImage)[:, :, 1]).astype(np.float), bMask, np.count_nonzero(bMask), tol=tol, cutoffPercent=cutoffPercent)

    munsellColor = ImageUtils.sRGBtoMunsell(np.mean(ycrcbImage[bMask], axis=0))
    effectiveColor = f.effective_color(munsellColor)
    if effectiveColor not in effectiveColorMap:
      effectiveColorMap[effectiveColor] = bMask
    else:
      effectiveColorMap[effectiveColor] = np.bitwise_or(effectiveColorMap[effectiveColor], bMask)

    # Store this mask for different computations.
    allClusterMasks.append(bMask)
    maskDirectionsList.append(f.get_mask_direction(bMask))
    maskPercentList.append(ImageUtils.percentPoints(bMask, totalPoints))

    print ("effective color: ", effectiveColor, " brightness: ", round(np.mean(ycrcbImage[:, :, 0][bMask]),2), "\n")
    #f.show(ImageUtils.plot_points_and_mask(ycrcbImage, [f.noseMiddlePoint], bMask))

    currMask = np.bitwise_xor(currMask, bMask)
    if ImageUtils.percentPoints(currMask, totalPoints) < 1:
      break

  print ("\n Light Direction: ", f.process_mask_directions(maskDirectionsList, maskPercentList))

  plot_color(ycrcbImage, allClusterMasks, totalPoints)

  effectiveColorMap = f.iterate_effectiveColorMap(effectiveColorMap, allClusterMasks)
  f.print_effectiveColorMap(effectiveColorMap, totalPoints)

  combinedMasks = f.combine_masks_close_to_each_other(effectiveColorMap)

  print ("\nCombined masks")
  for m in combinedMasks:
    print ("percent: ", ImageUtils.percentPoints(m, totalPoints))
    f.show_mask(m)

  f.show_orig_image()


if __name__ == "__main__":
  # Parse command line arguments
  parser = argparse.ArgumentParser(description='Image for processing')
  parser.add_argument('--image', required=True,metavar="path to video file")
  parser.add_argument('--cut', required=False,metavar="cut")
  parser.add_argument('--sat', required=False,metavar="sat")
  parser.add_argument('--show', default="True", required=False,metavar="show")
  args = parser.parse_args()

  cut = 1.0
  sat = 1.0
  if args.cut is not None:
    cut = float(args.cut)
  if args.sat is not None:
    sat = float(args.sat)

  analyze(args.image, cut, sat)
