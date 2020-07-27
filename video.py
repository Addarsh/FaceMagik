import os
import cv2
import argparse
import numpy as np
from image_utils import ImageUtils
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch

from face import Face
from deepface import DeepFace

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
def analyze(videoPath=None, imagePath=None, k=3, green="False"):
  if videoPath is None and imagePath is None:
    raise Exception("One of videoPath/imagePath needs to be non-empty")

  if imagePath is None:
    faceList = faces(videoPath)
  else:
    faceList = [Face(args.image, hdf5FileName=os.path.splitext(os.path.split(args.image)[1])[0] + ".hdf5")]

  # Set up figure to plot.
  fig = plt.figure()
  ax = fig.add_axes([0, 0, 1, 1])
  ax.set_xlim(0, len(faceList))
  ax.set_ylim(0, k+1)
  ax.axis('off')
  plt.ion()

  if imagePath != "":
    demography = DeepFace.analyze(imagePath, actions = ['gender', 'race'])
    print("Gender: ", demography["gender"])
    print("Race: ", demography["dominant_race"])

  count = 0
  for f in faceList:
    f.windowName = "image"

    if green == "True":
      f.to_YCrCb()

    medoids, allMasks, minCost = ImageUtils.best_clusters(f.distinct_colors(f.rFaceMask), f.image, f.rFaceMask, k)
    print ("Dividing image into K: ", k, " clusters, with MIN COST: ", minCost/k)

    # Sort masks by decreasing brightness.
    allMasks = sorted(allMasks, key=lambda m:-np.mean(f.brightImage[m[0]], axis=0)[2])
    resMasks = [m[0] for m in allMasks]
    iMasks = [m[1] for m in allMasks]

    # Find good mask (neglecting first and last mask).
    gMask = np.zeros(f.faceMask.shape, dtype=bool)
    for i in range(1, len(resMasks)-1):
      gMask = np.bitwise_or(gMask, resMasks[i])

    resColor = [0.0, 0.0, 0.0]
    for i, m in enumerate(resMasks):
      meanColor = np.mean(f.image[m], axis=0)
      print ("Mean color: ", ImageUtils.RGB2HEX(meanColor))
      print ("Percent: ", (np.count_nonzero(m)/np.count_nonzero(f.rFaceMask))*100.0)
      print ("Cluster Cost Mean: ", np.mean(ImageUtils.delta_e_mask_matrix(medoids[iMasks[i]], f.image[m])))
      print ("Mean brightness: ", np.mean(f.brightImage[m], axis=0)[2]*(100.0/255.0))
      print ("Mean sat: ", np.mean(f.satImage[m], axis=0)[1]*(100.0/255.0))
      print ("Std brightness: ", np.std(f.brightImage[m], axis=0)[2]*(100.0/255.0))
      print ("Mean ratio: ", np.mean(f.brightImage[m], axis=0)[2]/np.mean(f.satImage[m], axis=0)[1])
      print ("")

      # Find resultant color.
      if i != 0 and i != len(resMasks) -1:
        resColor = [x[0] + (np.count_nonzero(m)/np.count_nonzero(gMask))*x[1] for x in zip(resColor, meanColor)]

      # Add medoid to plot.
      ax.add_patch(mpatch.Rectangle((count, i), 1, 1, color=ImageUtils.RGB2HEX(np.mean(f.image[m], axis=0))))
      f.show_mask(m)

    print ("Res Color: ", ImageUtils.RGB2HEX(resColor))
    ax.add_patch(mpatch.Rectangle((count, k), 1, 1, color=ImageUtils.RGB2HEX(resColor)))

    print ("")
    for i in range(len(resMasks)):
      print ("Delta E with : ", i, " is: ", ImageUtils.delta_cie2000(np.mean(f.image[resMasks[i]], axis=0), resColor), np.mean(ImageUtils.delta_e_mask_matrix(resColor, f.image[resMasks[i]])))

    print ("Delta E with gMask: ", np.mean(ImageUtils.delta_e_mask_matrix(resColor, f.image[gMask])))

    # Plot medoids.
    plt.show(block=False)

    count += 1
    if f.show_orig_image() == ord("q"):
      break

if __name__ == "__main__":
  # Parse command line arguments
  parser = argparse.ArgumentParser(description='Video file path')
  parser.add_argument('--video', required=False,metavar="path to video file")
  parser.add_argument('--image', required=False,metavar="path to video file")
  parser.add_argument('--k', required=False,metavar="number of clusters")
  parser.add_argument('--green', required=False,metavar="green")
  args = parser.parse_args()

  analyze(args.video, args.image, int(args.k), args.green)
