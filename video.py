import os
import cv2
import argparse
import numpy as np
from image_utils import ImageUtils
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch

from face import Face

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
def analyze(videoPath=None, imagePath=None, k=3, delta_tol=4, sat_tol=5):
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
  ax.axis('off')
  plt.ion()

  count = 0
  for f in faceList:
    f.windowName = "image"

    faceClone = f.image.copy()
    faceMask = f.get_face_keypoints()
    """
    medoids, allMasks, allIndices = best_clusters(f, faceMask, delta_tol)


    print ("Dividing image into ", len(allMasks), " clusters")

    # Sort masks by decreasing brightness.
    tupleList = [(m, i) for m, i in zip(allMasks, allIndices)]
    stupleList = sorted(tupleList, key=lambda m:np.mean(f.brightImage[m[0]], axis=0)[2])
    resMasks = [m[0] for m in stupleList]
    iMasks = [m[1] for m in stupleList]

    ax.set_ylim(0, len(resMasks)+1)

    cdict = {0: [], 1: [], 2: [], 3: [], 4: [], 5:[]}
    for i, m in enumerate(resMasks):
      meanColor = np.mean(f.image[m], axis=0)
      print ("i: ", i)
      print ("Mean color: ", ImageUtils.RGB2HEX(meanColor))
      print ("Percent: ", (np.count_nonzero(m)/np.count_nonzero(faceMask))*100.0)
      print ("Cluster Cost Mean: ", np.mean(ImageUtils.delta_e_mask_matrix(medoids[iMasks[i]], f.image[m])))
      print ("Mean brightness: ", np.mean(f.brightImage[m], axis=0)[2]*(100.0/255.0))
      print ("Mean sat: ", np.mean(f.satImage[m], axis=0)[1]*(100.0/255.0))
      print ("Mean Hue: ", np.mean(f.hueImage[m], axis=0)[0])
      print ("Mean ratio: ", np.mean(f.ratioImage[m], axis=0)[1])
      print ("Std sat: ", np.std(f.satImage[m], axis=0)[1]*(100.0/255.0))
      print ("Std brightness: ", np.std(f.brightImage[m], axis=0)[2]*(100.0/255.0))
      print ("Medoid: ", ImageUtils.RGB2HEX(medoids[iMasks[i]]))

      if i != 0:
        pdiff = np.mean(f.ratioImage[m], axis=0)[1] - np.mean(f.ratioImage[resMasks[i-1]], axis=0)[1]
        print ("PREV RATIO DIFF: ", pdiff)
        print ("PREV DELTA: ", ImageUtils.average_delta_e_cie2000_masks(f.image[m], f.image[resMasks[i-1]]))
        if pdiff >= 0.4:
          print ("\tLEVEL CHANGE")

      print ("")

      # Add medoid to plot.
      #ax.add_patch(mpatch.Rectangle((count, i), 1, 1, color=ImageUtils.RGB2HEX(np.mean(f.image[m], axis=0))))
      ax.add_patch(mpatch.Rectangle((count, i), 1, 1, color=ImageUtils.RGB2HEX(medoids[iMasks[i]])))
      ImageUtils.plot_points_new(f.image, centroids(f, m, cdict), radius=5)
      for key in cdict:
        ImageUtils.plot_arrowed_line(f.image, cdict[key])
      f.show_mask(m)

    #ax.add_patch(mpatch.Rectangle((count, len(resMasks)), 1, 1, color=ImageUtils.RGB2HEX(res_color(resMasks, f))))
    plt.show(block=False)
    f.show_orig_image() == ord("q")
    """

    f.image = faceClone.copy()
    bri = np.max(f.brightImage[faceMask], axis=0)[2]
    delta = 10
    cd = [{"name": "left cheek"},{"name": "right cheek"},{"name": "forehead"}, {"name": "nose"}]
    cur_sat = 0
    prev_sat = 0
    while bri >= np.min(f.brightImage[faceMask], axis=0)[2]:
      mask = np.bitwise_and(faceMask, np.bitwise_xor(np.where(f.brightImage[:, :, 2] >= bri, True, False), np.where(f.brightImage[:, :, 2] >= bri + delta, True, False)))
      if np.count_nonzero(mask) == 0:
        print ("Skipped")
        bri -= delta
        continue
      print ("bri: ", bri*(100.0/255.0), ", sat mean: ", np.mean(f.satImage[mask], axis=0)[1]*(100.0/255.0) ,", sat std: ", np.std(f.satImage[mask], axis=0)[1]*(100.0/255.0), ", hue mean: ", np.mean(f.hueImage[mask], axis=0)[0], ", ratio mean: ", np.mean(f.ratioImage[mask], axis=0)[1], ", percent: ", (np.count_nonzero(mask)/np.count_nonzero(faceMask))*100.0)
      sbmasks = sub_masks(f, mask)
      for i, m in enumerate(sbmasks):
        if np.count_nonzero(m) == 0 or i != 3:
          continue
        cur_sat = np.mean(f.satImage[m], axis=0)[1]
        print ("\tMask name: ", cd[i]["name"], ", sat mean: ", np.mean(f.satImage[m], axis=0)[1]*(100.0/255.0),  ", delta sat: ", (cur_sat-prev_sat)*(100.0/255.0), ", sat std: ", np.std(f.satImage[m], axis=0)[1]*(100.0/255.0),", percent: ", (np.count_nonzero(m)/np.count_nonzero(f.get_nose_keypoints()))*100.0)

      print ("")
      if f.show_mask(mask) == ord("q"):
        break
      bri -= delta
      prev_sat = cur_sat

    break

    print ("")
    print ("ResColor")
    sMask = resMasks[-1]
    resMasks = resMasks[:-1]
    resColor = res_color(resMasks, f)
    print_color_info(resColor)

    # Plot medoids.
    ax.add_patch(mpatch.Rectangle((count, len(resMasks)+1), 1, 1, color=ImageUtils.RGB2HEX(resColor)))
    plt.show(block=False)
    f.show_masks_comb(resMasks)

    # Find mask of all sat < cutoff.
    ctf = ImageUtils.sRGBtoHSV(resColor)[0, 1]*(100.0/255.0) - sat_tol

    oldrm = np.zeros(f.faceMask.shape, dtype=bool)
    oldsm = sMask.copy()
    while True:
      print ("cutoff sat : ", ctf)
      hm = f.satImage[:, :, 1]*(100.0/255.0) < ctf
      sm = np.bitwise_and(hm, sMask)
      rm = np.bitwise_xor(sMask, sm)

      print_mask_info(sm, resColor, faceMask, f)
      if np.count_nonzero(rm) == 0:
        print ("broke here")
        break

      # Update res_color.
      newColor = res_color(resMasks + [rm], f)
      dec = ImageUtils.delta_cie2000(resColor, newColor)
      print ("\nDELTA BETWEEN OLD RES: ", ImageUtils.delta_cie2000(resColor, newColor))
      print ("RATIO DIFF BETWEEN NEW SPEC MASK and NEW RES COLOR: ", np.mean(f.ratioImage[sm], axis=0)[1] -  ImageUtils.sRGBtoHSV(newColor)[0, 2]/ImageUtils.sRGBtoHSV(newColor)[0, 1])
      print ("MASK PERCENTAGE CHANGE: ", (np.count_nonzero(oldsm)/np.count_nonzero(faceMask))*100.0 - (np.count_nonzero(sm)/np.count_nonzero(faceMask))*100.0)
      f.show_masks_comb(resMasks + [rm])
      #if (np.count_nonzero(oldsm)/np.count_nonzero(faceMask))*100.0 - (np.count_nonzero(sm)/np.count_nonzero(faceMask))*100.0 < 10:
      if np.mean(ImageUtils.delta_e_mask_matrix(np.mean(f.image[oldsm], axis=0), f.image[oldsm])) <= 2 or (np.count_nonzero(oldsm) <= np.count_nonzero(sm)):
        print ("")
        print ("SPECULAR MASK FOUND")
        if np.count_nonzero(oldrm) > 0:
          resMasks.append(oldrm)
        break

      oldrm = rm.copy()
      oldsm = sm.copy()
      resColor = newColor
      ax.add_patch(mpatch.Rectangle((count, len(resMasks)+1), 1, 1, color=ImageUtils.RGB2HEX(resColor)))
      print_color_info(resColor)
      ctf = ImageUtils.sRGBtoHSV(resColor)[0, 1]*(100.0/255.0) - sat_tol

    #nMasks =  new_masks(resColor, resMasks, f)
    nMasks = skip_shadow_masks(resColor, resMasks, f)
    resColor = res_color(nMasks, f)
    print_color_info(resColor)
    print_mask_info(oldsm, resColor, faceMask, f)
    print ("")
    print ("RATIO DIFF: ", np.mean(f.ratioImage[oldsm], axis=0)[1] -  ImageUtils.sRGBtoHSV(resColor)[0, 2]/ImageUtils.sRGBtoHSV(resColor)[0, 1])
    print_sats(resMasks, f)
    ax.add_patch(mpatch.Rectangle((count, len(resMasks)), 1, 1, color=ImageUtils.RGB2HEX(resColor)))
    f.show_masks_comb(nMasks)
    f.show_mask(oldsm, [255, 0, 0])
    count += 1

    #f.image[sMask] = medoids[0]
    #f.image[resMasks[0]] = medoids[1]

    if f.show_orig_image() == ord("q"):
      break

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
  allMasks = [f.get_left_cheek_keypoints(), f.get_right_cheek_keypoints(), f.get_left_forehead_points(), f.get_right_forehead_points(), f.get_left_nose_points(), f.get_right_nose_points()]
  for i, m in enumerate([lcmask, rcmask, lfhmask, rfhmask, lnsmask, rnsmask]):
    if (np.count_nonzero(m)/np.count_nonzero(allMasks[i]))*100.0 < 5:
      continue
    #if (np.count_nonzero(m) == 0):
    #  continue
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
  args = parser.parse_args()

  delta_tol = 4
  sat_tol = 5
  if args.tol is not None:
    delta_tol = int(args.tol)
  if args.sat is not None:
    sat_tol = int(args.sat)

  analyze(args.video, args.image, int(args.k), delta_tol, sat_tol)
