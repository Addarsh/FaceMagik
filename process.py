import os
import argparse
import skimage.draw
import random
import json
import math
import scipy
import cv2
import numpy as np
from inference import ANNOTATIONS_DIR, OUTPUT_DIR, CLASS, DATA
from imantics import Polygons
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

PROCESSED_DIR = os.path.join(OUTPUT_DIR, "processed")

def display_annotations(imagePath, ann):
  image = skimage.io.imread(imagePath)

  r = lambda: random.randint(0,255)
  color_map = {}
  for c in ann:
    if c[CLASS] in color_map:
      continue
    color_map[c[CLASS]] = (r(),r(),r())

  for c in ann:
    polygons = Polygons(c[DATA])
    mask = polygons.mask(image.shape[1], image.shape[0])
    mask.draw(image, color=color_map[c[CLASS]])

  # Save output
  skimage.io.imsave(os.path.join(PROCESSED_DIR, fname+"_x.jpg"), image)

def convexify(imagePath, ann):
  image = skimage.io.imread(imagePath)
  r = lambda: random.randint(0,255)

  polylist = []
  for c in ann:
    if c[CLASS] == FACE or c[CLASS] == EAR:
      for data in c[DATA]:
        polylist += data

  cv = scipy.spatial.ConvexHull(polylist)
  res = []
  res.append(np.array(polylist)[cv.vertices])
  polygons = Polygons(res)
  mask = polygons.mask(image.shape[1], image.shape[0])
  mask.draw(image, color=(r(),r(),r()))

  # Save output
  skimage.io.imsave(os.path.join(PROCESSED_DIR, fname+".jpg"), image)

"""
Merges face and ear. Currently tested only on short hair faces.
"""
def merge_face_ear(imagePath, ann):
  #image = skimage.io.imread(imagePath)
  image = cv2.imread(imagePath)

  face = []
  pears = []
  for c in ann:
    if c[CLASS] == FACE:
      if len(c[DATA]) > 1:
        print ("More than 1 face image is not supported yet")
        return
      face = c[DATA][0]
      continue

    if c[CLASS] == EAR:
      if len(c[DATA]) > 1:
        print ("More than 1 ear image is not supported yet")
        return
      pears.append(c[DATA][0])
      continue

  if len(pears) == 0 or len(pears) > 2:
    print ("Incorrect number of ears: ", len(pears) ," review please!")
    return

  face = to_points(face)
  ears = []
  for ear in pears:
    ears.append(to_points(ear))

  allPoints = []
  for ear in ears:
    allPoints += ear
  allPoints += face

  cv = scipy.spatial.ConvexHull(allPoints)

  chull = np.array(allPoints)[cv.vertices]
  tpoints = check_short_transition(ears, face, chull)

  if len(tpoints) == 0:
    print ("Face and ear have no transition points; something must be horribly wrong!")
    return

  vpts = []
  for i in range(0, len(tpoints), 2):
    tpt1, tpt2 = tpoints[i], tpoints[i+1]
    f = tpoints[i] if tpoints[i] in set(face) else tpoints[i+1]
    e = tpoints[i+1] if f == tpoints[i] else tpoints[i]

    if is_higher(f, e) and is_right(e, f):
      vpts += vpoints_face_ear(e, ears[0] if e in set(ears[0]) else ears[1] , clockwise=False, left=True, up=True)
      vpts += vpoints_face_ear(f, face , clockwise=True, left=False, up=False, extraArg=e)
    elif is_higher(f, e) and is_left(e, f):
      vpts += vpoints_face_ear(e, ears[0] if e in set(ears[0]) else ears[1] , clockwise=True, left=False, up=True)
      vpts += vpoints_face_ear(f, face , clockwise=False, left=True, up=False, extraArg=e)
    elif is_higher(e, f) and is_right(e, f):
      vpts += vpoints_face_ear(e, ears[0] if e in set(ears[0]) else ears[1] , clockwise=True, left=True, up=False)
      vpts += vpoints_face_ear(f, face , clockwise=False, left=False, up=True, extraArg=e)
    elif is_higher(e, f) and is_left(e, f):
      vpts += vpoints_face_ear(e, ears[0] if e in set(ears[0]) else ears[1] , clockwise=False, left=False, up=False)
      vpts += vpoints_face_ear(f, face , clockwise=True, left=True, up=True, extraArg=e)

  set_color(image, vpts, 4)

  cv2.imshow("image", image)
  key = cv2.waitKey(0)

"""
vpoints_face_ear returns the points in the vicinity of given ear or face point. left and up give
the directions along which to move and extraArg is for thresholding constraint in the case
of ear.
"""
def vpoints_face_ear(p, points, clockwise=True, left=True, up=True, extraArg=None):
  vpoints = [p]

  # points contains points in counter clockwise manner.
  idx = find_index(p, points)

  # move in counter clockwise manner.
  pdx = idx
  ndx = next_index(pdx, points, clockwise)

  upfunc = is_higher if up else is_lower
  leftfunc = is_left if left else is_right

  def always_True(p, e):
    return True
  if not extraArg:
    checkfunc = always_True
  else:
    checkfunc = is_higher if not up else is_lower

  while (leftfunc(points[ndx], points[pdx]) or upfunc(points[ndx], points[pdx])) \
    and checkfunc(points[ndx], extraArg):
    vpoints.append(points[ndx])
    pdx = ndx
    ndx = next_index(pdx, points, clockwise)

  return vpoints

"""
next_index returns next index to given index along given direction.
"""
def next_index(i, points, clockwise=False):
  if clockwise:
    return len(points)-1 if i == 0 else i-1
  return 0 if i == len(points)-1 else i+1

"""
check_short_transition will check for a transition from
face to ear or ear to face among the given set of points.
We return all pairs of such transiitons.
"""
def check_short_transition(ears, face, chull):
  fset = set(face)
  res = []

  for ear in ears:
    eset = set(ear)
    for i in range(len(chull)):
      if transition(tuple(chull[i]), tuple(chull[i-1]), eset, fset):
        res.append(tuple(chull[i]))
        res.append(tuple(chull[i-1]))

  return res

"""
transition checks if there is a transition between given points
and returns if so.
"""
def transition(p1, p2, hset, fset):
  return (p1 in hset and p2 in fset) \
    or (p1 in fset and p2 in hset)

"""
to_points converts points (list of lists) to
list of tuples to allow for hashing.
"""
def to_points(points):
  res = []
  for p in points:
    res.append((p[0], p[1]))
  return res

def set_color(img, points, radius=0):
  r = lambda: random.randint(0,255)
  clr = (r(), r(), r())
  for p in points:
    cv2.circle(img, (p[0], p[1]), 0, clr, radius)

"""
Returns true if p1 is higher than p2 (y coord); else returns false.
"""
def is_higher(p1, p2):
  return p1[1] < p2[1]

"""
Returns true if p1 is lower than p2 (y coord); else returns false.
"""
def is_lower(p1, p2):
  return p1[1] > p2[1]

"""
Returns true if p1 is to the right of p2 (x coord); else returns false.
"""
def is_right(p1, p2):
  return p1[0] > p2[0]

"""
Returns true if p1 is to the left of p2 (x coord); else returns false.
"""
def is_left(p1, p2):
  return p1[0] < p2[0]

"""
find_index returns the index of given point among given set of points.
"""
def find_index(p, points):
  for i in range(len(points)):
    if p == points[i]:
      return i
  raise Exception("Point: ", p, " not found in point set: ", points)


if __name__ == "__main__":
  # Parse command line arguments
  parser = argparse.ArgumentParser(
      description='Train Mask R-CNN to detect balloons.')
  parser.add_argument('--image', required=True,
                      metavar="path to test image file",
                      help="path to test image file",)
  parser.add_argument('--op', required=True,
                      metavar="operation",
                      help="operation",)
  args = parser.parse_args()

  fname = os.path.splitext(os.path.split(args.image)[1])[0]
  ann = []
  with open(os.path.join(ANNOTATIONS_DIR, fname+".json"), "r") as f:
    ann = json.load(f)

  if args.op == "display":
    display_annotations(args.image, ann)
  elif args.op == "convex":
    convexify(args.image, ann)
  else:
    merge_face_ear(args.image, ann)
