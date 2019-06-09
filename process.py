import os
import sys
import argparse
import skimage.draw
import random
import json
import math
import scipy
import cv2
import time
from skimage.filters import sobel
import numpy as np
from inference import ANNOTATIONS_DIR, OUTPUT_DIR, CLASS, DATA
from imantics import Polygons, Mask
from scipy.interpolate import interp1d
from rdp import rdp
from math_utils import MathUtils
from image_utils import ImageUtils
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
from output import (
  SVG_ATTR,
  SVG_DATA,
  STROKE,
  STROKE_WIDTH,
  FILL,
  CLOSED_PATH,
  SVG_FACE_EAR,
  SVG_HAIR,
  SVG_LEFT_REM_EAR,
  SVG_RIGHT_REM_EAR,
  SVG_LEFT_OPEN_EYE,
  SVG_RIGHT_OPEN_EYE,
  SVG_LEFT_EYEBROW,
  SVG_RIGHT_EYEBROW,
  SVG_LEFT_EYEBALL,
  SVG_RIGHT_EYEBALL,
  SVG_NOSE,
  SVG_LEFT_NOSTRIL,
  SVG_RIGHT_NOSTRIL,
  SVG_UPPER_LIP,
  SVG_LOWER_LIP,
  SVG_FACIAL_HAIR,
  SVG_LEFT_PUPIL,
  SVG_RIGHT_PUPIL,
  SVG_LEFT_UPPER_EYELID,
  SVG_LEFT_LOWER_EYELID,
  SVG_RIGHT_UPPER_EYELID,
  SVG_RIGHT_LOWER_EYELID,
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
It will compute points in clockwise direction but return in
counter clockwise direction.
"""
def merge_face_ear(ann):
  face = []
  pears = []
  hair = []
  nose = []

  for c in ann:
    if c[CLASS] == FACE:
      if len(c[DATA]) > 1:
        print ("More than 1 face image is not supported yet")
        return []
      face = c[DATA][0]
      continue

    if c[CLASS] == EAR:
      if len(c[DATA]) > 1:
        print ("More than 1 ear image is not supported yet")
        return []
      pears.append(c[DATA][0])
      continue

    if c[CLASS] == HAIR_ON_HEAD:
      if len(c[DATA]) > 1:
        hair = merge_hairs(c[DATA])
        break
      hair = c[DATA][0]
      continue

    if c[CLASS] == NOSE:
      if len(c[DATA]) > 1:
        print ("More than 1 nose not supported yet")
        return []
      nose = c[DATA][0]
      continue

  if len(pears) == 0:
    d  = {}
    face = to_points(face)
    d[FACE] = face
    d[EAR] = []
    d[HAIR_ON_HEAD] = to_points(hair)
    d[NOSE] = to_points(nose)
    return rdp(face, epsilon=0.5), [], d

  if len(pears) == 0 or len(pears) > 2:
    print ("Incorrect number of ears: ", len(pears) ," review please!")
    return []

  if len(hair) == 0:
    print ("hair not found: ", len(pears) ," review please!")
    return []

  face = to_points(face)
  ears = []
  for ear in pears:
    ears.append(to_points(ear))
  # dictionary that stores original boundary points.
  d = {}
  d[FACE] = face
  d[EAR] = ears
  d[HAIR_ON_HEAD] = to_points(hair)
  d[NOSE] = to_points(nose)

  allPoints = []
  for ear in ears:
    allPoints += ear
  allPoints += face

  cv = scipy.spatial.ConvexHull(allPoints)

  chull = np.array(allPoints)[cv.vertices]
  tpoints = check_short_transition(ears, face, chull)

  if len(tpoints) == 0:
    print ("Face and ear have no transition points; something must be horribly wrong!")
    return []

  pbetween = []
  usedEpts = [[] for i in range(len(ears))]
  for i in range(1, len(tpoints), 2):
    pfamily = []
    if tpoints[i] in set(face):
      pfamily = face
    elif tpoints[i] in set(ears[0]):
      pfamily = ears[0]
    else:
      pfamily = ears[1]

    j = 0 if i == len(tpoints)-1 else i+1
    dpoints = points_between(tpoints[i], tpoints[j],  pfamily)
    pbetween.append(dpoints)
    if pfamily == ears[0]:
      usedEpts[0] += dpoints
    elif len(ears) > 1 and pfamily == ears[1]:
      usedEpts[1] += dpoints

  vpts = []
  count = 0
  for i in range(0, len(tpoints), 2):
    tpt1, tpt2 = tpoints[i], tpoints[i+1]
    f = tpoints[i] if tpoints[i] in set(face) else tpoints[i+1]
    e = tpoints[i+1] if f == tpoints[i] else tpoints[i]

    evpts = []
    fvpts = []
    hpts = []

    """
    vpts is in clockwise order.
    """
    if MathUtils.is_higher(f, e) and MathUtils.is_right(e, f):
      evpts = vpoints_face_ear(e, ears[0] if e in set(ears[0]) else ears[1] , clockwise=False, left=True, up=True)
      fvpts = vpoints_face_ear(f, face , clockwise=True, left=False, up=False, extraArg=e)
      hpts = fvpts + list(reversed(evpts))
    elif MathUtils.is_higher(f, e) and MathUtils.is_left(e, f):
      evpts = vpoints_face_ear(e, ears[0] if e in set(ears[0]) else ears[1] , clockwise=True, left=False, up=True)
      fvpts = vpoints_face_ear(f, face , clockwise=False, left=True, up=False, extraArg=e)
      hpts = evpts + list(reversed(fvpts))
    elif MathUtils.is_higher(e, f) and MathUtils.is_right(e, f):
      evpts = vpoints_face_ear(e, ears[0] if e in set(ears[0]) else ears[1] , clockwise=True, left=True, up=False)
      fvpts = vpoints_face_ear(f, face , clockwise=False, left=False, up=True, extraArg=e)
      hpts = evpts + list(reversed(fvpts))
    elif MathUtils.is_higher(e, f) and MathUtils.is_left(e, f):
      evpts = vpoints_face_ear(e, ears[0] if e in set(ears[0]) else ears[1] , clockwise=False, left=False, up=False)
      fvpts = vpoints_face_ear(f, face , clockwise=True, left=True, up=True, extraArg=e)
      hpts = fvpts + list(reversed(evpts))

    vpts += hpts
    vpts += pbetween[count]

    count += 1

    if e in set(ears[0]):
      usedEpts[0] += evpts
    elif len(ears) > 1 and e in set(ears[1]):
      usedEpts[1] += evpts

  resPts = rdp(vpts, epsilon=0.5)
  remEarPts = rem_ear_points(usedEpts, ears)

  return list(reversed(resPts)), remEarPts, d

"""
rem_ear_points returns the remaining ear points post merge for
each ear.
"""
def rem_ear_points(usedEpts, ears):
  rem = [[] for i in range(len(usedEpts))]
  for i, ear in enumerate(ears):
    uset = set(usedEpts[i])
    res = []
    for p in ear:
      if p not in uset:
        res.append(p)

    idx = start_ear_point_idx(res)
    vals = []
    ndx = idx
    for j in range(len(res)):
      vals.append(res[ndx])
      ndx = 0 if ndx == len(res)-1 else ndx+1

    rem[i] = fit_spline(vals)
  return rem

"""
start_ear_point_idx returns the starting ear point index
among given ear points. It figures out the corner
points by finding the maximum difference between
consecutive points and choosing the larger y
coordinate value.
"""
def start_ear_point_idx(points):
  dmax = 0
  pos1, pos2 = -1, 0
  for i in range(-1, len(points)-1):
    p1, p2 = points[i], points[i+1]
    d = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    if d > dmax:
      dmax = d
      pos1 = i
      pos2 = i+1

  return pos2

"""
merge_face_hair will merge given face points with hair
label in annotations. All lists of points are in counter clockwise
direction.
"""
def merge_face_hair(mergedPts, d, imdims):
  hair = d[HAIR_ON_HEAD]
  mergedPts = to_points(mergedPts)

  allPoints = hair + mergedPts
  cv = scipy.spatial.ConvexHull(allPoints)
  chull = np.array(allPoints)[cv.vertices]

  tpoints = check_short_transition([hair], mergedPts, chull)

  if len(tpoints) == 0:
    print ("No transiiton with hair; try some other method.")
    return []

  pbetween = []
  for i in range(1, len(tpoints), 2):
    if not tpoints[i] in set(hair):
      continue
    j = 0 if i == len(tpoints)-1 else i+1
    dpoints = points_between(tpoints[i], tpoints[j],  hair)
    pbetween.append(dpoints)

  if len(pbetween) != 1:
    print ("Points between: ", len(pbetween), " is not equal to 1!")
    return []


  vvpts = []
  mergedPtsMask = Polygons([mergedPts]).mask(width=imdims[1], height=imdims[0])
  mergedPtsMask = [(p[1], p[0]) for p in np.argwhere(mergedPtsMask.array)]
  for i in range(0, len(tpoints), 2):
    tpt1, tpt2 = tpoints[i], tpoints[i+1]
    done = False
    for epts in d[EAR]:
      if tpoints[i] in set(epts) or tpoints[i+1] in set(epts):
        e = tpoints[i] if tpoints[i] in set(epts) else tpoints[i+1]
        h = tpoints[i+1] if e == tpoints[i] else tpoints[i]
        gpts = vpoints_hair_ear(h, hair, e, epts, mergedPtsMask, clockwise=MathUtils.is_left(tpoints[i], tpoints[i-2]))
        if MathUtils.is_left(tpoints[i], tpoints[i-2]):
          vvpts.insert(0, gpts)
        else:
          vvpts.append(gpts)
        done = True
        break

    if done:
      continue

    f = tpoints[i] if tpoints[i] in set(d[FACE]) else tpoints[i+1]
    h = tpoints[i+1] if f == tpoints[i] else tpoints[i]

    noseY = int((MathUtils.bottom_most_point(d[NOSE])[1] + MathUtils.top_most_point(d[NOSE])[1])/2)
    gpts = vpoints_hair_face(h, hair, f, mergedPtsMask, mergedPts, noseY, clockwise=MathUtils.is_left(tpoints[i], tpoints[i-2]))
    if MathUtils.is_left(tpoints[i], tpoints[i-2]):
      vvpts.insert(0, gpts)
    else:
      vvpts.append(gpts)

  if len(vvpts) !=2 :
    print ("Edge points: ", len(vvpts), " is not equal to 2!")
    return []

  hairPts = vvpts[0] + pbetween[0] + vvpts[1]

  # Return points in counter clockwise order to be unfiorm.
  return list(reversed(hairPts))

"""
merge_hairs will merge hair annotations for the same instance.
We use a convex hull to find intersection points and then use those
points to merge the two curves. The returned merged hair is in
counterclockwise order.
"""
def merge_hairs(hairAnn):
  hairs = []
  allPoints = []
  for h in hairAnn:
    hr = to_points(h)
    allPoints += hr
    hairs.append(hr)

  cv = scipy.spatial.ConvexHull(allPoints)
  chull = np.array(allPoints)[cv.vertices]

  tpoints = check_short_transition(hairs[1:], hairs[0], chull)

  if len(tpoints) == 0:
    print ("Hair and Hair have no transition points; something must be horribly wrong!")
    return []

  pbetween = []
  for i in range(1, len(tpoints), 2):
    j = 0 if i == len(tpoints)-1 else i+1
    dpoints = points_between(tpoints[i], tpoints[j],  hairs[0] if tpoints[i] in set(hairs[0]) else hairs[1])
    pbetween.append(dpoints)

  vpts = []
  count = 0
  for i in range(0, len(tpoints), 2):
    tpt1, tpt2 = tpoints[i], tpoints[i+1]
    f1 = tpoints[i] if tpoints[i] in set(hairs[0]) else tpoints[i+1]
    f2 = tpoints[i+1] if f1 == tpoints[i] else tpoints[i]

    vpts += list(reversed(pbetween[count]))
    f1pts = move_until(f1, hairs[0], positive=MathUtils.is_left(f1,f2))
    f2pts = move_until(f2, hairs[1], positive=MathUtils.is_left(f2,f1))
    if MathUtils.is_left(f1, f2):
      vpts += f1pts + f2pts
    else:
      vpts += f2pts + f1pts
    count += 1

  return vpts


"""
plot given points on given image.
"""
def view_image(image, points):
  for pts in points:
    set_color(image, pts, 0)

  windowName = "image"
  cv2.namedWindow(windowName,cv2.WINDOW_NORMAL)
  cv2.resizeWindow(windowName, 1024,1024)

  cv2.imshow(windowName, image)
  key = cv2.waitKey(0)

"""
fit_spline fits a cubic spline
through given points. The spline passes through
given points.
"""
def fit_spline(pts, k=3):
  # Define some points:
  points = np.array([[p[0] for p in pts],
                   [p[1] for p in pts]]).T  # a (nbre_points x nbre_dim) array

  # Linear length along the line:
  distance = np.cumsum(np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1)))
  distance = np.insert(distance, 0, 0)/distance[-1]

  # Interpolation.
  alpha = np.linspace(0, 1, len(pts))
  interpolator =  interp1d(distance, points, kind='quadratic', axis=0)

  interpVals = interpolator(alpha)

  return [(int(p[0]),int(p[1])) for p in interpVals]

"""
fit_polynomial fits a polynomial from given set of points.
"""
def fit_polynomial(points, k=3):
  xlims = xlimits(points)
  m = xMap(points)

  xarr = []
  yarr = []
  for x in range(xlims[0], xlims[1]+1):
    if x not in m:
      continue
    xarr.append(x)
    yarr.append(m[x])

  p = np.poly1d(np.polyfit(xarr, yarr, k))
  return p


"""
xMap returns the mapping from x to y for given set of points.
We assume that each x maps to exactly one y in the input points.
"""
def xMap(points):
  if len(points) == 0:
    raise Exception("xMap: Empty input points!")
  m = {}
  for p in points:
    m[p[0]] = p[1]
  return m

"""
xlimits returns the x coordinate limits tuple for the given set of points.
"""
def xlimits(points):
  if len(points) == 0:
    raise Exception("xlims: Empty input points!")

  min_x, max_x = sys.maxsize, 0
  for _, p in enumerate(points):
    min_x = min(p[0], min_x)
    max_x = max(p[0], max_x)

  return (min_x, max_x)

"""
vpoints_face_ear returns the points in the vicinity of given ear or face point. left and up give
the directions along which to move and extraArg is for thresholding constraint in the case
of ear.
"""
def vpoints_face_ear(p, points, clockwise=True, left=True, up=True, extraArg=None):
  vpoints = [p]

  # points contains points in counter clockwise manner.
  idx = MathUtils.find_index(p, points)

  # move in counter clockwise manner.
  pdx = idx
  ndx = MathUtils.next_index(pdx, points, clockwise)

  upfunc = MathUtils.is_higher if up else MathUtils.is_lower
  leftfunc = MathUtils.is_left if left else MathUtils.is_right

  def always_True(p, e):
    return True
  if not extraArg:
    checkfunc = always_True
  else:
    checkfunc = MathUtils.is_higher if not up else MathUtils.is_lower

  while (leftfunc(points[ndx], points[pdx]) or upfunc(points[ndx], points[pdx])) \
    and checkfunc(points[ndx], extraArg):
    vpoints.append(points[ndx])
    pdx = ndx
    ndx = MathUtils.next_index(pdx, points, clockwise)

  return vpoints

"""
vpoints_hair_face returns the extrapolated points from given hair point
to face mask.
"""
def vpoints_hair_face(h, hair, f, faceMask, face, noseY, clockwise=True):
  r = 10 # Num points in vicinity of h to determine polynomial curve. This number is emperical.
  idx = MathUtils.find_index(h, hair)

  polyPoints = [hair[idx]]
  ndx = idx
  for i in range(r):
    ndx = MathUtils.next_index(ndx, hair, clockwise)
    polyPoints.append(hair[ndx])

  # Move up facePoints close to nose Point to get a more accurate merge point.
  ndx = MathUtils.find_index(f, face)
  ndx = MathUtils.next_index(ndx, face, clockwise)
  count = 0
  while face[ndx][1] >= noseY:
    ndx = MathUtils.next_index(ndx,face, clockwise)
    count += 1

  polyPoints.append(face[ndx])

  # Flip vertices so polynomial can be fit.
  g = fit_polynomial([(p[1], p[0]) for p in polyPoints], k =2)
  res = []
  p = (hair[idx][0], hair[idx][1])
  x = hair[idx][1]
  count = 0
  maskSet = set(faceMask)
  maxcount = 1000
  while p not in maskSet and count < maxcount:
    res.append(p)
    x += 1
    p = (int(g(x)), x)
    count += 1

  # add a few more points so intersection is definite.
  for i in range(2*r):
    npi = -i-1 if not clockwise else i+1
    res.append((p[0]+npi, p[1]))

  return res

"""
vpoints_hair_ear returns the extrapolated points from given hair point
to face-ear mask with ear point as additional reference.
"""
def vpoints_hair_ear(h, hair, e, ear, mergedPtsMask, clockwise=True):
  r = 10 # Num points in vicinity of h to determine polynomial curve. This number is emperical.
  idx = MathUtils.find_index(h, hair)

  polyPoints = [hair[idx]]
  ndx = idx
  # Add points before given transition point.
  for i in range(r):
    ndx = MathUtils.next_index(ndx, hair, clockwise)
    polyPoints.append(hair[ndx])

  polyPoints = list(reversed(polyPoints))

  # Add points before given transition point.
  pdx = idx
  ndx = MathUtils.next_index(pdx, hair, not clockwise)
  upfunc = MathUtils.is_lower if not clockwise else MathUtils.is_higher
  rcount = 0
  while upfunc(hair[ndx], hair[pdx]) and rcount < r:
    polyPoints.append(hair[ndx])
    pdx = ndx
    ndx = MathUtils.next_index(pdx, hair, not clockwise)
    rcount += 1

  # Get the top most ear point.
  topEpt = e
  pdx = MathUtils.find_index(e, ear)
  ndx = MathUtils.next_index(pdx, ear, clockwise)
  while MathUtils.is_higher(ear[ndx], ear[pdx]):
    topEpt = ear[ndx]
    pdx = ndx
    ndx = MathUtils.next_index(pdx, ear, clockwise)

  # Append ear point to curve.
  polyPoints.append(topEpt)

  # Flip vertices so polynomial can be fit.
  g = fit_polynomial([(p[1], p[0]) for p in polyPoints], k =2)
  res = []
  p = (hair[idx][0], hair[idx][1])
  x = hair[idx][1]
  maskSet = set(mergedPtsMask)
  while p not in maskSet:
    res.append(p)
    x += 1
    p = (int(g(x)), x)

  res += enter_mask(res[-1], 2*r, maskSet, clockwise)

  # reverse list if clockwise (because it is on the left side)
  if clockwise:
    return list(reversed(res))
  return res

"""
process_nose processes nose annotations and
returns relevant curve.
"""
def process_nose(ann):
  nosePts = []
  for c in ann:
    if c[CLASS] == NOSE:
      if len(c[DATA]) != 1:
        raise Exception("Length of Nose Data points: ", len(c[DATA]), " is not 1")
      nosePts = to_points(c[DATA][0])
      break

  nosePts = to_points(nosePts)
  cv = scipy.spatial.ConvexHull(nosePts)
  chull = np.array(nosePts)[cv.vertices]

  # Find the 3 important points in the convex hull.
  topPt, leftPt, rightPt = find_nose_points(chull)
  topIdx = MathUtils.find_index((topPt[0], topPt[1]), nosePts)
  leftIdx = MathUtils.find_index((leftPt[0], leftPt[1]), nosePts)
  rightIdx = MathUtils.find_index((rightPt[0], rightPt[1]), nosePts)

  # Go to the left from the top most point.
  # Go to the right from top most point.
  leftPts = move_until(nosePts[topIdx], nosePts, positive=False, step=1, discard_pts=0)
  leftNosePt = leftPts[-1]
  rightPts = move_until(nosePts[topIdx], nosePts, positive=True, step=1, discard_pts=0)
  rightNosePt = rightPts[-1]

  # Move one of the left/right nose points further along until
  # Their y coordinate is less than equal to the other's y coordinate.
  if leftNosePt[1] != rightNosePt[1]:
    higherPt = leftNosePt if leftNosePt[1] < rightNosePt[1] else rightNosePt
    lowerPt = leftNosePt if leftNosePt[1] > rightNosePt[1] else rightNosePt
    clockwise = True if higherPt == rightNosePt else False
    pdx = MathUtils.find_index(higherPt, nosePts)
    ndx = pdx
    while nosePts[ndx][1] <= lowerPt[1]:
      pdx = ndx
      ndx = MathUtils.next_index(pdx, nosePts, clockwise=clockwise)

    if higherPt == leftNosePt:
      leftNosePt = nosePts[pdx]
    else:
      rightNosePt = nosePts[pdx]

  # Get points from left point to right point.
  res = [leftNosePt]
  ndx = MathUtils.find_index(leftNosePt, nosePts)
  fdx = MathUtils.find_index(rightNosePt, nosePts)
  while ndx != fdx:
    ndx = MathUtils.next_index(ndx, nosePts, clockwise=False)
    res.append(nosePts[ndx])

  return res

"""
process_mouth will process all elements of the mouth including
upper lip, lower lip, tongue and teeth. In case lips are not aligned
or a lip is missing, process_mouth will compensate accordingly.
"""
def process_mouth(ann):
  res = []
  upperPts = []
  for c in ann:
    if c[CLASS] == UPPER_LIP:
      if len(c[DATA]) != 1:
        raise Exception("Length of Upper Lip Data points: ", len(c[DATA]), " is not 1")
      res.append(to_points(c[DATA][0]))
      upperPts = to_points(c[DATA][0])
    elif c[CLASS] == LOWER_LIP:
      if len(c[DATA]) != 1:
        raise Exception("Length of Lower Lip Data points: ", len(c[DATA]), " is not 1")

  # Find left and right points of upper lip.
  lp = MathUtils.left_most_point(upperPts)
  rp = MathUtils.right_most_point(upperPts)
  tp = MathUtils.top_most_point(upperPts)
  bp = MathUtils.bottom_most_point(upperPts)


  lipWidth = bp[1] - tp[1]
  edgeWidth = rp[0] - lp[0]

  topPoint = (int((lp[0]+rp[0])/2), tp[1])
  bottomPoint = (topPoint[0], topPoint[1] + lipWidth)

  r = 2
  leftPoint = (lp[0] -  int(edgeWidth/4) ,  lp[1] + r*lipWidth)
  rightPoint = (rp[0] + int(edgeWidth/4) ,  rp[1] + r*lipWidth)

  leftMidPoint = (int((leftPoint[0]+topPoint[0])/2), topPoint[1] + int((leftPoint[1]-topPoint[1])/4))
  rightMidPoint = (int((rightPoint[0]+topPoint[0])/2), topPoint[1] + int((rightPoint[1]-topPoint[1])/4))


  leftBottomPoint = (int((leftPoint[0]+topPoint[0])/2), leftMidPoint[1] + int(lipWidth))
  righBottomPoint = (int((rightPoint[0]+topPoint[0])/2), rightMidPoint[1] + int(lipWidth))

  return [leftPoint, leftMidPoint, topPoint, rightMidPoint, rightPoint]
  #return [leftPoint, leftMidPoint, topPoint, rightMidPoint, rightPoint,  righBottomPoint, bottomPoint, leftBottomPoint ]

"""
add_smile adds smile to given facial expression.
"""
def add_smile(image, paths, ann):
  leftEyePts = []
  leftEyeballPts = []
  rightEyePts = []
  rightEyeballPts = []
  nosePts = []
  upperLipPts = []
  lowerLipPts = []
  leftEyebrowPts = []
  rightEyebrowPts = []
  facePts = []

  for c in ann:
    if c[CLASS] == EYE_OPEN:
      if len(c[DATA]) != 1:
        raise Exception("Length of Data points for label: ", c[CLASS], " is: ", len(c[DATA]), " which is not 1")
      if len(leftEyePts) == 0:
        leftEyePts = c[DATA][0]
      else:
        rightEyePts = c[DATA][0]
    elif c[CLASS] == EYEBALL:
      if len(c[DATA]) != 1:
        raise Exception("Length of Data points for label: ", c[CLASS], " is: ", len(c[DATA]), " which is not 1")
      if len(leftEyeballPts) == 0:
        leftEyeballPts = c[DATA][0]
      else:
        rightEyeballPts = c[DATA][0]
    elif c[CLASS] == NOSE:
      if len(c[DATA]) != 1:
        raise Exception("Length of Data points for label: ", c[CLASS], " is: ", len(c[DATA]), " which is not 1")
      nosePts = c[DATA][0]
    elif c[CLASS] == UPPER_LIP:
      if len(c[DATA]) != 1:
        raise Exception("Length of Data points for label: ", c[CLASS], " is: ", len(c[DATA]), " which is not 1")
      upperLipPts = c[DATA][0]
    elif c[CLASS] == LOWER_LIP:
      if len(c[DATA]) != 1:
        raise Exception("Length of Data points for label: ", c[CLASS], " is: ", len(c[DATA]), " which is not 1")
      lowerLipPts = c[DATA][0]
    if c[CLASS] == EYEBROW:
      if len(c[DATA]) != 1:
        raise Exception("Length of Data points for label: ", c[CLASS], " is: ", len(c[DATA]), " which is not 1")
      if len(leftEyebrowPts) == 0:
        leftEyebrowPts = c[DATA][0]
      else:
        rightEyebrowPts = c[DATA][0]

  if len(leftEyePts) == 0 or len(leftEyeballPts) == 0 or len(rightEyePts) == 0 \
    or len(rightEyeballPts) == 0 or len(nosePts) == 0 or len(upperLipPts) == 0 \
    or len(lowerLipPts) == 0 or len(leftEyebrowPts) == 0 or \
    len(rightEyebrowPts) == 0:
    raise Exception("Some face properties for smile may be missing")


  if not MathUtils.is_left(MathUtils.centroid(leftEyeballPts), MathUtils.centroid(rightEyeballPts)):
    temp = leftEyeballPts
    leftEyeballPts = rightEyeballPts
    rightEyeballPts = temp

  if not MathUtils.is_left(MathUtils.centroid(leftEyePts), MathUtils.centroid(rightEyePts)):
    temp = leftEyePts
    leftEyePts = rightEyePts
    rightEyePts = temp

  if not MathUtils.is_left(MathUtils.centroid(leftEyebrowPts), MathUtils.centroid(rightEyebrowPts)):
    temp = leftEyebrowPts
    leftEyebrowPts = rightEyebrowPts
    rightEyebrowPts = temp

  # Draw smiling face.
  ebrowpts = draw_eyebrow_expr(image, paths, ann, leftEyebrowPts, rightEyebrowPts)
  eyepts = draw_eye_expr(image, paths, ann, leftEyePts, rightEyePts, leftEyeballPts, rightEyeballPts)

  return ebrowpts + eyepts

"""
draw_eye_expr will draw eyes and eyeballs with given expression.
For now, it will draw only smile eyes.
"""
def draw_eye_expr(image, paths, ann, leftEyePts, rightEyePts, leftEyeballPts, rightEyeballPts):
  leftPts = draw_eye_helper(image, paths, ann, leftEyePts, leftEyeballPts, left=True)
  rightPts = draw_eye_helper(image, paths, ann, rightEyePts, rightEyeballPts, left=False)
  return leftPts + rightPts

def draw_eye_helper(image, paths, ann, leftEyePts, leftEyeballPts, left=True):
  ratio = 10

  leMask = get_mask(image, [leftEyePts])
  leMap = MathUtils.toMap(leMask)
  lelpt, lerpt = MathUtils.left_most_point(leMask), MathUtils.right_most_point(leMask)

  # Draw lower eye arc.
  # Heuristics.
  h = int((lerpt[0] - lelpt[0]+1)/ratio)
  apt = (int((lelpt[0] + lerpt[0])/2), MathUtils.k_point(lelpt, lerpt, 0.5)[1] + h)

  # Draw quadratic interpolation spline through key points.
  # This will be the lower curve.
  f = MathUtils.interp([lelpt, apt, lerpt])
  lcurve = [(x, int(f(x))) for x in range(lelpt[0], lerpt[0]+1)]

  # For upper curve we need to get the height of the eyeball.
  lebMask = get_mask(image, [leftEyeballPts])
  lebMap = MathUtils.toMap(lebMask)
  leblpt, lebrpt = MathUtils.left_most_point(lebMask), MathUtils.right_most_point(lebMask)
  np = 3
  # Take np points from the left and right and
  cpoints = []
  for x in range(leblpt[0], leblpt[0]+np):
    if x not in lebMap:
      continue
    cpoints.append((x, lebMap[x][-1]))
  for x in range(lebrpt[0], lebrpt[0]-np, -1):
    if x not in lebMap:
      continue
    cpoints.append((x, lebMap[x][-1]))

  center, rad = MathUtils.best_fit_circle(cpoints)

  # Find intersection Point.
  ipt = ()
  for p in lcurve:
    if p[0] == center[0]:
      ipt = p
      break

  # Find upper arc point and then draw the quadratic interpolation.
  upt = (ipt[0], ipt[1] - int(1.6*rad))
  f = MathUtils.interp([lelpt, upt, lerpt])
  ucurve = [(x, int(f(x))) for x in range(lelpt[0]+1, lerpt[0])]

  # Add eyeball from new center.
  newCenter = MathUtils.k_point(ipt, upt, 0.5)
  eyeballPts = MathUtils.circle_points(newCenter, rad, n=int(len(lebMask)/20))

  # Add pupil to eyeball.
  puplilPts =  MathUtils.circle_points(newCenter, int(rad/10), n=10)

  # Add upper eyelid.
  eyd = int((lerpt[0] - lelpt[0]+1)/10)
  if left:
    ledpt = (lelpt[0] -eyd, lelpt[1])
    redpt = (lerpt[0], lerpt[1]-eyd)
  else:
    ledpt = (lelpt[0], lelpt[1]-eyd)
    redpt = (lerpt[0] +eyd, lerpt[1])

  uedpt = (upt[0], upt[1]-eyd)
  sedpt = (upt[0]-eyd, upt[1]-eyd)
  fedpt = (upt[0]+eyd, upt[1]-eyd)

  f = MathUtils.interp([ledpt, sedpt, uedpt, fedpt, redpt])
  uedcurve = [(x, int(f(x))) for x in range(ledpt[0], redpt[0]+1)]

  # Add lower eyelid.
  nd = int((lerpt[0] - lelpt[0]+1)/5)

  if left:
    redpt = (lerpt[0], lerpt[1]+nd)
    ledpt = (ipt[0], ipt[1]+nd)
  else:
    ledpt = (lelpt[0], lelpt[1]+nd)
    redpt = (ipt[0], ipt[1]+nd)

  f = MathUtils.interp([ledpt, redpt])
  ledcurve = [(x, int(f(x))) for x in range(ledpt[0], redpt[0]+1)]

  if left:
    addPointsToPath(image, paths, SVG_LEFT_OPEN_EYE, lcurve + list(reversed(ucurve)) , ann)
    addPointsToPath(image, paths, SVG_LEFT_EYEBALL, eyeballPts, ann)
    addPointsToPath(image, paths, SVG_LEFT_PUPIL, (newCenter, int(rad/3)), ann)
    addPointsToPath(image, paths, SVG_LEFT_UPPER_EYELID, uedcurve, ann)
    addPointsToPath(image, paths, SVG_LEFT_LOWER_EYELID, ledcurve, ann)
  else:
    addPointsToPath(image, paths, SVG_RIGHT_OPEN_EYE, lcurve + list(reversed(ucurve)) , ann)
    addPointsToPath(image, paths, SVG_RIGHT_EYEBALL, eyeballPts, ann)
    addPointsToPath(image, paths, SVG_RIGHT_PUPIL, (newCenter, int(rad/3)), ann)
    addPointsToPath(image, paths, SVG_RIGHT_UPPER_EYELID, uedcurve, ann)
    addPointsToPath(image, paths, SVG_RIGHT_LOWER_EYELID, ledcurve, ann)

  return [lcurve, ucurve, eyeballPts, puplilPts, uedcurve, ledcurve]

"""
draw_eyebrow_expr will draw a given eyebrow expression.
For now, it will only draw smile eyebrow.
"""
def draw_eyebrow_expr(image, paths, ann, leftEyebrowPts, rightEyebrowPts):
  theta = 15
  ratio = 5
  lpts = draw_left_eyebrow_smile(get_mask(image, [leftEyebrowPts]), leftEyebrowPts, theta, ratio)
  addPointsToPath(image, paths, SVG_LEFT_EYEBROW, lpts, ann)

  rpts = draw_right_eyebrow_smile(get_mask(image, [rightEyebrowPts]), rightEyebrowPts, theta, ratio)
  addPointsToPath(image, paths, SVG_RIGHT_EYEBROW, rpts, ann)

  return [lpts, rpts]

"""
draw_left_eyebrow_smile returns the left eyebrow
when the person is smiling normally. theta and ratio
are Heuristics for calculating the smiling eyebrow.
"""
def draw_left_eyebrow_smile(leftEyebrowMask, leftEyebrowPts, theta, ratio):
  lebMap = MathUtils.toMap(leftEyebrowMask)
  llpt, lrpt = MathUtils.left_bottom_point(leftEyebrowPts), MathUtils.right_bottom_point(leftEyebrowPts)

  # Heuristics.
  xwidth = int((lrpt[0] - llpt[0]+1)/ratio)
  given_angle = MathUtils.angle(llpt,  (llpt[0] + xwidth, lebMap[llpt[0] + xwidth][-1]) )
  angle = max(-31, given_angle - theta) # in degrees.
  m = MathUtils.slope(angle)

  # First and second intermediate points on the eyebrow.
  fpt = (llpt[0] + xwidth, int(llpt[1] + xwidth*m))
  spt = MathUtils.k_point(fpt, lrpt, 4/5)

  # Draw cubic interpolation spline through key points.
  # This will be the lower curve.
  f = MathUtils.interp([llpt, fpt, spt, lrpt])
  lcurve = [(x, int(f(x))) for x in range(llpt[0], lrpt[0]+1)]

  # Find left eyebrow upper curve.
  # Using points on lower curve, we maintain the height
  # of the eyebrow at each given x position.
  ucurve = []
  for x in range(llpt[0], lrpt[0]+1):
    if x not in lebMap:
      continue
    ypts = lebMap[x]
    if len(ypts) == 1:
      continue
    yheight = ypts[-1] - ypts[0] + 1
    ucurve.append((x, int(f(x)) - yheight))

  return lcurve + list(reversed(ucurve))

"""
draw_left_eyebrow_smile returns the left eyebrow
when the person is smiling normally. theta and ratio
are Heuristics for calculating the smiling eyebrow.
"""
def draw_right_eyebrow_smile(rightEyebrowMask, rightEyebrowPts, theta, ratio):
  rebMap = MathUtils.toMap(rightEyebrowMask)
  rlpt, rrpt = MathUtils.left_bottom_point(rightEyebrowPts), MathUtils.right_bottom_point(rightEyebrowPts)

  # Heuristics.
  xwidth = int((rrpt[0] - rlpt[0]+1)/ratio)
  given_angle = MathUtils.angle(rrpt,  (rrpt[0] - xwidth, rebMap[rrpt[0] - xwidth][-1]))
  angle = min(37, given_angle + theta)   # in degrees.
  m = MathUtils.slope(angle)

  # First and second intermediate points on the eyebrow.
  fpt = (rrpt[0] - xwidth, int(rrpt[1] - xwidth*m))
  spt = MathUtils.k_point(rlpt, fpt, 1/5)

  # Draw cubic interpolation spline through key points.
  # This will be the lower curve.
  f = MathUtils.interp([rlpt, spt, fpt, rrpt])
  lcurve = [(x, int(f(x))) for x in range(rlpt[0], rrpt[0]+1)]

  # Find left eyebrow upper curve.
  # Using points on lower curve, we maintain the height
  # of the eyebrow at each given x position.
  ucurve = []
  for x in range(rlpt[0], rrpt[0]+1):
    if x not in rebMap:
      continue
    ypts = rebMap[x]
    if len(ypts) == 1:
      continue
    yheight = ypts[-1] - ypts[0] + 1
    ucurve.append((x, int(f(x)) - yheight))

  return lcurve + list(reversed(ucurve))

"""
post_process will take input annotations and post process
them so they can converted to vector graphics.
"""
def post_process(args):
  start = time.time()

  paths = {}
  image = cv2.imread(args.image)
  mergedPts, remEarPts, d = merge_face_ear(ann)
  addPointsToPath(image, paths, SVG_FACE_EAR, mergedPts, ann)

  hairPts = merge_face_hair(mergedPts, d, image.shape[:2])
  addPointsToPath(image, paths, SVG_HAIR, hairPts, ann)

  for c in ann:
    if c[CLASS] == EYE_OPEN:
      if len(c[DATA]) != 1:
        raise Exception("Length of Eye Open Data points: ", len(c[DATA]), " is not 1")
      facePts = to_points(c[DATA][0])
      if SVG_LEFT_OPEN_EYE not in paths:
        addPointsToPath(image, paths, SVG_LEFT_OPEN_EYE, facePts, ann)
      else:
        addPointsToPath(image, paths, SVG_RIGHT_OPEN_EYE, facePts, ann, left=False)
    elif c[CLASS] == EYEBROW:
      if len(c[DATA]) != 1:
        raise Exception("Length of Eyebrow Data points: ", len(c[DATA]), " is not 1")
      eyebrowPts = to_points(c[DATA][0])
      if SVG_LEFT_EYEBROW not in paths:
        addPointsToPath(image, paths, SVG_LEFT_EYEBROW, eyebrowPts, ann)
      else:
        addPointsToPath(image, paths, SVG_RIGHT_EYEBROW, eyebrowPts, ann, left=False)
    elif c[CLASS] == NOSTRIL:
      if len(c[DATA]) != 1:
        raise Exception("Length of Nostril data points: ", len(c[DATA]), " is not 1")
      nostrilPts = to_points(c[DATA][0])
      if SVG_LEFT_NOSTRIL not in paths:
        addPointsToPath(image, paths, SVG_LEFT_NOSTRIL, nostrilPts, ann)
      else:
        addPointsToPath(image, paths, SVG_RIGHT_NOSTRIL, nostrilPts, ann, left=False)
    elif c[CLASS] == UPPER_LIP:
      if len(c[DATA]) != 1:
        raise Exception("Length of Upper Lip data points: ", len(c[DATA]), " is not 1")
      addPointsToPath(image, paths, SVG_UPPER_LIP, to_points(c[DATA][0]), ann)
    elif c[CLASS] == LOWER_LIP:
      if len(c[DATA]) != 1:
        raise Exception("Length of Lower Lip data points: ", len(c[DATA]), " is not 1")
      addPointsToPath(image, paths, SVG_LOWER_LIP, to_points(c[DATA][0]), ann)

  for c in ann:
    if c[CLASS] == EYEBALL:
      if len(c[DATA]) != 1:
        raise Exception("Length of Eyeball Data points: ", len(c[DATA]), " is not 1")
      eyeballPts = to_points(c[DATA][0])
      if SVG_LEFT_EYEBALL not in paths:
        addPointsToPath(image, paths, SVG_LEFT_EYEBALL, eyeballPts, ann)
      else:
        addPointsToPath(image, paths, SVG_RIGHT_EYEBALL, eyeballPts, ann, left=False)

  nosePts = process_nose(ann)
  addPointsToPath(image, paths, SVG_NOSE, nosePts, ann)

  smilePts = add_smile(image, paths, ann)

  if len(remEarPts) > 0:
    addPointsToPath(image, paths, SVG_LEFT_REM_EAR, remEarPts[0], ann)
  if len(remEarPts) > 1:
    addPointsToPath(image, paths, SVG_RIGHT_REM_EAR, remEarPts[1], ann, left=False)

  #mouthPts = process_mouth(ann)
  #addPointsToPath(image, paths, SVG_LOWER_LIP, mouthPts, ann)

  print ("Time taken: ", time.time() - start)

  #view_image(image, [mergedPts, hairPts, nosePts] + remEarPts)
  view_image(image, smilePts)
  with open("paths.json", "w") as outputfile:
    json.dump(paths, outputfile)

"""
addPointsToPath will add given points to given path dictionary. In addition,
it will also segregate points in the label and add corresponding
attributes to the path.
"""
def addPointsToPath(image, paths, k, points, ann, left=True):
  attr = {STROKE: "#000", STROKE_WIDTH: 0, FILL:"none", CLOSED_PATH: False}
  if k != SVG_LEFT_EYEBALL and k != SVG_RIGHT_EYEBALL and k != SVG_HAIR:
    if k == SVG_LEFT_OPEN_EYE or k == SVG_RIGHT_OPEN_EYE or k == SVG_LEFT_LOWER_EYELID \
    or k == SVG_RIGHT_LOWER_EYELID:
      attr[STROKE_WIDTH] = 1
    elif k == SVG_LEFT_EYEBROW or k == SVG_RIGHT_EYEBROW or k == SVG_NOSE or \
    k == SVG_LEFT_REM_EAR or k == SVG_RIGHT_REM_EAR:
      attr[STROKE_WIDTH] = 2
    elif k == SVG_LEFT_UPPER_EYELID or k == SVG_RIGHT_UPPER_EYELID:
      attr[STROKE_WIDTH] = 3
    else:
      attr[STROKE_WIDTH] = 4

  paths[k] = {}
  paths[k] = {SVG_DATA: [], SVG_ATTR: []}
  paths[k][SVG_DATA].append(points)

  # Close paths for certain labels.
  if k == SVG_FACE_EAR or k == SVG_LEFT_EYEBALL or k == SVG_RIGHT_EYEBALL or \
    k == SVG_LEFT_NOSTRIL or k == SVG_RIGHT_NOSTRIL or k == SVG_LEFT_OPEN_EYE or\
    k == SVG_RIGHT_OPEN_EYE or k == SVG_LEFT_EYEBROW or k == SVG_RIGHT_EYEBROW or\
    k == SVG_UPPER_LIP or k == SVG_LOWER_LIP or k == FACIAL_HAIR or k == SVG_LEFT_EYEBALL \
    or k == SVG_RIGHT_EYEBALL:
    attr[CLOSED_PATH] = True

  if k == SVG_NOSE or k == SVG_LEFT_UPPER_EYELID or k == SVG_LEFT_LOWER_EYELID or \
  k == SVG_RIGHT_LOWER_EYELID or k == SVG_RIGHT_UPPER_EYELID or k == SVG_LEFT_LOWER_EYELID or \
  k == SVG_RIGHT_LOWER_EYELID or k == SVG_LEFT_REM_EAR or k == SVG_RIGHT_REM_EAR or \
  k == SVG_LEFT_PUPIL or k == SVG_RIGHT_PUPIL:
    paths[k][SVG_ATTR].append(attr)
    return
  if k == SVG_LEFT_NOSTRIL or k == SVG_RIGHT_NOSTRIL or k == SVG_LEFT_EYEBALL or k == SVG_RIGHT_EYEBALL:
    attr[FILL] = "#000"
    paths[k][SVG_ATTR].append(attr)
    return
  if k == SVG_LEFT_OPEN_EYE or k == SVG_RIGHT_OPEN_EYE:
    attr[FILL] = "#fff"
    paths[k][SVG_ATTR].append(attr)
    return

  # Remaining labels will be shaded according to inherent colors.
  # Get annotation mask for given label.
  bpts = get_ann_points(ann, label_map(k), left)
  mask = get_mask(image, bpts)

  # CV2 image is BGR by default; convert it to RGB.
  img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  # Segrate points divides points based on 2 dominant colors in the image.
  pts1, pts2 = MathUtils.segregate_points(img, mask)
  maxpts = pts1 if len(pts1) >= len(pts2) else pts2
  minpts = pts2 if len(pts1) >= len(pts2) else pts1

  # Fill out maximum color.
  attr[FILL] = ImageUtils.rgb_to_hex(ImageUtils.avg_color(img, maxpts))
  if k == SVG_FACE_EAR:
    attr[STROKE_WIDTH] = 5
  paths[k][SVG_ATTR].append(attr)

  # No need to calculate shading for labels other than face, hair and left and right ears.
  if k != SVG_FACE_EAR and k != SVG_HAIR and k != SVG_LEFT_REM_EAR and k != SVG_RIGHT_REM_EAR \
    and k != SVG_FACIAL_HAIR:
    return

  # Find clusters for minpts.
  clusters = []
  cdict = MathUtils.make_clusters(minpts)
  for key in cdict:
    clusters.append(cdict[key])

  # Choose top M clusters for drawing. Sorting is based on size of cluster.
  M = 3
  clusters = sorted(clusters, key=len, reverse=True)[:M]

  clusters = MathUtils.cv2_boundary_points(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), clusters)

  # Subdivide boundary points into delta intervals. Value of delta is emperical.
  temp = []
  for pts in clusters:
    temp.append([(int(pts[i][0]), int(pts[i][1])) for i in range(len(pts))])
  clusters = temp

  # Add new curves paths.
  paths[k][SVG_DATA] += clusters

  mincolor = ImageUtils.rgb_to_hex(ImageUtils.avg_color(img, minpts))
  for i in range(len(clusters)):
    attr_copy = attr.copy()
    attr_copy[STROKE] = "none"
    attr_copy[FILL] = mincolor
    attr_copy[CLOSED_PATH] = True
    paths[k][SVG_ATTR].append(attr_copy)

  if  k == SVG_LEFT_REM_EAR or k == SVG_RIGHT_REM_EAR:
    paths[k][SVG_DATA] = list(reversed(paths[k][SVG_DATA]))
    paths[k][SVG_ATTR] = list(reversed(paths[k][SVG_ATTR]))

"""
get_mask returns mask for given input list of polygons.
Note that the input is a list of list of points.
"""
def get_mask(image, bpts):
  mask = Polygons(bpts).mask(width=image.shape[1], height=image.shape[0])
  return [(int(p[1]), int(p[0])) for p in np.argwhere(mask.array)]

"""
label_map returns annotation label for given svg label.
"""
def label_map(k):
  if k == SVG_HAIR:
    return HAIR_ON_HEAD
  if k == SVG_FACE_EAR:
    return FACE
  if k == SVG_LEFT_EYEBROW or k == SVG_RIGHT_EYEBROW:
    return EYEBROW
  if k == SVG_LEFT_EYEBALL or k == SVG_RIGHT_EYEBALL:
    return EYEBALL
  if k == SVG_LEFT_REM_EAR or k == SVG_RIGHT_REM_EAR:
    return EAR
  if k == SVG_UPPER_LIP:
    return UPPER_LIP
  if k == SVG_LOWER_LIP:
    return LOWER_LIP
  if k == SVG_FACIAL_HAIR:
    return FACIAL_HAIR
  raise Exception("label_map: SVG Label: ", k, " not found!")

"""
get_ann_points returns annotation points for given label
from ann. left indicates that the first instance of the label
needs to be returned and right indicates that the second instance
needs to be returned. This is to distinguish between left and right
eyebrows, eyes, eyeballs and ears.
"""
def get_ann_points(ann, label, left=True):
  idx = 0 if left else 1
  for c in ann:
    if c[CLASS] == label:
      if idx == 0:
       return c[DATA]
      idx = 0
  raise Exception("Label: ", label, "  not found in annotations!")

"""
find_nose_points will return the leftmost, rightmost
and topmost point indices amond the given Convex hull nose points.
"""
def find_nose_points(nosePts):
  yminIdx = 0
  xminIdx = 0
  xmaxIdx = 0
  for i in range(1, len(nosePts)):
    p = nosePts[i]
    if p[1] < nosePts[yminIdx][1]:
      yminIdx = i
    if p[0] < nosePts[xminIdx][0]:
      xminIdx = i
    if p[0] > nosePts[xmaxIdx][0]:
      xmaxIdx = i

  return nosePts[yminIdx], nosePts[xminIdx], nosePts[xmaxIdx]

"""
move_until will move starting from given point
along given label in given direction until
violation of x direction constraint. step specifies
the number of points to be moved in given direction.
If step is default value, then it is calculated. Returns
points in counter-clockwise order.
"""
def move_until(p, points, positive=True, step=-1, discard_pts=2):
  r = 5
  res = [p]
  clockwise = not is_counter_clockwise(p, points, positive)
  func = MathUtils.is_right if positive else MathUtils.is_left
  pp = p
  np = move_x_points(pp, points, clockwise, step)
  while func(np, pp):
    res.append(np)
    pp = np
    np = move_x_points(pp, points, clockwise, step)

  if len(res) > discard_pts and discard_pts != 0:
    res = res[:-discard_pts]

  if clockwise:
    return list(reversed(res))
  return res

"""
move_x_points will move from given starting point
in the given direction in intervals based on number
of points in the label
"""
def move_x_points(p, points, clockwise=True, step = -1):
  if step != -1:
    r = step
  else:
    r = 1 if len(points) < 10 else int(len(points)/20)
  idx = MathUtils.find_index(p, points)
  ndx = idx
  for i in range(r):
    ndx = MathUtils.next_index(ndx, points, clockwise)
  return points[ndx]

"""
is_clockwise returns true if moving in the given
x direction is clockwise or counterclockwise movement
of the label.
"""
def is_counter_clockwise(p, points, positive=True):
  d = 1 if positive else -1
  r = 5
  idx = MathUtils.find_index(p, points)
  ndx = idx
  for i in range(r):
    ndx = MathUtils.next_index(ndx, points)

  return (points[ndx][0] - p[0])*d >= 0

"""
enter_mask will go further a little
and then draw the perpendicular line from that point.
This is to ensure that intersections are not missed.
"""
def enter_mask(p, k, maskSet, clockwise):
  # Try to move in Diagonal direction
  # and downward.
  yd = 1
  xd = 1 if clockwise else -1
  x, y = p
  res = []
  for i in range(1,k+1):
    res.append((x+xd*i, y+yd*i))
  return res

"""
slope returns slope between given points.
"""
def slope(p1, p2):
  x1, y1 = p1
  x2, y2 = p2
  if x1 == x2:
    return None
  return (y2-y1)/(x2-x1)

"""
points_between returns the points between the given indices in the given
points. Note that that point a and point b are assumed to be one after
the other while moving in clockwise direction.
"""
def points_between(a, b, points):
  i = MathUtils.find_index(a, points)
  j = MathUtils.find_index(b, points)

  ndx = i
  res = []
  while True:
    ndx = MathUtils.next_index(ndx, points, clockwise=True)
    if ndx == j:
      break
    res.append(points[ndx])

  return res

"""
check_short_transition is a general transition
function that will transition points in the convex
hull of the given set of points.
"""
def check_short_transition(ears, face, chull):
  fset = set(face)
  eset = set()
  for ear in ears:
    eset.update(ear)
  res = []

  for i in range(len(chull)):
    if transition(tuple(chull[i]), tuple(chull[i-1]), eset, fset):
      res.append(tuple(chull[i-1]))
      res.append(tuple(chull[i]))

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
    res.append((int(p[0]), int(p[1])))
  return res

def set_color(img, points, radius=0):
  r = lambda: random.randint(0,255)
  clr = (r(), r(), r())
  for p in points:
    cv2.circle(img, (p[0], p[1]), 0, clr, radius)


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
    post_process(args)
