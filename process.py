import os
import sys
import argparse
import skimage.draw
import random
import json
import math
import scipy
import cv2
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
    if is_higher(f, e) and is_right(e, f):
      evpts = vpoints_face_ear(e, ears[0] if e in set(ears[0]) else ears[1] , clockwise=False, left=True, up=True)
      fvpts = vpoints_face_ear(f, face , clockwise=True, left=False, up=False, extraArg=e)
      hpts = fvpts + list(reversed(evpts))
    elif is_higher(f, e) and is_left(e, f):
      evpts = vpoints_face_ear(e, ears[0] if e in set(ears[0]) else ears[1] , clockwise=True, left=False, up=True)
      fvpts = vpoints_face_ear(f, face , clockwise=False, left=True, up=False, extraArg=e)
      hpts = evpts + list(reversed(fvpts))
    elif is_higher(e, f) and is_right(e, f):
      evpts = vpoints_face_ear(e, ears[0] if e in set(ears[0]) else ears[1] , clockwise=True, left=True, up=False)
      fvpts = vpoints_face_ear(f, face , clockwise=False, left=False, up=True, extraArg=e)
      hpts = evpts + list(reversed(fvpts))
    elif is_higher(e, f) and is_left(e, f):
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
        gpts = vpoints_hair_ear(h, hair, e, epts, mergedPtsMask, clockwise=is_left(tpoints[i], tpoints[i-2]))
        if is_left(tpoints[i], tpoints[i-2]):
          vvpts.insert(0, gpts)
        else:
          vvpts.append(gpts)
        done = True
        break

    if done:
      continue

    f = tpoints[i] if tpoints[i] in set(d[FACE]) else tpoints[i+1]
    h = tpoints[i+1] if f == tpoints[i] else tpoints[i]
    gpts = vpoints_hair_face(h, hair, f, mergedPtsMask, clockwise=is_left(tpoints[i], tpoints[i-2]))
    if is_left(tpoints[i], tpoints[i-2]):
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
    f1pts = move_until(f1, hairs[0], positive=is_left(f1,f2))
    f2pts = move_until(f2, hairs[1], positive=is_left(f2,f1))
    if is_left(f1, f2):
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
    set_color(image, pts, 4)

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
vpoints_hair_face returns the extrapolated points from given hair point
to face mask.
"""
def vpoints_hair_face(h, hair, f, faceMask, clockwise=True):
  r = 10 # Num points in vicinity of h to determine polynomial curve. This number is emperical.
  idx = find_index(h, hair)

  polyPoints = [hair[idx]]
  ndx = idx
  for i in range(r):
    ndx = next_index(ndx, hair, clockwise)
    polyPoints.append(hair[ndx])
  polyPoints.append(f)

  # Flip vertices so polynomial can be fit.
  g = fit_polynomial([(p[1], p[0]) for p in polyPoints], k =2)
  res = []
  p = (hair[idx][0], hair[idx][1])
  x = hair[idx][1]
  count = 0
  maskSet = set(faceMask)
  while p not in maskSet:
    res.append(p)
    x += 1
    p = (int(g(x)), x)

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
  idx = find_index(h, hair)

  polyPoints = [hair[idx]]
  ndx = idx
  # Add points before given transition point.
  for i in range(r):
    ndx = next_index(ndx, hair, clockwise)
    polyPoints.append(hair[ndx])

  polyPoints = list(reversed(polyPoints))

  # Add points before given transition point.
  pdx = idx
  ndx = next_index(pdx, hair, not clockwise)
  upfunc = is_lower if not clockwise else is_higher
  rcount = 0
  while upfunc(hair[ndx], hair[pdx]) and rcount < r:
    polyPoints.append(hair[ndx])
    pdx = ndx
    ndx = next_index(pdx, hair, not clockwise)
    rcount += 1

  # Get the top most ear point.
  topEpt = e
  pdx = find_index(e, ear)
  ndx = next_index(pdx, ear, clockwise)
  while is_higher(ear[ndx], ear[pdx]):
    topEpt = ear[ndx]
    pdx = ndx
    ndx = next_index(pdx, ear, clockwise)

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
  topIdx = find_index((topPt[0], topPt[1]), nosePts)
  leftIdx = find_index((leftPt[0], leftPt[1]), nosePts)
  rightIdx = find_index((rightPt[0], rightPt[1]), nosePts)

  # Go to the left from the top most point.
  # Go to the right from top most point.
  leftPts = move_until(nosePts[topIdx], nosePts, positive=False, step=1, discard_pts=0)
  leftNosePt = leftPts[-1]
  rightPts = move_until(nosePts[topIdx], nosePts, positive=True, step=1, discard_pts=0)
  rightNosePt = rightPts[-1]

  # Get points from left point to right point.
  res = [leftNosePt]
  ndx = find_index(leftNosePt, nosePts)
  fdx = find_index(rightNosePt, nosePts)
  while ndx != fdx:
    ndx = next_index(ndx, nosePts, clockwise=False)
    res.append(nosePts[ndx])

  return res


"""
post_process will take input annotations and post process
them so they can converted to vector graphics.
"""
def post_process(args):
  paths = {}
  image = cv2.imread(args.image)
  mergedPts, remEarPts, d = merge_face_ear(ann)
  addPointsToPath(image, paths, SVG_FACE_EAR, mergedPts, ann)
  addPointsToPath(image, paths, SVG_LEFT_REM_EAR, remEarPts[0], ann)
  if len(remEarPts) > 1:
    addPointsToPath(image, paths, SVG_RIGHT_REM_EAR, remEarPts[1], ann)

  hairPts = merge_face_hair(mergedPts, d, image.shape[:2])
  addPointsToPath(image, paths, SVG_HAIR, hairPts, ann)

  # Add eyes to image.
  for c in ann:
    if c[CLASS] == EYE_OPEN:
      if len(c[DATA]) != 1:
        raise Exception("Length of Eye Open Data points: ", len(c[DATA]), " is not 1")
      facePts = to_points(c[DATA][0])
      if SVG_LEFT_OPEN_EYE not in paths:
        addPointsToPath(image, paths, SVG_LEFT_OPEN_EYE, facePts, ann)
      else:
        addPointsToPath(image, paths, SVG_RIGHT_OPEN_EYE, facePts, ann)
    elif c[CLASS] == EYEBROW:
      if len(c[DATA]) != 1:
        raise Exception("Length of Eyebrow Data points: ", len(c[DATA]), " is not 1")
      eyebrowPts = to_points(c[DATA][0])
      if SVG_LEFT_EYEBROW not in paths:
        addPointsToPath(image, paths, SVG_LEFT_EYEBROW, eyebrowPts, ann)
      else:
        addPointsToPath(image, paths, SVG_RIGHT_EYEBROW, eyebrowPts, ann)
    elif c[CLASS] == NOSTRIL:
      if len(c[DATA]) != 1:
        raise Exception("Length of Nostril data points: ", len(c[DATA]), " is not 1")
      nostrilPts = to_points(c[DATA][0])
      if SVG_LEFT_NOSTRIL not in paths:
        addPointsToPath(image, paths, SVG_LEFT_NOSTRIL, nostrilPts, ann)
      else:
        addPointsToPath(image, paths, SVG_RIGHT_NOSTRIL, nostrilPts, ann)

  for c in ann:
    if c[CLASS] == EYEBALL:
      if len(c[DATA]) != 1:
        raise Exception("Length of Eyeball Data points: ", len(c[DATA]), " is not 1")
      eyeballPts = to_points(c[DATA][0])
      if SVG_LEFT_EYEBALL not in paths:
        addPointsToPath(image, paths, SVG_LEFT_EYEBALL, eyeballPts, ann)
      else:
        addPointsToPath(image, paths, SVG_RIGHT_EYEBALL, eyeballPts, ann)

  nosePts = process_nose(ann)
  addPointsToPath(image, paths, SVG_NOSE, nosePts, ann)

  view_image(image, [mergedPts, hairPts, nosePts] + remEarPts)
  with open("paths.json", "w") as outputfile:
    json.dump(paths, outputfile)

"""
addPointsToPath will add given points to given path dictionary. In addition,
it will also segregate points in the label and add corresponding
attributes to the path.
"""
def addPointsToPath(image, paths, k, points, ann, left=True):
  attr = {STROKE: "#000", STROKE_WIDTH: 2, FILL:"none", CLOSED_PATH: False}
  paths[k] = {}
  paths[k] = {SVG_DATA: [], SVG_ATTR: []}
  paths[k][SVG_DATA].append(points)

  # Close paths for certain labels.
  if k == SVG_FACE_EAR or k == SVG_LEFT_EYEBALL or k == SVG_RIGHT_EYEBALL or \
    k == SVG_LEFT_NOSTRIL or k == SVG_RIGHT_NOSTRIL or k == SVG_LEFT_OPEN_EYE or\
    k == SVG_RIGHT_OPEN_EYE:
    attr[CLOSED_PATH] = True

  if k == SVG_NOSE:
    paths[k][SVG_ATTR].append(attr)
    return
  if k == SVG_LEFT_NOSTRIL or k == SVG_RIGHT_NOSTRIL:
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
  mask = Polygons(bpts).mask(width=image.shape[1], height=image.shape[0])
  mask = [(p[1], p[0]) for p in np.argwhere(mask.array)]

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

  # No need to calculate shading for labels other than face or hair.
  if k != SVG_FACE_EAR and k != SVG_HAIR and k != SVG_LEFT_REM_EAR and k != SVG_RIGHT_REM_EAR:
    return

  # Find clusters for minpts.
  clusters = []
  cdict = MathUtils.make_clusters(minpts)
  for key in cdict:
    clusters.append(cdict[key])

  # Choose top M clusters for drawing. Sorting is based on size of cluster.
  M = 2
  clusters = sorted(clusters, key=len, reverse=True)[:M]

  # Get boundary points of the clusters.
  temp = []
  for p in clusters:
    bdpts = MathUtils.boundary_points(p)
    if len(bdpts) == 0:
      # Boundary points too small, no need to perform shading.
      print ("No shading for label: ", k)
      return
    temp.append(bdpts)
  clusters = temp

  # Subdivide boundary points into delta intervals. Value of delta is emperical.
  temp = []
  for pts in clusters:
    delta = 1 if len(pts) <= 50 else 10
    temp.append([(int(pts[i][0]), int(pts[i][1])) for i in range(len(pts)) if i % delta == 0])
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
  func = is_right if positive else is_left
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
  idx = find_index(p, points)
  ndx = idx
  for i in range(r):
    ndx = next_index(ndx, points, clockwise)
  return points[ndx]

"""
is_clockwise returns true if moving in the given
x direction is clockwise or counterclockwise movement
of the label.
"""
def is_counter_clockwise(p, points, positive=True):
  d = 1 if positive else -1
  r = 5
  idx = find_index(p, points)
  ndx = idx
  for i in range(r):
    ndx = next_index(ndx, points)

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
  i = find_index(a, points)
  j = find_index(b, points)

  ndx = i
  res = []
  while True:
    ndx = next_index(ndx, points, clockwise=True)
    if ndx == j:
      break
    res.append(points[ndx])

  return res

"""
next_index returns next index to given index along given direction.
"""
def next_index(i, points, clockwise=False):
  if clockwise:
    return len(points)-1 if i == 0 else i-1
  return 0 if i == len(points)-1 else i+1

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
    post_process(args)
