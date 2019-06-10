from svgpathtools import parse_path, wsvg, Path, Line, Arc
from path_fitter import fitpath, pathtosvg, drawsvg
import json
import time

# SVG JSON file constants.
SVG_DATA = "SVG Data"
SVG_ATTR = "SVG Attributes"
SVG_FACE_EAR = "SVG Face Ear"
SVG_LEFT_REM_EAR = "SVG Left Rem Ear"
SVG_RIGHT_REM_EAR = "SVG Right Rem Ear"
SVG_HAIR = "SVG Hair"
SVG_LEFT_OPEN_EYE = "SVG Left Open Eye"
SVG_RIGHT_OPEN_EYE = "SVG Right Open Eye"
SVG_LEFT_EYEBROW = "SVG Left Eyebrow"
SVG_RIGHT_EYEBROW = "SVG Right Eyebrow"
SVG_LEFT_EYEBALL = "SVG Left Eyeball"
SVG_RIGHT_EYEBALL = "SVG Right Eyeball"
SVG_NOSE = "SVG Nose"
SVG_LEFT_NOSTRIL = "SVG Left Nostril"
SVG_RIGHT_NOSTRIL = "SVG Right Nostril"
SVG_LOWER_LIP = "SVG Lower Lip"
SVG_UPPER_LIP = "SVG Upper Lip"
SVG_FACIAL_HAIR = "SVG Facial Hair"
SVG_LEFT_PUPIL = "SVG Left Pupil"
SVG_RIGHT_PUPIL = "SVG Right Pupil"
SVG_LEFT_LOWER_EYELID = "SVG Left Lower Eyelid"
SVG_LEFT_UPPER_EYELID = "SVG Left Upper Eyelid"
SVG_RIGHT_LOWER_EYELID = "SVG Right Lower Eyelid"
SVG_RIGHT_UPPER_EYELID = "SVG Right Upper Eyelid"

# SVG Attribute constants.
STROKE = "stroke"
STROKE_WIDTH = "stroke_width"
FILL = "fill"
CLOSED_PATH = "closed path"

PATH = "path"
ATTR = "attr"
LABEL = "label"

"""
find_hair_intersection_points finds intersection points
of relevance between hair and face. It finds the right
points of intersection by finding maximum distance
between consecutive intersection points.
"""
def find_hair_intersection_points(hairPath, facePath):
  paramList = []
  dmax = -1
  i = 0
  res = []
  tpts = []
  for (T1, seg1, t1), (T2, seg2, t2) in hairPath.intersect(facePath):
    tpts.append(hairPath.point(T1))
    kh, th = hairPath.T2t(T1)
    kf, tf = facePath.T2t(T2)
    paramList.append(((kh, th), (kf, tf)))
    if i > 0:
      d = abs(hairPath[paramList[i][0][0]].point(paramList[i][0][1]) - \
      hairPath[paramList[i-1][0][0]].point(paramList[i-1][0][1]))
      if d > dmax:
        dmax = d
        res = [paramList[i-1], paramList[i]]
    i += 1

  return res, tpts

"""
find_eyeball_intersection_points finds the intersection
points between eye and eyeball. The intersection points
on eyeball are determined by finding the two segment that have intersection
points separated by maximum y value.
"""
def find_eyeball_intersection_points(eyeballPath, eyePath):
  paramList = []
  res = []
  tpts = []
  i = 0
  for (T1, seg1, t1), (T2, seg2, t2) in eyeballPath.intersect(eyePath):
    tpts.append(eyeballPath.point(T1))
    kh, th = eyeballPath.T2t(T1)
    kf, tf = eyePath.T2t(T2)
    if len(paramList) == 0 or abs(th - paramList[-1][0][1]) > 1e-3:
      paramList.append(((kh, th), (kf, tf)))

    i += 1

  if len(paramList) == 0:
    return

  if len(paramList) != 4:
    print("Number intersections != 4 is not supported yet: ", len(paramList))
    return None, tpts

  dmax = -1
  didx = -1
  for i in range(-1, len(paramList)-1):
    d = abs(eyeballPath[paramList[i][0][0]].point(paramList[i][0][1]).imag - eyeballPath[paramList[i+1][0][0]].point(paramList[i+1][0][1]).imag)
    if d > dmax:
      dmax = d
      didx = i

  # Merge all paths for eyeball.
  path = Path()
  rpath = Path()
  idx = didx
  ndx = 0 if idx == 3 else idx + 1
  for i in range(4):

    fullPath = []
    if i  % 2 == 0:
      # path from eyeball.
      ebp1 = paramList[idx][0]
      ebp2 = paramList[ndx][0]
      fullPath = eyeballPath
    else:
      # path from eye.
      ebp1 = paramList[idx][1]
      ebp2 = paramList[ndx][1]
      fullPath = eyePath

    sp = eyeballPath[paramList[idx][0][0]].point(paramList[idx][0][1])
    ep = eyeballPath[paramList[ndx][0][0]].point(paramList[ndx][0][1])
    rpath.append(Line(start=sp, end=ep))

    forward_path(fullPath, (ebp1, ebp2), path)

    idx = ndx
    ndx = 0 if idx == 3 else idx + 1

  return path, rpath, tpts

"""
draw_pupil draws circle for given pupil.
"""
def draw_pupil(opd, data, label):
  center, rad = data[0], data[1]
  top = Arc(start=complex(center[0]+rad, center[1]), radius=complex(rad, rad), rotation=0, large_arc=False, sweep=False, end=complex(center[0]-rad, center[1]))
  bottom = Arc(start=complex(center[0]-rad, center[1]), radius=complex(rad, rad), rotation=0, large_arc=False, sweep=False, end=complex(center[0]+rad, center[1]))

  opd[PATH].append(Path(top, bottom))
  opd[ATTR][0][FILL] = "#fff"


"""
forward_path append forward path from given intersection points
to given path object.
"""
def forward_path(fullPath, ipts, path):
  (k1, t1), (k2, t2) = ipts
  if k1 == k2:
    path.append(fullPath[k1].cropped(t1, t2))
    return
  _, p1 = fullPath[k1].split(t1)
  path.append(p1)

  if k2 >= k1+1:
    for k in range(k1+1, k2):
      path.append(fullPath[k])
  else:
    for k in range(k1+1, len(fullPath)):
      path.append(fullPath[k])
    for k in range(k2):
      path.append(fullPath[k])

  p2, _ = fullPath[k2].split(t2)
  path.append(p2)

"""
reverse_path append reverse path from given intersection points
to given path object.
"""
def reverse_path(fullPath, ipts, path):
  (k1, t1), (k2, t2) = ipts

  p1, _ = fullPath[k1].split(t1)

  path.append(p1.reversed())

  k = k1 -1 if k1 > 0 else len(fullPath)-k1-1
  while k != k2:
    path.append(fullPath[k].reversed())
    k = k -1 if k > 0 else len(fullPath)-k-1

  _, p2 = fullPath[k2].split(t2)
  path.append(p2.reversed())

"""
find_label returns dictionary corresponding to given label
in output list.
"""
def find_label(output, label):
  for o in output:
    if o[LABEL] == label:
      return o

  raise Exception("Label ", label, " not found in output!")

"""
merge_face_hair will merge face and hair
paths so that they intersect. The function
will modify the input map with the modified
path and attributes.
"""
def merge_face_hair(output):
  facePath = find_label(output, SVG_FACE_EAR)[PATH][0]
  hpd = find_label(output, SVG_HAIR)
  hairPath = hpd[PATH][0]

  iPts, tpts = find_hair_intersection_points(hairPath, facePath)

  path = Path()
  forward_path(hairPath, (iPts[0][0], iPts[1][0]), path)
  reverse_path(facePath, (iPts[1][1], iPts[0][1]), path)

  hpd[PATH][0] = path

  return tpts

def merge_eye_eyeball(output):
  lebd = find_label(output, SVG_LEFT_EYEBALL)
  lebPath = lebd[PATH][0]
  lePath = find_label(output, SVG_LEFT_OPEN_EYE)[PATH][0]

  path, rpath, tpts = find_eyeball_intersection_points(lebPath, lePath)
  if path:
    lebd[PATH][0] = path
    attr_copy = lebd[ATTR][0].copy()
    lebd[PATH].append(rpath)
    lebd[ATTR].append(attr_copy)

  rebd = find_label(output, SVG_RIGHT_EYEBALL)
  rebPath = rebd[PATH][0]
  rePath = find_label(output, SVG_RIGHT_OPEN_EYE)[PATH][0]

  path, rpath, tpts = find_eyeball_intersection_points(rebPath, rePath)
  if path:
    rebd[PATH][0] = path
    attr_copy = rebd[ATTR][0].copy()
    rebd[PATH].append(rpath)
    rebd[ATTR].append(attr_copy)

  return tpts

if __name__ == "__main__":
  start = time.time()

  d =  {}
  with open("paths.json", "r") as f:
    d = json.load(f)

  output = []
  for k, p in d.items():
    err = 10
    if k == SVG_FACE_EAR or k == SVG_HAIR or k == SVG_LEFT_REM_EAR or k == SVG_RIGHT_REM_EAR:
      err = 30
    elif k == SVG_LEFT_EYEBALL or k == SVG_RIGHT_EYEBALL or k == SVG_LEFT_OPEN_EYE \
    or k == SVG_RIGHT_OPEN_EYE or k == SVG_NOSE:
      err = 2
    elif k == SVG_LEFT_EYEBROW or k == SVG_RIGHT_EYEBROW:
      err = 5

    opd = {LABEL: k, PATH: [], ATTR: [{STROKE: attr[STROKE], STROKE_WIDTH: attr[STROKE_WIDTH], FILL: attr[FILL]} for attr in p[SVG_ATTR]]}
    dErr = err
    for i, data in enumerate(p[SVG_DATA]):
      if i > 0 and k == SVG_FACE_EAR:
        dErr = err*4
      else:
        dErr = err
      if k == SVG_LEFT_PUPIL or k == SVG_RIGHT_PUPIL:
        draw_pupil(opd, data, k)
      else:
        pf = fitpath(data, dErr)
        sp = pathtosvg(pf)
        if p[SVG_ATTR][i][CLOSED_PATH]:
          sp += " Z"
        path = parse_path(sp)
        opd[PATH].append(path)

    output.append(opd)

  intersections = merge_face_hair(output)
  #eipts = merge_eye_eyeball(output)
  eipts = []
  intersections = []


  pathList = []
  attrList = []
  for opd in output:
    #if opd[LABEL] != SVG_LEFT_EYEBALL:
    #  continue
    for j, path in enumerate(opd[PATH]):
      pathList.append(path)
      attrList.append(opd[ATTR][j])

  wsvg(pathList, attributes=attrList, filename='test.svg', nodes=eipts, node_radii = [1]*len(eipts))
  print ("Total time: ", time.time()-start)
