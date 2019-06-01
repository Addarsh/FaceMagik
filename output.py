from svgpathtools import parse_path, wsvg, Path
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

# SVG Attribute constants.
STROKE = "stroke"
STROKE_WIDTH = "stroke_width"
FILL = "fill"
CLOSED_PATH = "closed path"

PATH = "path"
ATTR = "attr"
LABEL = "label"


def find_intersection_points(hairPath, facePath):
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
      d = abs(paramList[i][0][0]-paramList[i-1][0][0])
      if d > dmax:
        dmax = d
        res = [paramList[i-1], paramList[i]]
    i += 1

  return res, tpts

"""
forward_path append forward path from given intersection points
to given path object.
"""
def forward_path(fullPath, ipts, path):
  (k1, t1), (k2, t2) = ipts
  _, p1 = fullPath[k1].split(t1)

  path.append(p1)
  for k in range(k1+1, k2):
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

  iPts, tpts = find_intersection_points(hairPath, facePath)

  path = Path()
  forward_path(hairPath, (iPts[0][0], iPts[1][0]), path)
  reverse_path(facePath, (iPts[1][1], iPts[0][1]), path)

  hpd[PATH][0] = path

  return tpts

if __name__ == "__main__":
  start = time.time()

  d =  {}
  with open("paths.json", "r") as f:
    d = json.load(f)

  output = []
  for k, p in d.items():

    err = 50
    if k == SVG_UPPER_LIP or k == SVG_LOWER_LIP:
      err = 20
    elif k == SVG_LEFT_OPEN_EYE or k == SVG_RIGHT_OPEN_EYE or k == SVG_LEFT_EYEBALL or k == SVG_RIGHT_EYEBALL:
      err = 10
    elif k != SVG_FACE_EAR and k != SVG_HAIR and k != SVG_LEFT_REM_EAR and k != SVG_RIGHT_REM_EAR:
      err = 50

    opd = {LABEL: k, PATH: [], ATTR: [{STROKE: attr[STROKE], STROKE_WIDTH: attr[STROKE_WIDTH], FILL: attr[FILL]} for attr in p[SVG_ATTR]]}
    for i, data in enumerate(p[SVG_DATA]):
      pf = fitpath(data, err)
      sp = pathtosvg(pf)
      if p[SVG_ATTR][i][CLOSED_PATH]:
        sp += " Z"
      path = parse_path(sp)
      opd[PATH].append(path)

    output.append(opd)

  intersections = merge_face_hair(output)
  intersections = []

  pathList = []
  attrList = []
  for opd in output:
    for j, path in enumerate(opd[PATH]):
      pathList.append(path)
      attrList.append(opd[ATTR][j])

  wsvg(pathList, attributes=attrList, filename='test.svg', nodes=intersections, node_radii = [5]*len(intersections))
  print ("Total time: ", time.time()-start)
