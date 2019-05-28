from svgpathtools import parse_path, wsvg, Path
from path_fitter import fitpath, pathtosvg, drawsvg
import json

# SVG JSON file constants.
SVG_DATA = "SVG Data"
SVG_COLOR = "SVG Color"
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

# SVG Attribute constants.
STROKE = "stroke"
STROKE_WIDTH = "stroke_width"
FILL = "fill"

PATH = "path"
ATTR = "attr"


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
merge_face_hair will merge face and hair
paths so that they intersect. The function
will modify the input map with the modified
path and attributes.
"""
def merge_face_hair(outputMap):
  facePath = outputMap[SVG_FACE_EAR][PATH]
  hairPath = outputMap[SVG_HAIR][PATH]

  iPts, tpts = find_intersection_points(hairPath, facePath)

  path = Path()
  forward_path(hairPath, (iPts[0][0], iPts[1][0]), path)
  reverse_path(facePath, (iPts[1][1], iPts[0][1]), path)

  outputMap[SVG_HAIR][PATH] = path

  return tpts

if __name__ == "__main__":
  d =  {}
  with open("paths.json", "r") as f:
    d = json.load(f)

  outputMap = {}
  for k, p in d.items():
    err = 50
    swidth = 5
    if k != SVG_FACE_EAR and k != SVG_HAIR and k != SVG_LEFT_REM_EAR and k != SVG_RIGHT_REM_EAR:
      err = 10
      swidth = 2
    pf = fitpath(p[SVG_DATA], err)
    sp = pathtosvg(pf)
    attr = {STROKE: "#000", STROKE_WIDTH: swidth, FILL:"none"}
    if k == SVG_FACE_EAR:
      sp += " Z"
      attr[FILL] = p[SVG_COLOR]
    elif k == SVG_HAIR:
      attr[FILL] = p[SVG_COLOR]
    elif k == SVG_LEFT_OPEN_EYE or k == SVG_RIGHT_OPEN_EYE:
      sp += " Z"
      attr[FILL] = "#ffffff"
    elif k == SVG_LEFT_EYEBROW or k == SVG_RIGHT_EYEBROW or \
      k == SVG_LEFT_EYEBALL or k == SVG_RIGHT_EYEBALL:
      sp += "Z"
      attr[FILL] = p[SVG_COLOR]
    elif k == SVG_LEFT_NOSTRIL or k == SVG_RIGHT_NOSTRIL:
      sp += " Z"
      attr[FILL] = "#000000"

    path = parse_path(sp)
    outputMap[k] = {PATH: path, ATTR: attr}

  intersections = merge_face_hair(outputMap)
  intersections = []

  pathList = []
  attrList = []
  for k in outputMap:
    #if k == SVG_RIGHT_EYEBROW:
    #  continue
    pathList.append(outputMap[k][PATH])
    attrList.append(outputMap[k][ATTR])

  wsvg(pathList, attributes=attrList, filename='test.svg', nodes=intersections, node_radii = [5]*len(intersections))
