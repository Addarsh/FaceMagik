from svgpathtools import parse_path, wsvg, Path
from path_fitter import fitpath, pathtosvg, drawsvg
import json

# SVG JSON file constants.
SVG_FACE_EAR = "SVG Face Ear"
SVG_LEFT_REM_EAR = "SVG Left Rem Ear"
SVG_RIGHT_REM_EAR = "SVG Right Rem Ear"
SVG_HAIR = "SVG Hair"

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
    print ("TPTS: ", T1)
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
  outputMap[SVG_HAIR][ATTR][FILL] = "#000"

  return tpts

if __name__ == "__main__":
  d =  {}
  with open("paths.json", "r") as f:
    d = json.load(f)

  outputMap = {}
  for k, p in d.items():
    pf = fitpath(p)
    sp = pathtosvg(pf)
    attr = {STROKE: "#000", STROKE_WIDTH: 5, FILL:"none"}
    if k == SVG_FACE_EAR:
      sp += " Z"
      attr[FILL] = "#654321"
    path = parse_path(sp)
    outputMap[k] = {PATH: path, ATTR: attr}

  intersections = merge_face_hair(outputMap)
  #intersections = []

  pathList = []
  attrList = []
  for k in outputMap:
    #if k != SVG_HAIR:
    #  continue
    pathList.append(outputMap[k][PATH])
    attrList.append(outputMap[k][ATTR])

  wsvg(pathList, attributes=attrList, filename='test.svg', nodes=intersections, node_radii = [5]*len(intersections))
