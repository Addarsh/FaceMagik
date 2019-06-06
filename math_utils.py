import math
import sys
import operator
import numpy as np
import scipy
import queue
import skimage
import cv2

from numpy.linalg import inv, eig
from lmfit import minimize, Parameters
from image_utils import ImageUtils
from scipy.interpolate import make_lsq_spline, interp1d

class MathUtils:
  """
  MathUtils provides functionality for curve fitting.
  """
  """
  n_polynomial fits a polynomial of degree n through given points.
  This is not a best fit polynomial. Number of points-1 = Degree of polynomial.
  TODO: Add a best fit polynomial curve function to MathUtils.
  """
  @staticmethod
  def n_polynomial(points):
    if len(points) <= 1:
      return points
    n = len(points)
    # Note: Degree of polynomial constructed = n-1.
    # base shift points.
    base_p = points[0]
    points = MathUtils.base_shift(points, base_p)

    A = np.ones((n,n))
    B = np.zeros((1,n))
    for j in range(n):
      x = points[j][0]
      for i in range(1, n):
        A[i][j] = x
        x *= points[j][0]
      B[0][j] = points[j][1]
    Ap = inv(A)
    c = np.matmul(B,Ap)

    base_x, base_y = base_p
    npoints = []
    dictp = {}
    for _, p in enumerate(points):
      dictp[p[0]] = p[1]
    for i in range(0, points[n-1][0]+1):
      if i in dictp:
        npoints.append((base_x+i, base_y+dictp[i]))
        continue
      y = c[0][0]
      x = i
      for j in range(1, n):
        y += c[0][j]*x
        x *= i
      y = int(round(y))
      npoints.append((base_x+i,base_y+y))
    return npoints

  """
  base_shift shifts given points with given base point as origin.
  """
  @staticmethod
  def base_shift(points, base_p):
    res = []
    base_x, base_y = base_p
    for _, p in enumerate(points):
      px, py = p
      res.append((px-base_x, py-base_y))
    return res

  """
  best_fit_circle tries to fit best circle using non-linear least squares among given points.
  """
  @staticmethod
  def best_fit_circle(points):
    # X coordinate and Y coordinate arrays.
    # These are created to use as datapoints in least sqaures optimization.
    x = [p[0] for  p in points]
    y = [p[1] for  p in points]

    # Initial guess for circle center.
    xc0 = sum(x)/len(x)
    yc0 = sum(y)/len(y)

    # residual for minimization function.
    def residual(params, x, y):
      xc = params['xc']
      yc = params['yc']
      n = len(x)

      # Check "Finding the circle that best fits a set of points - L. MAISONOBE"
      # for more details about the equation.
      rhat = sum([math.sqrt((i-xc)**2 + (j-yc)**2) for (i,j) in zip(x,y)])/n
      model = [math.sqrt((i-xc)**2 + (j-yc)**2)- rhat for (i,j) in zip(x,y)]
      return model

    # The circle center points are the paramters to be optimized.
    params = Parameters()
    params.add('xc', value=xc0)
    params.add('yc', value=yc0)

    out = minimize(residual, params, args=(x, y))
    rxc, ryc = int(round(out.params['xc'].value)), int(round(out.params['yc'].value))
    rad = sum([math.sqrt((i-rxc)**2 + (j-ryc)**2) for (i,j) in zip(x,y)])/len(x)

    return (rxc, ryc), rad

  """
  best_bezier_curve returns the bezier curve that is most likely at the
  boundary of shape. This is done by varying the control node in the given
  direction. Note that this is currently only implemented for quadratic bezier
  curves and variation is allowed only in the y direction.
  """
  @staticmethod
  def best_bezier_curve(pset, nodes, cnode, delta):
    if delta[0] != 0:
      raise Exception("Not supporting control node variation in x direction")
    # move control node in y direction.
    curve = MathUtils.bezier_curve([nodes[0], (cnode[0], cnode[1]), nodes[-1]])
    counts = MathUtils.num_pts_on_curve(curve, (nodes[0][0], nodes[-1][0]), pset)
    d = 1
    max_diff = 0
    best_curve = None
    while counts > 2:
      curve = MathUtils.bezier_curve([nodes[0], (cnode[0], cnode[1]+delta[1]*d), nodes[-1]])
      t_counts = MathUtils.num_pts_on_curve(curve, (nodes[0][0], nodes[-1][0]), pset)
      d +=1
      if abs(t_counts - counts) > max_diff:
        max_diff = abs(t_counts - counts)
        best_curve = curve
      counts = t_counts

    if best_curve == None:
      raise Exception("Could not find best bezier curve!")
    return best_curve

  """
  bspline will return a bsplien curve representing the given set of points
  between the given x coordinate limits. Input points is sorted.
  """
  @staticmethod
  def bspline(points):
    rpoints, vset = [], set()
    vlines = []
    for i in range(len(points)):
      if i == 0 or points[i][0] != points[i-1][0]:
        vlines += MathUtils.vline(list(vset))
        vset = set()
        rpoints.append(points[i])
        continue
      vset.add(points[i-1])
      vset.add(points[i])
      rpoints.pop()
      rpoints.append(points[i])
    if len(rpoints) == 1:
      vset.add(points[-1])
      vlines += MathUtils.vline(list(vset))
      rpoints.pop()
    return MathUtils.interp(rpoints) + vlines

  @staticmethod
  def interp(points):
    if len(points) == 0:
      return []
    if len(points) == 2:
      kind = "linear"
    elif len(points) == 3:
      kind = "quadratic"
    else:
      kind = "cubic"
    x = [p[0] for p in points]
    y = [p[1] for p in points]

    return scipy.interpolate.interp1d(x, y, kind=kind)

  @staticmethod
  def vline(points):
    if len(points) == 0:
      return []
    ypts = [p[1] for p in points]
    ymin, ymax = min(ypts), max(ypts)
    x = points[0][0]
    return [(x,j) for j in range(ymin, ymax+1)]

  """
  num_pts_on_curve returns the number of points of given curve (between given
  x coordinate limits) that occur in the given point set. Note that all points
  in the set are assumed to belong to the same label.
  """
  @staticmethod
  def num_pts_on_curve(curve, xlims, pset):
    xarr = np.linspace(0, 1.0, xlims[1]-xlims[0]+1, dtype=np.float64)
    pts = curve.evaluate_multi(np.asfortranarray(xarr))
    count = 0
    for j in range(len(pts[0,:])):
      x = int(xlims[0] + (xlims[1]-xlims[0]+1)*pts[0, j])
      y = int(pts[1,j])
      if (x,y) in pset:
        count += 1
    return count

  """
  eval_bezier evaluates given bezier for points in given xlims range and
  returns the points.
  """
  @staticmethod
  def eval_bezier(curve, xlims):
    xarr = np.linspace(0, 1.0, xlims[1] - xlims[0] + 1, dtype=np.float64)

    pts = curve.evaluate_multi(np.asfortranarray(xarr))
    res = []
    for j in range(len(pts[0,:])):
      x = int(xlims[0] + (xlims[1]-xlims[0]+1)*pts[0, j])
      y = int(pts[1, j])
      res.append((x,y))
    return res

  """
  Fit a polynomial of deg n through given x cordinate points.
  """
  @staticmethod
  def poly_fit(points, xlims, ybound, deg, w=None):
    x = [a[0] for a in points]
    y = [a[1] for a in points]
    p = np.polyfit(x, y, deg, w=w)

    res = []
    for i in range(xlims[0], xlims[1]+1):
      iv = [i**d for d in range(deg, -1, -1)]
      y = int(round(sum(np.multiply(p, iv))))
      if y >= ybound or y < 0:
        return res
      res.append((i,y))
    return res

  """
  max_eigen_vector returns eigen vector corresponding to maximum eigen value
  of the covariance matrix of given image matrix. The image matrix is of
  shape (3, n) where 3 is pixel dimension and n is number of observations.
  """
  @staticmethod
  def max_eigen_vector(m):
    w, v = eig(np.cov(m))

    max_eig_v = None
    max_val = 0
    for i in range(len(w)):
      if abs(w[i]) >= max_val:
        max_val = abs(w[i])
        max_eig_v = v[i]
    return np.reshape(max_eig_v, (m.shape[0],1))

  """
  segregates points divides up the given set of points into groups based on color
  """
  @staticmethod
  def segregate_points(img, points):
    m = np.zeros((len(points), img.shape[2]))
    for i in range(len(points)):
      m[i, :] = img[points[i][1], points[i][0]]

    q = np.reshape(np.mean(m, axis=0), (m.shape[1],1))
    v = MathUtils.max_eigen_vector(m.T)

    thresh = np.dot(q.T, v)
    def f(x):
      return np.dot(x, v) < thresh
    boolean_m = f(m)

    points_a, points_b = [], []
    avg_a, avg_b = 0, 0
    for i in range(len(points)):
      if boolean_m[i]:
        points_a.append(points[i])
      else:
        points_b.append(points[i])
    return points_a, points_b

  """
  find best points among given set of points.
  """
  @staticmethod
  def best_points(lab, points, n=1):
    if n < 1:
      return {}
    p_a, p_b = MathUtils.darkest_color(lab, points)
    if n == 1:
      return p_a
    pd_a, pl_a = MathUtils.darkest_color(lab, p_a)

    # fileter pl_a
    pld_a, _ = MathUtils.darkest_color(lab, pl_a+p_b)
    pldd_a, _ = MathUtils.darkest_color(lab, pld_a)
    return pd_a + pldd_a

  @staticmethod
  def iris_filter(gray, points, n):
    k = int(round(n))
    b = int(round(0.6 *len(points)))
    vals = []
    for _, p in enumerate(points):
      r = np.std(gray[p[1]-k:p[1]+k+1, p[0]-k: p[0]+k+1])
      vals.append((r,p[0],p[1]))

    rvals = sorted(vals, key=lambda x: -x[0])
    rpts = []
    for i in range(b):
      rpts.append((rvals[i][1], rvals[i][2]))
    return rpts

  def best_edge_val(gray, points, c, r, n):
    k = int(round(n))
    val = 0
    r = int(round(r))
    mval = np.mean(gray[c[1]-r:c[1]+r, c[0]-r:c[0]+r])
    for _, p in enumerate(points):
      val += np.std(gray[p[1]-k:p[1]+k+1, p[0]-k: p[0]+k+1])
    return val - mval**2


  def circle_val(gray, p, r, lmap):
    val = 0
    l = 1
    x = 0.05
    while x <= r:
      pts = MathUtils.circle_points(p, x)
      count = 0
      for _, p in enumerate(pts):
        try:
          val += gray[p[1],p[0]]
          count += 1
        except Exception as e:
          return sys.maxsize
      l += count
      x += 0.05
    return val/l

  """
  circle points returns points that lie on the circumference.
  The input is the center of the circle and the radius.
  """
  @staticmethod
  def circle_points(c, r, n=200):
    xc, yc = c
    points = []
    for i in range(n):
      theta = ((2*math.pi)/n)*i

      x = int(round(xc + r*math.cos(theta)))
      y = int(round(yc - r*math.sin(theta)))
      points.append((x,y))
    return points

  """
  compute_avg computes average LAB values of given points. img must
  be a LAB image.
  """
  @staticmethod
  def compute_avg(img, points):
    s = (0,0,0)
    n = len(points)
    for _, p in enumerate(points):
      s =  tuple(map(operator.add, s, img[p[1],p[0]]))
    return tuple(x/n for x in s)


  """
  Find median of given set of sorted points.
  """
  @staticmethod
  def median_point(points):
    even = len(points) %2 == 0
    mid = int(len(points)/2)
    return points[mid], mid

  """
  get_line returns (m,c) tuple for
  line between p1 and p2. For m = inf,
  returns empty tuple. Line eqx: y= mx + c
  """
  @staticmethod
  def get_line(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2:
      return ()
    m = (y2-y1)/(x2-x1)
    return (m, y1-m*x1)

  """
  slope takes input angle in degrees and
  returns the slope.
  """
  @staticmethod
  def slope(degrees):
    return math.tan(math.radians(degrees))

  """
  angle returns angle in degrees of slope
  between two points.
  """
  def angle(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.degrees(math.atan((y2-y1)/(x2-x1)))

  """
  line_distance returns distance of point p
  from line.
  """
  @staticmethod
  def line_distance(line_params, p):
    (m,c) = line_params
    x, y = p
    return abs((y-m*x-c)/(math.sqrt(1 + m**2)))

  """
  distance returns distance between two points.
  """
  @staticmethod
  def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

  """
  curve_map returns mapping of x to y for a curve that is a mathematical function.
  """
  @staticmethod
  def curve_map(pts):
    d = {}
    for p in pts:
      d[p[0]] = p[1]
    return d

  """
  toMap returns a dictionary from x coordinate to list of y coordinates
  for given set of points. The y cordinates for each x are sorted in y.
  """
  @staticmethod
  def toMap(points):
    pmap = {}
    for _, p in enumerate(points):
      x, y = p
      if x not in pmap:
        pmap[x] = []
      pmap[x].append(y)

    for x in pmap:
      pmap[x] = sorted(pmap[x])
    return pmap

  """
  xlims returns the x coordinate limits tuple for the given set of points.
  """
  @staticmethod
  def xlims(points):
    if len(points) == 0:
      return ()
    min_x, max_x = sys.maxsize, 0
    for _, p in enumerate(points):
      min_x = min(p[0], min_x)
      max_x = max(p[0], max_x)

    return (min_x, max_x)

  @staticmethod
  def ylims(points):
    if len(points) == 0:
      return ()
    min_y, max_y = sys.maxsize, 0
    for _, p in enumerate(points):
      min_y = min(p[1], min_y)
      max_y = max(p[1], max_y)

    return (min_y, max_y)

  @staticmethod
  def left_most_point(points):
    if len(points) == 0:
      raise Exception("left_most_point: input points is empty!")

    minp = points[0]
    for p in points:
      if p[0] < minp[0] or (p[0] == minp[0] and p[1] > minp[1]):
        minp = p

    return minp

  @staticmethod
  def right_most_point(points):
    if len(points) == 0:
      raise Exception("right_most_point: input points is empty!")

    minp = points[0]
    for p in points:
      if p[0] > minp[0] or (p[0] == minp[0] and p[1] > minp[1]):
        minp = p

    return minp

  @staticmethod
  def top_most_point(points):
    if len(points) == 0:
      raise Exception("top_most_point: input points is empty!")

    minp = points[0]
    for p in points:
      if p[1] < minp[1]:
        minp = p

    return minp

  @staticmethod
  def bottom_most_point(points):
    if len(points) == 0:
      raise Exception("bottom_most_point: input points is empty!")

    minp = points[0]
    for p in points:
      if p[1] > minp[1]:
        minp = p

    return minp

  """
  left_bottom_point returns the point that is left most and bottom most
  point in given set of points. We assume that the given set of points
  are in counter-clockwise direction.
  """
  @staticmethod
  def left_bottom_point(points):
    lpt = MathUtils.left_most_point(points)

    # Continue to move in counter clockwise direction till y coordinate
    # increases.
    pdx = MathUtils.find_index(lpt, points)
    ndx = MathUtils.next_index(pdx, points, clockwise=False)
    while MathUtils.is_lower(points[ndx], points[pdx]):
      pdx = ndx
      ndx = MathUtils.next_index(pdx, points, clockwise=False)

    return points[pdx]

  """
  right_bottom_point returns the point that is right most and bottom most
  point in given set of points. We assume that the given set of points
  are in counter-clockwise direction.
  """
  @staticmethod
  def right_bottom_point(points):
    lpt = MathUtils.right_most_point(points)

    # Continue to move in clockwise direction till y coordinate
    # increases.
    pdx = MathUtils.find_index(lpt, points)
    ndx = MathUtils.next_index(pdx, points, clockwise=True)
    while MathUtils.is_lower(points[ndx], points[pdx]):
      pdx = ndx
      ndx = MathUtils.next_index(pdx, points, clockwise=True)

    return points[pdx]

  """
  find_index returns the index of given point among given set of points.
  We assume that the given set of points are in counter-clockwise direction.
  """
  @staticmethod
  def find_index(p, points):
    for i in range(len(points)):
      if p == points[i]:
        return i
    raise Exception("Point: ", p, " not found in point set: ", points)

  """
  next_index returns next index to given index along given direction.
  Assumption is that input points are arranged in counterclockwise manner.
  """
  @staticmethod
  def next_index(i, points, clockwise=False):
    if clockwise:
      return len(points)-1 if i == 0 else i-1
    return 0 if i == len(points)-1 else i+1

  """
  Returns true if p1 is higher than p2 (y coord); else returns false.
  """
  @staticmethod
  def is_higher(p1, p2):
    return p1[1] < p2[1]

  """
  Returns true if p1 is lower than p2 (y coord); else returns false.
  """
  @staticmethod
  def is_lower(p1, p2):
    return p1[1] > p2[1]

  """
  Returns true if p1 is to the right of p2 (x coord); else returns false.
  """
  @staticmethod
  def is_right(p1, p2):
    return p1[0] > p2[0]

  """
  Returns true if p1 is to the left of p2 (x coord); else returns false.
  """
  @staticmethod
  def is_left(p1, p2):
    return p1[0] < p2[0]


  """
  cluster recursively clusters all points in pset that are connected to p.
  Returns the cluster of points.
  """
  def cluster(img, sp, pset):
    visited = set()
    q = queue.Queue()
    q.put(sp)
    visited.add(sp)
    while not q.empty():
      p = q.get()
      x, y = p
      for i in range(x-1, x+2):
        for j in range(y-1, y+2):
          if i < 0 or i >= img.shape[1] or j < 0 or j >= img.shape[0]:
            continue
          if (i,j) not in pset:
            continue
          if (i,j) in visited:
            continue
          visited.add((i,j))
          q.put((i,j))
    return visited

  @staticmethod
  def edge_points(points):
    pset = set(points)
    epts = []
    for _, p in enumerate(points):
      x, y = p
      if (x-1, y) in pset and (x+1, y) in pset \
      and (x,y-1) in pset and (x,y+1) in pset:
        continue
      epts.append((x,y))

    inset = pset - set(epts)
    res = []
    for _, p in enumerate(epts):
      x, y = p
      if (x-1, y) in inset or (x+1, y) in inset \
      or (x,y-1) in inset or (x,y+1) in inset:
        res.append((x,y))
        continue

    return MathUtils.complete_edge(res)

  """
  curve_boundareis returns all boundaries associated with given points.
  """
  @staticmethod
  def curve_boundaries(points):
    if len(points) == 0:
      return points

    clusters = MathUtils.make_clusters(points)
    boundaries = []
    for k in clusters:
      points = clusters[k]
      if len(points) < 50:
        continue
      xlims = MathUtils.xlims(points)
      pmap = MathUtils.toMap(points)

      boundary = set()
      # starting point must be on the edge.
      x0 = xlims[0]
      sp = (x0, pmap[x0][0])
      pp = (x0-1, sp[1])
      boundary.add(sp)

      prev_pt = pp
      curr_pt = (pp[0], pp[1]-1)
      bp = sp
      while True:
        prev_pt, curr_pt = MathUtils.find_next_point(prev_pt, curr_pt, bp, set(points))
        bp = curr_pt
        d = MathUtils.clock_dir((prev_pt[0]-curr_pt[0], prev_pt[1]-curr_pt[1]))
        curr_pt = (prev_pt[0] +d[0], prev_pt[1] + d[1])
        if bp == sp:
          break
        boundary.add(bp)

      boundaries.append(list(boundary))

    return boundaries


  @staticmethod
  def boundary_points(points):
    if len(points) < 10:
      return []
    xlims = MathUtils.xlims(points)
    pmap = MathUtils.toMap(points)

    boundary = []
    # starting point must be on the edge.
    x0 = xlims[0]
    sp = (x0, pmap[x0][0])
    pp = (x0-1, sp[1])
    boundary.append(sp)

    prev_pt = pp
    curr_pt = (pp[0], pp[1]-1)
    bp = sp
    while True:
      prev_pt, curr_pt = MathUtils.find_next_point(prev_pt, curr_pt, bp, set(points))
      bp = curr_pt
      d = MathUtils.clock_dir((prev_pt[0]-curr_pt[0], prev_pt[1]-curr_pt[1]))
      curr_pt = (prev_pt[0] +d[0], prev_pt[1] + d[1])
      if bp == sp:
        break
      boundary.append(bp)

    return boundary

  """
  Returns boundary points of given points using
  given gray scale image. Input is a list of connected
  points that belong to the same label.
  """
  @staticmethod
  def cv2_boundary_points(gray, clusters):
    np.place(gray, gray>=0, [0])

    for c in clusters:
      for p in c:
        gray[p[1], p[0]] = 255

    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    res = []
    for c in contours:
      pts = []
      for p in c:
        pts.append((p[0][0], p[0][1]))
      res.append(pts)

    return res

  @staticmethod
  def make_clusters(points):
    pset = set(points)
    class Node:
      def __init__ (self, label):
        self.label = label

      def __str__(self):
        return str(self.label)

    xlims = MathUtils.xlims(points)
    ylims = MathUtils.ylims(points)

    m = []
    count = 1
    for x in range(xlims[0], xlims[1]+1):
      n = []
      for y in range(ylims[0], ylims[1]+1):
        if (x,y) not in pset:
          n.append(Node(-1))
          continue
        l = Node(count)
        MathUtils.MakeSet(l)
        n.append(l)
        count += 1
      m.append(n)

    for x in range(xlims[0], xlims[1]+1):
      for y in range(ylims[0], ylims[1]+1):
        i, j = x - xlims[0] , y - ylims[0]
        if m[i][j].label == -1:
          continue
        if i == 0 and j == 0:
          continue

        if i == 0:
          if m[i][j-1].label == -1:
            continue
          MathUtils.Union(m[i][j], m[i][j-1])
        elif j == 0:
          if m[i-1][j].label == -1:
            continue
          MathUtils.Union(m[i][j], m[i-1][j])
        elif m[i-1][j].label == -1 and m[i][j-1].label == -1:
          continue
        elif m[i-1][j].label == -1:
          MathUtils.Union(m[i][j], m[i][j-1])
        elif m[i][j-1].label == -1:
          MathUtils.Union(m[i][j], m[i-1][j])
        else:
          MathUtils.Union(m[i][j], m[i-1][j])
          MathUtils.Union(m[i][j], m[i][j-1])

    groups = {}
    for x in range(xlims[0], xlims[1]+1):
      for y in range(ylims[0], ylims[1]+1):
        i, j = x - xlims[0] , y - ylims[0]
        if m[i][j].label == -1:
          continue
        parent = MathUtils.Find(m[i][j])
        if parent.label not in groups:
          groups[parent.label] = []
        groups[parent.label].append((xlims[0] +i, ylims[0]+j))

    return groups


  @staticmethod
  def MakeSet(x):
     x.parent = x
     x.rank   = 0

  @staticmethod
  def Union(x, y):
       xRoot = MathUtils.Find(x)
       yRoot = MathUtils.Find(y)
       if xRoot.rank > yRoot.rank:
           yRoot.parent = xRoot
       elif xRoot.rank < yRoot.rank:
           xRoot.parent = yRoot
       elif xRoot != yRoot: # Unless x and y are already in same set, merge them
           yRoot.parent = xRoot
           xRoot.rank = xRoot.rank + 1

  @staticmethod
  def Find(x):
    if x.parent == x:
      return x
    else:
      x.parent = MathUtils.Find(x.parent)
      return x.parent

  """
  distance_between_polygons takes given input points
  of two polygons and returns the distance between them.
  """
  @staticmethod
  def distance_between_polygons(poly1, poly2):
    return MathUtils.distance(MathUtils.centroid(poly1), MathUtils.centroid(poly2))

  """
  centroid returns centroid of given points.
  """
  @staticmethod
  def centroid(points):
    s = [0, 0]
    for p in points:
      s[0] += p[0]
      s[1] += p[1]
    n = len(points)
    return (int(s[0]/n), int(s[1]/n))

  """
  k_point returns point between p1 and p2 that divides
  p1 and p2 in ratio k i.e dist(p1->k)/dist(p1->p2) = k.
  """
  def k_point(p1, p2, k):
    return (int((1-k)*p1[0] + k*p2[0]), int((1-k)*p1[1] + k*p2[1]))

  """
  find_next_point returns the next point (moving clockwise) that is part
  of the given point set.
  """
  @staticmethod
  def find_next_point(pp, cp, bp, pset):
    while cp not in pset:
      d = MathUtils.next_d(pp, cp, bp)
      pp = cp
      cp = (cp[0]+ d[0], cp[1] + d[1])
    return pp, cp

  @staticmethod
  def next_d(prev_pt, curr_pt, bp):
    d = (curr_pt[0] - prev_pt[0], curr_pt[1] - prev_pt[1])
    if d == (0,0):
      raise Exception("d=0 for point: ", curr_pt)
    if curr_pt[0] == bp[0] or curr_pt[1] == bp[1]:
      return d
    return MathUtils.clock_dir(d)


  @staticmethod
  def clock_dir(d):
    if d == (0, -1):
      return (1, 0)
    if d == (1, 0):
      return (0, 1)
    if d == (0, 1):
      return (-1, 0)
    if d == (-1, 0):
      return (0, -1)
    raise Exception("d value incorrect: ", d)

  """
  shortest_path will returns points that get us from
  given start point to given end point.
  """
  @staticmethod
  def shortest_path(sp, target):
    pts = []
    x1, y1 = sp
    x2, y2 = target
    if (x2-x1) == 0:
      d = 1 if y2 > y1 else -1
      for j in range(1, abs(y2-y1)):
        pts.append((x1, y1+j*d))
      return pts
    if (y2-y1) == 0:
      d = 1 if x2 > x1 else -1
      for i in range(1, abs(x2-x1)):
        pts.append((x1+i*d, y1))
      return pts

    dx = 1 if (x2-x1) > 0 else -1
    dy = 1 if (y2-y1) > 0 else -1
    return [(x1+dx,y1+dy)] + MathUtils.shortest_path((x1+dx,y1+dy), target)



  """
  neighbor_count returns the number of neighbors
  for given point in given set.
  """
  @staticmethod
  def neighbor_count(p, pset):
    count = 0
    x, y = p
    for i in range(x-1, x+2):
      for j in range(y-1, y+2):
        if i == x and j == y:
          continue
        if (i,j) in pset:
          count += 1

    return count

  """
  Some edge points may not be pixel complete. This function will add
  all necessary points to make them complete.
  """
  @staticmethod
  def complete_edge(epts):
    # Find points that are only connected to one other point
    # and pair them to form the next edge.
    openpts = []
    for e in epts:
      if MathUtils.neighbor_count(e, set(epts)) == 1:
        openpts.append(e)

    if len(openpts) == 0:
      return epts

    if len(openpts) % 2 != 0:
      raise Exception("Edge cuve has: ",  len(openpts), " points open which is odd number!")

    done = set()
    groups = []
    for i, pt in enumerate(openpts):
      if pt in done:
        continue
      min_dist = 100000
      pos = -1
      for j in range(i+1, len(openpts)):
        if openpts[j] in done:
          continue
        d = abs(openpts[j][0]-pt[0]) + abs(openpts[j][1]-pt[1])
        if d < min_dist:
          min_dist = d
          pos = j

      if pos == -1:
        raise Exception("Could not find open point pair for point: ", pt)
      groups.append((pt, openpts[pos]))
      done.add(pt)
      done.add(openpts[pos])

    newpts = []
    for g in groups:
      newpts += MathUtils.shortest_path(g[0], g[1])
    return newpts + epts

  """
  num_neighbors returns the number of neighbors for the given point.
  Diagonal neighbors are not counted.
  """
  @staticmethod
  def num_neighbors(img, sp, pset):
    x, y = sp
    nbpts = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
    count = 0
    for _, p in enumerate(nbpts):
      x, y = p
      if x < 0 or x >= img.shape[1] or y < 0 or y >= img.shape[0]:
        continue
      if (x,y) in pset:
        count += 1
    return count

  """
  intersect returns all points that lie within points denoting a closed boundary.
  """
  @staticmethod
  def intersect(pts, bdpts):
    bdmap = MathUtils.toMap(bdpts)
    res = []
    for _, p in enumerate(pts):
      if p[0] not in bdmap:
        continue
      if p[1] < min(bdmap[p[0]]) or p[1] > max(bdmap[p[0]]):
        continue
      res.append(p)
    return res

  """
  points_inside returns points inside points representing the edge of a closed polygon.
  If the input points do not form a closed edge then the returned points are unreliable.
  """
  @staticmethod
  def pts_inside(epts):
    pmap = MathUtils.toMap(epts)
    pset = set(epts)
    res = []
    for x in pmap:
      yset = set(pmap[x])
      tlist = sorted(list(yset))
      ylist = []
      i = 0
      while i  < len(tlist):
        k = i+1
        while k < len(tlist) and (tlist[k] == tlist[k-1]+1):
          k += 1
        if MathUtils.both_side_neighbors(x, tlist[i:k] , pset):
          ylist.append(tlist[i])
        i = k
      lpts = []
      for i in range(1, len(ylist)):
        if i % 2 != 0:
          lpts += [(x,j) for j in range(ylist[i-1]+1, ylist[i])]
      res += lpts
    return res

  @staticmethod
  def both_side_neighbors(x, ylist, pset):
    if len(ylist) == 0:
      return True

    lfound = False
    if (x-1,ylist[0]-1) in pset or (x-1, ylist[-1]+1) in pset \
    or (x-1, ylist[0]) in pset or (x-1, ylist[-1]) in pset:
      lfound = True
    if not lfound:
      return False

    rfound = False
    if (x+1,ylist[0]-1) in pset or (x+1, ylist[-1]+1) in pset \
    or (x+1, ylist[0]) in pset or (x+1, ylist[-1]) in pset:
      return True
    return False

  """
  highest_point returns the point from the given set of points with the
  minimum y coordinate.
  """
  @staticmethod
  def highest_point(pts):
    miny = sys.maxsize
    res = ()
    for _, p in enumerate(pts):
      if p[1] < miny:
        res = p
        miny = p[1]
    return res

  """
  smooth_curves returns smooth version of upper and lower curves of given polygon.
  """
  @staticmethod
  def smooth_curves(img, points):
     epoints = MathUtils.edge_points(points)
     allcurves = MathUtils.curves(img, epoints)

     res = []
     for i, curve in enumerate(allcurves):
       fp, lp = (curve[0], curve[-1]) if curve[0][0] <= curve[-1][0] else (curve[-1], curve[0])
       pts = [fp]
       curve = sorted(curve, key=lambda x: x[0])
       d = MathUtils.num_divisions(curve)
       pts += [curve[i] for i in range(int(len(curve)/d), len(curve), int(len(curve)/d))]
       pts.append(lp)
       res.append(MathUtils.bspline(pts))

     allpoints = []
     for _, pts in enumerate(res):
       allpoints += pts
     return allpoints

  """
  curves returns the upper and lower curves for given edgepoints.
  The points returned are sorted in x.
  """
  @staticmethod
  def curves(img, epoints):
    eset = set(epoints)
    d = 1
    visited = set()
    allcurves = []
    cp = epoints[0]
    done = False
    while not done:
      curr_curve = []
      while True:
        visited.add(cp)
        curr_curve.append(cp)
        np = MathUtils.next_point(img, cp, visited, eset)
        if len(np) == 0:
          allcurves.append(curr_curve)
          done = True
          break
        if (np[0]-cp[0])*d >= 0:
          cp = np
          continue
        d *= -1
        allcurves.append(curr_curve)
        cp = np
        break
    return allcurves

  @staticmethod
  def next_point(img, p, visited, eset):
    x, y = p
    np = ()
    for i in range(x-1, x+2):
      for j in range(y-1, y+2):
        if i < 0 or i >= img.shape[1] or j < 0 or j >= img.shape[0]:
          continue
        if (i,j) not in eset:
          continue
        if (i,j) in visited:
          continue
        np = (i,j)
        break
    return np

  @staticmethod
  def num_divisions(pts):
    d = 200
    return min(len(pts), d)

  """
  fit_general_polygon will fit a general polygon with smooth Bsplines for given boundary points.
  A general polygon may need more than 2 B splines to completely be fit.
  These types of polygons are likely to occur on hair, face and even facial hair.
  """
  @staticmethod
  def fit_general_polygon(bpts, mask_set, k=3):
    if len(bpts) == 0:
      return []

    all_curves = MathUtils.boundary_functions(bpts)
    smooth_curves = []
    for i, c in enumerate(all_curves):
      smooth_curves.append(MathUtils.fit_smooth_spline(c, k))

    #This is for debugging purposes.
    """
    pts = []
    for c in smooth_curves:
      pts += c
    return pts
    """

    ptsMap = []
    for c in smooth_curves:
      cmap = MathUtils.toMap(c)
      ptsMap.append(cmap)

    return MathUtils.fill_curve_points(ptsMap, MathUtils.xlims(bpts), mask_set)

  """
  fill_curve_points fills points inside given curves.
  """
  @staticmethod
  def fill_curve_points(ptsMap, xlims, mask_set):
    pts = []
    for x in range(xlims[0], xlims[1]+1):
      ylist = MathUtils.sorted_ylist(ptsMap, x)
      if len(ylist) <= 1:
        continue
      pts += MathUtils.use_mask_data(x, ylist, mask_set)

    return pts

  """
  use_mask_data uses previous mask data to fill out points
  between y coordinate vertices for given x coordinate.
  """
  @staticmethod
  def use_mask_data(x, ylist, mask_set):
    pts = []
    for j in range(len(ylist)-1):
      ymin, ymax = ylist[j], ylist[j+1]
      lpts = []
      for y in range(ymin, ymax+1):
        if (x, y) in mask_set:
          lpts.append((x,y))
      if (len(lpts)/(ymax+1-ymin)) >= 0.5:
        gpts = [(x,k)  for k in range(ymin, ymax+1)]
        pts += gpts
    return pts

  """
  sorted_ylist returns the y list for given x from given
  curve maps, the points are sorted.
  """
  @staticmethod
  def sorted_ylist(ptsMap, x):
    ylist = []
    for pmap in ptsMap:
      if x in pmap:
        ylist.append(pmap[x][0])
    return sorted(ylist)

  """
  Fit a smooth spline to given set of point of open curve.
  """
  @staticmethod
  def fit_smooth_spline(pts, k=3):
    if len(pts) <= 15:
      return pts
    xlims = MathUtils.xlims(pts)
    if abs(xlims[1]-xlims[0]+1) <= k+2:
      return pts
    pmap = MathUtils.toMap(pts)

    pixel_div = 10 # Number of divisions between consecutive pixels.
    x = np.linspace(xlims[0], xlims[1], (xlims[1]-xlims[0])*pixel_div)
    y = [MathUtils.interp_y(i, pmap, 0) for i in x]

    n = int(len(pts)/20)
    t = sorted(np.linspace(xlims[0]+k,xlims[1]-k, n))
    t = np.r_[(x[0],)*(k+1), t, (x[-1],)*(k+1)]

    smooth_curve = make_lsq_spline(x, y, t, k)

    input_range = [i for i in range(xlims[0], xlims[1]+1)]
    y1 = list(smooth_curve(input_range))

    return [(input_range[i], int(y1[i])) for i in range(len(input_range))]

  """
  boundary_functions returns curves on the boundary that represent mathematical
  functions and when joined together, they give the entire boundary.
  """
  @staticmethod
  def boundary_functions(bpts):
    if len(bpts) == 0:
      return []
    bset = set(bpts)
    done = set()
    all_curves = []

    while (len(done)/len(bpts)) < 0.9:
      sp = MathUtils.boundary_starting_point(bpts, done)
      all_curves += MathUtils.boundary_fn_helper(sp, bset, done)
    return all_curves

  """
  boundary_starting_point returns the left most boundary point that
  has not been part of a function curve yet.
  """
  @staticmethod
  def boundary_starting_point(bpts, done):
    xlims = MathUtils.xlims(bpts)
    pmap = MathUtils.toMap(bpts)

    for x in range(xlims[0], xlims[1]+1):
      ylist = pmap[x]
      for y in ylist:
        if (x,y) in done:
          continue
        return (x,y)

    raise Exception("boundary starting point not found!")

  """
  boundary_fn_helper is a helper to find all curves associated with given
  boundary and starting from given point.
  """
  @staticmethod
  def boundary_fn_helper(sp, bset, done):
    all_curves = []
    curr_fn = [sp]
    vert_fn = []
    VERT_THRESH = 5 # 15 pixels that are just vertical i.e. only y changes by 1.
    done.add(sp)
    # d is the direction of movement and it can either be +1 (+x) or -1 (-x).
    d = 1
    pp = sp

    while done != bset:
      cp, is_vert = MathUtils.next_pt(pp, d, bset, done)
      if len(cp) == 0:
        # change direction.
        d *= -1
        if len(vert_fn) >= VERT_THRESH:
          all_curves.append(curr_fn[:-len(vert_fn)])
          all_curves.append(vert_fn)
        else:
          all_curves.append(curr_fn)

        cp = MathUtils.backtrack(d, bset, done, curr_fn)
        if len(cp) == 0:
          break
        curr_fn = [cp]
        vert_fn = []
      else:
        curr_fn.append(cp)
        if is_vert:
          vert_fn.append(cp)
        else:
          if len(vert_fn) >= VERT_THRESH:
            all_curves.append(curr_fn[:-len(vert_fn)])
            all_curves.append(vert_fn)
          vert_fn = []

      done.add(cp)
      pp = cp

    if len(vert_fn) >= VERT_THRESH:
      all_curves.append(curr_fn[:-len(vert_fn)])
      all_curves.append(vert_fn)
    else:
      all_curves.append(curr_fn)
    return all_curves


  """
  next_pt returns next point on boundary for given d.
  """
  @staticmethod
  def next_pt(cp, d, pset, done):
    x, y = cp
    if (x+d,y) in pset and (x+d, y) not in done:
      return (x+d, y), False
    if (x+d,y+1) in pset and (x+d, y+1) not in done:
      return (x+d, y+1), False
    if (x+d,y-1) in pset and (x+d, y-1) not in done:
      return (x+d, y-1), False
    if (x,y-1) in pset and (x, y-1) not in done:
      return (x, y-1), True
    if (x,y+1) in pset and (x, y+1) not in done:
      return (x, y+1), True

    return (), False

  """
  backtrack backtracks current point in given direction until
  we reach a point where the next pooint exist.
  """
  @staticmethod
  def backtrack(d, bset, done, curr_list):
    clist = curr_list.copy()
    while len(clist) > 0:
      cp = clist.pop()
      tp, _ = MathUtils.next_pt(cp, d, bset, done)
      if len(tp) != 0:
        return tp

    return ()


  """
  fill_curves returns set of points inside polygon containing given curves.
  note that both curves must have same x lims.
  """
  @staticmethod
  def fill_curves(pts1, pts2):
    if MathUtils.xlims(pts1) != MathUtils.xlims(pts2):
      raise Exception("Curves pts1 and pts2 do not have same range")

    xlims = MathUtils.xlims(pts1)
    pt1Map = MathUtils.curve_map(pts1)
    pt2Map = MathUtils.curve_map(pts2)

    pts = []
    for x in range(xlims[0], xlims[-1]+1):
      y1, y2 = pt1Map[x], pt2Map[x]
      for j in range(min(y1,y2), max(y1,y2)+1):
        pts.append((x,j))
    return pts

  """
  Interp y returns the float value of ordinate for given float
  value of x by interpolating y values corresponding to integer values
  closest to x in the given curve.
  """
  @staticmethod
  def interp_y(x, pmap, idx):
    if round(x) == x:
      return pmap[x][idx]
    ylower = pmap[math.floor(x)][idx]
    yupper = pmap[math.ceil(x)][idx]

    m = yupper - ylower
    c = yupper -m*math.ceil(x)
    return m*x + c

  """
  polygon_mask returns the mask for a given input polygon bounding vertices.
  The input image to ensure that the shape of the mask is always in bounds.
  """
  @staticmethod
  def polygon_mask(points, img):
    x = [p[0] for p in points]
    y = [p[1] for p in points]

    plist = []
    rr, cc = skimage.draw.polygon(x, y, shape=(img.shape[1], img.shape[0]))
    for i in range(len(rr)):
      plist.append((int(rr[i]),int(cc[i])))
    return plist

if __name__ == "__main__":
  points = [(1,2), (5,10), (3, -4)]
  print (MathUtils.best_fit_circle(points))
