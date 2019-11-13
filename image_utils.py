"""
image_utils is has helpul functions to process
an image.
"""
import os
import cv2
import numpy as np
import math
import csv
import json
import random
import time
import boto3
import bisect
import ijson
import matplotlib.pyplot as plt

from skimage import color
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_diff import delta_e_cie2000
from colormath.color_conversions import convert_color
from scipy.sparse import diags

class ImageUtils:
  windowName = ""

  BLACK = (0,0,0)
  WHITE = (255, 255, 255)

  """
  Initialize conditions for plottig.
  """
  @staticmethod
  def init():
    ImageUtils.windowName = "image"
    cv2.namedWindow(ImageUtils.windowName,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ImageUtils.windowName, 900,900)

  """
  Reads and returns a LAB scale and gray scale image from given image path.
  """
  @staticmethod
  def read(imagePath):
    img = cv2.imread(imagePath)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray, lab, img

  """
  Performs histogram equalization to adjust contrast in gray image.
  """
  @staticmethod
  def equalize_hist(gray):
    return cv2.equalizeHist(gray)

  """
  Performs adaptive equalization to adjust contrast in gray image.
  Currently, not using this function but could be used in the future.
  """
  @staticmethod
  def adaptive_hist(gray):
    clahe = cv2.createCLAHE(clipLimit=0.0,tileGridSize=(8,8))
    return clahe.apply(gray)

  """
  lab_diff computes the difference between color values
  c1 and c2 in LAB color space.
  """
  @staticmethod
  def lab_diff(c1, c2):
    color1 = LabColor(lab_l=c1[0], lab_a=c1[1], lab_b=c1[2])
    color2 = LabColor(lab_l=c2[0], lab_a=c2[1], lab_b=c2[2])
    return delta_e_cie2000(color1, color2)

  """
  plot_points plots given points on given gray image.
  """
  @staticmethod
  def plot_points(img, points):
    for (x,y) in points:
      cv2.circle(img, (x,y), 0, 255, 0)

  """
  Plots a rectangle with p1 as left-top point and p2 as right-botton point.
  """
  @staticmethod
  def plot_rect(img, p1, p2):
    cv2.rectangle(img, p1, p2, 0, 2)

  """
  color return the RGB tuple of given hex color. Hex color format is #FF0033.
  """
  @staticmethod
  def color(hexcolor):
    h = hexcolor.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2 ,4))

  """
  HEX2RGB returns RGB numpy array for given hex color.
  """
  def HEX2RGB(hexcolor):
    h = hexcolor.lstrip('#')
    return np.array([int(h[i:i+2], 16) for i in (0, 2 ,4)])

  """
  rgb_to_hex converts given RGB tuple to a hex string.
  """
  @staticmethod
  def rgb_to_hex(p):
    return '#{:02x}{:02x}{:02x}'.format(p[0], p[1] , p[2])

  """
  set_color sets given points to given color.
  """
  @staticmethod
  def set_color(img, pts, clr, radius=0):
    for _, p in enumerate(pts):
      cv2.circle(img, p, 0, clr, radius)

  """
  avg_color returns the average color of the given points. The points
  and image are in RGB LAB color space.
  """
  @staticmethod
  def avg_color(img, pts):
    if len(pts) == 0:
      return None
    c = (0,0,0)
    for _, p in enumerate(pts):
      nc = img[p[0],p[1]]
      c = (c[0] + nc[0]**2, c[1] + nc[1]**2, c[2] + nc[2]**2)

    l = len(pts)
    #return ImageUtils.brighter((int(c[0]/l),int(c[1]/l),int(c[2]/l)))
    return (int(math.sqrt(c[0]/l)),int(math.sqrt(c[1]/l)),int(math.sqrt(c[2]/l)))

  """
  avg_lab_color is the average color of the given points. The points and
  image are in LAB color space.
  """
  def avg_lab_color(img, pts):
    if len(pts) == 0:
      return None
    c = [0, 0, 0]
    l = len(pts)
    for p in pts:
      nc = img[p[0], p[1]]
      c = [sum(x) for x in zip(c,nc)]

    return (int(c[0]/l), int(c[1]/l), int(c[1]/l))

  """
  average intensity returns the average intensity of the given set of points.
  Input img must be dimension 1.
  """
  @staticmethod
  def avg_intensity(img, pts):
    avg = 0.0
    for p in pts:
      x, y = p
      avg += img[x, y]
    return avg/len(pts)

  """
  additiveBlendChannel will blend given RGB color channels additively.
  """
  @staticmethod
  def additiveBlendChannel(a, b, t):
    return int(math.sqrt(t*a*a + (1-t)*b*b))

  """
  additiveBlendRGB will additively blend the given RGB colors with the given
  ratio alpha:1-alpha in that order.
  """
  @staticmethod
  def additiveBlendRGB(c1, c2, alpha):
    r = ImageUtils.additiveBlendRGB(c1[0], c2[0], alpha)
    g = ImageUtils.additiveBlendRGB(c1[1], c2[1], alpha)
    b = ImageUtils.additiveBlendRGB(c1[2], c2[2], alpha)
    return (r,g,b)

  """
  Convert from RGB color to CYMK space.
  """
  @staticmethod
  def rgb_to_cmyk(r,g,b):
    rgb_scale = 255
    cmyk_scale = 100

    if (r == 0) and (g == 0) and (b == 0):
        # black
        return 0, 0, 0, cmyk_scale

    # rgb [0,255] -> cmy [0,1]
    c = 1 - r / float(rgb_scale)
    m = 1 - g / float(rgb_scale)
    y = 1 - b / float(rgb_scale)

    # extract out k [0,1]
    min_cmy = min(c, m, y)
    c = (c - min_cmy)
    m = (m - min_cmy)
    y = (y - min_cmy)
    k = min_cmy

    # rescale to the range [0,cmyk_scale]
    return c*cmyk_scale, m*cmyk_scale, y*cmyk_scale, k*cmyk_scale

  """
  Convert from CMYK color to RGB space.
  """
  @staticmethod
  def cmyk_to_rgb(c,m,y,k):
    rgb_scale = 255
    cmyk_scale = 100

    r = rgb_scale*(1.0-(c+k)/float(cmyk_scale))
    g = rgb_scale*(1.0-(m+k)/float(cmyk_scale))
    b = rgb_scale*(1.0-(y+k)/float(cmyk_scale))
    return (r,g,b)

  """
  subtractiveBlendRGB blends given list of color subtractively.
  """
  @staticmethod
  def subtractiveBlendRGB(list_of_colours):
    """input: list of rgb, opacity (r,g,b,o) colours to be added, o acts as weights.
    output (r,g,b)
    """
    C = 0
    M = 0
    Y = 0
    K = 0

    for (r,g,b,o) in list_of_colours:
        c,m,y,k = ImageUtils.rgb_to_cmyk(r, g, b)
        C+= o*c
        M+=o*m
        Y+=o*y
        K+=o*k

    return ImageUtils.cmyk_to_rgb(C, M, Y, K)

  """
  brighter makes the color by multiplying by constant.
  """
  def brighter(color):
    r,g, b = color
    c = min(1.25,255/max(r,b,b))
    return (int(r*c), int(g*c),  int(b*c))

  """
  filter_by_color returns all points of the image that belong to given
  color.
  """
  def filter_by_color(img, clr):
    points = []
    for i in range(img.shape[1]):
      for j in range(img.shape[0]):
        if not np.array_equal(img[j][i], clr):
          continue
        points.append((i,j))
    return points

  """
  rspectrum_lhtss calculates the reflectance spectrum for the
  given RGB color using the Leasr Hyperbolic Tangent Slop Squared (LHTSS)
  specified by Scott Burns on his site: http://scottburns.us/reflectance-curves-from-srgb/.
  This is the Python version of the specified MATLAB code.
  The reflectance spans the wavelength range 380-730 nm in 10 nm increments.
  rgb is a 3 element tuple of color in 0-255 range.
  rho is a (36,1) numpy array of reconstructed reflectance values, all (0->1).
  """
  def rspectrum_lhtss(rgb):
    T = ImageUtils.read_T_matrix()

    # convert from rgb tuple to list.
    rgb = [rgb[0], rgb[1], rgb[2]]

    numIntervals = T.shape[1]
    rho = np.zeros(numIntervals)

    # Black input color.
    if rgb == (0,0,0):
      rho = 0.0001*np.ones(numIntervals)
      return rho

    # White input color.
    if rgb == (255, 255, 255):
      rho = np.ones(numIntervals)
      return rho

    # 36 by 36 Jacobian having 4 main diagonal
    # and -2 on off diagonals, except for first and
    # last main diagonal are 2.
    D = diags([-2, 4, -2], [-1, 0, 1], shape=(numIntervals, numIntervals)).toarray()
    D[0][0] = 2
    D[numIntervals-1, numIntervals-1] = 2

    # Scale RGB values to between 0-1 and remove
    # gamma correction.
    for i in range(3):
      rgb[i] = rgb[i]/255.0
      if rgb[i] < 0.04045:
        rgb[i] = rgb[i]/12.92
      else:
        rgb[i] = math.pow((rgb[i] + 0.055)/1.055, 2.4)

    # Initialize paramters in optimization.
    z = np.zeros(numIntervals)
    lamda = np.zeros(3)
    ftol = math.pow(10, -8)
    count = 0
    NUM_ITERS = 100

    # Newton's method iteration.
    while count <= NUM_ITERS:
      d0 = (np.tanh(z) + 1)/2
      d1 = np.diag(np.power(1/np.cosh(z), 2)/2)
      d2 = np.diag(-np.power(1/np.cosh(z), 2)*np.tanh(z))
      F = np.concatenate((np.matmul(D,z) + np.matmul(d1, np.matmul(np.transpose(T), lamda)), np.matmul(T, d0)-rgb))
      J1 = np.concatenate((D+np.diag(np.matmul(d2, np.matmul(np.transpose(T), lamda))), np.matmul(d1, np.transpose(T))), axis=1)
      J2 = np.concatenate((np.matmul(T,d1), np.zeros((3,3))), axis=1)
      J = np.concatenate((J1, J2), axis=0)
      delta = np.matmul(np.linalg.inv(J), -F)
      z = z + delta[:numIntervals]
      lamda = lamda + delta[numIntervals:]
      if np.all(np.less(np.absolute(F)-ftol, np.zeros(numIntervals+3))):
        # Found solution.
        rho = (np.tanh(z) + 1)/2
        return rho
      count += 1

    raise Exception("rspectrum_lhtss: No solution found in iteration")

  """
  read_T_matrix reads the T matrix CSV containing T matrix used for predicting
  reflectance spectrum for a given color. Refer to http://scottburns.us/subtractive-color-mixture/
  for more details.
  Returned T is a (3,36) numpy array converting reflectance to D65 weighted linear rgb.
  """
  def read_T_matrix():
    T_matrix = []
    with open("T_matrix.csv") as f:
      csv_reader = csv.reader(f, delimiter=",")
      rows = []
      for i, row in enumerate(csv_reader):
        if i == 0:
          continue
        rows.append(row[1:])

      return np.stack(rows, axis = 0).astype(np.float)

  """
  reflectance_coeffs returns the absorption, scattering coefficient, a and b for
  given reflectance spectrum assumed to be at infinity. The optical
  depth is assumed to be 2mm at given reflectance.
  """
  def reflectance_coeffs(rho_infinity):
    D = 2# 2 mm.
    a = 0.5 * (1/rho_infinity + rho_infinity)
    b = np.sqrt(np.power(a,2)-1)

    rho = rho_infinity*0.99
    S = (np.arctanh(b/(a-rho))-np.arctanh(b/a))/(b*D)
    K = S*(a-1)

    return K, S, a, b

  """
  r_and_t_helper computes the reflectance and transmittance
  of given reflectance spectrum (at inifinite thickness) at given thickness
  for a foundation.
  """
  def r_and_t_helper(rho_infinity, x):
    K, S, a,b = ImageUtils.reflectance_coeffs(rho_infinity)
    R = np.sinh(b*S*x)/(a*np.sinh(b*S*x) + b*np.cosh(b*S*x))
    T = b/(a*np.sinh(b*S*x) + b*np.cosh(b*S*x))
    return R, T

  """
  plot_reflectance is a rest function to plot the reflectance of
  an RGB color for different optical depths ranging between 0.1 to 2 mm.
  """
  def plot_reflectance():
    rho_infinity  = ImageUtils.rspectrum_lhtss((255, 255, 0))
    x = [i for i in range(380, 731, 10)]
    for d in [0.1, 0.4, 0.7, 1, 1.3, 1.6, 1.9, 2.2]:
      R, _ = ImageUtils.r_and_t_helper(rho_infinity, d)
      plt.plot(x, R)

    plt.plot(x, rho_infinity)
    plt.show()

  """
  hash_tuple provides a string representation of the given tuple
  to use as key when saving dictionary as JSON.
  """
  def hash_tuple(t):
    return str(t[0]) + "-" + str(t[1]) + "-" + str(t[2])

  """
  reverse_hash_tuple returns a tuple from given string representing the dictionary
  key when saving to JSON.
  """
  def reverse_hash_tuple(tstr):
    tvals = tstr.split("-")
    return (int(tvals[0]), int(tvals[1]), int(tvals[2]))

  """
  compute_all_reflectances will compute reflectance (infinite optical depth)
  for all colors in the RGB space and store them in a JSON file.
  """
  def compute_all_reflectances():
    d = {}
    total = 256*256*256
    print ("Total: ", total)
    next_milestone = 0
    for i in range(256):
      for j in range(256):
        for k in range(256):
          rho_infinity = ImageUtils.rspectrum_lhtss((i, j, k))
          d[ImageUtils.hash_tuple((i, j, k))] = rho_infinity.tolist()
          if int(len(d)/total)*100 >= next_milestone:
            print (str(next_milestone) + "% complete")
            next_milestone += 1

    with open("reflectance.json", "w") as f:
      json.dump(d, f)


  """
  random_color generates a random RGB color each time it is called.
  """
  def random_color():
    return tuple(np.random.choice(range(256), size=3))

  """
  RGB2HEX converts RGB tuple to Hex string.
  """
  def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

  """
  flatten_rgb flattens the given RGB tuple into its corresponding index number
  from 1 to 256*256*256.
  """
  def flatten_rgb(rgb):
    return rgb[0]*256*256 + rgb[1]*256 + rgb[2] + 1

  """
  get_sorted_reflectance_list returns sorted reflectance list for all parts from S3.
  """
  def get_sorted_reflectance_list():
    # Find all reflectance files.
    s3 = boto3.resource("s3")
    my_bucket = s3.Bucket("addboxdrop")
    allobjs = []
    for obj in my_bucket.objects.all():
      if not obj.key.startswith("parts"):
        continue
      allobjs.append(obj.key)

    return sorted([int(obj.split("/")[1].split(".")[0]) for obj in allobjs])

  """
  get_reflectances returns the reflectance spectrum for all input RGBs.
  Input is (n, 3) and output is (n, 36).
  """
  def get_reflectances(rgbList):
    # Set of all colors for which we need to find the reflectance.
    rgbSet = set(rgbList)

    # Walk through entire reflectance file and map reflectance value
    # a given rgb color.
    refMap = {}
    with open("reflectance.json", "r") as f:
      currKey = None
      currRef = []
      for prefix, the_type, value in ijson.parse(f):
        if the_type == "map_key":
          keyTuple = ImageUtils.reverse_hash_tuple(value)
          if keyTuple in rgbSet:
            currKey = keyTuple
            continue
        if not currKey:
          continue
        if the_type == "number":
          currRef.append(float(value))
        elif the_type == "end_array":
          refMap[currKey] = currRef
          rgbSet.remove(currKey)
          currKey = None
          currRef = []
          if len(rgbSet) == 0:
            break

    return np.array([refMap[rgb] for rgb in rgbList])

  """
  show plots the image and blocks until user presses a key.
  If user presses 'q', returns False to indicate user wants to quit
  else returns True.
  """
  def show(img):
    cv2.imshow(ImageUtils.windowName, img)
    key = cv2.waitKey(0)
    if key & 0xff == ord("q"):
      return False
    return True

if __name__ == "__main__":
  rhoList = ImageUtils.get_reflectances([(0, 255, 0)])
  x = [i for i in range(380, 731, 10)]
  plt.plot(x, rhoList)
  plt.show()
