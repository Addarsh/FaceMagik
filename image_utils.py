"""
image_utils is has helpul functions to process
an image.
"""
import cv2
import numpy as np
import math

from skimage import color
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_diff import delta_e_cie2000
from colormath.color_conversions import convert_color

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
