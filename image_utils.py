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
import colour
import matplotlib.pyplot as plt

from skimage import color
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_diff import delta_e_cie2000
from colormath.color_conversions import convert_color
from scipy.sparse import diags
from scipy.optimize import minimize

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
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return gray, lab, img, hsv

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
  and image are in RGB color space.
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
  given RGB color using the Least Hyperbolic Tangent Slope Squared (LHTSS)
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
    divider = 2
    while count <= NUM_ITERS:
      d0 = (np.tanh(z) + 1)/divider
      d1 = np.diag(np.power(1/np.cosh(z), 2)/divider)
      d2 = np.diag(-np.power(1/np.cosh(z), 2)*np.tanh(z)*(2/divider))
      F = np.concatenate((np.matmul(D,z) + np.matmul(d1, np.matmul(np.transpose(T), lamda)), np.matmul(T, d0)-rgb))
      J1 = np.concatenate((D+np.diag(np.matmul(d2, np.matmul(np.transpose(T), lamda))), np.matmul(d1, np.transpose(T))), axis=1)
      J2 = np.concatenate((np.matmul(T,d1), np.zeros((3,3))), axis=1)
      J = np.concatenate((J1, J2), axis=0)
      delta = np.matmul(np.linalg.inv(J), -F)
      z = z + delta[:numIntervals]
      lamda = lamda + delta[numIntervals:]
      if np.all(np.less(np.absolute(F)-ftol, np.zeros(numIntervals+3))):
        # Found solution.
        rho = (np.tanh(z) + 1)/divider
        return rho
      count += 1

    raise Exception("rspectrum_lhtss: No solution found in iteration")

  """
  rspectrum_lhtss calculates the reflectance spectrum for the
  given RGB color using the Iterative Least Slope Squared (LHTSS)
  specified by Scott Burns on his site: http://scottburns.us/reflectance-curves-from-srgb/.
  This is the Python version of the specified MATLAB code.
  The reflectance spans the wavelength range 380-730 nm in 10 nm increments.
  rgb is a 3 element tuple of color in 0-255 range.
  rho is a (36,1) numpy array of reconstructed reflectance values, all (0->1).
  """
  def rspectrum_ilss(rgb):
    T = np.power(ImageUtils.read_T_matrix(), 1)

    # convert from rgb tuple to list.
    rgb = np.array([rgb[0], rgb[1], rgb[2]], dtype=np.float)

    numIntervals = T.shape[1]
    rho = np.ones(numIntervals)/2.0
    rhomin = 0.00001
    rhomax = 1

    if np.array_equal(rgb ,[0,0,0]):
      return rhomin*np.ones(numIntervals)

    if np.array_equal(rgb, [1, 1, 1]):
      return rhomax*np.ones(numIntervals)

    # Scale RGB values to between 0-1 and remove
    # gamma correction.
    for i in range(3):
      rgb[i] = rgb[i]/255.0
      if rgb[i] < 0.04045:
        rgb[i] = rgb[i]/12.92
      else:
        rgb[i] = math.pow((rgb[i] + 0.055)/1.055, 2.4)

    # 36 by 36 Jacobian having 4 main diagonal
    # and -2 on off diagonals, except for first and
    # last main diagonal are 2.
    #D = diags([-2, 4, -2], [-1, 0, 1], shape=(numIntervals, numIntervals)).toarray()
    D = diags([-6, 20, -6], [-1, 0, 1], shape=(numIntervals, numIntervals)).toarray()
    D[0][0] = 2
    D[numIntervals-1, numIntervals-1] = 2


    B = np.linalg.inv(np.concatenate((np.concatenate((D, np.transpose(T)), axis=1), np.concatenate((T, np.zeros((3, 3))), axis=1)), axis=0))

    R = np.matmul(B[:numIntervals,numIntervals:], rgb)

    NUM_ITERS = 10
    count = 0
    while count == 0 or (count <= NUM_ITERS and (np.any(np.less(rho-rhomin, np.zeros(numIntervals))) or np.any(np.less(rhomax-rho, np.zeros(numIntervals))))):
      # Create K1 for fixed reflectance at rhomax.
      fixed_upper_logical = rho >= rhomax
      fixed_upper = np.nonzero(fixed_upper_logical)[0]
      num_upper = len(fixed_upper)
      K1 = np.zeros((num_upper, numIntervals))
      K1[range(K1.shape[0]), fixed_upper] = 1

      # Create K0 for fixed reflectance at rhomin.
      fixed_lower_logical = rho <= rhomin
      fixed_lower = np.nonzero(fixed_lower_logical)[0]
      num_lower = len(fixed_lower)
      K0 = np.zeros((num_lower, numIntervals))
      K0[range(K0.shape[0]), fixed_lower] = 1

      # Set up linear system.
      K = np.concatenate((K1, K0), axis=0)
      C = np.matmul(np.matmul(B[:numIntervals,:numIntervals], np.transpose(K)), np.linalg.inv(np.matmul(K, np.matmul(B[:numIntervals, :numIntervals], np.transpose(K)))))
      rho = R - np.matmul(C, np.matmul(K, R) - np.concatenate((rhomax*np.ones(num_upper), rhomin*np.ones(num_lower)), axis=0))
      rho[fixed_upper_logical] = rhomax
      rho[fixed_lower_logical] = rhomin

      count += 1

    if count > NUM_ITERS:
      raise Exception("No solution found after: ", str(NUM_ITERS), " iterations")

    return rho

  """
  rspectrum_foundation estimates the reflectance of a given foundation color.
  """
  def rspectrum_foundation(rgb):
    T = ImageUtils.read_T_matrix()

    # convert from rgb tuple to list.
    rgb = np.array([rgb[0], rgb[1], rgb[2]], dtype=np.float)
    numIntervals = T.shape[1]
    rhomin = 0.0001
    rhomax = 1.0

    if np.array_equal(rgb ,[0,0,0]):
      return rhomin*np.zeros(numIntervals)

    if np.array_equal(rgb, [1, 1, 1]):
      return rhomax*np.ones(numIntervals)

    # Scale RGB values to between 0-1 and remove
    # gamma correction.
    for i in range(3):
      rgb[i] = rgb[i]/255.0
      if rgb[i] < 0.04045:
        rgb[i] = rgb[i]/12.92
      else:
        rgb[i] = math.pow((rgb[i] + 0.055)/1.055, 2.4)

    rho0 = 0.01*np.ones(numIntervals)

    def loss_fn(rho):
      return np.sum(np.diff(rho)**2)

    # Define constraints and bounds.
    cons = [{'type': 'eq', 'fun': lambda rho: (T @ rho) -rgb}, {'type': 'eq', 'fun': lambda rho: rho[0] -0.01}, {'type': 'ineq', 'fun': lambda rho: rho[-1] -0.4}]
    bounds = [(rhomin, rhomax)] * numIntervals

    minout = minimize(loss_fn, rho0, method='SLSQP',bounds=bounds,constraints=cons)
    print ("success: ", minout.success, minout.message)

    return minout.x


  """
  rspectrum_skin estimates the reflectance of a given skin color.
  """
  def rspectrum_skin(rgb):
    T = ImageUtils.read_T_matrix()

    # convert from rgb tuple to list.
    rgb = np.array([rgb[0], rgb[1], rgb[2]], dtype=np.float)
    numIntervals = T.shape[1]
    rhomin = 0.0001
    rhomax = 1

    if np.array_equal(rgb ,[0,0,0]):
      return rhomin*np.zeros(numIntervals)

    if np.array_equal(rgb, [1, 1, 1]):
      return rhomax*np.ones(numIntervals)

    # Scale RGB values to between 0-1 and remove
    # gamma correction.
    for i in range(3):
      rgb[i] = rgb[i]/255.0
      if rgb[i] < 0.04045:
        rgb[i] = rgb[i]/12.92
      else:
        rgb[i] = math.pow((rgb[i] + 0.055)/1.055, 2.4)

    rho0 = 0.01*np.ones(numIntervals)

    def loss_fn(rho):
      return np.sum(np.diff(rho)**2)

    # Define constraints and bounds.
    cons = [{'type': 'eq', 'fun': lambda rho: (T @ rho) -rgb}, {'type': 'eq', 'fun': lambda rho: rho[0] -0.01}, {'type': 'ineq', 'fun': lambda rho: rho[-1] -0.4}]
    bounds = [(rhomin, rhomax)] * numIntervals

    minout = minimize(loss_fn, rho0, method='SLSQP',bounds=bounds,constraints=cons)
    print ("success: ", minout.success, minout.message)

    return minout.x

  """
  sRGB_to_XYZ converts given sRGB array (0-255) to XYZ (0-1) under D65 illuminant.
  """
  def sRGB_to_XYZ(rgb):
    # Convert RGB to [0-1] and remove gamma correction.
    rgb = ImageUtils.remove_gamma_correction(np.array(rgb, dtype=float))
    illuminant_D65 = np.array([0.31270, 0.32900])
    RGB_to_XYZ_matrix = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],[0.0193339, 0.1191920, 0.9503041]])
    return colour.RGB_to_XYZ(rgb, illuminant_D65, illuminant_D65, RGB_to_XYZ_matrix, None)

  """
  sRGB_to_XYZ_matrix converts given sRGB array (n,3) (0-255) into XYZ array(n,3) (0-1).
  """
  def sRGB_to_XYZ_matrix(rgbArr):
    rgbArr = ImageUtils.remove_gamma_correction_matrix(rgbArr)
    RGB_to_XYZ_matrix = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],[0.0193339, 0.1191920, 0.9503041]])
    return np.transpose(RGB_to_XYZ_matrix @ np.transpose(rgbArr))

  """
  sRGB_to_XYZ converts given XYZ array (0-1) to sRGB (0-255) under D65 illuminant.
  """
  def XYZ_to_sRGB(xyz):
    XYZ_to_RGB_matrix = np.array([[3.2404542, -1.5371385, -0.4985314],[-0.9692660, 1.8760108, 0.0415560],[0.0556434, -0.2040259, 1.0572252]])
    illuminant_D65 = np.array([0.31270, 0.32900])
    rgb = colour.XYZ_to_RGB(xyz, illuminant_D65, illuminant_D65, XYZ_to_RGB_matrix, None)
    return ImageUtils.add_gamma_correction(rgb)

  """
  XYZ_to_sRGB_matrix converts given XYZ array (n, 3)(0-1) to sRGB(n,3) (0-255) under D65 illuminant.
  """
  def XYZ_to_sRGB_matrix(xyzArr):
    XYZ_to_RGB_matrix = np.array([[3.2404542, -1.5371385, -0.4985314],[-0.9692660, 1.8760108, 0.0415560],[0.0556434, -0.2040259, 1.0572252]])
    srgbArr = np.transpose(XYZ_to_RGB_matrix @ np.transpose(xyzArr))
    return ImageUtils.add_gamma_correction_matrix(srgbArr)

  """
  sRGB_to_uv converts given sRGB array(0-255) to uv(0-1) chromaticity.
  """
  def sRGB_to_uv(srgb):
    xyz = ImageUtils.sRGB_to_XYZ(srgb)
    xy = colour.XYZ_to_xy(xyz)
    return colour.xy_to_Luv_uv(xy)


  """
  chromatic_adaptation converts given
  """
  def chromatic_adaptation(imagePath, source_white):
    swhite_xyz = ImageUtils.sRGB_to_XYZ_matrix(np.array(source_white))
    d65_xyz = colour.xy_to_XYZ(np.array([0.31270, 0.32900]))

    m_bradford = np.array([[0.8951, 0.2664000, -0.1614000],[-0.7502000, 1.7135000, 0.0367000],[0.0389000, -0.0685000, 1.0296000]])
    m_gain = np.diag((m_bradford @ d65_xyz)/(m_bradford @ swhite_xyz))
    m_transform = np.linalg.inv(m_bradford) @ m_gain @ m_bradford

    # Chromatically adapt given image.
    _, _, img, _ = ImageUtils.read(imagePath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    flat_img = np.reshape(img, (-1,3))
    flat_xyz = ImageUtils.sRGB_to_XYZ_matrix(flat_img)
    adapted_xyz = np.transpose(m_transform @ np.transpose(flat_xyz))
    adapted_img = np.clip(ImageUtils.XYZ_to_sRGB_matrix(adapted_xyz), 0, 255).astype(np.uint8)
    img = cv2.cvtColor(np.reshape(adapted_img, img.shape), cv2.COLOR_RGB2BGR)

    print ("imgs shape: ", img.shape)
    cv2.namedWindow("image",cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image", 900,900)
    cv2.imshow("image", img)
    cv2.waitKey(0)


  """
  sRGB_to_SD converts given sRGB numpy array
  to a spectral distribution using color science library.
  """
  def sRGB_to_SD(srgb):
    # Convert RGB to [0-1] and remove gamma correction.
    rgb = ImageUtils.remove_gamma_correction(np.array(srgb, dtype=float))

    # D65
    illuminant_RGB = np.array([0.31270, 0.32900])
    illuminant_XYZ = np.array([0.31270, 0.32900])

    RGB_to_XYZ_matrix = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],[0.0193339, 0.1191920, 0.9503041]])

    xyz = colour.RGB_to_XYZ(rgb, illuminant_RGB, illuminant_XYZ, RGB_to_XYZ_matrix, None)
    cmfs = colour.STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer'].copy().align(colour.SpectralShape(380, 730, 10))
    sd = colour.recovery.XYZ_to_sd_Meng2015(xyz, cmfs, colour.ILLUMINANTS_SDS['D65'])
    return sd.values

  """
  SD_to_RGB converts given spectral distribution values to sRGB using color
  science library.
  """
  def SD_to_RGB(sd_values):
    x = ImageUtils.wavelength_arr()
    assert len(sd_values) == len(x)
    d = {}
    for i, l in enumerate(x):
      d[l] = sd_values[i]

    sd = colour.SpectralDistribution(d)
    xyz = colour.sd_to_XYZ(sd, colour.CMFS['CIE 1931 2 Degree Standard Observer'], colour.ILLUMINANTS_SDS['D65'])/100.0

    # D65
    illuminant_RGB = np.array([0.31270, 0.32900])
    illuminant_XYZ = np.array([0.31270, 0.32900])
    XYZ_to_RGB_matrix = np.array([[3.2404542, -1.5371385, -0.4985314],[-0.9692660, 1.8760108, 0.0415560],[0.0556434, -0.2040259, 1.0572252]])
    rgb = colour.XYZ_to_RGB(xyz, illuminant_XYZ, illuminant_RGB, XYZ_to_RGB_matrix, None)
    return ImageUtils.add_gamma_correction(rgb)

  """
  add_gamma_correction adds gamma correction to given RGB value and
  returns the new value. Input rgb has to be between 0-1.
  """
  def add_gamma_correction(rgb):
    for i in range(3):
      if rgb[i] < 0.0031308:
        rgb[i] = int(12.92*rgb[i]*255)
      else:
        rgb[i] = int(255*(1.055*(math.pow(rgb[i], 1/2.4))-0.055))

    return rgb

  """
  wavelength_arr returns the numpy array of wavelengths for which
  we are interested in Spectral distribution.
  """
  def wavelength_arr():
    return [i for i in range(380, 731, 10)]

  """
  remove_gamma_correction removes gamma correction from sRGB to given Linear RGB value.
  Input sRGB has to be between 0-255.
  """
  def remove_gamma_correction(rgb):
    for i in range(3):
      rgb[i] = rgb[i]/255.0
      if rgb[i] < 0.04045:
        rgb[i] = rgb[i]/12.92
      else:
        rgb[i] = math.pow((rgb[i] + 0.055)/1.055, 2.4)
    return rgb

  """
  numpy version of removing gamma correction to given array of RGB colors
  that have values between 0-255.
  """
  def remove_gamma_correction_matrix(rgbArr):
    rgbArr = (rgbArr/255.0).astype(np.float)
    return np.where(rgbArr < 0.04045, rgbArr/12.92, np.power((rgbArr + 0.055)/1.055, 2.4)).astype(np.float)

  """
  numpy version of adding gamma correction to given array of RGB colors
  that have values between 0-1.
  """
  def add_gamma_correction_matrix(rgbArr):
    return np.where(rgbArr < 0.0031308, 12.92*rgbArr*255, 255*(1.055*(np.power(rgbArr, 1/2.4))-0.055)).astype(int)

  """
  geometric_mean_mixing takes input as the given hex colors and then returns
  the color mix (geometric mean of reflectances).
  """
  def geometric_mean_mixing(c1, c2, alpha):
    rgb1, rgb2 = ImageUtils.color(c1), ImageUtils.color(c2)
    T = ImageUtils.read_T_matrix()
    rgb = np.matmul(T, np.power(ImageUtils.rspectrum_lhtss(rgb1), alpha)*np.power(ImageUtils.rspectrum_lhtss(rgb2), 1-alpha))
    return 255*rgb
    return ImageUtils.add_gamma_correction(rgb)

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
  R_foundation_depth given the reflectance and transmittance of the given foundation matrix
  at given depth percent (between 0-1 inclusive).
  """
  def R_foundation_depth(R_foundation, x_percent):
    # Assume 2 mm is the maximum depth of the foundation.
    D = 2
    a = 0.5 * ((1/R_foundation) + R_foundation)
    b = np.sqrt((a**2)-1)

    rho = R_foundation*0.99
    S = (np.arctanh(b/(a-rho))-np.arctanh(b/a))/(b*D)
    K = S*(a-1)

    # Find reflectance and transmittance at given depth.
    x = x_percent*D
    R = np.sinh(b*S*x)/(a*np.sinh(b*S*x) + b*np.cosh(b*S*x))
    T = b/(a*np.sinh(b*S*x) + b*np.cosh(b*S*x))

    return R, T

  """
  plot_reflectance is a rest function to plot the reflectance of
  an RGB color (in hex format) for different optical depths ranging between 0.1 to 2 mm.
  """
  def plot_reflectance(hexColors):
    x = ImageUtils.wavelength_arr()
    for hexColor in hexColors:
      rho  = ImageUtils.sRGB_to_SD(ImageUtils.color(hexColor))
      plt.plot(x, rho, label=hexColor, color=hexColor, linewidth=2)

    plt.legend()
    plt.show()


  """
  plot_skin_spectra will read a database of skin spectra and plot it.
  """
  def plot_skin_spectra():
    with open("skin_spectra.csv") as f:
      csv_reader = csv.reader(f, delimiter=",")
      x = None
      for i, row in enumerate(csv_reader):
        g = np.array(row[2:]).astype(np.float)
        if i == 0:
          x = g
        elif (i > 0*482 and i < 4*482) or (i >= 7*482 and i <= 9*482):
          plt.plot(x, g, linewidth=2)
    plt.show()


  """
  save_skin_spectra_uv will convert the reflectance spectrums of skin spectra
  to uv chromaticity and save it.
  """
  def save_skin_spectra_uv():
    skin_chrs = []
    with open("skin_spectra.csv") as f:
      csv_reader = csv.reader(f, delimiter=",")
      x = ImageUtils.wavelength_arr()
      for i, row in enumerate(csv_reader):
        if i == 0:
          continue
        rho = np.array(row[4:-1]).astype(np.float)
        if (i > 0*482 and i < 4*482) or (i >= 7*482 and i <= 9*482):
          d = {}
          for k, l in enumerate(x):
            d[l] = rho[k]

          sd = colour.SpectralDistribution(d)
          xyz = colour.colorimetry.sd_to_XYZ_integration(sd, colour.CMFS['CIE 1931 2 Degree Standard Observer'].copy().align(colour.SpectralShape(380, 730, 10)), colour.ILLUMINANTS_SDS['D65'].copy().align(colour.SpectralShape(380, 730, 10)))/100.0
          xy = colour.XYZ_to_xy(xyz)
          uv = colour.xy_to_Luv_uv(xy)
          skin_chrs.append(list(uv))

    plt.scatter(*zip(*skin_chrs))
    plt.show()

    with open("skin_uv.json", "w") as f:
      json.dump(skin_chrs, f)

  """
  plot_foundation_depth will plot given foundation at different thickness
  levels. We assume that the given foundation SD is at inifinite optical depth
  and is therefore opaque.
  """
  def plot_foundation_depth(skinColor, hexColor):
    x = ImageUtils.wavelength_arr()
    R_skin = ImageUtils.sRGB_to_SD(ImageUtils.color(skinColor))
    R_foundation = ImageUtils.sRGB_to_SD(ImageUtils.color(hexColor))
    for x_percent in [0.01, 0.05, 0.08, 0.1, 0.2, 0.3, 0.5, 1]:
      rho_f, T_f = ImageUtils.R_foundation_depth(R_foundation, x_percent)
      #rho_mix = np.power(np.power(R_skin,0.85) * np.power(rho_f,0.5), 1)
      #rho_mix = 0.0*R_skin + rho_f
      Rs = 0.1
      rho_mix = (1-Rs)*(rho_f + ((T_f**2)*R_skin)/(1-(R_skin*rho_f))) + Rs
      color = ImageUtils.RGB2HEX(ImageUtils.SD_to_RGB(rho_mix))
      plt.plot(x, rho_mix, label=color+"-"+str(x_percent), color=color, linewidth=2)

    plt.legend()
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

  """
  show_image opens up the given image in BGR space.
  """
  def show_image(imagePath):
    _, _, img, _ = ImageUtils.read(imagePath)
    img = np.clip(img, None, 255).astype(np.uint8)
    cv2.namedWindow("image",cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image", 900,900)
    cv2.imshow("image", img)
    cv2.waitKey(0)

  """
  show_gray opens up the given image in gray space.
  """
  def show_gray(imagePath):
    gray, _, _, hsv = ImageUtils.read(imagePath)
    gray2 = np.clip(gray*1.5, None, 255).astype(np.uint8)
    cv2.namedWindow("image",cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image", 900,900)
    cv2.imshow("image", gray2)
    cv2.waitKey(0)

if __name__ == "__main__":
  ImageUtils.save_skin_spectra_uv()
  #ImageUtils.plot_skin_spectra()
  #ImageUtils.chromatic_adaptation("test/ancha.JPG", ImageUtils.color("#caf0fc"))
  #ImageUtils.chromatic_adaptation("test/IMG_8803.JPG", ImageUtils.color("#fecaaf"))
  #ImageUtils.chromatic_adaptation("test/IMG_8803.JPG", ImageUtils.color("#fef2d5"))
  #ImageUtils.chromatic_adaptation("test/ancha_3.JPG", ImageUtils.color("#e5e2ce"))
  #ImageUtils.chromatic_adaptation("test/IMG_5862.JPG", ImageUtils.color("#e9e4dc"))
  #ImageUtils.show_gray("/Users/addarsh/Desktop/ancha.png")
  #ImageUtils.show_image("test/IMG_5862.JPG")
  #ImageUtils.plot_foundation_depth("#e19094","#daa48a")
  #ImageUtils.plot_foundation_depth("#e19094","#eabea1")
  #ImageUtils.plot_foundation_depth("#e19094","#d9b28b")
  #ImageUtils.plot_foundation_depth("#E2B9A7","#e3bca4")
  #ImageUtils.white_balance(ImageUtils.color("#F1DCC1"))
  #ImageUtils.plot_reflectance(["#E4C9AD", "#C39B82", "#CDA18B", "#ECC5B5"])
  #ImageUtils.plot_reflectance(["#946E59", "#E4CCB2", "#E9C8B3", "#FDF0E4", "#5F3E35"])
  #print ("Mixed color: ", ImageUtils.RGB2HEX(ImageUtils.geometric_mean_mixing("#D0B9C1", "#DAC4BC", 0.8)))
