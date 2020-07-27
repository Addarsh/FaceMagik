import os
import argparse
import cv2
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from image_utils import ImageUtils
from colour.plotting import *
from face import Face

windowName = "image"
cv2.namedWindow(windowName,cv2.WINDOW_NORMAL)
cv2.resizeWindow(windowName, 600,600)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Color check.')
parser.add_argument('--image', required=False,metavar="path or URL to image")
parser.add_argument('--k', required=False,metavar="channel to show")
parser.add_argument('--color', required=False,metavar="channel to show")
args = parser.parse_args()

line1 = None

refPts = []
skin_chrs = []
def click_and_crop(event, x, y, flags, param):
  global refPts, clone
  if event == cv2.EVENT_LBUTTONDOWN:
    refPts = [(x, y)]
  elif event == cv2.EVENT_LBUTTONUP:
    refPts.append((x, y))
    sbgrArr = clone[refPts[0][1]:refPts[1][1]+1, refPts[0][0]:refPts[1][0]+1]
    srgbArr = cv2.cvtColor(sbgrArr, cv2.COLOR_BGR2RGB)
    a = np.mean(cv2.cvtColor(sbgrArr, cv2.COLOR_BGR2LAB), axis=(0,1))
    a = np.array([[[a[0]*(100.0/255.0), a[1]-128, a[2]-128]]]).astype(np.float32)
    mean_color = (cv2.cvtColor(a, cv2.COLOR_LAB2BGR)*255.0).astype(int)
    print ("Mean color: ", ImageUtils.RGB2HEX(mean_color[0][0][::-1]))
    #plot_SD(srgb)

		# draw a rectangle around the region of interest
    cv2.rectangle(image, refPts[0], refPts[1], (0, 255, 0), 2)

"""
plot_SD plots spectral distribution of given srgb color.
"""
def plot_SD(rgb):
  x = [i for i in range(380, 731, 10)]
  rho  = ImageUtils.sRGB_to_SD(rgb)
  plt.plot(x, rho, label=ImageUtils.RGB2HEX(rgb), color=ImageUtils.RGB2HEX(rgb), linewidth=2)
  plt.draw()
  plt.legend()
  plt.show(block=False)

"""
set_saturation updates saturation of image by dcent percent.
"""
def set_saturation(img, dcent):
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  hsv[:,:, 1] = np.clip(hsv[:, :, 1] + dcent*(255.0/100.0), 0, 255).astype(np.uint8)
  return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

"""
pixels_diff calculates the difference in pixels
of given mask between given two RGB(n, 3) pixel numpy arrays.
"""
def pixels_diff(oarr, narr):
  dList = ImageUtils.delta_e_cie2000_vectors(oarr, narr)
  print ("\tdone computing pixels_diff with mean: ", np.mean(np.array(dList)))
  plt.hist(dList, bins=50)
  plt.xlim([0,10])
  plt.show(block=False)


image = cv2.imread(args.image)
clone = image.copy()

f = Face(args.image, hdf5FileName=os.path.splitext(os.path.split(args.image)[1])[0] + ".hdf5")
# This is to ascertain brightness of foreground and background masks.
ImageUtils.plot_histogram(f.image, f.rFaceMask, False,100)
ImageUtils.plot_histogram(f.image, f.bgMask, False,100)

cv2.setMouseCallback(windowName, click_and_crop)

while True:
	# display the image and wait for a keypress
  cv2.imshow(windowName, image)
  key = cv2.waitKey(0) & 0xFF

	# if the 'r' key is pressed, reset the cropping region
  if key == ord("r"):
    image = clone.copy()
  elif key == ord("c"):
    break

cv2.destroyAllWindows()
