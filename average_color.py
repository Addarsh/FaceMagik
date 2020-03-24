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

    #mean_color = np.sqrt(np.column_stack((np.mean(a[:, :, 0]**2), np.mean(a[:, :, 1]**2), np.mean(a[:, :, 2]**2)))).astype(int)
    srgb = mean_color[0][0][::-1]
    print ("Mean color: ", srgb)
    plot_SD(srgb)
    #compare_skin_spectra(np.reshape(srgbArr, (-1, 3)))

		# draw a rectangle around the region of interest
    cv2.rectangle(image, refPts[0], refPts[1], (0, 255, 0), 2)

def show_channel(image, k=-1):
  if k == 0:
    image[:, :, 1] = 0
    image[:, :, 2] = 0
  elif k == 1:
    image[:, :, 0] = 0
    image[:, :, 2] = 0
  elif k == 2:
    image[:, :, 0] = 0
    image[:, :, 1] = 0
  return image

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
compare_skin_spectra converts given srgb to uv chromaticity with known
skin uv chromaticity.
"""
def compare_skin_spectra(srgbList):
  uvlist = []
  for srgb in srgbList:
    uvlist.append(ImageUtils.sRGB_to_uv(srgb))

  global skin_chrs
  if len(skin_chrs) == 0:
    with open("skin_uv.json", "r") as f:
      skin_chrs = json.load(f)
    skin_chrs = np.array(skin_chrs)

    plt.scatter(*zip(*skin_chrs))

  plt.scatter(*zip(*uvlist), c="red")
  plt.draw()
  plt.show(block=False)

def plot_histogram(img, mask):
  median = np.median(img[mask])
  print ("Mean: ", np.mean(img[mask]), " median: ", median," Std: ", np.std(img[mask]))
  hist = cv2.calcHist([img],[0], mask.astype(np.uint8)*255,[256],[0,256])
  plt.plot(hist)
  plt.xlim([0,256])
  #plt.ylim([0, 10000])
  plt.show(block=False)
  return median

# set_brightness will set brightness
# value to b for every facemask, pixel in the given image.
def set_brightness(img, mask, b, k):
  img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  fmask = img[mask]
  fmask[:, k] = np.uint8(b)
  img[mask] = fmask
  return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

image = cv2.imread(args.image)
f = Face(args.image)

clone = image.copy()

if args.k != None:
  k = int(args.k)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  image = show_channel(image, k)
  median = plot_histogram(image[:, :, k], f.faceMask)
if args.color == "True":
  image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  if median != None:
    fmask = image[f.faceMask]
    fmask[:, k] = np.uint8(median)
    image[f.faceMask] = fmask
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    clone = image
elif args.color == "False":
  image = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)

cv2.setMouseCallback(windowName, click_and_crop)

while True:
	# display the image and wait for a keypress
  cv2.imshow(windowName, image)
  key = cv2.waitKey(1) & 0xFF

	# if the 'r' key is pressed, reset the cropping region
  if key == ord("r"):
    image = clone.copy()
  elif key == ord("c"):
    break

cv2.destroyAllWindows()
