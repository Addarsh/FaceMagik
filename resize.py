import os
import argparse
from PIL import Image

max_dim = 1600

def resize(imPath):
  img = Image.open(imPath)
  if max(img.size[0], img.size[1]) < max_dim:
    return

  mydir, fn = os.path.split(imPath)
  fname, ext = os.path.splitext(fn)
  fname += "_resized"


  oPath = os.path.join(mydir, fname + ext)
  if max(img.size[0], img.size[1]) == max_dim:
    img.save(os.path.join(mydir, f))
    return
  width, height = 0, 0
  if img.size[0] > img.size[1]:
    width = max_dim
    perc = (max_dim/float(img.size[0]))
    height = int((float(img.size[1])*float(perc)))
  else:
    height = max_dim
    perc = (max_dim/float(img.size[1]))
    width = int((float(img.size[0])*float(perc)))
  img = img.resize((width,height), Image.ANTIALIAS)
  img.save(oPath)

if __name__ == "__main__":
  # Parse command line arguments
  parser = argparse.ArgumentParser(
      description='Train Mask R-CNN to detect balloons.')
  parser.add_argument('--image', required=True,
                      metavar="path to test image file",
                      help="path to test image file",)
  args = parser.parse_args()

  resize(args.image)
