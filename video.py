import os
import cv2
import argparse
import numpy as np

from PIL import Image
from face import Face

# process analyzes video file
def process(videoPath):
  cap = cv2.VideoCapture(videoPath)
  dir, f = os.path.split(videoPath)
  imagePathPrefix = os.path.splitext(f)[0]

  if not cap.isOpened():
    raise Exception("Error opening video file")

  count = 0
  div = 100
  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      print("No more frames left")
      break
      
    if count % div != 0:
      count += 1
      continue
    print ("Frame number: ", count)

    frame = rotate_frame(frame)
    dirPath = os.path.join(os.path.join(dir, "output"), imagePathPrefix)
    imagePath = os.path.join(dirPath, "frame_" + str(count) + ".png")
    try:
      f = Face(imagePath, frame, dirPath)
      key = f.chakra_median_brightness()
    except Exception as e:
      print (e)
      count += 1
      continue

    cv2.destroyAllWindows()
    count += 1

    if key == ord("c"):
      break

"""
rotate_frame rotates frame 90 degrees clockwise to
correc the 90 degrees CCW ios rotation.
"""
def rotate_frame(frame):
  pImage = Image.fromarray(frame)
  pImage = pImage.transpose(Image.ROTATE_270)
  return np.array(pImage)

if __name__ == "__main__":
  # Parse command line arguments
  parser = argparse.ArgumentParser(description='Video file path')
  parser.add_argument('--file', required=False,metavar="path to video file")
  args = parser.parse_args()

  process(args.file)
