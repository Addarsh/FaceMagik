from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.http.multipartparser import MultiPartParser
from PIL import Image
import io
import numpy as np
import base64
import json
import cv2
import os

DATA_DIR = "data/new"
VIDEO_BASE_DIR = "video/"

@csrf_exempt
def index(request):
  if request.method == "POST":
    data, files = MultiPartParser(request.META, io.BytesIO(request.body),
                        request.upload_handlers, "utf-8").parse()

    with open(os.path.join(DATA_DIR, "face_vertices.json"), "w") as f:
      json.dump(json.loads(data["vertices"]), f)
    with open(os.path.join(DATA_DIR, "lighting.json"), "w") as f:
      json.dump(json.loads(data["lighting"]), f)
    with open(os.path.join(DATA_DIR, "triangle_indices.json"), "w") as f:
      json.dump(json.loads(data["triangleIndices"]), f)
    with open(os.path.join(DATA_DIR, "vertex_normals.json"), "w") as f:
      json.dump(json.loads(data["vertexNormals"]), f)


    # Save image to disk.
    image = Image.open(io.BytesIO(base64.b64decode(data["fileset"])))
    image = image.transpose(Image.ROTATE_270)
    image = np.asarray(image, dtype="uint8")
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(DATA_DIR, "face.png"), image)

    return JsonResponse({"message": "Upload complete."}, status=200)

"""
find_next_dir returns name of next directory name to create for given prefix
in given parent directory.
"""
def find_next_dir(parent, prefix):
  dirCount = 0
  dirName = ""
  for d in os.listdir(parent):
    if not os.path.isdir(os.path.join(parent, d)):
      continue
    dirCount += 1

  dirName = os.path.join(parent, prefix + str(dirCount))
  os.makedirs(dirName)
  return dirName

@csrf_exempt
def video(request):
  if request.method == "POST":
    data, files = MultiPartParser(request.META, io.BytesIO(request.body),
                        request.upload_handlers, "utf-8").parse()
    # Find video directory.
    dirName = json.loads(data["dirName"])[0]
    if  dirName == "":
      dirName = find_next_dir(VIDEO_BASE_DIR, "v")
    frameName = find_next_dir(dirName, "f")
    # Save information to JSON files.
    with open(os.path.join(frameName, "face_vertices.json"), "w") as f:
      json.dump(json.loads(data["vertices"]), f)
    with open(os.path.join(frameName, "lighting.json"), "w") as f:
      json.dump(json.loads(data["lighting"]), f)
    with open(os.path.join(frameName, "triangle_indices.json"), "w") as f:
      json.dump(json.loads(data["triangleIndices"]), f)
    with open(os.path.join(frameName, "vertex_normals.json"), "w") as f:
      json.dump(json.loads(data["vertexNormals"]), f)

    # Save image to disk.
    image = Image.open(io.BytesIO(base64.b64decode(data["fileset"])))
    image = image.transpose(Image.ROTATE_270)
    image = np.asarray(image, dtype="uint8")
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(frameName, os.path.split(frameName)[1] + ".png"), image)

    return JsonResponse({"message": "Upload complete.", "dirName": dirName}, status=200)
