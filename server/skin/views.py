from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.http.multipartparser import MultiPartParser
from PIL import Image
import io
import numpy as np
import base64
import json
import cv2

@csrf_exempt
def index(request):
  if request.method == "POST":
    data, files = MultiPartParser(request.META, io.BytesIO(request.body),
                        request.upload_handlers, "utf-8").parse()

    faceVertices = json.loads(data["vertices"])
    lighting = json.loads(data["lighting"])
    normals = json.loads(data["normals"])

    image = Image.open(io.BytesIO(base64.b64decode(data["fileset"])))
    image = image.transpose(Image.ROTATE_270)
    image = np.asarray(image, dtype="uint8")
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for v in faceVertices:
      image[v[0], v[1]] = [0, 255, 0]

    cv2.imwrite("test.png", image)
    return JsonResponse({"message": "Upload complete."}, status=200)
