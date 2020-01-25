from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.http.multipartparser import MultiPartParser
from PIL import Image
import io
import base64

@csrf_exempt
def index(request):
  if request.method == "POST":
    data, files = MultiPartParser(request.META, io.BytesIO(request.body),
                        request.upload_handlers, "utf-8").parse()
    with open("Output.txt", "w") as text_file:
      print(data["fileset"], file=text_file)

    image = Image.open(io.BytesIO(base64.b64decode(data["fileset"])))
    image = image.transpose(Image.ROTATE_270)

    image.save("test.png", "PNG")
    return JsonResponse({"message": "Upload complete."}, status=200)
