"""
Constants and configurations used by both training and inference scripts.
"""
from mrcnn.config import Config

# Label constants.
EYE_OPEN = "Eye (Open)"
EYEBALL = "Eyeball"
EYEBROW = "Eyebrow"
READING_GLASSES = "Reading glasses"
SUNGLASSES = "Sunglasses"
EYE_CLOSED = "Eye (Closed)"
NOSE = "Nose"
NOSTRIL = "Nostril"
UPPER_LIP = "Upper Lip"
LOWER_LIP = "Lower Lip"
TEETH = "Teeth"
TONGUE = "Tongue"
FACIAL_HAIR = "Facial Hair"
FACE = "Face"
HAIR_ON_HEAD = "Hair (on head)"
BALD_HEAD = "Bald Head"
EAR = "Ear"

# Map from label to class ID.
label_id_map = {
  EYE_OPEN: 1, EYEBALL: 2, EYEBROW: 3, READING_GLASSES: 4, SUNGLASSES: 5, EYE_CLOSED: 6,
  NOSE: 7, NOSTRIL: 8, UPPER_LIP:9, LOWER_LIP:10, TEETH:11, TONGUE: 12, FACIAL_HAIR:13,
  FACE: 14, HAIR_ON_HEAD: 15, BALD_HEAD: 16, EAR: 17
}


class FaceConfig(Config):
  """
  Derives from the base Config class and overrides some values.
  """
  # Give the configuration a recognizable name
  NAME = "face"

  # We use a GPU with 12GB memory, which can fit two images.
  # Adjust down if you use a smaller GPU.
  GPU_COUNT = 4
  IMAGES_PER_GPU = 1

  # Number of classes (including background)
  NUM_CLASSES = 17 + 1  # Background + classes

  # Number of training steps per epoch
  STEPS_PER_EPOCH = 100

  # Skip detections with < 90% confidence
  DETECTION_MIN_CONFIDENCE = 0.9

  # Remove mini mask to improve accuracy.
  USE_MINI_MASK = False

  # Use Resnet50 for faster training.
  BACKBONE = "resnet50"