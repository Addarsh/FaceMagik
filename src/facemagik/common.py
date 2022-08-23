"""
Constants and configurations used by both training and inference scripts.
"""
from mrcnn.config import Config
from enum import Enum
from dataclasses import dataclass
import numpy as np

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
    NOSE: 7, NOSTRIL: 8, UPPER_LIP: 9, LOWER_LIP: 10, TEETH: 11, TONGUE: 12, FACIAL_HAIR: 13,
    FACE: 14, HAIR_ON_HEAD: 15, BALD_HEAD: 16, EAR: 17
}

"""
Enum to describe scene brightness level.
"""


class SceneBrightness(Enum):
    DARK_SHADOW = 1
    SOFT_SHADOW = 2
    NEUTRAL_LIGHTING = 3
    TOO_BRIGHT = 4


"""
Direction of mask relative to nose center point of the face.
"""


class MaskDirection(Enum):
    LEFT = 1
    CENTER = 2
    RIGHT = 3


"""
Direction of light falling on the face.
"""


class LightDirection(Enum):
    CENTER = 1  # Light is either exactly facing or exactly opposite the person.
    CENTER_LEFT = 2  # Largely facing the user but also drifts to the left with maybe a slight shadow.
    CENTER_RIGHT = 3  # Largely facing the user but also drifts to the right with maybe a slight shadow.
    LEFT_CENTER_RIGHT = 4  # Center dominates but there is a bright region on the left and some shadow on the right.
    RIGHT_CENTER_LEFT = 5  # Center dominates but there is a bright region on the right and some shadow on the left.
    LEFT_CENTER = 6  # Light starts from the left and then falls to center. May nor may not be a shadow at center.
    LEFT_TO_RIGHT = 7  # Usually indicates there is a shadow in the scene.
    RIGHT_CENTER = 8  # Light starts from the right and then falls to center. May nor may not be a shadow at center.
    RIGHT_TO_LEFT = 9  # Usually indicates there is a shadow in the scene.


"""
Container class for skin tone.
"""


@dataclass
class SkinTone:
    DISPLAY_P3 = "displayP3"

    rgb: []
    hsv: []
    hls: []
    gray: float
    ycrcb: []
    percent_of_face_mask: float
    face_mask: np.ndarray
    profile: str


"""
Base Configuration for Mask RCNN model inference.
"""


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


"""
Inference configuration for Mask RCNN Model.
"""


class InferenceConfig(FaceConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
