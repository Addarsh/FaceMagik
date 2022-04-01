import multiprocessing.sharedctypes
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing as mp

from mrcnn import model as model_lib
from dataclasses import dataclass
from multiprocessing import Queue, Pool
from .utils import ImageUtils
from .face import Face
from .common import InferenceConfig, SceneBrightness, LightDirection

"""
Configuration details associated with skin detection algorithm.
"""


class SkinDetectionConfig:
    # Absolute path where image to be processed in located. Leave empty if passing Image directly in memory.
    IMAGE_PATH: str = ""

    # Numpy array of image that needs to be processed. Leave as None if fetching image from local absolute path.
    IMAGE: np.ndarray = None

    # If true, use new clustering algorithm else use older algorithm.
    USE_NEW_CLUSTERING_ALGORITHM: bool = False

    # Minimum Kmeans difference value until repeated Kmeans clustering is performed.
    KMEANS_TOLERANCE: float = 2.0

    # Minimum mask percent until which repeated Kmeans clustering is performed on teeth mask.
    KMEANS_TEETH_MASK_PERCENT_CUTOFF: float = 5.0

    # Minimum mask percent until which repeated Kmeans clustering is performed on face mask.
    KMEANS_FACE_MASK_PERCENT_CUTOFF: float = 2.0

    # If true, iterate teeth clusters for fine-grained results. Defaults to false.
    ITERATE_TEETH_CLUSTERS = False

    # If true, iterate face clusters for fine-grained results. Defaults to true.
    ITERATE_FACE_CLUSTERS = False

    # Factor to multiple image's per pixel brightness value. Should always be a positive value, defaults to 1.
    BRIGHTNESS_UPDATE_FACTOR: float = 1.0

    # Factor to multiple image's per pixel saturation value. Should always be a positive value, defaults to 1.
    SATURATION_UPDATE_FACTOR: float = 1.0

    # If true, combine effective color masks based on delta cie 2000 closeness value.
    COMBINE_MASKS: bool = False

    # If true, runs analysis in debug mode. Used during development.
    DEBUG_MODE: bool = False

    def __init__(self):
        pass

    def __repr__(self):
        return "SkinDetectionConfig(IMAGE_PATH: {0})".format(self.IMAGE_PATH)


class TeethNotVisibleException(ValueError):
    pass


"""
Container class for computed scene brightness and light direction.
"""


@dataclass
class SceneBrightnessAndDirection:
    scene_brightness_value: int
    primary_light_direction: LightDirection
    # Percentage of face mask that is in given mask direction (LEFT, CENTER or RIGHT).
    percent_per_direction: dict

    """
    Returns scene brightness enum based on brightness value.
    """

    def scene_brightness(self) -> SceneBrightness:
        if self.scene_brightness_value < 200:
            return SceneBrightness.DARK_SHADOW
        elif self.scene_brightness_value < 220:
            return SceneBrightness.SOFT_SHADOW
        elif self.scene_brightness_value > 225:
            return SceneBrightness.TOO_BRIGHT

        return SceneBrightness.NEUTRAL_LIGHTING


"""
Container class for skin tone.
"""


@dataclass
class SkinTone:
    DISPLAY_P3 = "displayP3"

    rgb: []
    percent_of_face_mask: int
    profile: str

    def brightness(self):
        return max(self.rgb)


"""
Class to analyze skin tone from a image of a face.
"""


class SkinToneAnalyzer:
    # Effective colors.
    pink_red = "PinkRed"
    maroon = "Maroon"
    orange = "Orange"
    orange_yellow = "OrangeYellow"
    yellow_green = "YellowishGreen"
    middle_green = "MiddleGreen"
    green_yellow = "GreenishYellow"
    light_green = "LightGreen"
    green = "Green"
    blue_green = "BluishGreen"
    green_blue = "GreenishBlue"
    blue = "Blue"
    none = "None"

    def __init__(self, maskrcnn_model, skin_config: object):

        # Detect face.
        if skin_config.IMAGE_PATH != "":
            face = Face(image_path=skin_config.IMAGE_PATH, maskrcnn_model=maskrcnn_model)
        else:
            face = Face(image=skin_config.IMAGE, maskrcnn_model=maskrcnn_model)

        if skin_config.BRIGHTNESS_UPDATE_FACTOR != 1.0 or skin_config.SATURATION_UPDATE_FACTOR != 1.0:
            new_img = ImageUtils.set_brightness(face.image, skin_config.BRIGHTNESS_UPDATE_FACTOR)
            new_img = ImageUtils.set_saturation(new_img, skin_config.SATURATION_UPDATE_FACTOR)
            face.image = new_img

        self.face = face
        self.image = self.face.image
        self.face_mask_to_process = self.face.get_face_until_nose_end_without_area_around_eyes()
        self.mouth_mask_to_process = self.face.get_mouth_points()
        self.face_mask_effective_color_map = {}
        self.skin_config = skin_config

    """
    Plots a figure with each cluster's color and Munsell value. Primary use for analysis of similar colors.
    """

    @staticmethod
    def plot_colors(image: np.ndarray, mask_list: [list], total_points: int) -> None:
        if len(mask_list) > 1:
            plt.close()
        fig = plt.figure()
        ax = fig.add_subplot(111)

        x = [i for i in range(len(mask_list))]
        percent_list = [ImageUtils.percentPoints(mask, total_points) for mask in mask_list]
        color_list = [np.mean(image[mask], axis=0) / 255.0 for mask in mask_list]
        munsell_color_list = [ImageUtils.sRGBtoMunsell(np.mean(image[mask], axis=0)) for mask in mask_list]
        ax.bar(x=x, height=percent_list, color=color_list, align="edge")
        # Set munsell color values on top of value.
        for i in range(len(x)):
            plt.text(i, percent_list[i], munsell_color_list[i])
        plt.show(block=False)

        ax.bar(x=x, height=percent_list, color=color_list, align="edge")
        # Set Munsell color values on top of value.
        for i in range(len(x)):
            plt.text(i, percent_list[i], munsell_color_list[i])
        plt.show(block=False)

    """
    The mapping from Munsell Hue to known color is a hueristic obtained by viewing many images and determining which 
    Munsell hues cluster together into a known color like "Orange" or "LightGreen". These clustered colors are then 
    used to ascertain image brightness. This kind of clustering can only work if the Munsell hue determines the final
    color of the pixel to a large extent. It is also likely that there can be better mapping to cluster colors.
    """

    @staticmethod
    def effective_color(munsell_color: str):
        if munsell_color == SkinToneAnalyzer.none:
            return SkinToneAnalyzer.none

        munsell_hue = munsell_color.split(" ")[0]
        hue_letter = ImageUtils.munsell_hue_letter(munsell_hue)
        hue_number = ImageUtils.munsell_hue_number(munsell_hue)

        delta = 0.5

        if hue_letter == "R":
            if hue_number < 7.5 - delta:
                return SkinToneAnalyzer.pink_red
            return SkinToneAnalyzer.maroon
        elif hue_letter == "YR":
            if hue_number < 2.5 - delta:
                return SkinToneAnalyzer.maroon
            elif hue_number < 9 - delta:
                return SkinToneAnalyzer.orange
            return SkinToneAnalyzer.orange_yellow
        elif hue_letter == "Y":
            if hue_number < 2.5 - delta:
                return SkinToneAnalyzer.orange_yellow
            # if hueNumber < 3 - delta:
            #  return SkinToneAnalyzer.orangeYellow
            elif hue_number < 7.5 - delta:
                return SkinToneAnalyzer.yellow_green
            return SkinToneAnalyzer.middle_green
            # if hueNumber < 5 - delta:
            #  return SkinToneAnalyzer.orangeYellow
            return SkinToneAnalyzer.yellow_green
        elif hue_letter == "GY":
            if hue_number < 2.5 - delta:
                return SkinToneAnalyzer.middle_green
                return SkinToneAnalyzer.yellow_green
            elif hue_number < 7.5 - delta:
                return SkinToneAnalyzer.green_yellow
            return SkinToneAnalyzer.light_green
        elif hue_letter == "G":
            if hue_number < 2 - delta:
                return SkinToneAnalyzer.light_green
            elif hue_number < 4.5 - delta:
                return SkinToneAnalyzer.green
            elif hue_number < 9 - delta:
                return SkinToneAnalyzer.green_blue
            return SkinToneAnalyzer.blue_green
        elif hue_letter == "BG":
            if hue_number < 2.5 - delta:
                return SkinToneAnalyzer.blue_green
            return SkinToneAnalyzer.blue

        # Old mapping. Deprecated.
        if hue_letter == "R":
            if hue_number < 7.5 - delta:
                return SkinToneAnalyzer.pink_red
            return SkinToneAnalyzer.maroon
        elif hue_letter == "YR":
            if hue_number < 2.5 - delta:
                return SkinToneAnalyzer.maroon
            return SkinToneAnalyzer.orange
        elif hue_letter == "Y":
            if hue_number < 7.5 - delta:
                return SkinToneAnalyzer.orange_yellow
            return SkinToneAnalyzer.yellow_green
        elif hue_letter == "GY":
            if hue_number < 2.5 - delta:
                return SkinToneAnalyzer.yellow_green
            elif hue_number < 7.5 - delta:
                return SkinToneAnalyzer.green_yellow
            return SkinToneAnalyzer.light_green
        elif hue_letter == "G":
            if hue_number < 2 - delta:
                return SkinToneAnalyzer.light_green
            elif hue_number < 4.5 - delta:
                return SkinToneAnalyzer.green
            elif hue_number < 9 - delta:
                return SkinToneAnalyzer.green_blue
            return SkinToneAnalyzer.blue_green
        elif hue_letter == "BG":
            return SkinToneAnalyzer.blue_green

    """
    Constructs a MaskRCNN model and returns it.
    """

    @staticmethod
    def construct_model(weights_relative_path):
        start_time = time.time()

        # Create model
        model = model_lib.MaskRCNN(mode="inference", config=InferenceConfig(), model_dir="")

        # Select weights file to load
        try:
            weights_path = os.path.join(os.getcwd(), weights_relative_path)
        except Exception as e:
            raise

        print("Loading weights from: ", weights_path)
        model.load_weights(weights_path, by_name=True)

        print("\nModel construction time: ", time.time() - start_time, " seconds\n")
        return model

    """
    Repeatedly divides mask into clusters using kmeans until difference between
    clusters is less than given tolerance. Returns the cluster with the largest
    diff value. diffImg has dimensions (W, H) and contains the values to peform
    clustering on. mask is boolean mask of same dimensions,
    """

    @staticmethod
    def brightest_cluster(diff_img: np.ndarray, mask: object, total_points: int, tol: int = 2,
                          cutoff_percent: int = 2) -> np.ndarray:
        c1_tuple, c2_tuple = ImageUtils.Kmeans_1d(diff_img, mask)
        c1_mask, centroid1 = c1_tuple
        c2_mask, centroid2 = c2_tuple

        if ImageUtils.percentPoints(c1_mask, total_points) < cutoff_percent or ImageUtils.percentPoints(c2_mask,
                                                                                                        total_points) \
                < cutoff_percent:
            # end cluster division.
            return mask
        if abs(centroid1 - centroid2) <= tol:
            # end cluster division.
            return mask
        return SkinToneAnalyzer.brightest_cluster(diff_img, c1_mask, total_points, tol, cutoff_percent)

    """
    Break given mask into smaller clusters for given YCrCb image. The Y value (brightness) of the image is used to
    perform Kmeans clustering to separate the mask into clusters. Additionally, these clusters are further combined into 
    a effective color map and returned along with the clusters.
    """

    @staticmethod
    def make_clusters(ycrcb_image: np.ndarray, mask_to_process: np.ndarray, kmeans_tolerance: float, cutoff_percent:
    float, debug_mode: bool) \
            -> [int, dict]:
        start_time = time.time()
        diff_img = (ycrcb_image[:, :, 0]).astype(float)
        curr_mask = mask_to_process.copy()
        total_points = np.count_nonzero(mask_to_process)

        all_cluster_masks = []
        effective_color_map = {}

        # Divide the image into smaller clusters.
        while True:
            # Find the brightest cluster.
            b_mask = SkinToneAnalyzer.brightest_cluster(diff_img, curr_mask, total_points,
                                                        tol=kmeans_tolerance,
                                                        cutoff_percent=cutoff_percent)
            # Find the least saturated cluster of the brightest cluster. This provides more fine-grained clusters
            # but is also more expensive. Comment it out if you want to plot "color of each cluster versus
            # the associated Munsell hue" to iterate/improve effective color mapping.
            # b_mask = SkinToneAnalyzer.brightest_cluster(255.0 -(ImageUtils.to_hsv(ycrcb_image)[:, :, 1]).astype(
            #    np.float), b_mask, np.count_nonzero(b_mask), tol=skin_config.KMEANS_TOLERANCE,
            #                                            cutoff_percent=skin_config.KMEANS_MASK_PERCENT_CUTOFF)

            munsell_color = ImageUtils.sRGBtoMunsell(np.mean(ycrcb_image[b_mask], axis=0))
            effective_color = SkinToneAnalyzer.effective_color(munsell_color)
            if effective_color not in effective_color_map:
                effective_color_map[effective_color] = b_mask
            else:
                effective_color_map[effective_color] = np.bitwise_or(effective_color_map[effective_color], b_mask)

            # Store this mask for different computations.
            all_cluster_masks.append(b_mask)

            if debug_mode:
                print("effective color: ", effective_color, " brightness: ",
                      round(np.mean(ycrcb_image[:, :, 0][b_mask]),
                            2), "\n")
                # f.show(ImageUtils.plot_points_and_mask(ycrcb_image, [f.noseMiddlePoint], bMask))

            curr_mask = np.bitwise_xor(curr_mask, b_mask)
            if ImageUtils.percentPoints(curr_mask, total_points) < 1:
                break

        print("\nClustering latency: ", time.time() - start_time, " seconds\n")

        return all_cluster_masks, effective_color_map

    """
    Break given mask into smaller clusters for given YCrCb image. The Y value (brightness) of the image is used to
    create these clusters. Additionally, these clusters are further combined into a effective color map and returned 
    along with the clusters.
    """

    @staticmethod
    def __make_new_clusters(ycrcb_image: np.ndarray, mask_to_process: np.ndarray) -> (list, dict):
        start_time = time.time()
        diff_img = (ycrcb_image[:, :, 0]).astype(float)

        # Break mask into smaller clusters.
        max_brightness = int(np.max(diff_img[mask_to_process]))
        min_brightness = int(np.min(diff_img[mask_to_process]))
        mask_clusters = []
        curr_mask = np.zeros(diff_img.shape, dtype=bool)
        curr_max_brightness = max_brightness
        for brightness in range(max_brightness, min_brightness - 1, -1):
            curr_mask = np.bitwise_or(curr_mask, np.bitwise_and(diff_img == brightness, mask_to_process))
            if curr_max_brightness - brightness >= 5:
                mask_clusters.append(curr_mask)
                curr_mask = np.zeros(diff_img.shape, dtype=bool)
                curr_max_brightness = brightness - 1

        if np.count_nonzero(curr_mask) > 0:
            mask_clusters.append(curr_mask)

        # Compute effective color of each cluster mask and group them.
        with Pool(processes=mp.cpu_count()) as pool:
            results = [pool.apply_async(ImageUtils.sRGBtoMunsell, (np.mean(ycrcb_image[m], axis=0),)) for m in
                       mask_clusters]
            munsell_color_list = [result.get() for result in results]
            effective_color_list = [SkinToneAnalyzer.effective_color(munsell_color) for munsell_color in
                                    munsell_color_list]

        effective_color_map = {}
        for effective_color, mask in zip(effective_color_list, mask_clusters):
            if effective_color not in effective_color_map:
                effective_color_map[effective_color] = mask
            else:
                effective_color_map[effective_color] = np.bitwise_or(effective_color_map[effective_color], mask)

        print("New Clustering latency: ", time.time() - start_time)

        return mask_clusters, effective_color_map

    """
    Static method that computes brightness of scene. Used in parallel execution.
    """

    @staticmethod
    def get_brightness(rgb_image: np.ndarray, ycrcb_image: np.ndarray, mask_to_process: np.ndarray, skin_config:
    SkinDetectionConfig, brightness_queue: multiprocessing.Queue):
        total_points = np.count_nonzero(mask_to_process)

        # Make clusters.
        if skin_config.USE_NEW_CLUSTERING_ALGORITHM:
            all_cluster_masks, effective_color_map = SkinToneAnalyzer.__make_new_clusters(ycrcb_image, mask_to_process)
            # Filter masks that are larger than 5% in size.
            effective_color_map = dict(filter(lambda elem: ImageUtils.percentPoints(elem[1], total_points) >= 5,
                                              effective_color_map.items()))
        else:
            all_cluster_masks, effective_color_map = SkinToneAnalyzer.make_clusters(ycrcb_image, mask_to_process,
                                                                                    skin_config.KMEANS_TOLERANCE,
                                                                                    skin_config.KMEANS_TEETH_MASK_PERCENT_CUTOFF,
                                                                                    skin_config.DEBUG_MODE)

        # Check mean brightness minimum coverage of the teeth.
        final_mask = np.zeros(mask_to_process.shape, dtype=bool)

        # Find mean brightness of first two masks in decreasing order of brightness.
        count = 0
        for effective_color in effective_color_map:
            final_mask = np.bitwise_or(final_mask, effective_color_map[effective_color])
            count += 1
            if count == 2:
                break

        mean_brightness = round(np.mean(np.max(rgb_image, axis=2)[final_mask]))
        if skin_config.DEBUG_MODE:
            print("\nMean brightness value: ", mean_brightness, " with percent: ", ImageUtils.percentPoints(
                final_mask, total_points), "\n")

        if brightness_queue is not None:
            brightness_queue.put(mean_brightness)

        return mean_brightness

    """
    Computes average brightness of the scene. The person in the image is expected to be smiling with teeth 
    visible else an exception is thrown.
    """

    def determine_brightness(self) -> int:
        if not self.face.is_teeth_visible():
            raise TeethNotVisibleException("Teeth not visible for config: " +
                                           str(self.skin_config))

        mask_to_process = self.mouth_mask_to_process
        total_points = np.count_nonzero(mask_to_process)
        ycrcb_image = ImageUtils.to_YCrCb(self.image)

        # Make clusters.
        if self.skin_config.USE_NEW_CLUSTERING_ALGORITHM:
            all_cluster_masks, effective_color_map = SkinToneAnalyzer.__make_new_clusters(ycrcb_image, mask_to_process)
            # Filter masks that are larger than 5% in size.
            effective_color_map = dict(filter(lambda elem: ImageUtils.percentPoints(elem[1], total_points) >= 5,
                                              effective_color_map.items()))
        else:
            all_cluster_masks, effective_color_map = SkinToneAnalyzer.make_clusters(ycrcb_image, mask_to_process,
                                                                                    self.skin_config.KMEANS_TOLERANCE,
                                                                                    self.skin_config.KMEANS_TEETH_MASK_PERCENT_CUTOFF,
                                                                                    self.skin_config.DEBUG_MODE)

        if self.skin_config.ITERATE_TEETH_CLUSTERS:
            # Iterate to optimize final clusters.
            effective_color_map = Face.iterate_effective_color_map(self.image, effective_color_map,
                                                                   all_cluster_masks)

        if self.skin_config.DEBUG_MODE:
            Face.print_effective_color_map(self.image, effective_color_map, total_points)

        # Check mean brightness minimum coverage of the teeth.
        final_mask = np.zeros(self.image.shape[:2], dtype=bool)

        # Find mean brightness of first two masks in decreasing order of brightness.
        count = 0
        for effective_color in effective_color_map:
            final_mask = np.bitwise_or(final_mask, effective_color_map[effective_color])
            count += 1
            if count == 2:
                break

        mean_brightness = round(np.mean(np.max(self.image, axis=2)[final_mask]))
        if self.skin_config.DEBUG_MODE:
            print("\nMean brightness value: ", mean_brightness, " with percent: ", ImageUtils.percentPoints(final_mask,
                                                                                                            total_points
                                                                                                            ), "\n")
        if self.skin_config.DEBUG_MODE:
            ImageUtils.show(self.image)

        return mean_brightness

    """
    Static method that returns Primary light direction. Used for parallel execution.
    """

    @staticmethod
    def get_primary_light_direction(ycrcb_image: np.ndarray, mask_to_process: np.ndarray,
                                    nose_middle_point: np.ndarray, rotation_matrix: np.ndarray, skin_config:
            SkinDetectionConfig, light_direction_queue: multiprocessing.Queue) -> LightDirection:
        total_points = np.count_nonzero(mask_to_process)

        # Make clusters.
        if skin_config.USE_NEW_CLUSTERING_ALGORITHM:
            all_cluster_masks, effective_color_map = SkinToneAnalyzer.__make_new_clusters(ycrcb_image, mask_to_process)
        else:
            all_cluster_masks, effective_color_map = SkinToneAnalyzer.make_clusters(ycrcb_image, mask_to_process,
                                                                                    skin_config.KMEANS_TOLERANCE,
                                                                                    skin_config.KMEANS_FACE_MASK_PERCENT_CUTOFF,
                                                                                    skin_config.DEBUG_MODE)

        # Get light direction from face mask clusters.
        mask_directions_list = [ImageUtils.get_mask_direction(b_mask, nose_middle_point, rotation_matrix,
                                                              skin_config.DEBUG_MODE)
                                for
                                b_mask in
                                all_cluster_masks]
        mask_percent_list = [ImageUtils.percentPoints(b_mask, total_points) for b_mask in all_cluster_masks]

        primary_light_direction, percent_per_direction = Face.process_mask_directions(mask_directions_list,
                                                                                      mask_percent_list)
        if light_direction_queue is not None:
            light_direction_queue.put((primary_light_direction, percent_per_direction))

        return primary_light_direction, percent_per_direction, effective_color_map

    """
    Instance method that returns primary light direction. Used for sequential execution.
    """

    def get_light_direction(self) -> LightDirection:
        ycrcb_image = ImageUtils.to_YCrCb(self.image)
        mask_to_process = self.face_mask_to_process
        node_middle_point = self.face.noseMiddlePoint
        rotation_matrix = self.face.rotMatrix

        return SkinToneAnalyzer.get_primary_light_direction(ycrcb_image, mask_to_process, node_middle_point,
                                                            rotation_matrix, self.skin_config,
                                                            None)

    """
    Computes Scene Brightness and Primary Light Direction. Primary light direction is executed in a separate process 
    to parallelize compute.
    """

    def get_scene_brightness_and_primary_light_direction(self) -> SceneBrightnessAndDirection:
        start_time = time.time()
        self.skin_config.DEBUG_MODE = False

        ycrcb_image = ImageUtils.to_YCrCb(self.image)
        mask_to_process = self.face_mask_to_process
        node_middle_point = self.face.noseMiddlePoint
        rotation_matrix = self.face.rotMatrix
        light_direction_queue = Queue()
        p = mp.Process(target=SkinToneAnalyzer.get_primary_light_direction, args=(ycrcb_image, mask_to_process,
                                                                                  node_middle_point, rotation_matrix,
                                                                                  self.skin_config,
                                                                                  light_direction_queue))
        p.start()

        scene_brightness_value = self.determine_brightness()

        p.join()

        primary_light_direction, percent_per_direction = light_direction_queue.get()

        print("Scene brightness and primary light direction detection latency: ", time.time() - start_time)

        return SceneBrightnessAndDirection(scene_brightness_value, primary_light_direction, percent_per_direction)

    """
    Computes Scene Brightness and Primary Light Direction. Scene brightness is executed in a separate process to 
    parallelize compute.
    """

    def get_primary_light_direction_and_scene_brightness(self) -> SceneBrightnessAndDirection:
        if not self.face.is_teeth_visible():
            raise TeethNotVisibleException("Teeth not visible for config: " +
                                           str(self.skin_config))

        start_time = time.time()
        self.skin_config.DEBUG_MODE = False

        rgb_image = self.image
        ycrcb_image = ImageUtils.to_YCrCb(self.image)
        mask_to_process = self.mouth_mask_to_process
        brightness_queue = Queue()
        p = mp.Process(target=SkinToneAnalyzer.get_brightness, args=(rgb_image, ycrcb_image, mask_to_process,
                                                                     self.skin_config, brightness_queue))
        p.start()

        primary_light_direction, percent_per_direction, effective_color_map = self.get_light_direction()

        p.join()

        # Store effective color map of face mask.
        self.face_mask_effective_color_map = effective_color_map

        scene_brightness_value = brightness_queue.get()

        print("Primary light direction detection and Scene brightness latency: ", time.time() - start_time)

        return SceneBrightnessAndDirection(scene_brightness_value, primary_light_direction, percent_per_direction)

    """
    Computes skin tones associated with the face.
    """

    def get_skin_tones(self):
        total_points = np.count_nonzero(self.face_mask_to_process)
        ycrcb_image = ImageUtils.to_YCrCb(self.image)

        if len(self.face_mask_effective_color_map) == 0:
            _, effective_color_map = SkinToneAnalyzer.__make_new_clusters(ycrcb_image, self.face_mask_to_process)
        else:
            effective_color_map = self.face_mask_effective_color_map

        return SkinToneAnalyzer.__get_skin_tones(self.image, effective_color_map, total_points)

    """
    Detects skin tone and primary light direction for given face image. Only used for debugging purposes. Do nto use 
    in production code.
    """

    def detect_skin_tone_and_light_direction(self) -> list:
        total_points = np.count_nonzero(self.face_mask_to_process)
        ycrcb_image = ImageUtils.to_YCrCb(self.image)

        # Make clusters.
        if self.skin_config.USE_NEW_CLUSTERING_ALGORITHM:
            all_cluster_masks, effective_color_map = SkinToneAnalyzer.__make_new_clusters(ycrcb_image,
                                                                                          self.face_mask_to_process)
        else:
            all_cluster_masks, effective_color_map = SkinToneAnalyzer.make_clusters(ycrcb_image,
                                                                                    self.face_mask_to_process,
                                                                                    self.skin_config.KMEANS_TOLERANCE,
                                                                                    self.skin_config.KMEANS_FACE_MASK_PERCENT_CUTOFF,
                                                                                    self.skin_config.DEBUG_MODE)

        # Get light direction from face mask clusters.
        mask_directions_list = [ImageUtils.get_mask_direction(b_mask, self.face.noseMiddlePoint, self.face.rotMatrix,
                                                              self.skin_config.DEBUG_MODE) for
                                b_mask in
                                all_cluster_masks]
        mask_percent_list = [ImageUtils.percentPoints(b_mask, total_points) for b_mask in all_cluster_masks]

        final_light_direction, percent_per_direction = Face.process_mask_directions(mask_directions_list,
                                                                                    mask_percent_list)
        print("\nFinal Light Direction: ", final_light_direction, " percent per direction: ", percent_per_direction)

        if self.skin_config.DEBUG_MODE:
            SkinToneAnalyzer.plot_colors(ycrcb_image, all_cluster_masks, total_points)

        if self.skin_config.ITERATE_FACE_CLUSTERS:
            # Iterate to optimize final clusters.
            effective_color_map = Face.iterate_effective_color_map(self.image, effective_color_map,
                                                                   all_cluster_masks)

        if self.skin_config.DEBUG_MODE:
            Face.print_effective_color_map(self.image, effective_color_map, total_points)

        if self.skin_config.COMBINE_MASKS:
            combined_masks = Face.combine_masks_close_to_each_other(self.image, effective_color_map)

            if self.skin_config.DEBUG_MODE:
                print("\nCombined masks")
                for m in combined_masks:
                    print("percent: ", ImageUtils.percentPoints(m, total_points))
                    self.face.show_mask(m)

        img = ImageUtils.plot_points_new(self.image, [self.face.noseMiddlePoint])
        ImageUtils.show(img)

        if self.skin_config.DEBUG_MODE:
            ImageUtils.show(self.image)

        return SkinToneAnalyzer.__get_skin_tones(self.image, effective_color_map, total_points)

    """
    Returns skin tones for given effective color map and total face mask points.
    """

    @staticmethod
    def __get_skin_tones(rgb_image, effective_color_map, total_points):
        # Return list of skin tones than are more than 10% in size.
        skin_tones = []
        skin_tone_percent_cutoff = 10
        for mask in effective_color_map.values():
            percent_of_face_mask = ImageUtils.percentPoints(mask, total_points)
            if percent_of_face_mask >= skin_tone_percent_cutoff:
                mean_color = np.round(np.mean(rgb_image[mask], axis=0)).astype(int)
                skin_tone = SkinTone(mean_color.tolist(), round(percent_of_face_mask), SkinTone.DISPLAY_P3)
                skin_tones.append(skin_tone)

        return skin_tones


if __name__ == "__main__":
    # Run this script from parent directory level (face_magik) of this module.
    # Example: python -m facemagik.skintone --image <image_path>
    # Stack Overflow answer: https://stackoverflow.com/questions/47319423

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Image for processing')
    parser.add_argument('--image', required=True, metavar="path to video file")
    parser.add_argument('--bri', required=False, metavar="bri")
    parser.add_argument('--sat', required=False, metavar="sat")
    args = parser.parse_args()

    # Multiprocessing library should use fork mode.
    mp.set_start_method('fork')

    skin_detection_config = SkinDetectionConfig()
    skin_detection_config.IMAGE_PATH = args.image
    skin_detection_config.DEBUG_MODE = True
    skin_detection_config.USE_NEW_CLUSTERING_ALGORITHM = True

    if args.bri is not None:
        skin_detection_config.BRIGHTNESS_UPDATE_FACTOR = float(args.bri)
    if args.sat is not None:
        skin_detection_config.SATURATION_UPDATE_FACTOR = float(args.sat)

    # Load Mask RCNN model. maskrcnn_model directory is located one level above where this script is run.
    maskrcnn_model = SkinToneAnalyzer.construct_model("../maskrcnn_model/mask_rcnn_face_0060.h5")

    analyzer = SkinToneAnalyzer(maskrcnn_model, skin_detection_config)
    #print("Brightness value: ", analyzer.determine_brightness())
    #print ("Primary light direction: ", analyzer.get_light_direction()[:2])
    #print("Scene brightness and light direction: ", analyzer.get_scene_brightness_and_primary_light_direction())
    #print("light direction and scene brightness: ", analyzer.get_primary_light_direction_and_scene_brightness())
    print("Skin Tones: ", analyzer.detect_skin_tone_and_light_direction())
    #print("Skin Tones production: ", analyzer.get_skin_tones())
