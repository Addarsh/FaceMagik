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
from .mesh import face_mask_with_direct_light
from .face import Face
from .common import InferenceConfig, SceneBrightness, LightDirection, SkinTone

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

@dataclass
class NoseMiddlePoint:
    x: int
    y: int


@dataclass
class FaceMaskInfo:
    face_mask_to_process: np.ndarray
    mouth_mask_to_process: np.ndarray
    nose_middle_point: NoseMiddlePoint
    left_eye_mask: np.ndarray
    right_eye_mask: np.ndarray

    def get_eye_masks(self):
        return [self.left_eye_mask, self.right_eye_mask]

    def get_nose_middle_point(self):
        return [self.nose_middle_point.y, self.nose_middle_point.x]

    @staticmethod
    def is_teeth_visible():
        return True


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

    def __init__(self, maskrcnn_model, skin_config: object, face_mask_info: FaceMaskInfo = None):
        if face_mask_info is None:
            # Detect face.
            if skin_config.IMAGE_PATH != "":
                face = Face(image_path=skin_config.IMAGE_PATH, maskrcnn_model=maskrcnn_model)
            else:
                face = Face(image=skin_config.IMAGE, maskrcnn_model=maskrcnn_model)

            if skin_config.BRIGHTNESS_UPDATE_FACTOR != 1.0 or skin_config.SATURATION_UPDATE_FACTOR != 1.0:
                new_img = ImageUtils.set_brightness(face.image, skin_config.BRIGHTNESS_UPDATE_FACTOR)
                new_img = ImageUtils.set_saturation(new_img, skin_config.SATURATION_UPDATE_FACTOR)
                face.image = new_img

            self.image = face.image
            self.face_mask_to_process = face.get_face_until_nose_end_without_area_around_eyes()
            #self.face_mask_to_process = face.get_face_mask_without_area_around_eyes()
            try:
                self.mouth_mask_to_process = face.get_mouth_points()
            except Exception:
                self.mouth_mask_to_process = None
            self.nose_middle_point = face.noseMiddlePoint
            self.rotation_matrix = ImageUtils.rotation_matrix(face.get_eye_masks())
            self.is_teeth_visible = face.is_teeth_visible()
            self.face = face
        else:
            self.image = skin_config.IMAGE
            self.face_mask_to_process = face_mask_info.face_mask_to_process
            self.mouth_mask_to_process = face_mask_info.mouth_mask_to_process
            self.nose_middle_point = face_mask_info.get_nose_middle_point()
            self.rotation_matrix = ImageUtils.rotation_matrix(face_mask_info.get_eye_masks())
            self.is_teeth_visible = FaceMaskInfo.is_teeth_visible()

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
        num_processes = min(4, mp.cpu_count())
        with Pool(processes=num_processes) as pool:
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
    Returns true if teeth are visible in image, false otherwise.
    """

    def are_teeth_visible_in_image(self):
        return self.is_teeth_visible

    """
    Computes average brightness of the scene. The person in the image is expected to be smiling with teeth 
    visible else an exception is thrown.
    """

    def determine_scene_brightness(self) -> int:
        if not self.is_teeth_visible:
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

        # Top 30% of brightest teeth pixels.
        top_brightness_mask = np.zeros(self.image.shape[:2], dtype=bool)
        for acm in all_cluster_masks:
            top_brightness_mask = np.bitwise_or(top_brightness_mask, acm)
            if ImageUtils.percentPoints(top_brightness_mask, total_points) >= 20:
                break

        average_teeth_brightness_value = round(np.mean(np.max(self.image, axis=2)[top_brightness_mask]))
        print("Scene brightness value: ", average_teeth_brightness_value)

        if self.skin_config.DEBUG_MODE:
            # Plot histogram.
            ImageUtils.plot_histogram(self.image, top_brightness_mask, channel=0, block=True, bins=50, fig_num=1,
                                      xlim=[0,
                                                                                                                 256])

        if self.skin_config.DEBUG_MODE:
            ImageUtils.show_mask(self.image, top_brightness_mask)
            ImageUtils.show(self.image)

        return average_teeth_brightness_value

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
    Returns light direction results in the form of primary light direction, percent per direction and effective color 
    map. Used for sequential execution.
    """

    def get_light_direction_result(self):
        start_time = time.time()
        self.skin_config.DEBUG_MODE = False

        ycrcb_image = ImageUtils.to_YCrCb(self.image)
        mask_to_process = self.face_mask_to_process
        node_middle_point = self.nose_middle_point
        rotation_matrix = self.rotation_matrix

        primary_light_direction = SkinToneAnalyzer.get_primary_light_direction(ycrcb_image, mask_to_process,
                                                                               node_middle_point, rotation_matrix,
                                                                               self.skin_config, None)
        print("Primary light direction computation time: ", time.time() - start_time)
        return primary_light_direction

    """
    Returns only primary light direction from light direction result.
    """

    def get_light_direction(self) -> LightDirection:
        return self.get_light_direction_result()[0]

    """
    Computes Scene Brightness and Primary Light Direction. Primary light direction is executed in a separate process 
    to parallelize compute.
    """

    def get_scene_brightness_and_primary_light_direction(self) -> SceneBrightnessAndDirection:
        start_time = time.time()
        self.skin_config.DEBUG_MODE = False

        ycrcb_image = ImageUtils.to_YCrCb(self.image)
        mask_to_process = self.face_mask_to_process
        node_middle_point = self.nose_middle_point
        rotation_matrix = self.rotation_matrix
        light_direction_queue = Queue()
        p = mp.Process(target=SkinToneAnalyzer.get_primary_light_direction, args=(ycrcb_image, mask_to_process,
                                                                                  node_middle_point, rotation_matrix,
                                                                                  self.skin_config,
                                                                                  light_direction_queue))
        p.start()

        scene_brightness_value = self.determine_scene_brightness()

        p.join()

        primary_light_direction, percent_per_direction = light_direction_queue.get()

        print("Scene brightness and primary light direction detection latency: ", time.time() - start_time)

        return SceneBrightnessAndDirection(scene_brightness_value, primary_light_direction, percent_per_direction)

    """
    Computes Scene Brightness and Primary Light Direction. Scene brightness is executed in a separate process to 
    parallelize compute. Currently used in production.
    """

    def get_primary_light_direction_and_scene_brightness(self) -> SceneBrightnessAndDirection:
        if not self.is_teeth_visible:
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

        primary_light_direction, percent_per_direction, effective_color_map = self.get_light_direction_result()

        p.join()

        # Store effective color map of face mask.
        self.face_mask_effective_color_map = effective_color_map

        scene_brightness_value = brightness_queue.get()

        print("Primary light direction detection and Scene brightness latency: ", time.time() - start_time)

        return SceneBrightnessAndDirection(scene_brightness_value, primary_light_direction, percent_per_direction)

    """
    Computes skin tones associated with the face. Used in production.
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
    Computes average brightness of the RGB image for given face mask.
    """

    def get_average_face_brightness(self) -> int:
        return round(np.mean(np.max(self.image, axis=2)[self.face_mask_to_process]))

    """
    Returns dominant hues found in face.
    """

    def hues_in_face(self):
        hue_values = ImageUtils.hue_values(self.image, self.face_mask_to_process)
        print("hue vals: ", hue_values)

        if self.skin_config.DEBUG_MODE:
            plt.figure(1)
            plt.hist(hue_values, bins=60, density=True, histtype="step")
            plt.xlim([0, 40])
            plt.show(block=True)

    """
    Returns skin tones filtered by mask percent.
    """

    def filter_skin_tones_by_mask_percent(self, skin_tones):
        max_cumulative_percent = 50
        sig_skin_tones = []
        cumulative_percent = 0
        average_brightness = 0.0
        average_chroma = 0.0
        average_sat = 0.0
        total_percent = 0.0
        final_mask = np.zeros(self.image.shape[:2], dtype=bool)
        for sk in skin_tones:
            # if round(sk.percent_of_face_mask) >= mask_percent_cutoff:
            if cumulative_percent < max_cumulative_percent:
                sig_skin_tones.append(sk)
                cumulative_percent += sk.percent_of_face_mask
                average_brightness += sk.percent_of_face_mask * sk.hsv[2]
                average_chroma += sk.percent_of_face_mask * sk.hls[2]
                average_sat += sk.percent_of_face_mask * sk.hsv[1]
                final_mask = np.bitwise_or(final_mask, sk.face_mask)
                total_percent += sk.percent_of_face_mask

        print("\n 50 filter")
        print("average brightness: ", round(average_brightness/total_percent))
        print("average chroma: ", round(average_chroma / total_percent))
        print("average sat: ", round(average_sat / total_percent))
        return sig_skin_tones, final_mask


    def desirable_regions_face_mask(self, mask, desirable_percent):
        image = self.image

        bright_image = np.max(image, axis=2).astype(float)
        total_points = np.count_nonzero(mask)

        # Break mask into smaller clusters.
        max_brightness = int(np.max(bright_image[mask]))
        min_brightness = int(np.min(bright_image[mask]))
        curr_mask = np.zeros(bright_image.shape, dtype=bool)
        for brightness in range(max_brightness, min_brightness - 1, -1):
            curr_mask = np.bitwise_or(curr_mask, np.bitwise_and(bright_image == brightness, mask))
            if ImageUtils.percentPoints(curr_mask, total_points) >= desirable_percent:
                break

        return curr_mask


    def compute_metrics_for_mask(self, mask):
        image = self.image
        bright_image = np.max(image, axis=2).astype(float)* (100.0/255.0)
        average_brightness = round(np.mean(bright_image[mask]))
        std_brightness = round(np.std(bright_image[mask]))

        gray_image = ImageUtils.to_gray(image).astype(float)*(100.0/255.0)
        mean_gray = round(np.mean(gray_image[mask]))

        s_image = ImageUtils.to_hls(image)[:, :, 2].astype(float) * (100.0/255.0)
        average_s = round(np.mean(s_image[mask]))
        std_s = round(np.std(s_image[mask]))

        ss_image = ImageUtils.to_hsv(image)[:, :, 1].astype(float) * (100.0/255.0)
        mean_ss = round(np.mean(ss_image[mask]), 2)
        std_ss = round(np.std(ss_image[mask]))

        l_image = ImageUtils.to_hls(image)[:, :, 1].astype(float) * (100.0 / 255.0)
        average_l = round(np.mean(l_image[mask]))

        print("mean brightness: ", average_brightness, " mean gray: ", mean_gray, " std brightness: ",
              std_brightness ," mean lightness: ",
              average_l)
        print("mean s: ", average_s, " std s: ", std_s, " s range: ", average_s-std_s, "-", average_s + std_s ,
              " std ss: ", std_ss,
              " mean ss: ", mean_ss)
        print("ratio: ", round((average_l - average_s)/average_brightness, 2))
        print("delta s: ", average_s - mean_ss)


    def get_over_and_under_mask(self):
        image = self.image
        bright_image = np.max(image, axis=2).astype(float) * (100.0 / 255.0)
        overexposed_mask = bright_image >= 94
        underexposed_mask = bright_image <= 8
        total_points = image.shape[0] * image.shape[1]
        print("\noverexposed percent: ", ImageUtils.percentPoints(overexposed_mask, total_points), " underexposed "
                                                                                                   "points: ", 
              ImageUtils.percentPoints(underexposed_mask, total_points), "\n")
        return overexposed_mask, underexposed_mask

    def get_teeth_facing_light(self):
        mmask = self.mouth_mask_to_process

        tmask = np.bitwise_and(self.face.lips_and_mouth_mask, face_mask_with_direct_light(self.image))
        #ImageUtils.show_mask(self.image, tmask)
        boundary_masks = ImageUtils.find_boundaries(tmask)

        # Pick boundary mask with most points.
        best_boundary_mask = None
        max_points_count = 0
        for b in boundary_masks:
            if np.count_nonzero(b) > max_points_count:
                best_boundary_mask = b
                max_points_count = np.count_nonzero(b)

        #ImageUtils.show_mask(self.image, best_boundary_mask)
        _, c, w, _ = ImageUtils.bbox(best_boundary_mask)

        # only use mask within column bounds.
        mmask[:, :c] = False
        mmask[:, c+w:] = False

        # constrain within nose edges.
        _, cn, wn, _ = ImageUtils.bbox(self.face.get_nose_keypoints())
        mmask[:, :cn] = False
        mmask[:, cn+wn:] = False

        return mmask

    """
    Detects skin tone and primary light direction for given face image. Only used for debugging purposes. Do not use 
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
        mask_directions_list = [ImageUtils.get_mask_direction(b_mask, self.nose_middle_point, self.rotation_matrix,
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
                    ImageUtils.face.show_mask(self.image, m)

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
                mean_color_rgb = np.round(np.mean(rgb_image[mask], axis=0), 2)
                hsv = ImageUtils.toHSVPreferredRange(ImageUtils.sRGBtoHSV(mean_color_rgb)[0])
                hls = ImageUtils.toHSVPreferredRange(ImageUtils.sRGBtoHLS(mean_color_rgb)[0])

                skin_tone = SkinTone(rgb=mean_color_rgb.tolist(), hsv=hsv.tolist(), hls=hls.tolist(), gray=0.0,
                                     ycrcb=[],
                                     percent_of_face_mask=round(percent_of_face_mask, 2), face_mask=mask.copy(),
                                     profile=SkinTone.DISPLAY_P3)
                skin_tones.append(skin_tone)

        return skin_tones


"""
Helper to read test face mask info from test images. Can be removed after tested on server.
"""


def get_test_face_mask_info():
    dir = "/Users/addarsh/virtualenvs/facemagik_server/facetone/"

    def get_mask(fname):
        return ImageUtils.get_boolean_mask(ImageUtils.read_grayscale_image(
            dir + fname))

    with open(dir + "nose_middle_point.txt", "r") as f:
        nose_middle_point = [int(v) for v in f.read().splitlines()]
    print("nose middle point: ", nose_middle_point)
    face_mask_info = FaceMaskInfo(face_mask_to_process=get_mask("test_face_mask.png"),
                                  mouth_mask_to_process=get_mask("test_mouth_mask.png"),
                                  nose_middle_point=NoseMiddlePoint(x=nose_middle_point[0], y=nose_middle_point[1]),
                                  left_eye_mask=get_mask("left_eye_mask.png"),
                                  right_eye_mask=get_mask("right_eye_mask.png"))
    return face_mask_info

"""
Performs detection in directory on one image at a time.
"""

def test_detection_dir(mrcnn_model, sk_config):
    from os import listdir
    from os.path import isfile, join
    path = "/Users/addarsh/virtualenvs/facemagik_server/facetone/rotation_mode"
    image_paths = [join(path,f) for f in listdir(path) if isfile(join(path, f)) and f.endswith(".png") and "tone" not
                   in f and "filtered" not in f]

    count = 0
    for p in image_paths:
        sk_config.IMAGE_PATH = p
        print("Path: ", p)
        an = SkinToneAnalyzer(mrcnn_model, sk_config, None)
        filtered_skin_tones = ImageUtils.smaller_cluster_skin_tones(analyzer.image, analyzer.face_mask_to_process)
        #shade_file_name = os.path.splitext(args.image)[0] + "_shade_clusters.png"
        filtered_file_name = os.path.splitext(p)[0] + "_filtered_" + str(round(abs(filtered_skin_tones[0].hls[
                                                                                          2]-filtered_skin_tones[
            -1].hls[2]))) + ".png"
        icc_profile_path = "/Users/addarsh/Desktop/anastasia-me/displayP3_icc_profile.txt"
        print("name: ", filtered_file_name)
        ImageUtils.save_skin_tones_to_file(filtered_file_name, filtered_skin_tones, icc_profile_path=icc_profile_path)
        count += 1
        print("done: ", count)


def show_only_face_mask(analyzer):
    clone = analyzer.image.copy()
    mask = np.bitwise_xor(np.ones(clone.shape[:2], dtype=bool), analyzer.face_mask_to_process)
    clone[mask] = [0, 0, 0]
    ImageUtils.show(clone)


def find_sat_mask(analyzer):
    image = analyzer.image
    s_image = ImageUtils.to_hsv(image)[:, :, 1].astype(float) * (100.0/255.0)
    b_image = ImageUtils.to_hsv(image)[:, :, 2].astype(float) * (100.0 / 255.0)
    mask = s_image < 15
    mask = np.bitwise_and(mask, b_image > 50)
    mask = np.bitwise_and(mask, analyzer.face.get_complete_face_mask())
    emask = np.zeros(analyzer.image.shape[:2], dtype=bool)
    for em in analyzer.face.get_eye_masks():
        #mask = np.bitwise_and(mask, em)
        emask = np.bitwise_or(emask, em)
    mask = np.bitwise_and(mask, emask)
    mask = np.bitwise_and(mask,analyzer.face.get_points_between_eyeballs())
    print("grey mask")

    bright_image = np.max(image, axis=2).astype(float)

    # use top 50%
    max_brightness = int(np.max(bright_image[mask]))
    min_brightness = int(np.min(bright_image[mask]))
    curr_mask = np.zeros(bright_image.shape, dtype=bool)
    print("max and min: ", max_brightness, min_brightness)
    for brightness in range(max_brightness, min_brightness - 1, -1):
        if ImageUtils.percentPoints(curr_mask, np.count_nonzero(mask)) > 50:
            break
        curr_mask = np.bitwise_or(curr_mask, np.bitwise_and(bright_image == brightness, mask))

    print("points between eyes")


    average_brightness = round(np.mean(bright_image[curr_mask]).astype(float)*(100.0/255.0))
    print("average brightness of grey mask: ", average_brightness)
    ImageUtils.show_mask(analyzer.image, curr_mask)


def find_gray_color_checker(analyzer):
    image = analyzer.image
    gray_image = ImageUtils.to_gray(image)

    """
    gray_color = 135
    middle_grey = np.reshape(np.array([gray_color, gray_color, gray_color]), (1,3))
    deltaImage = ImageUtils.delta_cie2000_matrix_v2(middle_grey, image)
    mask = deltaImage < 5
    """

    mask = np.bitwise_and(gray_image > 127, gray_image < 160)
    fr, _, _, hr = ImageUtils.bbox(analyzer.face.get_face_mask())
    # Discard all mask points above and including the face.
    mask[:fr + hr+1, :] = False

    # find contours.
    binary_clone = gray_image.copy()
    binary_clone[mask] = 255
    binary_clone[np.bitwise_xor(np.ones(image.shape[:2], dtype=bool), mask)] = 0

    ImageUtils.show(binary_clone)
    ImageUtils.show(gray_image)

    print("\nGray metrics")
    print("percent: ", ImageUtils.percentPoints(mask, image.shape[0]*image.shape[1]))
    print("mean gray value: ", np.mean(gray_image[mask]))
    return mask

def check_brightness(final_skin_tones):
    # Find smallest number of consecutive masks that add up to > 30%.
    start = 0
    end = start + 1
    cum_sum = final_skin_tones[start].percent_of_face_mask
    cur_tones = [final_skin_tones[start]]
    best_tones = final_skin_tones
    best_sum = 0.0
    best_start_pos = 0
    while start < len(final_skin_tones) - 1 and end < len(final_skin_tones):
        print("cum sum: ", cum_sum, " start and end: ", start, end, len(final_skin_tones))
        while end < len(final_skin_tones) and cum_sum < 30:
            cum_sum += final_skin_tones[end].percent_of_face_mask
            cur_tones.append(final_skin_tones[end])
            end += 1
        if end == len(final_skin_tones):
            break


        # Have to change this condition since it keeps looping when start = start and end = start +1/2 for ww-29
        while start < end-1 and cum_sum >= 30:
            if (len(cur_tones) < len(best_tones)) or (len(cur_tones) == len(best_tones) and cum_sum > best_sum):
                best_tones = cur_tones.copy()
                best_sum = cum_sum
                best_start_pos = start

            cur_tones.pop(0)
            cum_sum -= final_skin_tones[start].percent_of_face_mask
            start += 1


    # HSV = [24, 40, 73]
    foundation_color = np.array([186,141,112])
    # HSV = p179, 136, 107]
    #foundation_color = np.array([179, 136, 107])
    print("\nbrightness calc")
    print("best sum: ", best_sum, " best tones len: ", len(best_tones))
    for sk in best_tones:
        print("V: ", sk.hsv[2], " chroma: ", sk.hls[2], " sat: ", sk.hsv[1], " percent: ",
              sk.percent_of_face_mask, " delta cie: ", ImageUtils.delta_cie2000_v2(sk.rgb, foundation_color))

    print("\nbrighter tones with end pos: ", best_start_pos)
    for i in range(0, best_start_pos):
        sk = final_skin_tones[i]
        if round(sk.percent_of_face_mask) >= 1:
            print("V: ", sk.hsv[2], " chroma: ", sk.hls[2], " sat: ", sk.hsv[1],
                                                     " percent: ",
                  sk.percent_of_face_mask, " delta cie: ", ImageUtils.delta_cie2000_v2(sk.rgb, foundation_color))

    return best_tones


if __name__ == "__main__":
    # Run this script from parent directory level (face_magik) of this module.
    # Example: python -m facemagik.skintone --image <image_path>
    # Stack Overflow answer: https://stackoverflow.com/questions/47319423

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Image for processing')
    parser.add_argument('--image', required=True, metavar="path to image file")
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

    # Testing directory detection.
    #test_detection_dir(maskrcnn_model, skin_detection_config)
    #exit(0)

    face_mask_config = None
    # Uncomment if you want to test face mask info.
    # skin_detection_config.IMAGE = ImageUtils.read_rgb_image(args.image)
    # face_mask_config = get_test_face_mask_info()

    analyzer = SkinToneAnalyzer(maskrcnn_model, skin_detection_config, face_mask_config)
    #print("Brightness value: ", analyzer.determine_scene_brightness())
    #print("Average face brightness value: ", analyzer.get_average_face_brightness())
    print ("Primary light direction: ", analyzer.get_light_direction_result()[:2])
    #print("Scene brightness and light direction: ", analyzer.get_scene_brightness_and_primary_light_direction())
    #analyzer.detect_skin_tone_and_light_direction()
    #print("light direction and scene brightness: ", analyzer.get_primary_light_direction_and_scene_brightness())
    #print("light direction only: ", analyzer.get_light_direction())
    # uncomment to debug light direction.
    #analyzer.detect_skin_tone_and_light_direction()
    #analyzer.hues_in_face()

    # Show skin tones.
    #final_skin_tones = analyzer.get_skin_tones()
    final_skin_tones = ImageUtils.smaller_cluster_skin_tones(analyzer.image, analyzer.face_mask_to_process)
    #final_skin_tones = ImageUtils.smaller_cluster_skin_tones(analyzer.image, analyzer.face.get_forehead_points())
    desirable_fac_mask = analyzer.desirable_regions_face_mask(analyzer.face_mask_to_process, 50)
    analyzer.compute_metrics_for_mask(desirable_fac_mask)

    anti_mask = np.bitwise_xor(analyzer.face_mask_to_process, desirable_fac_mask)
    analyzer.compute_metrics_for_mask(anti_mask)

    print("Metrics for full mask")
    analyzer.compute_metrics_for_mask(analyzer.face_mask_to_process)

    # teeth mask
    print("\n\n Teeth if any")
    desirable_teeth_mask = analyzer.desirable_regions_face_mask(analyzer.mouth_mask_to_process, 15)
    #desirable_teeth_mask = np.bitwise_and(analyzer.mouth_mask_to_process, face_mask_with_direct_light(analyzer.image))
    #desirable_teeth_mask = analyzer.get_teeth_facing_light()
    # get rid of gums
    #desirable_teeth_mask = np.bitwise_and(desirable_teeth_mask,  analyzer.desirable_regions_face_mask(
    #    desirable_teeth_mask, 50))
    #ImageUtils.plot_histogram(analyzer.image, desirable_teeth_mask, channel=0, block=True, bins=50, fig_num=1,
    #                          xlim=[0,
    #                                256])

    """
    ImageUtils.plot_histogram(analyzer.image, np.ones(analyzer.image.shape[:2], dtype=bool), channel=2, block=True,
                              bins=256,
                              fig_num=1,xlim=[0,256])

    ImageUtils.plot_histogram(analyzer.image, np.ones(analyzer.image.shape[:2], dtype=bool), channel=1, block=True,
                              bins=50,
                              fig_num=1, xlim=[0, 256])
    ImageUtils.plot_histogram(analyzer.image, np.ones(analyzer.image.shape[:2], dtype=bool), channel=2, block=True,
                              bins=50,
                              fig_num=1, xlim=[0, 256])
    """

    #ImageUtils.plot_gray_histogram(ImageUtils.to_gray(analyzer.image), np.ones(analyzer.image.shape[:2], dtype=bool),
    #                              block=True, bins=255)

    analyzer.compute_metrics_for_mask(desirable_teeth_mask)
    ImageUtils.show_mask(analyzer.image, desirable_teeth_mask)

    print("\n\n Teeth Part 2")
    desirable_teeth_mask = analyzer.desirable_regions_face_mask(analyzer.mouth_mask_to_process, 30)
    analyzer.compute_metrics_for_mask(desirable_teeth_mask)
    ImageUtils.show_mask(analyzer.image, desirable_teeth_mask)

    filtered_skin_tones, bright_50_mask = analyzer.filter_skin_tones_by_mask_percent(final_skin_tones)
    mean_50_color_rgb = np.round(np.mean(analyzer.image[bright_50_mask], axis=0), 2)
    print("Mean 50 RGB: ", mean_50_color_rgb)

    cum_delta_percent = 0.0
    cum_percent = 0.0
    cutoff_print = 50.0
    count_idx = 1
    cutoff_reached = False
    shadow_mask = np.zeros(analyzer.image.shape[:2], dtype=bool)
    one_percent_mask = np.zeros(analyzer.image.shape[:2], dtype=bool)
    first_cum_mask = np.zeros(analyzer.image.shape[:2], dtype=bool)
    max_one_c = -1
    min_one_c = -1
    max_one_v = -1
    max_one_gray = -1
    min_one_gray = -1
    min_one_c = 100
    max_one_ss = -1
    min_one_ss = -1
    first_c = -1
    first_sat = -1
    one_percent_tones = []
    #foundation_color = np.array([179, 136, 107])
    foundation_color = np.array([186,141,112])
    #foundation_color = np.array([195, 144, 107])
    # mac foundation.
    #foundation_color = np.array([212, 181, 146])
    #foundation_color = np.array([220, 180, 140])
    #foundation_color = np.array([191, 144, 117])
    for i, sk in enumerate(final_skin_tones):
        if first_c == -1:
            first_c = round(sk.hls[2])
            first_sat = round(sk.hsv[1])
        if cum_percent < 1:
            first_cum_mask = np.bitwise_or(first_cum_mask, sk.face_mask)

        prev_d = -1
        next_d = -1
        if i > 0:
            prev_d = round(ImageUtils.delta_cie2000_v2(sk.rgb, final_skin_tones[i-1].rgb),2)
        if i < len(final_skin_tones) - 1:
            next_d = round(ImageUtils.delta_cie2000_v2(sk.rgb, final_skin_tones[i+1].rgb),2)

        delta_50_d = round(ImageUtils.delta_cie2000_v2(sk.rgb, mean_50_color_rgb),2)

        cum_percent += sk.percent_of_face_mask
        if delta_50_d <= 4:
            cum_delta_percent += sk.percent_of_face_mask
        delta_ss = round(sk.hsv[1] - sk.hls[2])
        w_ratio = 0.0
        if sk.hsv[2] > 0:
            w_ratio = (sk.hls[1] - sk.hls[2])/(sk.hsv[2])
        print("V: ", sk.hsv[2], " G: ", round(sk.gray,2)," S: ", sk.hls[2], " SS: ", sk.hsv[1]," percent: ",
              sk.percent_of_face_mask,
              " cum percent: ", round(cum_percent, 2), " L: ", sk.hls[1], " delta LS: ", round(sk.hls[1] - sk.hls[
                2]), " ratio: ", round(w_ratio, 2), " delta SS: ", delta_ss, " prevd: ", prev_d, " 50 d: ", delta_50_d,
              " delta cie: ", round(ImageUtils.delta_cie2000_v2(sk.rgb, foundation_color), 2), " cum delta percent: "
                                                                                               "",
              round(cum_delta_percent,2))
        if cum_percent > cutoff_print and not cutoff_reached:
            print(" Reached cum percent in : ", count_idx, "counts with brightness: ", sk.hsv[2])
            cutoff_reached = True
        if delta_ss >= 16:
            shadow_mask = np.bitwise_or(shadow_mask, sk.face_mask)

        # Update this to stop counting one percents once we hit two or more consecutive masks < 1 percent.
        if sk.percent_of_face_mask >= 1:
            one_percent_mask = np.bitwise_or(one_percent_mask, sk.face_mask)
            if max_one_c == -1:
                max_one_c = round(sk.hls[2])
                max_one_v = round(sk.hsv[2])
                max_one_ss = round(sk.hsv[1])
                max_one_gray = round(sk.gray)
            min_one_c = round(min(min_one_c, sk.hls[2]))
            min_one_v = round(sk.hsv[2])
            min_one_ss = round(max(min_one_ss, round(sk.hsv[1])))
            one_percent_tones.append(sk)
            min_one_gray = round(sk.gray)

        count_idx += 1

    max_chroma = round(np.mean((ImageUtils.to_hls(analyzer.image)[:, :, 2].astype(float)*(100.0/255.0))[
                                   first_cum_mask]))

    print("\nMAX chroma diff: ", first_c - min_one_c, " where upper: ", first_c, " and lower: ", min_one_c,"\n")
    print("\nMAX sat diff: ", first_sat - min_one_ss, " where upper: ", first_sat, " and lower: ", min_one_ss, "\n")

    print("one percent mask delta chroma: ", max_one_c - min_one_c,  " chroma diff: ", max_chroma - min_one_c," delta ss: ", min_one_ss - max_one_ss,
          " delta gray: ", max_one_gray - min_one_gray)

    print("\nTrue Delta brightness: ", round((((max_one_v*1.0)/100.0)**2.2 - ((min_one_v*1.0)/100.0)**2.2)*100.0),
          " from: ", max_one_v, " to: ", min_one_v,"\n")
    #eye_mask = find_sat_mask(analyzer)
    #print("eye mask")

    print("Saturation plot chroma (y) vs brightness (x)")
    x  = [t.hsv[2] for t in final_skin_tones]
    #x = [t.gray for t in final_skin_tones]
    y = [t.hls[2] for t in final_skin_tones]
    #y = [t.ycrcb[2] for t in final_skin_tones]
    #plt.plot(x, y)
    plt.xlim(100, 0)
    plt.ylim(0, 100)
    #plt.ylim(90, 130)
    #plt.show(block=True)

    overexposed_mask, underexposed_mask = analyzer.get_over_and_under_mask()
    #ImageUtils.show_mask(analyzer.image, overexposed_mask)
    #ImageUtils.show_mask(analyzer.image, underexposed_mask)
    #bgMask = analyzer.face.detect_background()
    #ImageUtils.show_mask(analyzer.image, underexposed_mask)


    #ImageUtils.show_mask(analyzer.image, bright_50_mask)
    ImageUtils.show(analyzer.image)

    best_tones = check_brightness(final_skin_tones)

    #analyzer.hues_in_face()
    #ImageUtils.show(analyzer.image)
    #ImageUtils.show(ImageUtils.to_YCrCb(analyzer.image))

    #print("Skin Tones production: ", final_skin_tones)
    shade_file_name = os.path.splitext(args.image)[0] + "_shade_clusters.png"
    filtered_file_name = os.path.splitext(args.image)[0] + "_filtered.png"
    one_percent_file_name = os.path.splitext(args.image)[0] + "_percent.png"
    icc_profile_path = "/Users/addarsh/Desktop/anastasia-me/displayP3_icc_profile.txt"
    ImageUtils.save_skin_tones_to_file(shade_file_name, final_skin_tones, icc_profile_path=icc_profile_path)
    ImageUtils.save_skin_tones_to_file(filtered_file_name,  filtered_skin_tones, icc_profile_path=icc_profile_path)
    ImageUtils.save_skin_tones_to_file(one_percent_file_name, one_percent_tones, icc_profile_path=icc_profile_path,
                                       skip_text=True)
    if (args.bri is not None and float(args.bri) != 1.0) or (args.sat is not None and float(args.sat) != 1.0):
        mod_file_name = os.path.splitext(args.image)[0] + "_mod.png"
        ImageUtils.save_image_to_file(analyzer.image, mod_file_name,
                                      icc_profile_path=icc_profile_path)

