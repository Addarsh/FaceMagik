import os
import argparse
import numpy as np
from image_utils import ImageUtils
import matplotlib.pyplot as plt
import time

from face import Face
from common import InferenceConfig, SceneBrightness
from mrcnn import model as model_lib

"""
Configuration details associated with skin detection algorithm.
"""


class SkinDetectionConfig:
    # Absolute path where image to be processed in located. Leave empty if passing Image directly in memory.
    IMAGE_PATH: str = ""

    # Numpy array of image that needs to be processed. Leave as None if fetching image from local absolute path.
    IMAGE: np.ndarray = None

    # Minimum Kmeans difference value until repeated Kmeans clustering is performed.
    KMEANS_TOLERANCE: float = 2.0

    # Minimum mask percent until which repeated Kmeans clustering is performed on teeth mask.
    KMEANS_TEETH_MASK_PERCENT_CUTOFF: float = 5.0

    # Minimum mask percent until which repeated Kmeans clustering is performed on face mask.
    KMEANS_FACE_MASK_PERCENT_CUTOFF: float = 2.0

    # If true, iterate teeth clusters for fine-grained results. Defaults to false.
    ITERATE_TEETH_CLUSTERS = False

    # If true, iterate face clusters for fine-grained results. Defaults to true.
    ITERATE_FACE_CLUSTERS = True

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

    def __init__(self, skin_config: object):
        # Load Mask RCNN model.
        maskrcnn_model = SkinToneAnalyzer.construct_model()

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
        self.skin_config = skin_config

    """
    Constructs a MaskRCNN model and returns it.
    """

    @staticmethod
    def construct_model():
        start_time = time.time()

        # Create model
        model = model_lib.MaskRCNN(mode="inference", config=InferenceConfig(), model_dir="")

        # Select weights file to load
        try:
            weights_path = os.path.join(os.getcwd(), "maskrcnn_model/mask_rcnn_face_0060.h5")
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
    Processes image to determine brightness of the scene. The person in the image is expected to be smiling.
    """

    def determine_brightness(self) -> SceneBrightness:
        if not self.face.is_teeth_visible():
            raise TeethNotVisibleException("Teeth not visible for config: " +
                                           str(self.skin_config))

        mask_to_process = self.face.get_mouth_points()
        total_points = np.count_nonzero(mask_to_process)
        ycrcb_image = ImageUtils.to_YCrCb(self.face.image)

        # Make clusters.
        all_cluster_masks, effective_color_map = SkinToneAnalyzer.make_clusters(ycrcb_image, mask_to_process,
                                                                                self.skin_config.KMEANS_TOLERANCE,
                                                                                self.skin_config.KMEANS_TEETH_MASK_PERCENT_CUTOFF,
                                                                                self.skin_config.DEBUG_MODE)

        if self.skin_config.ITERATE_TEETH_CLUSTERS:
            # Iterate to optimize final clusters.
            effective_color_map = self.face.iterate_effective_color_map(effective_color_map, all_cluster_masks)

        if self.skin_config.DEBUG_MODE:
            self.face.print_effective_color_map(effective_color_map, total_points)

        # Check mean brightness minimum coverage of the teeth.
        final_mask = np.zeros(self.face.faceMask.shape, dtype=bool)

        # Find mean brightness of first two masks in decreasing order of brightness.
        count = 0
        for effective_color in effective_color_map:
            final_mask = np.bitwise_or(final_mask, effective_color_map[effective_color])
            count += 1
            if count == 2:
                break

        mean_brightness = round(np.mean(np.max(self.face.image, axis=2)[final_mask]))
        if self.skin_config.DEBUG_MODE:
            print("\nMean brightness value: ", mean_brightness, " with percent: ", ImageUtils.percentPoints(final_mask,
                                                                                                            total_points
                                                                                                            ), "\n")
        if self.skin_config.DEBUG_MODE:
            self.face.show_orig_image()
            
        # Map to brightness value.
        if mean_brightness < 200:
            return SceneBrightness.DARK_SHADOW
        elif mean_brightness < 220:
            return SceneBrightness.SOFT_SHADOW
        elif mean_brightness > 225:
            return SceneBrightness.TOO_BRIGHT

        return SceneBrightness.NEUTRAL_LIGHTING

    """
    Returns light direction for given Face image.
    """

    def get_light_direction(self):
        mask_to_process = self.face.get_face_until_nose_end_without_area_around_eyes()
        total_points = np.count_nonzero(mask_to_process)
        ycrcb_image = ImageUtils.to_YCrCb(self.face.image)

        # Make clusters.
        all_cluster_masks, effective_color_map = SkinToneAnalyzer.make_clusters(ycrcb_image, mask_to_process,
                                                                                self.skin_config.KMEANS_TOLERANCE,
                                                                                self.skin_config.KMEANS_FACE_MASK_PERCENT_CUTOFF,
                                                                                self.skin_config.DEBUG_MODE)

        # Get light direction from face mask clusters.
        mask_directions_list = [self.face.get_mask_direction(b_mask, show_debug_info=self.skin_config.DEBUG_MODE) for
                                b_mask in
                                all_cluster_masks]
        mask_percent_list = [ImageUtils.percentPoints(b_mask, total_points) for b_mask in all_cluster_masks]

        return Face.process_mask_directions(mask_directions_list, mask_percent_list)

    """
    Process skin tone for given image.
    """

    def process_skin_tone(self) -> None:
        # mask_to_process = self.face.get_face_keypoints()
        # mask_to_process = self.face.get_face_until_nose_end()
        # mask_to_process = self.face.get_face_mask_without_area_around_eyes()
        mask_to_process = self.face.get_face_until_nose_end_without_area_around_eyes()
        total_points = np.count_nonzero(mask_to_process)
        ycrcb_image = ImageUtils.to_YCrCb(self.face.image)

        # Make clusters.
        all_cluster_masks, effective_color_map = SkinToneAnalyzer.make_clusters(ycrcb_image, mask_to_process,
                                                                                self.skin_config.KMEANS_TOLERANCE,
                                                                                self.skin_config.KMEANS_FACE_MASK_PERCENT_CUTOFF,
                                                                                self.skin_config.DEBUG_MODE)

        # Get light direction from face mask clusters.
        mask_directions_list = [self.face.get_mask_direction(b_mask, show_debug_info=self.skin_config.DEBUG_MODE) for
                                b_mask in
                                all_cluster_masks]
        mask_percent_list = [ImageUtils.percentPoints(b_mask, total_points) for b_mask in all_cluster_masks]

        final_light_direction = Face.process_mask_directions(mask_directions_list, mask_percent_list)
        print("\nFinal Light Direction: ", final_light_direction)

        if self.skin_config.DEBUG_MODE:
            SkinToneAnalyzer.plot_colors(ycrcb_image, all_cluster_masks, total_points)

        if self.skin_config.ITERATE_FACE_CLUSTERS:
            # Iterate to optimize final clusters.
            effective_color_map = self.face.iterate_effective_color_map(effective_color_map, all_cluster_masks)

        if self.skin_config.DEBUG_MODE:
            self.face.print_effective_color_map(effective_color_map, total_points)

        if self.skin_config.COMBINE_MASKS:
            combined_masks = self.face.combine_masks_close_to_each_other(effective_color_map)

            if self.skin_config.DEBUG_MODE:
                print("\nCombined masks")
                for m in combined_masks:
                    print("percent: ", ImageUtils.percentPoints(m, total_points))
                    self.face.show_mask(m)

        if self.skin_config.DEBUG_MODE:
            self.face.show_orig_image()

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


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Image for processing')
    parser.add_argument('--image', required=True, metavar="path to video file")
    parser.add_argument('--bri', required=False, metavar="bri")
    parser.add_argument('--sat', required=False, metavar="sat")
    args = parser.parse_args()

    skin_detection_config = SkinDetectionConfig()
    skin_detection_config.IMAGE_PATH = args.image
    skin_detection_config.COMBINE_MASKS = True
    skin_detection_config.DEBUG_MODE = True
    skin_detection_config.ITERATE_TEETH_CLUSTERS = True

    if args.bri is not None:
        skin_detection_config.BRIGHTNESS_UPDATE_FACTOR = float(args.bri)
    if args.sat is not None:
        skin_detection_config.SATURATION_UPDATE_FACTOR = float(args.sat)

    analyzer = SkinToneAnalyzer(skin_detection_config)
    analyzer.process_skin_tone()
    #print ("Light direction: ", analyzer.get_light_direction())
    #print("Brightness value: ", analyzer.determine_brightness())
