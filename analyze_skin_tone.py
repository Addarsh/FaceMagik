import os
import argparse
import numpy as np
from image_utils import ImageUtils
import matplotlib.pyplot as plt
import time

from face import Face
from common import InferenceConfig
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

    # Minimum mask percent until which repeated Kmeans clustering is performed.
    KMEANS_MASK_PERCENT_CUTOFF: float = 2.0

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


"""
Repeatedly divides mask into clusters using kmeans until difference between
clusters is less than given tolerance. Returns the cluster with the largest
diff value. diffImg has dimensions (W, H) and contains the values to peform
clustering on. mask is boolean mask of same dimensions,
"""


def brightest_cluster(diff_img: np.ndarray, mask: object, total_points: int, tol: int = 2,
                      cutoff_percent: int = 2) -> np.ndarray:
    c1_tuple, c2_tuple = ImageUtils.Kmeans_1d(diff_img, mask)
    c1_mask, centroid1 = c1_tuple
    c2_mask, centroid2 = c2_tuple

    if ImageUtils.percentPoints(c1_mask, total_points) < cutoff_percent or ImageUtils.percentPoints(c2_mask,
                                                                                                    total_points) < \
            cutoff_percent:
        # end cluster division.
        return mask
    if abs(centroid1 - centroid2) <= tol:
        # end cluster division.
        return mask
    return brightest_cluster(diff_img, c1_mask, total_points, tol, cutoff_percent)


"""
Plots a figure with each cluster's color and Munsell value. Primary use for analysis of similar colors.
"""


def plot_colors(image, mask_list, total_points):
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
Constructs a MaskRCNN model and returns it.
"""


def construct_model():
    # Create model
    model = model_lib.MaskRCNN(mode="inference", config=InferenceConfig(), model_dir="")

    # Select weights file to load
    try:
        weights_path = os.path.join(os.getcwd(), "maskrcnn_model/mask_rcnn_face_0060.h5")
    except Exception as e:
        raise

    print("Loading weights from: ", weights_path)
    model.load_weights(weights_path, by_name=True)

    return model


"""
analyze will process image with the given path. It will augment the image
by given factor of brightness and saturation before processing.
"""


def analyze(skin_config: object):
    # Load Mask RCNN model.
    start_time = time.time()
    maskrcnn_model = construct_model()
    print("\nModel construction time: ", time.time() - start_time, " seconds\n")

    # Detect face.
    f: Face = None
    if skin_config.IMAGE_PATH != "":
        f = Face(image_path=skin_config.IMAGE_PATH, maskrcnn_model=maskrcnn_model)
        #f = Face(image=ImageUtils.read_rgb_image(skin_config.IMAGE_PATH), maskrcnn_model=maskrcnn_model)

    if skin_config.BRIGHTNESS_UPDATE_FACTOR != 1.0 or skin_config.SATURATION_UPDATE_FACTOR != 1.0:
        new_img = ImageUtils.set_brightness(f.image, skin_config.BRIGHTNESS_UPDATE_FACTOR)
        new_img = ImageUtils.set_saturation(new_img, skin_config.SATURATION_UPDATE_FACTOR)
        f.image = new_img

    if skin_config.DEBUG_MODE:
        print("teeth visible: ", f.is_teeth_visible())
        # f.show_mask(f.get_mouth_points())

    # mask_to_process = f.get_face_keypoints()
    # mask_to_process = f.get_face_until_nose_end()
    # mask_to_process = f.get_face_mask_without_area_around_eyes()
    mask_to_process = f.get_face_until_nose_end_without_area_around_eyes()

    curr_mask = mask_to_process.copy()
    total_points = np.count_nonzero(mask_to_process)

    effective_color_map = {}
    all_cluster_masks = []
    mask_directions_list = []
    mask_percent_list = []

    ycrcb_image = ImageUtils.to_YCrCb(f.image)
    diff = (ycrcb_image[:, :, 0]).astype(float)

    # Divide the image into smaller clusters.
    start_time = time.time()
    while True:
        # Find the brightest cluster.
        b_mask = brightest_cluster(diff, curr_mask, total_points, tol=skin_config.KMEANS_TOLERANCE,
                                   cutoff_percent=skin_config.KMEANS_MASK_PERCENT_CUTOFF)
        # Find the least saturated cluster of the brightest cluster. This provides more fine-grained clusters
        # but is also more expensive. Comment it out if you want to plot "color of each cluster versus
        # the associated Munsell hue" to iterate/improve effective color mapping.
        # b_mask = brightest_cluster(255.0 -(ImageUtils.to_hsv(ycrcb_image)[:, :, 1]).astype(np.float), b_mask,
        #                          np.count_nonzero(b_mask), tol=tol, cutoffPercent=cutoff_percent)

        munsell_color = ImageUtils.sRGBtoMunsell(np.mean(ycrcb_image[b_mask], axis=0))
        effective_color = f.effective_color(munsell_color)
        if effective_color not in effective_color_map:
            effective_color_map[effective_color] = b_mask
        else:
            effective_color_map[effective_color] = np.bitwise_or(effective_color_map[effective_color], b_mask)

        # Store this mask for different computations.
        all_cluster_masks.append(b_mask)
        mask_directions_list.append(f.get_mask_direction(b_mask, show_debug_info=skin_config.DEBUG_MODE))
        mask_percent_list.append(ImageUtils.percentPoints(b_mask, total_points))

        if skin_config.DEBUG_MODE:
            print("effective color: ", effective_color, " brightness: ", round(np.mean(ycrcb_image[:, :, 0][b_mask]),
                                                                               2), "\n")
            # f.show(ImageUtils.plot_points_and_mask(ycrcb_image, [f.noseMiddlePoint], bMask))

        curr_mask = np.bitwise_xor(curr_mask, b_mask)
        if ImageUtils.percentPoints(curr_mask, total_points) < 1:
            break

    print("\nClustering latency: ", time.time() - start_time, " seconds\n")

    final_light_direction = Face.process_mask_directions(mask_directions_list, mask_percent_list)
    print("\nFinal Light Direction: ", final_light_direction)

    if skin_config.DEBUG_MODE:
        plot_colors(ycrcb_image, all_cluster_masks, total_points)

    start_time = time.time()
    effective_color_map = f.iterate_effectiveColorMap(effective_color_map, all_cluster_masks)
    print("\ncolor map iteration time: ", time.time() - start_time, " seconds\n")

    if skin_config.DEBUG_MODE:
        f.print_effective_color_map(effective_color_map, total_points)

    if skin_config.COMBINE_MASKS:
        start_time = time.time()
        combined_masks = f.combine_masks_close_to_each_other(effective_color_map)
        print ("\n combining masks time: ", time.time() - start_time, " seconds\n")

        if skin_config.DEBUG_MODE:
            print("\nCombined masks")
            for m in combined_masks:
                print("percent: ", ImageUtils.percentPoints(m, total_points))
                f.show_mask(m)

    if skin_config.DEBUG_MODE:
        f.show_orig_image()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Image for processing')
    parser.add_argument('--image', required=True, metavar="path to video file")
    parser.add_argument('--bri', required=False, metavar="bri")
    parser.add_argument('--sat', required=False, metavar="sat")
    args = parser.parse_args()

    skin_config = SkinDetectionConfig()
    skin_config.IMAGE_PATH = args.image
    skin_config.COMBINE_MASKS = True
    skin_config.DEBUG_MODE = False

    if args.bri is not None:
        skin_config.BRIGHTNESS_UPDATE_FACTOR = float(args.bri)
    if args.sat is not None:
        skin_config.SATURATION_UPDATE_FACTOR = float(args.sat)

    analyze(skin_config)
