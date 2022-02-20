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


def plot_color(image, mask_list, total_points):
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
construct_model constructs a MaskRCNN model and returns it.
"""


def construct_model():
    # Create model
    model = model_lib.MaskRCNN(mode="inference", config=InferenceConfig(),
                              model_dir="")
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


def analyze(bri=1.0, sat=1.0):
    # Load Mask RCNN model.
    start_time = time.time()
    maskrcnn_model = construct_model()
    print("\nModel construction time: ", time.time() - start_time, " seconds\n")

    # Detect face.
    #f = Face(image_path=args.image, maskrcnn_model=maskrcnn_model)
    f = Face(image=ImageUtils.read_rgb_image(args.image), maskrcnn_model=maskrcnn_model)

    f.windowName = "image"

    new_img = ImageUtils.set_brightness(f.image, bri)
    new_img = ImageUtils.set_saturation(new_img, sat)
    f.image = new_img

    print("teeth visible: ", f.is_teeth_visible())
    # f.show_mask(f.get_mouth_points())

    # maskToProcess = f.get_face_keypoints()
    # maskToProcess = f.get_face_until_nose_end()
    # maskToProcess = f.get_face_mask_without_area_around_eyes()
    maskToProcess = f.get_face_until_nose_end_without_area_around_eyes()

    tol = 2
    cutoff_percent = 2
    # tol = 2
    # cutoffPercent = 1
    curr_mask = maskToProcess.copy()
    total_points = np.count_nonzero(maskToProcess)

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
        b_mask = brightest_cluster(diff, curr_mask, total_points, tol=tol, cutoff_percent=cutoff_percent)
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
        mask_directions_list.append(f.get_mask_direction(b_mask))
        mask_percent_list.append(ImageUtils.percentPoints(b_mask, total_points))

        print("effective color: ", effective_color, " brightness: ", round(np.mean(ycrcb_image[:, :, 0][b_mask]),
                                                                           2), "\n")
        # f.show(ImageUtils.plot_points_and_mask(ycrcb_image, [f.noseMiddlePoint], bMask))

        curr_mask = np.bitwise_xor(curr_mask, b_mask)
        if ImageUtils.percentPoints(curr_mask, total_points) < 1:
            break

    print("\nClustering latency: ", time.time() - start_time, " seconds\n")

    final_light_direction = f.process_mask_directions(mask_directions_list, mask_percent_list)
    print("\nFinal Light Direction: ", final_light_direction)

    plot_color(ycrcb_image, all_cluster_masks, total_points)

    start_time = time.time()
    effective_color_map = f.iterate_effectiveColorMap(effective_color_map, all_cluster_masks)
    print("\ncolor map iteration time: ", time.time() - start_time, " seconds\n")
    f.print_effectiveColorMap(effective_color_map, total_points)

    combined_masks = f.combine_masks_close_to_each_other(effective_color_map)

    print("\nCombined masks")
    for m in combined_masks:
        print("percent: ", ImageUtils.percentPoints(m, total_points))
        f.show_mask(m)

    f.show_orig_image()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Image for processing')
    parser.add_argument('--image', required=True, metavar="path to video file")
    parser.add_argument('--bri', required=False, metavar="bri")
    parser.add_argument('--sat', required=False, metavar="sat")
    args = parser.parse_args()

    bri = 1.0
    sat = 1.0
    if args.bri is not None:
        bri = float(args.bri)
    if args.sat is not None:
        sat = float(args.sat)

    analyze(bri, sat)
