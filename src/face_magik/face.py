"""
This file implements a Face class that abstracts
some of the operations around face detection, light estimation
and albedo estimation of a given image.
"""
import os
import numpy as np
import time
import cv2
import h5py
import argparse
import json
import tensorflow as tf
import math
from scipy import ndimage
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from .image_utils import ImageUtils
from .common import (
    EYE_OPEN,
    EYEBALL,
    EYEBROW,
    READING_GLASSES,
    SUNGLASSES,
    EYE_CLOSED,
    NOSE,
    NOSTRIL,
    UPPER_LIP,
    LOWER_LIP,
    TEETH,
    TONGUE,
    FACIAL_HAIR,
    FACE,
    HAIR_ON_HEAD,
    BALD_HEAD,
    EAR,
)
from .common import label_id_map, MaskDirection, LightDirection


class Face:
    CLASS_IDS_KEY = "class_ids"
    MASKS_KEY = "masks"
    ID_LABEL_MAP = {v: k for k, v in label_id_map.items()}
    WINDOW_SIZE = 900

    def __init__(self, image_path="", maskrcnn_model=None, image=None):
        if image is None:
            self.image = ImageUtils.read_rgb_image(image_path)
        else:
            self.image = image

        # Convert from sRGB to display P3 if sRGB profile.
        if image_path != "" and self.is_sRGB_profile(image_path):
            print("image has sRGB profile")
            self.image = ImageUtils.sRGBtodisplayP3Image(self.image)

        self.brightImage = self.to_brightImage(self.image)

        if image is None:
            self.preds = self.detect_face(image_path, maskrcnn_model)
        else:
            print("MAKE PREDICTIONS")
            self.preds = self.make_predictions(maskrcnn_model)

        self.faceMask = self.get_face_mask()
        self.rotMatrix = self.rotation_matrix()
        self.noseMiddlePoint = ImageUtils.mean_coordinate(self.get_nose_keypoints())

        if image_path.startswith("server/data") or image_path.startswith("server/video"):
            # Load lighting, normals and face vertices from JSON file located at image path.
            dirPath, _ = os.path.split(image_path)
            with open(os.path.join(dirPath, "face_vertices.json"), "r") as f:
                self.faceVertices = np.array(json.load(f))
            with open(os.path.join(dirPath, "lighting.json"), "r") as f:
                self.lighting = json.load(f)
            with open(os.path.join(dirPath, "triangle_indices.json"), "r") as f:
                self.triangleIndices = np.array(json.load(f))
            with open(os.path.join(dirPath, "vertex_normals.json"), "r") as f:
                self.vertexNormals = np.array(json.load(f))
                self.compute_vertex_normals()

            # Constants array from Ramamoorthi's paper.
            self.c = np.array([0.429043, 0.511664, 0.743125, 0.886227, 0.247708])

        if image_path != "":
            self.windowName = image_path
        else:
            self.windowName = "image"

    """
    show_gray will show grayscale of RGB image.
    """

    def show_gray(self):
        cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.windowName, Face.WINDOW_SIZE, Face.WINDOW_SIZE)
        cv2.imshow(self.windowName, cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY))
        return cv2.waitKey(0) & 0xFF

    """
    get_profile returns the color profile of the picture.
    Formula found here: https://stackoverflow.com/questions/50641637/identify-colour-space-of-any-image-if-icc-profile-is-empty-pil
    """

    def is_sRGB_profile(self, image_path):
        from PIL import Image

        image = Image.open(image_path)
        exif = image._getexif() or {}
        if exif == {}:
            # Assume that the color profile is sRGB
            return True

        return exif.get(0xA001) == 1 or exif.get(0x0001) == 'R98'

    """
    show_masks will display multiple masks in the image.
    """

    def show_masks(self, masks, colorList=[]):
        if len(colorList) == 0:
            colorList = [ImageUtils.random_color() for c in masks]
        assert len(masks) == len(colorList), "Num masks != num colors"
        clone = self.image.copy()
        for i, mask in enumerate(masks):
            clone[mask] = np.array(colorList[i])
        return self.show(clone)

    """
    show_masks_comb will combine given mask list and show it.
    """

    def show_masks_comb(self, masks):
        if len(masks) == 0:
            return self.show_orig_image()
        res = np.zeros(self.faceMask.shape, dtype=bool)
        for m in masks:
            res = np.bitwise_or(res, m)
        return self.show_mask(res)

    """
    show_mask will display given mask.
    """

    def show_mask(self, mask, color=[0, 255, 0]):
        clone = self.image.copy()
        clone[mask] = np.array(color)
        return self.show(clone)

    """
    show_mask_with_image will display given mask with given image.
    """

    def show_mask_with_image(self, image, mask, color=[0, 255, 0]):
        clone = image.copy()
        clone[mask] = np.array(color)
        return self.show(clone)

    """
    show_skin_tone sets given skin tone (sRGB) to background mask.
    """

    def show_skin_tone(self, skinTone):
        print("\tSkin tone: ", ImageUtils.RGB2HEX(skinTone))
        self.show_masks([self.background_mask()], [skinTone])

    """
    show_orig_image shows the original RGB image without modifications.
    """

    def show_orig_image(self):
        return self.show(self.image)

    """
  show_irrad_mask shows given irradiance mask.
  """

    def show_irrad_mask(self, vertices, E):
        clone = self.image.copy()
        clone[:, :] = [0, 0, 0]
        clone[vertices[:, 0], vertices[:, 1]] = np.clip(E * 100, None, 255).astype(np.uint8)
        return self.show(clone)

    """
    show is an internal helper function to display given RGB image.
    """

    def show(self, image):
        # cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)
        # cv2.resizeWindow(self.windowName, (100, 100))
        # cv2.imshow(self.windowName, cv2.cvtColor(ImageUtils.ResizeWithAspectRatio(image, width=600), cv2.COLOR_RGB2BGR))
        cv2.imshow(self.windowName, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return cv2.waitKey(0) & 0xFF

    """
    detect_face will detect face in the given image and segment out eyes,
    eyebrows, nose, lips etc using a MaskRCNN network. It returns a predictions
    dictionary.
    """

    def detect_face(self, image_path: str, maskrcnn_model) -> list:
        face_mask_dir_path, image_name = os.path.split(image_path)
        hdf5_file_name = os.path.splitext(image_name)[0] + ".hdf5"
        hdf5_full_path = os.path.join(face_mask_dir_path, hdf5_file_name)
        if os.path.exists(hdf5_full_path):
            start_time = time.time()
            preds = {Face.CLASS_IDS_KEY: []}
            with h5py.File(hdf5_full_path, 'r') as f:
                preds[Face.MASKS_KEY] = np.zeros((self.image.shape[0], self.image.shape[1], len(f.keys())), dtype=bool)
                for i, class_id in enumerate(f.keys()):
                    preds[Face.CLASS_IDS_KEY].append(int(class_id.split("-")[0]))
                    preds[Face.MASKS_KEY][:, :, i] = f[class_id][:]
            print("\nReading from Hdf5 file: ", time.time() - start_time, " seconds \n")
            return preds

        print("Running on {}".format(image_path))

        # Detect face.
        preds = self.make_predictions(maskrcnn_model)

        if not os.path.exists(face_mask_dir_path):
            os.makedirs(face_mask_dir_path)

        with h5py.File(hdf5_full_path, 'w') as f:
            for i, class_id in enumerate(preds[Face.CLASS_IDS_KEY]):
                mask = preds[Face.MASKS_KEY][:, :, i]
                dset = f.create_dataset(str(class_id) + "-" + str(i), mask.shape, dtype=bool, chunks=True,
                                        compression="gzip")
                dset[...] = mask
        return preds

    """
    Makes predictions using given Mask RCNN model on image.
    """

    def make_predictions(self, maskrcnn_model) -> list:
        start_time = time.time()
        preds = maskrcnn_model.detect([self.image], verbose=1)[0]
        print("\nModel detection time: ", time.time() - start_time, " seconds\n")
        return preds

    """
    specularity returns the 2D specularity array as shown in Shen et. al. 2009.
    """

    def specularity(self):
        Vmin = np.amin(self.image, axis=-1)
        mu = np.mean(Vmin)
        sigma = np.std(Vmin)
        eta = 0.1
        Tv = mu + eta * sigma
        tau = Vmin.copy()
        tau = np.where(tau > Tv, Tv, tau)
        return Vmin - tau

    """
    compute_specular_masks computes and stores all specular regions on the face  as the dominant
    specular region (based on maximum mean specularity).
    It is 3D array of shape (IMG_WIDTH, IMG_HEIGHT, num regions).
    """

    def compute_specular_masks(self):
        beta = self.specularity()
        bImg = np.bitwise_and(beta != 0, self.faceMask)

        # Label all specular clusters.
        label_im, nb_labels = ndimage.label(bImg)
        sizes = ndimage.sum(bImg, label_im, range(nb_labels + 1))
        indices = np.transpose(np.argwhere(sizes > int(len(np.nonzero(self.faceMask)[0]) / 1000.0)))

        self.allMasks = np.repeat(label_im[:, :, np.newaxis], len(indices), axis=2) == indices
        if self.allMasks.shape[2] == 0:
            # No masks found.
            print("NO Specular Masks found")
            return

        domMask = self.allMasks[:, :, np.argmax(
            np.sum(np.where(self.allMasks, np.repeat(beta[:, :, np.newaxis], len(indices), axis=2), 0), axis=(0, 1)) /
            sizes[indices])]

        # Find surrounding region of given dominant mask.
        contours, _ = cv2.findContours(cv2.threshold((200 * domMask).astype(np.uint8), 127, 255, 0)[1], cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cimg = np.zeros(self.image.shape[:2])
        cv2.drawContours(cimg, contours, 0, color=255, thickness=3)

        self.surrMask = cimg == 255
        self.domMask = np.bitwise_and(np.bitwise_xor(domMask, self.surrMask), domMask)

    """
    evaluate_k calculates the value of constant k (Shen et al) using given dominant and specular masks.
    k is obtained via least squares minimization.
    """

    def evaluate_k(self, domMask, surrMask):
        beta = self.specularity()
        domVals = np.sum(np.where(np.repeat(domMask[:, :, np.newaxis], 3, axis=2), self.image, 0),
                         axis=(0, 1)) / np.count_nonzero(domMask)
        surrVals = np.sum(np.where(np.repeat(surrMask[:, :, np.newaxis], 3, axis=2), self.image, 0),
                          axis=(0, 1)) / np.count_nonzero(surrMask)
        domBetaVals = np.sum(
            np.where(np.repeat(domMask[:, :, np.newaxis], 3, axis=2), np.repeat(beta[:, :, np.newaxis], 3, axis=2), 0),
            axis=(0, 1)) / np.count_nonzero(domMask)
        surrBetaVals = np.sum(
            np.where(np.repeat(surrMask[:, :, np.newaxis], 3, axis=2), np.repeat(beta[:, :, np.newaxis], 3, axis=2), 0),
            axis=(0, 1)) / np.count_nonzero(surrMask)

        fun = lambda k: ((domVals - surrVals) - k * (domBetaVals - surrBetaVals)) @ (
                (domVals - surrVals) - k * (domBetaVals - surrBetaVals))
        res = minimize(fun, [0.6], method="SLSQP")
        return res.x

    """
    remove_specular_highlights removes specular highlights from given image using Shen et al. 2009.
    https://www.researchgate.net/publication/24410295_Simple_and_efficient_method_for_specularity_removal_in_a_image
    It returns the diffuse image as a result.
    """

    def remove_specular_highlights(self):
        self.compute_specular_masks()

        k = self.evaluate_k(self.domMask, self.surrMask)
        print("k: ", k)

        beta = self.specularity()
        return (self.image.copy() - k * np.repeat(beta[:, :, np.newaxis], 3, axis=2)).astype(np.uint8)

    """
    detect_background detects background of given face image.
    """

    def detect_background(self):
        interpreter = tf.lite.Interpreter(
            model_path="background_detection/deeplabv3_1_default_1.tflite")

        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # check the type of the input tensor
        floating_model = input_details[0]['dtype'] == np.float32

        # NxHxWxC, H:1, W:2
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]

        ow, oh = self.image.shape[:2]
        img = cv2.resize(self.image, (width, height))

        # add N dim
        input_data = np.expand_dims(img, axis=0)

        if floating_model:
            input_data = (np.float32(input_data) - 127.5) / 127.5

        interpreter.set_tensor(input_details[0]['index'], input_data)

        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        results = np.squeeze(output_data)

        out = np.apply_along_axis(np.argmax, 2, results)

        bg = np.where(out == 0, 255, 0).astype(np.uint8)
        return cv2.resize(bg, (oh, ow), interpolation=cv2.INTER_NEAREST).astype(np.bool)

    """
    biclustering_Kmeans returns the dominant and surrounding masks of given mask.
    The mask clustering is done using Kmeans of the original mask.
    """

    def biclustering_Kmeans(self, mask):
        clf = KMeans(n_clusters=2)
        labels = clf.fit_predict(self.image[mask])

        flatMask = np.ndarray.flatten(mask)
        indices = np.argwhere(flatMask)

        # Compute masks of both clusters.
        aMask, bMask = flatMask.copy(), flatMask.copy()
        aMask[indices] = np.reshape(labels == 0, (len(labels == 0), 1))
        bMask[indices] = np.reshape(labels == 1, (len(labels == 1), 1))
        aMask = np.reshape(aMask, mask.shape)
        bMask = np.reshape(bMask, mask.shape)

        beta = self.specularity()
        if np.sum(beta[aMask]) / np.count_nonzero(aMask) >= np.sum(beta[bMask]) / np.count_nonzero(bMask):
            return aMask, bMask

        return bMask, aMask

    """
    biclustering_Kmeans returns the dominant and surrounding masks (in that order) of given mask.
    It differs from original implementation in that it calculates dominant based on greater average
    brightness of pixels.
    """

    def biclustering_Kmeans_mod(self, image, mask,
                                func=lambda img, m1, m2: (m1, m2) if np.mean(img[m1][:, 0]) >= np.mean(
                                    img[m2][:, 0]) else (m2, m1)):
        clf = KMeans(n_clusters=2)
        labels = clf.fit_predict(image[mask])

        flatMask = np.ndarray.flatten(mask)
        indices = np.argwhere(flatMask)

        # Compute masks of both clusters.
        aMask, bMask = flatMask.copy(), flatMask.copy()
        aMask[indices] = np.reshape(labels == 0, (len(labels == 0), 1))
        bMask[indices] = np.reshape(labels == 1, (len(labels == 1), 1))
        aMask = np.reshape(aMask, mask.shape)
        bMask = np.reshape(bMask, mask.shape)

        return func(image, aMask, bMask)

    """
    remove_specular_highlights_modified is the modified version of remove specular highlights.
    This algorithm calculates the dffuse image using Shen et al but stores the specular
    region for further processing. The stored specular region can be reused in diffuse calculations.
    """

    def remove_specular_highlights_modified(self):
        self.compute_specular_masks()

        domdomMask, domSurrMask = self.biclustering_Kmeans(self.domMask)

        k = self.evaluate_k(domdomMask, domSurrMask)

        beta = self.specularity()
        self.diffuse = (self.image.copy() - k * np.repeat(beta[:, :, np.newaxis], 3, axis=2)).astype(np.uint8)

        specularMask = np.zeros(self.image.shape[:2], dtype=bool)
        for i in range(self.allMasks.shape[2]):
            # domdomMask, _ = self.biclustering_Kmeans(self.allMasks[:, :, i])
            specularMask = np.bitwise_or(specularMask, self.allMasks[:, :, i])

        self.specularMask = specularMask
        return specularMask

    """
    distinct_colors returns distinct colors from given mask.
    The colors are obtained by repeated subdivision of the mask
    based on hue and saturation.
    """

    def distinct_colors(self, mask, tol=0.05):
        # Divide by hue and sat.
        allHueMasks = self.divide_all_hue(mask, tol=tol)
        # allHueMasks = self.divide_all_brightness(mask, tol=tol)
        allMasks = []
        for m in allHueMasks:
            allMasks += self.divide_all_sat(m, tol=tol)

        # Return distinct colors(medians) from allMasks.
        colors = []
        mset = set()
        for m in allMasks:
            mc = np.median(self.image[m], axis=0)
            if ImageUtils.RGB2HEX(mc) not in mset:
                colors.append(mc)
                mset.add(ImageUtils.RGB2HEX(mc))

        return np.array(colors)

    """
    compute_diffuse will compute the irradiance(E) and diffuse color using the
    normals of the face and spherical harmonic coefficients.
    """

    def compute_diffuse(self):
        if self.lighting == None:
            return

        tVertices = self.allVerts.astype(int)
        self.adjust_SH_coeffs()

        Mred = self.compute_M_matrix(self.lighting["red"])
        Mgreen = self.compute_M_matrix(self.lighting["green"])
        Mblue = self.compute_M_matrix(self.lighting["blue"])

        # Convert to n by 4 array.
        normals = np.append(self.allVertNorms, np.ones((self.allVertNorms.shape[0], 1)), 1)
        Ered = np.einsum('ij,ij->i', normals @ Mred, normals)
        Egreen = np.einsum('ij,ij->i', normals @ Mgreen, normals)
        Eblue = np.einsum('ij,ij->i', normals @ Mblue, normals)
        E = np.column_stack((Ered, Egreen, Eblue))
        self.show_irrad_mask(tVertices, E)

        # Calculate intensity at each vertex point and show.
        clone = self.image.copy()
        intensity = ImageUtils.add_gamma_correction_matrix(
            ImageUtils.remove_gamma_correction_matrix(clone[tVertices[:, 0], tVertices[:, 1]]) * ((1.0) / E))
        clone[:, :] = [255, 255, 255]
        for i in range(intensity.shape[0]):
            cv2.circle(clone, (tVertices[i, 1], tVertices[i, 0]), 1,
                       [int(intensity[i][0]), int(intensity[i][1]), int(intensity[i][2])], 1)
        self.show(clone)

    """
    compute_diffuse_new will compute the irradiance(E) and diffuse color using the
    normals of the face and spherical harmonic coefficients using an alternate
    formula to calculate the irradiance.
    """

    def compute_diffuse_new(self):
        if self.lighting == None:
            return

        tVertices = self.allVerts.astype(int)
        self.adjust_SH_coeffs()

        # Compute harmonic functions.
        normals = self.allVertNorms
        normMatrix = np.column_stack((np.ones((normals.shape[0], 1)), normals[:, 1], normals[:, 2], normals[:, 0],
                                      normals[:, 1] * normals[:, 0], normals[:, 1] * normals[:, 2],
                                      (3 * normals[:, 2] * normals[:, 2]) - 1, normals[:, 2] * normals[:, 0],
                                      (normals[:, 0] * normals[:, 0]) - (normals[:, 1] * normals[:, 1])))

        Ered = normMatrix @ self.lighting["red"]
        Egreen = normMatrix @ self.lighting["green"]
        Eblue = normMatrix @ self.lighting["blue"]
        E = np.column_stack((Ered, Egreen, Eblue))
        self.show_irrad_mask(tVertices, E)

        # Calculate intensity at each vertex point and show.
        print("E, verts: ", E.shape[0], tVertices.shape[0])
        clone = self.image.copy()
        intensity = ImageUtils.add_gamma_correction_matrix(
            ImageUtils.remove_gamma_correction_matrix(clone[tVertices[:, 0], tVertices[:, 1]]) * ((1.0) / E))
        clone[:, :] = [255, 255, 255]
        for i in range(intensity.shape[0]):
            cv2.circle(clone, (tVertices[i, 1], tVertices[i, 0]), 1,
                       [int(intensity[i][0]), int(intensity[i][1]), int(intensity[i][2])], 1)
        self.show(clone)

    """
    adjust_SH_coeffs adjusts the signs of SH coefficients provided by ARKit depending
    on the sign x, y and z components of the face normals.
    """

    def adjust_SH_coeffs(self):
        normals = self.allVertNorms

        cntNormals = normals.shape[0]
        self.cntXpos = np.count_nonzero(normals[:, 0] >= 0)
        self.cntXneg = cntNormals - self.cntXpos
        self.cntYpos = np.count_nonzero(normals[:, 1] >= 0)
        self.cntYneg = cntNormals - self.cntYpos
        self.cntZpos = np.count_nonzero(normals[:, 2] >= 0)
        self.cntZneg = cntNormals - self.cntZpos
        print("X pos cnt: ", self.cntXpos, " X neg cnt: ", self.cntXneg)
        print("Y pos cnt: ", self.cntYpos, " Y neg cnt: ", self.cntYneg)
        print("Z pos cnt: ", self.cntZpos, " Z neg cnt: ", self.cntZneg)

        self.reverseXZ = False
        if abs(self.cntYpos - self.cntYneg) > abs(self.cntZpos - self.cntZneg):
            self.reverseXZ = True
            print("reversed X and Z")

        self.show_positive_normals()

        self.adjust_SH_coeffs_helper(self.lighting["red"])
        self.adjust_SH_coeffs_helper(self.lighting["green"])
        self.adjust_SH_coeffs_helper(self.lighting["blue"])

    """
    show_positive_normals displays positive normals mask in X, Y and Z directions.
    """

    def show_positive_normals(self):
        normals = self.allVertNorms
        verts = self.allVerts.astype(int)

        # X direction.
        clone = self.image.copy()
        vmask = verts[normals[:, 0] >= 0]
        clone[vmask[:, 0], vmask[:, 1]] = [255, 0, 0]
        self.show(clone)

        # Y direction.
        clone = self.image.copy()
        vmask = verts[normals[:, 1] >= 0]
        clone[vmask[:, 0], vmask[:, 1]] = [0, 255, 0]
        self.show(clone)

        # Z direction.
        clone = self.image.copy()
        vmask = verts[normals[:, 2] >= 0]
        clone[vmask[:, 0], vmask[:, 1]] = [0, 0, 255]
        self.show(clone)

    """
    adjust_SH_coeffs helper will adjust coefficients
    for given channel (red, blue or green) based on whether
    X,Y and Z are flipped or not.
    """

    def adjust_SH_coeffs_helper(self, L):
        if self.reverseXZ:
            flipZ = True
            flipY = True
            flipX = True if ((self.cntXpos >= self.cntXneg and L[3] < 0) or (
                    self.cntXpos < self.cntXneg and L[3] >= 0)) else False
        else:
            flipZ = True if ((self.cntZpos >= self.cntZneg and L[2] < 0) or (
                    self.cntZpos < self.cntZneg and L[2] >= 0)) else False
            flipZ = True
            flipY = True
            flipX = True

        print("flipX: ", flipX, " flipY: ", flipY, " flipZ: ", flipZ)
        L[1] = -L[1] if flipY else L[1]
        L[2] = -L[2] if flipZ else L[2]
        L[3] = -L[3] if flipX else L[3]
        L[4] = L[4] if flipY == flipX else -L[4]
        L[5] = L[5] if flipY == flipZ else -L[5]
        L[7] = L[7] if flipX == flipZ else -L[7]

    """
    compute_M_matrix computes the M matrix specified in Ramamoorthi's paper.
    The M matrix is multiplied with normalized normal vector at each vertex
    to obtain the irradiance at the vertex. We need to flip spherical harmonic
    values of x and z directions based on observation.
    """

    def compute_M_matrix(self, L):
        c = self.c
        return np.array([
            [c[0] * L[8], c[0] * L[4], c[0] * L[7], c[1] * L[3]], [c[0] * L[4], -c[0] * L[8], c[0] * L[5], c[1] * L[1]],
            [c[0] * L[7], c[0] * L[5], c[2] * L[6], c[1] * L[2]],
            [c[1] * L[3], c[1] * L[1], c[1] * L[2], c[3] * L[0] - c[4] * L[6]],
        ])

    """
    compute_vertex_normals computes all vertices and their normals
    from given mesh vertices (and corresponding normals).
    """

    def compute_vertex_normals(self):
        faceVertices = self.faceVertices
        triangleIndices = np.reshape(self.triangleIndices, (-1, 3))
        triangle2DArr = faceVertices[triangleIndices]
        vertexNormals = self.vertexNormals[triangleIndices]
        allAreas = self.compute_areas(triangle2DArr)

        minxarr = np.min(triangle2DArr[:, :, 0], axis=1)
        maxxarr = np.max(triangle2DArr[:, :, 0], axis=1)
        minyarr = np.min(triangle2DArr[:, :, 1], axis=1)
        maxyarr = np.max(triangle2DArr[:, :, 1], axis=1)

        self.allVerts = np.zeros((0, 2))
        self.allVertNorms = np.zeros((0, 3))
        for i in range(triangle2DArr.shape[0]):
            bbox = np.mgrid[minxarr[i]:maxxarr[i], minyarr[i]:maxyarr[i]].reshape(2, -1).T
            v0, v1, v2 = triangle2DArr[i][0], triangle2DArr[i][1], triangle2DArr[i][2]

            bLen = bbox.shape[0]
            areas0 = self.compute_areas(
                np.stack((np.repeat(v1[np.newaxis, :], bLen, axis=0), np.repeat(v2[np.newaxis, :], bLen, axis=0), bbox),
                         axis=1))
            areas1 = self.compute_areas(
                np.stack((np.repeat(v2[np.newaxis, :], bLen, axis=0), np.repeat(v0[np.newaxis, :], bLen, axis=0), bbox),
                         axis=1))
            areas2 = self.compute_areas(
                np.stack((np.repeat(v0[np.newaxis, :], bLen, axis=0), np.repeat(v1[np.newaxis, :], bLen, axis=0), bbox),
                         axis=1))

            areas = np.stack((areas0, areas1, areas2), axis=1)
            tol = 0.001
            masks = np.stack((areas0 >= tol, areas1 >= tol, areas2 >= tol), axis=1)

            idxArr = np.where(masks.all(axis=1))[0]
            w = areas[idxArr] / allAreas[i]

            # n by 3 where n is len(idxArr)
            vNorms = w @ vertexNormals[i]
            verts = bbox[idxArr]
            if verts.shape[0] == 0:
                continue

            self.allVerts = np.concatenate((self.allVerts, verts), axis=0)
            self.allVertNorms = np.concatenate((self.allVertNorms, vNorms), axis=0)

    """
    compute_areas will compute deteminants of given array of triangle(each triangle is represented by 3 vertices).
    The input shape of triangles is (n, 3, 2) and the points are in counter clockwise order.
    The out is an array of areas of dimension (n, 1).
    """

    def compute_areas(self, a):
        # Get vector array from triangle vertex array.
        return np.linalg.det(np.stack((a[:, 1] - a[:, 0], a[:, 2] - a[:, 0]), axis=1))

    """
    get_face_mask returns face mask numpy array from stored dictionary of face predictions.
    """

    def get_face_mask(self):
        faceMask = np.zeros(self.image.shape[:2], dtype=bool)
        faceFound = False
        for i, class_id in enumerate(self.preds[Face.CLASS_IDS_KEY]):
            if Face.ID_LABEL_MAP[class_id] == EYE_OPEN or Face.ID_LABEL_MAP[class_id] == EYE_CLOSED \
                    or Face.ID_LABEL_MAP[class_id] == NOSTRIL or Face.ID_LABEL_MAP[class_id] == TEETH \
                    or Face.ID_LABEL_MAP[class_id] == TONGUE or Face.ID_LABEL_MAP[class_id] == UPPER_LIP \
                    or Face.ID_LABEL_MAP[class_id] == LOWER_LIP or Face.ID_LABEL_MAP[class_id] == SUNGLASSES \
                    or Face.ID_LABEL_MAP[class_id] == EYEBROW or Face.ID_LABEL_MAP[class_id] == FACE:
                faceMask = np.bitwise_xor(faceMask, self.preds[Face.MASKS_KEY][:, :, i])
                if Face.ID_LABEL_MAP[class_id] == FACE:
                    if faceFound:
                        raise Exception("Only 1 face allowed per image")
                    faceFound = True
        if not faceFound:
            raise Exception("Face not found in image")

        return faceMask

    """
    get_face_keypoints returns keypoints of the face that are best representatives
    of skin tone on the face (forehead, cheeks and nose).
    """

    def get_face_keypoints(self):
        keypoints = np.zeros(self.faceMask.shape, dtype=bool)
        for m in [self.get_forehead_points(), self.get_left_cheek_keypoints(), self.get_right_cheek_keypoints(),
                  self.get_nose_keypoints()]:
            keypoints = np.bitwise_or(keypoints, m)
        return keypoints

    """
    get_face_until_nose_end returns keypoints of the face until (and including) the nose end.
    The idea is to not include points below the nose in case there is a beard.
    """

    def get_face_until_nose_end(self):
        faceMask = self.get_face_mask()

        noseMasks = self.get_attr_masks(NOSE)
        assert len(noseMasks) == 1, "Want 1 mask for nose!"
        noseRowMin, _, _, noseHeight = self.bbox(noseMasks[0])
        faceMask[noseRowMin + noseHeight:, :] = False
        return faceMask

    """
    get_face_mask_without_area_around_eyes returns mask of face without area under and above
    eyes to reduce noise in the skin tone algorithm.
    """

    def get_face_mask_without_area_around_eyes(self):
        faceMask = self.get_face_mask()
        leftEyeAreaMask = self.get_area_around_left_eye()
        rightEyeAreaMask = self.get_area_around_right_eye()

        return np.bitwise_and(faceMask, np.logical_not(np.bitwise_or(leftEyeAreaMask, rightEyeAreaMask)))

    """
    get_face_until_nose_end_without_area_around_eyes returns mask of face without area under and above
    eyes to reduce noise in the skin tone algorithm.
    """

    def get_face_until_nose_end_without_area_around_eyes(self):
        faceMask = self.get_face_until_nose_end()
        leftEyeAreaMask = self.get_area_around_left_eye()
        rightEyeAreaMask = self.get_area_around_right_eye()

        return np.bitwise_and(faceMask, np.logical_not(np.bitwise_or(leftEyeAreaMask, rightEyeAreaMask)))

    """
    get_eye_white_brightness returns eye white brightness that indicates brightness
    of image.
    """

    def get_eye_white_brightness(self):
        whiteMask = self.get_eye_white_points()
        medoids, allMasks, minCost = ImageUtils.best_clusters(self.distinct_colors(whiteMask, tol=0.005), self.image,
                                                              whiteMask, 3, numIters=100)
        maxBrightness = 0
        for m in allMasks:
            print("Mean brightness: ", np.mean(self.brightImage[m[0]], axis=0)[2] * (100.0 / 255.0), "percent: ",
                  (np.count_nonzero(m[0]) / np.count_nonzero(whiteMask)) * 100.0)
            maxBrightness = max(maxBrightness, np.mean(self.brightImage[m[0]], axis=0)[2] * (100.0 / 255.0))
            self.show_mask(m[0])

        return maxBrightness

    """
    get_left_cheek_keypoints returns keypoints from left cheek that are best representatives of skin
    tone on the face.
    """

    def get_left_cheek_keypoints(self):
        eye_masks = self.get_attr_masks(EYE_OPEN)
        face_mask = self.get_attr_masks(FACE)[0]
        nose_mask = self.get_attr_masks(NOSE)
        assert len(eye_masks) == 2, "Want 2 masks for eye open!"
        assert len(nose_mask) == 1, "Want 1 mask for nose!"

        nose_mask = nose_mask[0]
        left_eye_mask = eye_masks[0] if self.bbox(eye_masks[0])[1] <= self.bbox(eye_masks[1])[1] else eye_masks[1]
        nose_row_min, nose_col_min, nose_width, nose_height = self.bbox(nose_mask)
        left_eye_row_min, left_eye_col_min, left_eye_width, left_eye_height = self.bbox(left_eye_mask)

        # row_min = left_eye_row_min+left_eye_height
        row_min = left_eye_row_min + 2 * left_eye_height
        row_max = nose_row_min + nose_height
        col_min = max(np.nonzero(face_mask[row_min, :])[0][0], np.nonzero(face_mask[row_max, :])[0][0])
        col_max = min(nose_col_min, left_eye_col_min + left_eye_width)
        height = row_max - row_min + 1
        # row_max = row_max - int(height/4)

        keypoints = self.faceMask.copy()
        keypoints[:, col_max - 10:] = False
        keypoints[:row_min + 20, :] = False
        keypoints[row_max:, :] = False
        return keypoints

    """
    get_right_cheek_keypoints returns keypoints from right cheek that are best representatives of skin
    tone on the face.
    """

    def get_right_cheek_keypoints(self):
        eye_masks = self.get_attr_masks(EYE_OPEN)
        face_mask = self.get_attr_masks(FACE)[0]
        nose_mask = self.get_attr_masks(NOSE)
        assert len(eye_masks) == 2, "Want 2 masks for eye open!"
        assert len(nose_mask) == 1, "Want 1 mask for nose!"

        nose_mask = nose_mask[0]
        right_eye_mask = eye_masks[0] if self.bbox(eye_masks[0])[1] >= self.bbox(eye_masks[1])[1] else eye_masks[1]
        nose_row_min, nose_col_min, nose_width, nose_height = self.bbox(nose_mask)
        right_eye_row_min, right_eye_col_min, right_eye_width, right_eye_height = self.bbox(right_eye_mask)

        # row_min = right_eye_row_min+right_eye_height
        row_min = right_eye_row_min + 2 * right_eye_height
        row_max = nose_row_min + nose_height
        col_min = max(nose_col_min + nose_width, right_eye_col_min)
        col_max = min(np.nonzero(face_mask[row_min, :])[0][-1], np.nonzero(face_mask[row_max, :])[0][-1])
        height = row_max - row_min + 1
        # row_max = row_max - int(height/5)

        keypoints = self.faceMask.copy()
        keypoints[:, :col_min + 10] = False
        keypoints[:row_min + 20, :] = False
        keypoints[row_max:, :] = False
        return keypoints

    """
    get_nose_keypoints returns keypoints from nose that are best representatives of skin tone on the face.
    """

    def get_nose_keypoints(self):
        nose_masks = self.get_attr_masks(NOSE)
        nostril_masks = self.get_attr_masks(NOSTRIL)
        assert len(nose_masks) == 1, "Want 1 mask for nose!"
        if len(nostril_masks) == 0:
            nose_mask = nose_masks[0]
            row, _, _, h = self.bbox(nose_mask)
            nose_mask[row + h - 10:, :] = False
            return nose_mask

        row_max = 10 ** 10
        for mask in nostril_masks:
            row, _, _, _ = self.bbox(mask)
            if row < row_max:
                row_max = row
        nose_mask = nose_masks[0]
        nose_mask[row_max - 10:, :] = 0
        return nose_mask

    """
    get_left_nose_points returns left half points on the nose.
    """

    def get_left_nose_points(self):
        nsmask = self.get_nose_keypoints().copy()
        _, cmin, w, _ = self.bbox(nsmask)
        xmid = int(cmin + float(w) / 2)
        nsmask[:, xmid:] = False
        return nsmask

    """
    get_right_nose_points returns right half points on the nose.
    """

    def get_right_nose_points(self):
        nsmask = self.get_nose_keypoints().copy()
        _, cmin, w, _ = self.bbox(nsmask)
        xmid = int(cmin + float(w) / 2)
        nsmask[:, :xmid] = False
        return nsmask

    """
    does_facial_hair_exist checks if facial hair exists on face. Doensn't work
    reliably for stubble/lighter beards.
    """

    def does_facial_hair_exist(self):
        return len(self.get_attr_masks(FACIAL_HAIR)) > 0

    """
    get_area_around_eyes returns mask of area around eyes.
    """

    def get_area_around_eyes(self):
        eyeMasks = self.get_attr_masks(EYE_OPEN)
        eyebrowMasks = self.get_attr_masks(LE)
        assert len(eyeMasks) == 2, "Want 2 masks for eyes!"

    """
    get_area_around_left_eye returns mask of area around left eye.
    """

    def get_area_around_left_eye(self):
        eyeMasks = self.get_attr_masks(EYE_OPEN)
        eyebrowMasks = self.get_attr_masks(EYEBROW)
        assert len(eyeMasks) == 2, "Want 2 masks for eyes!"
        assert len(eyebrowMasks) == 2, "Want 2 masks for eyebrows!"

        leftEyeMask = eyeMasks[0] if self.bbox(eyeMasks[0])[1] <= self.bbox(eyeMasks[1])[1] else eyeMasks[1]
        leftEyebrowMask = eyebrowMasks[0] if self.bbox(eyebrowMasks[0])[1] <= self.bbox(eyebrowMasks[1])[1] else \
            eyebrowMasks[1]

        eyeRowMin, eyeColMin, eyeWidth, eyeHeight = self.bbox(leftEyeMask)
        eyebrowRowMin, eyebrowColMin, eyebrowWidth, eyebrowHeight = self.bbox(leftEyebrowMask)

        # Bounding box.
        rowMin = eyebrowRowMin + eyebrowHeight / 2
        rowMax = eyeRowMin + eyeHeight + 3 * eyeHeight / 4
        colMin = eyeColMin - 3 * eyeHeight / 4
        colMax = eyeColMin + eyeWidth + eyeHeight / 2

        finalMask = np.zeros(self.faceMask.shape, dtype=bool)
        finalMask[int(rowMin):int(rowMax) + 1, int(colMin):int(colMax) + 1] = True
        return finalMask

    """
    get_area_around_right_eye returns mask of area around right eye.
    """

    def get_area_around_right_eye(self):
        eyeMasks = self.get_attr_masks(EYE_OPEN)
        eyebrowMasks = self.get_attr_masks(EYEBROW)
        assert len(eyeMasks) == 2, "Want 2 masks for eyes!"
        assert len(eyebrowMasks) == 2, "Want 2 masks for eyebrows!"

        rightEyeMask = eyeMasks[0] if self.bbox(eyeMasks[0])[1] >= self.bbox(eyeMasks[1])[1] else eyeMasks[1]
        rightEyebrowMask = eyebrowMasks[0] if self.bbox(eyebrowMasks[0])[1] >= self.bbox(eyebrowMasks[1])[1] else \
            eyebrowMasks[1]

        eyeRowMin, eyeColMin, eyeWidth, eyeHeight = self.bbox(rightEyeMask)
        eyebrowRowMin, eyebrowColMin, eyebrowWidth, eyebrowHeight = self.bbox(rightEyebrowMask)

        # Bounding box.
        rowMin = eyebrowRowMin + eyebrowHeight / 2
        rowMax = eyeRowMin + eyeHeight + 3 * eyeHeight / 4
        colMin = eyeColMin - eyeHeight / 2
        colMax = eyeColMin + eyeWidth + 3 * eyeHeight / 4

        finalMask = np.zeros(self.faceMask.shape, dtype=bool)
        finalMask[int(rowMin):int(rowMax) + 1, int(colMin):int(colMax) + 1] = True
        return finalMask

    """
    get_neck_points returns some points of the neck of the person.
    """

    def get_neck_points(self):
        faceMasks = self.get_attr_masks(FACE)
        hairMasks = self.get_attr_masks(HAIR_ON_HEAD)
        eyeMasks = self.get_attr_masks(EYE_OPEN)
        assert len(faceMasks) == 1, "Want 1 mask for face!"
        assert len(eyeMasks) == 2, "Want 2 masks for eye!"

        # Use facemask to determine maximum height of neck mask.
        # Use eyes to restrict width of neck mask.
        faceMask = faceMasks[0]
        rmin, _, _, h = self.bbox(faceMasks[0])
        _, c1, w1, _ = self.bbox(eyeMasks[0])
        _, c2, w2, _ = self.bbox(eyeMasks[1])
        w1, w2 = int(w1 / 2), int(w2 / 2)

        mask = np.zeros(faceMask.shape, dtype=bool)
        mask[rmin + int(0.9 * h):rmin + int(1.2 * h), min(c1 + w1, c2 + w2):max(c1 + w1, c2 + w2)] = True
        mask = np.bitwise_xor(np.bitwise_and(mask, faceMask), mask)
        return mask

    """
    get_forehead_points returns points on the forehead.
    """

    def get_forehead_points(self):
        faceMasks = self.get_attr_masks(FACE)
        assert len(faceMasks) == 1, "Want 1 mask for face!"
        faceMask = faceMasks[0]
        eyebrowMasks = self.get_attr_masks(EYEBROW)
        assert len(eyebrowMasks) > 0, "Want atleast 1 mask for eyebrows"

        mask = faceMask.copy()
        for ebMask in eyebrowMasks:
            rmin, _, _, h = self.bbox(ebMask)
            mask[rmin:, :] = False

            # rmax = rmin + h
            # mask[rmax:,:] = False
            # mask = np.bitwise_xor(mask, ebMask)

        rmin, cmin, w, _ = self.bbox(faceMask)
        # delta_pixels = 30
        delta_pixels = 0
        mask[:, :cmin + delta_pixels] = False
        mask[:, cmin + w - delta_pixels:] = False
        mask[:rmin + 10, :] = False
        return mask

    """
    Returns the rotation matrix to transform from original coordinates to one that is perpendicular to the line 
    segment joining the eyes.
    """

    def rotation_matrix(self):
        eyeMasks = self.get_attr_masks(EYE_OPEN)
        assert len(eyeMasks) == 2, "Want 2 masks for eye!"

        # Eye line.
        leftEyeMask = eyeMasks[0] if self.bbox(eyeMasks[0])[1] <= self.bbox(eyeMasks[1])[1] else eyeMasks[1]
        rightEyeMask = eyeMasks[0] if self.bbox(eyeMasks[0])[1] > self.bbox(eyeMasks[1])[1] else eyeMasks[1]

        leftEyeCords = np.argwhere(leftEyeMask)
        xMaxIndex = np.argmax(leftEyeCords, axis=0)[1]
        leftMaxCord = leftEyeCords[xMaxIndex]

        rightEyeCords = np.argwhere(rightEyeMask)
        xMinIndex = np.argmin(rightEyeCords, axis=0)[1]
        rightMinCord = rightEyeCords[xMinIndex]

        leftY, leftX = leftMaxCord[0], leftMaxCord[1]
        rightY, rightX = rightMinCord[0], rightMinCord[1]
        theta = math.atan(float(leftY - rightY) / float(rightX - leftX))
        rotMatrix = np.array([[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]])

        return rotMatrix

    """
    Class encapsulating face mask results and other aspects of input image.
    """

    @staticmethod
    def get_light_direction(mask_direction_list: list, percent_per_direction) -> LightDirection:

        # From the mask results, find the light direction.
        # The algorithm determines direction of the light based on percentPerDirection
        # and order of mask directions in the maskDirectionList.
        start_direction = mask_direction_list[0]
        end_direction = MaskDirection.CENTER
        if start_direction == MaskDirection.LEFT and MaskDirection.RIGHT in percent_per_direction:
            end_direction = MaskDirection.RIGHT
        elif start_direction == MaskDirection.RIGHT and MaskDirection.LEFT in percent_per_direction:
            end_direction = MaskDirection.LEFT
        elif start_direction == MaskDirection.CENTER:
            # Find the first non-CENTER direction while iterating from the end.
            end_direction = MaskDirection.CENTER
            for i in range(len(mask_direction_list) - 1, -1, -1):
                direction = mask_direction_list[i]
                if direction != MaskDirection.CENTER:
                    end_direction = direction
                    break

        print("star direction: ", start_direction)
        print("end direction: ", end_direction)
        left_percent = 0 if MaskDirection.LEFT not in percent_per_direction else percent_per_direction[
            MaskDirection.LEFT]
        center_percent = 0 if MaskDirection.CENTER not in percent_per_direction else percent_per_direction[
            MaskDirection.CENTER]
        right_percent = 0 if MaskDirection.RIGHT not in percent_per_direction else percent_per_direction[
            MaskDirection.RIGHT]

        max_percent = max(left_percent, center_percent, right_percent)
        if max_percent == center_percent:
            # Light is predominantly in the direction or behind the person.
            if start_direction == MaskDirection.CENTER:
                if end_direction == MaskDirection.CENTER:
                    return LightDirection.CENTER
                return LightDirection.CENTER_LEFT if end_direction == MaskDirection.LEFT else \
                    LightDirection.CENTER_RIGHT

            if start_direction == MaskDirection.LEFT:
                if end_direction == MaskDirection.CENTER:
                    return LightDirection.LEFT_CENTER

                if max_percent >= 50:
                    return LightDirection.LEFT_CENTER_RIGHT

                return LightDirection.LEFT_TO_RIGHT

            # Start direction is right.
            if end_direction == MaskDirection.CENTER:
                return LightDirection.RIGHT_CENTER

            if max_percent >= 50:
                return LightDirection.RIGHT_CENTER_LEFT

            return LightDirection.RIGHT_TO_LEFT

        if start_direction == MaskDirection.LEFT:
            return LightDirection.LEFT_CENTER if end_direction == MaskDirection.CENTER else LightDirection.LEFT_TO_RIGHT

        return LightDirection.RIGHT_CENTER if end_direction == MaskDirection.CENTER else LightDirection.RIGHT_TO_LEFT

    """
    Processes sorted list of mask directions and maskPercentList and returns the light direction. maskPercentList is 
    the list of percent count of each mask in the maskDirectionsList. Both elements should be in the same order.
    """

    @staticmethod
    def process_mask_directions(mask_directions_list: list, mask_percent_list: list) -> LightDirection:
        assert len(mask_directions_list) > 0, "Mask directions list cannot be empty!"
        assert len(mask_percent_list) == len(
            mask_directions_list), "Mask Percent list and direction list should be the same length!"

        # Coalesce consecutive masks in the same direction. Count the mask
        # percent for each direction and store in a map. Skip coalesced masks
        # from the final result that are less than 5% in size.
        combined_mask_direction_list = []
        percent_per_direction: dict[MaskDirection, ] = {}
        min_percent = 3
        direction = mask_directions_list[0]
        total_percent = 0
        for i in range(1, len(mask_directions_list)):
            if mask_directions_list[i] == direction:
                total_percent += mask_percent_list[i]
                continue

            if total_percent >= min_percent:
                combined_mask_direction_list.append(direction)
                if direction not in percent_per_direction:
                    percent_per_direction[direction] = round(total_percent)
                else:
                    percent_per_direction[direction] += round(total_percent)

            direction = mask_directions_list[i]
            total_percent = mask_percent_list[i]

        if total_percent >= min_percent:
            combined_mask_direction_list.append(direction)
            if direction not in percent_per_direction:
                percent_per_direction[direction] = round(total_percent)
            else:
                percent_per_direction[direction] += round(total_percent)

        print("percentPerDirection: ", percent_per_direction)

        return Face.get_light_direction(combined_mask_direction_list, percent_per_direction)

    """
    get_left_forehead_points returns left half points on the forehead.
    """

    def get_left_forehead_points(self):
        fhmask = self.get_forehead_points().copy()
        _, cmin, w, _ = self.bbox(fhmask)
        xmid = int(cmin + float(w) / 2)
        fhmask[:, xmid:] = False
        return fhmask

    """
    get_right_forehead_points returns right half points on the forehead.
    """

    def get_right_forehead_points(self):
        fhmask = self.get_forehead_points().copy()
        _, cmin, w, _ = self.bbox(fhmask)
        xmid = int(cmin + float(w) / 2)
        fhmask[:, :xmid] = False
        return fhmask

    """
    get_eye_white_points returns mask of eye whites.
    """

    def get_eye_white_points(self):
        eMask = np.zeros(self.faceMask.shape, dtype=bool)
        ebMasks = self.get_attr_masks(EYEBALL)
        eyMasks = self.get_attr_masks(EYE_OPEN)
        assert len(ebMasks) == 2, "Want 2 eyeball masks"
        assert len(eyMasks) == 2, "Want 2 eye masks"
        for m in eyMasks:
            eMask = np.bitwise_or(eMask, m)
        for m in ebMasks:
            eMask = np.bitwise_xor(eMask, m)
        return eMask

    """
    Return true if teeth are visible in image and false otherwise.
    """

    def is_teeth_visible(self):
        teeth_masks = self.get_attr_masks(TEETH)
        return len(teeth_masks) > 0

    """
    Returns mask of mouth including teeth and tongue and excluding upper and lower lips.
    """

    def get_mouth_points(self):
        ulip_masks = self.get_attr_masks(UPPER_LIP)
        llip_masks = self.get_attr_masks(LOWER_LIP)
        assert len(ulip_masks) == 1, "Want 1 upper lip mask"
        assert len(llip_masks) == 1, "Want 1 lower lip mask"
        ulip_mask = ulip_masks[0]
        llip_mask = llip_masks[0]
        rmin_ulip, cmin_ulip, w_ulip, h_ulip = self.bbox(ulip_mask)
        rmin_llip, cmin_llip, w_llip, h_llip = self.bbox(llip_mask)

        # Find mask that encompasses lips and mouth. Determine (ymin, ymax) for each x.
        lips_and_mouth_mask = np.zeros(self.faceMask.shape, dtype=bool)
        cmin = min(cmin_ulip, cmin_llip)
        cmax = max(cmin_ulip + w_ulip, cmin_llip + w_llip) + 1
        rmax_ulip = rmin_ulip + h_ulip
        rmax_llip = rmin_llip + h_llip
        for x in range(cmin, cmax):
            # Upper lip ymin point.
            ymin = rmin_ulip
            while ymin <= rmax_ulip and not ulip_mask[ymin, x]:
                ymin = ymin + 1
            if ymin > rmax_ulip:
                continue
            ymax = rmax_llip
            while ymax >= rmin_llip and not llip_mask[ymax, x]:
                ymax = ymax - 1
            if ymax < rmin_llip:
                continue

            lips_and_mouth_mask[ymin:ymax + 1, x] = True

        # Find mouth mask by excluding lips from "lips and mouth" mask.
        return np.bitwise_xor(lips_and_mouth_mask,
                                   np.bitwise_and(lips_and_mouth_mask, np.bitwise_or(ulip_mask, llip_mask)))

    """
    Prints the given effective color map for debugging.
    """

    def print_effective_color_map(self, effective_color_map, total_points):
        print("\nEffective Color Map: ")
        ycrcb_image = ImageUtils.to_YCrCb(self.image)
        prev_mask = np.zeros(ycrcb_image.shape[:2], dtype=bool)
        sorted_effective_color_map = sorted(effective_color_map, key=lambda h: 255.0 - np.mean(
            self.to_brightImage(self.image)[effective_color_map[h]], axis=0)[2])
        for mHue in sorted_effective_color_map:
            comb_mask = effective_color_map[mHue]
            prev_delta_cie = 0 if np.count_nonzero(prev_mask) == 0 else ImageUtils.delta_cie2000(
                np.mean(ycrcb_image[comb_mask], axis=0), np.mean(ycrcb_image[prev_mask], axis=0))

            print("\npercent: ", ImageUtils.percentPoints(comb_mask, total_points), "Munsell hue: ", mHue,
                  " Musell sat: ",
                  round(np.mean(self.to_satImage(ycrcb_image)[comb_mask], axis=0)[1] * (100.0 / 255.0), 2),
                  " brightness: ", round(np.mean(np.max(self.image, axis=2)[comb_mask]), 2), " mean + std: ",
                  round(np.mean(np.max(self.image, axis=2)[comb_mask]) + np.std(np.max(self.image, axis=2)[comb_mask]),
                        2), " hue: ", round(ImageUtils.sRGBtoHSV(np.mean(self.image[comb_mask], axis=0))[0, 0] * 2, 2),
                  " sat: ", round(np.mean(self.to_satImage(self.image)[comb_mask], axis=0)[1] * (100.0 / 255.0), 2),
                  " red: ", round(np.mean(self.image[comb_mask][:, 0]), 2), " green: ",
                  round(np.mean(self.image[comb_mask][:, 1]), 2), " blue: ",
                  round(np.mean(self.image[comb_mask][:, 2]), 2), " prev delta: ", round(prev_delta_cie, 2))

            self.show_mask_with_image(ycrcb_image, comb_mask)
            prev_mask = comb_mask

    """
    Iterate effectiveColorMap to make each cluster for each color to be more accurate. Each mask's delta_cie is 
    compared with the cluster's color to determine it's new cluster (based on smallest delta_cie).
    """

    def iterate_effective_color_map(self, effective_color_map: dict, all_cluster_masks: list):
        start_time = time.time()
        ycrcb_image = ImageUtils.to_YCrCb(self.image)
        result_color_map = effective_color_map.copy()
        for i in range(5):
            temp_color_map = {}
            for m in all_cluster_masks:
                min_dte = 1000.0
                best_hue = ""
                for mHue in result_color_map:
                    cb_mask = result_color_map[mHue]
                    dte = ImageUtils.delta_cie2000(np.mean(ycrcb_image[cb_mask], axis=0), np.mean(ycrcb_image[m],
                                                                                                  axis=0))
                    if dte < min_dte:
                        min_dte = dte
                        best_hue = mHue

                if best_hue not in temp_color_map:
                    temp_color_map[best_hue] = m
                else:
                    temp_color_map[best_hue] = np.bitwise_or(temp_color_map[best_hue], m)
            result_color_map = temp_color_map.copy()

        print("\ncolor map iteration time: ", time.time() - start_time, " seconds\n")
        return result_color_map

    """
    Combine masks that are close to each other and may have delta_cie small enough that they can be merged in a 
    single cluster. Returns the new list of masks.
    """

    def combine_masks_close_to_each_other(self, effective_color_map):
        start_time = time.time()
        still_rem = True
        temp_close_masks = sorted([effective_color_map[e] for e in effective_color_map],
                                  key=lambda m: 255.0 - np.mean(self.to_brightImage(self.image)[m], axis=0)[2])
        ycrcb_image = ImageUtils.to_YCrCb(self.image)
        while still_rem:
            still_rem = False
            for i in range(1, len(temp_close_masks)):
                prev_delta_cie = ImageUtils.delta_cie2000(np.mean(ycrcb_image[temp_close_masks[i]], axis=0),
                                                          np.mean(ycrcb_image[temp_close_masks[i - 1]], axis=0))
                next_delta_cie = 100 if i == len(temp_close_masks) - 1 else ImageUtils.delta_cie2000(
                    np.mean(ycrcb_image[temp_close_masks[i]], axis=0), np.mean(ycrcb_image[temp_close_masks[i + 1]],
                                                                               axis=0))
                if prev_delta_cie >= 5 or next_delta_cie < prev_delta_cie:
                    # Check merge in the next index.
                    continue

                # Merge with previous mask.
                now_masks = []
                for j in range(i - 1):
                    now_masks.append(temp_close_masks[j])
                now_masks.append(np.bitwise_or(temp_close_masks[i - 1], temp_close_masks[i]))
                for j in range(i + 1, len(temp_close_masks)):
                    now_masks.append(temp_close_masks[j])
                temp_close_masks = now_masks.copy()
                still_rem = True
                break

        print("\n combining masks time: ", time.time() - start_time, " seconds\n")
        return temp_close_masks

    """
    good_face_points returns a mask containing points that are usually good to sample for skin tone. It excludes 
    points near the eyes and lips.
    """

    def good_face_points(self):
        faceMasks = self.get_attr_masks(FACE)
        eyeMasks = self.get_attr_masks(EYE_OPEN)
        ulipMasks = self.get_attr_masks(UPPER_LIP)
        llipMasks = self.get_attr_masks(LOWER_LIP)
        assert len(faceMasks) == 1, "Want 1 mask for face!"
        assert len(ulipMasks) == 1, "Want 1 mask for upper lip!"
        assert len(llipMasks) == 1, "Want 1 mask for lower lip!"
        assert len(eyeMasks) == 2, "Want 2 masks for eye!"

        mask = faceMasks[0].copy()

        rmin1, _, _, h1 = self.bbox(eyeMasks[0])
        rmin2, _, _, h2 = self.bbox(eyeMasks[0])
        tol = 1.0
        rmin = min(rmin1, rmin2) - tol * max(h1, h2)
        rmax = max(rmin1 + h1, rmin2 + h2) + tol * max(h1, h2)
        mask[int(rmin):int(rmax) + 1, :] = False

        tol = 0.5
        tol_w = 0.1
        rmin, cmin, w, h = self.bbox(ulipMasks[0])
        mask[int(rmin - tol * h):int(rmin + h + tol * h), int(cmin - tol_w * w):int(cmin + w + tol_w * w)] = False

        rmin, cmin, w, h = self.bbox(llipMasks[0])
        mask[int(rmin - tol * h):int(rmin + h + tol * h), int(cmin - tol_w * w):int(cmin + w + tol_w * w)] = False

        return mask

    """
    to_ratioImage converts sRGB image to HSV like saturation image with H = 0, S = Brightness/saturation (float), V = 0.
  """

    def to_ratioImage(self, image):
        ratioImage = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float)
        ratioImage[:, :, 0] = 0
        flatMask = ratioImage[ratioImage[:, :, 1] == 0]
        flatMask[:, 1] = 0.1
        ratioImage[ratioImage[:, :, 1] == 0] = flatMask
        ratioImage[:, :, 1] = ratioImage[:, :, 2] / (ratioImage[:, :, 1])
        ratioImage[:, :, 2] = 0
        return ratioImage

    """
    to_brightImage converts sRGB image to HSV image with H = 0, S = 0 and only brightness values.
    """

    def to_brightImage(self, image):
        hsvImage = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float)
        hsvImage[:, :, 0] = 0
        hsvImage[:, :, 1] = 0
        return hsvImage

    """
    to_brightImage converts sRGB image to HSV image with H = 0, V = 0 and only saturation values.
    """

    def to_satImage(self, image):
        hsvImage = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float)
        hsvImage[:, :, 0] = 0
        hsvImage[:, :, 2] = 0
        return hsvImage

    def to_LSatImage(self, image):
        hslImage = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float)
        hslImage[:, :, 0] = 0
        hslImage[:, :, 1] = 0
        return hslImage

    def to_LBriImage(self, image):
        hslImage = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float)
        hslImage[:, :, 0] = 0
        hslImage[:, :, 2] = 0
        return hslImage

    """
    to_hueImage converts sRGB image to HSV image with S = 0, V = 0 and only hue values.
    """

    def to_hueImage(self, image):
        hsvImage = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float)
        hsvImage[:, :, 0] *= 2
        hsvImage[:, :, 1] = 0
        hsvImage[:, :, 2] = 0
        return hsvImage

    """
    background_mask returns the mask of the background i.e. everything other than the face.
    """

    def background_mask(self):
        return np.bitwise_xor(np.ones(self.image.shape[:2], dtype=bool), self.get_attr_masks(FACE)[0])

    """
    check_if_good_image returns true if image is good to analyze for skin tone else returns false.
    """

    def check_if_good_image(self):
        ImageUtils.plot_histogram(self.image, self.faceMask, False)
        hist = cv2.calcHist([self.image], [0], self.faceMask.astype(np.uint8) * 255, [256], [0, 256])

        print("Median Red: ", np.median(self.image[self.faceMask][:, 0]))
        if np.median(self.image[self.faceMask][:, 0]) <= 140:
            print("Image too dark: ", np.median(self.image[self.faceMask][:, 0]))
            self.show(self.image)
            return False

        if hist[-1] / max(hist) >= 0.2:
            # Red pixel == 255, too bright.
            print("Image too bright: ", hist[-1] / max(hist))
            self.show(self.image)
            return False

        return True

    """
    divide_all_hue divides given mask into hue masks repeatedly.
    """

    def divide_all_hue(self, mask, image=np.zeros((0, 3)), tol=0.05):
        if np.array_equal(image, np.zeros((0, 3))):
            image = self.to_hueImage(self.image)

        if np.count_nonzero(mask) / np.count_nonzero(self.faceMask) < tol:
            return [mask]
        a, b = self.divide_by_hue(mask, False, image)
        if np.array_equal(b, np.zeros(self.faceMask.shape, dtype=bool)) or \
                (np.mean(image[a], axis=0)[0] - np.mean(image[b], axis=0)[0]) < 2 \
                or np.count_nonzero(mask) / np.count_nonzero(self.faceMask) < tol:
            return [mask]
        return self.divide_all_hue(a, image) + self.divide_all_hue(b, image)

    """
    divide_all_sat divides given mask into saturation masks repeatedly.
    """

    def divide_all_sat(self, mask, image=np.zeros((0, 3)), tol=0.05):
        if np.array_equal(image, np.zeros((0, 3))):
            image = self.to_satImage(self.image)

        a, b = self.divide_by_saturation(mask, False, image)
        if np.array_equal(b, np.zeros(self.faceMask.shape, dtype=bool)) or \
                (np.mean(image[a], axis=0)[1] - np.mean(image[b], axis=0)[1]) * 100.0 / 255.0 < 2 \
                or np.count_nonzero(mask) / np.count_nonzero(self.faceMask) < tol:
            return [mask]
        return self.divide_all_sat(b, image) + self.divide_all_sat(a, image)

    """
    divide_all_brightness divides given mask into brightness masks repeatedly.
    """

    def divide_all_brightness(self, mask, tol=0.05):
        a, b = self.divide_by_brightness(mask, False)
        if np.array_equal(b, np.zeros(self.faceMask.shape, dtype=bool)) or \
                (np.mean(self.brightImage[a], axis=0)[2] - np.mean(self.brightImage[b], axis=0)[2]) * 100.0 / 255.0 < 1 \
                or np.count_nonzero(mask) / np.count_nonzero(self.faceMask) < 0.05:
            return [mask]
        return self.divide_all_brightness(a) + self.divide_all_brightness(b)

    def join_masks(self, masks):
        res = np.zeros(self.faceMask.shape, dtype=bool)
        for m in masks:
            res = np.bitwise_or(res, m)
        return res

    """
    closeness_dist returns the percentage of points in given mask that are perceptually close to given sRGB color.
    """

    def closeness_dist(self, sRGB, mask, k=3):
        delta = ImageUtils.delta_e_mask_matrix(sRGB, self.image[mask])
        return (np.count_nonzero(delta <= k) / np.count_nonzero(mask)) * 100

    """
    best_brightness finds the brightness for given mask that best captures the skin tone within the given larger mask.
    """

    def best_brightness(self, meanRGB, mask, largerMask):
        minV = np.min(ImageUtils.sRGBtoHSV(self.image[mask])[:, 2])
        maxV = np.max(ImageUtils.sRGBtoHSV(self.image[mask])[:, 2])

        meanHSV = ImageUtils.sRGBtoHSV(meanRGB)
        print("\tMean brightness: ", meanHSV[0][2] * (100.0 / 255.0))

        bestBrightness = 0
        closeList = []
        closest = 0
        for v in range(minV, maxV + 1):
            meanHSV[0][2] = v
            meanRGB = ImageUtils.HSVtosRGB(meanHSV)
            closeness = self.closeness_dist(meanRGB[0], largerMask)
            if closeness > closest:
                closest = closeness
                bestBrightness = v
            closeList.append(closeness)

        print("\tBest Brightness: ", bestBrightness * (100.0 / 255.0))
        print("\tAverage closeness: ", sum(closeList) / len(closeList))
        print("\tMax closeness: ", max(closeList))

        meanHSV[0][2] = bestBrightness
        meanRGB = ImageUtils.HSVtosRGB(meanHSV)
        return meanRGB[0], max(closeList)

    """
    best_sat finds the saturation for given mask that best captures the skin tone within the given larger mask.
    """

    def best_sat(self, sRGB, mask, largerMask):
        minSat = np.min(ImageUtils.sRGBtoHSV(self.image[mask])[:, 1])
        maxSat = np.max(ImageUtils.sRGBtoHSV(self.image[mask])[:, 1])

        meanHSV = ImageUtils.sRGBtoHSV([sRGB])
        print("\tMean Sat: ", meanHSV[0][1] * (100.0 / 255.0))

        bestSat = 0
        closeList = []
        closest = 0
        for s in range(minSat, maxSat + 1):
            meanHSV[0][1] = s
            meanRGB = ImageUtils.HSVtosRGB(meanHSV)
            closeness = self.closeness_dist(meanRGB[0], largerMask)
            if closeness > closest:
                closest = closeness
                bestSat = s
            closeList.append(closeness)

        print("\tBest Sat: ", bestSat * (100.0 / 255.0))
        print("\tAverage closeness: ", sum(closeList) / len(closeList))
        print("\tMax closeness: ", max(closeList))

        meanHSV[0][1] = bestSat
        meanRGB = ImageUtils.HSVtosRGB(meanHSV)
        return meanRGB[0], max(closeList)

    """
    Detects skin tone.
    """

    def detect_skin_tone(self):
        # ImageUtils.plot_histogram(self.image, self.faceMask, block=True)

        higherSatMask, lowerSatMask = self.divide_by_saturation(self.faceMask, False)
        a, b = self.divide_by_saturation(higherSatMask, False)
        c, d = self.divide_by_saturation(lowerSatMask, False)
        satMaskList = [a, b, c, d]

        # Pick best neck points.
        akpts, bkpts = self.biclustering_Kmeans_mod(self.image, self.get_neck_points())
        aDeltaList, bDeltaList = [], []
        aTotalScore, bTotalScore = 0.0, 0.0
        for i, m in enumerate(satMaskList):
            delta = ImageUtils.delta_e_mask(self.image, akpts, m)
            close = self.closeness_dist(np.median(self.image[akpts], axis=0), m)
            aTotalScore += delta * np.count_nonzero(m)
            aDeltaList.append((m, delta, close, i))
        aTotalScore /= np.count_nonzero(self.faceMask)

        for i, m in enumerate(satMaskList):
            delta = ImageUtils.delta_e_mask(self.image, bkpts, m)
            close = self.closeness_dist(np.median(self.image[bkpts], axis=0), m)
            bTotalScore += delta * np.count_nonzero(m)
            bDeltaList.append((m, delta, close, i))
        bTotalScore /= np.count_nonzero(self.faceMask)

        print("left score, right score: ", aTotalScore, bTotalScore)
        nkPts, deltaList = (akpts, aDeltaList) if aTotalScore < bTotalScore else (bkpts, bDeltaList)

        # Store neckpoints.
        self.nkPts = nkPts

        # Find top two masks (minimum delta) and merge them.
        for (m, d, c, i) in deltaList:
            print("mask number: ", i, ", count: ", np.count_nonzero(m) / np.count_nonzero(self.faceMask), " delta: ", d,
                  " closeness: ", c)
        sDeltaList = sorted(deltaList, key=lambda x: -x[2])
        goodMask = np.zeros(self.faceMask.shape, dtype=bool)
        for (m, _, _, k) in sDeltaList[:2]:
            print("Picking mask number: ", k)
            goodMask = np.bitwise_or(goodMask, m)

        print("Mask percent: ", np.count_nonzero(goodMask) / np.count_nonzero(self.faceMask))

        # Divide by Hue and then find brightness.
        self.show_mask(goodMask)

        allSkinTones = []
        allHueMasks = self.divide_all_hue(goodMask)
        for m in allHueMasks:
            print("HUE: ", np.mean(self.to_hueImage(self.image)[m], axis=0)[0], " Mask percent: ",
                  (np.count_nonzero(m) / np.count_nonzero(goodMask)) * 100.0)
            self.show_mask(m)
            if np.count_nonzero(m) / np.count_nonzero(goodMask) < 0.05:
                print("Skipped because mask size too small\n")
                continue
            meanRGB, closeness = self.best_brightness(np.mean(self.image[m], axis=0), m, goodMask)
            finalRGB, closeness = self.best_sat(meanRGB, m, goodMask)
            closeness = 0
            finalRGB = meanRGB
            print("Skin Tone: ", ImageUtils.RGB2HEX(finalRGB), " closeness: ", closeness, " brightness: ",
                  ImageUtils.sRGBtoHSV(finalRGB)[0, 2] * (100.0 / 255.0))
            self.show_skin_tone(finalRGB)
            allSkinTones.append((m, finalRGB, closeness))
            print("")

        # Merge masks with similar skin tones.
        mergedSkinTones = []
        for i, t in enumerate(allSkinTones):
            print("MASK: ", i, " tone: ", ImageUtils.RGB2HEX(t[1]), " close: ", t[2])
            for j in range(i + 1, len(allSkinTones)):
                q = allSkinTones[j]
                print("\tColor diff with j: ", j, " is: ", ImageUtils.delta_cie2000(t[1], q[1]))

    """
    divide_by_hue divides given mask using hue image into higher and lower hue sub masks.
    """

    def divide_by_hue(self, mask, show_masks_info=True, image=np.zeros((0, 3))):
        if np.array_equal(image, np.zeros((0, 3))):
            image = self.to_hueImage(self.image)

        print("divide by hue")
        try:
            higherHueMask, lowerHueMask = self.biclustering_Kmeans_mod(image, mask,
                                                                       func=lambda img, m1, m2: (m1, m2) if np.mean(
                                                                           img[m1][:, 0]) >= np.mean(
                                                                           img[m2][:, 0]) else (m2, m1))
            if np.count_nonzero(higherHueMask) == 0 or np.count_nonzero(higherHueMask) == 0:
                return mask, np.zeros(self.faceMask.shape, dtype=bool)
            if show_masks_info:
                self.masks_info(higherHueMask, lowerHueMask)
            return higherHueMask, lowerHueMask
        except Exception as e:
            return mask, np.zeros(self.faceMask.shape, dtype=bool)

    """
    divide_by_saturation divides given mask using saturation image into higher and lower saturation sub masks.
    """

    def divide_by_saturation(self, mask, show_masks_info=True, image=np.zeros((0, 3))):
        if np.array_equal(image, np.zeros((0, 3))):
            image = self.to_satImage(self.image)
        print("divide by saturation")
        try:
            higherSatMask, lowerSatMask = self.biclustering_Kmeans_mod(image, mask,
                                                                       func=lambda img, m1, m2: (m1, m2) if np.mean(
                                                                           img[m1][:, 1]) >= np.mean(
                                                                           img[m2][:, 1]) else (m2, m1))
            if np.count_nonzero(higherSatMask) == 0 or np.count_nonzero(lowerSatMask) == 0:
                return mask, np.zeros(self.faceMask.shape, dtype=bool)
            if show_masks_info:
                self.masks_info(higherSatMask, lowerSatMask)
            return higherSatMask, lowerSatMask
        except Exception as e:
            return mask, np.zeros(self.faceMask.shape, dtype=bool)

    """
    divide_by_brightness divides given mask using brightness image into higher and lower brightness sub masks. It 
    returns the mask with higher saturation first.
    """

    def divide_by_brightness(self, mask, show_masks_info=True):
        print("divide by brightness")
        higherBriMask, lowerBriMask = self.biclustering_Kmeans_mod(self.brightImage, mask,
                                                                   func=lambda img, m1, m2: (m1, m2) if np.mean(
                                                                       img[m1][:, 2]) >= np.mean(img[m2][:, 2]) else (
                                                                       m2, m1))
        if show_masks_info:
            self.masks_info(higherBriMask, lowerBriMask)
        return higherBriMask, lowerBriMask

    """
    divide_by_ratio will divide given mask using brightness by brightness/saturation ratio. It returns the mask with 
    higher brightness/saturation ratio first.
    """

    def divide_by_ratio(self, mask, show_masks_info=True):
        print("divide by ratio")
        higherRatioMask, lowerRatioMask = self.biclustering_Kmeans_mod(self.to_ratioImage(self.image), mask,
                                                                       func=lambda img, m1, m2: (m1, m2) if np.mean(
                                                                           img[m1][:, 1]) >= np.mean(
                                                                           img[m2][:, 1]) else (m2, m1))
        if show_masks_info:
            self.masks_info(higherRatioMask, lowerRatioMask)
        return higherRatioMask, lowerRatioMask

    """
    masks_info will print various measurements about the input masks like brightness, saturation, delta_e_cie2000 
    diff etc. which can then be analyzed later. The function also displays the masks.
    """

    def masks_info(self, leftMask, rightMask):
        mask = np.bitwise_or(leftMask, rightMask)
        print("\tMask division: ", np.count_nonzero(leftMask) / np.count_nonzero(mask),
              np.count_nonzero(rightMask) / np.count_nonzero(mask))
        print("\tCounts: ", (np.count_nonzero(leftMask) / np.count_nonzero(self.faceMask)) * 100.0,
              (np.count_nonzero(rightMask) / np.count_nonzero(self.faceMask)) * 100.0)
        print("\tSat values: ", np.mean(self.to_satImage(self.image)[leftMask], axis=0)[1] * 100.0 / 255.0,
              np.mean(self.to_satImage(self.image)[rightMask], axis=0)[1] * 100.0 / 255.0, abs(
                np.mean(self.to_satImage(self.image)[leftMask], axis=0)[1] * 100.0 / 255.0 -
                np.mean(self.to_satImage(self.image)[rightMask], axis=0)[1] * 100.0 / 255.0))
        print("\tBrightness values: ", np.mean(self.brightImage[leftMask], axis=0)[2] * 100.0 / 255.0,
              np.mean(self.brightImage[rightMask], axis=0)[2] * 100.0 / 255.0, abs(
                np.mean(self.brightImage[leftMask], axis=0)[2] * 100.0 / 255.0 -
                np.mean(self.brightImage[rightMask], axis=0)[2] * 100.0 / 255.0))
        print("\tHue values: ", np.mean(self.to_hueImage(self.image)[leftMask], axis=0)[0],
              np.mean(self.to_hueImage(self.image)[rightMask], axis=0)[0])
        print("\tSat std: ", np.std(self.to_satImage(self.image)[leftMask], axis=0)[1] * 100.0 / 255.0,
              np.std(self.to_satImage(self.image)[rightMask], axis=0)[1] * 100.0 / 255.0)
        print("\tBrightness std: ", np.std(self.brightImage[leftMask], axis=0)[2] * 100.0 / 255.0,
              np.std(self.brightImage[rightMask], axis=0)[2] * 100.0 / 255.0)
        print("\tRatio values: ", np.mean(self.to_ratioImage(self.image)[leftMask], axis=0)[1],
              np.mean(self.to_ratioImage(self.image)[rightMask], axis=0)[1], abs(
                np.mean(self.to_ratioImage(self.image)[leftMask], axis=0)[1] -
                np.mean(self.to_ratioImage(self.image)[rightMask], axis=0)[1]))
        print("\tMean Colors: ", ImageUtils.RGB2HEX(np.mean(self.image[leftMask], axis=0)),
              ImageUtils.RGB2HEX(np.mean(self.image[rightMask], axis=0)))
        print("\tMedian Colors: ", ImageUtils.RGB2HEX(np.median(self.image[leftMask], axis=0)),
              ImageUtils.RGB2HEX(np.median(self.image[rightMask], axis=0)))
        print("\tColor diff between left/right masks: ", ImageUtils.delta_e_mask(self.image, leftMask, rightMask))
        print("\tLeft Color closeness with mask: ", self.closeness_dist(np.median(self.image[leftMask], axis=0), mask))
        print("\tRight Color closeness with mask: ",
              self.closeness_dist(np.median(self.image[rightMask], axis=0), mask))
        print("")
        self.show_masks([leftMask, rightMask], [[0, 255, 0], [255, 0, 0]])

    """
    best_clusters repeatedly divides each mask until the cluster mean (when calculating delta_e of cluster w.r.t 
    medoid) is less than the given tolerance. This calls the other overloaded best_clusters method.
    """

    def best_clusters(self, faceMask, delta_tol=3):
        # number of clusters.
        k = 2
        allMedoids, allMasks, allIndices, _ = ImageUtils.best_clusters(self.distinct_colors(faceMask, tol=0.0005),
                                                                       self.image, faceMask, k)

        while True:
            masks = []
            medoids = []
            indices = []

            for i, m in zip(allIndices, allMasks):
                if np.mean(ImageUtils.delta_e_mask_matrix(allMedoids[i], self.image[m])) <= delta_tol:
                    # No need to sub divide mask.
                    masks.append(m)
                    indices.append(len(medoids))
                    medoids.append(allMedoids[i])
                    continue

                cmeds, cmasks, _, _ = ImageUtils.best_clusters(self.distinct_colors(m, tol=0.0005), self.image, m, k,
                                                               delta_tol)
                if len(cmasks) == 0:
                    # Some kind of exception occrured, just use the mask and move on.
                    masks.append(m)
                    indices.append(len(medoids))
                    medoids.append(allMedoids[i])
                    continue

                # Append both sub masks.
                masks.append(cmasks[0])
                indices.append(len(medoids))
                medoids.append(cmeds[0])

                masks.append(cmasks[1])
                indices.append(len(medoids))
                medoids.append(cmeds[1])

            if len(masks) == len(allMasks):
                # Iterations complete, return masks.
                break

            allMasks = masks.copy()
            allIndices = indices.copy()
            allMedoids = medoids.copy()

        return allMedoids, allMasks, allIndices

    """
    divide_mask will divide given mask into clusters based on delta_e_cie2000 differences.
    """

    def divide_mask(self, mask):
        DELTA = 5

        domMask, surrMask = self.biclustering_Kmeans_mod(self.image, mask)
        if ImageUtils.delta_e_mask(self.image, domMask, surrMask) <= DELTA:
            return [mask]

        domSubMasks = self.divide_mask(domMask)
        surrSubMasks = self.divide_mask(surrMask)

        return domSubMasks + surrSubMasks

    """
    adjust_brightness will set the brightness of color to brightness v.
    """

    def adjust_brightness(self, color, v):
        hsv = ImageUtils.sRGBtoHSV(color)
        hsv[0, 2] = v
        return ImageUtils.HSVtosRGB(hsv)

    # Modify image brightness to given brightness (in percent).
    def apply_brightness(self, mask, br):
        clone = self.image.copy()
        clone = cv2.cvtColor(clone, cv2.COLOR_RGB2HSV)
        fmask = clone[mask]
        fmask[:, 2] = np.uint8(br * (255.0) / (100.0))
        clone[mask] = fmask
        clone = cv2.cvtColor(clone, cv2.COLOR_HSV2RGB)
        return clone

    """
    bbox returns bounding box (x1, y1, w, h) (top left point, width and height) of given mask.
    """

    def bbox(self, mask):
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, cmin, cmax - cmin + 1, rmax - rmin + 1

    """
    get_custom_mask will return a mask of points with given attributes.
    """

    def get_custom_mask(self, attrs):
        allMasks = []
        for attr in attrs:
            allMasks += self.get_attr_masks(attr)
        if len(allMasks) == 0:
            raise Exception("No mask found for any attribute!")
        mask = np.zeros(self.image.shape[:2], dtype=bool)
        for m in allMasks:
            mask = np.bitwise_xor(mask, m)
        return mask

    """
    get_attr_masks returns a list of masks for given attribute. Attribute can be one of EYE_OPEN, NOSE, 
    UPPER_LIP etc. It will return an empty list for any attribute that doesn't exist in the face.
    """

    def get_attr_masks(self, attr):
        masks = []
        for i, class_id in enumerate(self.preds[Face.CLASS_IDS_KEY]):
            if label_id_map[attr] == class_id:
                masks.append(self.preds[Face.MASKS_KEY][:, :, i])
        return masks


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Image path')
    parser.add_argument('--image', required=False, metavar="path or URL to image")
    parser.add_argument('--mask', required=False, metavar="path to face mask file")
    args = parser.parse_args()

    f = Face(args.image, hdf5FileName=args.mask)
    f.detect_background()
    # f.detect_skin_tone()
