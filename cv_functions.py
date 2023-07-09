import cv2 as cv
import numpy as np


def blur(img, width, height):
    return cv.blur(img, (width, height))


def match_shape(shape):
    match shape:
        case "RECT":
            return cv.MORPH_RECT
        case "ELLIPSE":
            return cv.MORPH_ELLIPSE
        case "CROSS":
            return cv.MORPH_CROSS
        case _:
            return None


def erode(img, size, iterations, shape):
    element = cv.getStructuringElement(shape, (2 * size + 1, 2 * size + 1), (-1, -1))
    return cv.erode(img, element, iterations=iterations)


def dilate(img, size, iterations, shape):
    shape = match_shape(shape)
    element = cv.getStructuringElement(shape, (2 * size + 1, 2 * size + 1), (-1, -1))
    return cv.dilate(img, element, iterations=iterations)  # TODO: add bordertype dropdown


def hsv_threshold(img, lower_bound, upper_bound, invert):
    # remember cv2 uses BGR but easyopencv uses rgb!
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    thresh = cv.inRange(hsv, lower_bound, upper_bound)
    if invert:
        thresh = np.array(thresh)
        return np.bitwise_not(thresh)
    return thresh


def match_approx_method(approx_method):
    match approx_method:
        case "CHAIN NONE":
            return cv.CHAIN_APPROX_NONE
        case "CHAIN SIMPLE":
            return cv.CHAIN_APPROX_SIMPLE
        case "CHAIN TC89 L1":
            return cv.CHAIN_APPROX_TC89_L1
        case "CHAIN TC89 KCOS":
            return cv.CHAIN_APPROX_TC89_KCOS
        case _:
            return None


def contour(thresh, approx_method):
    return cv.findContours(thresh, cv.RETR_EXTERNAL, approx_method)
