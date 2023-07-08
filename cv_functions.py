import cv2

def gaussian_blur(img, amount, kernel=(11,11)):
    return cv2.GaussianBlur(img, kernel, amount)

def erode(img, iterations, kernel=(11,11), anchor=(-1,-1)):
    return cv2.erode(img, iterations=iterations, kernel=kernel, anchor=anchor)

def dilate(img, iterations, kernel=(11,11), anchor=(-1,-1)):
    return cv2.dilate(img, iterations=iterations, kernel=kernel, anchor=anchor)

def hsv_threshold(img, lower_bound, upper_bound):
    # remember cv2 uses BGR
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, lower_bound, upper_bound)