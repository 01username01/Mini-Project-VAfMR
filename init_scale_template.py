import numpy as np
import cv2 as cv
# finds corners of the scaling template, and places the template arbitrarily into the camera frame, with the correct scaling
# inputs: image is the template image, padded with white, so that the pixel size is the same as the other pictures
# sidelength_m: length of the fiducial marker in meters, sidelength_p: lengthof the marker in pixels
def init_scale_template(image, sidelength_m, sidelength_p):
    scaling_factor = -sidelength_m/sidelength_p
    #height, width = image.shape
    # detect corner locations of the template image
    print(image.dtype)
    features = cv.goodFeaturesToTrack(image, maxCorners=500, qualityLevel=0.01, minDistance=7, blockSize=7, useHarrisDetector=False) # (num_new_keypoints, 1,2) 
    # place template at an arbitrary point in the world
    # for simplicity chosen was that the point (0, 0), the top left of the image lies at the origin of the world frame
    coordinates_3d = np.zeros((features.shape[0], 1, 3))
    coordinates_3d[:, :, :2] = features*scaling_factor
    return coordinates_3d, features

