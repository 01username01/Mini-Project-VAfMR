import numpy as np
import cv2 as cv
from utils import linearTriangulationBatch
from recompute_3d import recompute_3d_points


def scale_correction(current_img, S, K, template, starting_state):

    starting_keypoints = starting_state["starting_keypoints"]
    starting_img = starting_state["starting_img"]
    starting_t = starting_state["starting_t"]
    T_cw_start = starting_state["T_cw_start"]

    template_img = template["template_img"]
    pts_2D_marker = template["pts_2D_marker"]
    pts_3D_marker = template["pts_3D_marker"]



    new_start_found = False
    S_corrected = None
    t_corrected_norm = None
    marker_found = False
    correction_found = False
    pts_2D_marker_world, status_marker, err_marker = cv.calcOpticalFlowPyrLK(template_img, current_img, pts_2D_marker, None)
    detected_pts_2D_marker = pts_2D_marker_world[status_marker == 1] # (num_det_pts, 2)
    detected_pts_3D_marker = pts_3D_marker[status_marker == 1] # (num_det_pts, 3)
    # position of the camera relative to the marker
    success_marker, rvec_marker, tvec_marker, inliers_marker = cv.solvePnPRansac(detected_pts_3D_marker, detected_pts_2D_marker, K, None, flags=cv.SOLVEPNP_P3P, iterationsCount=100, reprojectionError=2.0, confidence=0.999)
    marker_found = inliers_marker.shape[0] >= 150 # condition if the marker is found, if there are enough inliers, marker was located
    if marker_found: 
        if starting_t is not None: # see if marker was already found in previous frames
            abs_movement = np.linalg.norm(tvec_marker-starting_t)
            if abs_movement > 0.2: # check if the two frames are far enough apart
                print("Images far enough apart, applying scale correction")
                correction_found = True
                
                pts_2D, status, err = cv.calcOpticalFlowPyrLK(starting_img, current_img, starting_keypoints, None)
                #recompute all 3d points with the correct scale
                points_3D_prev_frame, T_mat_now_prev, inliers_2D_new = recompute_3d_points(starting_keypoints[status==1], pts_2D[status==1], K, abs_movement)
                R_cw_start = T_cw_start[:, :3]
                R_wc_start = R_cw_start.T 
                t_cw_start = T_cw_start[:, 3]
                t_wc_start = -R_wc_start @ t_cw_start
                points_3D_world_frame = R_wc_start @ points_3D_prev_frame.T + t_wc_start.reshape(3,1)
                S_corrected = {
                    "P": inliers_2D_new.T,         # (2, K)
                    "X": points_3D_world_frame,         # (3, K)
                    "C": np.zeros((2, 0)),    # (2, 0)
                    "F": np.zeros((2, 0)),    # (2, 0)
                    "T": np.zeros((12, 0)),   # (12, 0)
                }
                print(points_3D_world_frame)
                t_corrected_norm = abs_movement
            else:
                print("Still too close")
        else:
            print("Marker found for the first time")
            new_start_found = True
            starting_state["starting_keypoints"] = S["P"].T.astype(np.float32)[:, None, :] # (num_kp, 1, 2)
            starting_state["starting_img"] = current_img
            starting_state["starting_t"] = tvec_marker
    else:
        print("No marker found")
        starting_state["starting_keypoints"] = None
        starting_state["starting_img"] = None
        starting_state["starting_t"] = None


    return correction_found, new_start_found, starting_state, S_corrected, t_corrected_norm
