import numpy as np
import cv2 as cv
from utils import linearTriangulationBatch


def processFrame(img_prev, img_next, S, K):
    # Find the keypoints in the new image using KLT

    keypoints = S["P"].T.astype(np.float32)[:, None, :] # (num_kp, 1, 2)
    landmarks = S["X"].T.astype(np.float32)[:, None, :] # (num_kp, 1, 3)

    print(f"Number of keypoints and landmakts in S_prev: {keypoints.shape[0]}")
    
    pts_2D, status, err = cv.calcOpticalFlowPyrLK(img_prev, img_next, keypoints, None)
    
    detected_pts_2D = pts_2D[status == 1] # (num_det_pts, 2)
    detected_pts_3D = landmarks[status == 1] # (num_det_pts, 3)

    print(f"Number of keypoints found in next image: {detected_pts_2D.shape[0]}")

    # Find camera pose
    success, rvec, tvec, inliers = cv.solvePnPRansac(detected_pts_3D, detected_pts_2D, K, None, flags=cv.SOLVEPNP_P3P, iterationsCount=100, reprojectionError=2.0, confidence=0.99)

    if not success:
        raise RuntimeError("solvePnPRansac failed to find a valid camera pose.")
    
    R, _ = cv.Rodrigues(rvec)
    trans_mat = np.hstack((R, tvec.reshape(3,1))) # (3, 4) T_cw from world to camera

    # Get the new inliers
    inlier_pts_2D = detected_pts_2D[inliers][:, 0, :] # (num_inlier, 2)
    inlier_pts_3D = detected_pts_3D[inliers][:, 0, :] # (num_inlier, 3)
    
    print(f"Number of inliers for P3P: {inlier_pts_2D.shape[0]}")

    # The inliers are going to be part of the new X and P
    P_next = inlier_pts_2D.T # (2, num_inlier)
    X_next = inlier_pts_3D.T # (3, num_inlier)

    
    # Update candidates
    C_prev = S["C"].T.astype(np.float32)[:, None, :] # (num_cand, 1, 2)
    F_prev = S["F"] # (2, num_cand)
    T_prev = S["T"] # (12, num_cand)

    print(f"Number of previous candidates: {C_prev.shape[0]}")

    # Find canditates in new image
    if C_prev.shape[0] > 0:

        C_tracked, C_status, err = cv.calcOpticalFlowPyrLK(img_prev, img_next, C_prev, None)

        # Only keep the ones that were found in the new image
        C_next = C_tracked[C_status == 1] # (num_new_cand, 2)
        F_next = F_prev[:, C_status.flatten() == 1].T # (num_new_cand, 2)
        T_next = T_prev[:, C_status.flatten() == 1].T # (num_new_cand, 12)

        print(f"Number of previous candidates found in next image: {C_next.shape[0]}")
      
    else:
        C_next = np.zeros((0, 2)) # (0, 2)
        F_next = np.zeros((0, 2)) # (0, 2)
        T_next = np.zeros((0, 12)) # (0, 12)
    
    # Triangulate candidates if the fullfill the criteria
    
    # Check criteria
    alpha = 2 * np.tan(0.1 / 2) # comming from b/z > 10

    print(f"Angles criterium: must be bigger than {alpha}")

    # Convert C and F to homogeneous coordinates
    ones = np.ones(C_next.shape[0])[:, None]
    C_next_hom = np.concatenate((C_next, ones), axis=1) # (num_new_cand, 3)
    F_next_hom = np.concatenate((F_next, ones), axis=1) # (num_new_cand, 3)
    
    # calculate the bearing vectors
    v1 = (np.linalg.inv(K) @ F_next_hom.T).T # (num_new_cand, 3) from previous images
    v2 = (np.linalg.inv(K) @ C_next_hom.T).T # (num_new_cand, 3) from the current image
    
    # normalize
    v1_n = v1 / np.linalg.norm(v1, axis=1, keepdims=True) # (num_new_cand, 3)
    v2_n = v2 / np.linalg.norm(v2, axis=1, keepdims=True) # (num_new_cand, 3)

    # Convert the bearing vectors to the world frame
    T_next_reshaped = T_next.reshape(-1, 3, 4) # (num_new_cand, 3, 4)
    R_F = T_next_reshaped[:, :, :3].transpose(0,2,1) # (num_new_cand, 3, 3)
    v1_w = np.einsum('nij,nj->ni', R_F, v1_n) # (num_new_cand, 3)
    v2_w = (trans_mat[:, :3].T @ v2_n.T).T # (num_new_cand, 3)
    
    # Calculate the angles between the normalized bearing vectors
    dot_prod = np.clip(np.sum(v1_w * v2_w, axis=1,), -1.0, 1.0) # (num_new_cand,)
    angles =  np.arccos(dot_prod) # (num_new_cand,)

    print(f"Number of canditates fullfilling the criteria: {np.sum(angles > alpha)}")

    if np.sum(angles > alpha) > 0:

        # Filter out the points that fullfill the criterium 
        pts_C = C_next_hom[angles > alpha].T # (3, num_tri_pts)
        pts_F = F_next_hom[angles > alpha].T # (3, num_tri_pts)

        M_C = K @ trans_mat 
        M_C_batch = np.broadcast_to(M_C, (pts_C.shape[1], 3, 4)) # (num_tri_pts, 3,4)
        M_F_batch = K @ (T_next.reshape(-1, 3, 4)[angles > alpha]) # (num_tri_pts, 3, 4)

        # Triangulate the points
        additional_X_hom = linearTriangulationBatch(pts_F, pts_C, M_F_batch, M_C_batch) # (4, num_tri_pts)
        additional_X = (additional_X_hom[:3, :] / additional_X_hom[3, :]) # (3, num_tri_pts)

        # Remove the freashly triangulated points from C, F and T
        C_next = C_next[angles <= alpha] # (num_new_cand - num_tri_pts, 2)
        F_next = F_next[angles <= alpha] # (num_new_cand - num_tri_pts, 2)
        T_next = T_next[angles <= alpha] # (num_new_cand - num_tri_pts, 2)

        # Append the freshly trinagulated points to X and P
        P_next = np.concatenate((P_next, pts_C[:2, :]), axis=1) # (2, num_total traing_pts)
        X_next = np.concatenate((X_next, additional_X), axis=1) # (3, num_total traing_pts)
        
    # Find new candidates
    
    # Use Shi Tomasi to find corners
    new_keypoints = cv.goodFeaturesToTrack(img_next, maxCorners=2000, qualityLevel=0.01, minDistance=7, blockSize=7, useHarrisDetector=False) # (num_new_keypoints, 1,2)
    new_keypoints = new_keypoints[:, 0, :] # (num_new_keypoints, 2)

    print(f"Number of detected keaypoints: {new_keypoints.shape[0]}")
    
    # Check if the new points aren't already in the old C or P
    existing_pts = np.vstack([P_next.T, C_next]) # (num_exisitng_pts, 2)
    
    diff = new_keypoints[:, None, :] - existing_pts[None, :, :] #  (num_new_keypoints, 1, 2) - (1, num_existing_pts, 2) -> (num_new_keypoints, num_existing_pts, 2)
    dist = np.linalg.norm(diff, axis=2)  # (num_new_keypoints, num_existing_pts)
    
    mask_new = np.all(dist > 3.0, axis=1) # further apart than 3 pixels
    new_unique_keypoints = new_keypoints[mask_new] # (num_new_unique_keypoints, 2)

    print(f"Number of new unique keypoints: {new_unique_keypoints.shape[0]}")

    # Append the unique new keypoints to the candidate list 
    C_next = np.concatenate((C_next, new_unique_keypoints), axis=0) # (num_next_c, 2)
    
    # Append the unique new keypoints to F and T
    F_next = np.concatenate((F_next, new_unique_keypoints), axis=0) # (num_next_f, 2)
    T_add = np.tile(trans_mat.flatten(), (new_unique_keypoints.shape[0], 1)) # (num_new_unique_keypoints, 12)
    T_next = np.concatenate((T_next, T_add), axis=0) # (num_next_t, 2)

    # Creat next state dictionary
    S_next = {
        "P": P_next, # (2, N)
        "X": X_next, # (3, N)
        "C": C_next.T, # (2, M)
        "F": F_next.T, # (2, M)
        "T": T_next.T, # (2, M)
    }

    return S_next, trans_mat # S_next : dict, trans_mat : np.array (3, 4)
