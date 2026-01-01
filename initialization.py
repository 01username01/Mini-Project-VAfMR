import numpy as np
import cv2 as cv

from utils import decomposeEssentialMatrix, disambiguateRelativePose, linearTriangulation


def initialization(img0, img1, K):
    # Use SIFT to find features and descript them

    # Convert the BGR images to gray scale if needed
    if img0.ndim == 3:
        img0 = cv.cvtColor(img0,cv.COLOR_BGR2GRAY) # CV loads the images as BGR
        img1 = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)

    # Applying SIFT detector to get the features und corresponding descriptors
    sift = cv.SIFT_create()
    img0_features, img0_descriptor = sift.detectAndCompute(img0, None)
    img1_features, img1_descriptor = sift.detectAndCompute(img1, None)

    # Feature matching
    bf = cv.BFMatcher()
    matches = bf.knnMatch(img0_descriptor.astype(np.float32), img1_descriptor.astype(np.float32), 2)

    # Filtering the matches
    pts_0 = []
    pts_1 = []
    match_points = []
    for best, second_best in matches:
        if best.distance < 0.75 * second_best.distance:
            pts_0.append(list(img0_features[best.queryIdx].pt))
            pts_1.append(list(img1_features[best.trainIdx].pt))
            match_points.append(best)

    pts_0 = np.array(pts_0)
    pts_1 = np.array(pts_1)
    
    # Get Fundamental Matrix aswell as the inliers
    F, inlier_mask = cv.findFundamentalMat(pts_0, pts_1, cv.FM_RANSAC, 1.0, 0.999)

    # Creat list of inlier points
    inliers_0 = pts_0[inlier_mask.flatten() == 1]
    inliers_1 = pts_1[inlier_mask.flatten() == 1]
    match_points_inliers = [m for m, keep in zip(match_points, inlier_mask.flatten()) if keep == 1]

    # Creat list of homgenious inliers
    inliers_0_hom = np.vstack([inliers_0.T, np.ones((1, inliers_0.shape[0]))])
    inliers_1_hom = np.vstack([inliers_1.T, np.ones((1, inliers_1.shape[0]))])

    
    # Plot the matches (inlier only)
    img_matches = cv.drawMatches(
        img0, img0_features,
        img1, img1_features,
        match_points_inliers,
        None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    img_matches_small = cv.resize(img_matches, None, fx=0.5, fy=1, interpolation=cv.INTER_AREA)
    cv.imshow("Matches (inlier only)", img_matches_small)
    cv.waitKey(0)
    cv.destroyAllWindows()
    

    # Calculate the Essential Matrix 
    E_ = K.T @ F @ K
    # Ensure rank 2 
    U, S, Vt = np.linalg.svd(E_)
    S[2] = 0
    E = U @ np.diag(S) @ Vt
    print(np.linalg.matrix_rank(E))

    # Calculate the rotation matrix and the translation vector from E
    rots, u3 = decomposeEssentialMatrix(E)

    # Select the correct R and T
    R, T = disambiguateRelativePose(rots, u3, inliers_0_hom, inliers_1_hom, K, K)
    trans_mat = np.concatenate((R, T[:, None]), axis=1)
    
    # Creat projection matrix
    M1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))
    M2 = K @ trans_mat

    # Triangulate points
    points_3D_hom = linearTriangulation(inliers_0_hom, inliers_1_hom, M1, M2)
    points_3D_ = (points_3D_hom[:3, :] / points_3D_hom[3, :]).T

    # Remove points that have a negative value for Z
    points_3D = points_3D_[points_3D_[:,2] >= 0]
    inliers_img1 = inliers_1[points_3D_[:,2] >= 0]


    return points_3D, trans_mat, inliers_img1

