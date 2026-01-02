import numpy as np
import cv2 as cv

from utils import decomposeEssentialMatrix, disambiguateRelativePose, linearTriangulation


def recompute_3d_points(pts_0, pts_1, K, abs_movement):
    

    # Get Fundamental Matrix aswell as the inliers
    F, inlier_mask = cv.findFundamentalMat(pts_0, pts_1, cv.FM_RANSAC, 1.0, 0.99)

    # Creat list of inlier points
    inliers_0 = pts_0[inlier_mask.flatten() == 1]
    inliers_1 = pts_1[inlier_mask.flatten() == 1]

    # Creat list of homgenious inliers
    inliers_0_hom = np.vstack([inliers_0.T, np.ones((1, inliers_0.shape[0]))])
    inliers_1_hom = np.vstack([inliers_1.T, np.ones((1, inliers_1.shape[0]))])

    
    # # Plot the matches (inlier only)
    # img_matches = cv.drawMatches(
    #     img0, img0_features,
    #     img1, img1_features,
    #     match_points_inliers,
    #     None,
    #     flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    # )
    # img_matches_small = cv.resize(img_matches, None, fx=0.5, fy=1, interpolation=cv.INTER_AREA)
    # cv.imshow("Matches (inlier only)", img_matches_small)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    

    # Calculate the Essential Matrix 
    E_ = K.T @ F @ K
    # Ensure rank 2 
    U, S, Vt = np.linalg.svd(E_)
    S[2] = 0
    E = U @ np.diag(S) @ Vt
    print(np.linalg.matrix_rank(E))

    # Calculate the rotation matrix and the translation vector from E
    rots, u3 = decomposeEssentialMatrix(E)

    u3 *= abs_movement

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

