import numpy as np
from scipy import signal
from scipy.linalg import expm, logm
import scipy
from scipy.optimize import least_squares
import matplotlib.pyplot as plt


def distortPoints(x, D, K):
    """Applies lens distortion D(2) to 2D points x(Nx2) on the image plane. """

    k1, k2 = D[0], D[1]

    u0 = K[0, 2]
    v0 = K[1, 2]

    xp = x[:, 0] - u0
    yp = x[:, 1] - v0

    r2 = xp**2 + yp**2
    xpp = u0 + xp * (1 + k1*r2 + k2*r2**2)
    ypp = v0 + yp * (1 + k1*r2 + k2*r2**2)

    x_d = np.stack([xpp, ypp], axis=-1)

    return x_d


def projectPoints(points_3d, K, D=np.zeros([4, 1])):
    """
    Projects 3d points to the image plane (3xN), given the camera matrix (3x3) and
    distortion coefficients (4x1).
    """
    # get image coordinates
    projected_points = np.matmul(K, points_3d[:, :, None]).squeeze(-1)
    projected_points /= projected_points[:, 2, None]

    # apply distortion
    projected_points = distortPoints(projected_points[:, :2], D, K)

    return projected_points


def decomposeEssentialMatrix(E):
    """ Given an essential matrix, compute the camera motion, i.e.,  R and T such
     that E ~ T_x R
     
     Input:
       - E(3,3) : Essential matrix

     Output:
       - R(3,3,2) : the two possible rotations
       - u3(3,1)   : a vector with the translation information
    """
    pass
    u, _, vh = np.linalg.svd(E)

    # Translation
    u3 = u[:, 2]

    # Rotations
    W = np.array([ [0, -1,  0],
                   [1,  0,  0],
                   [0,  0,  1]])

    R = np.zeros((3,3,2))
    R[:, :, 0] = u @ W @ vh
    R[:, :, 1] = u @ W.T @ vh

    for i in range(2):
        if np.linalg.det(R[:, :, i]) < 0:
            R[:, :, i] *= -1

    if np.linalg.norm(u3) != 0:
        u3 /= np.linalg.norm(u3)

    return R, u3

def disambiguateRelativePose(Rots,u3,points0_h,points1_h,K1,K2):
    """ DISAMBIGUATERELATIVEPOSE- finds the correct relative camera pose (among
     four possible configurations) by returning the one that yields points
     lying in front of the image plane (with positive depth).

     Arguments:
       Rots -  3x3x2: the two possible rotations returned by decomposeEssentialMatrix
       u3   -  a 3x1 vector with the translation information returned by decomposeEssentialMatrix
       p1   -  3xN homogeneous coordinates of point correspondences in image 1
       p2   -  3xN homogeneous coordinates of point correspondences in image 2
       K1   -  3x3 calibration matrix for camera 1
       K2   -  3x3 calibration matrix for camera 2

     Returns:
       R -  3x3 the correct rotation matrix
       T -  3x1 the correct translation vector

       where [R|t] = T_C2_C1 = T_C2_W is a transformation that maps points
       from the world coordinate system (identical to the coordinate system of camera 1)
       to camera 2.
    """
    pass

    # Projection matrix of camera 1
    M1 = K1 @ np.eye(3,4)

    total_points_in_front_best = 0
    for iRot in range(2):
        R_C2_C1_test = Rots[:,:,iRot]
        
        for iSignT in range(2):
            T_C2_C1_test = u3 * (-1)**iSignT
            
            M2 = K2 @ np.c_[R_C2_C1_test, T_C2_C1_test]
            P_C1 = linearTriangulation(points0_h, points1_h, M1, M2)
            
            # project in both cameras
            P_C2 = np.c_[R_C2_C1_test, T_C2_C1_test] @ P_C1
            
            num_points_in_front1 = np.sum(P_C1[2,:] > 0)
            num_points_in_front2 = np.sum(P_C2[2,:] > 0)
            total_points_in_front = num_points_in_front1 + num_points_in_front2
                  
            if (total_points_in_front > total_points_in_front_best):
                # Keep the rotation that gives the highest number of points
                # in front of both cameras
                R = R_C2_C1_test;
                T = T_C2_C1_test;
                total_points_in_front_best = total_points_in_front;

    return R, T

def linearTriangulation(p1, p2, M1, M2):
    """ Linear Triangulation
     Input:
      - p1 np.ndarray(3, N): homogeneous coordinates of points in image 1
      - p2 np.ndarray(3, N): homogeneous coordinates of points in image 2
      - M1 np.ndarray(3, 4): projection matrix corresponding to first image
      - M2 np.ndarray(3, 4): projection matrix corresponding to second image

     Output:
      - P np.ndarray(4, N): homogeneous coordinates of 3-D points
    """
    pass
    assert(p1.shape == p2.shape), "Input points dimension mismatch"
    assert(p1.shape[0] == 3), "Points must have three columns"
    assert(M1.shape == (3,4)), "Matrix M1 must be 3 rows and 4 columns"
    assert(M2.shape == (3,4)), "Matrix M1 must be 3 rows and 4 columns"

    num_points = p1.shape[1]
    P = np.zeros((4, num_points))

    # Linear Algorithm
    for i in range(num_points):
        # Build matrix of linear homogeneous system of equations
        A1 = cross2Matrix(p1[:, i]) @ M1
        A2 = cross2Matrix(p2[:, i]) @ M2
        A = np.r_[A1, A2]

        # Solve the homogeneous system of equations
        _, _, vh = np.linalg.svd(A, full_matrices=False)
        P[:, i] = vh.T[:,-1]

    # Dehomogenize (P is expressed in homoegeneous coordinates)
    P /= P[3,:]

    return P


def linearTriangulationBatch(p1, p2, M1, M2):
    """ Linear Triangulation
     Input:
      - p1 np.ndarray(3, N): homogeneous coordinates of points in image 1
      - p2 np.ndarray(3, N): homogeneous coordinates of points in image 2
      - M1 np.ndarray(N, 3, 4): projection matrix corresponding to first image
      - M2 np.ndarray(N, 3, 4): projection matrix corresponding to second image

     Output:
      - P np.ndarray(4, N): homogeneous coordinates of 3-D points
    """
    pass
    assert(p1.shape == p2.shape), "Input points dimension mismatch"
    assert(p1.shape[0] == 3), "Points must have three columns"

    num_points = p1.shape[1]
    P = np.zeros((4, num_points))

    # Linear Algorithm
    for i in range(num_points):
        # Build matrix of linear homogeneous system of equations
        A1 = cross2Matrix(p1[:, i]) @ M1[i]
        A2 = cross2Matrix(p2[:, i]) @ M2[i]
        A = np.r_[A1, A2]

        # Solve the homogeneous system of equations
        _, _, vh = np.linalg.svd(A, full_matrices=False)
        P[:, i] = vh.T[:,-1]

    # Dehomogenize (P is expressed in homoegeneous coordinates)
    P /= P[3,:]

    return P

def cross2Matrix(x):
    """ Antisymmetric matrix corresponding to a 3-vector
     Computes the antisymmetric matrix M corresponding to a 3-vector x such
     that M*y = cross(x,y) for all 3-vectors y.

     Input: 
       - x np.ndarray(3,1) : vector

     Output: 
       - M np.ndarray(3,3) : antisymmetric matrix
    """
    M = np.array([[0,   -x[2], x[1]], 
                  [x[2],  0,  -x[0]],
                  [-x[1], x[0],  0]])
    return M

def Matrix2Cross(M):
    """
    Computes the 3D vector x corresponding to an antisymmetric matrix M such that M*y = cross(x,y)
    for all 3D vectors y.
    Input:
     - M(3,3) : antisymmetric matrix
    Output:
     - x(3,1) : column vector
    See also CROSS2MATRIX
    """
    x = np.array([-M[1, 2], M[0, 2], -M[0, 1]])

    return x

def twist2HomogMatrix(twist):
    """
    twist2HomogMatrix Convert twist coordinates to 4x4 homogeneous matrix
    Input: -twist(6,1): twist coordinates. Stack linear and angular parts [v;w]
    Output: -H(4,4): Euclidean transformation matrix (rigid body motion)
    """
    v = twist[:3]  # linear part
    w = twist[3:]   # angular part

    se_matrix = np.concatenate([cross2Matrix(w), v[:, None]], axis=1)
    se_matrix = np.concatenate([se_matrix, np.zeros([1, 4])], axis=0)

    H = expm(se_matrix)

    return H

def HomogMatrix2twist(H):
    """
    HomogMatrix2twist Convert 4x4 homogeneous matrix to twist coordinates
    Input:
     -H(4,4): Euclidean transformation matrix (rigid body motion)
    Output:
     -twist(6,1): twist coordinates. Stack linear and angular parts [v;w]

    Observe that the same H might be represented by different twist vectors
    Here, twist(4:6) is a rotation vector with norm in [0,pi]
    """

    se_matrix = logm(H)

    # careful for rotations of pi; the top 3x3 submatrix of the returned se_matrix by logm is not
    # skew-symmetric (bad).

    v = se_matrix[:3, 3]

    w = Matrix2Cross(se_matrix[:3, :3])
    twist = np.concatenate([v, w])

    if np.iscomplexobj(twist):
        imax = np.max(np.abs(np.imag(twist)))
        if imax > 1e-8:
            raise ValueError(f"Twist has significant imaginary part (max |Im|={imax}). Check HomogMatrix2twist.")
        twist = np.real(twist)

    return twist

# def runBA(hidden_state, observations, K):
#     """
#     Update the hidden state, encoded as explained in the problem statement, with 20 bundle adjustment iterations.
#     """
#     with_pattern = True
#     hidden_state = hidden_state.astype(np.float32)
#     observations = observations.astype(np.float32)
#     K = K.astype(np.float32)

#     pattern = None
#     if with_pattern:
#         num_frames = int(observations[0])
#         num_observations = (observations.shape[0] - 2 - num_frames) / 3

#         # Factor 2, one error for each x and y direction.
#         num_error_terms = int(2 * num_observations)
#         # Each error term will depend on one pose (6 entries) and one landmark position (3 entries),
#         # so 9 nonzero entries per error term:
#         pattern = scipy.sparse.lil_matrix((num_error_terms, hidden_state.shape[0]), dtype=np.int8)
        
#         # Fill pattern for each frame individually:
#         observation_i = 2  # iterator into serialized observations
#         error_i = 0  # iterating frames, need another iterator for the error

#         for frame_i in range(num_frames):
#             num_keypoints_in_frame = int(observations[observation_i])
#             # All errors of a frame are affected by its pose.
#             pattern[error_i:error_i + 2 * num_keypoints_in_frame, frame_i*6:(frame_i + 1)*6] = 1

#             # Each error is then also affected by the corresponding landmark.
#             landmark_indices = observations[observation_i + 2 * num_keypoints_in_frame + 1:
#                                             observation_i + 3 * num_keypoints_in_frame + 1]

#             for kp_i in range(landmark_indices.shape[0]):
#                 pattern[error_i + kp_i * 2:error_i + (kp_i+1) * 2,
#                         num_frames * 6 + int(landmark_indices[kp_i] - 1) * 3:num_frames * 6 + int(landmark_indices[kp_i]) * 3] = 1


#             observation_i = observation_i + 1 + 3 * num_keypoints_in_frame
#             error_i = error_i + 2 * num_keypoints_in_frame

#         pattern = scipy.sparse.csr_matrix(pattern)

    def baError(hidden_state):
        plot_debug = False
        num_frames = int(observations[0])
        T_W_C = hidden_state[:num_frames * 6].reshape([-1, 6]).T
        p_W_landmarks = hidden_state[num_frames * 6:].reshape([-1, 3]).T

        error_terms = []
        
        # Iterator into the observations that are encoded as explained in the problem statement.
        observation_i = 1

        for i in range(num_frames):
            single_T_W_C = twist2HomogMatrix(T_W_C[:, i])
            num_frame_observations = int(observations[observation_i + 1])
            keypoints = np.flipud(observations[observation_i + 2:observation_i + 2 + num_frame_observations*2].reshape([-1, 2]).T)

            landmark_indices = observations[observation_i + 2 + num_frame_observations*2:observation_i + 2 + num_frame_observations * 3]
            
            # Landmarks observed in this specific frame.
            p_W_L = p_W_landmarks[:, landmark_indices.astype(np.int) - 1]
            
            # Transforming the observed landmarks into the camera frame for projection.
            T_C_W = np.linalg.inv(single_T_W_C)
            p_C_L = np.matmul(T_C_W[:3, :3], p_W_L.transpose(1, 0)[:, :, None]).squeeze(-1) + T_C_W[:3, -1]

            # From exercise 1.
            projections = projectPoints(p_C_L, K)
            
            # Can be used to verify that the projections are reasonable.
            if plot_debug:
                plt.clf()
                plt.close()
                plt.plot(projections[:, 0], projections[:, 1], 'o')
                plt.plot(keypoints[0, :], keypoints[1, :], 'x')
                plt.axis('equal')
                plt.show()

            error_terms.append(keypoints.transpose(1, 0) - projections)
            observation_i = observation_i + num_frame_observations * 3 + 1

        return np.concatenate(error_terms).flatten()

    res_1 = least_squares(baError, hidden_state, max_nfev=20, verbose=2, jac_sparsity=pattern)
    hidden_state = res_1.x

    return hidden_state

def cropProblem(hidden_state, observations, ground_truth, cropped_num_frames):
    # Determine which landmarks to keep; assuming landmark indices increase with frame indices.
    num_frames = int(observations[0])
    assert cropped_num_frames < num_frames

    observation_i = 2
    for i in range(cropped_num_frames):
        num_observations = int(observations[observation_i])
        if i == (cropped_num_frames - 1):
            cropped_num_landmarks = int(observations[observation_i+1+num_observations*2:observation_i+num_observations*3+1].max())
        observation_i = observation_i + num_observations * 3 + 1

    cropped_hidden_state = np.concatenate([hidden_state[:6*cropped_num_frames],
                                           hidden_state[6*num_frames:6*num_frames+3*cropped_num_landmarks]], axis=0)
    cropped_observations = np.concatenate([np.asarray([cropped_num_frames, cropped_num_landmarks]),
                                           observations[2:observation_i]], axis=0)
    cropped_ground_truth = ground_truth[:, :cropped_num_frames]

    return cropped_hidden_state, cropped_observations, cropped_ground_truth


def alignEstimateToGroundTruth(pp_G_C, p_V_C):
    """
    Returns the points of the estimated trajectory p_V_C transformed into the ground truth frame G.
    The similarity transform Sim_G_V is to be chosen such that it results in the lowest error between
    the aligned trajectory points p_G_C and the points of the ground truth trajectory pp_G_C.
    All matrices are 3xN

    Input:
    pp_G_C.shape: (3, N) ground truth
    p_V_C.shape: (3, N) estimate

    Intermediate: 
    Sim_G_V: similarity transform

    Output:
    p_G_C.shape: (3, N) aligned
    """
    if True:
        ### x: twisted
        ### S_GV: normal

        # pp_G_C_with_1 = np.vstack([pp_G_C, np.ones((1, pp_G_C.shape[1]))])
        # p_V_C_with_1 = np.vstack([p_V_C, np.ones((1, p_V_C.shape[1]))])
        def error_fun(x):
            T_GV = twist2HomogMatrix(x[:6])
            s_GV = x[6]
            f = s_GV * T_GV[:3, :3] @ p_V_C + T_GV[:3, 3].reshape(-1,1)
            Y = pp_G_C
            error = f - Y
            return error.flatten() 
        
        tau_0 = HomogMatrix2twist(np.eye(4)).reshape(-1, 1)
        # print(tau_0)
        x_0 = np.vstack([tau_0, np.ones((1, 1))]).reshape(-1)
        res = least_squares(error_fun, x_0)
        T_GV = twist2HomogMatrix(res.x[:6])
        s_GV = res.x[6]

        p_G_C = s_GV * T_GV[:3, :3] @ p_V_C + T_GV[:3, 3].reshape(-1,1)
        return p_G_C


def plotMap(hidden_state, observations, axis_range, title):
    plt.clf()
    plt.close()

    num_frames = int(observations[0])
    T_W_frames = hidden_state[:num_frames*6].reshape([-1, 6]).T
    p_W_landmarks = hidden_state[num_frames*6:].reshape([-1, 3]).T

    p_W_frames = np.zeros([3, num_frames])
    for i in range(num_frames):
        T_W_frame = twist2HomogMatrix(T_W_frames[:, i])
        p_W_frames[:, i] = T_W_frame[:3, -1]

    plt.plot(p_W_landmarks[2, :], -p_W_landmarks[0, :], '.')
    plt.plot(p_W_frames[2, :], -p_W_frames[0, :], 'rx', linewidth=3)

    plt.axis('equal')
    plt.xlim(axis_range[:2])
    plt.ylim(axis_range[2:])
    plt.title(title)
    plt.show()


def runBA(hidden_state, observations, K):
    """
    Update the hidden state, encoded as explained in the problem statement, with 20 bundle adjustment iterations.
    
    Input:
    -hidden_state.shape: (2610,)
    -observations.shape: (4329,)
    -K.shape: (3, 3)

    num_f: 4.0
    num_l: 862.0

    Output: 
    -hidden_state.shape: (2610,)

    """
    # TODO: Your code here

    # for i in range(len(observations)):
    #     print(i)
    #     if i >=2:
    #         print(f'k_i: {observations[i]}')
    # print(observations[2,0])
    # print(observations[2].shape)

    if True:
        n = int(observations[0]) # number of frames 
        m = observations[1] # number of total landmarks 
        

        

    
    
        jac_input = int(2*(observations.shape[0]-2-n)//3)
        jac_pattern = scipy.sparse.lil_matrix((jac_input, hidden_state.shape[0]), dtype=np.int8)
        i_error = 0
        k_index = 1
        k = 0
        for i_n in range(n):
            k_index += k*3+1
            k = int(observations[k_index])
            p_index = k_index + 1
            l_index = p_index + 2*k
            jac_pattern[i_error:i_error+k*2, i_n*6:(i_n+1)*6] = 1
            landmark_index_list = observations[l_index:l_index+k].astype(int)
            for i in range(landmark_index_list.shape[0]):
                jac_pattern[i_error+i*2:i_error+(i+1)*2, 
                            n*6+int(landmark_index_list[i]-1)*3:n*6+int(landmark_index_list[i])*3] = 1
            i_error += 2*k
        jac_pattern = scipy.sparse.csr_matrix(jac_pattern) 

        def error_fun(x):
            # print(f'n: {n}')
            taus = x[:n*6]
            Ps = x[n*6:]
            Ps = Ps.reshape(-1, 3)
            k_index = 1
            k = 0
            f_list = []
            Y_list = []
            
            for i_n in range(n):
                k_index += k*3+1
                k = int(observations[k_index])
                p_index = k_index + 1
                l_index = p_index + 2*k

                

                T_WC = twist2HomogMatrix(taus[i_n*6:(i_n+1)*6])
                T_CW = np.linalg.inv(T_WC)

                landmark_index_list = observations[l_index:l_index+k].astype(int)
                P_W = Ps[landmark_index_list-1].T # (3,N)
                P_C = T_CW[:3, :3] @ P_W + T_CW[:3, 3].reshape(-1,1)

                # print(f'P_C: {P_C.shape}')
                
                projection = projectPoints(P_C.T, K)
                # i_error += projection.flatten(order='C').reshape(-1,1).shape[0]
                f_list.append(projection.flatten(order='C').reshape(-1,1))
                # print(projection.flatten(order='F').shape)

                obs = observations[p_index:p_index+k*2].reshape(-1,2)
                obs = obs[:, ::-1].reshape(-1,1)
                Y_list.append(obs)

                
            # print(f_list)
            f = np.vstack(f_list)
            Y = np.vstack(Y_list)   
                        
                
            if False:
                # create f
                taus = x[:n]
                Ps = x[n:].T # (3, m)
                f_list = []
                for i_n in range(n):
                    T = twist2HomogMatrix(taus[i_n])
                    Ps_W = T[:3, :3] @ Ps + T[:3, 3].reshape(-1,1) # (3, m)
                    ps = K @ (Ps_W/Ps_W[2, :])[:2, :] # (2, m)
                    f_list.append(ps.T) # (m, 2)
                f = np.vstack(f_list) # total landmarks * frames (m*n, 2)
                
                # create Y
                Y_list = []
                for i_n in range(n):
                    O_i = observations[i_n+2]
                    k_i = O_i[0] # number of observed landmarks
                    Y_list.append(O_i[1:k_i+1])
                Y = np.vstack(Y_list) # (k*n, 2)
            
            error = f - Y
            # print(f'error.shape: {error.shape}')
            return error.reshape(-1)

        x_0 = hidden_state

        res = least_squares(error_fun, x_0, jac_sparsity=jac_pattern, max_nfev=20, verbose=2)
        hidden_state = res.x

        return hidden_state