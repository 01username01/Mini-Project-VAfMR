import os
import cv2 as cv
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

from initialization import initialization
from process_Frame import processFrame
from init_scale_template import init_scale_template
from scale_correction import scale_correction
from utils import *

# --- Setup ---
ds = 0  # 0: KITTI, 1: Malaga, 2: Parking, 3: Own Dataset
BA_param = 5 # after how many frames should BA be executed
correct_scale = True

# Define dataset paths
# (Set these variables before running)
kitti_path = "data\kitti"
malaga_path = "data\malaga-urban-dataset-extract-07"
parking_path = "data\parking"
# own_dataset_path = "/path/to/own_dataset"

if ds == 0:
    assert 'kitti_path' in locals(), "You must define kitti_path"
    ground_truth = np.loadtxt(os.path.join(kitti_path, 'poses', '05.txt'))
    ground_truth = ground_truth[:, [-9, -1]]  # same as MATLAB(:, [end-8 end])
    last_frame = 4540
    K = np.array([
        [7.18856e+02, 0, 6.071928e+02],
        [0, 7.18856e+02, 1.852157e+02],
        [0, 0, 1]
    ])
elif ds == 1:
    assert 'malaga_path' in locals(), "You must define malaga_path"
    img_dir = os.path.join(malaga_path, 'malaga-urban-dataset-extract-07_rectified_800x600_Images')
    left_images = sorted(glob(os.path.join(img_dir, '*.png')))
    last_frame = len(left_images)
    K = np.array([
        [621.18428, 0, 404.0076],
        [0, 621.18428, 309.05989],
        [0, 0, 1]
    ])
elif ds == 2:
    assert 'parking_path' in locals(), "You must define parking_path"
    last_frame = 598
    K = np.loadtxt(os.path.join(parking_path, 'K.txt'))
    ground_truth = np.loadtxt(os.path.join(parking_path, 'poses.txt'))
    ground_truth = ground_truth[:, [-9, -1]]
elif ds == 3:
    # Own Dataset
    # TODO: define your own dataset and load K obtained from calibration of own camera
    assert 'own_dataset_path' in locals(), "You must define own_dataset_path"

else:
    raise ValueError("Invalid dataset index")

# --- Bootstrap ---
bootstrap_frames_dict = {0 : [0, 3], 1 : [0, 1], 2 : [0, 1], 3 : [0, 1]}
bootstrap_frames = bootstrap_frames_dict[ds]

if ds == 0:
    img0 = cv.imread(os.path.join(kitti_path, '05', 'image_0', f"{bootstrap_frames[0]:06d}.png"), cv.IMREAD_GRAYSCALE)
    img1 = cv.imread(os.path.join(kitti_path, '05', 'image_0', f"{bootstrap_frames[1]:06d}.png"), cv.IMREAD_GRAYSCALE)
elif ds == 1:
    img0 = cv.imread(left_images[bootstrap_frames[0]], cv.IMREAD_GRAYSCALE)
    img1 = cv.imread(left_images[bootstrap_frames[1]], cv.IMREAD_GRAYSCALE)
elif ds == 2:
    img0 = cv.imread(os.path.join(parking_path, 'images', f"img_{bootstrap_frames[0]:05d}.png"), cv.IMREAD_GRAYSCALE)
    img1 = cv.imread(os.path.join(parking_path, 'images', f"img_{bootstrap_frames[1]:05d}.png"), cv.IMREAD_GRAYSCALE)
elif ds == 3:
    # Load images from own dataset
    img0 = cv.imread(os.path.join(own_dataset_path, f"{bootstrap_frames[0]:06d}.png"), cv.IMREAD_GRAYSCALE)
    img1 = cv.imread(os.path.join(own_dataset_path, f"{bootstrap_frames[1]:06d}.png"), cv.IMREAD_GRAYSCALE)
else:
    raise ValueError("Invalid dataset index")

#correction for scale drift initialization
if correct_scale:
    target_height, target_width = img0.shape
    scale_template = cv.imread(os.path.join(kitti_path, '05', 'template.png'), cv.IMREAD_GRAYSCALE)
    height, width = scale_template.shape

    blank_image = np.full((target_height,target_width), 255)
    height_offset = int((target_height-height)/2)
    width_offset = int((target_width-width)/2)
    # Here, y_offset+height <= blank_image.shape[0] and x_offset+width <= blank_image.shape[1]
    blank_image[height_offset:height_offset+height, width_offset:width_offset+width] = scale_template
    scale_template = blank_image.astype(np.uint8)
    points_3d_marker, points_2d_marker = init_scale_template(scale_template, 0.2, height)
    marker_template = {"template_img": scale_template, "pts_2D_marker": points_2d_marker, "pts_3D_marker": points_3d_marker}
    starting_state = {"starting_keypoints": None, "starting_img": None, "starting_t": None, "T_cw_start": None, "index": 0}


# Initialization
# points_3D : (num_inliers, 3), tans_mat : (3, 4), inlier_1 : (num_inlier, 2) (from the second image)
points_3D, trans_mat, inliers_1 = initialization(img0, img1, K)

# Calculate camera position after the first frame
R_cw = trans_mat[:, :3]
R_wc = R_cw.T 
t_cw = trans_mat[:, 3]
t_wc = -R_wc @ t_cw
print(f"Camera position after the first frame: {t_wc}")

# # Creat birds eye view plot containing 3d landmarks aswell as the cam pos
# X_3d_points = points_3D[:, 0]
# Z_3d_points = points_3D[:, 2]

# X_cam = [0, t_wc[0]]
# Z_cam = [0, t_wc[2]]

# plt.ion() 

# fig, ax = plt.subplots(figsize=(6, 6))

# sc_points = ax.scatter(X_3d_points, Z_3d_points, s=5, c="blue", label="3D points")

# sc_cam = ax.scatter(X_cam, Z_cam, s=30, c="red", label="Camera")

# ax.set_xlabel("X (left-right)")
# ax.set_ylabel("Z (forward)")
# ax.set_title("Bird's Eye View")
# ax.axis("equal")
# ax.grid(True)
# ax.legend()

# plt.show()

"""# Needed for BA
# Calculate the twist from the transformation matrix
R_cw = trans_mat[:, :3]
R_wc = R_cw.T 
t_cw = trans_mat[:, 3]
t_wc = -R_wc @ t_cw # points from world frame to camera frame expressed in world coordinates
# Assamble the transformation matrix in the opposit direction
T_wc = np.eye((4,4), dtype=float) # (4, 4)
T_wc[0:3, 0:3] = R_wc
T_wc[0:3, 3] = t_wc
# Calculate the twist
twist = HomogMatrix2twist(T_wc) # (6, 1)"""

# Check keyframe distance
b = np.linalg.norm(t_wc)
print(f"b: {b}")
z = np.mean(points_3D, axis=0)[2]
print(f"Z: {z}")
metric = (b/z)
print(f"b/z= {metric}")


# Initialize the state and the pose
# P : keypoints/features (2, K), X : 3D_points /landmarks (3, K)
S_current = {
    "P": inliers_1.T,         # (2, K)
    "X": points_3D.T,         # (3, K)
    "C": np.zeros((2, 0)),    # (2, 0)
    "F": np.zeros((2, 0)),    # (2, 0)
    "T": np.zeros((12, 0)),   # (12, 0)
}

"""# Plot ground truth trajectory
plt.figure(figsize=(8, 6))
plt.plot(ground_truth[:, 0], ground_truth[:, 1], 'b-', label='Ground Truth')
plt.xlabel("X (meters)")
plt.ylabel("Z (meters)")
plt.title("Ground Truth Trajectory (X-Z)")
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()"""

prev_img = img1

"""# For BA
twists = (twist.T).copy() # (1, 6) 
landmarks_3D = S_current["X"].copy() # (3, K)
keypoints_2D = list((S_current["P"].copy())[None, :, :]) # (1, 2, K)
label = [[range(landmarks_3D.shape[1])]] # (1, num_landmarks)
"""

# Trajectory log
trajectory_x = []
trajectory_z = []
trajectory_t_wc = []

# Counters
keypoint_counter = []
candidate_counter = []

# Create plots
plt.ion()

# Trajectory plot
fig_traj, ax_traj = plt.subplots()
ax_traj.set_title("Camera Trajectory")
ax_traj.set_xlabel("X")
ax_traj.set_ylabel("Z")
ax_traj.grid(True)
ax_traj.set_aspect('equal', adjustable='datalim')

# Keypoint/candidate counter plot
fig_count, ax_count = plt.subplots()
ax_count.set_title("Keypoints per frame")
ax_count.set_xlabel("Frame index")
ax_count.set_ylabel("Count")
ax_count.grid(True)

ln_landmarks, = ax_count.plot([], [], label="P (#landmarks)")
ln_candidates, = ax_count.plot([], [], label="C (#candidates)")
ax_count.legend()

"""# Video
# fourcc = cv.VideoWriter_fourcc(*"mp4v")
# out_video = cv.VideoWriter("vo_features.mp4", fourcc, 20, (img1.shape[1], img1.shape[0]))
"""
T_next = trans_mat
first_frame = bootstrap_frames[1] + 1

# --- Continuous operation ---
for i in range(bootstrap_frames[1] + 1, last_frame + 1):
    print(f"\n\nProcessing frame {i}\n=====================")
    
    if ds == 0:
        image_path = os.path.join(kitti_path, '05', 'image_0', f"{i:06d}.png")
    elif ds == 1:
        image_path = left_images[i]
    elif ds == 2:
        image_path = os.path.join(parking_path, 'images', f"img_{i:05d}.png")
    elif ds == 3:
        image_path = os.path.join(own_dataset_path, f"{i:06d}.png")
    else:
        raise ValueError("Invalid dataset index")
    
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Warning: could not read {image_path}")
        continue
    
    # Simulate 'pause(0.01)'
    cv.waitKey(10)

    # Creat everything needed for BA
    hidden_state = []
    observations = []

    # Process the images
    S_next, T_next = processFrame(prev_img, image, S_current, K)

    
    # # Check if it is time for bundle adjustment
    # if (i + 1) % BA_param == 0:
    #     optimized_hidden_state = runBA(hidden_state, observations, K)
    

    # Calculate camera position after the first frame
    R_cw = T_next[:, :3]
    R_wc = R_cw.T 
    t_cw = T_next[:, 3]
    t_wc = -R_wc @ t_cw
    print(f"Camera position after the first frame: {t_wc}")

    # # Update the birds eye plot
    # P = S_next["X"].T[:, [0, 2]]
    # print(P.shape)
    # C = [[t_wc[0], t_wc[2]]]              
    # sc_points.set_offsets(P)
    # sc_cam.set_offsets(C)
    # plt.pause(0.001)

    # Update the camera trajectory plot
    trajectory_x.append(t_wc[0])
    trajectory_z.append(t_wc[2])
    trajectory_t_wc.append(t_wc)

    # Do the scale correction
    if correct_scale:
        correction_found, new_start_found, starting_state, S_corrected, t_corrected_norm = scale_correction(image, S_next, K, marker_template, starting_state)
        if new_start_found:
            starting_state["T_cw_start"] = T_next
            starting_state["index"] = i
        if correction_found:
            print("Correcting scale")
            start_frame = starting_state["index"]
            S_next = S_corrected
            t_wc_start = trajectory_t_wc[start_frame-first_frame]
            scaling_factor = t_corrected_norm/np.linalg.norm(t_wc-t_wc_start)
            for j in range(start_frame, i+1):
                trajectory_t_wc[j-first_frame] *= scaling_factor
                trajectory_x[j-first_frame] *= scaling_factor
                trajectory_z[j-first_frame] *= scaling_factor


    ax_traj.plot(trajectory_x, trajectory_z, 'b-')
    plt.pause(0.001)

    
    # Update the counter plot
    keypoint_counter.append(S_next["P"].shape[1])
    candidate_counter.append(S_next["C"].shape[1])

    ln_landmarks.set_xdata(range(len(keypoint_counter)))
    ln_landmarks.set_ydata(keypoint_counter)

    ln_candidates.set_xdata(range(len(candidate_counter)))
    ln_candidates.set_ydata(candidate_counter)

    ax_count.relim()
    ax_count.autoscale_view()
    plt.pause(0.001)

    prev_img = image
    S_current = S_next
    
    """# Update the lists for BA

    # Get the correct twistvector
    R_cw = T_next[:, :3]
    R_wc = R_cw.T 
    t_cw = T_next[:, 3]
    t_wc = -R_wc @ t_cw # points from world frame to camera frame expressed in world coordinates
    # Assamble the transformation matrix in the opposit direction
    T_wc = np.eye((4,4), dtype=float) # (4, 4)
    T_wc[0:3, 0:3] = R_wc
    T_wc[0:3, 3] = t_wc
    # Calculate the twist
    twist = HomogMatrix2twist(T_wc) # (6, 1)
    twists.append()

    # Append the new landwarks
    landmarks_3D = np.concatenate([landmarks_3D, additional_x], axis=1) # (3, total_num_landmarks)
    indices = np.argmax(np.all(landmarks_3D[:, :, None] == S_next["X"][:, None, :], axis=0), axis=0)
    label.append(indices)
    keypoints_2D.append(S_next["P"])
    

    # Check if it is time for bundle adjustment
    if (i + 1) % BA_param == 0:
        optimized_hidden_state = runBA(hidden_state, observations, K)"""

    """# ====== FEATURE VISUALIZATION FOR VIDEO ======
    vis = cv.cvtColor(image, cv.COLOR_GRAY2BGR)

    # Draw P (green)
    for (x,y) in S_next["P"].T:
        cv.circle(vis, (int(x), int(y)), 2, (0,255,0), -1)

    # Draw C (blue)
    for (x,y) in S_next["C"].T:
        cv.circle(vis, (int(x), int(y)), 2, (255,0,0), -1)

    out_video.write(vis)
    cv.imshow("Features", vis)
    """
    
"""out_video.release()
cv.destroyAllWindows()"""
