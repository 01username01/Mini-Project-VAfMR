import os
import cv2 as cv
cv.setRNGSeed(42)
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from pathlib import Path

from initialization import initialization
from process_Frame import processFrame
from BundleAdjustment import do_BundleAdjustment
from init_scale_template import init_scale_template
from scale_correction import scale_correction
from utils import *

# --- Setup ---
ds = 0  # 0: KITTI, 1: Malaga, 2: Parking, 3: Own Dataset
path_length = 160 # 160

# Define dataset paths
# (Set these variables before running)
BASE_DIR = Path.cwd() # OS-independent path
DATA_DIR = BASE_DIR / "data"
kitti_path = DATA_DIR / "kitti"
malaga_path = DATA_DIR / "malaga-urban-dataset-extract-07"
parking_path = DATA_DIR / "parking"
correct_scale = True


if ds == 0:
    assert 'kitti_path' in locals(), "You must define kitti_path"
    ground_truth = np.loadtxt(os.path.join(kitti_path, 'poses', '05.txt'))
    ground_truth = ground_truth[:, [-9, -1]]  # same as MATLAB(:, [end-8 end])
    ground_truth_3d = np.loadtxt(os.path.join(kitti_path, 'poses', '05.txt'))[:, [3, 7, 11]]
    last_frame = 4540
    K = np.array([
        [7.18856e+02, 0, 6.071928e+02],
        [0, 7.18856e+02, 1.852157e+02],
        [0, 0, 1]
    ])
elif ds == 1:
    assert 'malaga_path' in locals(), "You must define malaga_path"
    img_dir = os.path.join(malaga_path, 'malaga-urban-dataset-extract-07_rectified_800x600_Images')
    left_images = sorted(glob(os.path.join(img_dir, '*left.jpg')))
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
    ground_truth_3d = np.loadtxt(os.path.join(parking_path, 'poses.txt'))[:, [3, 7, 11]]
elif ds == 3:
    # Own Dataset
    # TODO: define your own dataset and load K obtained from calibration of own camera
    assert 'own_dataset_path' in locals(), "You must define own_dataset_path"

else:
    raise ValueError("Invalid dataset index")

# --- Bootstrap ---
bootstrap_frames_dict = {0 : [0, 3], 1 : [0, 1], 2 : [0, 2], 3 : [0, 1]}
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
    scale_template = cv.imread(os.path.join(DATA_DIR, 'template.jpeg'), cv.IMREAD_GRAYSCALE)
    scale_template = cv.resize(scale_template, (200, 200))
    height, width = scale_template.shape

    blank_image = np.full((target_height,target_width), 255)
    height_offset = int((target_height-height)/2)
    width_offset = int((target_width-width)/2)
    # Here, y_offset+height <= blank_image.shape[0] and x_offset+width <= blank_image.shape[1]
    blank_image[height_offset:height_offset+height, width_offset:width_offset+width] = scale_template
    scale_template = blank_image.astype(np.uint8)
    points_3d_marker, points_2d_marker = init_scale_template(scale_template, 0.16, height)
    marker_template = {"template_img": scale_template, "pts_2D_marker": points_2d_marker, "pts_3D_marker": points_3d_marker}
    starting_state = {"starting_keypoints": None, "starting_img": None, "starting_t": None, "T_cw_start": None, "index": 0}

    #calculate initial scale
    pts_2D_0, status_0, err_marker = cv.calcOpticalFlowPyrLK(scale_template, img0, points_2d_marker, None)
    detected_pts_2D_0 = pts_2D_0[status_0 == 1] # (num_det_pts, 2)
    detected_pts_3D_0 = points_3d_marker[status_0 == 1] # (num_det_pts, 3)
    _, _, tvec_0, _ = cv.solvePnPRansac(detected_pts_3D_0, detected_pts_2D_0, K, None, flags=cv.SOLVEPNP_P3P, iterationsCount=100, reprojectionError=2.0, confidence=0.999)
    pts_2D_1, status_1, err_marker = cv.calcOpticalFlowPyrLK(scale_template, img1, points_2d_marker, None)
    detected_pts_2D_1 = pts_2D_1[status_1 == 1] # (num_det_pts, 2)
    detected_pts_3D_1 = points_3d_marker[status_1 == 1] # (num_det_pts, 3)
    _, _, tvec_1, _ = cv.solvePnPRansac(detected_pts_3D_1, detected_pts_2D_1, K, None, flags=cv.SOLVEPNP_P3P, iterationsCount=100, reprojectionError=2.0, confidence=0.999)
    scaling_factor = np.linalg.norm(tvec_1-tvec_0)
else:
    scaling_factor = 1


# Initialization
# points_3D : (num_inliers, 3), tans_mat : (3, 4), inlier_1 : (num_inlier, 2) (from the second image)
points_3D, trans_mat, inliers_1 = initialization(img0, img1, K, scaling_factor)

# Calculate camera position after the first frame
R_cw = trans_mat[:, :3]
R_wc = R_cw.T 
t_cw = trans_mat[:, 3]
t_wc = -R_wc @ t_cw
print(f"Camera position after the first frame: {t_wc}")
print(R_cw)

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

# Check keyframe distance
print(f"number of 3D points: {points_3D.shape}")
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
    "T_it": np.zeros((1,0))
}

# Plot ground truth trajectory
plt.figure(figsize=(8, 6))
plt.plot(ground_truth[:, 0], ground_truth[:, 1], 'b-', label='Ground Truth')
plt.scatter(ground_truth[0, 0], ground_truth[1, 0], color="blue")
plt.text(ground_truth[0, 0], ground_truth[1, 0], "start", color="blue", fontsize=12, verticalalignment="center")
plt.xlabel("X (meters)")
plt.ylabel("Z (meters)")
plt.title("Ground Truth Trajectory (X-Z)")
plt.grid(True)
plt.axis('equal')
plt.legend()
if ds == 0:
    dataset = 'kitti'
elif ds == 1:
    dataset = 'malaga'
elif ds == 2:
    dataset = 'parking'
elif ds == 3:
    dataset = 'selfmade'
else:
    dataset = 'undefined_dataset'
plt.savefig("plots/path_ground_truth_"+dataset+".png", dpi=300, bbox_inches="tight")
# plt.show()


prev_img = img1

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
ax_traj.set_title("ONLY CORRECT IF BA IS DISABLED - Camera Trajectory")
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

# BA stuff
is_BA_active = True
BA_repetion_rate = 5  # 5, 1 # How often BA is repeated  
BA_window = 20  # 10, 20 # How many frames are optimized over
BA_counter = 0  # iterator
S_list = []  # list of all S
T_CW_list = []  # list of all T 
### also see variable: 'path_length'
T_next = trans_mat
first_frame = bootstrap_frames[1] + 1

# --- Continuous operation ---
for i in range(bootstrap_frames[1] + 1, last_frame + 1):
    print(f"\n\nProcessing frame {i}\n=====================")
    if i == path_length + 4:
        break
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
    T_CW_index = len(T_CW_list)
    S_next, T_next = processFrame(prev_img, image, S_current, K, T_CW_index)
    S_list.append(S_next)
    T_CW_list.append(T_next)


    BA_counter += 1
    print(f'BA_counter: {BA_counter}')
    print(f'len(S_list): {len(S_list)}')
    if (BA_counter >= BA_repetion_rate and len(S_list) >= BA_window and is_BA_active):
        S_list, T_CW_list = do_BundleAdjustment(S_list, T_CW_list, K, BA_window, max_nfev=30)  # max_nfev = 30
        S_next = S_list[-1]
        T_next = T_CW_list[-1]
        BA_counter = 0    
    

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

if ds == 0:
    dataset = 'kitti'
elif ds == 1:
    dataset = 'malaga'
elif ds == 2:
    dataset = 'parking'
elif ds == 3:
    dataset = 'selfmade'
else:
    dataset = 'undefined_dataset'

# method = 'basic'
# ax_traj.figure.savefig("plots_BA/path_"+method+"_"+dataset+".png", dpi=300, bbox_inches="tight")

ground_truth_3d_cropped = ground_truth_3d[4:path_length+4]

path = []
for T in T_CW_list:
    R_cw = T[:, :3]
    R_wc = R_cw.T 
    t_cw = T[:, 3]
    t_wc = -R_wc @ t_cw
    path.append(t_wc.reshape(1,3))
path = np.vstack(path)

aligned_path = alignEstimateToGroundTruth(ground_truth_3d_cropped.T, path.T).T

if is_BA_active:
    BA_label = 'BA'
else:
    BA_label = 'noBA'

# aligned path (works with and without BA)
plt.ioff()
fig, ax = plt.subplots()
ax.plot(ground_truth_3d_cropped[:, 0], ground_truth_3d_cropped[:, 2], label='Ground Truth')
# ax.plot(no_BA_path[:, 0], no_BA_path[:, 2], label='no_BA_path')
ax.plot(path[:, 0], path[:, 2], label=BA_label + '_path')
plt.scatter(ground_truth_3d_cropped[0, 0], ground_truth_3d_cropped[0, 2], color="black")
# plt.scatter(no_BA_path[0, 0], no_BA_path[0, 2], color="black")
plt.scatter(path[0, 0], path[0, 2], color="black")
ax.set_aspect('equal', adjustable='datalim')
ax.grid(True)
ax.legend()
plt.savefig("plots/"+BA_label+"_non-aligned_"+dataset+".png", dpi=300, bbox_inches="tight")
plt.show()  

# aligned path (works with and without BA)
plt.ioff()
fig, ax = plt.subplots()
ax.plot(ground_truth_3d_cropped[:, 0], ground_truth_3d_cropped[:, 2], label='Ground Truth')
# ax.plot(no_BA_aligned_path[:, 0], no_BA_aligned_path[:, 2], label='no_BA_aligned_path')
ax.plot(aligned_path[:, 0], aligned_path[:, 2], label=BA_label + '_aligned_path')
plt.scatter(ground_truth_3d_cropped[0, 0], ground_truth_3d_cropped[0, 2], color="black")
# plt.scatter(no_BA_aligned_path[0, 0], no_BA_aligned_path[0, 2], color="black")
plt.scatter(aligned_path[0, 0], aligned_path[0, 2], color="black")
ax.set_aspect('equal', adjustable='datalim')
ax.grid(True)
ax.legend()
plt.savefig("plots/"+BA_label+"_aligned_"+dataset+".png", dpi=300, bbox_inches="tight")
plt.show()  


"""out_video.release()
cv.destroyAllWindows()"""
