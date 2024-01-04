from numpy import linalg as LA
import argparse
import get_dataset_info as dataset
import computer_vision as cv
import pipeline as pl


parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=int, required=True)
parser.add_argument('-T_robust', action='store_true')
args = parser.parse_args()
print('Dataset:', args.dataset, 'T_robust:', args.T_robust)


print('\n\n\n### Initializing ###\n')

data_set = args.dataset-1
T_robust = args.T_robust
K, img_names, init_pair, pixel_threshold = dataset.get_dataset_info(data_set)
K_inv = LA.inv(K)
imgs = cv.load_image(img_names, multi=True)
n_imgs = imgs.shape[0]
n_camera_pairs = n_imgs-1

abs_rots, x1_norm_RA, x2_norm_RA, inliers_RA = pl.compute_rotation_averaging(imgs, init_pair, K, pixel_threshold, plot=False)
x1_init_norm_feasible_inliers, x2_init_norm_feasible_inliers, des1_init_feasible_inliers, des2_init_feasible_inliers, X_init_feasible_inliers, X_init_idx = pl.compute_initial_3D_points(imgs, init_pair, K, pixel_threshold, plot=False)
trans, valid_cameras, x_norm_TR, X_idx_TR, inliers_TR = pl.compute_translation_registration(K, imgs, init_pair, 10*pixel_threshold, abs_rots, x1_init_norm_feasible_inliers, x2_init_norm_feasible_inliers, des1_init_feasible_inliers, X_init_feasible_inliers, X_init_idx, ransac=T_robust)
abs_rots_opt, trans_opt = pl.refine_rotations_and_translations(trans, abs_rots, X_init_feasible_inliers, valid_cameras, X_idx_TR, x_norm_TR, inliers_TR)
cameras = pl.create_cameras(abs_rots, trans)
cameras_opt = pl.create_cameras(abs_rots_opt, trans_opt)
pl.triangulate_final_3D_reconstruction(imgs, K, pixel_threshold, cameras, valid_cameras, inliers_RA, x1_norm_RA, x2_norm_RA, 'Final 3D Reconstruction with LM=False', T_robust)
pl.triangulate_final_3D_reconstruction(imgs, K, pixel_threshold, cameras_opt, valid_cameras, inliers_RA, x1_norm_RA, x2_norm_RA, 'Final 3D Reconstruction with LM=True', T_robust)