import numpy as np
from numpy import linalg as LA
from scipy.io import loadmat
import scipy
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.pyplot import cm
import matplotlib as mpl
import cv2
import computer_vision as cv
import pipeline as pl
from tqdm import trange
import time
from get_dataset_info import *
from scipy.spatial.transform import Rotation
import sys


print('\n\n\n### Initializing ###\n')

data_set = 1
K, img_names, init_pair, pixel_threshold = get_dataset_info(data_set)
K_inv = LA.inv(K)
imgs = cv.load_image(img_names, multi=True)
n_imgs = imgs.shape[0]
n_camera_pairs = n_imgs-1

abs_rots, x1_norm_RA, x2_norm_RA, inliers_RA = pl.compute_rotation_averaging(imgs, init_pair, K, pixel_threshold, plot=False)
x1_init_norm_feasible_inliers, x2_init_norm_feasible_inliers, des1_init_feasible_inliers, des2_init_feasible_inliers, X_init_feasible_inliers, X_init_idx = pl.compute_initial_3D_points(imgs, init_pair, K, pixel_threshold, plot=False)
trans, valid_cameras, x_norm_TR, X_idx_TR, inliers_TR = pl.compute_translation_registration(K, imgs, init_pair, pixel_threshold, abs_rots, x1_init_norm_feasible_inliers, x2_init_norm_feasible_inliers, des1_init_feasible_inliers, X_init_feasible_inliers, X_init_idx, ransac=True)
abs_rots_opt, trans_opt = pl.refine_rotations_and_translations(trans, abs_rots, X_init_feasible_inliers, valid_cameras, X_idx_TR, x_norm_TR, inliers_TR)
# cameras = pl.create_cameras(abs_rots, trans)
cameras = pl.create_cameras(abs_rots_opt, trans_opt)
pl.triangulate_final_3D_reconstruction(imgs, K, pixel_threshold, cameras, valid_cameras, inliers_RA, x1_norm_RA, x2_norm_RA)