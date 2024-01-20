import numpy as np
from numpy import linalg as LA
from scipy.io import loadmat
from scipy import optimize
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.pyplot import cm
import matplotlib as mpl
import computer_vision as cv
from scipy.spatial.transform import Rotation


def plot_cameras_and_axes(ax, C_list, axis_list, s, valid_idx, col):

    for i in range(np.size(C_list,1)):

        C = C_list[:,i]
        axis = axis_list[:,i]
        ax.plot(C[0], C[1], C[2], 'o', color=col[i],  label='Camera {}'.format(valid_idx[i]+1), alpha=0.7)

        x_axis = C[0] + s*axis[0]
        y_axis = C[1] + s*axis[1]
        z_axis = C[2] + s*axis[2]

        ax.plot([x_axis, C[0]], [y_axis, C[1]], [z_axis, C[2]], '-', color=col[i], lw=3, alpha=0.7)

def plot_cameras_and_3D_points(X_arr, C_arr, axis_arr, s, title, valid_idx, multi=False):
    
    fig = plt.figure(figsize=(12,8))
    ax = plt.axes(projection='3d')
    col = cm.rainbow(np.linspace(0, 1, np.size(C_arr,1)))

    if multi:
        for i in range(len(X_arr)):
            X = X_arr[i]
            ax.plot(X[0], X[1], X[2], '.', color=col[i], ms=0.8)
    else:
        ax.plot(X_arr[0], X_arr[1], X_arr[2], '.', color='magenta', ms=0.4)
    plot_cameras_and_axes(ax, C_arr, axis_arr, s, valid_idx, col)

    ax.set_xlabel(r'$X$')
    ax.set_ylabel(r'$Y$')
    ax.set_zlabel(r'$Z$')
    ax.set_aspect('equal')
    ax.set_title(title)
    # ax.view_init(elev=-45, azim=-45, roll=180)
    fig.tight_layout()
    plt.legend(loc="lower right")
    plt.show()

def plot_3D_points(X):
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot(X[0], X[1], X[2], '.', ms=1, color='magenta', label='X')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    ax.set_aspect('equal')
    ax.legend(loc="lower right")
    fig.tight_layout()
    plt.show()

def compute_rotation_averaging(imgs, init_pair, K, pixel_threshold, plot=False):
    print('\n\n\n### Computing rotation averaging ###\n')

    K_inv = LA.inv(K)
    n_imgs = imgs.shape[0]
    n_camera_pairs = n_imgs-1

    marg = 0.75
    min_its = 15000
    max_its = 20000
    scale_its = 4
    alpha = 0.99
    P1 = cv.get_canonical_camera()
    rel_cameras = [P1]

    x1_norm_RA = []
    x2_norm_RA = []
    inliers_RA = []

    for i in range(n_camera_pairs):    
        print('\nCamera pair:', i+1, '/', n_camera_pairs)

        img1 = imgs[i]
        img2 = imgs[i+1]
        x1, x2, _, _ = cv.compute_sift_points(img1, img2, marg, flann=True, verbose=True)
        x1_norm = cv.dehomogenize(K_inv @ x1)
        x2_norm = cv.dehomogenize(K_inv @ x2)
        x1_norm_RA.append(x1_norm)
        x2_norm_RA.append(x2_norm)

        E, inliers = cv.estimate_E_robust(K, x1_norm, x2_norm, min_its, max_its, scale_its, alpha, pixel_threshold, essential_matrix=True, homography=True, verbose=True)
        inliers_RA.append(inliers)

        x1_norm_inliers = x1_norm[:,inliers]
        x2_norm_inliers = x2_norm[:,inliers]

        P2_arr = cv.extract_P_from_E(E)
        X_arr = cv.compute_triangulated_X_from_extracted_P2_solutions(P1, P2_arr, x1_norm_inliers, x2_norm_inliers)
        P2, X = cv.extract_valid_camera_and_points(P1, P2_arr, X_arr, verbose=True)
        rel_cameras.append(P2)

        if plot:
            percentile = 90
            feasable_pts = cv.compute_feasible_points(P1, P2, X, percentile)
            P_arr = np.array([P1, P2])
            C_arr, axis_arr = cv.compute_camera_center_and_normalized_principal_axis(P_arr, multi=True)
            plot_cameras_and_3D_points(X[:,feasable_pts], C_arr, axis_arr, s=1, title=None, valid_idx=[0,1], multi=False)

    rel_cameras = np.array(rel_cameras)
    rel_rots = rel_cameras[:,:,:-1]
    abs_rots = cv.compute_absolute_rotations(rel_rots, init_pair[0], verbose=True)
    
    return abs_rots, x1_norm_RA, x2_norm_RA, inliers_RA 

def compute_initial_3D_points(imgs, init_pair, K, pixel_threshold, plot=False):
    print('\n\n\n### Computing initial 3D-points ###\n')

    K_inv = LA.inv(K)
    min_its = 40000
    max_its = 40000
    scale_its = 3
    alpha = 0.99
    marg = 0.75

    x1_init, x2_init, des1_init, des2_init = cv.compute_sift_points(imgs[init_pair[0]], imgs[init_pair[1]], marg, flann=True, verbose=True)
    x1_init_norm = cv.dehomogenize(K_inv @ x1_init)
    x2_init_norm = cv.dehomogenize(K_inv @ x2_init)

    E, inliers = cv.estimate_E_robust(K, x1_init_norm, x2_init_norm, min_its, max_its, scale_its, alpha, pixel_threshold, essential_matrix=True, homography=True, verbose=True)

    x1_init_norm_inliers = x1_init_norm[:,inliers]
    x2_init_norm_inliers = x2_init_norm[:,inliers]
    des1_init_inliers = des1_init[inliers]
    des2_init_inliers = des2_init[inliers]

    P1 = cv.get_canonical_camera()
    P2_arr = cv.extract_P_from_E(E)
    X_arr = cv.compute_triangulated_X_from_extracted_P2_solutions(P1, P2_arr, x1_init_norm_inliers, x2_init_norm_inliers)
    P2, X_init_inliers = cv.extract_valid_camera_and_points(P1, P2_arr, X_arr, verbose=True)

    percentile = 90
    feasible_pts = cv.compute_feasible_points(P1, P2, X_init_inliers, percentile)

    x1_init_norm_feasible_inliers = x1_init_norm_inliers[:,feasible_pts]
    x2_init_norm_feasible_inliers = x2_init_norm_inliers[:,feasible_pts]
    des1_init_feasible_inliers = des1_init_inliers[feasible_pts]
    des2_init_feasible_inliers = des2_init_inliers[feasible_pts]
    X_init_feasible_inliers = X_init_inliers[:,feasible_pts]
    X_init_idx = np.ones(X_init_feasible_inliers.shape[1], dtype=bool)

    if plot:
        plot_3D_points(X_init_feasible_inliers)
    
    return x1_init_norm_feasible_inliers, x2_init_norm_feasible_inliers, des1_init_feasible_inliers, des2_init_feasible_inliers, X_init_feasible_inliers, X_init_idx

def compute_translation_registration(K, imgs, init_pair, pixel_threshold, abs_rots, x1_init_norm_feasible_inliers, x2_init_norm_feasible_inliers, des1_init_feasible_inliers, X_init_feasible_inliers, X_init_idx):
    print('\n\n\n### Computing translation registration ###\n')

    K_inv = LA.inv(K)
    marg = 0.75
    min_its = 15000
    max_its = 20000
    scale_its = 1
    alpha = 0.99

    trans = []
    x_norm_TR = []
    X_idx_TR = []
    inliers_TR = []

    n_imgs = imgs.shape[0]
    valid_cameras = np.ones(n_imgs, dtype=bool)

    for i in range(n_imgs):
        print('\nImage:', i+1, '/', n_imgs)

        if (i != init_pair[1]) and (i != init_pair[0]):
            img2 = imgs[i]            
            _, x2, X_idx = cv.compute_sift_points_TR(x1_init_norm_feasible_inliers, des1_init_feasible_inliers, img2, marg, flann=True, verbose=True)
            x_norm = cv.dehomogenize(K_inv @ x2)
        elif i == init_pair[0]:
            x_norm = x1_init_norm_feasible_inliers
            X_idx = X_init_idx
        elif i == init_pair[1]:
            x_norm = x2_init_norm_feasible_inliers
            X_idx = X_init_idx

        X = X_init_feasible_inliers[:,X_idx]        
        R = abs_rots[i]

        T, inliers = cv.estimate_T_robust(K, R, X[:-1], x_norm, min_its, max_its, scale_its, alpha, pixel_threshold, verbose=True)
        
        if np.isnan(T[0]):
            valid_cameras[i] = False

        x_norm_TR.append(x_norm)
        X_idx_TR.append(X_idx)
        trans.append(T)
        inliers_TR.append(inliers)
        
    trans = np.array(trans)

    return trans, valid_cameras, x_norm_TR, X_idx_TR, inliers_TR

def refine_rotations_and_translations(trans, abs_rots, X_init_feasible_inliers, valid_cameras, X_idx_TR, x_norm_TR, inliers_TR):
    print('\n\n\n### Refining translations and rotations ###\n')

    def fun(params, n_valid_cams, xs_norm, X_init, X_idx_TR, inliers_TR, valid_cameras):

        trans = params[:n_valid_cams * 3].reshape((n_valid_cams, 3))
        q_arr = params[n_valid_cams * 3:].reshape((n_valid_cams, 4))
        rots = []

        for i in range(n_valid_cams):
            R = Rotation.from_quat(q_arr[i] / LA.norm(q_arr[i])).as_matrix()
            U, _, VT = LA.svd(R, full_matrices=False)
            R = U @ VT
            rots.append(R)

        xs_proj = []
        t = 0
        for i in range(len(valid_cameras)):

            if valid_cameras[i]:

                X_idx = X_idx_TR[i]
                inliers_T = inliers_TR[i]

                R = rots[t] 
                X = X_init[:,X_idx][:,inliers_T]
                T = trans[t]
                t += 1

                x_proj = cv.dehomogenize(R @ X + T[:,None])
                xs_proj.append(x_proj)
        xs_proj = np.concatenate(xs_proj, 1)

        return (xs_proj - xs_norm).ravel()


    xs_norm = []
    n_trans = trans.shape[0]

    for i in range(n_trans):
        if valid_cameras[i]:
            x_norm = x_norm_TR[i]
            inliers_T = inliers_TR[i]
            xs_norm.append(x_norm[:,inliers_T])
    xs_norm = np.concatenate(xs_norm, 1)

    n_valid_cams = np.sum(valid_cameras)

    q_arr = []
    for i in range(n_trans): # n_imgs
        if valid_cameras[i]:
            R = abs_rots[i]
            q = Rotation.from_matrix(R).as_quat()
            q_arr.append(q)
    q_arr = np.concatenate(q_arr, 0)

    x0 = np.concatenate((trans[valid_cameras].ravel(), q_arr), 0)
    res = optimize.least_squares(fun, x0, method='lm', args=(n_valid_cams, xs_norm, X_init_feasible_inliers[:-1], X_idx_TR, inliers_TR, valid_cameras))

    trans_opt_valid = res.x[:n_valid_cams * 3].reshape((n_valid_cams, 3))
    q_opt_valid = res.x[n_valid_cams * 3:].reshape((n_valid_cams, 4))

    abs_rots_opt_valid = []
    for i in range(n_valid_cams):
        R = Rotation.from_quat(q_opt_valid[i] / LA.norm(q_opt_valid[i])).as_matrix()
        U, _, VT = LA.svd(R, full_matrices=False)
        R = U @ VT
        abs_rots_opt_valid.append(R)
    abs_rots_opt_valid = np.array(abs_rots_opt_valid)

    trans_opt = []
    abs_rots_opt = []
    t = 0

    for i in range(n_trans): # n_imgs
        if valid_cameras[i]:
            abs_rots_opt.append(abs_rots_opt_valid[t])
            trans_opt.append(trans_opt_valid[t])
            t += 1
        else:
            abs_rots_opt.append(abs_rots[i])
            trans_opt.append(trans[i])
    
    return abs_rots_opt, trans_opt

def create_cameras(abs_rots, trans):
    cameras = []

    for i in range(len(trans)):
        R = abs_rots[i]
        T = trans[i]
        P = np.column_stack((R, T))
        cameras.append(P)
    cameras = np.array(cameras)
    
    return cameras

def triangulate_final_3D_reconstruction(imgs, K, pixel_threshold, cameras, valid_cameras, inliers_RA, x1_norm_RA, x2_norm_RA, title):
    print('\n\n\n### Triangulating final 3D-reconstruction ###\n')

    K_inv = LA.inv(K)
    marg = 0.75
    alpha = 0.99

    X_final = []
    valid_idx = []
    n_valid_cameras = np.sum(valid_cameras)

    for i in range(valid_cameras.shape[0]):
        if valid_cameras[i]:
            valid_idx.append(i)
    valid_idx = np.array(valid_idx)
    print('Valid camera indices:', valid_idx)

    for idx in range(n_valid_cameras-1):
        print('Camera pair:', idx+1, '/', n_valid_cameras-1)

        i = valid_idx[idx]
        ij = valid_idx[idx+1]
        
        P1 = cameras[i]
        P2 = cameras[ij]

        if i+1 < ij:
            img1 = imgs[i]
            img2 = imgs[ij]
            x1, x2, _, _ = cv.compute_sift_points(img1, img2, marg, flann=True, verbose=True)
            x1_norm = cv.dehomogenize(K_inv @ x1)
            x2_norm = cv.dehomogenize(K_inv @ x2)

            min_its = 0
            max_its = 10000
            scale_its = 1
            E, inliers = cv.estimate_E_robust(K, x1_norm, x2_norm, min_its, max_its, scale_its, alpha, pixel_threshold, essential_matrix=True, homography=True, verbose=True)
        else:
            inliers = inliers_RA[i]
            x1_norm = x1_norm_RA[i]
            x2_norm = x2_norm_RA[i]
        
        x1_norm_inliers = x1_norm[:,inliers]
        x2_norm_inliers = x2_norm[:,inliers]

        percentile = 90
        X_inliers = cv.triangulate_3D_point_DLT(P1, P2, x1_norm_inliers, x2_norm_inliers, verbose=False)
        feasible_pts = cv.compute_feasible_points(P1, P2, X_inliers, percentile, ransac=False)
        X_final.append(X_inliers[:,feasible_pts])

    C_arr, axis_arr = cv.compute_camera_center_and_normalized_principal_axis(cameras[valid_idx], multi=True)
    plot_cameras_and_3D_points(X_final, C_arr, axis_arr, s=0.5, title=title, valid_idx=valid_idx, multi=True)