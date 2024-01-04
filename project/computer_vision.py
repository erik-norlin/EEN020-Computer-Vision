import numpy as np
from numpy import linalg as LA
from scipy.io import loadmat
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
from matplotlib.pyplot import cm
import cv2
from tqdm import trange


def load_image(path, multi=False):
    if multi:
        img = []    
        for im in path:
            img.append(cv2.imread(im)) 
        img = np.array(img)
    else:
        img = cv2.imread(path)  
    return img 

def convert_mat_to_np(path, string):    
    mat_data = loadmat(path)
    array = np.array(mat_data[string])
    return array

def dehomogenize(x):
    x_deh = x/x[-1]
    return x_deh

def homogenize(x, multi=False):
    if multi:
        ones = np.ones(np.size(x,1))
        x_h = np.vstack([x, ones])
    else:
        x_h = np.append(x, 1)
    return x_h

def normalize_vector(v):
    norm_v = v/(v @ v)**0.5
    return norm_v

def compute_camera_center(P):
    M = P[:,:3]
    P4 = P[:,-1]
    C = -1*(LA.inv(M) @ P4)
    return C

def add_noise(pts, std):
    pts[:-1,:] += np.random.normal(0, std, pts[:-1,:].shape)
    return pts

def compute_normalized_principal_axis(P):
    M = P[:,:3]
    m3 = P[-1,:3]
    v = LA.det(M) * m3
    return normalize_vector(v)

def compute_camera_center_and_normalized_principal_axis(P, multi=False):

    if multi:
        C_arr = np.array([homogenize(compute_camera_center(P[i])) for i in range(np.size(P,0))])
        axis_arr = np.array([homogenize(compute_normalized_principal_axis(P[i])) for i in range(np.size(P,0))])
    else:
        C_arr = np.array([homogenize(compute_camera_center(P))])
        axis_arr = np.array([homogenize(compute_normalized_principal_axis(P))])

    if np.size(C_arr, 0) != 4: # (n,4) => (4,n)
        C_arr = C_arr.T
        axis_arr = axis_arr.T
    
    return C_arr, axis_arr

def compute_homography(P, pi, K):
    R = P[:,:3]
    t = P[:,-1]
    H = R - np.outer(t, np.transpose(pi[:-1]))
    H_tot = K @ H @ LA.inv(K)
    return H, H_tot

def transform_image(img, H):
    width = np.size(img, 1)
    height = np.size(img, 0)
    transf_img = cv2.warpPerspective(img, H, (width, height))
    return transf_img

def uncalibrate(p_cal, K):
    p_uncal = LA.inv(K) @ p_cal
    return p_uncal

def calibrate(p_uncal, K):
    p_cal = K @ p_uncal
    return p_cal

def compute_3D_point(pi, x2D):
    pi = dehomogenize(pi)
    s = -np.transpose(pi[:-1]) @ x2D
    x3D = x2D / s
    return x3D

def transform_and_dehomogenize(P, X):
    x = dehomogenize(transform(P, X))
    return x

def transform(P, x):
    y = P @ x
    return y

def rotate_and_translate_P2_points(x, theta, x_trans, y_trans):

    cos = np.cos(theta)
    sin= np.sin(theta)
    R = np.array([[cos, -sin, x_trans],[sin, cos, y_trans],[0, 0, 1]])

    x_rot = R @ x
    return x_rot

def find_mean_and_std(x, axis):
    mean = np.sum(x[axis,:]) / np.size(x,1)
    std =  (np.sum((x[axis,:] - mean)**2) / np.size(x,1))**0.5
    return mean, std

def normalize_x_and_y(img_pts, verbose=False):

    x_mean, x_std = find_mean_and_std(img_pts, axis=0)
    y_mean, y_std = find_mean_and_std(img_pts, axis=1)

    N = np.array([[1/x_std, 0, -x_mean/x_std],
                    [0, 1/y_std, -y_mean/y_std],
                    [0, 0, 1]])
    
    img_pts_norm = N @ img_pts

    if verbose:
        print('\nx_mean:', x_mean, 'x_std:', x_std, 'y_mean:', y_mean, 'y_std:', y_std)
        print('x_mean:', np.mean(img_pts_norm[0,:]), 'x_std:', np.std(img_pts_norm[0,:]), 'y_mean:', np.mean(img_pts_norm[1,:]), 'y_std:', np.std(img_pts_norm[1,:]))

    return img_pts_norm, N

def check_mean_and_std(x):
    print('\nx_mean:', np.mean(x[0,:]), '\nx_std:', np.std(x[0,:]), '\ny_mean:', np.mean(x[1,:]), '\ny_std:', np.std(x[1,:]))


def unitize(a, b):
    denom = 1.0 / np.sqrt(a**2 + b**2)
    ra = a * denom
    rb = b * denom
    return ra, rb

def homography_to_RT(H, x1, x2):
    # Check the right sign for H
    if LA.det(H) < 0:
        H *= -1 
        
    N = x1.shape[1]
    if x1.shape[0] != 3:
        x1 = np.vstack([x1, np.ones((1, N))])
    if x2.shape[0] != 3:
        x2 = np.vstack([x2, np.ones((1, N))])

    positives = np.sum(np.sum(x2 * (H @ x1), axis=0) > 0)
    if positives < (N / 2):
        H *= -1

    U, S, VT = np.linalg.svd(H, full_matrices=False)
    V = VT.T
    s1 = S[0] / S[1]
    s3 = S[2] / S[1]
    zeta = s1 - s3
    a1 = np.sqrt(1 - s3**2)
    b1 = np.sqrt(s1**2 - 1)
    a, b = unitize(a1, b1)
    c, d = unitize(1+s1*s3, a1*b1)
    e, f = unitize(-b/s1, -a/s3)
    v1, v3 = V[:, 0], V[:, 2]
    n1 = b * v1 - a * v3
    n2 = b * v1 + a * v3
    R1 = U @ np.array([[c, 0, d], [0, 1, 0], [-d, 0, c]]) @ VT
    R2 = U @ np.array([[c, 0, -d], [0, 1, 0], [d, 0, c]]) @ VT
    t1 = e * v1 + f * v3
    t2 = e * v1 - f * v3
    if n1[2] < 0:
        t1 = -t1
        n1 = -n1
    if n2[2] < 0:
        t2 = -t2
        n2 = -n2

    # Move from Triggs' convention H = R*(I - t*n') to H&Z notation H = R - t*n'
    t1 = R1 @ t1
    t2 = R2 @ t2

    # Verify that we obtain the initial homography back
    # H /= S[1]
    # print(np.linalg.norm(R1 - zeta * np.outer(t1, n1) - H), np.linalg.norm(R2 - zeta * np.outer(t2, n2) - H))

    return R1, t1, R2, t2

    # Example usage:
    # H is the homography matrix, x1 and x2 are the corresponding 2D points
    # R1, t1, R2, t2 = homography_to_RT(H, x1, x2)

def estimate_camera_DLT(Xmodel, img_pts, verbose=False):

    n = np.size(img_pts,1)
    M = []

    for i in range(n):

        X = Xmodel[0,i]
        Y = Xmodel[1,i]
        Z = Xmodel[2,i]

        x = img_pts[0,i]
        y = img_pts[1,i]

        m = np.array([[X, Y, Z, 1, 0, 0, 0, 0, -x*X, -x*Y, -x*Z, -x],
                      [0, 0, 0, 0, X, Y, Z, 1, -y*X, -y*Y, -y*Z, -y]])

        M.append(m)

    M = np.concatenate(M, 0)
    U, S, VT = LA.svd(M, full_matrices=False)
    P = np.stack([VT[-1, i:i+4] for i in range(0, 12, 4)], 0)

    if verbose:
        M_approx = U @ np.diag(S) @ VT
        v = VT[-1,:] # last row of VT because optimal v should be last column of V
        Mv = M @ v
        print('\n||Mv||:', (Mv @ Mv)**0.5)
        print('||v||^2:', v @ v)
        print('max{||M - M_approx||}:', np.max(np.abs(M - M_approx)))
        print('S:', S)

    return P

def estimate_T_DLT_1(img_pts, verbose=False):

    n = img_pts.shape[1]
    M = []

    for i in range(n):

        x = img_pts[0,i]
        y = img_pts[1,i]

        m = np.array([[1, 0, -x],
                      [0, 1, -y]])
        
        M.append(m)

    M = np.concatenate(M, 0)
    U, S, VT = LA.svd(M, full_matrices=False)
    T = VT[-1,:]

    if verbose:
        M_approx = U @ np.diag(S) @ VT
        v = VT[-1,:]
        Mv = M @ v
        print('\n||Mv||:', (Mv @ Mv)**0.5)
        print('||v||^2:', v @ v)
        print('max{||M - M_approx||}:', np.max(np.abs(M - M_approx)))
        print('S:', S)

    return T

def estimate_T_DLT_2(R, img_pts, verbose=False):

    n = img_pts.shape[1]
    M = []

    for i in range(n):

        xx = create_skew_symmetric_matrix(img_pts[:,i])
        m = np.column_stack((xx, xx @ R))
        M.append(m)

    M = np.concatenate(M, 0)
    U, S, VT = LA.svd(M, full_matrices=False)
    T = VT[-1,:3]

    if verbose:
        M_approx = U @ np.diag(S) @ VT
        v = VT[-1,:]
        Mv = M @ v
        print('\n||Mv||:', (Mv @ Mv)**0.5)
        print('||v||^2:', v @ v)
        print('max{||M - M_approx||}:', np.max(np.abs(M - M_approx)))
        print('S:', S)

    return T

def estimate_H_DLT(img1_pts, img2_pts, verbose=False):

    n = np.size(img1_pts,1)
    M = []

    for i in range(n):

        x = img1_pts[0,i]
        y = img1_pts[1,i]

        u = img2_pts[0,i]
        v = img2_pts[1,i]

        m = np.array([[x, y, 1, 0, 0, 0, -u*x, -u*y, -u],
                      [0, 0, 0, x, y, 1, -v*x, -v*y, -v]])

        M.append(m)

    M = np.concatenate(M, 0)
    U, S, VT = LA.svd(M, full_matrices=False)
    H = np.stack([VT[-1, i:i+3] for i in range(0, 9, 3)], 0)

    if verbose:
        M_approx = U @ np.diag(S) @ VT
        v = VT[-1,:] # last row of VT because optimal v should be last column of V
        Mv = M @ v
        print('\n||Mv||:', (Mv @ Mv)**0.5)
        print('||v||^2:', v @ v)
        print('max{||M - M_approx||}:', np.max(np.abs(M - M_approx)))
        print('S:', S)

    return H

def triangulate_3D_point_DLT(P1, P2, img1_pts, img2_pts, verbose=False):
    
    n = np.size(img1_pts,1)
    X = []

    for i in range(n):

        x1 = img1_pts[0,i]
        y1 = img1_pts[1,i]
        
        x2 = img2_pts[0,i]
        y2 = img2_pts[1,i]

        M = np.array([[P1[0,0]-x1*P1[2,0], P1[0,1]-x1*P1[2,1], P1[0,2]-x1*P1[2,2], P1[0,3]-x1*P1[2,3]],
                      [P1[1,0]-y1*P1[2,0], P1[1,1]-y1*P1[2,1], P1[1,2]-y1*P1[2,2], P1[1,3]-y1*P1[2,3]],
                      [P2[0,0]-x2*P2[2,0], P2[0,1]-x2*P2[2,1], P2[0,2]-x2*P2[2,2], P2[0,3]-x2*P2[2,3]],
                      [P2[1,0]-y2*P2[2,0], P2[1,1]-y2*P2[2,1], P2[1,2]-y2*P2[2,2], P2[1,3]-y2*P2[2,3]]])

        U, S, VT = LA.svd(M, full_matrices=False)
        X.append(VT[-1,:])
        
        if verbose:
            M_approx = U @ np.diag(S) @ VT
            v = VT[-1,:]
            Mv = M @ v
            print('\n||Mv||:', (Mv @ Mv)**0.5)
            print('||v||^2:', v @ v)
            print('max{||M - M_approx||}:', np.max(np.abs(M - M_approx)))
            print('S:', S)

    X = dehomogenize(np.stack(X,1))
    return X # in P3

def compute_triangulated_X_from_extracted_P2_solutions(P1, P2_arr, x1_norm, x2_norm, verbose=False):
    X_arr = np.array([triangulate_3D_point_DLT(P1, P2, x1_norm, x2_norm, verbose=verbose) for P2 in P2_arr])
    return X_arr

def enforce_fundamental(F):
    U, S, VT = LA.svd(F, full_matrices=False)
    # if LA.det(U @ VT) < 0:
    #     VT = -VT
    S[-1] = 0
    F = U @ np.diag(S) @ VT
    return F  

def estimate_F_DLT(img1_pts, img2_pts, enforce=False, verbose=False): # Computes F such that x2.T @ F @ x1 = 0

    n = np.size(img1_pts,1)
    M = []

    for i in range(n):

        # x1 = img_pts_1[0,i]
        # y1 = img_pts_1[1,i]
        # z1 = img_pts_1[2,i]
        
        # x2 = img_pts_2[0,i]
        # y2 = img_pts_2[1,i]
        # z2 = img_pts_2[2,i]

        # m = np.array([[x2*x1, x2*y1, x2*z1, y2*x1, y2*y1, y2*z1, z2*x1, z2*y1, z2*z1]])

        x = img1_pts[:,i]
        y = img2_pts[:,i]
        m = np.outer(y, x).flatten()
        M.append([m])


    M = np.concatenate(M, 0)
    U, S, VT = LA.svd(M, full_matrices=False)
    F = VT[-1,:].reshape(3,3)
    if enforce:
        F = enforce_fundamental(F)

    if verbose:
        for i in range(np.size(img1_pts,1)):
            epi_const = img2_pts[:,i].T @ F @ img1_pts[:,i]
            print('x2^T @ F @ x1:', epi_const)
        
        print('\nDet(F):', LA.det(F))
        M_approx = U @ np.diag(S) @ VT
        v = VT[-1,:]
        Mv = M @ v
        print('||Mv||:', (Mv @ Mv)**0.5)
        print('||v||^2:', v @ v)
        print('max{||M - M_approx||}:', np.max(np.abs(M - M_approx)))
        print('S:', S)

    F = F/F[-1,-1]
    return F

def enforce_essential(E):
    U, _, VT = LA.svd(E, full_matrices=False)
    if LA.det(U @ VT) < 0:
        VT = -VT
    E =  U @ np.diag([1,1,0]) @ VT
    return E

def estimate_E_DLT(img1_pts_norm, img2_pts_norm, enforce=False, verbose=False):

    n = np.size(img1_pts_norm,1)
    M = []

    for i in range(n):

        x = img1_pts_norm[:,i]
        y = img2_pts_norm[:,i]
        m = np.outer(y, x).flatten()
        M.append([m])

    M = np.concatenate(M, 0)
    U, S, VT = LA.svd(M, full_matrices=False)
    E = VT[-1,:].reshape(3,3)
    if enforce:
        E = enforce_essential(E)

    if verbose:
        for i in range(np.size(img1_pts_norm,1)):
            epi_const = img2_pts_norm[:,i].T @ E @ img1_pts_norm[:,i]
            print('x2^T @ E @ x1:', epi_const)
        
        print('\nDet(E):', LA.det(E))
        M_approx = U @ np.diag(S) @ VT
        v = VT[-1,:]
        Mv = M @ v
        print('||Mv||:', (Mv @ Mv)**0.5)
        print('||v||^2:', v @ v)
        print('max{||M - M_approx||}:', np.max(np.abs(M - M_approx)))
        print('S:', S)

    E = E/E[-1,-1]
    return E

def estimate_E_robust_simple(K, x1_norm, x2_norm, n_its, n_samples, err_threshold_px):
    
    err_threshold = err_threshold_px / K[0,0]
    best_inliers = None
    best_E = None
    max_inliers = 0

    for t in trange(n_its):

        rand_mask = np.random.choice(np.size(x1_norm,1), n_samples, replace=False)
        E = estimate_E_DLT(x1_norm[:,rand_mask], x2_norm[:,rand_mask], enforce=True, verbose=False)

        distance1_arr, distance2_arr = compute_epipolar_errors(E, x1_norm, x2_norm)
        inliers = ((distance1_arr**2 + distance2_arr**2) / 2) < err_threshold**2
        n_inliers = np.sum(inliers)

        if n_inliers > max_inliers:
            best_inliers = np.copy(inliers)
            best_E = np.copy(E)
            max_inliers = n_inliers
            print(np.sum(inliers), end='\r')
        
    return best_E, best_inliers

def compute_E_validity(E):
    rank = LA.matrix_rank(E)
    valid = True if rank == 2 else False
    return valid

def compute_E_inliers(E, x1_norm, x2_norm, err_threshold):
    
    distance1_arr, distance2_arr = compute_epipolar_errors(E, x1_norm, x2_norm)
    inliers = ((distance1_arr**2 + distance2_arr**2) / 2) < err_threshold**2
    n_inliers = np.sum(inliers)
    epsilon_E = n_inliers / x1_norm.shape[1]

    return epsilon_E, inliers

def verbose_E_robust(t, T_E, T_H, epsilon_E, epsilon_H, inliers, method):
    print('Iteration:', t, 'T_E:', T_E, 'T_H:', T_H, 'epsilon_E:', np.round(epsilon_E, 2), 'epsilon_H:', np.round(epsilon_H, 2), 'No. inliers:', np.sum(inliers), 'From:', method)

def estimate_E_robust(K, x1_norm, x2_norm, min_its, max_its, scale_its, alpha, err_threshold_px, essential_matrix=True, homography=True, verbose=False):
    
    err_threshold = err_threshold_px / K[0,0]
    best_E = None
    best_inliers = None
    n_points = x1_norm.shape[1]
    n_E_samples = 8
    n_H_samples = 4
    best_epsilon_E = 0
    best_epsilon_H = 0
    T_E = max_its
    T_H = max_its

    t = 0
    while t < T_E and t < T_H:
        t += 1

        if essential_matrix:
            rand_mask = np.random.choice(n_points, n_E_samples, replace=False)
            E = estimate_E_DLT(x1_norm[:,rand_mask], x2_norm[:,rand_mask], enforce=True, verbose=False)
            E_valid = compute_E_validity(E)

            if E_valid:
                epsilon_E, inliers = compute_E_inliers(E, x1_norm, x2_norm, err_threshold)
                    
                if epsilon_E > best_epsilon_E:
                    best_E = np.copy(E)
                    best_inliers = np.copy(inliers)
                    best_epsilon_E = epsilon_E
                    T_E = compute_ransac_iterations(alpha, best_epsilon_E, n_E_samples, min_its, max_its, scale_its)

                    if verbose:
                        verbose_E_robust(t, T_E, T_H, best_epsilon_E, best_epsilon_H, best_inliers, method='E 8-point alg.')
        
        if homography:
            rand_mask = np.random.choice(n_points, n_H_samples, replace=False)
            H = estimate_H_DLT(x1_norm[:,rand_mask], x2_norm[:,rand_mask], verbose=False)
            x2_norm_proj = dehomogenize(H @ x1_norm)
            distance_arr = compute_point_point_distance(x2_norm_proj, x2_norm)
            inliers = distance_arr < (3*err_threshold)
            n_inliers = np.sum(inliers)
            epsilon_H = n_inliers / n_points

            if epsilon_H > best_epsilon_H:
                
                # num, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, np.eye(3))
                R1, T1, R2, T2 = homography_to_RT(H, x1_norm, x2_norm)
                E1 = compute_E_from_R_and_T(R1, T1)
                E2 = compute_E_from_R_and_T(R2, T2)

                E1_valid = compute_E_validity(E1)
                E2_valid = compute_E_validity(E2)

                if E1_valid:
                    epsilon_E, inliers = compute_E_inliers(E1, x1_norm, x2_norm, err_threshold)
                        
                    if epsilon_E > best_epsilon_E:
                        best_E = np.copy(E1)
                        best_inliers = np.copy(inliers)
                        best_epsilon_E = epsilon_E
                        best_epsilon_H = epsilon_H
                        T_E = compute_ransac_iterations(alpha, best_epsilon_E, n_E_samples, min_its, max_its, scale_its)
                        T_H = compute_ransac_iterations(alpha, best_epsilon_H, n_H_samples, min_its, max_its, scale_its)

                        if verbose:
                            verbose_E_robust(t, T_E, T_H, best_epsilon_E, best_epsilon_H, best_inliers, method='H 4-point alg.')

                if E2_valid:
                    epsilon_E, inliers = compute_E_inliers(E2, x1_norm, x2_norm, err_threshold)
                        
                    if epsilon_E > best_epsilon_E:
                        best_E = np.copy(E2)
                        best_inliers = np.copy(inliers)
                        best_epsilon_E = epsilon_E
                        best_epsilon_H = epsilon_H
                        T_E = compute_ransac_iterations(alpha, best_epsilon_E, n_E_samples, min_its, max_its, scale_its)
                        T_H = compute_ransac_iterations(alpha, best_epsilon_H, n_H_samples, min_its, max_its, scale_its)
                        
                        if verbose:
                            verbose_E_robust(t, T_E, T_H, best_epsilon_E, best_epsilon_H, best_inliers, method='H 4-point alg.')
        
    print('Bailout at iteration:', t)
    return best_E, best_inliers

def estimate_T_robust(K, R, X, x_norm, min_its, max_its, scale_its, alpha, err_threshold_px, DLT1=False, verbose=False):
    
    err_threshold = err_threshold_px / K[0,0]
    best_T = np.full(3, np.nan)
    best_inliers = np.zeros(x_norm.shape[1], dtype=bool)
    best_epsilon = 0
    n_points = x_norm.shape[1]
    n_samples = 2
    ransac_its = max_its

    t = 0
    while t < ransac_its:
        t += 1

        rand_mask = np.random.choice(n_points, n_samples, replace=False)
        if DLT1:
            T = estimate_T_DLT_1(x_norm[:,rand_mask], verbose=False)
        else:
            T = estimate_T_DLT_2(R, x_norm[:,rand_mask], verbose=False)

        x_norm_proj = dehomogenize(R @ X + T[:,None])
        distance_arr = compute_point_point_distance(x_norm_proj, x_norm)
        inliers = distance_arr < err_threshold
        n_inliers = np.sum(inliers)
        epsilon = n_inliers / n_points

        if epsilon > best_epsilon:
            best_T = np.copy(T)
            best_inliers = np.copy(inliers)
            best_epsilon = epsilon
            ransac_its = compute_ransac_iterations(alpha, best_epsilon, n_samples, min_its, max_its, scale_its)
            if verbose:
                print('Iteration:', t, 'T:', ransac_its, 'epsilon:', np.round(best_epsilon, 2), 'No. inliers:', np.sum(inliers))
    
    print('Bailout at iteration:', t)
    return best_T, best_inliers

def compute_ransac_iterations(alpha, epsilon, s, min_its, max_its, scale):
    T = scale * np.ceil(np.log(1-alpha) / np.log(1-epsilon**s))
    if np.isinf(T) or T > max_its:
        T = max_its
    elif T < min_its:
        T = min_its
    return T

def normalize_camera(P):
    if not np.isclose(P[-1,-1], 0):
        P= P/P[-1,-1]
        if P[2,2] < 0:
            P = -P
    return P

def convert_E_to_F(E, K1, K2):
    F = LA.inv(K2).T @ E @ LA.inv(K1)
    F = F / F[-1,-1]
    return F

def compute_and_plot_lines(l, img, ax):

    col = cm.rainbow(np.linspace(0, 1, np.size(l,1)))

    for i in range(np.size(l,1)):

        a = l[0,i]
        b = l[1,i]
        c = l[2,i]

        x = np.linspace(0, np.size(img,1), 2)
        y = (-a*x - c) / b # ax + by + c = 0 ==> y = (-ax - c) / b

        ax.plot(x, y, '-', lw=3, color=col[i], alpha=0.7)

def compute_RMS_error(distance1_arr, distance2_arr):
    e_rms = np.sqrt(np.sum(distance1_arr**2 + distance2_arr**2) / (2*np.size(D1, 0)))
    return e_rms

def compute_point_point_distance(x_proj, x_img):
    distance_arr = LA.norm(x_proj - x_img, axis=0)
    return distance_arr

def compute_point_line_distance_2D(l, p, multi=False):

    if multi:
        a = l[0,:]
        b = l[1,:]
        c = l[2,:]

        x = p[0,:]
        y = p[1,:]
    else:
        a = l[0]
        b = l[1]
        c = l[2]

        x = p[0]
        y = p[1]
    
    distance_arr = np.abs(a*x + b*y + c) / (a**2 + b**2)**0.5

    # numerator = np.abs(np.einsum("ij, ij->j", p, l))
    # denominator = LA.norm(l[:-1], axis=0)
    # D = numerator / denominator

    return distance_arr

def compute_epipolar_lines(F, x1, x2):
    l2 = F @ x1
    l1 = F.T @ x2

    # l2 = np.einsum("ij,jk->ik", F, x1)
    # l1 = np.einsum("ji,jk->ik", F, x2)
    return l1, l2

def compute_epipolar_errors(F, x1, x2):
    l1, l2 = compute_epipolar_lines(F, x1, x2)
    distance1_arr = compute_point_line_distance_2D(l1, x1, multi=True)
    distance2_arr = compute_point_line_distance_2D(l2, x2, multi=True)
    return distance1_arr, distance2_arr

def RQ_decompose(a):

    m, n = a.shape
    e = np.eye(m)
    p = e[:, ::-1]
    q0, r0 = LA.qr(p @ a[:, :m].T @ p)

    r = p @ r0.T @ p
    q = p @ q0.T @ p

    fix = np.diag(np.sign(np.diag(r)))
    r = r @ fix
    q = fix @ q

    if n > m:
        q = np.concatenate((q, LA.inv(r) @ a[:, m:n]), axis=1)

    return r, q

def compute_feasible_points(P1, P2, X, percentile, ransac=True):
    
    if ransac:
        x1 = P1 @ X
        x2 = P2 @ X
        x1_filter = x1[-1,:] > 0
        x2_filter = x2[-1,:] > 0

    X_bar = np.mean(X, axis=1)
    X_norm = LA.norm(X - X_bar[:,None], axis=0)
    norm_percentile = np.percentile(X_norm, percentile)
    outlier_filter = X_norm < norm_percentile

    if ransac:
        feasible_pts = x1_filter * x2_filter * outlier_filter
    else:
        feasible_pts = outlier_filter
    return feasible_pts

def compute_absolute_rotations(rel_rots, origin_idx, verbose=False):
    
    abs_rots = [rel_rots[0]]
    for i in range(len(rel_rots)-1):

        Ri = abs_rots[i]
        R2 = rel_rots[i+1]
        if LA.det(R2) < 0:
            print('WARNING: det(R{}) < 0, not a rotation!'.format(i), LA.det(R2))
            R2 = -R2
        U, _, VT = LA.svd(R2, full_matrices=False)
        R2 = U @ VT
        Rj = R2 @ Ri
        abs_rots.append(Rj)

    R0 = abs_rots[origin_idx]    
    for i in range(len(abs_rots)):

        Ri = abs_rots[i]
        Ri = LA.inv(R0) @ Ri
        abs_rots[i] = Ri

        if verbose:
            print('det(R{}):'.format(i), LA.det(Ri))
        
    return np.array(abs_rots)

def compute_rots_trans_RA(rel_cameras):

    P1 = rel_cameras[0]
    rots = [P1[:,:-1]]
    trans = [P1[:,-1]]

    for i in range(rel_cameras.shape[0]-1):
        P2 = rel_cameras[i+1]
        P2 /= P2[-1,2]
        R2 = P2[:,:-1]
        T2 = P2[:,-1]

        Ri = rots[i]
        Ti = trans[i]

        Rj = R2 @ Ri
        Tj = T2 + (Rj @ Ri.T @ Ti)

        rots.append(Rj)
        trans.append(Tj)

    rots = np.array(rots)
    trans = np.array(trans)

    return rots, trans

def compute_sift_points(img1, img2, marg, flann=False, verbose=False):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if flann:
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
    else:
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < marg*n.distance:
            good_matches.append([m])

    x1 = np.stack([kp1[match[0].queryIdx].pt for match in good_matches],1)
    x2 = np.stack([kp2[match[0].trainIdx].pt for match in good_matches],1)
    x1 = homogenize(x1, multi=True)
    x2 = homogenize(x2, multi=True)

    des1 = np.stack([des1[match[0].queryIdx] for match in good_matches],0)
    des2 = np.stack([des2[match[0].trainIdx] for match in good_matches],0)

    if verbose:
        print('Number of matches:', np.size(matches,0))
        print('Number of good matches:', np.size(x1,1))

    return x1, x2, des1, des2

def compute_sift_points_TR(x1, des1, img2, marg, flann=False, verbose=False):
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # nfeatures=0
    # sigma=1.6
    # contrastThreshold=0.04
    # edgeThreshold=10

    # nfeatures=nfeatures,
    # sigma=2/sigma,  
    # contrastThreshold=2/contrastThreshold,
    # edgeThreshold=2/edgeThreshold,

    sift = cv2.SIFT_create()
    kp2, des2 = sift.detectAndCompute(img2, None)

    if flann:
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
    else:
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < marg*n.distance:
            good_matches.append([m])

    x_idx = np.array([match[0].queryIdx for match in good_matches])
    x1 = np.stack([x1[:,match[0].queryIdx] for match in good_matches],1)
    x2 = np.stack([kp2[match[0].trainIdx].pt for match in good_matches],1)
    x2 = homogenize(x2, multi=True)

    if verbose:
        print('Number of matches:', np.size(matches,0))
        print('Number of good matches:', np.size(x1,1))

    return x1, x2, x_idx

def get_sift_plot_points(img1_pts, img2_pts, img1):
    x = [img1_pts[0,:], np.size(img1,1)+img2_pts[0,:]]
    y = [img1_pts[1,:], img2_pts[1,:]]
    return x, y

def plot_sift_points(x1, x2, img1, img_match, path, save=False):
    x, y = get_sift_plot_points(x1, x2, img1)

    fig = plt.figure(figsize=(18,9))
    ax = plt.axes()

    ax.plot(x, y, 'o-', ms=5, lw=1, color='magenta')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_aspect('equal')
    ax.imshow(cv2.cvtColor(img_match, cv2.COLOR_BGR2RGB))
    fig.tight_layout()
    if save:
        fig.savefig(path, dpi=300)
    plt.show()

def extract_skew_symmetric_vector(T):
    t = np.array([T[2,1], T[0,2], T[1,0]])
    return t

def create_skew_symmetric_matrix(t):
    T = np.array([[0, -t[2], t[1]],
                  [t[2], 0, -t[0]],
                  [-t[1], t[0], 0]])
    return T

def compute_E_from_R_and_T(R, T):
    E = create_skew_symmetric_matrix(T) @ R
    return E

def extract_valid_camera_and_points(P1, P_arr, X_arr, verbose=False):
        
    x1_arr = np.array([transform(P1, X) for X in X_arr])
    valid_coords_P1 = np.array([np.sum(x[-1] > 0) for x in x1_arr])

    x2_arr = np.array([transform(P_arr[i], X_arr[i]) for i in range(np.size(P_arr, 0))])
    valid_coords_P2 = np.array([np.sum(x[-1] > 0) for x in x2_arr]) 

    valid_coords = valid_coords_P1 + valid_coords_P2
    valid_coords_ind = np.argmax(valid_coords)
    X_valid = X_arr[valid_coords_ind]
    P2_valid = P_arr[valid_coords_ind]
    # P2_valid = P2_valid / P2_valid[-1,-1]
    
    if verbose:
        print('No. valid coords for each camera pair:', valid_coords)
        print('Argmax(P2_arr):', valid_coords_ind)
    
    return P2_valid, X_valid

def get_canonical_camera():
    P = np.concatenate((np.eye(3), np.zeros(3)[:,np.newaxis]), 1)
    return P

def get_camera_rotation(P):
    R = P[:,:-1]
    return R

def get_camera_translation(P):
    T = P[:,-1]
    return T

def extract_P_from_E(E):

    U, _, VT = LA.svd(E, full_matrices=False)

    if LA.det(U @ VT) < 0:
        VT = -VT

    W = np.array([[0,1,0],[-1,0,0],[0,0,1]])
    Z = np.array([[0,-1,0],[1,0,0],[0,0,0]])

    S1 = U @ Z @ U.T
    S2 = U @ Z.T @ U.T

    R1 = U @ W @ VT
    R2 = U @ W.T @ VT

    t1 = extract_skew_symmetric_vector(S1)
    t2 = extract_skew_symmetric_vector(S2)

    P1 = np.concatenate((R1, t1[:, np.newaxis]), 1)
    P2 = np.concatenate((R1, t2[:, np.newaxis]), 1)
    P3 = np.concatenate((R2, t1[:, np.newaxis]), 1)
    P4 = np.concatenate((R2, t2[:, np.newaxis]), 1)

    P_arr = np.array([P1, P2, P3, P4])
    return P_arr

def compute_average_error(x_proj, x_img):
    err = ((x_proj[0,:] - x_img[0,:])**2 + (x_proj[1,:] - x_img[1,:])**2)**0.5
    avg_err = np.sum(err) / np.size(err,0)
    return avg_err

def compute_reprojection_error_wrt_X(P1, P2, Xj, x1j, x2j):

    x1j_proj = transform_and_dehomogenize(P1, Xj)
    x2j_proj = transform_and_dehomogenize(P2, Xj)
    
    r1 = np.array([x1j[0] - x1j_proj[0], x1j[1] - x1j_proj[1]])
    r2 = np.array([x2j[0] - x2j_proj[0], x2j[1] - x2j_proj[1]])
    res = np.concatenate((r1, r2), 0)
    reproj_err = LA.norm(res)**2

    return reproj_err, res

def compute_total_reprojection_error_wrt_X(P1, P2, X_est, x1, x2, verbose=False):

    n_pts = np.size(X_est,1)
    reproj_err_tot = []
    res_tot = []

    for j in range(n_pts):

        Xj = X_est[:,j]
        x1j = x1[:,j]
        x2j = x2[:,j]

        reproj_err, res = compute_reprojection_error_wrt_X(P1, P2, Xj, x1j, x2j)
        reproj_err_tot.append(reproj_err)
        res_tot.append(res)
    
    res_tot = np.concatenate(res_tot, 0)

    if verbose:
        print('\nTotal reprojection error:', round(np.sum(reproj_err_tot), 2))
        print('Median reprojection error:', round(np.median(reproj_err_tot), 2))
        print('Avg. reprojection error:', round(np.mean(reproj_err_tot), 2))

    return reproj_err_tot, res_tot

def compute_jacobian_of_residual_wrt_X(Pi, Xj):
    jac1 = (Pi[-1,:] * (Pi[0,:] @ Xj) / (Pi[-1,:] @ Xj)**2) - (Pi[0,:] / (Pi[-1,:] @ Xj))
    jac2 = (Pi[-1,:] * (Pi[1,:] @ Xj) / (Pi[-1,:] @ Xj)**2) - (Pi[1,:] / (Pi[-1,:] @ Xj))
    jac = np.row_stack((jac1, jac2))
    return jac

def linearize_reprojection_error_wrt_X(P1, P2, Xj, x1j, x2j):

    _, res = compute_reprojection_error_wrt_X(P1, P2, Xj, x1j, x2j)
    jac1 = compute_jacobian_of_residual_wrt_X(P1, Xj)
    jac2 = compute_jacobian_of_residual_wrt_X(P2, Xj)
    jac = np.row_stack((jac1, jac2))
    
    return res, jac

def compute_update(res, jac, mu):
    I = np.eye(jac.shape[1])
    delta = -LA.inv(jac.T @ jac + mu*I) @ jac.T @ res
    return delta

def optimize_X(P1, P2, X, x1, x2, mu_init, n_its, verbose=False):

    n_pts = np.size(X,1)
    ts = []

    for j in trange(n_pts):
        
        Xj = X[:,j]
        x1j = x1[:,j]
        x2j = x2[:,j]

        t = 0
        converged = False
        mu = mu_init
    
        while (t <= n_its) and converged is not True:
        
            t += 1
            res, jac = linearize_reprojection_error_wrt_X(P1, P2, Xj, x1j, x2j)
            delta_Xj = compute_update(res, jac, mu)
            Xj_opt = dehomogenize(Xj + delta_Xj)
            
            reproj_err, _ = compute_reprojection_error_wrt_X(P1, P2, Xj, x1j, x2j)
            reproj_err_opt, _ = compute_reprojection_error_wrt_X(P1, P2, Xj_opt, x1j, x2j)

            if np.isclose(reproj_err_opt, reproj_err):
                converged = True
            elif reproj_err_opt < reproj_err:
                Xj = Xj_opt
                mu /= 10
            else:
                mu *= 10
        
        X[:,j] = Xj
        ts.append(t)

    if verbose:
        print('\nAvg its:', np.mean(ts))
        print('Max its:', np.max(ts))
        print('Min its:', np.min(ts))

    return X

def remove_error_2P_points(x_proj, x_img, err):

    x_keep = (((x_proj[0,:] - x_img[0,:])**2 + (x_proj[1,:] - x_img[1,:])**2)**0.5 < err)
    x_proj_keep = x_proj[:,x_keep]
    x_img_keep = x_img[:,x_keep]
    
    return x_proj_keep, x_img_keep

def plot_cameras_and_axes(ax, C_list, axis_list, s):

    col = cm.rainbow(np.linspace(0, 1, np.size(C_list,1)))

    for i in range(np.size(C_list,1)):

        C = C_list[:,i]
        axis = axis_list[:,i]

        ax.plot(C[0], C[1], C[2], 'o', color=col[i],  label='Camera {}'.format(i+1), alpha=0.7)

        x_axis = C[0] + s*axis[0]
        y_axis = C[1] + s*axis[1]
        z_axis = C[2] + s*axis[2]

        ax.plot([x_axis, C[0]], [y_axis, C[1]], [z_axis, C[2]], '-', color=col[i], lw=3, alpha=0.7)

def plot_cameras_and_3D_points(X, C_arr, axis_arr, s, path):
    
    fig = plt.figure(figsize=(8,6))
    ax = plt.axes(projection='3d')

    ax.plot(X[0], X[1], X[2], '.', ms=0.6, color='magenta', label='3D points')
    plot_cameras_and_axes(ax, C_arr, axis_arr, s)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    ax.axis('equal')

    plt.legend(loc="lower right")
    fig.savefig(path, dpi=300)

    plt.show()

def plot_image_points_projected_points_and_image(x_proj, x_img, img, path):

    fig = plt.figure(figsize=(10,8))

    plt.plot(x_img[0], x_img[1], 'D', color='blue', ms=1.2, label='Image points')
    plt.plot(x_proj[0], x_proj[1], 's', color='red', ms=0.4, label='Projected points')

    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.axis('equal')
    # plt.gca().invert_yaxis()

    plt.legend(loc="lower right")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    fig.tight_layout()
    fig.savefig(path, dpi=300)

    plt.show()