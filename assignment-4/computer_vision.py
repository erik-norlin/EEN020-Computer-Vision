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
from icecream import ic


def load_image(path):
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

def compute_normalized_principal_axis(P):
    M = P[:,:3]
    m3 = P[-1,:3]
    v = LA.det(M) * m3
    return normalize_vector(v)

def compute_camera_center_and_normalized_principal_axis(P):
    C = compute_camera_center(P)
    p_axis = compute_normalized_principal_axis(P) 
    return C, p_axis

def compute_camera_and_normalized_principal_axis(P, multi=False):

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

def normalize_x_and_y(img_pts):

    axis = 0
    x_mean, x_std = find_mean_and_std(img_pts, axis)

    axis = 1
    y_mean, y_std = find_mean_and_std(img_pts, axis)

    print('\nx_mean:', x_mean, '\nx_std:', x_std, '\ny_mean:', y_mean, '\ny_std:', y_std)

    N = np.array([[1/x_std, 0, -x_mean/x_std],
                    [0, 1/y_std, -y_mean/y_std],
                    [0, 0, 1]])
    
    img_pts_norm = N @ img_pts
    
    print('\nx_mean:', np.mean(img_pts_norm[0,:]), '\nx_std:', np.std(img_pts_norm[0,:]), '\ny_mean:', np.mean(img_pts_norm[1,:]), '\ny_std:', np.std(img_pts_norm[1,:]))

    return img_pts_norm, N

def check_mean_and_std(x):
    print('\nx_mean:', np.mean(x[0,:]), '\nx_std:', np.std(x[0,:]), '\ny_mean:', np.mean(x[1,:]), '\ny_std:', np.std(x[1,:]))

# def DLT_P3(X_model, x_model):

#     n = np.size(x_model,1)
#     M = []

#     for i in range(n):

#         X = X_model[0,i]
#         Y = X_model[1,i]
#         Z = X_model[2,i]

#         x = x_model[0,i]
#         y = x_model[1,i]

#         m = np.array([[X, Y, Z, 1, 0, 0, 0, 0, -x*X, -x*Y, -x*Z, -x],
#                       [0, 0, 0, 0, X, Y, Z, 1, -y*X, -y*Y, -y*Z, -y]])

#         M.append(m)

#     M = np.concatenate(M, 0)
#     U, S, VT = LA.svd(M, full_matrices=False)
#     M_approx = U @ np.diag(S) @ VT

#     v = VT[-1,:] # last row of VT because optimal v should be last column of V
#     Mv = M @ v

#     print('||Mv||:', (Mv @ Mv)**0.5)
#     print('||v||^2:', v @ v)
#     print('max{||M - M_approx||}:', np.max(np.abs(M - M_approx)))
#     # print('S:', S)
    
#     return VT

# def estimate_camera_DLT_v2(X_model, x_model):

#     VT = DLT_P3(X_model, x_model)
#     P = np.stack([VT[-1, i:i+4] for i in range(0, 12, 4)], 0)

#     return P

def estimate_camera_DLT(Xmodel, img_pts, print_svd=False):

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

    if print_svd:
        M_approx = U @ np.diag(S) @ VT
        v = VT[-1,:] # last row of VT because optimal v should be last column of V
        Mv = M @ v
        print('\n||Mv||:', (Mv @ Mv)**0.5)
        print('||v||^2:', v @ v)
        print('max{||M - M_approx||}:', np.max(np.abs(M - M_approx)))
        print('S:', S)

    return P

def triangulate_3D_point_DLT(P1, P2, img1_pts, img2_pts, print_svd=False):
    
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
        
        if print_svd:
            M_approx = U @ np.diag(S) @ VT
            v = VT[-1,:] # last row of VT because optimal v should be last column of V
            Mv = M @ v
            print('\n||Mv||:', (Mv @ Mv)**0.5)
            print('||v||^2:', v @ v)
            print('max{||M - M_approx||}:', np.max(np.abs(M - M_approx)))
            print('S:', S)

    X = dehomogenize(np.stack(X,1))
    return X

def get_triangulated_X_from_extracted_P2_solutions(P1, P2_arr, x1_norm, x2_norm):
    X_arr = np.array([triangulate_3D_point_DLT(P1, P2, x1_norm, x2_norm, print_svd=False) for P2 in P2_arr])
    return X_arr

def enforce_fundamental(F):
    U, S, VT = LA.svd(F, full_matrices=False)
    # if LA.det(U @ VT) < 0:
    #     VT = -VT
    S[-1] = 0
    F = U @ np.diag(S) @ VT
    return F  

def estimate_F_DLT(img_pts_1, img_pts_2, enforce=False, verbose=False): # Computes F such that x2.T @ F @ x1 = 0

    n = np.size(img_pts_1,1)
    M = []

    for i in range(n):

        # x1 = img_pts_1[0,i]
        # y1 = img_pts_1[1,i]
        # z1 = img_pts_1[2,i]
        
        # x2 = img_pts_2[0,i]
        # y2 = img_pts_2[1,i]
        # z2 = img_pts_2[2,i]

        # m = np.array([[x2*x1, x2*y1, x2*z1, y2*x1, y2*y1, y2*z1, z2*x1, z2*y1, z2*z1]])

        x = img_pts_1[:,i]
        y = img_pts_2[:,i]
        m = np.outer(y, x).flatten()
        M.append([m])


    M = np.concatenate(M, 0)
    U, S, VT = LA.svd(M, full_matrices=False)
    F = VT[-1,:].reshape(3,3)
    if enforce:
        F = enforce_fundamental(F)

    if verbose:
        for i in range(np.size(img_pts_1,1)):
            epi_const = img_pts_2[:,i].T @ F @ img_pts_1[:,i]
            print('x2^T @ F @ x1:', epi_const)
        
        print('\nDet(F):', LA.det(F))
        M_approx = U @ np.diag(S) @ VT
        v = VT[-1,:] # last row of VT because optimal v should be last column of V
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

def estimate_E_DLT(img_pts_1, img_pts_2, enforce=False, verbose=False):

    n = np.size(img_pts_1,1)
    M = []

    for i in range(n):

        x = img_pts_1[:,i]
        y = img_pts_2[:,i]
        m = np.outer(y, x).flatten()
        M.append([m])

    M = np.concatenate(M, 0)
    U, S, VT = LA.svd(M, full_matrices=False)
    E = VT[-1,:].reshape(3,3)
    if enforce:
        E = enforce_essential(E)

    if verbose:
        for i in range(np.size(img_pts_1,1)):
            epi_const = img_pts_2[:,i].T @ E @ img_pts_1[:,i]
            print('x2^T @ E @ x1:', epi_const)
        
        print('\nDet(E):', LA.det(E))
        M_approx = U @ np.diag(S) @ VT
        v = VT[-1,:] # last row of VT because optimal v should be last column of V
        Mv = M @ v
        print('||Mv||:', (Mv @ Mv)**0.5)
        print('||v||^2:', v @ v)
        print('max{||M - M_approx||}:', np.max(np.abs(M - M_approx)))
        print('S:', S)

    E = E/E[-1,-1]
    return E

def estimate_E_robust(K, x1_norm, x2_norm, n_its, n_samples, err_threshold_px):
    
    err_threshold = err_threshold_px / K[0,0]
    best_inliers = None
    best_E = None
    max_inliers = 0

    for t in trange(n_its):

        rand_mask = np.random.choice(np.size(x1_norm,1), n_samples, replace=False)
        E = estimate_E_DLT(x1_norm[:,rand_mask], x2_norm[:,rand_mask], enforce=True, verbose=False)

        D1, D2 = compute_epipolar_errors(E, x1_norm, x2_norm)
        inliers = ((D1**2 + D2**2) / 2) < err_threshold**2

        n_inliers = np.sum(inliers)

        if n_inliers > max_inliers:
            best_inliers = np.copy(inliers)
            best_E = np.copy(E)
            max_inliers = n_inliers

            print(np.sum(inliers), end='\r')
        
    return best_E, best_inliers

def compute_ransac_iterations(alpha, epsilon, s):
    T = np.ceil(np.log(1-alpha) / np.log(1-epsilon**s))
    return T

def normalize_camera(P):
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

        ax.plot(x, y, '-', lw=3, color=col[i], alpha=0.7) #  label='Line {}'.format(i+1)

def compute_RMS_error(D1, D2):
    e_rms = np.sqrt(np.sum(D1**2 + D2**2) / (2*np.size(D1, 0)))
    return e_rms

def compute_point_line_distance_2D(l, p):
    a = l[0,:]
    b = l[1,:]
    c = l[2,:]

    x = p[0,:]
    y = p[1,:]

    D = np.abs(a*x + b*y + c) / (a**2 + b**2)**0.5

    # numerator = np.abs(np.einsum("ij, ij->j", p, l))
    # denominator = LA.norm(l[:-1], axis=0)
    # D = numerator / denominator

    return D

def compute_epipolar_lines(F, x1, x2):
    # l3 = F @ x1
    # l4 = F.T @ x2

    l2 = np.einsum("ij,jk->ik", F, x1)
    l1 = np.einsum("ji,jk->ik", F, x2)
    return l1, l2

def compute_epipolar_errors(F, x1, x2):
    l1, l2 = compute_epipolar_lines(F, x1, x2)
    D1 = compute_point_line_distance_2D(l1, x1)
    D2 = compute_point_line_distance_2D(l2, x2)
    return D1, D2

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

def compute_sift_points(img1, img2, marg):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # img1 = cv2.medianBlur(img1, ksize = 5)
    img1 = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX)
    # img1 = img1.astype(np.float32)

    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.medianBlur(img2, ksize = 5)
    img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX)
    # img2 = img2.astype(np.float32)


    # sift = cv2.SIFT_create(int nfeatures=0, int nOctaveLayers=3, double contrastThreshold=0.04, double edgeThreshold=10, double sigma=1.6)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    # search_params = dict(checks=50)   # or pass empty dictionary

    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    # matches = flann.knnMatch(des1, des2, k=2)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    print('Number of matches:', np.size(matches,0))

    good_matches = []
    for m, n in matches:
        if m.distance < marg*n.distance:
            good_matches.append([m])

    draw_params = dict(matchColor=(255,0,255), singlePointColor=(0,255,0), matchesMask=None, flags=cv2.DrawMatchesFlags_DEFAULT)
    img_match = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, **draw_params)

    x1 = np.stack([kp1[match[0].queryIdx].pt for match in good_matches],1)
    x2 = np.stack([kp2[match[0].trainIdx].pt for match in good_matches],1)

    x1 = homogenize(x1, multi=True)
    x2 = homogenize(x2, multi=True)

    print('Number of good matches:', np.size(x1,1))

    return x1, x2, img_match

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
    # ax.legend(loc="upper right")
    ax.imshow(cv2.cvtColor(img_match, cv2.COLOR_BGR2RGB))
    fig.tight_layout()
    if save:
        fig.savefig(path, dpi=300)
    plt.show()

def point_line_distance_2D(l, p):
    a = l[0]
    b = l[1]
    c = l[2]

    x = p[0]
    y = p[1]

    d = np.abs(a*x + b*y + c) / (a**2 + b**2)**0.5
    return d

def get_skew_vector(T):
    t = np.array([T[2,1], T[0,2], T[1,0]])
    return t

def extract_valid_camera_and_points(P1, P_arr, X_arr):
        
    x1_arr = np.array([transform(P1, X) for X in X_arr])
    valid_coords_P1 = np.array([np.sum(x[-1] > 0) for x in x1_arr])

    x2_arr = np.array([transform(P_arr[i], X_arr[i]) for i in range(np.size(P_arr, 0))])
    valid_coords_P2 = np.array([np.sum(x[-1] > 0) for x in x2_arr]) 

    valid_coords = valid_coords_P1 + valid_coords_P2
    valid_coords_ind = np.argmax(valid_coords)
    X_valid = X_arr[valid_coords_ind]
    P2_valid = P_arr[valid_coords_ind]
    
    print('No. valid coords for each camera pair:', valid_coords)
    print('Argmax(P2_arr):', valid_coords_ind)

    return P2_valid, X_valid

def get_canonical_camera():
    P = np.concatenate((np.eye(3), np.zeros(3)[:,np.newaxis]), 1)
    return P

def extract_P_from_E(E):

    U, S, VT = LA.svd(E, full_matrices=False)
    # print(U @ np.diag([1,1,0] @ VT))
    print(S)

    if LA.det(U @ VT) < 0:
        VT = -VT

    W = np.array([[0,1,0],[-1,0,0],[0,0,1]])
    Z = np.array([[0,-1,0],[1,0,0],[0,0,0]])

    S1 = U @ Z @ U.T
    S2 = U @ Z.T @ U.T

    R1 = U @ W @ VT
    R2 = U @ W.T @ VT

    t1 = get_skew_vector(S1)
    t2 = get_skew_vector(S2)

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

def compute_reprojection_error(P1, P2, Xj, x1j, x2j):

    x1j_proj = transform_and_dehomogenize(P1, Xj)
    x2j_proj = transform_and_dehomogenize(P2, Xj)
    
    r1 = np.array([x1j[0] - x1j_proj[0], x1j[1] - x1j_proj[1]])
    r2 = np.array([x2j[0] - x2j_proj[0], x2j[1] - x2j_proj[1]])
    res = np.concatenate((r1, r2), 0)
    reproj_err = LA.norm(res)**2

    return reproj_err, res

def compute_total_reprojection_error(P1, P2, X_est, x1, x2):

    n_pts = np.size(X_est,1)
    reproj_err_tot = []
    res_tot = []

    for j in range(n_pts):

        Xj = X_est[:,j]
        x1j = x1[:,j]
        x2j = x2[:,j]

        reproj_err, res = compute_reprojection_error(P1, P2, Xj, x1j, x2j)
        reproj_err_tot.append(reproj_err)
        res_tot.append(res)
    
    res_tot = np.concatenate(res_tot, 0)

    print('\nTotal reprojection error:', round(np.sum(reproj_err_tot), 2))
    print('Median reprojection error:', round(np.median(reproj_err_tot), 2))
    print('Avg. reprojection error:', round(np.mean(reproj_err_tot), 2))

    return reproj_err_tot, res_tot

def compute_jacobian_of_r(Pi, Xj):
    J1 = (Pi[-1,:] * (Pi[0,:] @ Xj) / (Pi[-1,:] @ Xj)**2) - (Pi[0,:] / (Pi[-1,:] @ Xj))
    J2 = (Pi[-1,:] * (Pi[1,:] @ Xj) / (Pi[-1,:] @ Xj)**2) - (Pi[1,:] / (Pi[-1,:] @ Xj))
    J = np.row_stack((J1, J2))
    return J

def linearize_reprojection_error(P1, P2, Xj, x1j, x2j):

    _, r = compute_reprojection_error(P1, P2, Xj, x1j, x2j)

    J1 = compute_jacobian_of_r(P1, Xj)
    J2 = compute_jacobian_of_r(P2, Xj)
    J = np.row_stack((J1, J2))
    
    return r, J

def compute_update(r, J, mu):
    I = np.eye(np.size(J,1))
    delta_Xj = -LA.inv(J.T @ J + mu*I) @ J.T @ r
    return delta_Xj

def optimize_X(P1, P2, X, x1, x2, mu_init, n_its):

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
            r, J = linearize_reprojection_error(P1, P2, Xj, x1j, x2j)
            delta_Xj = compute_update(r, J, mu)
            Xj_opt = dehomogenize(Xj + delta_Xj)
            
            reproj_err, _ = compute_reprojection_error(P1, P2, Xj, x1j, x2j)
            reproj_err_opt, _ = compute_reprojection_error(P1, P2, Xj_opt, x1j, x2j)

            if np.isclose(reproj_err_opt, reproj_err):
                converged = True
            elif reproj_err_opt < reproj_err:
                Xj = Xj_opt
                mu /= 10
            else:
                mu *= 10
        
        X[:,j] = Xj
        ts.append(t)

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

        # ax.plot(x_axis, y_axis, z_axis, 'o', label='Axis')
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