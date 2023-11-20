import numpy as np
from scipy.io import loadmat
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
from matplotlib.pyplot import cm
import cv2

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
    C = -1*(np.linalg.inv(M) @ P4)
    return C

def compute_normalized_principal_axis(P):
    M = P[:,:3]
    m3 = P[-1,:3]
    v = np.linalg.det(M) * m3
    return normalize_vector(v)

def compute_camera_center_and_normalized_principal_axis(P):
    C = compute_camera_center(P)
    p_axis = compute_normalized_principal_axis(P) 
    return C, p_axis

def compute_camera_and_normalized_principal_axis(P, multi=False):

    if multi:
        C_arr = np.array([homogenize(compute_camera_center(P[i])) for i in range(np.size(P))])
        axis_arr = np.array([homogenize(compute_normalized_principal_axis(P[i])) for i in range(np.size(P))])
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
    H_tot = K @ H @ np.linalg.inv(K)
    return H, H_tot

def transform_image(img, H):
    width = np.size(img, 1)
    height = np.size(img, 0)
    transf_img = cv2.warpPerspective(img, H, (width, height))
    return transf_img

def uncalibrate(p_cal, K):
    p_uncal = np.linalg.inv(K) @ p_cal
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

def normalise_x_and_y(img_pts):

    axis = 0
    x_mean, x_std = find_mean_and_std(img_pts, axis)

    axis = 1
    y_mean, y_std = find_mean_and_std(img_pts, axis)

    print('x_mean:', x_mean, '\nx_std:', x_std, '\ny_mean:', y_mean, '\ny_std:', y_std)

    N = np.array([[1/x_std, 0, -x_mean/x_std],
                    [0, 1/y_std, -y_mean/y_std],
                    [0, 0, 1]])
    
    img_pts_norm = N @ img_pts

    return img_pts_norm, N

def estimate_camera_DLT(Xmodel, img_pts):

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
    U, S, VT = np.linalg.svd(M, full_matrices=False)
    P = np.stack([VT[-1, i:i+4] for i in range(0, 12, 4)], 0)
    # print(np.shape(U), '\n\n',np.shape(S), '\n\n', np.shape(VT))
    M_approx = U @ np.diag(S) @ VT

    v = VT[-1,:] # last row of VT because optimal v should be last column of V
    Mv = M @ v
    print('||Mv||:', (Mv @ Mv)**0.5)
    print('||v||^2:', v @ v)
    print('max{||M - M_approx||}:', np.max(np.abs(M - M_approx)))
    # print('S:', S)

    return P

def rq(a):

    m, n = a.shape
    e = np.eye(m)
    p = e[:, ::-1]
    q0, r0 = np.linalg.qr(p @ a[:, :m].T @ p)

    r = p @ r0.T @ p
    q = p @ q0.T @ p

    fix = np.diag(np.sign(np.diag(r)))
    r = r @ fix
    q = fix @ q

    if n > m:
        q = np.concatenate((q, np.linalg.inv(r) @ a[:, m:n]), axis=1)

    return r, q

def compute_sift_points(img1, img2, marg):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < marg*n.distance:
            good_matches.append([m])

    draw_params = dict(matchColor=(255,0,255), singlePointColor=(0,255,0), matchesMask=None, flags=cv2.DrawMatchesFlags_DEFAULT)
    img_match = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, **draw_params)

    img1_pts = np.float32([kp1[match[0].queryIdx].pt for match in good_matches])
    img2_pts = np.float32([kp2[match[0].trainIdx].pt for match in good_matches])

    print('Number of good matches:', np.size(img1_pts,0))

    return img1_pts, img2_pts, img_match

def get_sift_plot_points(img1_pts, img2_pts, img1):
    x = [img1_pts[:,0], np.size(img1,1)+img2_pts[:,0]]
    y = [img1_pts[:,1], img2_pts[:,1]]
    return x, y

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
    
    fig = plt.figure(figsize=(10,8))
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
    plt.gca().invert_yaxis()

    plt.legend(loc="lower right")
    plt.imshow(img)
    fig.savefig(path, dpi=300)

    plt.show()