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

def plot_cameras_and_axes(ax, C_list, axis_list, s):

    col = cm.rainbow(np.linspace(0, 1, np.size(C_list,0)))

    for i in range(np.size(C_list,0)):

        C = C_list[i,:]
        axis = axis_list[i,:]

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