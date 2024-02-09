from math import sqrt
import sys
import numpy as np
import cv2 as cv

def scaler(points, pt_mean):
    s = 0
    for m in range(len(points)):
        s += (points[m][0]-pt_mean[0]) **2 + (points[m][1]-pt_mean[1]) **2
    s = sqrt(s / (2 * len(points)))

    return s

def DLT(points1, points2):
    #Basic DLT
    A = []
    for m in range(len(points1)):
        u = points1[m][0]
        v = points1[m][1]
        u_p = points2[m][0]
        v_p = points2[m][1]
        arr1 = [u,v,1,0,0,0,-(u_p*u),-(u_p*v),-u_p]
        arr2 = [0,0,0,u,v,1,-(v_p*u),-(v_p*v),-v_p]
        A.append(arr1)
        A.append(arr2)
    A = np.array(A)
    U,S,V = np.linalg.svd(A)
    V = np.transpose(V)
    V = np.reshape(V[:,-1],(3,3))

    return V

def NDLT(points1, points2):
    #Normalization using Hartley algorithm
    ''' 
    (m1,m2), (m1',m2') are centroid of points1, points2
    s, s' are scalar quantities

         [s^-1  ,    0  , -s^-1*m1]
    T  = [0     , s^-1  , -s^-1*m2]
         [0     ,    0  ,     1   ]

         [s'^-1 ,   0  , -s'^-1*m1']
    T' = [   0  , s'^-1, -s'^-1*m2']
         [   0  ,   0  ,     1    ]
    '''
    #Compute similarity transform T and T'
    points1 = np.array(points1)
    points2 = np.array(points2)
    mean = points1.mean(axis=0)
    mean_prime = points2.mean(axis=0)

    s = scaler(points1, mean)
    s_prime = scaler(points2, mean_prime)

    m1, m2 = mean[0], mean[1] 
    m1_prime, m2_prime = mean_prime[0], mean_prime[1] # m1' and  m2'

    T = [   [1/s    , 0  , -((1/s)*m1)],
            [0      , 1/s, -((1/s)*m2)],
            [0      , 0  ,           1]]
    T_prime = [ [1/s_prime  , 0         ,-((1/s_prime)*m1_prime)],
                [0          , 1/s_prime ,-((1/s_prime)*m2_prime)],
                [0          , 0         ,           1           ]]
    T = np.array(T)
    T_prime = np.array(T_prime)

    #Normalization via similarity transform
    x_tilde = np.transpose(np.dot(T, np.transpose(points1)))
    x_prime_tilde = np.transpose(np.dot(T_prime, np.transpose(points2)))

    #Apply basic DLT
    H_tilde = DLT(x_tilde, x_prime_tilde)
    #Denormalization : H = T^-1 * H^~ * T
    H = np.matmul(np.linalg.inv(T_prime), np.matmul(H_tilde, T))
    
    return H

def loss(H):
    #L2 norm
    gt_pairs = np.load(sys.argv[3])
    p_s = gt_pairs[0] #source
    p_t = gt_pairs[1] #groundtruth
    
    p_s = np.c_[p_s, np.transpose(np.ones(100))]
    p_s = np.transpose(p_s)
    p_t_hat = np.dot(H,p_s)
    p_t_hat = np.transpose(p_t_hat) 

    # p_t_hat = H * p_s / lamda
    error = 0
    for m in range(p_t_hat.shape[0]):
        error += sqrt((p_t[m][0]-p_t_hat[m][0]/p_t_hat[m][2]) **2 + (p_t[m][1] - p_t_hat[m][1]/p_t_hat[m][2]) **2)
    error /= p_t_hat.shape[0]
    print("Loss:",error)
    return error

def get_sift_correspondences(img1, img2):
    '''
    Input:
        img1: numpy array of the first image
        img2: numpy array of the second imagey

    Return:
        points1: numpy array [N, 2], N is the number of correspondences
        points2: numpy array [N, 2], N is the number of correspondences
    '''
    #sift = cv.xfeatures2d.SIFT_create()# opencv-python and opencv-contrib-python version == 3.4.2.16 or enable nonfree
    sift = cv.SIFT_create()             # opencv-python==4.5.1.48
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    matcher = cv.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k = 2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.6488 * n.distance:
            good_matches.append(m)

    #print("find", len(good_matches), "pairs")
    good_matches = sorted(good_matches, key=lambda x: x.distance)    
    good_matches = good_matches[:int(sys.argv[4])]

    points1 = np.array([kp1[m.queryIdx].pt for m in good_matches])
    points2 = np.array([kp2[m.trainIdx].pt for m in good_matches])
    
    img_draw_match = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.namedWindow("images", cv.WINDOW_NORMAL)
    cv.resizeWindow('images', 1280, 720)
    cv.imshow('images', img_draw_match)
    cv.waitKey(0)
    return points1, points2


if __name__ == '__main__':
    #python3 1.py images/1-0.png images/1-1.png groundtruth_correspondences/correspondence_01.npy 8 NDLT
    if len(sys.argv) < 6:
        print('[USAGE] python3 1.py [source image path] [target image path] [correspondence path] [k_pairs] [DLT/NDLT]')
        sys.exit(1)
    img1 = cv.imread(sys.argv[1])
    img2 = cv.imread(sys.argv[2])
   
    gt_correspondences = np.load(sys.argv[3])
    points1, points2 = get_sift_correspondences(img1, img2)
    points1 = np.c_[points1, np.transpose(np.ones(points1.shape[0]))]
    points2 = np.c_[points2, np.transpose(np.ones(points2.shape[0]))]

    if sys.argv[5] == 'DLT':
        H = DLT(points1, points2)
    elif sys.argv[5] == 'NDLT':
        H = NDLT(points1, points2)
    else:
        print("Wrong input.")

    loss(H)
