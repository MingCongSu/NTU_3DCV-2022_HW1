from math import sqrt
import sys
import numpy as np
import cv2 as cv

def scaler(points, pt_mean):
    s = 0
    for m in range(len(points)):
        s += (points[m][0]-pt_mean[0]) * (points[m][0]-pt_mean[0]) + (points[m][1]-pt_mean[1]) * (points[m][1]-pt_mean[1])
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

WINDOW_NAME = 'window'
path = 'images/book.jpg'
# book aspect ratio(width : height) = 3 : 4
def on_mouse(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        param[0].append([x, y])  


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('[USAGE] python3 2.py images/book.jpg')
        sys.exit(1)

    img = cv.imread(sys.argv[1])
    img = cv.resize(img, (0, 0), fx = 0.4, fy = 0.4, interpolation = cv.INTER_AREA) #original size * 0.4
    print("Please collect points clockwisely!")
    
    points_add= []
    cv.namedWindow(WINDOW_NAME)
    cv.setMouseCallback(WINDOW_NAME, on_mouse, [points_add])
    while True:
        img_ = img.copy()
        for i, p in enumerate(points_add):
            # draw points on img_
            cv.circle(img_, tuple(p), 2, (0, 255, 0), -1)
        cv.imshow(WINDOW_NAME, img_)

        key = cv.waitKey(20) % 0xFF
        if key == 27: break # exist when pressing ESC
    cv.destroyAllWindows()
    print('{} Points added:'.format(len(points_add)))

    points1 = points_add
    print(points1)
    points2 = [[0,0], [449,0], [449,599], [0,599]] #[width(x), height(y)] = 450 * 600
    points1 = np.c_[points1, np.transpose(np.ones(4))]
    points2 = np.c_[points2, np.transpose(np.ones(4))]
    H = NDLT(points1, points2)


    warp1 = np.ones_like(img) * 255
    warp2 = np.ones_like(img) * 255
    for m in range(600): #y
        for n in range(450): #x
            index = np.dot(np.linalg.inv(H), np.array([n,m,1])) #H^-1 * pt_hat = (lamda^-1) * ps
            index /= index[2]
            a = [int(index[0]), int(index[1])]
            b = [int(index[0])+1, int(index[1])]
            c = [int(index[0])+1, int(index[1])+1]
            d = [int(index[0]), int(index[1])+1]
            wa = (1-index[0]%1)*(1-index[1]%1)
            wb = (index[0]%1)*(1-index[1]%1)
            wc = (index[0]%1)*(index[1]%1)
            wd = (1-index[0]%1)*(index[1]%1)
            warp1[m][n] = img[int(index[1])][int(index[0])]
            #x = wa*a + wb*b + wc*c + wd*d
            temp = wa*img[a[1],a[0]] + wb*img[b[1],b[0]] + wc*img[c[1],c[0]] + wd*img[d[1],d[0]]
            warp2[m][n] = temp

    #cv.imshow('backward_warping', warp1)
    cv.imshow('bilinear_interpolation',warp2)
    cv.waitKey(0)
    cv.destroyAllWindows()