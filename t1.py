import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_svd_input(left_coord, right_coord):
    output_array = []

    for i in range(len(left_coord)):
        xy1 = [left_coord[i].pt[0],left_coord[i].pt[1],1]
        zro = [0,0,0]
        # for right_x
        temp = [-1*left_coord[i].pt[0]*right_coord[i].pt[0] ,-1*left_coord[i].pt[1]*right_coord[i].pt[0],-1*right_coord[i].pt[0]]
        output_array.append(np.concatenate((xy1, zro, temp), axis=0))
        # for right_y
        temp = [-1*left_coord[i].pt[0]*right_coord[i].pt[1] ,-1*left_coord[i].pt[1]*right_coord[i].pt[1],-1*right_coord[i].pt[1]]
        output_array.append(np.concatenate((zro, xy1, temp), axis=0))

    return output_array

def stitch_background(img1, img2, savepath=''):
    "The output image should be saved in the savepath."
    "Do NOT modify the code provided."
    
    
   
    # TO DO: implement your solution here
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    print(des1.shape)
    feature_map = {}
    for i in range(des1.shape[0]):
        dist_list = np.linalg.norm(des1[i]- des2, axis=1)
        right_idx = range(0,des2.shape[0])
        left_idx = [i]*des2.shape[0]
        out = list(zip(dist_list, left_idx, right_idx))

        out = sorted(out, key=lambda x: x[0])
        feature_map[i] = out[0:2]

    valid_features = {}
    ratio_test = 0.75
    valid_kp1 = []
    valid_kp2 = []

    for k,v in feature_map.items():
        if v[0][0]< ratio_test*v[1][0]:
            valid_features[k] = v
            valid_kp1.append(kp1[v[0][1]])
            valid_kp2.append(kp2[v[0][2]])
    #     print(len(valid_features))

    # RANSAC
    k = 3000
    n = 4
    t = 0.3
    d = 200
    max_goodfit = 0
    good_fit_set = []
    for ransac_iter in range(k):
        x = np.random.randint(0, len(valid_kp1), size=(n))
        left_coord = [valid_kp1[i] for i in x]
        right_coord = [valid_kp2[i] for i in x]
        svd_input = get_svd_input(left_coord, right_coord)
        U, s, Vt = np.linalg.svd(svd_input)
        mat_x = Vt[8].reshape(3, 3)
        normalized_h = np.divide(mat_x, mat_x[2,2])

        inlier = []
        for i in range(len(valid_kp1)):
            if i not in x:
                test_p = np.matrix([valid_kp1[i].pt[0],valid_kp1[i].pt[1],1]).T
                target = np.matrix([valid_kp2[i].pt[0],valid_kp2[i].pt[1],1]).T
                pred = normalized_h*test_p
                pred = np.divide(pred,pred[2,0])
                dist = np.linalg.norm(pred-target)
                if dist < t:
                    inlier.append(i)

        if len(inlier) > max_goodfit:
            max_goodfit = len(inlier)
            good_fit_set = inlier

    # final H matrix
    left_coord = [valid_kp1[i] for i in good_fit_set]
    right_coord = [valid_kp2[i] for i in good_fit_set]
    svd_input = get_svd_input(left_coord, right_coord)
    U, s, Vt = np.linalg.svd(svd_input)
    mat_x = Vt[8].reshape(3, 3)
    normalized_h = np.divide(mat_x, mat_x[2,2])

    # corners:
    rows = img1.shape[0]
    cols = img1.shape[1]
    l_top = [0,cols-1]
    l_bottom = [rows-1, cols-1]


    max_width = cols*2
    max_height = rows+100
    result_img = cv2.warpPerspective(img1, normalized_h, (max_width,max_height))
    plt.imshow(result_img)
    plt.show()

    
    
    return result_img
   #return
if __name__ == "__main__":
    img1 = cv2.imread('t1_1.png',cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('t1_2.png',cv2.IMREAD_GRAYSCALE)
    savepath = 'task1.png'
    stitch_background(img1, img2, savepath=savepath)

