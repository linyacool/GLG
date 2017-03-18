# coding=utf-8
import numpy as np
import cv2
import scipy.misc
import os
#import matplotlib.pyplot as plt
import time
import GLG
L = 256
subsize_H = 30
subsize_W = 30

def aglg(img):
    #image = np.copy(img)
    Ori_height,Ori_width = np.shape(img)
    height,width = np.shape(img)
    if height % subsize_H != 0 or width % subsize_W != 0:
        height = int(height / subsize_H) * subsize_H
        width = int(width / subsize_W) * subsize_W
    #res = [[0 for i in range(width)] for j in range(height)]
    res = np.zeros((height, width))
    M = int(height / subsize_H)
    N = int(width / subsize_W)
    M += 1
    N += 1
    T = [[[] for j in range(N + 1)] for i in range(M + 1)]
    for i in range(0, M):
        for j in range(0, N):
            x = i * subsize_H
            y = j * subsize_W
            if i == M - 1 and j != N - 1:
                temp_img = img[x:Ori_height+1,y:y+subsize_W]
            elif j == N - 1 and i != M - 1:
                temp_img = img[x:x+subsize_H, y:Ori_width+1]
            elif i == M - 1 and j == N - 1:
                temp_img = img[x:Ori_height+1, y:Ori_width+1]
            else:
                temp_img = img[x:x+subsize_H,y:y+subsize_W]
            T[i+1][j+1], temp = GLG.glg(temp_img)
    A = [[[0 for k in range(L)] for j in range(N + 2)] for i in range(M + 2)]
    # for thr four corner components
    A[1][1] = T[1][1]
    A[1][N+1] = T[1][N]
    A[M+1][1] = T[M][1]
    A[M+1][N+1] = T[M][N]
    # for the boundary components
    for j in range(2,N+1):
        for k in range(0,L):
            if T[1][j-1][k] == L-1:
                A[1][j][k] = T[1][j][k]
            elif T[1][j][k] == L-1:
                A[1][j][k] = T[1][j-1][k]
            else:
                A[1][j][k] = (T[1][j-1][k] + T[1][j][k]) / 2
            #if T[M][j-1][k] == L-1:
            #   A[M+1][j][k] = T[M][j][k]
            #elif T[M][j][k] == L-1:
            #    A[M+1][j][k] = T[M][j-1][k]
            #else:
            A[M+1][j][k] = (T[M][j-1][k] + T[M][j][k]) / 2
    for i in range(2,M+1):
        for k in range(0,L):
            if T[i-1][1][k] == L-1:
                A[i][1][k] = T[i][1][k]
            elif T[i][1][k] == L-1:
                A[i][1][k] = T[i-1][1][k]
            else:
                A[i][1][k] = (T[i-1][1][k] + T[i][1][k]) / 2
            #if T[i - 1][N][k] == L - 1:
            #   A[i][N+1][k] = T[i][N][k]
            #elif T[i][N][k] == L - 1:
            #    A[i][N+1][k] = T[i-1][N][k]
            #else:
            A[i][N+1][k] = (T[i-1][N][k] + T[i][N][k]) / 2
    #for the interior components
    for i in range(2,M+1):
        for j in range(2,N+1):
            for k in range(0,L):
                p = 0
                sum = 0
                if T[i-1][j-1][k] != L-1:
                    p = p + 1
                    sum = sum + T[i-1][j-1][k]
                if T[i-1][j][k] != L-1:
                    p = p + 1
                    sum = sum + T[i-1][j][k]
                if T[i][j-1][k] != L-1:
                    p = p + 1
                    sum = sum + T[i][j-1][k]
                if T[i][j][k] != L-1:
                    p = p + 1
                    sum = sum + T[i][j][k]
                if p != 0:
                    A[i][j][k] = sum / p
                else:
                    A[i][j][k] = k
    #perform bilinear intrpolation
    for i in range(M-1):  #i+1
        for j in range(N-1): #j+1
            for x in range(subsize_H):
                for y in range(subsize_W):
                    X = i * subsize_H + x
                    Y = j * subsize_W + y
                    k = img[X][Y]
                    temp = float((subsize_H - x) * ((subsize_W - y) * A[i+1][j+1][k] + (y + 1) * A[i+1][j+2][k]) + (x + 1) * ((subsize_H - y) * A[i+2][j+1][k] + (y+1) * A[i+2][j+2][k])) / float((subsize_H + 1) * (subsize_W + 1))
                    res[X][Y] = temp
    return res

def main():
    parser = GLG.build_parser()
    options = parser.parse_args()
    if not os.path.isfile(options.img):
        parser.error("Image %s does not exist.)" % options.network)
    res = options.res
    img = cv2.imread(options.img,cv2.IMREAD_GRAYSCALE)
    height,width = np.shape(img)
    flag = False
    if height % subsize_H == 0:
        height += 1
        flag = True
    if width % subsize_W == 0:
        width += 1
        flag = True
    if flag:
        img = cv2.resize(img, (height, width))
    image = aglg(img)
    print 'ten:',GLG.ten(image)
    scipy.misc.imsave(res,image)
if __name__ == '__main__':
    main()