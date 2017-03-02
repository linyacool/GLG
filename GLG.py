# coding=utf-8
import numpy as np
import cv2
import scipy.misc
import os
#import matplotlib.pyplot as plt
import warnings
from argparse import ArgumentParser
warnings.filterwarnings("ignore")
ALPHA = 0.8
M = 256
Threshold = 10

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--image',
                        dest = 'img', help = 'input image',
                        metavar = 'INPUT_IMAGE.jpg', required = True)
    parser.add_argument('--result',
                        dest='res', help='output image',
                        metavar='OUTPUT_IMAGE.jpg', required=True)
    return parser

def Trans_and_CalcD(H = [],T = []):
    M = len(H)
    Tar = np.zeros(M+1)
    for i in range(M):
        if H[i] != 0:
            Tar[T[i]] = H[i]
    D = 0
    for i in range(0,M-1):
        for j in range(i+1,M):
            D = D + Tar[i] * Tar[j] * (j - i)
    return D

def ten(img):
    height, width = np.shape(img)
    '''
    ix = [-1,0,1   iy = [1,2,1
          -2,0,2         0,0,0
          -1,0,1]        -1,-2,-1]
    '''
    ans = 0
    for i in range(1,height-1):
        for j in range(1,width-1):
            Sx = img[i-1][j+1] + 2 * img[i][j+1] + img[i+1][j+1] - (img[i-1][j-1] + 2 * img[i][j-1] + img[i+1][j-1])
            Sy = img[i-1][j-1] + 2 * img[i-1][j] + img[i-1][j+1] - (img[i+1][j-1] + 2 * img[i+1][j] + img[i+1][j+1])
            temp = Sx * Sx + Sy * Sy
            if temp > Threshold:
                ans = ans + temp
    return ans

def glg(img):
    height,width = np.shape(img)
    Npix = height * width
    scipy.misc.imsave('original_img.jpg', img)
    hist = cv2.calcHist([img], [0], None, [M], [0.0, 255.0])
    #show histogram of the original image
    '''
    bins = np.arange(257)
    item = img[:, :]
    hist, bins = np.histogram(item, bins)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()
    '''
    temp = [0]
    temp_gray_level = np.zeros(M)
    cnt = 1
    for i in range(M):
        if hist[i] != 0:
            temp.append(hist[i])
            temp_gray_level[cnt] = i
            cnt = cnt + 1
    n = len(temp) - 1
    G = [[0] for i in range(n+2)]
    gray_level = [[0 for _ in range(n+1)] for __ in range(n+1)]
    G[n] = temp
    gray_level[n] = temp_gray_level
    L = [[0] for i in range(n+2)]
    R = [[0] for i in range(n+2)]
    for k in range(M):
        if hist[k] != 0:
            L[n].append(k)
            R[n].append(k)
    N = np.zeros(n+2).astype(float)
    T = [[0 for k in range(M+1)] for i in range(n+2)]
    D = np.zeros(n+2)
    maxD = 0
    Iopt = n - 1

    while n >= 3:
        #compute Gn-1,Ln-1,Rn-1,i'
        a = min(G[n][1:n+1])
        ia = G[n].index(a)
        left = True
        if ia == 1:
            b = G[n][ia+1]
            left = False
        elif ia == n:
            b = G[n][ia-1]
        else:
            if G[n][ia-1] <= G[n][ia+1]:
                b = G[n][ia-1]
                left = True
            else:
                b = G[n][ia+1]
                left = False
        if left:
            ii = ia - 1
        else:
            ii = ia
        for i in range(1,ii):
            G[n-1].append(G[n][i])
            gray_level[n-1][i] = gray_level[n][i]
        G[n-1].append(a+b)
        gray_level[n-1][ii] = gray_level[n][ii]
        for i in range(ii+1,n):
            G[n-1].append(G[n][i+1])
            gray_level[n-1][i] = gray_level[n][i+1]

        for i in range(1,ii+1):
            L[n-1].append(L[n][i])
        for i in range(ii+1,n):
            L[n-1].append(L[n][i+1])

        for i in range(1,ii):
            R[n-1].append(R[n][i])
        for i in range(ii,n):
            R[n-1].append(R[n][i+1])

        if L[n-1][1] != R[n-1][1]:
            N[n-1] = (M - 1)/float(n - 1)
        else:
            N[n-1] = (M - 1)/float(n - 1 - ALPHA)
        for k in range(0,M):
            if k <= L[n-1][1]:
                T[n - 1][k] = 0
                continue
            if k >= R[n-1][n-1]:
                T[n-1][k] = M - 1
                continue
            i = 0
            for x in range(1,n):
                if k >= L[n-1][x] and k < R[n-1][x]:
                    i = x
                    if i > 0 and L[n-1][i] != R[n-1][i]:
                        if L[n-1][1] == R[n-1][1]:
                            ans = int((i - ALPHA - (R[n - 1][i] - k) / float(R[n - 1][i] - L[n - 1][i])) * float(N[n - 1]) + 1 + 0.5)
                            T[n - 1][k] = ans
                        else:
                            ans = int((i - (R[n - 1][i] - k) / float(R[n - 1][i] - L[n - 1][i])) * float(N[n - 1]) + 1 + 0.5)
                            T[n - 1][k] = ans
                    elif i > 0 and L[n-1][i] == R[n-1][i]:
                        if L[n-1][1] == R[n-1][1]:
                            T[n - 1][k] =int(((i - ALPHA) * float(N[n - 1])) + 0.5)
                        else:
                            T[n - 1][k] =int((i * float(N[n - 1])) + 0.5)
                elif k == R[n-1][x]:
                    i = x
                    if L[n-1][1] == R[n-1][1]:
                        T[n - 1][k] = int(((float (i) - ALPHA) * float(N[n - 1])) + 0.5)
                    else:
                        T[n - 1][k] = int((i * float(N[n - 1])) + 0.5)
             #can be deleted
                if i == 0:
                    T[n-1][k] = T[n-1][k-1]
        D[n-1] = Trans_and_CalcD(hist,T[n-1])
        if D[n - 1] > maxD:
            maxD = D[n - 1]
            Iopt = n - 1
        #print n - 1, D[n - 1]
        n = n - 1
    return T[Iopt],D[Iopt]/(float (Npix) * (Npix - 1))

def main():
    parser = build_parser()
    options = parser.parse_args()
    if not os.path.isfile(options.img):
        parser.error("Image %s does not exist.)" % options.network)
    res = options.res
    img = cv2.imread(options.img, cv2.IMREAD_GRAYSCALE)
    Trans,PixDist = glg(img)
    height, width = np.shape(img)
    #reconstruct the enhangced image
    image = np.copy(img)
    for i in range(0,height):
        for j in range(0,width):
            image[i][j] = Trans[img[i][j]]
    scipy.misc.imsave(res,image)
    print "The PixDist is %.1lf" %PixDist
if __name__ == '__main__':
    main()