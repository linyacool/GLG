# coding=utf-8
import numpy as np
import cv2
import scipy.misc
import os
#import matplotlib.pyplot as plt
import warnings
import GLG
warnings.filterwarnings("ignore")
ALPHA = 0.8
M = 256
GROUP = 20

def fglg(img):
    height,width = np.shape(img)
    Npix = height * width
    scipy.misc.imsave('original_img.jpg',img)
    hist = cv2.calcHist([img],[0],None,[256],[0.0,255.0])

    # show histogram of the original image
    #plt.hist(hist.flatten(), 256)
    #plt.show()
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
    T = [0 for i in range(M+2)]

    while n - 1 >= GROUP:
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
        n = n - 1


    n = n + 1
    if L[n-1][1] != R[n-1][1]:
        N = (M - 1)/float(n - 1)
    else:
        N = (M - 1)/float(n - 1 - ALPHA)
    for k in range(0,M):
        if k <= L[n-1][1]:
            T[k] = 0
            continue
        if k >= R[n-1][n-1]:
            T[k] = M - 1
            continue
        i = 0
        for x in range(1,n):
            if k >= L[n-1][x] and k < R[n-1][x]:
                i = x
                if i > 0 and L[n-1][i] != R[n-1][i]:
                    if L[n-1][1] == R[n-1][1]:
                        ans = int((i - ALPHA - (R[n - 1][i] - k) / float(R[n - 1][i] - L[n - 1][i])) * float(N) + 1 + 0.5)
                        T[k] = ans
                    else:
                        ans = int((i - (R[n - 1][i] - k) / float(R[n - 1][i] - L[n - 1][i])) * float(N) + 1 + 0.5)
                        T[k] = ans
                elif i > 0 and L[n-1][i] == R[n-1][i]:
                    if L[n-1][1] == R[n-1][1]:
                        T[k] =int(((i - ALPHA) * float(N)) + 0.5)
                    else:
                        T[k] =int((i * float(N)) + 0.5)
            elif k == R[n-1][x]:
                i = x
                if L[n-1][1] == R[n-1][1]:
                    T[k] = int(((float (i) - ALPHA) * float(N)) + 0.5)
                else:
                    T[k] = int((i * float(N)) + 0.5)
             #There can be delete
            #if i == 0:
             #   T[n-1][k] = T[n-1][k-1]
    D = GLG.Trans_and_CalcD(hist,T)

    return T,D/(float (Npix) * (Npix - 1))

def main():
    parser = GLG.build_parser()
    options = parser.parse_args()
    if not os.path.isfile(options.img):
        parser.error("Image %s does not exist.)" % options.network)
    res = options.res
    img = cv2.imread(options.img,cv2.IMREAD_GRAYSCALE)
    Trans,PixDist = fglg(img)
    height, width = np.shape(img)
    #reconstruct the enhangced image
    image = np.copy(img)
    for i in range(0,height):
        for j in range(0,width):
            image[i][j] = Trans[img[i][j]]
    #print GLG.ten(img),GLG.ten(image)
    scipy.misc.imsave(res,image)
    print 'ten:', GLG.ten(image)
    print "The PixDist is %.1lf" %PixDist
if __name__ == '__main__':
    main()