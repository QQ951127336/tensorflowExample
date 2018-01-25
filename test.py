# coding:utf-8
import cv2
import matplotlib.pyplot as plt



# 自适应中值滤波函数 adaptiveProcess
def adaptiveProcess(im, row, col, kernelSize, maxSize):
    pixels = []
    for a in range((-(kernelSize // 2)), kernelSize // 2 + 1):
        for b in range((-(kernelSize // 2)), kernelSize // 2 + 1):
            q = int(im[row + a, col + b])
            pixels.append(q)
    pixels.sort()
    min = pixels[0]
    max = pixels[kernelSize * kernelSize - 1]
    med = pixels[kernelSize * kernelSize // 2]
    zxy = im[row][col]
    if med > min and med < max:
        if zxy > min and zxy < max:
            return zxy
        else:
            return med
    else:
        kernelSize = kernelSize + 2
        if kernelSize <= maxSize:
            return adaptiveProcess(im, row, col, kernelSize, maxSize)
        else:
            return med


# 以灰度图像读取
im = cv2.imread("0047.jpg", 0)
plt.subplot(411)
plt.imshow(im, 'gray')

minSize = 3
maxSize = 7
m = im.shape[0]
n = im.shape[1]
im1 = im
# 扩展图像边缘
im1 = cv2.copyMakeBorder(im, maxSize // 2, maxSize // 2, maxSize // 2, maxSize // 2, borderType=cv2.BORDER_REFLECT,
                         dst=im1, value=0)
# 自适应中值滤波
for i in range(m):
    for j in range(n):
        im1[i][j] = adaptiveProcess(im1, i, j, minSize, maxSize)
plt.subplot(412)
plt.imshow(im1, 'gray')

# 图像膨胀腐蚀，核为3*3矩阵
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
im2 = cv2.morphologyEx(im1, cv2.MORPH_OPEN, kernel)
plt.subplot(413)
plt.imshow(im2, 'gray')

# 直方图均衡化
im3 = cv2.equalizeHist(im2)

plt.subplot(414)
plt.imshow(im3, 'gray')
plt.show()
