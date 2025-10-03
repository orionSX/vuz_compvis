import cv2
import numpy as np

image = cv2.imread("D:\GitHub\compvis\lab3\my_img.jpg")
image_BW = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def gk(size, sigma):
    kernel = np.zeros((size, size))
    center = size // 2
    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))

    return kernel / np.sum(kernel)


def gf(image, ksize, sigma):
    kernel = gk(ksize, sigma)
    krad = ksize // 2
    filtered_image = image.copy()

    for i in range(krad, image.shape[0] - krad):
        for j in range(krad, image.shape[1] - krad):
            row_start, row_end = i - krad, i + krad + 1
            col_start, col_end = j - krad, j + krad + 1
            neighborhood_region = image[row_start:row_end, col_start:col_end]
            filtered_image[i, j] = np.sum(neighborhood_region * kernel)

    return filtered_image


for size in [3, 5, 7]:
    kernel = gk(size, 1.0)
    print(f"Матрица {size}x{size}:\n{kernel}\n")

params = [
    (7, 1.5),
    (25, 5.0),
]
results = []

for ksize, sigma in params:
    filtered = gf(image_BW, ksize, sigma)
    results.append(filtered)

r1, r2 = results[0], results[1]
print(f"Разница по параметрам:{np.abs(np.mean(r1) - np.mean(r2))}")


derevnya, opencv = gf(image_BW, 25, 5.0), cv2.GaussianBlur(image_BW, (25, 25), 5.0)
print(f"Разница с OpenCV:{np.abs(np.mean(derevnya) - np.mean(opencv))}")


cv2.imshow("Orig", image_BW)
cv2.imshow("Mine", derevnya)
cv2.imshow("OpenCV", opencv)
cv2.waitKey(0)
cv2.destroyAllWindows()
