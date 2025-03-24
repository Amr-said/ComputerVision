import numpy as np 
import cv2 as cv
from PIL import Image
from numpy.core.shape_base import _accumulate
from scipy import ndimage
import streamlit as st
import matplotlib.pyplot as plt
import base64
import math
import cv2
import numpy as np
from collections import defaultdict


@st.cache_data


#Step1: Noise Reduction

def gaussian_kernel(size, sigma, verbose=False):
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
    kernel_2D *= 1.0 / kernel_2D.max()
    return kernel_2D

@st.cache_data

def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)
@st.cache_data

def convolution(image, kernel, average=False, verbose=False):
    
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape
    output = np.zeros(image.shape)
    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)
    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image
    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            if average:
                output[row, col] /= kernel.shape[0] * kernel.shape[1]
    return output


@st.cache_data

#Step2: Gradient Calculation
def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    
    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)
    
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)
    
    return (G, theta)

@st.cache_data

#Step3: Non-maximum suppression
def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    
    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                q = 255
                r = 255
                
               #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i,j] >= q) and (img[i,j] >= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass
    
    return Z

@st.cache_data

#Step4: Double threshold
def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    
    highThreshold = img.max() * highThresholdRatio;
    lowThreshold = highThreshold * lowThresholdRatio;
    
    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)
    
    weak = np.int32(25)
    strong = np.int32(255)
    
    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)
    
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    return (res, weak, strong)

@st.cache_data

#Step5: Edge Tracking by Hysteresis
def hysteresis(img, weak, strong=255):
    M, N = img.shape  
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img


@st.cache_data

#Main Function:
def canny_apply(img_name,kernel_size,sigma):
    verbose = True
    # str_img_name=str(img_name)
    # img = cv.imread(str_img_name)
    kernel = gaussian_kernel(kernel_size, sigma, verbose=verbose)
    noisy_img  = convolution(img_name, kernel, True, verbose=verbose)
    (grad_img, theta) = sobel_filters(noisy_img)
    n_sup_img = non_max_suppression(grad_img, theta)
    (res_img, weak_mat, strong_mat) = threshold(n_sup_img)
    hys_img = hysteresis(res_img, weak_mat, strong_mat)
    img_final = Image.fromarray(hys_img)
    return img_final


@st.cache_data


def line_detection_non_vectorized(image, edge_image, num_rhos=180, num_thetas=180):
    edge_height, edge_width = edge_image.shape[:2]
    edge_height_half, edge_width_half = edge_height / 2, edge_width / 2
    #
    d = np.sqrt(np.square(edge_height) + np.square(edge_width))
    dtheta = (180 / num_thetas)
    drho = ((2 * d) / num_rhos)
    #
    thetas = np.arange(0, 180, step=dtheta)
    rhos = np.arange(-d, d, step=drho)
    #
    cos_thetas = np.cos(np.deg2rad(thetas))
    sin_thetas = np.sin(np.deg2rad(thetas))
    #
    accumulator = np.zeros((len(rhos), len(thetas)))
    #
    output_img = np.zeros((edge_height, edge_width, 3), np.uint8)
    #
      #
    figure = plt.figure(figsize=(12, 12))
    subplot1 = figure.add_subplot(1, 4, 1)
    subplot1.imshow(image)
    subplot2 = figure.add_subplot(1, 4, 2)
    subplot2.imshow(edge_image, cmap="gray")
    subplot3 = figure.add_subplot(1, 4, 3)
    subplot3.set_facecolor((0, 0, 0))
    subplot4 = figure.add_subplot(1, 4, 4)
    subplot4.imshow(output_img)
    # 
    for y in range(edge_height):
        for x in range(edge_width):
            if edge_image[y][x] != 0:
                edge_point = [y - edge_height_half, x - edge_width_half]
                ys, xs = [], []
                for theta_idx in range(len(thetas)):
                    rho = (edge_point[1] * cos_thetas[theta_idx]) + (edge_point[0] * sin_thetas[theta_idx])
                    theta = thetas[theta_idx]
                    rho_idx = np.argmin(np.abs(rhos - rho))
                    accumulator[rho_idx][theta_idx] += 1
                    ys.append(rho)
                    xs.append(theta)

    out_img = image.copy()
    indices, top_thetas, top_rhos = peak_votes(accumulator, rhos, thetas, 50)
    for i in range(len(indices)):
        rho = top_rhos[i]
        theta = top_thetas[i]
        a = np.cos(np.deg2rad(theta))
        b = np.sin(np.deg2rad(theta))
        x0 = (a * rho) + edge_width_half
        y0 = (b * rho) + edge_height_half
        x1 = int(x0 + 200 * (-b))
        y1 = int(y0 + 200 * (a))
        x2 = int(x0 - 200 * (-b))
        y2 = int(y0 - 200 * (a))
        out_img = cv.line(out_img, (x1, y1), (x2, y2), (0, 255, 0), 1)

    st.image(out_img, caption="Hough Line Image",use_column_width=True)
    return out_img

@st.cache_data


def peak_votes(accumulator, rhos, thetas, n):
    """ Finds the max number of votes in the hough accumulator """
    idx = np.argpartition(accumulator.flatten(), -n)[-n:]
    indices = idx[np.argsort((-accumulator.flatten())[idx])]
    top_rhos = rhos[(indices / accumulator.shape[1]).astype(int)]
    top_thetas = thetas[indices % accumulator.shape[1]]

    return indices, top_thetas, top_rhos

@st.cache_data


def hough_lines(path,size,sigma):
    decoded_data = base64.b64decode(path)
    
    np_data = np.fromstring(decoded_data, np.uint8)
    image = cv.imdecode(np_data, cv.IMREAD_COLOR)
    # edge_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    if len(image.shape) == 3:
        edge_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
       edge_image = image
    edge_image= canny_apply(edge_image,size,sigma)
    edge_image = np.asarray(edge_image)
    line_detection_non_vectorized(image, edge_image)
    # edge_image = cv.GaussianBlur(edge_image, (5, 5), 1)
    # edge_image = cv.dilate(edge_image, (3, 3), iterations=1)


@st.cache_data

def detectCircles(img,size,sigma, threshold, region, radius=None):
    """

    :param img:
    :param threshold:
    :param region:
    :param radius:
    :return:
    """
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.GaussianBlur(img, (5, 5), 1.5)
    # img = base64.b64decode(img)
    
    # img = np.fromstring(img, np.uint8)
    # img = cv.imdecode(img, cv.IMREAD_COLOR)
    img = canny_apply(img, size, sigma)
    img=  np.asarray(img)
    
    
    (M, N) = img.shape
    if radius == None:
        R_max = np.max((M, N))
        R_min = 3
    else:
        [R_max, R_min] = radius

    R = R_max - R_min
    # Initializing accumulator array.
    # Accumulator array is a 3 dimensional array with the dimensions representing
    # the radius, X coordinate and Y coordinate resectively.
    # Also appending a padding of 2 times R_max to overcome the problems of overflow
    A = np.zeros((R_max, M + 2 * R_max, N + 2 * R_max))
    B = np.zeros((R_max, M + 2 * R_max, N + 2 * R_max))

    # Precomputing all angles to increase the speed of the algorithm
    theta = np.arange(0, 360) * np.pi / 180
    edges = np.argwhere(img[:, :])  # Extracting all edge coordinates
    for val in range(R):
        r = R_min + val
        # Creating a Circle Blueprint
        bprint = np.zeros((2 * (r + 1), 2 * (r + 1)))
        (m, n) = (r + 1, r + 1)  # Finding out the center of the blueprint
        for angle in theta:
            x = int(np.round(r * np.cos(angle)))
            y = int(np.round(r * np.sin(angle)))
            bprint[m + x, n + y] = 1
        constant = np.argwhere(bprint).shape[0]
        for x, y in edges:  # For each edge coordinates
            # Centering the blueprint circle over the edges
            # and updating the accumulator array
            X = [x - m + R_max, x + m + R_max]  # Computing the extreme X values
            Y = [y - n + R_max, y + n + R_max]  # Computing the extreme Y values
            A[r, X[0]:X[1], Y[0]:Y[1]] += bprint
        A[r][A[r] < threshold * constant / r] = 0

    for r, x, y in np.argwhere(A):
        temp = A[r - region:r + region, x - region:x + region, y - region:y + region]
        try:
            p, a, b = np.unravel_index(np.argmax(temp), temp.shape)
        except:
            continue
        B[r + (p - region), x + (a - region), y + (b - region)] = 1

    return B[:, R_max:-R_max, R_max:-R_max]

@st.cache_data


def displayCircles(A, img):
    """

    :param A:
    :param img:
    :return:
    """
    circleCoordinates = np.argwhere(A)  # Extracting the circle information
    for r, x, y in circleCoordinates:
        cv2.circle(img, (y, x), r, color=(0, 255, 0), thickness=2)
    cv2.imwrite("outputcir.jpg",img)  
    st.image(img, caption="Hough Circle Image",use_column_width=True)
    return img
    
@st.cache_data


def hough_circles(source: np.ndarray,size,sigma, min_radius: int = 10, max_radius: int = 100) -> np.ndarray:
    """

    :param source:
    :param min_radius:
    :param max_radius:
    :return:
    """
    print("jjjdkldd",source)
    src = np.copy(source)
    circles = detectCircles(src, size, sigma, threshold=16, region=15, radius=[max_radius, min_radius])
    return displayCircles(circles, src)

# img_bgr = cv2.imread("images/CirclesInput.jpg")
# img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
# hough_circles(img_rgb)








# _________________________________________Hough-Elipse________________________________________________#
@st.cache_data

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges



@st.cache_data



def hough_ellipse(edges, threshold, min_minor_axis, max_minor_axis):
    height, width = edges.shape
    acc = np.zeros((height, width, 180), dtype=np.int)

    for y in range(height):
        for x in range(width):
            if edges[y, x] > 0:
                for a in range(min_minor_axis, max_minor_axis + 1):
                    for angle in range(180):
                        rad = np.radians(angle)
                        b = int(round(a * np.sin(rad)))
                        yc = y + b
                        xc = x + a
                        if 0 <= yc < height and 0 <= xc < width:
                            acc[yc, xc, angle] += 1

    candidates = np.where(acc >= threshold)
    return candidates


@st.cache_data

def draw_ellipses(image, candidates, min_minor_axis, max_minor_axis):
    for y, x, angle in zip(*candidates):
        color = (0, 255, 0)
        thickness = 2
        cv2.ellipse(image, (x, y), (min_minor_axis, max_minor_axis), angle, 0, 360, color, thickness)
    return image


