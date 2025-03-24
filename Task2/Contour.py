import streamlit as st
import cv2
import numpy as np
from inspect import getclosurevars
from skimage import io
import matplotlib.pyplot as plt
import copy

k = 40


neighbors = np.array([[i, j] for i in range(-1, 2) for j in range(-1, 2)])


# @st.cache_data
# @st.cache


def internalEnergy(snake,alpha,beta):
    iEnergy=0
    snakeLength=len(snake)
    for index in range(snakeLength-1,-1,-1):  #??
        nextPoint = (index+1)%snakeLength
        currentPoint = index % snakeLength
        previousePoint = (index - 1) % snakeLength
        iEnergy = iEnergy+ (alpha *(np.linalg.norm(snake[nextPoint] - snake[currentPoint] )**2))\
                  + (beta * (np.linalg.norm(snake[nextPoint] - 2 * snake[currentPoint] + snake[previousePoint])**2))
    return iEnergy


# @st.cache_data

def interval_mapping(image, from_min, from_max, to_min, to_max):
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    return to_min + (scaled * to_range)


# @st.cache_data

def basicImageGradiant(image):
    s_mask = 17
    sobelx = np.abs(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=s_mask))
    sobelx = interval_mapping(sobelx, np.min(sobelx), np.max(sobelx), 0, 255)
    sobely = np.abs(cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=s_mask))
    sobely = interval_mapping(sobely, np.min(sobely), np.max(sobely), 0, 255)
    gradient = 0.5 * sobelx + 0.5 * sobely
    print(sobelx)
    print(sobely)
    return gradient


# @st.cache_data

def imageGradient(gradient, snak):
    sum = 0
    snaxels_Len= len(snak)
    for index in range(snaxels_Len-1):
        point = snak[index]
        sum = sum+((gradient[point[1]][point[0]]))
    return sum


# @st.cache_data

def externalEnergy(grediant,image,snak):
    sum = 0
    snaxels_Len = len(snak)
    for index in range(snaxels_Len - 1):
        point = snak[index]
        sum = +(image[point[1]][point[0]])
    pixel = 255 * sum
    eEnergy = k * (pixel - imageGradient(grediant, snak)) 
    return eEnergy


# @st.cache_data

def totalEnergy(grediant, image, snake,alpha,beta,gamma):
    iEnergy = internalEnergy(snake,alpha,beta)
    eEnergy=externalEnergy(grediant, image, snake)
    tEnergy = iEnergy+(gamma * eEnergy)
    return tEnergy


# @st.cache_data

def _pointsOnCircle(center, radius, num_points=50):
    points = np.zeros((num_points, 2), dtype=np.int32)
    for i in range(num_points):
        theta = float(i)/num_points * (2 * np.pi)
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        p = [x, y]
        points[i] = p
        
    return points


# @st.cache_data

def isPointInsideImage(image, point):
    
    return np.all(point < np.shape(image)) and np.all(point > 0)


# @st.cache_data

def activeContour(image_file, center, radius,alpha,beta,gamma,num_points):
    
    image = cv2.imread(image_file,0)
    # print(image.shape)
    snake = _pointsOnCircle(center, radius, num_points)
    grediant = basicImageGradiant(image)

    snakeColon =  copy.deepcopy(snake)

    for i in range(200):
        for index,point in enumerate(snake):
            min_energy2 = float("inf")
            for cindex,movement in enumerate(neighbors):
                next_node = (point + movement)
                if not isPointInsideImage(image, next_node):
                    continue
                if not isPointInsideImage(image, point):
                    continue

                snakeColon[index]=next_node

                totalEnergyNext = totalEnergy(grediant, image, snakeColon,alpha,beta,gamma)

                if (totalEnergyNext < min_energy2):
                    min_energy2 = copy.deepcopy(totalEnergyNext)
                    indexOFlessEnergy = copy.deepcopy(cindex)
            snake[index] = (snake[index]+neighbors[indexOFlessEnergy])
        snakeColon = copy.deepcopy(snake)


    
    return image, snake


# @st.cache_data

def get_chain_code(x, y):
    chain_code = []
    dx = np.array([0, 1, 1, 1, 0, -1, -1, -1])
    dy = np.array([-1, -1, 0, 1, 1, 1, 0, -1])

    for i in range(len(x) - 1):
        direction = np.where((dx == x[i + 1] - x[i]) & (dy == y[i + 1] - y[i]))[0][0]
        chain_code.append(direction)

    return chain_code


# @st.cache_data

def get_perimeter(chain_code):
    dx = np.array([0, 1, 1, 1, 0, -1, -1, -1])
    dy = np.array([-1, -1, 0, 1, 1, 1, 0, -1])
    distances = np.sqrt(dx**2 + dy**2)
    return sum([distances[code] for code in chain_code])


# @st.cache_data

def get_area(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


