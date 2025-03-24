import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import cv2
import matplotlib.pyplot as plt
import random

# adding Gaussian noise, Salt & Pepper noise, and Uniform noise
def add_gaussian_noise(image, mean=0, sigma=25):
        gauss = np.random.normal(mean, sigma, image.shape)
        noisy = image + gauss
        return np.clip(noisy, 0, 255).astype(np.uint8)

def add_salt_and_pepper_noise(image, salt_prob=0.05, pepper_prob=0.05):
        noisy = image.copy()
        total_pixels = image.size
        num_salt = int(total_pixels * salt_prob)
        salt_coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
        noisy[salt_coords[0], salt_coords[1]] = 255

        num_pepper = int(total_pixels * pepper_prob)
        pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
        noisy[pepper_coords[0], pepper_coords[1]] = 0

        return noisy

def add_uniform_noise(image, min_val=0, max_val=255, intensity=50):
        noise = np.random.randint(min_val, max_val, image.shape, dtype=np.uint8)
        noisy_image = cv2.add(image, noise)
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

# histogram equalization
def histogram_equalization(image):
        equalized_image = cv2.equalizeHist(image)
        return equalized_image
    
# median filter
def apply_median_filter(image, kernel_size=3):
        filtered_image = cv2.medianBlur(image, kernel_size)
        return filtered_image

    # average filter
def apply_average_filter(image, kernel_size=3):
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        filtered_image = cv2.filter2D(image, -1, kernel)
        return filtered_image

    # Gaussian filter
def apply_gaussian_filter(image, kernel_size=3, sigma=0):
        filtered_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        return filtered_image    

# image normalization
def image_normalization(image):
        normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        return normalized_image
# displaying histogram
def display_histogram(image):
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        plt.figure(figsize=(6, 4))
        plt.title("Histogram")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.plot(hist)
        st.pyplot()    