# Add all necessary imports here
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Implement a function that reads an image in grayscale given the image path


def read_img(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img

# Implement a function that calculates mean of grayscale image


def mean(image):
    mean_value = np.mean(image)
    return mean_value

# Implement a function that returns a binary image that has the same size as the input grayscale image
# the output image should have ones at pixels with intensity values greater than or equal to a given threshold
# and zeros at pixels with intensity values less than that given  threshold


def segment(image, threshold):
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary

# Create a binary mask for any image with threshold equal to the mean value of grayscale intensity values.
# Then apply the mask to the image.
# Plot the image, mask, and masked region of the image.


# Read the image
img = read_img('GettyImages-126375144-920x745.jpg')

# Calculate the mean value of the image
img_mean = mean(img)

# Create the binary mask using the mean value
mask = segment(img, img_mean)

# Apply the mask to the original image
masked_img = cv2.bitwise_and(img, img, mask=mask)

# Plot the images
plt.figure(figsize=(12, 12))
plt.subplot(131)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.subplot(132)
plt.imshow(mask, cmap='gray')
plt.title('Binary Mask')
plt.subplot(133)
plt.imshow(masked_img, cmap='gray')
plt.title('Masked Image')
plt.show()
