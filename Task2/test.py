import streamlit as st
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import cv2
import matplotlib.pyplot as plt
import random
import Code_functions

def main():
    grayscale_image = None  # grayscale_image at a higher scope
    noisy_grayscale_image = None  # noisy grayscale image

    # Streamlit app title
    st.title("Image Preprocessing App")

    # Create a sidebar for options
    st.sidebar.header("Options")

    # Upload image
    uploaded_image = st.sidebar.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        # Read the uploaded image
        image = Image.open(uploaded_image)
        image = np.array(image)

        # original image
        st.sidebar.image(image, caption="Original Image", use_column_width=True)

        # Check if the image is in color (3 channels) or grayscale (1 channel)
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert RGB image to grayscale
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif len(image.shape) == 2:
            
            grayscale_image = image
        else:
            # Handle other cases or show an error message
            st.warning("Unsupported image format. Please provide an RGB or grayscale image.")
            grayscale_image = None

        # Display grayscale image if it's not None
        if grayscale_image is not None:
            st.image(grayscale_image, caption="Grayscale Image", use_column_width=True)
            
            
        # Standardize image
        standardize_option = st.sidebar.checkbox("Standardize Image")
        if standardize_option:
            if grayscale_image is not None:
                standardized_image = cv2.equalizeHist(grayscale_image)
                st.image(standardized_image, caption="Standardized Grayscale Image", use_column_width=True)
            else:
                st.warning("Please select an image and convert it to grayscale to standardize.")

        # Data augmentation
        augmentation_option = st.sidebar.checkbox("Apply Data Augmentation")
        if augmentation_option:
            if grayscale_image is not None:
                augmented_images = []
                augmented_images.append(grayscale_image)  # Original image

                # Rotate by 45 degrees
                height, width = grayscale_image.shape
                rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 45, 1)
                rotated_image = cv2.warpAffine(grayscale_image, rotation_matrix, (width, height))
                augmented_images.append(rotated_image)

                # Mirror horizontally
                horizontally_mirrored = cv2.flip(grayscale_image, 1)
                augmented_images.append(horizontally_mirrored)

                # Flip vertically
                vertically_flipped = cv2.flip(grayscale_image, 0)
                augmented_images.append(vertically_flipped)

                # Randomly adjust brightness
                brightness_factor = st.sidebar.slider("Brightness Factor", 0.5, 10.0, 1.0)
                brightness_adjusted = cv2.convertScaleAbs(grayscale_image, alpha=brightness_factor, beta=0)
                augmented_images.append(brightness_adjusted)

                st.image(augmented_images, caption=["Original", "Rotated", "Horizontally Mirrored", "Vertically Flipped", "Brightness Adjusted"], use_column_width=True)
            else:
                st.warning("Data augmentation is currently supported for grayscale images only.")

    # section for noise addition and image filters
    st.sidebar.header("Image Filters & Noise")

    # Checkbox to add noise
    noise_option = st.sidebar.checkbox("Add Noise")
    if noise_option:
        noise_type = st.sidebar.radio("Select Noise Type", ("Gaussian Noise", "Salt & Pepper Noise", "Uniform Noise"))

        apply_to_grayscale = st.sidebar.checkbox("Apply to Grayscale Image")

        if not apply_to_grayscale:
            st.warning("Please select an image type (Grayscale) to apply noise.")

        if apply_to_grayscale and grayscale_image is not None:
            noisy_grayscale_image = None
            if noise_type == "Gaussian Noise":
                mean = st.sidebar.slider("Mean", 0, 255, 0)
                sigma = st.sidebar.slider("Sigma", 1, 100, 25)
                noisy_grayscale_image = add_gaussian_noise(grayscale_image, mean, sigma)
            elif noise_type == "Salt & Pepper Noise":
                salt_prob = st.sidebar.slider("Salt Probability", 0.01, 0.5, 0.05)
                pepper_prob = st.sidebar.slider("Pepper Probability", 0.01, 0.5, 0.05)
                noisy_grayscale_image = add_salt_and_pepper_noise(grayscale_image, salt_prob, pepper_prob)
            elif noise_type == "Uniform Noise":
                min_val = st.sidebar.slider("Min Value", 0, 100, 0)
                max_val = st.sidebar.slider("Max Value", 100, 255, 255)
                intensity = st.sidebar.slider("Intensity", 1, 100, 50)
                noisy_grayscale_image = add_uniform_noise(grayscale_image, min_val, max_val, intensity)

            st.image(noisy_grayscale_image, caption=f"Noisy Grayscale Image ({noise_type})", use_column_width=True)

    # histogram and processing
    st.sidebar.header("Histogram Processing")

    # Checkbox to apply histogram equalization
    hist_eq_option = st.sidebar.checkbox("Apply Histogram Equalization")
    if hist_eq_option:
        if grayscale_image is not None:
            equalized_image = histogram_equalization(grayscale_image)
            st.image(equalized_image, caption="Histogram Equalized Grayscale Image", use_column_width=True)
        else:
            st.warning("Please select an image and convert it to grayscale to equalize histogram.")

    # section for image filters
    st.sidebar.header("Image Filters")

    # median filter
    median_filter_option = st.sidebar.checkbox("Apply Median Filter")
    if median_filter_option:
        if noisy_grayscale_image is not None:
            kernel_size = st.sidebar.slider("Select Kernel Size", 3, 15, 3, key="median_kernel_size")
            filtered_image = apply_median_filter(noisy_grayscale_image, kernel_size)
            st.image(filtered_image, caption="Median Filtered Grayscale Image", use_column_width=True)
        else:
            st.warning("Please select an image and convert it to grayscale to apply the median filter.")

    # apply average filter
    average_filter_option = st.sidebar.checkbox("Apply Average Filter")
    if average_filter_option:
        if noisy_grayscale_image is not None:
            kernel_size = st.sidebar.slider("Select Kernel Size", 3, 15, 3, key="average_kernel_size")
            filtered_image = apply_average_filter(noisy_grayscale_image, kernel_size)
            st.image(filtered_image, caption="Average Filtered Grayscale Image", use_column_width=True)
        else:
            st.warning("Please select an image and convert it to grayscale to apply the average filter.")

    # apply Gaussian filter
    gaussian_filter_option = st.sidebar.checkbox("Apply Gaussian Filter")
    if gaussian_filter_option:
        if noisy_grayscale_image is not None:
            kernel_size = st.sidebar.slider("Select Kernel Size", 3, 15, 3, key="gaussian_kernel_size")
            sigma = st.sidebar.slider("Select Sigma Value", 0, 5, 0, key="gaussian_sigma")
            filtered_image = apply_gaussian_filter(noisy_grayscale_image, kernel_size, sigma)
            st.image(filtered_image, caption="Gaussian Filtered Grayscale Image", use_column_width=True)
        else:
            st.warning("Please select an image and convert it to grayscale to apply the Gaussian filter.")

    # section for image normalization
    st.sidebar.header("Image Normalization")

    # apply image normalization
    normalization_option = st.sidebar.checkbox("Apply Image Normalization")
    if normalization_option:
        if grayscale_image is not None:
            normalized_image = image_normalization(grayscale_image)
            st.image(normalized_image, caption="Normalized Grayscale Image", use_column_width=True)
        else:
            st.warning("Please select an image and convert it to grayscale to apply image normalization.")

    # section for histogram visualization
    st.sidebar.header("Histogram Visualization")

    # show histogram
    hist_option = st.sidebar.checkbox("Show Histogram")
    if hist_option:
        if grayscale_image is not None:
            display_histogram(grayscale_image)
        else:
            st.warning("Please select an image and convert it to grayscale to show the histogram.")

    # convert image to binary format
    binary_option = st.sidebar.checkbox("Convert to Binary Image")
    if binary_option:
        if grayscale_image is not None:
            _, binary_image = cv2.threshold(grayscale_image, 128, 255, cv2.THRESH_BINARY)
            st.image(binary_image, caption="Binary Grayscale Image", use_column_width=True)

if __name__ == "__main__":
    main()
