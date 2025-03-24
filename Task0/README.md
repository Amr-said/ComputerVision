# Computer Vision Course - Project #1

---
# Table of Content

| Requiered Part | Title |
| ----------- | ----------- |
| [#Part 0](#part-0) | Graphical User Interface - GUI |
| [#Part 1](#part-1) | Add additive noise to the image |
| [#Part 2](#part-2) | Filter the noisy image |
| [#Part 3](#part-3) | Detect edges in the image |
| [#Part 4](#part-4) | Draw histogram and distribution curve |
| [#Part 5](#part-5) | Equalize the image |
| [#Part 6](#part-6) | Normalize the image |
| [#Part 7](#part-7) | Local and global thresholding |
| [#Part 8](#part-8) | Transformation from color image to gray scale |
| [#Part 9](#part-9) | Frequency domain filters |
| [#Part 10](#part-10) | Hybrid images |

---

# Part 0

## Graphical User Interface - GUI

We display all the requirments in one window
using the GUI you can change some parameters like:

    1. type of the filter
    2. noise value
    3. threshold 

![GUI](ScreenShots/gui1.JPG)
![GUI](ScreenShots/gui2.JPG)
![GUI](ScreenShots/gui3.JPG)

---

# Part 1

## Add additive noise to the image

We add three different type of noise :

* Uniform
* Guassian
* Salt & Pepper

You can change between them using combo box and change the amount of noise using slider

### Uniform

![Uniform Noise](ScreenShots/Uniform.png)

### Guassian

![Gaussian Noise](ScreenShots/Gaussian_Noise.png)

### Salt & Pepper

![Salt & Pepper Noise](ScreenShots/Salt&pepper.png)
---

# Part 2

## Filter the noisy image using the following low pass filters

in this section we use function "avgFilter" to manipulate the noisy image with a specific filter we choose and the result is ...

## - original image

![original of Image](ScreenShots/Original.png)

## - Average filter

![Average filter](ScreenShots/Average_filters.png)

## - Gaussian filter

![Gaussian filter](ScreenShots/Gaussian_filter.png)

## - median filters

![median filters](ScreenShots/Median_filter.png)

---

# Part 3

## Detect edges in the image using the following masks

## - Sobel , Roberts , Prewitt and Canny edge detectors

in this section we use function "edgFilters" to manipulate the image with a specific filter we choose and the result is ...

## - original pic

![original pic](Screenshots/Original.png)

## - Sobel filter

![edge detection Sobel](ScreenShots/SobelEdgeDetection.png)

## - Roberts pic

![edge detection Roberts](ScreenShots/RobertsEdgeDetection.png)

## - Prewitt filters

![edge detection Prewitt](ScreenShots/PrewittEdgeDetection.png)

---

# Part 4

## Draw histogram and distribution curve

In this part we've implemented a function called "df" that takes an image data array and return the histogram values for each intensity value.

Using that values to draw a "BarGraphItem" on pyqtgraph we got the following output...

![Histogram of Image](ScreenShots/Histogram.png)

---

# Part 5

## Equalize the image

Using the previous histogram to generate a histogram equalization function by looping over the whole image array and equalize the output of the process we got the following image...

![Equalization of Image](ScreenShots/Equalized.png)

You can see the difference in histograms, now it's values distributed over larger range of data.

---

# Part 6

## Normalize the image

Normalization process doesn't depend on histogram, as we know.

By calculating the mean and standard deviation for image data array and using the following equation:

`New Value = (Original Value - mean) / std^2`

We got the following image...

![Normalization of Image](ScreenShots/Normalized.png)

---

# Part 7

## Local and global thresholding

Global Thresholding: We implemented the global function where it iterates over all the image's pixels and assign it to a new value where it becomes 255 or 0 according to whether or not it's greater than a given threshold provided by the user.
![Global_thresh_img](Screenshots/Global Thresholding.png)
Local Thresholding: We implemented the local function to work as it divides the image to many smaller windows where their size is selected by the user, the mean is calculated to the selected window and then is used as the threshold value to this window by applying the same technique used in the global function, the user have the option to choose if a constant is needed to be subtracted from the mean before applying it to the window.
![Local_thresh_img](Screenshots/Local Thresholding.png)

---

# Part 8

## Transformation from color image to gray scale image and plot of R, G, and B histograms with its distribution function (cumulative curve that you use it for mapping and histogram equalization)

The  transformation of a colored image to a gray scale image was done by selecting each corresponding pixel in the 3 channels(ie. RGB) and multiplying these value by certain constant values respectively and add them together. the result is then equal to the pixel value needed to acheive the suitabale grey color for that pixel.
![Gray_scale_img](Screenshots/RGB Channels.png)
Where its histogram is displayed as follows
<!-- ![Gray_histogram](Screenshots/Histogram_gray.png) -->

---

# Part 9

## Frequency domain filters (high pass and low pass)

in this section we use function "freqFilters" to manipulate the image with a LPS or HPS in the frequency domain and get the pic back and the result is ...

## - original pic

![original pic](ScreenShots/Original.png)

## - low pass filter

![Low pass filter](ScreenShots/LowPass.png)


## - High Pass filter
![High pass filter](ScreenShots/HighPass.png)

---

# Part 10

## Hybrid images

After implementing the high pass and the low pass filters required in part 9, hybrid images could be acheiving by simply applying the high filter on an image and the low pass on another and adding the results of each together to reach the merging between the two images needed.

![Hybrid_imgs](ScreenShots/gui3.JPG)

---
