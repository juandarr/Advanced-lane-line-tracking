# **Advanced Lane Finding using OpenCV**

The goals/steps of this project are the following:

1. [x] Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
2. [x] Apply a distortion correction to raw images.
3. [x] Use color transforms, gradients, etc., to create a thresholded binary image.
4. [x] Apply a perspective transform to rectify binary image ("birds-eye view").
5. [x] Detect lane pixels and fit to find the lane boundary.
6. [x] Determine the curvature of the lane and vehicle position with respect to the center.
7. [x] Warp the detected lane boundaries back onto the original image.
8. [x] Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort_output.png "Image 1. Undistorted"
[image2]: ./output_images/binary_combined_thresholds.png "Image 2. Binary image combined thresholds"
[image3]: ./output_images/perspective_transform.png "Image 3. Perspective transform in straight lanes road"
[image4]: ./output_images/lane_line_detection.png "Image 4. Left and right lane line detection"
[image5]: ./output_images/curvature_position.png "Image 5. Curvature and vehicle position respect to the middle of the lane"
[image6]: ./output_images/estimations_warped.png "Image 6. Warped back lane region in the undistorted image in addition to numerical estimations"
[video1]: ./estimation_output_project_video.mp4 "Output Video: simple pipeline version"
[video2]: ./improved_estimation_output_project_video.mp4 "Output Video: improved pipeline version"

## Steps description

### Here I will consider the steps individually and describe how I addressed each step in my implementation.  

---
### 1. Camera Calibration

The implementation of the camera calibration is shown in the second block of code of the jupyter notebook 'Advanced Lane Finding.ipynb'. 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection after using `cv2.findChessboardCorners()`. 

The lists of points `objpoints` and `imgpoints`  are used to compute the camera calibration matrix and distortion coefficients, which are the outputs of the `cv2.calibrateCamera()` function.  

## **Pipeline (single images)**

Steps 2 through 8 are the main ones used in the pipeline function. 

### 2. Distortion correction of raw images
As an example of the camera calibration step (see the respective section in the jupyter notebook), I applied this distortion correction to the `calibration5.jpg` image in the folder `camera_cal`. After applying the function `cv2.undistort()` I obtained the following result: 

![alt text][image1]


### 3. Color and gradient thresholds

This step is implemented in the jupyter notebook. The method considers thresholds over the gradient in the x-direction of the lightness channel, which is especially useful to identify vertical or near vertical transitions in color a color channel (in this case the L channel of the HLS -hue, lightness, saturation- color space) and thresholds over the HSL color space to filter white and yellow lane lines from the image (Initially tried just with the saturation channel from the HSL color space but didn't get the best results). The gradients in the x-direction are achieved using `cv2.Sobel()` function over the (1,0) axis, which is the x-axis in this case. After conditioning the values (taking the absolute value and scaling) a binary result is defined for those values that are within the specified gradient thresholds. A similar approach is carried out over the white and yellow regions in HSL color space image, where the thresholds are used to define a binary image (using two filters, one for white and another for yellow) with values inside the minimum and maximum threshold values for each channel of the HSL color space. Finally, both binary outputs are combined. Here are the results after applying and combining the color and thresholds the image `test5.jpg` from the folder `test_images`:

![alt text][image2]

### 4. Perspective transform

The perspective transform step (implemented in the jupyter notebook) uses the function `cv2.getPerspectiveTransform()` to obtain a transform matrix (M) to map source points to destination points. We can also obtain an inverse transform matrix (Minv) to map destination points to source points changing the source to destination and destination to source in the perspective transform function. The election of the source and destination points are key in this step. The defined values are the following:

```python
    offset_left = 245 # offset in x direction at the left side of the warped image
    offset_right = 285 # offset in the x directions at the right side of the warped image

    #b. Define four source points src=np.float32([[,],[,],[,],[,]])
    src = np.float32([[206,img_size[1]], 
                     [582,460], 
                     [701,460], 
                     [1100,img_size[1]]])

    #c. Define four destination points dst=np.float32([[,],[,],[,],[,]])
    dst = np.float32([[offset_left,img_size[1]],
                     [offset_left,0],
                     [img_size[0]-offset_right,0],
                     [img_size[0]-offset_right,img_size[1]]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 206, 720      | 245, 720        | 
| 582, 460      | 245, 0      |
| 701, 460      | 995, 0      |
| 1100, 720     | 995, 720        |

Once a correct perspective transform matrix is obtained, we expect an image transformation where the lane lines appear parallel and well defined. The following image shows the perspective transform working as expected:

![alt text][image3]

### 5. Detect lane pixels and find the lane lines

In this step, we use histograms and windows to identify the lane lines. Histograms are useful to detect the initial position of the most active pixels in the lower half of the image.
Since the image has been warped (for parallel lines to appear parallel in the warped space), we can expect vertical regions where the lane is present to present the biggest group of
active pixels. Once we have detected this initial position, we can iterate with windows in the direction bottom-up with the goal of detecting each lane line inside the window boundaries. A set of hyperparameters can be defined to tune the search, such as the number of windows in the y-direction and the number of active pixels required to recenter a window for the next iteration. The following image shows the result of this process:

![alt text][image4]

### 6. Determine the lane curvature and vehicle position respect to the center of the lane

This section implements the radius of curvature formula using a transformation of length values
from the pixel space to the real world space. In reality, there are two different pixel worlds to consider and we need to map them to the real world. The first one is the one mapped by the camera which is the one reflected in the undistorted images as shown on the left side of image 3. The second one is the pixel world from the warped space, here lengths in the x and y directions have a different mapping to real-world values. An image in this pixel world is present on the right side of image 3. 

After fitting the pixel values detected in each line to a second order polynomial, we use its coefficients to obtain the radius of curvature. Given that in general the left and right line lanes are parallel in the real world, we can expect that they will appear parallel and with a similar radius of curvature in the warped space. The vehicle position respect to the center of the lane is calculated defining the center of the image as the center of the vehicle and the center of the lane as the reference point. Thus, we need to obtain the x coordinate of the right and left line lanes, then calculate the middle point. 

All these ideas are presented in detail in the jupyter notebook in the respective section. The following image is the result of this process. LCR stands for 'Left lane radius of curvature', RCR for 'Right lane radius of curvature' and Position, to the vehicle position respect to the middle of the lane. Given that the middle of the lane is the reference point, when the vehicle is right to the middle point we will get positive values. When the vehicle is positioned to the left of the middle point of the lane we get negative values. 

![alt text][image5]

### 7. Warp the detected lane lines back onto the original image

This part of the pipeline process defines a region contained within the left and right lanes and warps it back onto the undistorted image using the inverse matrix obtained after applying the perspective transform from the destination to the source (from warped space -top view- to the undistorted image space -camera view-).

### 8. Output numerical estimation of curvature and vehicle position

In this step, we attach the text with the numerical estimations from step 7 at the top of the image. For the radius of curvature, we take the average value between the left lane line and right lane line radius of curvature. The next image shows the result of both steps. Notice the green region warped back onto the undistorted image and the numerical estimations at the top.

![alt text][image6]

---

## **Pipeline (video)**

### Sanity check, histogram and window lane line tracking only

The conversion time of the whole `video_project.mp4` is about 15 minutes. Given that I didn't use a look-ahead filter here, this time performance is expected.
Here's a [link to my video result](https://youtu.be/kMyI6nWGlGE)

---

## **Pipeline (video) improved version with Line class**

### Line tracking, sanity check, look-ahead filter, reset and smoothing  (single images)

In this case, I implemented a Line class and created a couple of instances to represent the left and right lane lines. The look-ahead filter greatly improved the time performance of the whole pipeline with processing times per frame going from 1 second to about 0.1 seconds. The conversion time of the whole `video_project.mp4` is about 3.5 minutes. Smoothing is also used with a buffer_size of 5 samples. 

Here is a [link to my video result](https://youtu.be/Vc3exxHYV_M)

---

## **Discussion**

#### 1. Briefly discuss any problems/issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One of the main challenges of the project is to apply a filter able to separate shadows and dark areas of the road over light material from the lane lines. The defined thresholds for the gradient in the x-direction (using ``cv2.Sobel` over x) and color filtering have a huge impact in the performance of the lane finding pipeline and will not work in all scenarios, as was shown in the challenge videos, where constant irregularities in the lane asphalt, shadows, and strong curves will make the pipeline fail. The color filtering was fine-tuned to detect yellow and white lines. If relevant features of lane lines are present in a different color channel/space invisible to this implementation we will have to redefine it. 

Another challenge in the project was the time it takes for the pipeline to complete one pass. If we use histogram and window iterations only, it takes about 1 second to complete one frame. I implemented the tip that recommends to keep track of the lanes and identify subsequent lanes not based in histograms and windows iterations (which is computationally expensive) but using the previously detected lane as the reference to define a margin inside which we should find the next lane line. This operation takes around 0.1 seconds, a huge improvement from the 1 second that takes the initial implementation. This approach, however, is only useful once a lane line has been identified. 

For live lane line tracking, the results in this project are not enough. At the minimum, the time that takes to process one frame is about 0.1 seconds. In real life scenarios, this time can be more than enough for accidents to occur and depending on the context they could be life-threatening. The speed of conversion is, therefore, a relevant requirement in certain applications and we can improve with more computational power and more time/space efficient conversion algorithms.




