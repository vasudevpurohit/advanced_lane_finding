## FINDING LANE LINES ON THE ROAD

The aim of this project is to find lane lines on the road using a more robust approach than Canny edge detection and Hough transforms.

### Following are the files & folders contained in this repository:
---------------------------------------------------------------

1. The main code that runs on a video is contained in 'advanced_lane_finding.py'
2. 'advanced_lane_finding_image.py' runs on a single image.
3. All the test images on which the code has been run are contained in 'test_images'
4. The output images obtained are stored in 'output_images'
5. The video pipeline is implemented on 'project_video.mp4'
5. 'Project Writeup.docx' contains the pipeline for the code, shortcomings, and the ways to improve the pipeline.


### Running the code:
-------------------

To run the code on different videos, you would have to change the name of the input video on which the code needs
to be run, and the name of the output video to which the appended video will be stored.

Hence, you might want to change the following lines of code accordingly,

1. To run the code on different input videos, ref line 184 of 'advanced_lane_finding.py':  
		cap = cv2.VideoCapture('<filename>')

2. To store the corresponding appended videos, ref line 187 of 'advanced_lane_finding.py':  
		out = cv2.VideoWriter('<filename>', fourcc, 25, (1280,  720))

3. To run the image pipeline on different images, ref line 164 of 'advanced_lane_finding_images.py':
		test = plt.imread('test_images/<filename>')

4. To save the output change, ref line 172 of 'advanced_lane_finding_images.py':
		test = plt.imread('output_images/<filename>')