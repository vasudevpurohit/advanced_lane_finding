import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob

img = glob.glob('camera_cal/calibration*')
nx = 9 #no. of corners along x
ny = 6 #no. of corners along y
obj_points = np.zeros((ny,nx,3),dtype='float32')                                
for i in range(ny):
    for j in range(nx):
        obj_points[i][j] = [j,i,0] #defining the object point array
obj_points = np.reshape(obj_points,(nx*ny,1,3)) #matching its shape to the i_array
o_array = [] #array to store all the object 3D points
i_array = [] #array to store all the image 2D points

for fname in img:                                                               
    image = plt.imread(fname)
    image_g = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    x_size = image_g.shape[1]
    y_size = image_g.shape[0]
    ret, corners = cv2.findChessboardCorners(image_g,(nx,ny),None)
    
    if ret == True:
        i_array.append(corners)
        o_array.append(obj_points)

ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(o_array,i_array,(y_size,x_size),None,None)


#undistorting a real world image
test = plt.imread('test_images/test6.jpg')
test_udst = cv2.undistort(test,cameraMatrix,distCoeffs,None,cameraMatrix) #undistorting the pulled test image

#thresholding 'test' image based on color and gradient
test_g = cv2.cvtColor(test_udst,cv2.COLOR_BGR2GRAY)  #grayscale test image

grad_x = cv2.Sobel(test_g,cv2.CV_64F,1,0)    #gradient computation
abs_gradx = np.abs(grad_x)  #absolute gradient values
scaled_absx = np.uint8(255*(abs_gradx/np.max(abs_gradx)))   #scaled gradients that lie between 0 & 255

min_g = 20  #minimum gradient threshold
max_g = 110 #maximum gradient threshold

bin_g = np.zeros((y_size,x_size),dtype='uint8') #empty array to store values for 'activated' pixels
bin_g[(scaled_absx >= min_g)&(scaled_absx <= max_g)] = 1

#thresholding based on saturation channel
test_c = cv2.cvtColor(test_udst,cv2.COLOR_BGR2HLS)
S = test_c[:,:,2]   #extracting the saturation channel

min_s = 155  #minimum saturation threshold
max_s = 255 #maximum saturation threshold
bin_s = np.zeros((y_size,x_size),dtype='uint8')
bin_s[(S > min_s) & (S <= max_s)] = 1

#thresholding based on color
min_c = 200  #minimum saturation threshold
max_c = 255 #maximum saturation threshold
bin_c = np.zeros((y_size,x_size),dtype='uint8')
bin_c[(test_g > min_c) & (test_g <= max_c)] = 1

bin_comb = np.zeros((y_size,x_size),dtype='uint8')  #initialising the combined array with zeros
bin_comb[(bin_g == 1)|(bin_s == 1)|(bin_c==1)] = 1


vertices = np.array([[(150,720),(590,450),(750,450),(1250,720)]])
mask = np.zeros((y_size,x_size),dtype='float32')
mask = cv2.fillPoly(mask,vertices,255)
mask = np.uint8(mask)
mask_im = cv2.bitwise_and(bin_comb,mask)

warped = np.array([[[200,720]],[[200,0]],[[1000,0]],[[800,720]]],np.float32)
vertices = np.reshape(vertices,(4,1,2))
vertices = np.float32(vertices)
M_persp = cv2.getPerspectiveTransform(vertices,warped)  #transform matrix to carry out the perspective transformation
M_persp_inv = np.linalg.inv(M_persp)                    #inverse transform matrix to be used later
trans_im = np.zeros((y_size,x_size),dtype='uint8')      #initialising an empty array to store the transformed img
trans_im = cv2.warpPerspective(mask_im,M_persp,(x_size,y_size),flags=cv2.INTER_LINEAR)


#finding polynomials for lane lines
sum = np.sum(trans_im,axis=0)   #finding peaks for the sliding window search to begin
mid = np.int(x_size/2)
left_start = np.argmax(sum[:mid])       #starting point - left lane
right_start = np.argmax(sum[mid:])+mid  #starting point - right lane
left_current = left_start
right_current = right_start

window_margin = 100
window_no = 10
window_height = np.int(y_size/window_no)
minpix = 50
pixels_captured = []
trans_im_copy_left = np.zeros((y_size,x_size),dtype='uint8')
trans_im_copy_right = np.zeros((y_size,x_size),dtype='uint8')
trans_3c = np.dstack((trans_im_copy_left,trans_im_copy_left,trans_im_copy_left))*255 #could have used the _right one as well
indices_l = []
indices_r = []
co_l = np.array([],dtype='float32')     #empty array to store all the x,y co-ordinates of the left lane
co_r = np.array([],dtype='float32')     #empty array to store all the x,y co-ordinates of the right lane
count = 0
#left lane
upper_bound = y_size
for i in range(window_no):
    indices = []
    boundx_1 = np.int(left_current - window_margin) #left bound for left lane
    boundx_2 = np.int(left_current + window_margin) #right bound for left lane
    lower_bound = upper_bound - window_height   #for the current iteration
    for j in range(boundx_1,boundx_2):
        for k in range(lower_bound,upper_bound):
            if trans_im[k][j] == 1:
                indices.append([k,j])
                trans_im_copy_left[k][j] = 1
    start_point = (boundx_1,upper_bound)    #defining the start point of the search window
    end_point = (boundx_2,lower_bound)      #defining the end point of the search window
    cv2.rectangle(trans_3c,start_point,end_point,(255,0,0),3)   #constructing a red rectangle over the image
    indices = np.array(indices,dtype='float32')     #converting the index list to an array
    co_l = np.append(co_l,indices)                  #storing these in a master array for co-ordinates, used later
    if indices.shape[0] >= minpix:
        left_current = np.mean(indices,axis=0)[1]   #changing the mid-point of the next rectangle based on the current search
    upper_bound = lower_bound   #for the next iteration
    
my = 30/720 #metres per pixel in y direction
mx = 3.7/700 #metres per pixel in x direction   
co_l = np.reshape(co_l,(np.int(co_l.shape[0]/2),2)) #reshaping the co-ordinate array for (x,y) format
poly_l = np.polyfit(co_l[:,0],co_l[:,1],2)
y_new=np.linspace(0,(y_size-1),y_size)
x_newl = np.polyval(poly_l,y_new)

trans_im_copy_left = np.dstack((trans_im_copy_left,trans_im_copy_left,trans_im_copy_left))*255

search_im_l = cv2.bitwise_or(trans_im_copy_left,trans_3c)

#right lane
upper_bound = y_size
for i in range(window_no):
    indices = []
    boundx_1 = np.int(right_current - window_margin) #left bound for left lane
    boundx_2 = np.int(right_current + window_margin) #right bound for left lane
    lower_bound = upper_bound - window_height   #for the current iteration
    for j in range(boundx_1,boundx_2):
        for k in range(lower_bound,upper_bound):
            if trans_im[k][j] == 1:
                indices.append([k,j])
                trans_im_copy_right[k][j] = 1
    start_point = (boundx_1,upper_bound)
    end_point = (boundx_2,lower_bound)
    cv2.rectangle(trans_3c,start_point,end_point,(255,0,0),3)
    indices = np.array(indices,dtype='float32')
    co_r = np.append(co_r,indices)
    if indices.shape[0] >= minpix:
        right_current = np.mean(indices,axis=0)[1]
    upper_bound = lower_bound   #for the next iteration
co_r = np.reshape(co_r,(np.int(co_r.shape[0]/2),2)) #reshaping the co-ordinate array for (x,y) format
poly_r = np.polyfit(co_r[:,0],co_r[:,1],2)
x_newr = np.polyval(poly_r,y_new)

trans_im_copy_right = np.dstack((trans_im_copy_right,trans_im_copy_right,trans_im_copy_right))*255

search_im_r = cv2.bitwise_or(trans_im_copy_right,trans_3c)
search_im = cv2.bitwise_or(search_im_l,search_im_r)

#visualising the results
# plt.imshow(search_im)
# plt.plot(x_newl,y_new,'-',color='green',markersize=0.5)
# plt.plot(x_newr,y_new,'-',color='blue',markersize=0.5)

#calculating the radius of curvature
# A_new_l = (mx/(np.power(my,2)))*poly_l[0] #scaled coeffecients
# A_new_r = (mx/(np.power(my,2)))*poly_r[0]
# B_new_l = (mx/my)*poly_l[1]
# B_new_r = (mx/my)*poly_r[1]
R_l = (np.power(1+np.power(((2*poly_l[0]*mx/(my*my))*(np.max(y_new)*my))+(poly_l[1]*mx/my),2),1.5))/np.absolute(2*poly_l[0]*mx/(my*my))
R_r = (np.power(1+np.power(((2*poly_r[0]*mx/(my*my))*(np.max(y_new)*my))+(poly_r[1]*mx/my),2),1.5))/np.absolute(2*poly_r[0]*mx/(my*my))

#transforming the perspective image to the actual image again
templ = np.transpose(np.array((x_newl,y_new)))
tempr = np.transpose(np.array((x_newr,y_new)))[::-1]
co = np.vstack((templ,tempr))   #storing all the coordinates in a single array
final_im = np.zeros((y_size,x_size,3),dtype='uint8')
final_im = cv2.fillPoly(final_im,np.int32([co]),(0,255,0))      #final image with the lane region colored green

final_im_tr = cv2.warpPerspective(final_im,M_persp_inv,(x_size,y_size),flags=cv2.INTER_LINEAR)

lane_im = cv2.addWeighted(final_im_tr,0.3,test_udst,1,0)

plt.imshow(lane_im)

#next step would be to modify the current program for a video and the searching would have to be changed accordingly
#the smoothing can also be carried out over n frames and then this lane output can be added to a running video to 
#visualise how this is working
#see how you can add the radius and offset values to the video and create a separate video for visualising through
# a bird's eye view