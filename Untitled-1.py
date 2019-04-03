#%%
import cv2 
import numpy as np
from matplotlib import pyplot as plt
import os



#%%
# saving right camera pictures for camera calibration 
cap = cv2.VideoCapture('C:/Users/cleme/Documents/Projet/Untitled Folder/Video/Angle1.mp4')
count = 0
test = True
while test:
    ret, frame = cap.read()
    print(ret)
    name = '/Users/cleme/Documents/Projet/Untitled Folder/right_data/frame{}.jpg'.format(str(count))
    print('creating ...' + name)
    cv2.imwrite(name, frame)
    count +=1
    if count == 15:
        test = False
cap.release()

#%%
# saving left camera pictures for camera calibration
cap = cv2.VideoCapture('C:/Users/cleme/Documents/Projet/Untitled Folder/Video/Angle2.mp4')
count = 0
test = True
while test:
    ret, frame = cap.read()
    print(ret)
    name = '/Users/cleme/Documents/Projet/Untitled Folder/left_data/frame{}.jpg'.format(str(count))
    print('creating ...' + name)
    cv2.imwrite(name, frame)
    count +=1
    if count == 15:
        test = False
cap.release()


#%%
# image printing tests
folder_right = '/Users/cleme/Documents/Projet/Untitled Folder/right_data/'
folder_left = '/Users/cleme/Documents/Projet/Untitled Folder/left_data/'
sample_filename_right = folder_right + np.random.choice(os.listdir(folder_right))
sample_filename_left = folder_left + np.random.choice(os.listdir(folder_left))
img_right = cv2.imread(sample_filename_right)
img_left = cv2.imread(sample_filename_left)
gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
plt.subplot(221)
plt.imshow(img_right)
plt.subplot(222)
plt.imshow(img_left)
plt.subplot(223)
plt.imshow(gray_right)
plt.subplot(224)
plt.imshow(gray_left)

#%%


def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	
	global refPt, cropping
 
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True
		#crd.append(refPt[0])
		
		
		
 
	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		cropping = False
 
		# draw a rectangle around the region of interest
		cv2.rectangle(im, refPt[0], refPt[1], (0, 0, 255), 2)
		cv2.imshow("image", im)
		



#%%
#  
count = 0
right_calibration_folder = '/Users/cleme/Documents/Projet/Untitled Folder/right_calibration/'
right_data_folder = '/Users/cleme/Documents/Projet/Untitled Folder/right_data/'
folder = os.listdir(right_data_folder)
print(folder)
#%%

coord_right = []
for image in folder:
	
	img = cv2.imread(right_data_folder + image)
	im = img.copy()
	test = True
	
	while True:
		# display the image and wait for a keypress
		cv2.setMouseCallback("image", click_and_crop)
		cv2.imshow("image", im)
		key = cv2.waitKey(1) & 0xFF
		cv2.imwrite(right_calibration_folder + 'right_calib_frame{}.jpg'.format(str(count)), im)
 
	# if the 'c' key is pressed, break from the loop and go to next frame
		if key == ord("c"):
			
			count +=1
			coord_right.append(crd)
			print(coord_right)
			break
	# if there are two reference points, then crop the region of interest
	# from teh image and display it
	if len(refPt) == 2:
		#roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
		#cv2.imshow("ROI", roi)
		cv2.waitKey(0)

	cv2.destroyAllWindows()




#%%

#%%
# calibration de la camera de gauche
count = 0
left_calibration_folder = '/Users/cleme/Documents/Projet/Untitled Folder/left_calibration/'
left_data_folder = '/Users/cleme/Documents/Projet/Untitled Folder/left_data/'
folder = os.listdir(left_data_folder)
print(folder)
#%%

coord_left = []
for image in folder:
	print (refPt)
	img = cv2.imread(left_data_folder + image)
	im = img.copy()
	test = True
	
	while True:
		# display the image and wait for a keypress
		cv2.setMouseCallback("image", click_and_crop)
		cv2.imshow("image", im)
		key = cv2.waitKey(1) & 0xFF
		cv2.imwrite(left_calibration_folder + 'left_calib_frame{}.jpg'.format(str(count)), im)
 
	# if the 'c' key is pressed, break from the loop and go to next frame
		if key == ord("c"):
			
			count +=1
			coord_left.append(crd)
			print(coord_left)
			break
	# if there are two reference points, then crop the region of interest
	# from teh image and display it
	if len(refPt) == 2:
		#roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
		#cv2.imshow("ROI", roi)
		cv2.waitKey(0)

	cv2.destroyAllWindows()

#%%


image = cv2.imread(right_calibration_folder + 'right_calib_frame0.jpg')
img_conv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
plt.imshow(img_conv)


for i in img_conv:
	for pixel in i:
		if (pixel [0]==255):
			pixel [0]=0
			pixel [1]=0
			pixel [2]=255

plt.imshow(img_conv)
			

			




#%%
cameraMatrix = np.zeros((3,3), dtype = np.float32)
cameraMatrix[0,1] = 1
b = cameraMatrix[:2]
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
print(objp)




