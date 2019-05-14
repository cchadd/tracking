"""
    Read a video and apply a mask on it
"""

#%% Mask
import numpy as np
import cv2
import matplotlib.pyplot as plt
#%%
absolute_path = "/home/tdesfont/0-documents/shared_soccer/video/"
file_name = "Angle2.mp4"

#%% Import mask
mask = np.load("mask.npy")
plt.figure()
plt.imshow(mask)
plt.show()

#%% Read a video

processed = 1
coloured = 1

cap = cv2.VideoCapture(absolute_path+file_name)

dt = 20
image_count = 20

count = 0
while(cap.isOpened()):
    count += 1

    ret, frame = cap.read()
    if coloured:
        gray = frame
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if processed:
        gray = cv2.bitwise_and(gray, gray, mask = mask.astype('int8'))

    if count % dt == 0:
        cv2.imshow('frame',gray)
        if image_count > 0:
            cv2.imwrite('./roi_frames/frame_{}.jpg'.format(count//dt), gray)
            image_count -= 1
        else:
            break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

