import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('./frames/Angle1_frame_1.jpg',0)
edges = cv2.Canny(img,100,200)

plt.imshow(img, cmap = 'gray')
plt.imshow(edges, cmap = 'viridis', alpha=0.5)
plt.show()

#%%
#img1 = img.copy()
#edges = cv2.Canny(img1,50,150,apertureSize = 3)
#
#lines = cv2.HoughLines(edges,1,np.pi/180,190)
#
#for i, _ in enumerate(lines):
#    for rho,theta in lines[i]:
#        a = np.cos(theta)
#        b = np.sin(theta)
#        x0 = a*rho
#        y0 = b*rho
#        x1 = int(x0 + 1000*(-b))
#        y1 = int(y0 + 1000*(a))
#        x2 = int(x0 - 1000*(-b))
#        y2 = int(y0 - 1000*(a))
#
#        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)
#        print(rho, theta)
#
#cv2.imwrite('houghlines3.jpg',img1)

#%%

img2 = img.copy()
edges = cv2.Canny(img2, 50, 150, apertureSize = 3)
minLineLength = 50
maxLineGap = 12
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength, maxLineGap)

for i in range(lines.shape[0]):
    for x1,y1,x2,y2 in lines[i]:
        cv2.line(img2,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imwrite('proba_houghlines_detection.jpg',img2)
