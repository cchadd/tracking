#%% Imports
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

#%%
class FramesGetter(object):
    '''
    Classto perform actions on frames from a video
    '''
    def __init__(self):
        pass

    def test_t(self):
        print("hello")

    def get_frames(path_to_video, path_to_store, name_to_frames, \
                   frame_delay=10, frame_count=5):
        '''
        Inputs:
        -------
        path_to_store (str):
            path to folder where frames will be stored
        name (str):
            name to be giver to frames (e.g. name_Number_Of_Frame)
        frame_delay (int, float):
            time between each frame
        frame_count (int):
            number of frames to be recorded

        '''
        capture = cv2.VideoCapture(path_to_video)
        count = 0
        delay = 0
        while count < frame_count:
            delay += 1
            ret, frame = cap.read()
            if delay % frame_delay == 0:
                count += 1
                file_name = path_to_store + name_to_frames +'_{}.jpg'.format(count)
                print('Created frame {}'.format(count))
                cv2.imwrite(path_to_store, frame)

class Calibrator(FramesGetter):
    def __init__(self, path_to_frames, num_frame):

        assert isinstance(path_to_frames, str)
        assert isinstance(num_frame, int)

        self.__frames = path_to_frames
        self.__num_frame = num_frame
        
    def __draw_on_frames(self, event, coord_x, coord_y, flags, param):
        '''
        Displays a frame and lets the user draw on it
        '''
        global refPt, cropping, crd
        if event == cv2.EVENT_LBUTTONDOWN:
            refPt = [(x, y)]
            cropping = True
            crd.append(refPt[0])
        elif event == cv2.EVENT_LBUTTONUP:
            refPt.append((x, y))
            cropping = False
            cv2.rectangle(im, refPt[0], refPt[1], (0, 0, 255), 2)
            cv2.imshow("image", im)

    def __get_obj_point(self):
        '''
        Returns coordinates of selected points
        Inputs:
        -------
        path (str):
            String to folder with calibration frames
        num_frame (int):
            number of frames used to perform calibration
        '''

        global crd, im
        coord = []
        count = 0

        #selecting num_frame in folder
        frames = os.listdir(self.__path_to_frames)[:self.__num_frame]

        for frame in frames:
            image = cv2.imread(path + frame)
            im = image.copy()
            while True:
                cv2.setMouseCallback("image", __draw_on_frames)
                cv2.imshow("image", im)
                key = cv2.waitKey(1) & 0xFF
                cv2.imwrite('calibrated.jpg', im)
                if key == ord('c'):
                    count += 1
                    coord.append(crd)
                    crd = []
                    break
        cv2.destroyAllWindows()
        return coord
