"""
Camera Calibration Class
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

class FramesProcess(object):
    '''
    Class to perform actions on frames from a video
    Functions:
        get_frames: stores frames from video 
        get_coordinates: gets coordinates from user's clicks
        create_handler: creates handler for interactive window
        _draw_on_frames: draws on selected frames
        _get_obj_points: records coordinate from user's click 


    '''

    def __init__(self):
        pass
    
    def get_frames(self, path_to_video, path_to_store, name_to_frames, \
                   frame_delay=10, frame_count=5):
        '''
        Inputs:
        -------
        path_to_store (str):
            path to folder where frames will be stored
        name_to_frames (str):
            name to be given to frames (e.g. name_to_frames_Number_Of_Frame)
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
            ret, frame = capture.read()
            if delay % frame_delay == 0:
                count += 1
                file_name = path_to_store + name_to_frames +'_{}.jpg'.format(count)
                print('Created frame {}'.format(count))
                cv2.imwrite(file_name, frame)

        capture.release()

    def get_coordinates(self, path_to_frames, num_frame):
        self._create_handler()
        coordinate = FramesProcess._get_obj_point(path_to_frames, num_frame)
        return coordinate

    def get_a_frame(self, path_to_frames,  num_frame):
        '''
        Frame getter

        Inputs:
        -------
        path_to_frames (str)
            Path to frames
        num_frame (int)
            Number of the frame to return

        Outputs:
        --------
        image (array)
            The choose image
        '''
        
        image = cv2.imread(path_to_frames + sorted(os.listdir(path_to_frames))[num_frame])
        return image 


    def _create_handler(self):
        global crd
        crd = []
        cv2.namedWindow('Choose calibration points')

    def _draw_on_frames(event, x, y, flags, param):
        '''
        Displays a frame and let the user draw on it
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
            cv2.imshow("Choose calibration points", im)

    def _get_obj_point(path_to_frames, num_frame):
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
        frames = os.listdir(path_to_frames)[:num_frame]

        for frame in frames:
            image = cv2.imread(path_to_frames + frame)
            im = image.copy()
            while True:
                cv2.setMouseCallback("Choose calibration points", FramesProcess._draw_on_frames)
                cv2.imshow("Choose calibration points", im)
                key = cv2.waitKey(1) & 0xFF
                cv2.imwrite('calibrated.jpg', im)
                if key == ord('c'):
                    count += 1
                    coord.append(crd)
                    crd = []
                    break
        cv2.destroyAllWindows()
        return coord

