import numpy as np 
import cv2
from calibration.framesProcess import FramesProcess


class MeanSelectionCalib(FramesProcess):

    def __init__(self, path_to_video, path_to_frames, real_coordinates, name_to_frames='', num_frames=5, frame_delay=10):
        '''
        Calibrator using a the mean selection method. Will average coord on num_frames frames
        and calibrate the camera with the averaged points.

        Inputs:
        -------

        path_to_video (str)
            Path to video we want to calibrate the camera

        path_to_frames (str)
            Path to the folder in which frames will be stored for further process
        
        real_coordinates (dict)
            Dictionnary with real coordinate {0: (x0, y0),
                                              1, (x1, y1),
                                              ...}
            By convention: 
                - selected points for calibration will have a positive key
                - selected points for testing calibration will have a negative key

        name_to_frame (str)
            Name to be giver to the frames name_to_frames_Number_Of_frame
        
        num_frames (int)
            Number of frames on which perform the calibration
        
        frame_delay (int, float)
            Delay between to recorded frames
        '''

        assert isinstance(real_coordinates, dict)
        assert isinstance(num_frames, int)
        assert isinstance(frame_delay, (int, float))
        

        FramesProcess.__init__(self)
        self.__path_to_video = path_to_video
        self.__path_to_frames = path_to_frames
        self.__name_to_frames = name_to_frames
        self.__num_frames = num_frames
        self.__frame_delay = frame_delay
        self.__image_coordinates = None
        self.__real_coordinates_dict = real_coordinates
        self.__real_coordinates = []
        self.__testing_coordinates = []
        
        #Calibration properties
        self.__camera_matrix = None
        self.__distorsion = None 
        self.__rot_matrix = None
        self.__tran_matrix = None
        
    @property
    def camera_matrix(self):
        return self.__camera_matrix

    @property
    def distortion(self):
        return self.__distorsion
        
    def calibrate_camera(self):
        self.get_frames(
                    self.__path_to_video,
                    self.__path_to_frames,
                    self.__name_to_frames,
                    self.__frame_delay,
                    self.__num_frames)

        image_coordinate = self.get_coordinates(
                                            self.__path_to_frames,
                                            self.__num_frames)

        sample_frame = self.get_a_frame(
            self.__path_to_frames, 0)
        
        self.__compute_mean_array(image_coordinate)
        self.__get_object_point()

        try:
            assert self.__image_coordinates.shape == self.__real_coordinates.shape
            print('Success ! Same number of image and object points...')
        except:
            print('Dimension of image and object points do not match')

        image_coord_shaped, real_coord_shaped = self.__reshaping()

        camera_matrix = cv2.initCameraMatrix2D(
            [real_coord_shaped],
            [image_coord_shaped],
            sample_frame.shape[:2])

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            [real_coord_shaped],
            [image_coord_shaped],
            sample_frame.shape[:2],
            camera_matrix,
            None,
            flags=cv2.CALIB_USE_INTRINSIC_GUESS)

        self.__camera_matrix = mtx
        self.__distorsion = dist
        self.__rot_matrix = rvecs
        self.__tran_matrix = tvecs



    def __compute_mean_array(self, image_coordinate_list):
        '''
        Computes average matrix on the frames
        Inputs:
        -------
        image_coordinate_list (list)
            List of tupple-like or list_like elements
        '''

        try:
            points_mean = []
            for coord_num in range(0, len(image_coordinate_list[0])):
                mean = [0, 0]
                for frame_num in range(0, len(image_coordinate_list)):
                    mean[0] += image_coordinate_list[frame_num][coord_num][0]
                    mean[1] += image_coordinate_list[frame_num][coord_num][1]
                mean[0] = mean[0]/len(image_coordinate_list)
                mean[1] = mean[1]/len(image_coordinate_list)
                points_mean.append(mean)
            self.__image_coordinates = np.array(list(points_mean))
        
        except IndexError:
            raise IndexError('Number of coord per frame do not match')

    
    def __get_object_point(self):
        '''
        Stored selected coordinates (key >= 0) 
        order = increasing key's value
        '''
        
        selected_keys = [
            key for key in sorted(
                self.__real_coordinates_dict.keys()) if key >= 0]
        for key in selected_keys:
            self.__real_coordinates.append(list(
                self.__real_coordinates_dict[key]))
        self.__real_coordinates = np.array(
            self.__real_coordinates)

    def __reshaping(self ):
        ''' Reshape image and real coordinates to perform the calibration

        Outputs:
        --------
        reshaped_array (array)
            Reshaped array
        '''

        image_p_vec = self.__image_coordinates.copy()
        real_p_vec = self.__real_coordinates.copy()

        selected_keys = [
            key for key in sorted(
                self.__real_coordinates_dict.keys()) if key >= 0]

        image_p_vec = image_p_vec[selected_keys]

        n = real_p_vec.shape[0]
        v = np.ones((n, 1))

        real_p_vec = np.hstack((real_p_vec, v))
        real_p_vec = real_p_vec.reshape(1, -1, 3).astype('float32')
        print (image_p_vec, real_p_vec)
        image_p_vec = image_p_vec.reshape(1, -1, 2).astype('float32')

        return image_p_vec, real_p_vec
        