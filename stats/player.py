#%%
import numpy as np
import pandas as pd
from collections import deque

class Player(object):
    """
    Player class to perform individual stats and performance
    analysis
    """


    def __init__(self, name, surname, number, role, team, maxlen=5):
        """
        Player constructor

        Inputs:
        -------
        
        name (str)
            Player's name

        surname (str)
            Player's surname 

        number (int)
            Player's number

        role (str)
            Player's role on the field

        team (str)
            Name of the player's team

        Attributes:
        -----------
        stats (DataFrame)
            A DataFrame containing all the computed stats
                columns = ['position', 'speed', 'acceleration',  ...]
                index = (timestamps)

        last_positions (deque):
            'max_len' last recorded postions to compute instant speed and
            acceleration

        last_times (deque)
            'maxlen' last recorded times to compute instant speed and 
            acceleration. Times will be datetime like values
         
        """
        assert isinstance(name, str)
        assert isinstance(surname, str)
        assert isinstance(number, int)
        assert isinstance(role, str)       
        
        self.name = name
        self.surname = surname
        self.number = number
        self.role = role
        self.team = team
        self.__stats = pd.DataFrame(columns=['position', 'speed', 'acceleration', 'has_ball'])
        self.__last_positions = deque([], maxlen=maxlen)
        self.__last_times = deque([], maxlen=maxlen)
        self.__instant_speed = 0
        self.__instant_acc = 0

    def update(self, obs, time):
        """
        Update player stats form observables got from camera

        obs = {
            'position': (x, y),
            ...}
        """
        self.__last_positions.append(obs['position'])
        self.__last_times.append(time)

    def main(self):
        """
        Main function.
        Will be called periodically got refresh players stats.
        """
        self.__compute_speed()
        


    def compute_speed(self, window=5, smoothing_factor=1):
        
        dt = self.__stats.index
        self.__stats['speed'][-1:] = sum(df['position'][-window:])


#%%

#%%
