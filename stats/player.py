import numpy as np
import pandas as pd

class Player(object):
    """
    Player class to perform individual stats and performance
    analysis
    """


    def __init__(self, name, surname, number, role, team):
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

    def update(self, obs):
        """
        Update player stats form observables got from camera

        obs = {
            'position': (x, y),
            ...}
        """


    def main(self):
        """
        Main function.
        Will be called periodically got refresh players stats.
        """
        self.__compute_speed()
        


    def __compute_speed(self, smoothing_factor=1):
        
        dt = self.__stats.index
        self.__stats['speed'] = self.__stats['position'].diff() / dt


#%%
import numpy as np
import pandas as pd
df = pd.DataFrame(columns=['position', 'speed', 'acceleration'])



#%%
