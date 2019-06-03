import numpy as np
import pandas as pd

class Player(object):

    def __init__(self, name, surname, number, role, team):
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

    def __compute_speed(self, smoothing_factor=1):
        
        dt = self.__stats.index
        self.__stats['speed'] = self.__stats['position'].diff() / dt

#%%
import numpy as np
import pandas as pd
df = pd.DataFrame(columns=['position', 'speed', 'acceleration'])



#%%
