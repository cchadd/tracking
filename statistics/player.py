import numpy as np
import pandas as pd

class Player(object):

    def __init__(self, name, surname, number, team):
        self.__name = name
        self.__surname = surname
        self.__number = number
        self.__team = team
        self.__stats = pd.DataFrame(columns=['position', 'speed', 'acceleration', 'has_ball'])


    def __compute_speed(self, smoothing_factor=1):
        
        dt = self.__stats.index
        self.__stats['speed'] = self.__stats['position'].diff() / dt

#%%
import numpy as np
import pandas as pd
df = pd.DataFrame(columns=['position', 'speed', 'acceleration'])



#%%
