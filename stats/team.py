#%%
from stats.player import Player
import yaml

class Team(object):
    
    def __init__(self, team_name, composition, color):
        assert isinstance(team_name, str)
        assert isinstance(composition, dict)
        assert isinstance(color, str)

        self.composition = dict.fromkeys(composition)

        for player in self.composition.keys():
            player_param = composition[player]
            self.composition[player] = Player(team=team_name, **player_param)

                


class Game(object):

    def __init__(self, yaml_file):
        with open(yaml_file) as ffile:
            params = yaml.load(ffile)

        team_1 = params['team_A']
        team_2 = params['team_B']
        

        self.team = Team(**team_1)
        self.adv = Team(**team_2)

#%%
game = Game('./stats/config/sample_game_config.yaml')

#%%
