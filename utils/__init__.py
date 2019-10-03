"""Utilits tools"""

from collections import deque


class MultiBuffer(object):

    def __init__(self, keys, maxlen=10):

        assert isinstance(keys, list)
        assert isinstance(maxlen, int)

        self.__keys = keys
        self.values = dict.fromkeys(keys, deque([], maxlen=maxlen))

    def append(self, obs):

        assert isinstance(obs)
        assert set(obs.keys()) - set(self.__keys) == set(), \
            'Keys {} not in MultiBuffer'.format(set(obs.keys()) - set(self.__keys))