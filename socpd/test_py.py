import pandas as pd
from typing import List, Tuple
from tqdm import tqdm
class Person:
    def __init__(self, model, id:int, features):
            super().__init__(model)
            self.model = model
            self.id: int = id
            self._feature = features
            self.score  = 0
            self.grid = self.model.grid
    
    @staticmethod
    def get_agents(model, df,s):
            return [Person(model, i, df.iloc[i]) for i in range(s)]
    def getscore(self):
            self.score = sum(self._features)
            
    def find_new_home(self):
            """ Move to random free spot and update free spots. """
            new_spot = self.random.choice(self.model.grid.empty)
            self.grid.move_to(self, new_spot)


