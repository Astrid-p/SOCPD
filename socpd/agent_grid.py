"""
Agentpy Agent Module
Content: Agent Classes

"""

"""comma module"""

import json
import numpy as np
import os
import pandas as pd


from math import exp, log
from typing import List, Tuple
from tqdm import tqdm

from .objects import Object
from .agent import Agent

from .hypothesis import PARAMS_INDIVIDUAL, \
    PARAMS_IPF_WEIGHTS

class Individual2D(Agent):

    def setup(self):
        # linking with model #--> Agent
        
        self.random = self.model.random
        self.status_quo = self.model.status_quo
        #self.pos3D = self.model._3D
        #Grid method
        self.grid = self.model.grid
        
        #local variable
        self.status= False
        self.moving= False
        self.features = None
        self.influencing_profile : pd.DataFrame
        #self.influenced_score :  pd.Series
        self.move_to_pos = None
        self.share_similar_diet :float = .0

        #call-out attribute from hypothesis/model 
        self.env_beta = self.model._env_beta
        self.status_var = self.model.status_var
  
            # Hypothesis's params:
        self.params_self  = self.model.rules['actions_to_self']
        self.params_nw = self.model.rules['actions_to_nw']
            # for masking actions
        self_adopt = self.model.self_adopt #self_decisions_to_adopt
        homo_neg_adopt  = self.model.homo_neg_adopt # neg_nw_actions
        homo_pos_adopt = self.model.homo_pos_adopt # pos_nw_actions
        
        # get local attribute __________________________________________________________
        # ____________________ to SELF _____________________
        # to separate scores of moving and adopt from actions_to_self (hypothesis)
        self.masked_self_adopt = np.isin(self.params_self.index, self_adopt)
        
        #____________________ to Network _______________________
        
        # to update influencing_profile by agent's status
        self.masked_neg = np.isin(self.params_nw.index, homo_neg_adopt)
        self.masked_pos = np.isin(self.params_nw.index, homo_pos_adopt)
        
        # to separate scores of moving and adopt (from actions_to_nw) after homo cross-checked 
        self.masked_nw_adopt = np.isin(self.params_nw.index, homo_neg_adopt + homo_pos_adopt)
        


    @staticmethod
    def get_p(sum_betas:float):
        '''logit reversed'''
        return exp(sum_betas)/(1+exp(sum_betas))
    
    #_________________________________________________________________
    # SETUP agent's status at t = 0
    def get_status_step0(self) -> None:
        """
        Get agent status at t = 0 
        Returns: None -> change agents' status to True if status_var_yes = 1
        """
        fs = self.features
        self.status = fs[f'{self.status_var}_yes'] ==1 


    #________________________________________________________________
    # Set influencing profile by statu
    def update_influencing_profile_by_status(self):
        '''Set feature-customized influencing-profile of each agents     
        Required: 
            Hypothesis rules (param_nw)
            updated status (from t0)
            features
        Return: None
            Update influencing_profile by status 
            actions against agent's status will be zero out
        '''
        params_nw = np.array(self.params_nw)
        features = np.array(self.features)
        influencing_profile = np.multiply(params_nw,features)
        # zero-out the influence doesn't match agent's status
        if self.status:
            influencing_profile[self.masked_neg,:] = 0.0
        else:
            influencing_profile[self.masked_pos,:] = 0.0

        # set agent's influencing profile
        self.influencing_profile = influencing_profile
    
    def update_agent_combined(self):
        
        '''Finalize all score combination
            - update segregration
            - update self.moving 
            - update self.move_away_pos
            - update self.status'''
            
        """Get scores for all actions to self"""
        
        _score_self = np.array(self.params_self).dot(np.array(self.features))
            # self influence on network tie
        _score_self_stay = np.sum(_score_self[~self.masked_self_adopt])
            # self influence on adopting the practice 
        _score_self_adopt = np.sum(_score_self[self.masked_self_adopt])
        _score_adopt = _score_self_adopt + self.env_beta + self.status_quo
        
        """ Get scores from neighbor by shared similarity"""
        neighbors = self.grid.neighbors(self)   
        
        # Update segregration ( p of neighbors with same status ______________________________________________________
        similar = len([n for n in neighbors if n.status == self.status])
        ln = len(neighbors)
        self.share_similar_diet = similar / ln if ln > 0 else 0     
        
        # Get scores from neighbor' actions ________________________________________________________
        if ln == 0:
            # if there is no neighbor, randomly move to an empty spot
            self.move_to_pos = self.random.choice(self.model.grid.empty)
            prob_move = self.get_p(-_score_self_stay)
            
        else:
            # scored neighbors by similarity (for both adopt and stay)
            neigh_pfs = np.array([n.influencing_profile for n in neighbors])
            _neigh_scores = neigh_pfs.dot(np.array(self.features)) 
            
            # move away from neighbor with lowest score_to_move ______________________________________________________
            _neigh_scores_stay = np.sum(_neigh_scores[:, ~self.masked_nw_adopt], axis = 1) + _score_self_stay
                    
            weak_neigh = None
            for i, nb in enumerate(neighbors):
                if i == np.argmin(_neigh_scores_stay):
                    weak_neigh = self.grid.positions[nb]
                    choice = [2,-2]
                    path = np.array([self.random.choice(choice), self.random.choice(choice)])
                    self.move_to_pos = tuple(path + weak_neigh)
                break
            

            #_____________NOTE: REVERSED EFFECT____________
            # actions to move measure satisfaction of network ties
            # from network tie with lowest satisfactions -> get unsatisfied probability by using reversed beta
            prob_move = self.get_p(-np.min(_neigh_scores_stay)) 
            _score_nw_adopt =  np.sum(_neigh_scores[:, self.masked_nw_adopt]) 
            _score_adopt += _score_nw_adopt 
        
        """ Update self.moving """
        if self.move_to_pos in self.model.grid.empty:
            self.moving = self.random.random() <= prob_move
        else:
            self.moving = False
        
        """ UPDATE self.status """
        prob_adopt = self.get_p(_score_adopt)
        self.status = self.random.random() <= prob_adopt
    
    

    '''STEP 4 : taking action following  moving and status'''
    def find_new_friends(self):
        self.grid.move_to(self, self.move_to_pos)
        
    def change_agent_features_by_status(self):
        if self.status:
            self.features[self.features.index.str.contains(pat = f'{self.status_var}')] = [0,1]
        else:
            self.features[self.features.index.str.contains(pat = f'{self.status_var}')] = [1,0]    

class Individual3D(Agent):

    def setup(self):
        # linking with model #--> Agent
        
        self.random = self.model.random
        self.status_quo = self.model.status_quo
        
        #Grid method
        self.grid = self.model.grid
        
        #local variable
        self.status= False
        self.moving= False
        self.features = None
        self.influencing_profile : pd.DataFrame
        #self.influenced_score :  pd.Series
        self.move_to_pos = None
        self.share_similar_diet :float = .0

        #call-out attribute from hypothesis/model 
        self.env_beta = self.model._env_beta
        self.status_var = self.model.status_var
  
            # Hypothesis's params:
        self.params_self  = self.model.rules['actions_to_self']
        self.params_nw = self.model.rules['actions_to_nw']
            # for masking actions
        self_adopt = self.model.self_adopt #self_decisions_to_adopt
        homo_neg_adopt  = self.model.homo_neg_adopt # neg_nw_actions
        homo_pos_adopt = self.model.homo_pos_adopt # pos_nw_actions
        
        # get local attribute __________________________________________________________
        # ____________________ to SELF _____________________
        # to separate scores of moving and adopt from actions_to_self (hypothesis)
        self.masked_self_adopt = np.isin(self.params_self.index, self_adopt)
        
        #____________________ to Network _______________________
        
        # to update influencing_profile by agent's status
        self.masked_neg = np.isin(self.params_nw.index, homo_neg_adopt)
        self.masked_pos = np.isin(self.params_nw.index, homo_pos_adopt)
        
        # to separate scores of moving and adopt (from actions_to_nw) after homo cross-checked 
        self.masked_nw_adopt = np.isin(self.params_nw.index, homo_neg_adopt + homo_pos_adopt)
        


    @staticmethod
    def get_p(sum_betas:float):
        '''logit reversed'''
        return exp(sum_betas)/(1+exp(sum_betas))
    
    #_________________________________________________________________
    # SETUP agent's status at t = 0
    def get_status_step0(self) -> None:
        """
        Get agent status at t = 0 
        Returns: None -> change agents' status to True if status_var_yes = 1
        """
        fs = self.features
        self.status = fs[f'{self.status_var}_yes'] ==1 


    #________________________________________________________________
    # Set influencing profile by statu
    def update_influencing_profile_by_status(self):
        '''Set feature-customized influencing-profile of each agents     
        Required: 
            Hypothesis rules (param_nw)
            updated status (from t0)
            features
        Return: None
            Update influencing_profile by status 
            actions against agent's status will be zero out
        '''
        params_nw = np.array(self.params_nw)
        features = np.array(self.features)
        influencing_profile = np.multiply(params_nw,features)
        # zero-out the influence doesn't match agent's status
        if self.status:
            influencing_profile[self.masked_neg,:] = 0.0
        else:
            influencing_profile[self.masked_pos,:] = 0.0

        # set agent's influencing profile
        self.influencing_profile = influencing_profile
    
    def update_agent_combined(self):
        
        '''Finalize all score combination
            - update segregration
            - update self.moving 
            - update self.move_away_pos
            - update self.status'''
            
        """Get scores for all actions to self"""
        
        _score_self = np.array(self.params_self).dot(np.array(self.features))
            # self influence on network tie
        _score_self_stay = np.sum(_score_self[~self.masked_self_adopt])
            # self influence on adopting the practice 
        _score_self_adopt = np.sum(_score_self[self.masked_self_adopt])
        _score_adopt = _score_self_adopt + self.env_beta + self.status_quo
        
        """ Get scores from neighbor by shared similarity"""
        neighbors = self.grid.neighbors(self)   
        
        # Update segregration ( p of neighbors with same status ______________________________________________________
        similar = len([n for n in neighbors if n.status == self.status])
        ln = len(neighbors)
        self.share_similar_diet = similar / ln if ln > 0 else 0     
        
        # Get scores from neighbor' actions ________________________________________________________
        if ln == 0:
            # if there is no neighbor, randomly move to an empty spot
            self.move_to_pos = self.random.choice(self.model.grid.empty)
            prob_move = self.get_p(-_score_self_stay)
            
        else:
            # scored neighbors by similarity (for both adopt and stay)
            neigh_pfs = np.array([n.influencing_profile for n in neighbors])
            _neigh_scores = neigh_pfs.dot(np.array(self.features)) 
            
            # move away from neighbor with lowest score_to_move ______________________________________________________
            _neigh_scores_stay = np.sum(_neigh_scores[:, ~self.masked_nw_adopt], axis = 1) + _score_self_stay
                    
            weak_neigh = None
            for i, nb in enumerate(neighbors):
                if i == np.argmin(_neigh_scores_stay):
                    weak_neigh = self.grid.positions[nb]
                    choice = [2,-2]
                    path = np.array([self.random.choice(choice), self.random.choice(choice), self.random.choice(choice)])
                    self.move_to_pos = tuple(path + weak_neigh)
                break


            #_____________NOTE: REVERSED EFFECT____________
            # actions to move measure satisfaction of network ties
            # from network tie with lowest satisfactions -> get unsatisfied probability by using reversed beta
            prob_move = self.get_p(-np.min(_neigh_scores_stay)) 
            _score_nw_adopt =  np.sum(_neigh_scores[:, self.masked_nw_adopt]) 
            _score_adopt += _score_nw_adopt 
        
        """ Update self.moving """
        if self.move_to_pos in self.model.grid.empty:
            self.moving = self.random.random() <= prob_move
        else:
            self.moving = False
        
        """ UPDATE self.status """
        prob_adopt = self.get_p(_score_adopt)
        self.status = self.random.random() <= prob_adopt
    
    

    '''STEP 4 : taking action following  moving and status'''
    def find_new_friends(self):
        self.grid.move_to(self, self.move_to_pos)
        
    def change_agent_features_by_status(self):
        if self.status:
            self.features[self.features.index.str.contains(pat = f'{self.status_var}')] = [0,1]
        else:
            self.features[self.features.index.str.contains(pat = f'{self.status_var}')] = [1,0]    

class Populating(Object):
    def __init__(self, model):
        super().__init__(model)
        self.nprandom = model.nprandom
        self.all_possible_features = self.model.all_possible_features
           
    def sampling_from_ipf(self, dir_params: str, pop:int) -> pd.DataFrame:
        """
        Sample from IPF distribution saved
        as `weights.csv` in the parameters folder

        Parameters
        ----------
        pop (int): size of data sample
        dir_params (str): path to the parameters folder

        Returns
        -------
        sample (pandas.dataFrame): dataframe containing the sampling
        """
        fpath_weights = os.path.join(dir_params, PARAMS_IPF_WEIGHTS)
        assert os.path.isfile(fpath_weights)
        df_weights = pd.read_csv(fpath_weights, sep=",", index_col=0)
        weights = df_weights["weight"] / df_weights["weight"].sum()
        indices = df_weights.index
        

        sample_indices = self.nprandom.choice(indices, pop, p=weights)
        sample = df_weights.loc[sample_indices].drop(["weight"], axis=1)
        sample = sample.reset_index(drop=True)
        return sample

    def populate_ipf(self, dir_params: str, pop:int) -> List:
        """
        Create a population of individual agents
        with the given weights obtained via IPF

        Args:
            pop (int): size of data sample.
            dir_params (str): path to parameters folder.

        Returns:
            List[Individual]: A list containing instances of
            the individual class, each representing an
            agent with specific features.
        """
        _features = pd.DataFrame()

        sample = self.sampling_from_ipf(dir_params, pop)

        # one-hot encoding
        encoded_columns = pd.get_dummies(sample).reindex(
            columns=self.all_possible_features,
            fill_value=0
        )
        _features = pd.concat([_features, encoded_columns], axis=1)
        #________________added
        cols = sorted(_features.columns)
        _features = _features[cols]

        # Add 'baseline' column filled with ones if this is not present yet
        if 'baseline' not in _features.columns:
            _features.insert(0, "baseline", 1)

        return [_features.iloc[i] for i in
                tqdm(range(pop), desc="Populating individuals", unit="i")]

    def populate_simple(self, dir_params: str, pop:int) -> List:
        """
        Create a population of individual agents
        with the given feature parameters.

        Args:
            pop (int): population size, i.e., number of agents.
            dir_params (str): dir to the folder containing
            feature parameter file.
            #from_scratch (bool, optional): flag of creating hypothesis
            from scratch or reading from files. Defaults to False.

        Returns:
            list[Individual]: a list of Individual agents
        """
        assert pop > 0, 'Size must be positive!'
        assert isinstance(pop, int), 'Pop - population size must be integer!'
        assert os.path.isdir(dir_params), "Given folder doesn't exist!"

        fpath_params_individual = os.path.join(dir_params, PARAMS_INDIVIDUAL)
        with open(fpath_params_individual) as f:
            features = json.load(f)
        features = {k.lower():v for k, v in features.items()}

        _features = pd.DataFrame()
        for feature, distribution in features.items():
            _features[feature] = self.nprandom.choice(
                distribution[0], pop, p=distribution[1]
            )

            # Define all possible columns (including those not in the sample)
            # When the sample size is too small,
            # this doesn't cover all categories,
            # the resulting DataFrame thus lacks those columns.
            # To solve the issue, we ensure all possible categories are present
            # when creating the dummy variables

        # one-hot encoding
        categorical_cols = _features.select_dtypes(include=['object'])
        encoded_cols = pd.get_dummies(categorical_cols).reindex(
            columns=self.all_possible_features,
            fill_value=0
        )
        _features.drop(categorical_cols.columns, axis=1, inplace=True)
        _features = pd.concat([_features, encoded_cols], axis=1)
        #________________added
        cols = sorted(_features.columns)
        _features = _features[cols]

        # Add 'baseline' column filled with ones
        _features.insert(0, "baseline", 1)

        return [_features.iloc[i] for i in
                tqdm(range(pop), desc="Populating individuals", unit="i")]
    
