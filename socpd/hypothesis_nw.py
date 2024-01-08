"""Hypothesis class definition"""

import json
import os
import pandas as pd
from typing import Dict, Set, List, Union

PARAMS_INDIVIDUAL = 'params_individual.json'
PARAMS_IPF_WEIGHTS = "ipf_weight_matrix.pickle"
PARAMS_IPF_KEYMAPS = "ipf_keymaps.pickle"



class Hypothesis:
    """
    The Hypothesis class is responsible for managing and validating
    hypotheses specified by the user.

    """
    _required_params = ['size', 'steps', 'self','neighbor' 'status',]
                        #'lockdown_policies', 'lockdown']
    
    
    @staticmethod
    def _get_one_hot_encoded_features(fpath_params_individual: str) -> List:
        """
        One-hot encode categorical features in the
        `params_individual.json` file and return the
        feature list.

        Args:
            fpath_params_individual (str): Path to the
            individual parameters JSON file.

        Returns:
            features (list): List of one-hot encoded features.

        """
        with open(fpath_params_individual) as f:
            params_individual = json.load(f)
        features = []
        for key, value in params_individual.items():
            if isinstance(value[0][0], str):
                features += [key + '_' + v for v in value[0]]
            else:
                features += [key]
        return features

    @staticmethod
    def _check_status_variable(fpath_params_individual: str,
                                 status_var:str):    
        '''Validate error: if status_var is not in json file, if not categorized by yes/no'''
        status_var = status_var.lower()
        
        with open(fpath_params_individual) as f:
            params_individual = json.load(f)
        params_individual = {k.lower():v for k, v in params_individual.items()}
        
        if not status_var in params_individual:
            raise ValueError(f"variable to specify agent's status -'{status_var}' not found in json file")
        
        desired_group_name = ['no', 'yes']
        status_groups = sorted([g.lower() for g in params_individual[f'{status_var}'][0]])
        if desired_group_name != status_groups:
            raise ValueError(f" Varible '{status_var}' should be categorized with 'no'/'yes' in json file")

    @staticmethod
    def detect_wrong_input_format(Dict_params):
        desired_dict = {'dir_params':str,
                        'status_var': str,
                        'all_possible_actions': list,
                        'actions_to_self' : list,
                        'actions_to_nw': list,
                        'actions_to_nw_neg': list,
                        'actions_to_nw_pos': list
                        }

        dict_params = {key.lower(): value for key, value in Dict_params.items()}


        if not set(desired_dict.keys()) == set(dict_params.keys()):
            raise ValueError("Hypothesis_settings's Keys do not match with the default \nList of desired keys:\n%s" % '\n'.join(desired_dict.keys()) )

        wrongtype_key = []
        for key, value in dict_params.items():
            if not type(value) == desired_dict.get(key):
                wrongtype_key.append(key)
        if len(wrongtype_key) >0:
            raise TypeError("Mismatch type for Hypothesis_settings's value \n\
                            -> Default type:\n%s" % '\n'.join([f'{k} - {desired_dict[k]}' for k in wrongtype_key])
                            ) 
            
    @classmethod
    def create_empty_hypotheses(cls, dir_params: str,
                                status_var:str,
                                all_possible_actions: List[str]) -> None:
        """
        Preparation step:
        To generate empty CSV file for storing hypothesis (betas) of actions according to agents'features 
        Agent features will be generated from the json file
        Args:
            dir_params (str): The directory of the folder that contains the agent and model parameter files.
        Returns:
            None: This function does not return anything
            as it creates empty csv files int the specified directory
        """
        # setup hypothesis                
        fpath_params_individual = os.path.join(dir_params, PARAMS_INDIVIDUAL)
        cls._check_status_variable(fpath_params_individual, status_var)
        
        #get json file, check if the files exist
        if not os.path.exists(fpath_params_individual):
            raise FileNotFoundError(f"'{PARAMS_INDIVIDUAL}' \
            file is missing in the directory '{dir_params}'")
        

        #structure the empty hypothesis params file
        feas = cls._get_one_hot_encoded_features(fpath_params_individual)  
        feas = [f.lower() for f in feas]
        columns = ['actions', 'baseline']
        columns += feas
        df = pd.DataFrame(0, index=range(len(all_possible_actions)), columns=columns)
        df['actions'] = sorted(all_possible_actions)

        output_fpath = os.path.join(dir_params,f'agent_behaviors_on_{status_var}.csv')
        df.to_csv(output_fpath, sep=';', index=False)

    @classmethod
    def validate_n_read_hypotheses(cls, hypothesis_params: Dict ) -> Dict[str, pd.DataFrame]:
        """
        - Validate and read Hypo data/ model params file for agent's behaviorial settings
        - Setup cls Hypothesis to carry agent's all possible hypothesized attributes  
        Args:
            dir_params (str): path of the parameters folder
            all_possible_actions(str): list all actions
            all_possible_features(str):list of all agent's features
            actions_to_self(tr): list of impacting on self 
            actions_to_nw(str):List of actions impacting on neighbors
            pos/neg _nw_actions: Lists of actions decide by agent'status

   
        Returns: cls-Hypothesis carrying agent's all possible hypothesized attributes (formated) 
            - dir_params, tatus_var (lowercase), actions (lowercase), features (sored/lowercase)
            - categorized actionns by purposes (list -> lowercase)
            - agent profiles (params) by actions to self and to neighbors
        """
        #_____________________
        def list_equal(list1, list2): 
            if sorted(list1) == sorted(list2):
                return True
            else:
                return False
        def ifsubset(small, big):
            if len(small) ==0:
                return True
            else: 
                return set(small).issubset(set(big))
        def lowerlist(l):
            if len(l) ==0:
                return l
            else:
                return [e.lower() for e in l]
        
        #__________________________________________________________________
        # Validate Hypothesis_settings_____________________________________
            # Validate formmat
        cls.detect_wrong_input_format(hypothesis_params)
            # re-setup to lowercase
        dir_params           = hypothesis_params['dir_params']
        status_var           = hypothesis_params['status_var']
        all_possible_actions = hypothesis_params['all_possible_actions']
        actions_to_self      = lowerlist(hypothesis_params['actions_to_self'])
        actions_to_nw        = lowerlist(hypothesis_params['actions_to_nw'])    
        homo_neg = lowerlist(hypothesis_params['actions_to_nw_neg'])
        homo_pos = lowerlist(hypothesis_params['actions_to_nw_pos']) 

        cls.sim_setup_n_validate_param_file(dir_params,
                                            status_var,
                                            all_possible_actions)  
        
        # Validate hypothesized purposes of all actions
        if not list_equal(cls.all_possible_actions,
                         actions_to_self+actions_to_nw):
            raise ValueError("combination (self-actions + nw-actions) needs to match with 'all_possible_actions'")

        
        if not all([#ifsubset(self_adopt, actions_to_self),
                    ifsubset(homo_pos + homo_neg, actions_to_nw),
                    #ifsubset(actions_to_stay, cls.all_possible_actions),
                    ]):
            raise ValueError("negative and postive action should be included in actions to nw")

        # _________________________________________________________________
        # Setup agent's formated profiles___________________________________
            # access data; format data to lowercase feature and actions, float values
        file_name = f'agent_behaviors_on_{cls.status_var}.csv'
        fpath_params = os.path.join(cls.dir_params, file_name)
        df = pd.read_csv(fpath_params, delimiter=';', decimal=".") 
        df = df.fillna(0)
                # format lowercase for data file   
        df.columns = df.columns.str.lower()
        df['actions'] = df['actions'].str.lower()
                # format float values for data file
        for col in df.columns:
            if col != "actions": #skip col of the effect/action names
                df[col] = df[col].astype(float)

            # rearrange order of data's cols by cls-Hypothesis's features, set actions as index
        cols = ['baseline', 'actions']
        cols += Hypothesis.all_possible_features  
        df = df[cols]
        df = df.set_index('actions')
            # sparating profile by actions to self and to neighbors
        data_dfs = {}
        grouped = df.groupby(df.index.isin(actions_to_self))
        data_dfs['actions_to_self']=grouped.get_group(True)
        data_dfs['actions_to_nw']=grouped.get_group(False)
        
        #__________________________________________________________________
        # update final hypotheses_________________________________________
        Hypothesis.homo_neg = sorted(homo_neg)
        Hypothesis.homo_pos = sorted(homo_pos)
        Hypothesis.rules = data_dfs

    
    @classmethod
    def sim_setup_n_validate_param_file(cls, dir_params: str,
                            status_var:str,
                        all_possible_actions:List[str])  -> None:
        """Simple aalidate the hypothesis data (model params file) and simple setup cls-Hypothesis 
        Args:
            dir_params (str): dir to the folder containing
            status_var (str): specify target variable measure the phenomenon/agent's status
            hypothesis and parameter files.
        SimSetup and Raises error:
            -  SimSetup cls Hypothesis with sim_setup_hypothesis_params (formated status_var, actions, features)
            - ValueError: If any validation checks on:
                + wrong arg status_var (to json)
                + unmatched arg actions with hypodata 's action
                + unmatched json one-hot features with hypodata's features
        """
        
        # SimSetup___________________________________________
        # simple Hypothesis params setup
        cls.sim_setup_hypothesis_params(dir_params,
                                    status_var,
                                    all_possible_actions)
        
        # Validating ___________________________________________
        #  Validating hypo data/model params file
        required_features = ["actions", "baseline"]
        required_features += cls.all_possible_features
        required_actions = cls.all_possible_actions
            # Raising missing file / incorrect file name
        fname = f'agent_behaviors_on_{cls.status_var}.csv' 
        fpath = os.path.join(dir_params, fname) 
        if not os.path.isfile(fpath):
            raise ValueError(f"Hypothesis file specified by '{fname}' -> not found")   
            # Acessing model params file     
        hypothesis_data = pd.read_csv(fpath, sep=";", decimal=",")
            # Raise missing feature 
        missing_features = set(required_features) - set([c.lower() for c in hypothesis_data.columns])
        if any(missing_features):
            raise ValueError("Missing features in %s" % ' - '.join(
                (fname, ", ".join(missing_features))
                ) )
            # Raise missing actions         
        missing_actions = set(required_actions) - set([a.lower() for a in hypothesis_data["actions"]])
        if any(missing_actions):
            raise ValueError("Missing actions in %s" % ' - '.join(
                (fname, ", ".join(missing_actions))
                ) )
        
    @classmethod   
    def sim_setup_hypothesis_params(cls, dir_params: str,
                                    status_var:str,
                                    all_possible_actions:List[str],
                                    ) -> None:
        
        '''set hypothesis features and status_var based on json file
        and hypothesis actions based on the input'''
        # assign the variable used to specify status of the agents  
        status_var = status_var.lower()
        path_individual = os.path.join(dir_params, PARAMS_INDIVIDUAL)
        cls._check_status_variable(path_individual, status_var) 

        # create onehot encoding for agent features based on json file (crosstab proportions)
        cols = cls._get_one_hot_encoded_features(path_individual)  
        cols = [c.lower() for c in cols]
        
        #____ simple set up for hypothesis --> attach hypo parameters into Hypothesis______________________
        # name of dir_params storinng hypothesis parameters
        Hypothesis.dir_params = dir_params
        # Name of varible to specify agent status ---> lower case
        Hypothesis.status_var = status_var
        # All possible actions lower cases
        Hypothesis.all_possible_actions = [a.lower() for a in all_possible_actions] 
        # All possible features lower cases and sorted      
        Hypothesis.all_possible_features = sorted(cols)
