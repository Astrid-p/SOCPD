"""
Agentpy Model Module
Content: Main class for agent-based models
"""
#shared
import numpy as np

#agentpy
from .model import Model
from .grid import Grid
from .sequences import AgentList

#comma
from .agentCM import Individual, Populating
from .hypothesis import Hypothesis
 

class SocPD(Model, Hypothesis):  
    def setup(self) :
               
        # Set-up Hypothesis 
        Hypo_dict = self.p.Hypothesis_settings #_____________ added
        Hypothesis.validate_n_read_hypotheses(Hypo_dict)#__________________added

        #call-out hypothesis on actions
        self._dir_params = Hypothesis.dir_params
        self.status_var = Hypothesis.status_var
        #self.env_beta = Hypothesis.env_beta
        self.all_possible_features = Hypothesis.all_possible_features
        
        self.self_adopt = Hypothesis.self_adopt
        self.homo_neg_adopt = Hypothesis.homo_neg_adopt
        self.homo_pos_adopt = Hypothesis.homo_pos_adopt     
        self.rules = Hypothesis.rules

         # Private variables
        self.status_quo : float = .0
        self._use_ipf : bool = None
        pop = self.pop = self.p['pop']
        den = self.p['den']
        s = int(np.ceil(np.sqrt(pop/den)))
        self.report('grid_size', s)

        # Update default values with input value
        ###### seed, pop, use_ipf, _steps, grid_size (by density)
        if 'env_beta' in self.p:
            self._env_beta = float(self.p['env_beta'])
        else:
            self._env_beta = 0.0
        self.report("intervention's effect-env_beta", self._env_beta)
        if 'use_ipf' in self.p:
            self._use_ipf = self.p['use_ipf']
        else:
            self._use_ipf = False 
        self.report('use_ipf', self._use_ipf)
                #----------------------------------------------#
        #_______GRID_____________________________________________
        # setup model's Grid
        self.grid = Grid(self, (s, s), track_empty=True)

        #_______AGENTS___________________________________________
        #__generate agents
        self.agents = AgentList(self, pop, Individual)
        #__generate features then update agent's features and status
        self.Populating = Populating(self)
        if self._use_ipf:
            _feature_iter = self.Populating.populate_ipf(self._dir_params, pop)
        else:
            _feature_iter = self.Populating.populate_simple(self._dir_params, pop)
        #__update agent's features 
        for i, a in enumerate(self.agents):
            a.features = _feature_iter[i]
        #for a, fs in zip (self.agents, _feature_iter):
        #    a.features  =  fs
        #__update Agent status based on feature generation        
        self.agents.get_status_step0()
        
        #_______ADD AGENTS ON THE GRID_________________________
        self.grid.add_agents(self.agents, random=True, empty=True)


    def update(self): 
        positive_agents = self.agents.select(self.agents.status==True)
        self.status_quo = len(positive_agents)/self.pop

        # Stop simulation if all are positive or negative
        if self.status_quo == 0 or self.status_quo == 1:
            self.stop()
        #record status
        self['positive'] = self.status_quo
        self.record('positive')
        self['negative'] = 1 - self['positive']
        self.record('negative')
        
        
        # update from t = 0 
        self.agents.update_influencing_profile_by_status()  
        self.agents.update_agent_combined()
        self.unhappy = self.agents.select(self.agents.moving == True)
        
    def step(self) : 
        self.unhappy.find_new_friends()
        self.agents.change_agent_features_by_status()
        
    def get_segregation(self):
        # Calculate average percentage of similar neighbors
        return round(sum(self.agents.share_similar_diet) / self.pop, 2)

    def end(self):
        # Measure segregation at the end of the simulation
        self.report('segregation', self.get_segregation())        
        self.report(f'Final {self.status_var} proportion', self.status_quo) 
        self.report(f'Peak {self.status_var} proportion', max(self.log['positive']))

