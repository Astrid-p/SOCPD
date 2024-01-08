"""
VEGCON Model Module
Content: Model implementation using BA network generation
"""
#shared
import numpy as np
import networkx as nx

#agentpy
from .model import Model
from .network import Network
from .sequences import AgentList

#comma
from .agent_nw_ipf import Individual, Populating
from .hypothesis_nw import Hypothesis
 

class SocPD(Model, Hypothesis):  
    def setup(self) :
               
        # Set-up Hypothesis 
        Hypo_dict = self.p.Hypothesis_settings #_____________ added
        Hypothesis.validate_n_read_hypotheses(Hypo_dict)#__________________added

        #call-out hypothesis on actions
        self._dir_params = Hypothesis.dir_params
        self.status_var = Hypothesis.status_var
        self.all_possible_features = Hypothesis.all_possible_features
  
        self.homo_neg = Hypothesis.homo_neg
        self.homo_pos = Hypothesis.homo_pos    
        self.rules = Hypothesis.rules
        

        '''
        REMOVED FEATURES
        # Private attributes_________________________________________________________________
        # status_quo, env_beta,use_ipf, pop, 
        self.status_quo : float = .0
        if 'env_beta' in self.p:
            self._env_beta = float(self.p['env_beta'])
        else:
            self._env_beta = 0.0
        self.report("intervention's effect-env_beta", self._env_beta)
        '''
        # Define method for agent generation
        #self._use_ipf : bool = None
        if 'use_ipf' in self.p:
            self._use_ipf = self.p['use_ipf']
        else:
            self._use_ipf = False 
        self.report('use_ipf', self._use_ipf)
                
        #----------------------------------------------#
        #_______NETWORK GENERATION_____________________

        pop = self.pop = self.p['pop']
        m = self.p['m']
        p = self.p['p']
        q = self.p['q']
        graph = nx.extended_barabasi_albert_graph(
            n = pop, 
            m = m, # there are 1-p-q chance a new nodes can connect with m other existing nodes based on attachment preference
            p = p, # each existing nodes has p prob to form a new links to the others with attachment preference
            q = q, # each existing nodes has q prob to rewire one of thier existin links with attachment preference
            seed=self.random)
        self.network = Network(self, graph)
        
            # Report network's attributes
        degree_sequence = np.array([d for _, d in graph.degree()])
        self.report("Max_nw_size", np.max(degree_sequence))
        self.report("Min_nw_size", np.min(degree_sequence))
        self.report("AVG_nw_size", np.mean(degree_sequence))
        
        
        #_______AGENT GENERATION______________________
        
        # generate agents and features
        self.agents = AgentList(self, pop, Individual)
        self.Populating = Populating(self)
        if self._use_ipf:
            _feature_iter = self.Populating.populate_ipf(self._dir_params, pop)
        else:
            _feature_iter = self.Populating.populate_simple(self._dir_params, pop)
        for i, a in enumerate(self.agents):
            a.features = _feature_iter[i]

        # update agent's features and status step 0            
        self.agents.get_status_step0()
        
        #_______ADD AGENTS ON network_________________________
        
        self.network.add_agents(self.agents, self.network.nodes)

    def update(self): 
        """
        Defines the model's actions after each simulation step (including `t==0`)
        - Update influencing profile by the current status
        - Get agents' new status 
        """
        # Percentage of postive agents in the population
        positive_p = len(self.agents.select(self.agents.status==True))/self.pop
        
        # Stop simulation if all agents are positive or negative
        if positive_p <= 0.01 or positive_p >= 0.99:
            self.stop()
        
        #REMOVED FEATURES
        # self.status_quo = log(positive_p/(1-positive_p))

        # Recording positive and negative percentage
        self['positive'] = positive_p
        self.record('positive')
        self['negative'] = 1 - self['positive']
        self.record('negative')
        
        # update agent profile (from step t = 0)
        self.agents.update_influencing_profile_by_status()  
        # update influence score 
        self.agents.update_agent_combined()
        # REMOVED FEATURES
        #self.unhappy = self.agents.select(self.agents.moving == True)
        #self.unhappy.find_new_friends()

        
    def step(self) : 
        """  Update agent profile by new status (excluding `t==0`).
        """
        self.agents.change_agent_features_by_status()
        
    # REMOVED FEATURE
    # def get_segregation(self):
        # Calculate average percentage of similar neighbors
    #    return round(sum(self.agents.share_similar_diet) / self.pop, 2)

    def end(self):    
        # Reporting result  
        self.report(f'Final_{self.status_var}_proportion', self.positive) 
        self.report(f'Peak_{self.status_var}_proportion', max(self.log['positive']))






