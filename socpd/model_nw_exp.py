"""
Agentpy Model Module
Content: Main class for agent-based models
"""
#shared
import numpy as np
from math import log
import networkx as nx

#agentpy
from .model import Model
from .network import Network
from .sequences import AgentList

#comma
from .agent_nw import Individual, Populating
from .hypothesis_nw import Hypothesis
 
dir_params = 'parameters_demo'
all_possible_actions=  [ 'Veg_prob',
                          'concerns_health',
                         'concerns_animal_welfare',
                         'concerns_environement',
                         'concerns_convinience',
                         'concerns_familiarity',
                         'concerns_nutritional_adeq',
                         'concerns_taste',
                         'concerns_price',
                         'concerns_hunger',
                         'homo_veg_inf',
                         'homo_om_inf',

                        ]

Hypothesis_settings = { 'dir_params' : dir_params , # folder name stored actions parameters
                        'status_var':'Vegetarian',  # the variables deciding agents' status
                        # all actions from the actions parameters
                        'all_possible_actions':all_possible_actions,
                        # all actions/events influencing directly on the agent
                        'actions_to_self':['Veg_prob',
                                            'concerns_health',
                                            'concerns_animal_welfare',
                                            'concerns_environement',
                                            'concerns_convinience',
                                            'concerns_familiarity',
                                            'concerns_nutritional_adeq',
                                            'concerns_taste',
                                            'concerns_price',
                                            'concerns_hunger',],
                        # all actions/events from the agent influencing on others in thier networks
                        'actions_to_nw':['homo_veg_inf',
                                        'homo_om_inf',],

                        # actions/events from the agents having negative effects on thier networks 
                        'actions_to_nw_neg':['homo_om_inf'],
                        # actions/events from the agents having positive effects on thier networks 
                        'actions_to_nw_pos':['homo_veg_inf',],
                        }
 
class SocPD(Model, Hypothesis):  
    def setup(self) :
               
        # Set-up Hypothesis 
        Hypo_dict = Hypothesis_settings #_____________ added
        Hypothesis.validate_n_read_hypotheses(Hypo_dict)#__________________added

        #call-out hypothesis on actions
        self._dir_params = Hypothesis.dir_params
        self.status_var = Hypothesis.status_var
        self.all_possible_features = Hypothesis.all_possible_features
  
        self.homo_neg = Hypothesis.homo_neg
        self.homo_pos = Hypothesis.homo_pos    
        self.rules = Hypothesis.rules
        
        # Private attributes_________________________________________________________________
            # status_quo, env_beta,use_ipf, pop, 
        self.status_quo : float = .0
        if 'env_beta' in self.p:
            self._env_beta = float(self.p['env_beta'])
        else:
            self._env_beta = 0.0
        self.report("intervention's effect-env_beta", self._env_beta)

            # Specifying use ipf
        #self._use_ipf : bool = None
        if 'use_ipf' in self.p:
            self._use_ipf = self.p['use_ipf']
        else:
            self._use_ipf = False 
        self.report('use_ipf', self._use_ipf)
                
        #----------------------------------------------#
        #_______PREPARE NETWORK_____________________

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
        self.report("Avg_nw_size", np.mean(degree_sequence))
        
        
        #_______GENERATE AGENTS______________________
        self.agents = AgentList(self, pop, Individual)
        
            # generate features then update agent's features and status
            # Specifying use_ipf
        self.Populating = Populating(self)
        if self._use_ipf:
            _feature_iter = self.Populating.populate_ipf(self._dir_params, pop)
        else:
            _feature_iter = self.Populating.populate_simple(self._dir_params, pop)
            
            # update agent's features and status step 0
        for i, a in enumerate(self.agents):
            a.features = _feature_iter[i]
        self.agents.get_status_step0()
        
        #_______ADD AGENTS ON network_________________________
        
        self.network.add_agents(self.agents, self.network.nodes)

    def update(self): 
        positive_p = len(self.agents.select(self.agents.status==True))/self.pop
        self.status_quo = log(positive_p/(1-positive_p))

        # Stop simulation if all are positive or negative
        if positive_p == 0 or positive_p == 1:
            self.stop()
        #record status
        self['positive'] = positive_p
        self.record('positive')
        self['negative'] = 1 - self['positive']
        self.record('negative')
        
        
        # update from t = 0 
        self.agents.update_influencing_profile_by_status()  
        #if self.nw3D:
        self.agents.update_agent_combined()
        #self.unhappy = self.agents.select(self.agents.moving == True)

        
    def step(self) : 
        #self.unhappy.find_new_friends()
        self.agents.change_agent_features_by_status()
        
    #def get_segregation(self):
        # Calculate average percentage of similar neighbors
    #    return round(sum(self.agents.share_similar_diet) / self.pop, 2)

    def end(self):
        # Measure segregation at the end of the simulation
        #self.report('segregation', self.get_segregation())        
        self.report(f'Final_{self.status_var}_proportion', self.positive) 
        self.report(f'Peak_{self.status_var}_proportion', max(self.log['positive']))






