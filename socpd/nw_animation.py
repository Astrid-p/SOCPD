""" Visualization for SoCPD-grid network
"""

from socpd.visualization import animate
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import IPython
import numpy as np

import networkx as nx


matplotlib.rcParams['animation.embed_limit'] = 2**128

def model_stackplot(data, ax, labels):
    """ Stackplot of people's status over time. """
    x = data.index.get_level_values('t')
    y = [data[var] for var in ['positive','negative']]

    sns.set()

    ax.stackplot(x, y, labels=labels,
                 colors = ['forestgreen', 'darksalmon'])

    ax.legend()
    ax.set_xlim(0, max(1, len(x)-1))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time steps")
    ax.set_ylabel("Percentage of population")


def network_graph(m, ax, width):
    seed = m.reporters['seed']
    G = m.network.graph
    pos = nx.spring_layout(G, k = 0.2, seed=seed)
    color_dict = {True:'forestgreen', False:'darksalmon'}
    colors = [color_dict[c] for c in m.agents.status]
    options = {
    "node_color": colors,
    "node_size": 25,
    "edge_color": "grey",
    "linewidths": 0,
    "width": width}
    nx.draw(G, pos, ax, **options)

def combined(m, axs, *fargs ):
    ax1, ax2 = axs
    
    ax1.set_title(f"{fargs[0]}\nTime-step: {m.t},", fontdict = {'fontsize': 18})
    
    ax2.set_title(f"{fargs[1]}\n{fargs[2][0]} proportion at step {m.t}: {m.positive}", 
                  fontdict = {'fontsize': 18})

    #Network animation________________________________________________________ 
    network_graph(m, ax1, fargs[3])
    #stackplot animation_______________________________________________________
    model_stackplot(m.output.variables[fargs[4]], ax2, fargs[2])
        
def generate_animation(model, p, ani_dict):
    g_name = ani_dict['graph_nw_name']
    s_name = ani_dict['stack_plot_name']
    labels = ani_dict['pos_neg_label_list']
    width  = ani_dict['width']
    mod    = ani_dict['model_name']
    o_name = ani_dict['output_name']
    
    fig = plt.figure(figsize=(18,12))
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)
    ax2 = plt.subplot2grid((2, 3), (0, 2))

    plot_fargs = [g_name, s_name, labels, width, mod]
    animation = animate(model(p), fig, (ax1,ax2), combined, fargs = plot_fargs)
    plt.tight_layout(pad=4)
    animation.save(f"{o_name}.gif")
    
   
    return IPython.display.HTML(animation.to_jshtml())

def stack_ani(m, ax, *fargs ):
    

    ax.set_title(f"{fargs[0]}\n{fargs[1][0]} proportion at step {m.t}: {m.positive}", 
                  fontdict = {'fontsize': 18})
    
    #stackplot animation_______________________________________________________
    model_stackplot(m.output.variables[fargs[2]], ax, fargs[1])

def generate_stack(model,p, ani_dict):
    s_name = ani_dict['stack_plot_name']
    labels = ani_dict['pos_neg_label_list']
    mod    = ani_dict['model_name']
    out_path = ani_dict["out_path"]
    fig, ax = plt.subplots()

    plot_fargs = [s_name, labels, mod]
    animation = animate(model(p), fig, ax, stack_ani, fargs = plot_fargs)
    animation.save(f"{out_path}.gif")
    return IPython.display.HTML(animation.to_jshtml())