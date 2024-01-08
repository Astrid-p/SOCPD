""" Visualization for SoCPD-grid network
"""

from socpd.visualization import gridplot, animate
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import IPython
import numpy as np



def model_stackplot(data, ax, labels):
    """ Stackplot of people's status over time. """
    x = data.index.get_level_values('t')
    y = [data[var] for var in ['positive','negative']]

    sns.set()

    ax.stackplot(x, y, labels=labels,
                 colors = ['#9ACD32', '#D2B48C'])

    ax.legend()
    ax.set_xlim(0, max(1, len(x)-1))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time steps")
    ax.set_ylabel("Percentage of population")

def combined_plot(model, axs, *fargs ):
    ax1, ax2 = axs

    ax1.set_title(f"{fargs[0]}\nTime-step: {model.t}, "
                 f"Avg similar {fargs[4]}: {model.get_segregation()}\n", fontdict = {'fontsize': 15})
    ax2.set_title(f"{fargs[1]}\n{fargs[2][0]} proportion at step {model.t}: {model.positive}", 
                  fontdict = {'fontsize': 15})
    
    """ Grid animation_____________________________________________________________ 
    """
    group_grid = model.grid.attr_grid('status')
    cmap = matplotlib.colors.ListedColormap(matplotlib.colormaps["Set2"].colors[1:5])#.reversed()
    gridplot(group_grid, cmap=cmap, ax=ax1)
    ax1.grid(False)
    
    """ stackplot animation_______________________________________________________
    """
    model_stackplot(model.output.variables[fargs[3]], ax2, fargs[2])

def animation_plot(model, p, plot_fargs):
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 6), ) # Prepare figure 
    animation = animate(model(p), fig, axs, combined_plot, fargs = plot_fargs)
    fig.tight_layout(pad = 3.5)
   
    return IPython.display.HTML(animation.to_jshtml())

#________________________________________________________________________________________
""" 3D plot fir grid network generation"""

def grid_3d(m, ax):
    #ndim = m.p['ndim']
    pos = m.grid.positions.values()
    pos = np.array(list(pos)).T # Transform
    cmap = matplotlib.colors.ListedColormap(matplotlib.colormaps["Set2"].colors[1:5])#.reversed()
    ax.scatter(*pos, cmap = cmap,  c=m.agents.status)
    ax.set_xlim(0, m.grid_size)
    ax.set_ylim(0, m.grid_size)
    #if ndim == 3:
    ax.set_zlim(0, m.grid_size)
    #ax.set_axis_off()

        
def combined_plot3d(model,axs, *fargs ):

    ax1 , ax2 = axs   
    ax1.set_title(f"{fargs[0]}\nTime-step: {model.t}, "
                 f"Avg similar {fargs[4]}: {model.get_segregation()}", fontdict = {'fontsize': 15})
    ax2.set_title(f"{fargs[1]}\n{fargs[2][0]} proportion at step {model.t}: {model.positive}", fontdict = {'fontsize': 15})
    
    #Grid animation_____________________________________________________________ 
    grid_3d(model, ax1)
    #stackplot animation_______________________________________________________
    model_stackplot(model.output.variables[fargs[3]], ax2, fargs[2])


def animation_plot3d(model, p, plot_fargs):
    
    #projection = '3d' if p['ndim']== 3 else None
    fig = plt.figure(figsize = (12,6))
    ax1 = fig.add_subplot(121, projection= '3d')
    ax2 = fig.add_subplot(122)
    
    #fig, axs = plt.subplots(1, 2, figsize=(12, 6), ) # Prepare figure 
    animation = animate(model(p), fig, (ax1, ax2), combined_plot3d, fargs = plot_fargs)
    fig.tight_layout(pad = 3.5)
    return IPython.display.HTML(animation.to_jshtml())

def generate_animation(model, p, plot_fargs, ndim_3d:bool =False):
    if ndim_3d: 
        return animation_plot3d(model, p, plot_fargs)
    else:
        return animation_plot(model, p, plot_fargs)

        
 
        