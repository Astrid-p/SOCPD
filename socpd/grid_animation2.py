""" Visualization for SoCPD-grid network
"""

from socpd.visualization import gridplot, animate
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import IPython



def model_stackplot(data, ax , labels):
    """ Stackplot of people's status over time. """
    x = data.index.get_level_values('t')
    y = [data[var] for var in ['positive', 'negative']]

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
    ax2.set_title(f"{fargs[1]}\n{fargs[2][0]} proportion at step {model.t}: {model.status_quo}", 
                  fontdict = {'fontsize': 15})
    
    """ Grid animation_____________________________________________________________ 
    """
    group_grid = model.grid.attr_grid('status')
    cmap = matplotlib.colors.ListedColormap(matplotlib.colormaps["Set2"].colors[1:5])
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