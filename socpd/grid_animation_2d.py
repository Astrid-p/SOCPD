""" Visualization for SoCPD-grid network
"""

from socpd.visualization import gridplot
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns



def model_stackplot(data, ax , labels):
    """ Stackplot of people's status over time. """
    x = data.index.get_level_values('t')
    y = [data[var] for var in ['negative','positive']]

    sns.set()

    ax.stackplot(x, y, labels=labels,
                 colors = ['#D2B48C','#9ACD32', ])

    ax.legend()
    ax.set_xlim(0, max(1, len(x)-1))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time steps")
    ax.set_ylabel("Percentage of population")

def animation_plot_grid(model, axs, *fargs ):
    ax1, ax2 = axs
    ax1 = projection = '3d' if fargs[5]['ndim'] == 3 else None
    ax1 = fig.add_subplot(111, projection=projection)
    ax1.set_title(f"{fargs[2]}\nTime-step: {model.t}, "
                 f"Avg similar {fargs[0]}: {model.get_segregation()}\n", fontdict = {'fontsize': 15})
    ax2.set_title(f"{fargs[3]}\n{fargs[4][0]} proportion at step {model.t}: {model.status_quo}", 
                  fontdict = {'fontsize': 15})
    
    """ Grid animation_____________________________________________________________ 
    """
    group_grid = model.grid.attr_grid('status')
    cmap = matplotlib.colors.ListedColormap(matplotlib.colormaps["Set2"].colors[1:5])
    gridplot(group_grid, cmap=cmap, ax=ax1)
    ax1.grid(False)
    
    """ stackplot animation_______________________________________________________
    """
    model_stackplot(model.output.variables[fargs[1]], ax2, fargs[4])