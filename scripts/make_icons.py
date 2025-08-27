import matplotlib as mp
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
import numpy as np

n_turns = 2


def make_main_icon():
    n_turns = 1.5
    fig = plt.figure(frameon=False)
    fig.set_size_inches(1, 1)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)

    spline_effect = [
        pe.Stroke(linewidth=12, foreground="white"),
        pe.Stroke(linewidth=6, foreground="cornflowerblue"),
    ]

    patch = mp.patches.Rectangle([0, 0], width=1, height=1, facecolor='midnightblue',
                                edgecolor='white', linewidth=10,
                                transform=ax.transAxes)
    ax.add_patch(patch)

    circle = mp.patches.Circle((0.5, 0.5), radius=0.35,
                               path_effects=spline_effect, color='none',
                               transform=ax.transAxes)
    ax.add_patch(circle)

    ax.plot([0.3, 0.5, 0.7], [0.5, 0.5, 0.5], 'o', path_effects=spline_effect, transform=ax.transAxes)
    ax.plot([0.5, 0.5, 0.5], [0.3, 0.5, 0.7], 'o', path_effects=spline_effect, transform=ax.transAxes)

    #theta = np.linspace(0, n_turns*2*np.pi, 1000)
    #x = 2 * theta * np.cos(theta)
    #y = 2 * theta * np.sin(theta)
    #plt.plot(x, y, color='none', solid_capstyle='round', path_effects=spline_effect)

    #b = max(np.abs(x.min()), x.max()) + 7.5
    #xs = 0.5*(x.max() - (-x.min()))
    #ys = np.mean(y)
    #ax.axis(xmin=-b+xs, xmax=b+xs, ymin=-b+ys, ymax=b+ys)
    fig.savefig('main-icon.png', transparent=False, bbox_inches='tight')


#make_spiral()
#make_cells()
#make_exclude()
#make_tile()
make_main_icon()
