import os, sys
import numpy as np
from scipy.special import beta
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
plt.style.use('seaborn')

def beta_pdf(x, a, b):
    return x**(a-1) * (1-x)**(b-1) / beta(a, b)

if __name__ == '__main__':
    n = 10000
    x = np.linspace(0, 1, n)

    # make plot
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 5)
    l, = plt.plot(x, beta_pdf(x, 1, 1), lw=2)
    fig.subplots_adjust(left=0.25, bottom=0.25)

    # make sliders
    ax_a = plt.axes([0.25, 0.1, 0.65, 0.03])
    ax_b = plt.axes([0.25, 0.15, 0.65, 0.03])
    s_a = Slider(ax_a, 'a', 1e-5, 10.0, valinit=1)
    s_b = Slider(ax_b, 'b', 1e-5, 10.0, valinit=1)

    # plot initial pdf
    a = s_a.val; b = s_b.val
    y = beta_pdf(x, a, b)

    # update plot
    def update(val):
        a = s_a.val; b = s_b.val
        y = beta_pdf(x, a, b)
        l.set_ydata(y)
        fig.canvas.draw_idle()

    # update plot on slider change
    s_a.on_changed(update)
    s_b.on_changed(update)

    plt.show()


