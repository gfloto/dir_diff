'''Functions for drawing contours of Dirichlet distributions.'''

# Author: Thomas Boggs

from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from scipy.stats import multivariate_normal
import numpy as np
import scipy

_corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
_AREA = 0.5 * 1 * 0.75**0.5
_triangle = tri.Triangulation(_corners[:, 0], _corners[:, 1])

# For each corner of the triangle, the pair of other corners
_pairs = [_corners[np.roll(range(3), -i)[1:]] for i in range(3)]
# The area of the triangle formed by point xy and another pair or points
tri_area = lambda xy, pair: 0.5 * np.linalg.norm(np.cross(*(pair - xy)))

def xy2bc(xy, tol=1.e-4):
    '''Converts 2D Cartesian coordinates to barycentric.

    Arguments:

        `xy`: A length-2 sequence containing the x and y value.
    '''
    coords = np.array([tri_area(xy, p) for p in _pairs]) / _AREA
    return np.clip(coords, tol, 1.0 - tol)

class LogitNormal(object):
    def __init__(self, mu, sigma):
        '''Creates LogitNormal distribution with parameter `mu` and `sigma`.'''
        self._mu = np.array(mu)
        self._sigma = np.diag(sigma)

    def pdf(self, x):
        '''Returns pdf value for `x`.'''
        epsilon = 1e-5
        x = np.clip(x, epsilon, 1 - epsilon)
        x_D = x[-1] # last element of x
        x_without_D = x[:-1] # first D-1 elements of x
        logit_x = np.log(x_without_D / x_D)  # transforming to logit space
        norm_const = 1 / np.prod(x)  # Jacobian determinant
        # multivariate normal pdf
        pdf_val = norm_const * multivariate_normal.pdf(logit_x, mean=self._mu, cov=self._sigma)
        return pdf_val


    def sample(self, N):
        '''Generates a random sample of size `N`.'''
        sample_norm = np.random.multivariate_normal(self._mu, self._sigma, N)
        exp_y = np.exp(sample_norm)
        ones_column = np.ones((N, 1))  # Create a column of ones
        exp_y = np.hstack((exp_y, ones_column))  # Append a column of ones to exp_y
        sample_logit = exp_y / (1 + np.sum(exp_y, axis=1, keepdims=True))
        return sample_logit


def draw_pdf_contours(dist, border=False, nlevels=600, subdiv=8, **kwargs):
    '''Draws pdf contours over an equilateral triangle (2-simplex).

    Arguments:

        `dist`: A distribution instance with a `pdf` method.

        `border` (bool): If True, the simplex border is drawn.

        `nlevels` (int): Number of contours to draw.

        `subdiv` (int): Number of recursive mesh subdivisions to create.

        kwargs: Keyword args passed on to `plt.triplot`.
    '''
    from matplotlib import ticker, cm
    import math

    refiner = tri.UniformTriRefiner(_triangle)
    trimesh = refiner.refine_triangulation(subdiv=subdiv)
    pvals = [dist.pdf(xy2bc(xy)) for xy in zip(trimesh.x, trimesh.y)]

    plt.tricontourf(trimesh, pvals, nlevels, cmap='gist_earth', **kwargs)
    plt.axis('equal')
    plt.xlim(0, 1)
    plt.ylim(0, 0.75**0.5)
    plt.axis('off')
    if border is True:
        plt.triplot(_triangle, linewidth=1)

def plot_points(X, barycentric=True, border=True, **kwargs):
    '''Plots a set of points in the simplex.

    Arguments:

        `X` (ndarray): A 2xN array (if in Cartesian coords) or 3xN array
                       (if in barycentric coords) of points to plot.

        `barycentric` (bool): Indicates if `X` is in barycentric coords.

        `border` (bool): If True, the simplex border is drawn.

        kwargs: Keyword args passed on to `plt.plot`.
    '''
    if barycentric is True:
        X = X.dot(_corners)
    plt.plot(X[:, 0], X[:, 1], 'k.', ms=1, **kwargs)
    plt.axis('equal')
    plt.xlim(0, 1)
    plt.ylim(0, 0.75**0.5)
    plt.axis('off')
    if border is True:
        plt.triplot(_triangle, linewidth=1)

if __name__ == '__main__':
    f = plt.figure(figsize=(6, 3))
    mus = [[0, 0],
           [0.2, 0.35]]
    sigmas = [[0.5, 0.5],
              [0.6, 0.8]]
    for i, (mu, sigma) in enumerate(zip(mus, sigmas)):
        plt.subplot(1, len(mus), i + 1)
        dist = LogitNormal(mu, sigma)
        draw_pdf_contours(dist)
        title = f'$\mu$ = {tuple(mu)}\n$\sigma$ = {tuple(sigma)}'
        plt.title(title, fontdict={'fontsize': 8})
        # plt.subplot(2, len(mus), i + 1 + len(mus))
        # plot_points(dist.sample(5000))
    plt.savefig('logitnormal_plots.png')
    print('Wrote plots to "logitnormal_plots.png".')

