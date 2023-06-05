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
        logit_x = np.log(x / (1 - x))  # transforming to logit space
        norm_const = np.prod(1 / (x * (1 - x)))
        pdf_val = norm_const * multivariate_normal.pdf(logit_x, mean=self._mu, cov=self._sigma)
        return pdf_val

    def sample(self, N):
        '''Generates a random sample of size `N`.'''
        sample_norm = np.random.multivariate_normal(self._mu, self._sigma, N)
        sample_logit = scipy.special.expit(sample_norm)
        # normalize to make the samples sum to 1
        sample_logit /= np.sum(sample_logit, axis=1, keepdims=True)
        return sample_logit

def draw_pdf_contours(dist, border=False, nlevels=400, subdiv=8, **kwargs):
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

    plt.tricontourf(trimesh, pvals, nlevels, cmap='jet', **kwargs)
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
    f = plt.figure(figsize=(8, 6))
    mus = [[0.999] * 3,
           [0.5] * 3,
           [0.6, 0.2, 0.999]]
    sigmas = [[0.9] * 3,
              [1.8] * 3,
              [1.2, 0.5, 2.2]]
    for i, (mu, sigma) in enumerate(zip(mus, sigmas)):
        plt.subplot(2, len(mus), i + 1)
        dist = LogitNormal(mu, sigma)
        draw_pdf_contours(dist)
        title = '$\mu$ = (%.3f, %.3f, %.3f)\n$\sigma$ = (%.3f, %.3f, %.3f)' % (tuple(mu) + tuple(sigma))
        plt.title(title, fontdict={'fontsize': 8})
        plt.subplot(2, len(mus), i + 1 + len(mus))
        plot_points(dist.sample(5000))
    plt.savefig('logitnormal_plots.png')
    print('Wrote plots to "logitnormal_plots.png".')

