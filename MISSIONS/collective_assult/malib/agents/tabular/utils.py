# Created by yingwen at 2019-03-10
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


FONTSIZE = 18
EPS = 1e-6

# TODO: ADD esp and bolzmann policy here

def U(payoff):
    return payoff[(0, 0)] - payoff[(0, 1)] - payoff[(1, 0)] + payoff[(1, 1)]

def V(alpha, beta, payoff):
    u = U(payoff)
    return alpha * beta * u + alpha * (payoff[(0, 1)] - payoff[(1, 1)]) + beta * (payoff[(1, 0)] - payoff[(1, 1)]) + payoff[(1, 1)]

def projection(PI, threshold=0):
    #print(PI)
    # 1- the unit vector is perpendicular to the 1 surface. Using this along with the
    # 	passed point P0 we get a parametric line equation:
    #   P = P0 + t * I, where t is the parameter and I is the unit vector.
    # 	the unit of projection has \sum P = 1 = \sum P0 + nt, where n is the dimension of P0
    # 	hence the point of projection, P' = P0 + ( (1 - \sum P0) / n ) I
    # * compute sum
    t = sum(PI)
    #print(t)
    # * compute t
    t = (1.0 - t) / len(PI)

    # * compute P'
    for i in range(len(PI)):
        PI[i] += t

    # 2- if forall p in P', p >=0 (and consequently <=1), we found the point.
    #	other wise, pick a negative dimension d, make it equal zero while decrementing
    #	other non zero dimensions. repeat until no negatives remain.
    done = False
    while not done:
        # comulate negative dimensions
        # and count positive ones. note that there must be at least
        # one positive dimension
        n = 0
        excess = 0
        for i in range(len(PI)):
            if PI[i] < threshold:
                excess += threshold-PI[i]
                PI[i] = threshold
            elif PI[i] > threshold:
                n += 1

        # none negative? then done
        if excess == 0:
            done = True
        else:
            # otherwise decrement by equal steps
            for i in range(len(PI)):
                if PI[i] > threshold:
                    PI[i] -= excess / n
    #print(PI)
    return PI


def makehash():
    import collections
    return collections.defaultdict(makehash)


def sigmoid(x, derivative=False):
  return x * (1 - x) if derivative else np.clip(1./(1.+np.exp(-x)),EPS, 1-EPS)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def chain_files(file_names):
    for file_name in file_names:
        with open(file_name) as f:
            for line in f:
                yield line


def drange(start=0.0, stop=1.0, step=0.1):
    eps = 1.0e-6
    r = start
    while r < stop + eps if stop > start else r > stop - eps:
        yield min(max(min(start, stop), r), max(start, stop))
        r += step


def pv(*args, **kwargs):
    import sys
    import inspect
    import pprint

    for name in args:
        record = inspect.getouterframes(inspect.currentframe())[1]
        frame = record[0]
        val = eval(name, frame.f_globals, frame.f_locals)

        prefix = kwargs['prefix'] if 'prefix' in kwargs else ''
        iostream = sys.stdout if 'stdout' in kwargs and kwargs['stdout'] \
            else sys.stderr

        print('%s%s: %s' % (prefix, name, pprint.pformat(val)), file=iostream)


def weighted_mean(samples, weights):
    return sum(x * w for x, w in zip(samples, weights)) / sum(weights) \
        if sum(weights) > 0.0 else 0.0


def mean(samples):
    return sum(samples) / len(samples) if len(samples) else 0.0


def flatten(x):
    return [y for l in x for y in flatten(l)] if type(x) is list else [x]


def forward(*args):
    print('\t'.join(str(i) for i in args))


def random_seed(seed):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)


def minmax(low, x, high):
    return min(max(low, x), high)

def timeit(func):
    import functools

    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        import time

        startTime = time.time()
        func(*args, **kwargs)
        elapsedTime = time.time() - startTime
        print('function [{}] finished in {} s'.format(
            func.__name__, elapsedTime))
    return newfunc


def plot_dynamics(history_pi_0, history_pi_1, pi_alpha_gradient_history, pi_beta_gradient_history, title=''):
    cmap = plt.get_cmap('viridis')
    colors = range(len(history_pi_1))
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)

    scatter = ax.scatter(history_pi_0, history_pi_1, c=colors, s=1)
    ax.scatter(0.5, 0.5, c='r', s=10., marker='*')
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label('Iterations', rotation=270, fontsize=FONTSIZE)

    skip = slice(0, len(history_pi_0), 5)
    ax.quiver(history_pi_0[skip],
              history_pi_1[skip],
              pi_alpha_gradient_history[skip],
              pi_beta_gradient_history[skip],
              units='xy', scale=10., zorder=3, color='blue',
              width=0.007, headwidth=3., headlength=4.)

    ax.set_ylabel("Policy of Player 2", fontsize=FONTSIZE)
    ax.set_xlabel("Policy of Player 1", fontsize=FONTSIZE)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.set_xticklabels(map(str, np.arange(0, 11, 2) / 10), fontsize=FONTSIZE-6)
    ax.set_yticklabels(map(str, np.arange(0, 11, 2) / 10), fontsize=FONTSIZE-6)
    ax.set_title(title, fontsize=FONTSIZE+8)
    plt.tight_layout()
    plt.savefig('{}.pdf'.format(title))
    plt.show()


def plot_Q(data, row_labels, col_labels, title, ax=None,
          cbar_kw={}, cbarlabel="", **kwargs):
    fig = plt.figure(figsize=(6.5, 5))
    ax = fig.add_subplot(111)

    im, cbar = heatmap(data, row_labels, col_labels, ax=ax,
                       cmap="YlGn", cbarlabel="Q Values")
    texts = annotate_heatmap(im, valfmt="{x:.1f} t")
    # print(data)
    ax.set_title(title, pad=50)
    ax.xaxis.set_label_position("top")
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Beta')
    fig.tight_layout()
    plt.savefig('{}.pdf'.format(title), bbox_inches='tight', format='pdf')
    plt.show()


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.1f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Arguments:
        im         : The AxesImage to be labeled.
    Optional arguments:
        data       : Data used to annotate. If None, the image's data is used.
        valfmt     : The format of the annotations inside the heatmap.
                     This should either use the string format method, e.g.
                     "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
        textcolors : A list or array of two color specifications. The first is
                     used for values below a threshold, the second for those
                     above.
        threshold  : Value in data units according to which the colors from
                     textcolors are applied. If None (the default) uses the
                     middle of the colormap as separation.

    Further arguments are passed on to the created text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[im.norm(data[i, j]) > threshold])
            text = im.axes.text(j, i, "{0:.1f}".format(data[i, j]), **kw)
            texts.append(text)

    return texts