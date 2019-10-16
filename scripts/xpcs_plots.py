#!/usr/bin/env python

import sys
import numpy
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
import pylab
import h5py

if sys.platform == 'darwin':
    matplotlib.use("macOSX")


def trim_axs(axs, N):
    """little helper to massage the axs list to have correct length..."""
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]


def gen_image(dset, basename, cbar=True, log=False):
    figsize = (numpy.array(dset.shape) / 100.0)[::-1]
    fig = plt.figure()
    fig.set_size_inches(figsize)
    if cbar:
        figsize[1] *= 1.1
        plt.title(basename)
    else:
        plt.axes([0, 0, 1, 1])  # Make the plot occupy the whole canvas
        plt.axis('off')
    if log:
        image_filename = basename + '_log.png'
        im = plt.imshow(dset,norm=LogNorm())
    else:
        image_filename = basename + '.png'
        im = plt.imshow(dset)
    if cbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
    plt.savefig(image_filename, dpi=100)


def plot_pixelSum(xpcs_h5file):
    pixelSum_dset = xpcs_h5file['exchange/pixelSum']
    basename = 'scattering_pattern'
    gen_image(pixelSum_dset, basename)
    gen_image(pixelSum_dset, basename, log=True)
    gen_image(pixelSum_dset, basename + '_pre', cbar=False)
    gen_image(pixelSum_dset, basename + '_pre', cbar=False, log=True)


def plot_intensity_vs_time(xpcs_h5file):
    i_vs_t = xpcs_h5file['exchange/frameSum']
    pylab.plot(i_vs_t[0], i_vs_t[1])
    plt.xlabel("Elapsed Time (s)")
    plt.ylabel("Average Intensity (photons/pixel/frame)")
    plt.title(xpcs_h5file.filename.rstrip('.hdf'))
    plt.savefig('total_intensity_vs_time.png')


def plot_intensity_t_vs_q(xpcs_h5file):
    basename = xpcs_h5file.filename.rstrip('.hdf')
    q = xpcs_h5file['/xpcs/sqlist']
    pmt_t = xpcs_h5file['/exchange/partition-mean-partial']
    markers = ['o', 'x', '+', 'v', '^', '<', '>', 's', 'p', '*', 'h',
    'D']
    n_markers = len(markers)
    n_plots = pmt_t.shape[0]
    fig = plt.figure()
    fig.set_size_inches((8,6.25))
    ax = plt.gca()
    for i in range(0, n_plots):
        if i >= n_markers:
            markerfacecolor = 'gray'
            i_marker = i - n_markers
        else:
            markerfacecolor = 'None'
            i_marker = i
        ax.plot(q[0], pmt_t[i], color='k', marker=markers[i_marker],
                    alpha=0.5,
                    markerfacecolor=markerfacecolor,
                    markeredgecolor='k',
                    markersize=4,
                    markeredgewidth=0.3,
                    linestyle = 'None',
                    label='{:d}'.format(i+1))
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel("q (A^-1)")
    ax.set_ylabel("Intensity (photons/pixel/frame)")
    ax.legend(numpoints=1)
    plt.title('{} Intensity Mean Partial'.format(basename))
    plt.tight_layout()
    plt.savefig('{}_intensity_t.png'.format(basename), dpi=150)

def plot_intensity_vs_q(xpcs_h5file):
    basename = xpcs_h5file.filename.rstrip('.hdf')
    q = xpcs_h5file['/xpcs/sqlist']
    pmt = xpcs_h5file['/exchange/partition-mean-total']
    fig = plt.figure()
    fig.set_size_inches((8,6.25))
    plt.loglog(q[0], pmt[0], color='k')
    plt.xlabel("q (A^-1)")
    plt.ylabel("Intensity (photons/pixel/frame)")
    plt.title('{} Intensity Mean Total'.format(basename))
    plt.tight_layout()
    plt.savefig(basename + '_intensity.png', dpi=150)


def plot_g2_all(xpcs_h5file):
    def sfig(bn, gs, ge):
        fig.suptitle('{} Correlations g2 {:d} to {:d}'.format(bn, gs, ge))
        plt.savefig('{}_g2_corr_{:03d}_{:03d}.png'.format(bn, gs, ge), dpi=100)
    
    basename = xpcs_h5file.filename.rstrip('.hdf')
    exp = xpcs_h5file['/measurement/instrument/detector/exposure_period']
    dt = xpcs_h5file['/exchange/tau'][0]*exp[0][0]
    g2_all = xpcs_h5file['/exchange/norm-0-g2']
    g2_err_all = xpcs_h5file['/exchange/norm-0-stderr']
    dynamicQ = xpcs_h5file['/xpcs/dqlist'][0]
    n_plots = dynamicQ.shape[0]

    g_index = 0
    while g_index < n_plots:
        g_start = g_index
        pylab.clf()
        fig, axs = plt.subplots(nrows=3, ncols=3, constrained_layout=True)
        fig.set_size_inches((10,10.25))
        for i in range(0, 3): # x, left-right axis
            for j in range(0, 3): # y, top-down axis
                ax = axs[i,j]
                ax.errorbar(dt, g2_all[:, g_index], yerr=g2_err_all[:, g_index],
                                fmt='ko', fillstyle='none', capsize=2, markersize=5)
                ax.set_title('q={:f}'.format(dynamicQ[g_index]))
                ax.set_yscale('linear')
                ax.set_xscale('log')
                g_index += 1
                if g_index == n_plots:
                    sfig(basename, g_start, g_index - 1)
                    break
        sfig(basename, g_start, g_index - 1)

if __name__ == '__main__':
    h5filename = sys.argv[1]
    print('opening ' + h5filename)
    x_h5_file = h5py.File(h5filename, 'r')
    plot_intensity_vs_time(x_h5_file)
    pylab.clf()
    plot_intensity_vs_q(x_h5_file)
    pylab.clf()
    plot_intensity_t_vs_q(x_h5_file)
    pylab.clf()
    plot_g2_all(x_h5_file)
    pylab.clf()
    plot_pixelSum(x_h5_file)
