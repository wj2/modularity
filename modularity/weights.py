import numpy as np
import scipy.stats as sts

import matplotlib.pyplot as plt
import seaborn as sns
# from pipeline.imports import *


def dic_to_vec(dic):
    vec = []
    for i in [0, 1]:
        for j in [0, 1]:
            for k in [0, 1]:
                if '%u%u%u' % (i, j, k) in dic[0]:
                    vec.append(-1)
                if '%u%u%u' % (i, j, k) in dic[1]:
                    vec.append(1)

    return np.asarray(vec)


def rotate_weights(weights, n_neurons):
    R = sts.special_ortho_group.rvs(n_neurons)
    wn = {}
    for key in weights:
        wn[key] = np.asarray([np.dot(w, R) for w in weights[key]])
    return wn


def SNR(x):
    mean = np.nanmean(x, 0)
    std = np.nanstd(x, 0)
    snr = mean/std

    snr[np.isnan(snr)] = 0
    snr[np.isinf(snr)] = 0

    return snr


def specialization(weights_A, weights_B, n_angles=5):
    thresholds = np.linspace(0, np.pi / 2, n_angles + 1)

    # take the absolute value of the signal-to-noise ratio
    data1 = np.abs(SNR(np.squeeze(weights_A)))
    data2 = np.abs(SNR(np.squeeze(weights_B)))
    # compute fractions in angles and compare it to a null model
    angle_data = np.arctan2(data2, data1)
    lengths = np.sqrt(data1 ** 2 + data2 ** 2)
    fractions, bins = np.histogram(angle_data, thresholds)
    fractions = fractions/np.sum(fractions)

    # compute Spearman correlation
    corr_data = scipy.stats.spearmanr(data1, data2)[0]

    # compute mean Specialization
    spec_data = np.nanmean(np.abs(data1 - data2) / (data1 + data2))

    return fractions, corr_data, spec_data


def specialization_plot(weights, varA, varB, specializations, correlations, fractions, specializations_null, correlations_null, fractions_null, axs=None):
    pair_key = '%s\n%s' % (varA, varB)
    n_angles = len(fractions[pair_key])

    angles = fractions[pair_key]
    angles_null = fractions_null[pair_key]
    spec_data = specializations[pair_key]
    spec_null = specializations_null[pair_key]
    corr_data = correlations[pair_key]
    corr_null = correlations_null[pair_key]

    # plot
    variables = [varA, varB]
    thresholds = np.linspace(0, np.pi / 2, n_angles + 1)
    dx = thresholds[1] - thresholds[0]

    data1 = np.abs(SNR(np.squeeze(weights[varA])))
    data2 = np.abs(SNR(np.squeeze(weights[varB])))

    import matplotlib.cm as cm
    cmap = cm.get_cmap('cool', n_angles)
    if axs is None:
        f, axs = plt.subplots(1, 4, figsize=(12, 3.5), gridspec_kw={'width_ratios': [1, 1, 1.2, 1.2]})

    # first plot: scatter of decoding weights
    n_neurons = len(data1)
    axs[0].scatter(data1, data2, alpha=min(0.8, 40./n_neurons), marker='o', color='k', s=16)
    axs[0].set_xlabel('D.I. %s' % variables[0])
    axs[0].set_ylabel('D.I. %s' % variables[1])

    axs[0].axhline([0], color='k', linestyle='--')
    axs[0].axvline([0], color='k', linestyle='--')
    [xm, xM] = axs[0].get_xlim()
    [ym, yM] = axs[0].get_ylim()
    axs[0].set_xlim([np.min([xm, ym]), np.max([xM, yM])])
    axs[0].set_ylim([np.min([xm, ym]), np.max([xM, yM])])

    # second plot: angle distribution
    for i in range(len(thresholds) - 1):
        if not i:
            axs[1].bar(thresholds[i], angles[i], alpha=0.5, width=dx - 0.05, color=cmap(i), label='Data')
            axs[1].errorbar(thresholds[i], np.nanmean(angles_null, 1)[i], 2 * np.nanstd(angles_null, 1)[i],
                            color='k', linestyle='', capsize=6, marker='_', label='Null')
        else:
            axs[1].bar(thresholds[i], angles[i], alpha=0.5, width=dx - 0.05, color=cmap(i))
            axs[1].errorbar(thresholds[i], np.nanmean(angles_null, 1)[i], 2 * np.nanstd(angles_null, 1)[i],
                            color='k', linestyle='', capsize=6, marker='_')
    axs[1].set_xlabel('Angle $\\theta$')
    axs[1].set_ylabel('Fraction of neurons')
    axs[1].set_xticks([-dx / 2, np.pi / 2 - dx / 2])
    axs[1].set_xticklabels(['0', '$\pi/2$'])
    axs[1].legend()
    sns.despine(ax=axs[0])
    sns.despine(ax=axs[1])

    # third plot: correlation against null model that preserves conditional distributions
    visualize_data_vs_null(corr_data, corr_null, ax=axs[2],
                           value='DI correlation\n%s-%s' % (variables[0], variables[1]))

    axs[2].annotate("", xy=(0.95, 0.3), xytext=(0.7, 0.3),
                    arrowprops=dict(arrowstyle="->", linewidth=3, alpha=0.5, color=pltcolors[1]), va='center',
                    color=pltcolors[1], fontweight='bold', xycoords='axes fraction', textcoords='axes fraction')
    axs[2].text(0.9, 0.32, "Mixed", va='bottom', ha='right', color=pltcolors[1], fontweight='bold',
                transform=axs[2].transAxes)

    axs[2].annotate("", xy=(0.05, 0.3), xytext=(0.3, 0.3),
                    arrowprops=dict(arrowstyle="->", linewidth=3, alpha=0.5, color=pltcolors[2]), va='center',
                    color=pltcolors[2], fontweight='bold', xycoords='axes fraction', textcoords='axes fraction')
    axs[2].text(0.1, 0.32, "Spec.", va='bottom', ha='left', color=pltcolors[2], fontweight='bold',
                transform=axs[2].transAxes)

    # fourth plot: mean specialization against null model that preserves conditional distributions
    visualize_data_vs_null(spec_data, spec_null, ax=axs[3],
                           value='Mean Specialization\n%s-%s' % (variables[0], variables[1]))

    axs[3].annotate("", xy=(0.95, 0.3), xytext=(0.7, 0.3),
                    arrowprops=dict(arrowstyle="->", linewidth=3, alpha=0.5, color=pltcolors[2]), va='center',
                    color=pltcolors[2], fontweight='bold', xycoords='axes fraction', textcoords='axes fraction')
    axs[3].text(0.9, 0.32, "Spec.", va='bottom', ha='right', color=pltcolors[2], fontweight='bold',
                transform=axs[3].transAxes)

    axs[3].annotate("", xy=(0.05, 0.3), xytext=(0.3, 0.3),
                    arrowprops=dict(arrowstyle="->", linewidth=3, alpha=0.5, color=pltcolors[1]), va='center',
                    color=pltcolors[1], fontweight='bold', xycoords='axes fraction', textcoords='axes fraction')
    axs[3].text(0.1, 0.32, "Mixed", va='bottom', ha='left', color=pltcolors[1], fontweight='bold',
                transform=axs[3].transAxes)


def specialization_test(weights, nshuffles=100, plot=True, axs=None):
    # Define the dictionaries we will fill
    correlations = {}
    specializations = {}
    fractions = {}

    correlations_null = {}
    specializations_null = {}
    fractions_null = {}

    variables = list(weights.keys())
    n_neurons = len(weights[variables[0]][0])

    # Data: loop over pairs, if they are orthogonal, compute specialization
    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            pair_key = '%s\n%s' % (variables[i], variables[j])

            fracs, corr, spec = specialization(weights[variables[i]],
                                               weights[variables[j]])

            specializations[pair_key] = spec
            correlations[pair_key] = corr
            fractions[pair_key] = fracs

            specializations_null[pair_key] = []
            correlations_null[pair_key] = []
            fractions_null[pair_key] = []

    # Null model: rotate everyone, repeat [Data]
    for n in range(nshuffles):
        wn = rotate_weights(weights, n_neurons)
        for i in range(len(variables)):
            for j in range(i + 1, len(variables)):
                pair_key = '%s\n%s' % (variables[i], variables[j])
                fracs, corr, spec = specialization(wn[variables[i]], wn[variables[j]])
                specializations_null[pair_key].append(spec)
                correlations_null[pair_key].append(corr)
                fractions_null[pair_key].append(fracs)

    # plots
    nplots = int((len(variables)*(len(variables)-1))/2)
    if plot:
        if axs is None:
            f, axs = plt.subplots(nplots, 4, figsize=(12, 3*nplots), gridspec_kw={'width_ratios': [1, 1, 1.2, 1.2]})
        if nplots == 1:
            axs = np.asarray([axs])

        plot_index = 0
        for i in range(len(variables)):
            for j in range(i+1, len(variables)):
                specialization_plot(weights, variables[i], variables[j],
                                    specializations, correlations, fractions,
                                    specializations_null, correlations_null, fractions_null,
                                    axs=axs[plot_index, :])
                plot_index += 1

    return specializations, correlations, specializations_null, correlations_null
