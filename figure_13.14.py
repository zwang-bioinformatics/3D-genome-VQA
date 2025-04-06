###################################

import os
os.environ['LD_LIBRARY_PATH'] = "/home/asiciliano/anaconda3/envs/analysis/lib"

import string
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from util import *

import cv2
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import scienceplots
import lovelyplots

###################################

data_src = "/home/asiciliano/3D-genome-VQA/data/experimental_contact_data/"
save_dir = "/home/asiciliano/3D-genome-VQA/figures/"

color_map = {
    "Ideal (Simulator)" : "#6EA6CD",
    "Ideal (Device)" : "#364B9A",
    "Noise (Simulator)" : "#F67E4B",
    "Noise (Device)" : "#DD3D2D"
}

data = {

    "backend": [],
    'lambda': [],
    r'$\lambda$': [],

    'alpha': [],
    r'$\alpha$': [],

    "avg_inferred": [],
    "max_inferred": [],
    "avg_w_dice": [],
    "avg_dice_max": [],
    "inferred_auc": [],
    "inferred_ap": [],
    "empirical_shannon_entropy": [],
}

###################################

for l in os.listdir(data_src): 
    lm = tuple(map(int,l.split("-")[1].split('.')))
    pwr = lm[0] / lm[1]

    pi_c = np.load(data_src + l + "/pi_c.npy")
    pos_msk = pi_c == 1

    for param_source in os.listdir(data_src + l + "/"):
        if param_source == "pi_c.npy": continue
        
        for alpha in os.listdir(data_src + l + "/" + param_source + "/"):

            for trial in tqdm(os.listdir(data_src + l + "/" + param_source + "/" + alpha + "/")):

                src_dir = data_src + l + "/" + param_source + "/" + alpha + "/" + trial + "/"

                simulator = sampled_metrics(
                    np.load(src_dir + "sampled_states.npy"),
                    pos_msk,
                    np.sum(pos_msk),
                    '{:018b}', 8,
                    inferred_power = 1 / pwr
                )

                device = sampled_metrics(
                    np.load(src_dir + "sampled_states_sherbrooke.npy"),
                    pos_msk,
                    np.sum(pos_msk),
                    '{:018b}', 8,
                    inferred_power = 1 / pwr
                )

                for m, backend in [(simulator, param_source.title() + " (Simulator)"), (device, param_source.title() + " (Device)"), ]:

                    for u in m: 
                        data[u] += [m[u]]

                    data[r'$\lambda$'] += ["$\\frac{" + str(lm[0]) + "}" + "{" + str(lm[1]) + "}$"]
                    data['lambda'] += [pwr]
                    data['alpha'] += [float(alpha.split("-")[1])]
                    data[r'$\alpha$'] += [r'$\alpha = ' + str(alpha.split("-")[1]) + '$']
                    data['backend'] += [backend]

data = pd.DataFrame(data)

print(data)

###################################
with plt.style.context(["nature", "science"]):
    for info in [
        {
            "metric": "inferred_auc",
            "label": "AUC",
        },
        {
            "metric": "inferred_ap",
            "label": "AP",
        },
        {
            "metric": "empirical_shannon_entropy",
            "label": r"$\Tilde{H}$",
        },
        {
            "metric": "avg_inferred",
            "label": r"Avg$(\Tilde{\pi}_c)$ ",
        },

        {
            "metric": "max_inferred",
            "label": r"Max$(\Tilde{\pi}_c)$ ",
        },

    ]:

        legend_font_size = 46
        label_font_size = 50
        tick_fontsize = 36
        alpha_fontsize = 40
        lambda_fontsize = 44
        left_margin = 0.175

        fig, ax = plt.subplots(figsize=(8,9))

        g = sns.catplot(
            data=data, 
            x=r"$\alpha$", 
            y=info["metric"],
            order = [r'$\alpha = 0.0$', r'$\alpha = 1.0$'], 
            hue= "backend", 
            col=r'$\lambda$', 
            col_order = [r"$\frac{1}{3}$", r"$\frac{1}{2}$", r"$\frac{2}{3}$"],
            hue_order = ["Ideal (Simulator)", "Ideal (Device)", "Noise (Simulator)", "Noise (Device)"],

            kind="bar",
            height=9, 
            aspect=0.495,
            capsize=0.5,
            linewidth=2.5,
            palette = color_map,
        )

        for i, ax in enumerate(g.axes.flat):
            label_key = ["Ideal (Simulator)", "Ideal (Device)", "Noise (Simulator)", "Noise (Device)"]
            x_labels = [t.get_text() for t in ax.get_xticklabels()]
            b = 0

            lines = {
                k: [[None,None], [None,None]]
                for k in label_key
            }

            for bar in sorted(ax.patches, key =lambda p: p.get_x()):
                if bar.get_label() != '_nolegend_': continue
                
                backend_label = label_key[b % len(label_key)]

                lines[label_key[b % len(label_key)]][0][b // len(label_key)] = bar.get_x() + bar.get_width() / 2 # x
                lines[label_key[b % len(label_key)]][1][b // len(label_key)] = bar.get_height()  # y
                b += 1
            
            for line in lines: 
                if not all(lines[line][d][0] and lines[line][d][1] for d in range(2)): continue
                ax.plot(lines[line][0], lines[line][1], marker='o', color=color_map[line], linewidth = 3.5, zorder=-1)

        sns.move_legend(g, 
            3, bbox_to_anchor=(left_margin / 2, 1, 1 - left_margin / 2, .102),
            ncol = 2,
            title = None,
            frameon=False,
            fontsize = legend_font_size,
            borderaxespad = 0,
            mode = "expand",
        )

        for spine in g.axes.flat[0].spines.values(): spine.set_linewidth(3)

        if info["metric"] not in ["empirical_shannon_entropy"]: g.axes.flat[0].set_yscale('log')

        g.axes.flat[0].tick_params(left=True, right = False, top = False, bottom=True, which='both')
        g.axes.flat[0].tick_params(axis='y', labelsize=tick_fontsize, pad = tick_fontsize / 3, width = 0, length = 0, which = 'both')

        g.axes.flat[0].set_ylabel(
            info["label"], 
            position=(0, 0.5), 
            ha='right', 
            va='center', 
            fontsize = label_font_size,
            rotation = 0,
            labelpad = label_font_size*1.5
        )
        g.axes.flat[0].yaxis.set_label_coords(-2*left_margin, 0.5)

        sns.despine(ax=g.axes.flat[0], left = False, right = True, bottom = True, top = True)

        for ax_ in g.axes.flat[1:]:
            if info["metric"] not in ["empirical_shannon_entropy", "empirical_shannon_entropy_filt"]: ax_.set_yscale('log')
            sns.despine(ax=ax_, left=True, right = True, bottom = True, top = True)
            ax_.tick_params(left=False, right = False, top = False, bottom=False, which='both')

        for ax_ in g.axes.flat:
            ax_.tick_params(axis='x', labelsize=alpha_fontsize, pad = alpha_fontsize / 3, rotation = -22.5)
            if info["metric"] in ["empirical_shannon_entropy"]: ax_.set_ylim(None, math.log2(4096)) # , "empirical_shannon_entropy_filt"
            else: ax_.set_ylim(None, 1)
            
            ax_.set_xlabel(ax_.get_title(), fontsize=lambda_fontsize, labelpad = lambda_fontsize / 2)
            ax_.set_title('')
            ax_.yaxis.minorticks_off()
            ax_.yaxis.set_major_locator(ticker.MaxNLocator(6, prune='both'))
            ax_.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:0.2f}'))

        plt.tight_layout()
        plt.subplots_adjust(left=left_margin)

        if info["metric"] not in ["empirical_shannon_entropy"]: 
            plt.savefig(
                save_dir + f"sub_figures/experimental_{info["metric"]}.png", 
                dpi=300, transparent=False, 
                bbox_extra_artists=[g._legend],
            )
            plt.close()
        else: 
            plt.savefig(
                save_dir + f"experimental_{info["metric"]}.png", 
                dpi=300, transparent=False, 
                bbox_extra_artists=[g._legend],
            )
            plt.close()       

###################################

with plt.style.context(["nature", "science"]):

    file_path = save_dir + "quadrant_cooler.png"

    rows, columns = 2, 2
    fig = plt.figure(figsize=(32,25))

    for i,image_info in enumerate([
        ("",save_dir + "sub_figures/experimental_inferred_auc.png"),
        ("",save_dir + "sub_figures/experimental_inferred_ap.png"),
        ("",save_dir + "sub_figures/experimental_avg_inferred.png"),
        ("",save_dir + "sub_figures/experimental_max_inferred.png"),

    ]): 

        label, file = image_info
        ax = fig.add_subplot(rows, columns, i + 1) 
        ax.autoscale(tight=True)        
        plt.imshow(cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)) 
        plt.axis('off') 
        plt.title("("+string.ascii_uppercase[i]+") " + label, fontsize=75, pad=50, loc='left') 
    
        plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(file_path, dpi=300, bbox_inches = 'tight', transparent = False)
    plt.clf()
    plt.close()
