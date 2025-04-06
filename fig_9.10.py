import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn import metrics
import cv2
import string
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import scienceplots
import lovelyplots

from util import *

data_dir = "/home/asiciliano/3D-genome-VQA/data/simulated_contact_data/"
save_dir = "/home/asiciliano/3D-genome-VQA/figures/"

data = {

    "group": [],
    "alpha": [],
    "length": [],

    "avg_dice_max": [],
    "avg_w_dice": [],
    "max_inferred": [],
    "avg_inferred": [],

    "inferred_auc": [],
    "inferred_ap": []
}

for length in sorted(map(int,os.listdir(data_dir))):

    ##################

    full_filt = np.array([[i,j] for i in range(length) for j in range(i + 2, length)])

    fmt = '{:0' + str(3*(length - 2)) + 'b}'

    for group in tqdm(os.listdir(f"{data_dir}{length}/groups/"), desc = f"loading data (N = {length})"): 

        group_set = frozenset(map(lambda g: tuple(sorted(map(int,g.split(".")))), group[2:].split("_")))

        pos_msk = np.array([
            tuple(sorted([int(pair[0]), int(pair[1])])) in group_set
            for pair in full_filt
        ])
        n_c = np.sum(pos_msk)

        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]: 

            for trial in os.listdir(f"{data_dir}{length}/groups/{group}/alpha-{alpha}/"): 

                states = np.load(f"{data_dir}{length}/groups/{group}/alpha-{alpha}/{trial}/sampled_states.npy")
                states, counts = np.unique(states.astype(int), return_counts=True)
                
                sm = sampled_metrics(states, pos_msk, n_c, fmt, length, inferred_power = 3)  

                for m in sm: 
                    if m == "empirical_shannon_entropy": continue
                    data[m] += [sm[m]]

                data["group"] += [group]
                data["alpha"] += [alpha]
                data["length"] += [str(length)]

                # print(sm)

                # avg_inferred = 0
                # avg_dice_max = 0
                # avg_w_dice = 0
                # max_inferred = None
                # cnt = 0

                # empirical = []
                # pred_inferred = []
                # truths = []

                # for i, state in enumerate(states): 

                #     distogram = fetch_distogram(state2struct(state, fmt, length))

                #     if np.sum(distogram < 1/4): continue
                    
                #     infered_pi_c = np.array([
                #         1 if d <= 1.5 else (1.5 / d)**3
                #         for d in distogram
                #     ])

                #     cnt += counts[i]
                #     infer = np.mean(infered_pi_c[pos_msk])
                #     avg_inferred += counts[i] * infer

                #     if max_inferred is None: max_inferred = infer
                #     elif infer > max_inferred: max_inferred = infer

                #     dice_max = None
                #     for t_i in np.unique(infered_pi_c): 
                #         pi_c_thresh = infered_pi_c > t_i
                #         intersect_i = np.sum(pi_c_thresh[pos_msk])
                #         dice_i = 2*intersect_i/(n_c + np.sum(pi_c_thresh))

                #         if dice_max is None or dice_i > dice_max[1]: dice_max = [t_i, dice_i]

                #     avg_dice_max += counts[i]*dice_max[1]
                #     avg_w_dice += counts[i]*2*np.sum(infered_pi_c[pos_msk]) / (n_c + np.sum(infered_pi_c))

                #     empirical += [counts[i], counts[i]]
                #     pred_inferred += [infer, np.mean(infered_pi_c[~pos_msk])]
                #     truths += [1, 0]
            
                ##################

                # data["avg_inferred"] += [avg_inferred / cnt]
                # data["max_inferred"] += [max_inferred]
                # data["avg_w_dice"] += [avg_w_dice / cnt]
                # data["avg_dice_max"] += [avg_dice_max / cnt]
                # data["inferred_auc"] += [max([0.5, metrics.roc_auc_score(truths, pred_inferred, sample_weight = empirical)])]
                # data["inferred_ap"] += [metrics.average_precision_score(truths, pred_inferred, sample_weight = empirical)]
                # data["group"] += [group]
                # data["alpha"] += [alpha]
                # data["length"] += [str(length)]

                ##################

data = pd.DataFrame(data)

##################

with plt.style.context(["nature", "science", "grid", "colors5-light"]):

    for info in [

        ################

        (r"Avg$(DSI_m)$", "avg_dice_max"),
        (r"Avg$(DSI_w)$", "avg_w_dice"),

        ("AUC", "inferred_auc"),
        ("AP", "inferred_ap"),
        
        (r"Avg$(\Tilde{\pi}_c)$", "avg_inferred"),
        (r"Max$(\Tilde{\pi}_c)$", "max_inferred"),

    ]:

        fig, ax = plt.subplots()

        sns.lineplot(
            data = data,
            x = "alpha",  y = info[1], 
            hue = "length", 
            hue_order = ["6", "7", "8", "9", "10"]
        )

        plt.legend(title=r"$N$", bbox_to_anchor=(1.025, 1), loc='upper left', frameon=False, fontsize=11, title_fontsize = 11.5)

        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])

        plt.xlabel(r"$\alpha$", fontsize=12)
        plt.ylabel(info[0], fontsize=12)

        ax.autoscale(tight=True)
        plt.margins(y = 0.05, x = 0.05)
        
        plt.tight_layout()

        fl_name = f"{info[1]}.png"
                        
        plt.savefig(save_dir + "sub_figures/" + fl_name, dpi=300, transparent=False)
        plt.close()


##################


with plt.style.context(["nature", "science"]):

    rows, columns = 1, 2

    for pairing in [
        (
            "full_inferred_dice", "avg_w_dice", "avg_dice_max"
        )
    ]:

        fig = plt.figure(figsize=(25, 15)) 
        for i, fl in enumerate([
            save_dir + "sub_figures/" + f"{pairing[1]}.png",
            save_dir + "sub_figures/" + f"{pairing[2]}.png"
        ]): 
            fig.add_subplot(rows, columns, i + 1) 
            plt.imshow(cv2.cvtColor(cv2.imread(fl), cv2.COLOR_BGR2RGB))
            plt.axis('off') 
            plt.title("("+string.ascii_uppercase[i]+")", fontsize=50, pad=35, loc='left') 

        plt.tight_layout()
        plt.savefig(save_dir + f"{pairing[0]}.png", dpi=300, transparent=False)
        plt.clf()
        plt.close()

    ################

###################################

with plt.style.context(["nature", "science"]):

    rows, columns = 2, 2

    for quadrant in [
        (
            "quadrant_inferred", 
            "inferred_auc", "inferred_ap",
            "avg_inferred", "max_inferred"
        )
    ]:

        fig = plt.figure(figsize=(25, 20)) 
        for i, fl in enumerate([
            save_dir + "sub_figures/" + f"{quadrant[1]}.png",
            save_dir + "sub_figures/" + f"{quadrant[2]}.png",
            save_dir + "sub_figures/" + f"{quadrant[3]}.png",
            save_dir + "sub_figures/" + f"{quadrant[4]}.png",

        ]): 
            fig.add_subplot(rows, columns, i + 1) 
            plt.imshow(cv2.cvtColor(cv2.imread(fl), cv2.COLOR_BGR2RGB))
            plt.axis('off') 
            plt.title("("+string.ascii_uppercase[i]+")", fontsize=50, pad=35, loc='left') 

        plt.tight_layout()
        plt.savefig(save_dir + f"{quadrant[0]}.png", dpi=300, transparent=False)
        plt.clf()
        plt.close()

    ################