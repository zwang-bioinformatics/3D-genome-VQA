###################################

import numpy as np
from numba import njit
from sklearn import metrics

###################################

unit_length = 1
tns = unit_length * (1 / (3**0.5)) * np.array([
    [1,1,1], [-1,-1,-1],
    [-1,1,1], [1,-1,-1],
    [1,-1,1], [-1,1,-1],
    [1,1,-1], [-1,-1,1]
])

###################################

@njit
def collapse_to_coord(translations):
    out = np.zeros(translations.shape)
    out[:2] = translations[:2]
    for i in range(2,len(translations)): out[i] = out[i-1] + translations[i]
    return out

@njit
def fetch_distogram(struct): 
    out = np.zeros(((struct.shape[0] - 1) * (struct.shape[0] - 2)) // 2)
    c = 0
    for i in range(struct.shape[0]):
        for j in range(i + 2, struct.shape[0]): 
            out[c] = np.sqrt(np.sum(np.square(struct[i] - struct[j]), axis=0))
            c += 1
    return out

###################################

def state2struct(state, fmt, length):
    translations = np.zeros((length,3))
    translations[1] = tns[0]
    bits = np.array(list(map(int,fmt.format(state))))[::-1] 
    spins = 1 - 2*bits
    for i in range(2, length): 
        x_i = 0*(length - 2) + (i - 2)
        y_i = 1*(length - 2) + (i - 2)
        z_i = 2*(length - 2) + (i - 2)
        translations[i] = unit_length * (1 / (3**0.5)) * np.array([spins[x_i], spins[y_i], spins[z_i]])
    return collapse_to_coord(translations)

###################################

def sampled_metrics(states, pos_msk, n_c, fmt, length, inferred_power = 3):   

    states, counts = np.unique(states.astype(int), return_counts=True)

    avg_inferred = 0
    avg_dice_max = 0
    avg_w_dice = 0
    max_inferred = None
    cnt = 0

    empirical = []
    pred_inferred = []
    truths = []

    empirical_shannon_entropy = 0

    for i, state in enumerate(states): 

        distogram = fetch_distogram(state2struct(state, fmt, length))

        empirical_shannon_entropy += - (counts[i] / 4096) * (np.log(counts[i] / 4096) / np.log(2))

        if np.sum(distogram < 1/4): continue # clash
        
        infered_pi_c = np.array([
            1 if d <= 1.5 else (1.5 / d)**inferred_power
            for d in distogram
        ])

        cnt += counts[i]
        infer = np.mean(infered_pi_c[pos_msk])
        avg_inferred += counts[i] * infer

        if max_inferred is None: max_inferred = infer
        elif infer > max_inferred: max_inferred = infer

        dice_max = None
        for t_i in np.unique(infered_pi_c): 
            pi_c_thresh = infered_pi_c > t_i
            intersect_i = np.sum(pi_c_thresh[pos_msk])
            dice_i = 2*intersect_i/(n_c + np.sum(pi_c_thresh))

            if dice_max is None or dice_i > dice_max[1]: dice_max = [t_i, dice_i]

        avg_dice_max += counts[i]*dice_max[1]
        avg_w_dice += counts[i]*2*np.sum(infered_pi_c[pos_msk]) / (n_c + np.sum(infered_pi_c))

        empirical += [counts[i], counts[i]]
        pred_inferred += [infer, np.mean(infered_pi_c[~pos_msk])]
        truths += [1, 0]

    ##################

    return {
        "avg_inferred": avg_inferred / cnt,
        "max_inferred": max_inferred,
        "avg_w_dice": avg_w_dice / cnt,
        "avg_dice_max": avg_dice_max / cnt,
        "inferred_auc": max([0.5, metrics.roc_auc_score(truths, pred_inferred, sample_weight = empirical)]),
        "inferred_ap": metrics.average_precision_score(truths, pred_inferred, sample_weight = empirical),
        "empirical_shannon_entropy": empirical_shannon_entropy
    }

    ##################

###################################

def p_x_ref(r, x2, pi_c, m_x, epsilon = 0.0001):

    # proportional to the probability of a distance of x2 occuring given pi_c and m_x
    # x2 = distance squared (E[i,j])
    # pi_c = contact probability
    # m_x = maximum distance (|i - j|)

    if pi_c == 0: 
        # neutral
        g = 4*(x2 - r**2)
        g = np.clip(g, a_min = np.min(g), a_max = 700) # prevent overflow
        return 1 - (1 / (1 + np.exp(g)))

    elif pi_c == 1: 
        # attract
        pi_x_c = pi_c / (np.clip(x2, a_min = epsilon, a_max = m_x**2) / (m_x**2))
        px_raw = np.clip(1 - (np.abs(x2 - r**2) / (m_x**2 - r**2)), a_min = 0, a_max = 1)

        return np.power(px_raw, pi_x_c)

    else: 
        assert False, "NOT IMPLEMENTED!"

###################################

def spatial_distance_rmsd(distogram, matrix, unit, p, scale_factor = None): 

    if not scale_factor: 
        scale_factor = (np.dot((matrix / unit)**(-p), distogram)) / np.sum(np.square(distogram))

    rmsd = sum(
        np.square(scale_factor*distogram[idx] - (matrix[idx] / unit)**(-p)) 
        for idx in range(matrix.shape[0])
    )

    rmsd = (rmsd / matrix.shape[0]) ** (0.5)
    
    return rmsd, scale_factor

###################################
