import numpy as np
import matplotlib.pyplot as plt
import pandas.plotting as pd_plotting

get_key_by_value = lambda d, val: list(d.keys())[list(d.values()).index(val)]


def get_key_by_value_wrapper(d, val):
    try:
        return get_key_by_value(d, val)
    except ValueError:
        try:
            val = val.replace(' ', '_')
            return get_key_by_value(d, val)
        except ValueError:
            for key, value in d.items():
                if val in value:
                    return key
    return 999


def pred_to_logits(pred, dataset):
    verb_hot = np.zeros(len(dataset.verb_map))
    target_hot = np.zeros(len(dataset.target_map))
    instrument_hot = np.zeros(len(dataset.instrument_map) - 1)
    for instance in range(len(pred['instrument'])):
        i_idx = get_key_by_value_wrapper(dataset.instrument_map, pred['instrument'][instance])
        if i_idx == -1:
            return verb_hot, target_hot, instrument_hot
        try:
            v_idx = get_key_by_value_wrapper(dataset.verb_map, pred['verb'][instance])
            t_idx = get_key_by_value_wrapper(dataset.target_map, pred['target'][instance])
        except:
            print('Error in pred_to_logits caused by invalid verb or target')
            return verb_hot, target_hot, instrument_hot
        set_idx(instrument_hot, i_idx)
        set_idx(verb_hot, v_idx)
        set_idx(target_hot, t_idx)
    return verb_hot, target_hot, instrument_hot


def set_idx(v, idx):
    ignore = [999]
    if idx not in ignore:
        v[idx] = 1


def visualize_prediction(write_path, label, pred, read_path):
    plt.imshow(plt.imread(read_path))
    plt.title(f'Pred: {pred}\nLabel: {label}')
    plt.savefig(write_path, bbox_inches='tight')


def print_table(metrics_df, out_name='metrics_table.png'):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')
    table = pd_plotting.table(ax, metrics_df, loc='center', cellLoc='center', colWidths=[0.20] * len(metrics_df.columns))
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.5, 1.5)
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_fontsize(14)
            cell.set_facecolor('#2E7D32')
            cell.set_text_props(weight='bold', color='white')
        elif i == len(metrics_df):
            cell.set_facecolor('#A5D6A7')
            cell.set_text_props(weight='bold')
        else:
            cell.set_facecolor('#E8F5E9')
        cell.set_edgecolor('#E0E0E0')
        cell.set_height(0.15)
    for k, cell in table.get_celld().items():
        if k[1] == 0:
            cell.set_width(0.25)
    plt.savefig(out_name, bbox_inches='tight', dpi=300)
