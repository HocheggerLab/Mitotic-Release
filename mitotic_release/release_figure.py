import pandas as pd
import numpy as np
from mitotic_release.general_functions import save_fig
import pathlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import platform
if platform.system() == 'Darwin':
    mpl.use('MacOSX')  # avoid matplotlib warning about interactive backend

plt.style.use('../data/style/HHlab_style01.mplstyle')
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']



def data_prep(df):
    df.loc[df['MI'] == ['NaN'], 'MI'] = np.NaN
    df['MI'] = df['MI'].fillna(method="ffill").astype('float')
    return df

def mitotic_index_plot(df, path, plate_name):
    print("generating figure")
    row_number = df['row'].max() +1
    fig, ax = plt.subplots(ncols=row_number, figsize=(row_number * 3, 3))
    for row in range(row_number):
        df1 = df[(df['row'] == row)]
        sns.lineplot(y='MI',
                     x='time',
                     data=df1,
                     hue='column',
                     palette=[colors[0], colors[2], colors[3], colors[1]],
                     ax=ax[row])
        ax[row].set_title(set(df1['siRNA']).pop())
        ax[row].set_xlabel('Time [min]')
        ax[row].set_ylim(0, 80)
        if row == 0:
            ax[row].legend([0, 0.125, 0.25, 0.5],
                           title='1NMPP1[uM]',
                           bbox_to_anchor=(row_number + 1.8, 1.2))

            ax[row].set_ylabel('% Mitotic Index')
        else:
            ax[row].legend().remove()
            ax[row].set_ylabel(None)
    save_fig(path, f"{plate_name}_Figure", tight_layout=False)


def auc_plot(df, path, plate_name):
    df_grouped = df.groupby(['well', 'row', 'column', 'siRNA', '1NM_conc', 'time']).mean(numeric_only=True).reset_index()
    gene_list = df_grouped['siRNA'].unique()
    value_list = []
    for i in range(len(df_grouped) // 24):
        x = df_grouped.iloc[i * 24:(i + 1) * 24, 5]
        y = df_grouped.iloc[i * 24:(i + 1) * 24, 7]
        area = np.trapz(y, x=x, axis=- 1)
        value_list.append(area)

    df_auc = pd.DataFrame()
    df_auc['auc_values'] = value_list
    df_auc['siRNA'] = [cond for cond in gene_list for i in range(4)]
    df_auc['concentration'] = [0, 0.125, 0.25, 0.5] * (len(value_list) // 4)

    fig, ax = plt.subplots()
    sns.barplot(x='siRNA', y='auc_values', hue='concentration', data=df_auc,
                palette=[colors[0], colors[2], colors[3], colors[1]])
    ax.set_title(f"{plate_name}_AUC curves", fontsize=12)
    ax.set_xticklabels(gene_list, rotation=30, ha='right')
    ax.set_ylabel('Area under Curve')
    ax.legend(title='1NMPP1[uM]', bbox_to_anchor=(1.05, 0.6))
    save_fig(path, f"{plate_name}_AUC_Figure", tight_layout=False)



# test
if __name__ == '__main__':
    path = pathlib.Path.home() / 'Desktop'
    df = pd.read_csv('../data/sample.csv', index_col=0)
    mitotic_index_plot(data_prep(df), path, 'sample_figure')
    auc_plot(data_prep(df), path, 'sample_AUC_figure')

