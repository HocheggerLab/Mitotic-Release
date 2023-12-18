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
plt.style.use('./HHlab_style01.mplstyle')
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']



def data_prep(df):
    for image in df.image.unique():
        if df.loc[df['image'] == image, 'MI'].isna().sum() > 5:
            return df.drop(df.loc[df['image'] == image].index)
        df.loc[df['MI'] == 'NaN', 'MI'] = np.NaN
        df['MI'] = df['MI'].fillna(method="ffill").astype('float')
        return df

def mitotic_index_plot(df, path):
    for condition in df.condition.unique():
        fig, ax = plt.subplots(figsize=(3, 3))
        df1 = df.loc[(df['condition'] == condition)]
        sns.lineplot(y='MI',
                     x='time',
                     data=df1,
                     hue='1NM_conc',
                     palette=[colors[0], colors[2], colors[3], colors[1]],
                     ax=ax)
        ax.set_title(condition)
        ax.set_xlabel('Time [min]')
        ax.set_ylim(0, 90)
        ax.legend([0, 0.125, 0.25, 0.5],
                           title='1NMPP1[uM]',
                           bbox_to_anchor=(1, 1))

        ax.set_ylabel('% Mitotic Index')
        save_fig(path, f"{condition}_Figure", tight_layout=False)






# test
if __name__ == '__main__':
    path = pathlib.Path.home() / 'Desktop/MR_assay_B55_OA_TM_HeLa/mitotic_index_data'
    df = pd.read_csv('/Users/hh65/Desktop/MR_assay_B55_OA_TM_HeLa/mitotic_index_data/MR_assay_B55_OA_TM_HeLa_final_data.csv')
    mitotic_index_plot(data_prep(df), path)


