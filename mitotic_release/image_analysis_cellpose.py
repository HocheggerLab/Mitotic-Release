from mitotic_release import Defaults
from mitotic_release.data_structure import MetaData, ExpPaths
from mitotic_release.flatfield_corr import flatfieldcorr
from mitotic_release.general_functions import save_fig, generate_image, filter_segmentation, omero_connect, scale_img, \
    color_label
from  pathlib import Path
from cellpose import models

from skimage import measure, io
import pandas as pd
import numpy as np
import keras
from tqdm import tqdm
from operator import itemgetter
import matplotlib as mpl
import matplotlib.pyplot as plt



class Image:
    """
    generates the corrected images and segmentation masks.
    Stores corrected images as dict, and n_mask, c_mask and cyto_mask arrays.
    """

    def __init__(self, well, omero_image, meta_data, exp_paths, flatfield_dict):

        self._well = well
        self._omero_image = omero_image
        self._meta_data = meta_data
        self._paths = exp_paths
        self._get_metadata()
        self._flatfield_dict = flatfield_dict


    def _get_metadata(self):
        self.channels = self._meta_data.channels
        self.cell_line = self._meta_data.well_conditions(self._well.getId())['Cell_Line']
        row_list = list('ABCDEFGHIJKL')
        self.well_pos = f"{row_list[self._well.row]}{self._well.column}"
        self.image_id = self._omero_image.getId()
        self.mi_model = self._get_mi_model()
        self.model = models.CellposeModel(gpu=False, model_type=Defaults.CELLPOSE_MODEL)


    def _get_mi_model(self):
        mi_model_path = Path('../data/MI_model/mi_model01.h5')
        return keras.models.load_model(mi_model_path)

    def mitotic_index(self):
        df_MI = pd.DataFrame()
        pixels = self._omero_image.getPrimaryPixels()
        time_points = [3, 5]
        fig, ax = plt.subplots(ncols= len(time_points), figsize=(40, 10))
        for time in tqdm(range(self._omero_image.getSizeT())):
            img = pixels.getPlane(0, 0, time) / list(self._flatfield_dict.values())[0]
            scaled = scale_img(img)
            mask = self.segment_cellpose(scaled)
            dict_mit_index = {
                'well': self.well_pos,
                'image': self._omero_image.getId(),
                'time': time*10,
                'MI': [],
                'cell_count': []
                        }
            nuclei_data = self.analyse_image(img, mask)

            if time in time_points:
                fig_number = time_points.index(time)
                ax[fig_number].imshow(scaled, cmap='gray')
                ax[fig_number].axis('off')
                ax[fig_number].title.set_text(f'timepoint: {time*10}min')
                for number, box in enumerate(nuclei_data['data']):
                    box1 = box[np.newaxis, :, :, np.newaxis]
                    minr, maxr, minc, maxc = nuclei_data['coords'][number]
                    proba_1 = self.mi_model.predict(box1, verbose=0)
                    predict_1 = np.round(proba_1).astype('int')
                    if predict_1 == 1:
                        rect = mpl.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                                      fill=False, edgecolor='red', linewidth=1)
                        ax[fig_number].add_patch(rect)
                    elif predict_1 == 0:
                        rect = mpl.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                                  fill=False, edgecolor='blue', linewidth=1)
                        ax[fig_number].add_patch(rect)
            df_timepoint = self.get_mi_df(nuclei_data['data'], dict_mit_index)
            df_MI = pd.concat([df_MI, df_timepoint])
        save_fig(Path.home() / 'Desktop','test')
        return df_MI

    def segment_cellpose(self, img):
        """Use stardist model for segmentation and return masks, colored masks and cell counts for Figure"""
        n_channels = [[0, 0]]
        n_masks, n_flows, n_styles = self.model.eval(img, channels=n_channels)
        return filter_segmentation(n_masks)

    @staticmethod
    def analyse_image(img, mask):
        dict_1 = {'data': [], 'coords': []}
        w = 10
        for region in measure.regionprops(mask):
            b = img[region.bbox[0]:region.bbox[2], region.bbox[1]:region.bbox[3]]
            if 10 <= len(b) <= 30:
                centroid = region.centroid
                i = centroid[0]
                j = centroid[1]
                imin = int(round(max(0, i - w)))
                imax = int(round(min(mask.shape[0], i + w + 1)))
                jmin = int(round(max(0, j - w)))
                jmax = int(round(min(mask.shape[1], j + w + 1)))
                box = img[imin:imax, jmin:jmax]
                box1 = box[:, :, np.newaxis]
                if box1.shape == (21, 21, 1):
                    dict_1['data'].append(box1)
                    dict_1['coords'].append([imin, imax, jmin, jmax])

        return dict_1

    def get_mi_data(self, data_01, dict_mit_index):
        predict_1 = self.mi_model.predict(np.array(data_01), verbose=0)
        count = np.count_nonzero(predict_1 >= 0.5)
        mit_index = count / len(predict_1) * 100
        dict_mit_index['MI'] = round(mit_index,2)
        dict_mit_index['cell_count'] = len(predict_1)

    def get_mi_df(self, nuclei_data, dict_mit_index):
        if len(np.array(nuclei_data).shape) == 4:
            self.get_mi_data(np.array(nuclei_data), dict_mit_index)
        else:
            dict_mit_index['MI'].append('NaN')
        return pd.DataFrame \
            .from_dict(dict_mit_index, orient='index') \
            .T \
            .replace("NaN", np.nan) \
            .fillna(method="ffill")



        #         ax[number].axis('off')
        #         ax[number].title.set_text('timepoint ' + str(tp))

# import matplotlib.patches as mpatches
#
# w = 10
# for ID in image_list:
#     object = conn.getObject("Image", ID)
#     tps = [0, 3, 7, 20]
#     pixels = object.getPrimaryPixels()
#     sns.set(font='Arial')
#     fig, ax = plt.subplots(ncols=
#                            len(tps), figsize=(40, 10))
#
#     for number, tp in enumerate(tps):
#         img = pixels.getPlane(0, 0, tp)
#
#         percentiles = np.percentile(img, (1, 99))
#         scaled = exposure.rescale_intensity(img, in_range=tuple(percentiles))
#         Segmented, ColorLabels, Cell_Number = Get_Nuclei(scaled)
#
#         ax[number].imshow(scaled, cmap='gray')
#         ax[number].axis('off')
#         ax[number].title.set_text('timepoint ' + str(tp))
#
#         for region in measure.regionprops(Segmented):
#             b = img[region.bbox[0]:region.bbox[2], region.bbox[1]:region.bbox[3]]
#             if 0 <= len(b) <= 100:
#                 centroid = region.centroid
#                 i = centroid[0]
#                 j = centroid[1]
#                 imin = int(round(max(0, i - w)))
#                 imax = int(round(min(Segmented.shape[0], i + w + 1)))
#                 jmin = int(round(max(0, j - w)))
#                 jmax = int(round(min(Segmented.shape[1], j + w + 1)))
#                 box = img[imin:imax, jmin:jmax]
#                 box1 = box[np.newaxis, :, :, np.newaxis]
#
#                 if box1.shape == (1, 21, 21, 1):
#
#                     proba_1 = model1.predict(box1, verbose=0)
#                     predict_1 = np.round(proba_1).astype('int')
#                     if predict_1 == 1:
#
#                         minr, minc, maxr, maxc = region.bbox
#                         rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
#                                                   fill=False, edgecolor='red', linewidth=1)
#                         ax[number].add_patch(rect)
#                     elif predict_1 == 0:
#
#                         minr, minc, maxr, maxc = region.bbox
#                         rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
#                                                   fill=False, edgecolor='blue', linewidth=1)
#                         ax[number].add_patch(rect)
#     save_fig(f"{object.name}_img")
#     plt.show()



if __name__ == "__main__":
    @omero_connect
    def feature_extraction_test(conn=None):
        meta_data = MetaData(1059, conn)
        exp_paths = ExpPaths(meta_data)
        well = conn.getObject("well", 11553)
        omero_image = conn.getObject("Image", 400921)
        flatfield_dict = flatfieldcorr(meta_data, exp_paths)
        image = Image(well, omero_image, meta_data, exp_paths, flatfield_dict)
        df_final = image.mitotic_index()
        print(df_final)




    feature_extraction_test()
