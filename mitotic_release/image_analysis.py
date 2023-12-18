from mitotic_release import Defaults
from mitotic_release.data_structure import MetaData, ExpPaths
from mitotic_release.flatfield_corr import flatfieldcorr
from mitotic_release.general_functions import save_fig, filter_segmentation, omero_connect, scale_img
from pathlib import Path
from csbdeep.utils import normalize
from skimage import measure, filters
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from tqdm import tqdm
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
        self.exp_data = self._meta_data.well_conditions(self._well.getId())
        row_list = list('ABCDEFGHIJKL')
        self.well_pos = f"{row_list[self._well.row]}{self._well.column+1}"
        self.image_id = self._omero_image.getId()
        self.mi_model = self._get_mi_model()
        
    @staticmethod
    def _get_mi_model():
        if Defaults.MAGNIFICATION == '20x':
            print("Using 20x model")
            mi_model_path = Path('../MI_Classification/CNN_Training/TrainingData/MI_CNN_model20x.h5')
        else:
            print("Using 10x model")
            mi_model_path = Path('../data/MI_model/mi_model01.h5')
        return keras.models.load_model(mi_model_path)

    def mitotic_index(self):
        df_mi = pd.DataFrame()
        pixels = self._omero_image.getPrimaryPixels()
        t = self._omero_image.getSizeT()
        time_points = [0, 2, 5, 15] if t > 1 else [0]
        flatfield_mask = list(self._flatfield_dict.values())[0]
        for time in tqdm(range(t)):
            img = pixels.getPlane(0, 0, time) / flatfield_mask
            scaled = scale_img(img)
            blurred_img = filters.gaussian(scaled, sigma=4)
            mask = self.segment_stardist(blurred_img)
            print(np.max(mask))
            dict_mit_index = {
                'well': self.well_pos,
                'row': self._well.row,
                'column': self._well.column,
                'image': self._omero_image.getId(),
                'cell_line': self.exp_data['cell_line'],
                'siRNA': self.exp_data['condition'],
                '1NM_conc': self.exp_data['1NM_uM'],
                'time': time * 10,
                'MI': [],
                'cell_count': []
            }
            nuclei_data = self.analyse_image(img, mask) # generates a dictionary of nuclei images and coordiates
            if time in time_points:
                fig, ax = plt.subplots(ncols=len(time_points), figsize=(10 * len(time_points), 10))
                if len(time_points) == 1:
                    ax = [ax]
                fig_number = time_points.index(time)
                ax[fig_number].imshow(scaled, cmap='gray')
                ax[fig_number].axis('off')
                ax[fig_number].title.set_text(f'timepoint: {time * 10}min')
                for number, box in enumerate(nuclei_data['data']):
                    box1 = box[np.newaxis, :, :, np.newaxis]
                    rect = self.generate_patches(box1, nuclei_data['coords'][number])
                    ax[fig_number].add_patch(rect)
            df_timepoint = self.get_mi_df(nuclei_data['data'], dict_mit_index)
            df_mi = pd.concat([df_mi, df_timepoint])
        save_fig(self._paths.segmentation_check, f'{self.well_pos}_{self.image_id}_segmentation_check')
        plt.close()
        return df_mi


    @staticmethod
    def segment_stardist(img):
        """Use stardist model for segmentation and return masks, colored masks and cell counts for Figure"""
        label_objects, nb_labels = Defaults.STARDIST_MODEL.predict_instances(normalize(img))
        return filter_segmentation(label_objects)

    @staticmethod
    def analyse_image(img, mask):
        dict_1 = {'data': [], 'coords': []}
        w = 20 if Defaults.MAGNIFICATION == '20x' else 10
        for region in measure.regionprops(mask):
            b = img[region.bbox[0]:region.bbox[2], region.bbox[1]:region.bbox[3]]
            centroid = region.centroid
            i = centroid[0]
            j = centroid[1]
            imin = int(round(max(0, i - w)))
            imax = int(round(min(mask.shape[0], i + w + 1)))
            jmin = int(round(max(0, j - w)))
            jmax = int(round(min(mask.shape[1], j + w + 1)))
            coords = [imin, imax, jmin, jmax]
            box = img[imin:imax, jmin:jmax]
            if box.shape == (41, 41):
                box1 = box[:, :, np.newaxis]
                dict_1['data'].append(box1)
                dict_1['coords'].append(coords)
        return dict_1

    def generate_patches(self, box, coords):
        imin, imax, jmin, jmax = coords

        proba_1 = self.mi_model.predict(box, verbose=0)

        if proba_1 >= 0.5:
            return mpl.patches.Rectangle((jmin, imin), jmax - jmin, imax - imin,
                                         fill=False, edgecolor='red', linewidth=1)
        else:
            return mpl.patches.Rectangle((jmin, imin), jmax - jmin, imax - imin,
                                         fill=False, edgecolor='white', linewidth=1)

    def get_mi_data(self, data_01, dict_mit_index):
        predict_1 = self.mi_model.predict(np.array(data_01), verbose=0)
        count = np.count_nonzero(predict_1 >= 0.5)
        mit_index = count / len(predict_1) * 100
        dict_mit_index['MI'] = round(mit_index, 2)
        dict_mit_index['cell_count'] = len(predict_1)

    def get_mi_df(self, nuclei_data, dict_mit_index):
        if len(np.array(nuclei_data).shape) == 4:
            self.get_mi_data(np.array(nuclei_data), dict_mit_index)
        else:
            dict_mit_index['MI'] = np.NaN
            dict_mit_index['cell_count'] = np.NaN
        return pd.DataFrame \
            .from_dict(dict_mit_index, orient='index').T


if __name__ == "__main__":
    @omero_connect
    def feature_extraction_test(conn=None):
        meta_data = MetaData(201, conn)
        exp_paths = ExpPaths(meta_data)
        well = conn.getObject("well", 251)
        omero_image = conn.getObject("Image", 1427)
        flatfield_dict = flatfieldcorr(meta_data, exp_paths)
        image = Image(well, omero_image, meta_data, exp_paths, flatfield_dict)
        df_final = image.mitotic_index()
        print(df_final)


    feature_extraction_test()
