
from mitotic_release import Defaults
from mitotic_release.data_structure import MetaData, ExpPaths
from mitotic_release.flatfield_corr import flatfieldcorr

from mitotic_release.general_functions import omero_connect, scale_img
import ezomero
import  numpy as np
import napari
from tqdm import tqdm
from skimage import io
from skimage.segmentation import clear_border
from csbdeep.utils import normalize
import gc
from skimage import measure
import keras
import pathlib

PLATE_ID = 1102
IMAGE_ID = 410860
SEGMENTATION_CHANNEL = 0

def mitotic_timecourse(flatfield_dict, img_id, conn):
    om_img, data = ezomero.get_image(conn, img_id)
    image_list = []
    for i in range(len(flatfield_dict)):
        data_corr = data[:, 0, :, :, i] // list(flatfield_dict.values())[i]
        data_corr = data_corr[:, 30:1050, 30:1050].astype('uint16')
        image_list.append(data_corr)
    del data
    gc.collect()
    return image_list


def segment_image(img):
    """Use image from segmentation channel to perform segmentation and return segmentation channel"""
    tc_segmented = []
    for time in tqdm(range(img.shape[0])):
        image_to_segment = img[time, :, :]
        mask = segment_stardist(image_to_segment)
        tc_segmented.append(mask)
    return np.stack(tc_segmented).astype('uint16')


def segment_stardist(img):
    """Use stardist model for segmentation and return masks, colored masks and cell counts for Figure"""
    scaled = scale_img(img, (0.1, 99.9))
    label_objects, nb_labels = Defaults.STARDIST_MODEL.predict_instances(normalize(scaled))
    cleared = clear_border(label_objects)
    sizes = np.bincount(cleared.ravel())
    mask_sizes = (sizes > 20)
    mask_sizes[0] = 0
    return mask_sizes[cleared]

def analyse_image(img, mask):
    dict_1 = {'data': [], 'coords': []}
    w = 10
    for region in measure.regionprops(mask):
        b = img[region.bbox[0]:region.bbox[2], region.bbox[1]:region.bbox[3]]
        print(b.shape)
        if 10 <= len(b) <= 30:
            centroid = region.centroid
            i = centroid[0]
            j = centroid[1]
            imin = int(round(max(0, i - w)))
            imax = int(round(min(mask.shape[0], i + w + 1)))
            jmin = int(round(max(0, j - w)))
            jmax = int(round(min(mask.shape[1], j + w + 1)))
            coords = [imin, jmin, imax, jmax]
            box = img[imin:imax, jmin:jmax]
            if box.shape == (21, 21):
                box1 = box[:, :, np.newaxis]
                dict_1['data'].append(box1)
                dict_1['coords'].append(coords)
    return dict_1


def get_mi_model():
    mi_model_path = pathlib.Path('../data/MI_model/mi_model01.h5')
    return keras.models.load_model(mi_model_path)

def make_bbox(bbox_extents):
    minr = bbox_extents[0]
    minc = bbox_extents[1]
    maxr = bbox_extents[2]
    maxc = bbox_extents[3]

    bbox_rect = np.array(
        [[minr, minc], [maxr, minc], [maxr, maxc], [minr, maxc]])
    bbox_rect = np.moveaxis(bbox_rect, 2, 0)
    return bbox_rect

def show_images(image_list, mask):
    # Create a new napari viewer
    viewer = napari.Viewer()
    # Add the image layer
    for img in image_list:
        viewer.add_image(img)
        viewer.add_image(mask)
    # for box_data in box_data_timecourse:
    #     print(box_data['coords'])
    #     bbox_rects = make_bbox(box_data['coords'])
    #     viewer.add_shapes(
    #         bbox_rects,
    #         face_color='transparent',
    #         edge_color='green',
    #         name='bounding box',
    #     )


    # Create a new labels layer for the bounding boxes
    # for coords in box_data['coords']:
    #     box_layer.data[coords[0]:coords[1], coords[2]:coords[3]] = 1

   # for box, coords in box_data:
   #  imin, imax, jmin, jmax = coords
   #  proba_1 = mi_model.predict(box, verbose=0)
   #  predict_1 = np.round(proba_1).astype('int')
   #  if predict_1 == 1:
   #      return mpl.patches.Rectangle((jmin, imin), jmax - jmin, imax - imin,
   #                                   fill=False, edgecolor='red', linewidth=1)
   #  elif predict_1 == 0:
   #      return mpl.patches.Rectangle((jmin, imin), jmax - jmin, imax - imin,
   #                                   fill=False, edgecolor='blue', linewidth=1)
   #

    napari.run()

if __name__ == '__main__':

    @omero_connect
    def main(conn=None):
        # meta_data = MetaData(PLATE_ID, conn)
        # exp_paths = ExpPaths(meta_data)
        # flatfield_dict = flatfieldcorr(meta_data, exp_paths)
        # image_list = mitotic_timecourse(flatfield_dict, IMAGE_ID, conn)
        img = io.imread('/Users/hh65/Desktop/Index.idx.ome.tiff')
        image_list = [img]
        mask = segment_image(image_list[SEGMENTATION_CHANNEL])
        bbox = analyse_image(img[0,:,:],mask[0,:,:])
        box_data_timecourse = []
        print(bbox['data'][0].shape)
        # for time in range(mask.shape[0]):
        #     box_data = analyse_image(image_list[SEGMENTATION_CHANNEL][time,:,:], mask[time,:,:])
        #     box_data_timecourse.append(box_data)
        show_images(image_list, mask)

    main()