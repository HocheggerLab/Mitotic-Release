from omero.gateway import BlitzGateway
import json
import getpass
import functools
import numpy as np
from skimage import exposure, filters, io, measure
from skimage.measure import find_contours

from stardist.models import StarDist2D
from csbdeep.utils import normalize
import napari
import keras
from pathlib import Path

flatfield_mask = io.imread('/Users/hh65/Desktop/test_screen/flatfield_correction_images/flatfield_H2B.tif')
STARDIST_MODEL = StarDist2D.from_pretrained('2D_versatile_fluo')

def omero_connect(func):
    """
    decorator to log in and generate omero connection
    :param func: function to be decorated
    :return: wrapper function: passes conn to function and closes it after execution
    """

    @functools.wraps(func)
    def wrapper_omero_connect(*args, **kwargs):
        try:
            with open('../data/secrets/config_test.json') as file:
                data = json.load(file)
            username = data['username']
            password = data['password']
            server = data['server']
        except IOError:
            username = input("Username: ")
            password = getpass.getpass(prompt='Password: ')
            server = "ome2.hpx.sussex.ac.uk"
        conn = BlitzGateway(username, password, host=server)
        value = None
        try:
            print('Connecting to Omero')
            if conn.connect():
                value = func(*args, **kwargs, conn=conn)
                print('Disconnecting from Omero')
            else:
                print(f'Failed to connect to Omero: {conn.getLastError()}')
        finally:
            # No side effects if called without a connection
            conn.close()
        return value

    return wrapper_omero_connect


@omero_connect
def get_img(image_id, conn=None):
    img = conn.getObject("Image", image_id)
    pixels = img.getPrimaryPixels()
    return pixels.getPlane(0,0,0) / flatfield_mask

def segmentation(img):
    percentile: tuple[float, float] = (1, 99)
    percentiles = np.percentile(img, (percentile[0], percentile[1]))
    scaled_img = exposure.rescale_intensity(img, in_range=tuple(percentiles))
    blurred_img = filters.gaussian(scaled_img, sigma=4)
    label_objects, nb_labels = STARDIST_MODEL.predict_instances(normalize(blurred_img))
    print(np.max(label_objects))
    # Calculate average size of nuclei
    # region_areas = [region.area for region in measure.regionprops(label_objects)]
    # average_size = np.mean(region_areas)
    #
    # # Filter out small objects
    # filtered_label_objects = np.zeros_like(label_objects)
    # for region in measure.regionprops(label_objects):
    #     if region.area >= 0.3 * average_size:  # Keeping only nuclei >= 30% of average size
    #         filtered_label_objects[region.coords[:, 0], region.coords[:, 1]] = region.label

    return scaled_img, label_objects

def get_nuclei (img, label_objects):
    width = 20

    nuclei_data = {'data': [], 'coords': []}
    for region in measure.regionprops(label_objects):

        centroid = region.centroid
        i = centroid[0]
        j = centroid[1]
        imin = int(round(max(0, i - width)))
        imax = int(round(min(label_objects.shape[0], i + width + 1)))
        jmin = int(round(max(0, j - width)))
        jmax = int(round(min(label_objects.shape[1], j + width + 1)))
        box = img[imin:imax, jmin:jmax]
        coords = [imin, imax, jmin, jmax]
        if box.shape == (41, 41):
            box1 = box[:, :, np.newaxis]
            nuclei_data['data'].append(box1)
            nuclei_data['coords'].append(coords)
    return nuclei_data

def classify_nuclei(nuclei_data):
    mi_model_path = Path('../MI_Classification/CNN_Training/TrainingData/MI_CNN_model20x.h5')
    model = keras.models.load_model(mi_model_path)
    predictions = model.predict(np.array(nuclei_data['data']), verbose=0)
    return predictions, nuclei_data['coords']


def display_image_with_bounding_boxes(image, label_objects, coords, predictions):
    
    viewer = napari.Viewer()
    viewer.add_image(image, name='Image')

    contours = find_contours(label_objects, level=0.5)
    all_contours = list(contours)
    # Add all contours as a single layer
    viewer.add_shapes(all_contours, shape_type='path', edge_color='white', name='Contours')

    # Prepare lists to store boxes and their colors
    boxes = []
    colors = []

    for coord, prediction in zip(coords, predictions):
        imin, imax, jmin, jmax = coord

        # Correctly define the box for Napari's [x, y] format
        box = np.array([[imin, jmin], [imin, jmax], [imax, jmax], [imax, jmin]])
        boxes.append(box)

        # Assign color based on prediction
        color = [1, 0, 0, 1] if prediction > 0.5 else [1, 1, 0, 1]
        colors.append(color)

    # Add all boxes in a single shapes layer
    viewer.add_shapes(boxes, shape_type='polygon', edge_color=colors, face_color='transparent', name='Bounding Boxes')

    napari.run()

if __name__ == '__main__':

    image = get_img(1427)
    scaled, masks = segmentation(image)
    nuclei_data = get_nuclei(image, masks)

    predictions, coords = classify_nuclei(nuclei_data)

    display_image_with_bounding_boxes(image, masks, coords, predictions)
