from omero.gateway import BlitzGateway
import numpy as np
from skimage import exposure, filters, measure, io
from stardist.models import StarDist2D
import random
from config import username, password
from csbdeep.utils import normalize
STARDIST_MODEL = StarDist2D.from_pretrained('2D_versatile_fluo')
flatfield_correction = io.imread('/Users/hh65/Desktop/test_screen/flatfield_correction_images/flatfield_H2B.tif')
class OmeroImages:

    def __init__(self, image_id, timepoints, sample_number):
        self.image_ID = image_id
        self.timepoints = timepoints
        self.sample_number = sample_number
        self.image_list = self.get_images()
        self.width = 20
        self.nuclei_list = self.get_nuclei(self.image_list)

    def get_images(self):
        image_list=[]
        conn = BlitzGateway(username, password, host='localhost')
        conn.connect()
        img_object = conn.getObject("Image", self.image_ID)
        pixels = img_object.getPrimaryPixels()
        for i in self.timepoints:
            img = pixels.getPlane(0, 0, i) / flatfield_correction
            image_list.append(img)
        conn.close()
        return image_list


    def get_nuclei(self, image_list):
        nuclei_list = []
        for img in image_list:
            percentile: tuple[float, float] = (1, 99)
            percentiles = np.percentile(img, (percentile[0], percentile[1]))
            scaled_img = exposure.rescale_intensity(img, in_range=tuple(percentiles))
            blurred_img = filters.gaussian(scaled_img, sigma=4)
            label_objects, nb_labels = STARDIST_MODEL.predict_instances(normalize(blurred_img))
            for region in random.sample(measure.regionprops(label_objects), self.sample_number):
                centroid = region.centroid
                i = centroid[0]
                j = centroid[1]
                imin = int(round(max(0, i-self.width)))
                imax = int(round(min(label_objects.shape[0], i+self.width+1)))
                jmin = int(round(max(0, j-self.width)))
                jmax = int(round(min(label_objects.shape[1], j+self.width+1)))
                box=img[imin:imax, jmin:jmax]
                nuclei_list.append(box)
        random.shuffle(nuclei_list)
        return nuclei_list
