from omero.gateway import BlitzGateway
from cellpose import models
from skimage import (
    io, exposure, feature, filters, measure, segmentation, color, morphology
)
import random
from config import username, password


class OmeroImages:

    def __init__(self, image_id, timepoints, sample_number):
        self.image_ID = image_id
        self.timepoints = timepoints
        self.sample_number = sample_number
        self.image_list = self.get_images()
        self.width = 10
        self.nuclei_list = self.get_nuclei(self.image_list)

    def get_images(self):
        image_list=[]
        conn = BlitzGateway(username, password, host='ome2.hpc.sussex.ac.uk')
        conn.connect()
        object = conn.getObject("Image", self.image_ID)
        pixels = object.getPrimaryPixels()
        for i in self.timepoints:
            img = pixels.getPlane(0, 0, i)
            image_list.append(img)
        conn.close()
        return image_list


    def get_nuclei(self, image_list):
        nuclei_list = []
        for img in image_list:
            n_model = models.CellposeModel(gpu=False, model_type='nuclei')
            n_channels = [[0, 0]]
            n_masks, n_flows, n_styles = n_model.eval(img, diameter=15, channels=n_channels)
            for region in random.sample(measure.regionprops(n_masks), self.sample_number):
                centroid = region.centroid
                i = centroid[0]
                j = centroid[1]
                imin = int(round(max(0, i-self.width)))
                imax = int(round(min(n_masks.shape[0], i+self.width+1)))
                jmin = int(round(max(0, j-self.width)))
                jmax = int(round(min(n_masks.shape[1], j+self.width+1)))
                box=img[imin:imax, jmin:jmax]
                nuclei_list.append(box)
        random.shuffle(nuclei_list)
        return nuclei_list
