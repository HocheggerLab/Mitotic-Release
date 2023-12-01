# omero_screen/__init__.py

__version__ = '0.1.0'


import os
import logging
from stardist.models import StarDist2D
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# generate class to provide default settings for program
class Defaults:
    """Store the default variables to read the Excel input file"""
    DEFAULT_DEST_DIR = "Desktop"  # Decides where the final data folder will be made
    FLATFIELD_TEMPLATES = "flatfield_correction_images"
    DATA = "mitotic_index_data"
    IMGS_CORR = "segmentation_check"
    TEMP_WELL_DATA = "temp_well_data"
    NS = 'openmicroscopy.org/omero/client/mapAnnotation'
    FEATURELIST = ['label', 'area', 'intensity_max', 'intensity_mean']
    STARDIST_MODEL = StarDist2D.from_pretrained('2D_versatile_fluo')
    CELLPOSE_MODEL = 'Nuclei_Hoechst'

SEPARATOR = "==========================================================================================\n"
