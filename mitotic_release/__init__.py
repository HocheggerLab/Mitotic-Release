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
    MAGNIFICATION = '10x'
    FLATFIELD_TEMPLATES = "flatfield_correction_images"
    DATA = "mitotic_index_data"
    IMGS_CORR = "segmentation_check"
    TEMP_WELL_DATA = "temp_well_data"
    NS = 'openmicroscopy.org/omero/client/mapAnnotation'
    FEATURELIST = ['label', 'area', 'intensity_max', 'intensity_mean']
    STARDIST_MODEL = StarDist2D.from_pretrained('2D_versatile_fluo')
    CELLPOSE_MODEL = 'Nuclei_Hoechst'

SEPARATOR = "==========================================================================================\n"

def setup_logging():
    # Set a less verbose level for the root logger to avoid noisy logs from external libraries
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    )

    # Create and configure your application's main logger
    app_logger_name = "omero-screen-napari"  # Use a unique name for your application's logger
    app_logger = logging.getLogger(app_logger_name)
    app_logger.setLevel(logging.DEBUG)  # Or DEBUG, as per your requirement

    # Optionally, add any specific handlers/formatters to your app logger here


# Ensure this is called when your package is imported
setup_logging()