from omero_images import OmeroImages
from training_gui import TrainingScreen, count
import matplotlib.pyplot as plt

image_id = 921385
timepoints = [5, 10]
nuclei_per_img = 50


imgs = OmeroImages(image_id, timepoints, nuclei_per_img)
screen = TrainingScreen(imgs)
print(count)



