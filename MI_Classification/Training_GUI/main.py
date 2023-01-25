from omero_images import OmeroImages
from training_gui import TrainingScreen, count
import matplotlib.pyplot as plt

image_id = 247869
timepoints = [3, 10, 15, 20]
nuclei_per_img = 3


imgs = OmeroImages(image_id, timepoints, nuclei_per_img)
screen = TrainingScreen(imgs)
print(count)



