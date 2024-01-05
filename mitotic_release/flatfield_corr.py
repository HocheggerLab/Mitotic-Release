from mitotic_release.aggregator import ImageAggregator
from mitotic_release.general_functions import (
    save_fig,
    scale_img,
    generate_image,
    generate_random_image,
    omero_connect,
)
from mitotic_release.data_structure import MetaData, ExpPaths
from mitotic_release import SEPARATOR
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import matplotlib
from pathlib import Path
import glob
import platform
from itertools import chain
import random

if platform.system() == "Darwin":
    matplotlib.use("MacOSX")  # avoid matplotlib warning about interactive backend


def flatfieldcorr(meta_data, exp_paths) -> dict:
    """

    :return:
    """
    plate = meta_data.plate_obj
    channels = meta_data.channels
    template_path = exp_paths.flatfield_templates
    if len(glob.glob(f"{str(template_path)}/*.tif")) == 1:
        return load_corr_dict(template_path, channels)
    return generate_corr_dict(plate, channels, template_path)


@omero_connect
def multiplate_mask(plate_list, name, conn=None):
    """
    Generate a flatfield mask for multiple plates
    :param plate_list: list of omero plate objects
    :return: a dictionary with channel_name : flatfield correction masks
    """
    print(f"\nAssembling Flatfield Correction Masks\n{SEPARATOR}")
    image_list = []
    for plate_id in plate_list:
        plate = conn.getObject("Plate", plate_id)
        image_list += get_images(plate)
    agg = ImageAggregator(60)
    for image in tqdm(image_list):
        image_array = generate_image(image, 0)
        agg.add_image(image_array)
    blurred_agg_img = agg.get_gaussian_image(30)
    norm_mask = blurred_agg_img / blurred_agg_img.mean()
    io.imsave(Path.cwd().parent / "data" / f"{name}_flatfieldmask.tif", norm_mask)


def load_corr_dict(path, channels):
    print(f"Loading Flatfield Correction Masks from File\n{SEPARATOR}")
    corr_img_list = glob.glob(f"{str(path)}/*.tif")
    array_list = list(map(io.imread, corr_img_list))
    channel_list = list(channels.keys())
    return dict(zip(channel_list, array_list))


def get_images(plate):
    well_list = list(plate.listChildren())
    image_list = []
    for well in well_list:
        images = list(well.listChildren())
        image_list.append(images)
    all_images = [obj.getImage() for obj in list(chain(*image_list))]

    return random.sample(all_images, 50) if len(all_images) > 100 else all_images


def generate_corr_dict(plate, channels, template_path):
    """
    Saves each flat field mask file with well position and channel name
    :return: a dictionary with channel_name : flatfield correction masks
    """
    print(f"\nAssembling Flatfield Correction Masks for each Channel\n{SEPARATOR}")
    image_list = get_images(plate)
    corr_dict = {}
    # iteration extracts tuple (channel_name, channel_number) as channel
    for channel in list(channels.items()):
        corr_img_id = f"flatfield_{channel[0]}"
        norm_mask = aggregate_imgs(image_list, channel)
        io.imsave(template_path / f"{corr_img_id}.tif", norm_mask)
        for _ in range(5):  # generate 5 test images randomly chosen from the image_list
            example = gen_example(image_list, channel, norm_mask)
            example_fig(example, channel, template_path)
        corr_dict[channel[0]] = norm_mask  # associates channel name with flatfield mask
    return corr_dict


def aggregate_imgs(image_list, channel):
    """
    Aggregate images in well for specified channel and generate correction mask using the Aggregator Module
    :param channel: dictionary from self.exp_data.channels
    :return: flatfield correction mask for given channel
    """
    agg = ImageAggregator(60)
    for image in tqdm(image_list):
        image_array = generate_image(image, channel[1])
        agg.add_image(image_array)
    blurred_agg_img = agg.get_gaussian_image(30)
    return blurred_agg_img / blurred_agg_img.mean()


def gen_example(image_list, channel, mask):
    example_img_id, example_time, example_img = generate_random_image(
        image_list, channel
    )
    scaled = scale_img(example_img)
    corr_img = example_img / mask
    corr_scaled = scale_img(corr_img)
    # order all images for plotting
    return [
        example_img_id,
        example_time,
        [
            (scaled, "original image"),
            (np.diagonal(example_img), "diag. intensities"),
            (corr_scaled, "corrected image"),
            (np.diagonal(corr_img), "diag. intensities"),
            (mask, "flatfield correction mask"),
        ],
    ]


def example_fig(data_list, channel, path):
    image_id = data_list[0]
    image_time = data_list[1]
    data_tuple_list = data_list[2]
    fig, ax = plt.subplots(1, 5, figsize=(20, 5))
    for i, data_tuple in enumerate(data_tuple_list):
        plt.sca(ax[i])
        if i in [0, 2, 4]:
            plt.imshow(data_tuple[0], cmap="gray")
        else:
            plt.plot(data_tuple[0])
            plt.ylim(data_tuple[0].min(), 5 * data_tuple[0].min())
        plt.title(data_tuple[1])
    # save and close figure
    fig_id = f"{image_id}_{image_time*10}min_{channel[0]}_flatfield_check"  # using channel name
    save_fig(path, fig_id)
    plt.close(fig)


# test


if __name__ == "__main__":
    multiplate_mask([1545, 1546, 1547, 1548, 1550, 1551, 1552, 1553], "PPasescreen1")
    
