from mitotic_release import SEPARATOR
from mitotic_release.data_structure import MetaData, ExpPaths
from mitotic_release.flatfield_corr import flatfieldcorr
from mitotic_release.image_analysis import Image
from mitotic_release.general_functions import omero_connect
import pandas as pd
import pathlib


# Functions to loop through well object, assemble data for images and ave quality control data

def well_loop(well, meta_data, exp_paths, flatfield_dict):
    well_pos = f"row_{well.row}_col{well.column}"
    df_well_path = exp_paths.temp_well_data / f'data_{well_pos}'
    # check if file already exists to load dfs and move on
    if pathlib.Path.exists(df_well_path):
        print(f"\nWell has already been analysed, loading data\n{SEPARATOR}")
        df_well = pd.read_pickle(str(df_well_path))
    # analyse the images to generate the dfs
    else:
        print(f"\nSegmenting and Analysing Images\n{SEPARATOR}")
        df_well = pd.DataFrame()
        image_number = len(list(well.listChildren()))
        for number in range(image_number):
            omero_img = well.getImage(number)
            image = Image(well, omero_img, meta_data, exp_paths, flatfield_dict)
            df_image = image.mitotic_index()
            df_well = pd.concat([df_well, df_image])
            df_well.to_pickle(str(df_well_path))


    return df_well


if __name__ == "__main__":
    @omero_connect
    def well_test(conn=None):
        meta_data = MetaData(1059, conn)
        exp_paths = ExpPaths(meta_data)
        well = conn.getObject("well", 11553)
        flatfield_dict = flatfieldcorr(meta_data, exp_paths)
        df_final = well_loop(well, meta_data, exp_paths, flatfield_dict)
        print(df_final)

    df_final = well_test()
    print(df_final)
