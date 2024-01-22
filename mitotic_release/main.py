from mitotic_release import SEPARATOR
from mitotic_release.general_functions import omero_connect
from mitotic_release.data_structure import MetaData, ExpPaths
from mitotic_release.flatfield_corr import flatfieldcorr
from mitotic_release.well_loop import well_loop
import pandas as pd


@omero_connect
def main(plate_id, conn=None):
    meta_data = MetaData(plate_id, conn)
    exp_paths = ExpPaths(meta_data)
    df_final = pd.DataFrame()
    flatfield_dict = flatfieldcorr(meta_data, exp_paths)
    for count, well in enumerate(list(meta_data.plate_obj.listChildren())):
        print(
            f"Analysing well row:{well.row}/col:{well.column} - {count + 1} of {meta_data.plate_length}.\n{SEPARATOR}"
        )
        well_data = well_loop(well, meta_data, exp_paths, flatfield_dict)
        df_final = pd.concat([df_final, well_data])
    df_final.to_csv(exp_paths.final_data / f"{meta_data.plate}_final_data.csv")


if __name__ == "__main__":
    main(1641)
