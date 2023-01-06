# Program to analyse Omero high-throughput microscopy images

current version 0.1.0

This is a first version of the program; additional bug fixes and testing necessary
Requires a connection to an active Omero server



## Contributors
Alex Herbert and Helfrid Hochegger

## Purpose of the program

Analysing mitotic release data for cdk1as release experiments (see Rata et al 2018).

Images are first pre-processed using flatfield correction, then segmenthed using stardist models.
A pretrained CNN classifies the nuclei as mitotic or interphase.

The input metadata are provided via an Excel file (see data/sample_metadata.xlsx) and need to be uploaded 
to the Omero server using the GDSC HCC script on the server.

#TODO generation of Figure still has bug when NaN in dataframe. Bugfix not yet tested




