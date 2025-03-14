This repository contains a series of Jupyter notebooks used for processing medical imaging data as part of the RSNA Pneumonia Detection Challenge . Below is the order in which the notebooks should be executed, along with a brief description of their purpose:

RSNA_Pneumonia_Detection_Challenge_dcm_to_png.ipynb
Converts DICOM (.dcm) images to PNG (.png) format.
This is the first step in the pipeline, preparing the raw medical images for further processing.

RSNA_Resized_800.ipynb
Resizes all PNG images to a uniform resolution of 800x800 pixels.
Ensures consistency in image size across the dataset, which is critical for machine learning model training.

RSNA_Pneumonia_Detection_v4.ipynb
Contains the final steps of the workflow, including data analysis, model training, or evaluation (to be detailed later).
This notebook builds on the processed images from the previous steps.

Note:
This README provides a high-level overview of the workflow to help you understand the sequence of the notebooks. A comprehensive README with detailed explanations of each step, dependencies, and results will be written at the end of the project.

This README is concise and serves its purpose of providing an initial understanding of the workflow while signaling that more detailed documentation will follow.
