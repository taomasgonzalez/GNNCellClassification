import os
import scanpy as sc
import pandas as pd
import requests
import cv2
import zipfile
import matplotlib.pyplot as plt


def load_ann_data(filenames, filepaths):
    ann_data = {}

    for filepath, filename in zip(filepaths, filenames):
        patient_id = os.path.splitext(filename)[0]
        print(f"Loading AnnData for sample {patient_id} …")
        ad = sc.read_h5ad(filepath)
        ann_data[patient_id] = ad

    return ann_data


def load_histology(ann_data, img_dir):
    histology_imgs = {}

    for patient_id in ann_data.keys():
        print(f"Loading Image for sample {patient_id} …")
        tif_name = f"{patient_id}.tif"
        tif_path = os.path.join(img_dir, tif_name)

        img = cv2.imread(tif_path, cv2.IMREAD_COLOR)
        if img is None:
            print("error")
            raise FileNotFoundError(f"Could not load histology image at {tif_path}")
        else:
            histology_imgs[patient_id] = img
    return histology_imgs
