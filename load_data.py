import argparse
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
        ad.obs["sample_id"] = patient_id
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


def main(data_dir, img_dir):
    files = [file for file in os.listdir(data_dir) if file.endswith(".h5ad")]
    filepaths = [os.path.join(data_dir, file) for file in files]
    # ann_data[k] is the per patient AnnData
    ann_data = load_ann_data(filenames=files, filepaths=filepaths)
    # histology_images[k] per patient is the TIFF array
    histology_imgs = load_histology(ann_data=ann_data, img_dir=img_dir)
    # ann_data.X, ann_data.obsm["spatial"], histology_images[s]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument(
        "--data_dir", "-d",
        type=str,
        default="dataset/data",
        help="Path to the directory containing .h5ad files"
    )

    parser.add_argument(
        "--img_dir", "-i",
        type=str,
        default="dataset/images",
        help="Path to the directory containing images files"
    )

    args = parser.parse_args()
    main(args.data_dir, args.img_dir)
