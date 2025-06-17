import os
import cv2
import csv
import re
import pandas as pd
import numpy as np
import scanpy as sc
import math
from matplotlib import pyplot as plt
import networkx as nx


def create_graphs(graph_dir, ann_data, histology_imgs):
    offsets = {'151676': 310, '151669': 276, '151507': 236, '151508': 232, '151672': 264, \
        '151670': 339, '151673': 260, '151675': 228, '151510': 204, '151671': 238, \
        '151674': 234, '151509': 220}
    thickness = 48

    for patient_id, data in ann_data.items():
        print(f"Creating Graphs for sample {patient_id} …")

        offset = offsets[patient_id]
        hires_scale = ann_data[patient_id].uns['spatial'][patient_id]['scalefactors']['tissue_hires_scalef']
        spot_pixels = ann_data[patient_id].obsm['spatial'] * hires_scale
        spot_pixels = spot_pixels.astype(int)
        image = ann_data[patient_id].uns['spatial'][patient_id]['images']['hires']
        hires_shape = image.shape
        assert min(hires_shape[0], hires_shape[1]) > spot_pixels.max()

        flipped_image = np.flip(image, 0)
        image_cpy = flipped_image.copy()
        x_pixels = spot_pixels[:, 0]
        y_pixels = spot_pixels[:, 1]

        for xi, yi in zip(x_pixels, y_pixels):
            x0 = max(0, xi - thickness)
            x1 = min(image.shape[1], xi + thickness)
            y0 = max(0, yi - thickness) + offsets[patient_id]
            y1 = min(image.shape[0], yi + thickness) + offsets[patient_id]
            image_cpy[y0:y1, x0:x1, :] = [0, 0, 0]


        graph_img_path = os.path.join(graph_dir, f'{patient_id}_graph.jpg')
        if not cv2.imwrite(graph_img_path, np.flip(image_cpy, 0)):
            raise FileNotFoundError(f"Could not write graph at {graph_img_path}")

        graph_path = os.path.join(graph_dir, f'{patient_id}_adj')
        adj = calculate_adj_matrix(x_pixels=x_pixels,y_pixels=y_pixels, image=flipped_image, thickness=thickness,  \
                                   alpha=1, histology=True, offset=offsets[patient_id])

        # verify that it's a square matrix
        assert adj.ndim == 2 and adj.shape[0] == adj.shape[1]
        # distances should be >= 0
        assert np.all(adj >= 0.0)
        # there should be no funny numbers (infinity, etc)
        assert np.isfinite(adj).all()
        # should be a symmetric matrix
        assert np.allclose(adj, adj.T, atol=1e-8)
        # diagonals should be 0 (or close to)
        assert np.allclose(np.diag(adj), 0.0, atol=1e-4)

        # force diagonals to zero.
        np.fill_diagonal(adj, 0.0)
        print("Max value: ", np.max(adj))
        np.save(graph_path, adj)


def pairwise_distance(x):
    # Squared norms of each row
    sq_norms = np.sum(x*x, axis=1)

    # ||xi - xj||**2 = ||xi||**2 + ||xj||**2 - 2 * xi·xj
    sq_norms = sq_norms[:, None] + sq_norms[None, :] - 2.0 * x.dot(x.T)
    # Numerical safety: clip any small negatives to zero
    np.maximum(sq_norms, 0, out=sq_norms)
    return np.sqrt(sq_norms)


def get_region_colors(x_pixels, y_pixels, offset, image, thickness=49, alpha=1):
    # thickness to control the range of the region surrounding each spot.
    # alpha to control the color scale relative to the distances
    thickness_half = thickness // 2

    # Obtain the average color of the region surrounding each spot.
    # color_avgs[i] will have the average per-channel color (color0, color1, color2) of the region.
    color_avgs = []
    for x_pixel, y_pixel in zip(x_pixels, y_pixels):
        # define region limits
        x0 = max(0, x_pixel - thickness_half)
        x1 = min(image.shape[1], x_pixel + thickness_half)
        y0 = max(0, y_pixel - thickness_half) + offset
        y1 = min(image.shape[0], y_pixel + thickness_half) + offset

        avg_colors = np.mean(image[x0:x1, y0:y1], axis=(0, 1))
        color_avgs.append(avg_colors)

    color_avgs = np.array(color_avgs)

    # Obtain variance of each color channel
    color_avgs_vars = np.var(color_avgs, axis=0)

    print("Variances of color0, color1, color2 = ", color_avgs_vars)

    normalized_color_avgs = np.sum(color_avgs * color_avgs_vars, axis=1) / np.sum(color_avgs_vars)
    normalized_color_avgs -= np.mean(normalized_color_avgs)
    normalized_color_avgs /= np.std(normalized_color_avgs)
    return normalized_color_avgs


def calculate_adj_matrix(x_pixels, y_pixels, offset, image, thickness=49, alpha=1, histology=True):
    if histology:
        print("Calculating adj matrix using histology image...")

        normalized_color_avgs = get_region_colors(x_pixels, y_pixels, offset, image, thickness, alpha)

        z_scale = np.max([np.std(x_pixels), np.std(y_pixels)]) * alpha
        z = normalized_color_avgs * z_scale

        print("Var of x, y, z = ", np.var(x_pixels), np.var(y_pixels), np.var(z))
        X = np.array([x_pixels, y_pixels, z]).T.astype(np.float64)
    else:
        print("Calculating adj matrix using xy only...")
        X = np.array([x_pixels, y_pixels]).T.astype(np.float64)

    return pairwise_distance(X)


def visualize_graph(graph, color):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(graph, pos=nx.spring_layout(graph, seed=42), with_labels=False,
                     node_color=color, cmap="Set2")
    plt.show()
