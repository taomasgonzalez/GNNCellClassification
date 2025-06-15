import numpy as np
from graph import get_region_colors
import os
import random
import scanpy as sc
from scipy.sparse import vstack
from sklearn.decomposition import TruncatedSVD
import torch


def train_val_test_split(ann_data, seed):
    """
    returns 3 random lists based on the seed and the patients in the annotated data.
    The function assumes 12 patients in total and does a predefined split:
    1) list of names of the patients that should used for training (8 patients).
    2) list of names of the patients that should used for validation (2 patients).
    3) list of names of the patients that should used for testing (2 patients).
    """
    random.seed(seed)
    train_patients = random.sample(list(ann_data.keys()), 8)

    rest = [patient for patient in ann_data.keys() if patient not in train_patients]
    val_patients = random.sample(rest, 2)
    print(f"val_patients: {val_patients}")
    test_patients = [patient for patient in rest if patient not in val_patients]

    return train_patients, val_patients, test_patients


def get_coo_connections(ann_data):
    """
    """
    return { patient: data.obsp['spatial_connectivities'].tocoo() for patient, data in ann_data.items() }


def get_edge_indices(coo_connections):
    edge_indices = {}

    for patient, coo in coo_connections.items():
        row = torch.from_numpy(coo.row).long()
        col = torch.from_numpy(coo.col).long()
        edge_indices[patient] = torch.stack([row, col], dim=0)

    return edge_indices


def get_edge_features(ann_data, edge_indices, graph_dir):
    edge_features = {}

    for patient in ann_data.keys():
        filename = str(f"{patient}_adj.npy")
        adj_distances = np.load(os.path.join(graph_dir, filename))
        adj_tensor = torch.from_numpy(adj_distances)
        row, col = edge_indices[patient]
        distances = adj_tensor[row, col]
        edge_features[patient] = distances.unsqueeze(1).float()

    return edge_features


def get_normalized_umi_count(ann_data):
    normalized_data = {}

    for patient in ann_data.keys():
        sc.pp.normalize_total(ann_data[patient])
        sc.pp.log1p(ann_data[patient])
        normalized_data[patient] = ann_data[patient].X

    return normalized_data


def get_normalized_color_avgs(ann_data):
    offsets = {'151676': 310, '151669': 276, '151507': 236, '151508': 232, '151672': 264, \
               '151670': 339, '151673': 260, '151675': 228, '151510': 204, '151671': 238, \
               '151674': 234, '151509': 220}
    thickness = 48

    normalized_color_avgs = {}
    for patient_id, data in ann_data.items():
        offset = offsets[patient_id]
        hires_scale = ann_data[patient_id].uns['spatial'][patient_id]['scalefactors']['tissue_hires_scalef']
        spot_pixels = ann_data[patient_id].obsm['spatial'] * hires_scale
        spot_pixels = spot_pixels.astype(int)
        
        image = ann_data[patient_id].uns['spatial'][patient_id]['images']['hires']
        hires_shape = image.shape
        assert min(hires_shape[0], hires_shape[1]) > spot_pixels.max()

        flipped_image = np.flip(image, 0)
        x_pixels = spot_pixels[:, 0]
        y_pixels = spot_pixels[:, 1]
        
        normalized_color_avgs[patient_id] = get_region_colors(x_pixels, y_pixels, offset=offset, image=flipped_image, \
                                                              thickness=thickness, alpha=1)
    
    return normalized_color_avgs



def get_pca_reduced(normalized_data, train_patients):
    train_data = list(data for patient, data in normalized_data.items() if patient in train_patients)
    assert len(train_data) == 8
    train_stack = vstack(list(train_data))
    # svd.components_ now holds a shared basis between all the spots in the training set.
    # This basis will be freezed for using it when validating and testing.
    svd = TruncatedSVD(n_components=50, random_state=0)
    svd.fit(train_stack)

    reduced_data = {patient: svd.transform(X) for patient, X in normalized_data.items()}

    return reduced_data


def get_data_x(ann_data, reduced_data, normalized_color_avgs):
    data_x = {}
    for patient_id, data in ann_data.items():
        pca = reduced_data[patient_id]
        color_avgs = normalized_color_avgs[patient_id]
        color_avgs = color_avgs.reshape(-1, 1)
        x_np = np.hstack([pca, color_avgs])
        
        x_tensor = torch.from_numpy(x_np).float()
        
        data_x[patient_id] = x_tensor
        print(data_x[patient_id].shape)

    return data_x


def get_data_y(ann_data):
    data_y = {}

    for patient_id, data in ann_data.items():
        print("patient_id:", patient_id)
        codes = data.obs["sce.layer_guess"].cat.codes.values.copy()
        codes[codes == -1] = max(codes) + 1

        data_y[patient_id] = torch.tensor(codes, dtype=torch.long)

    return data_y


def get_data_pos(ann_data):
    data_pos = {}

    for patient_id, data in ann_data.items():
        y_tensor = torch.from_numpy(data.obs[['array_row', 'array_col']].values)
        y_tensor = y_tensor.to(torch.int16)
        data_pos[patient_id] = y_tensor

    return data_pos
