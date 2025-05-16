import numpy as np
import scanpy as sc


def prefilter_genes(ann_data, min_counts=None, max_counts=None, min_cells=10, max_cells=None):
    if min_cells is None and min_counts is None and max_cells is None and max_counts is None:
        raise ValueError('Provide one of min_counts, min_genes, max_counts or max_genes.')
    id_tmp = np.asarray([True]*ann_data.shape[1],dtype=bool)
    id_tmp = np.logical_and(id_tmp, sc.pp.filter_genes(ann_data.X, min_cells=min_cells)[0]) if min_cells is not None  else id_tmp
    id_tmp = np.logical_and(id_tmp, sc.pp.filter_genes(ann_data.X, max_cells=max_cells)[0]) if max_cells is not None  else id_tmp
    id_tmp = np.logical_and(id_tmp, sc.pp.filter_genes(ann_data.X, min_counts=min_counts)[0]) if min_counts is not None  else id_tmp
    id_tmp = np.logical_and(id_tmp, sc.pp.filter_genes(ann_data.X, max_counts=max_counts)[0]) if max_counts is not None  else id_tmp
    ann_data._inplace_subset_var(id_tmp)


def prefilter_specialgenes(ann_data, Gene1Pattern="ERCC", Gene2Pattern="MT-"):
    id_tmp1 = np.asarray([not str(name).startswith(Gene1Pattern) for name in ann_data.var_names], dtype=bool)
    id_tmp2 = np.asarray([not str(name).startswith(Gene2Pattern) for name in ann_data.var_names], dtype=bool)
    id_tmp = np.logical_and(id_tmp1, id_tmp2)
    ann_data._inplace_subset_var(id_tmp)
