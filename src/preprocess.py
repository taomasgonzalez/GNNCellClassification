import argparse
from load_data import load_ann_data, load_histology
from graph import calculate_adj_matrix, create_graphs
from gene_filtering import prefilter_genes, prefilter_specialgenes
import os


def main(data_dir, img_dir, graph_dir):
    files = [file for file in os.listdir(data_dir) if file.endswith(".h5ad")]
    filepaths = [os.path.join(data_dir, file) for file in files]
    # ann_data[k] is the per patient AnnData
    ann_data = load_ann_data(filenames=files, filepaths=filepaths)
    # histology_images[k] per patient is the TIFF array
    histology_imgs = load_histology(ann_data=ann_data, img_dir=img_dir)
    print("obs: ", ann_data['151674'].obs)
    print("var: ", ann_data['151674'].var)
    print("uns: ", ann_data['151674'].uns)
    print("obsm: ", ann_data['151674'].obsm['spatial'])
    print("obsp: ", ann_data['151674'].obsp)
    print("X: ", ann_data['151674'].X)

    create_graphs(graph_dir=graph_dir, ann_data=ann_data, histology_imgs=histology_imgs)
    # ann_data.X, ann_data.obsm["spatial"], histology_images[s]
    prefilter_genes(ann_data, min_cells=3)
    prefilter_specialgenes(ann_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument(
        "--data_dir", "-d",
        type=str,
        default="../dataset/data",
        help="Path to the directory containing .h5ad files"
    )

    parser.add_argument(
        "--img_dir", "-i",
        type=str,
        default="../dataset/images",
        help="Path to the directory containing images files"
    )

    parser.add_argument(
        "--graph_dir", "-g",
        type=str,
        default="../out/graphs",
        help="Path to the directory where generated graphs will be output to"
    )

    args = parser.parse_args()
    main(args.data_dir, args.img_dir, args.graph_dir)
