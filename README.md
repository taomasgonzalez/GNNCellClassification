# GNNCellClassification

> **Note:** This is an ongoing project. Features, results, and documentation are subject to change.

A multimodal Graph Neural Network (GCN) pipeline for automated classification of cortical layers in the human dorsolateral prefrontal cortex (DLPFC) using spatial transcriptomics and histological imaging data.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Data](#data)
- [Installation](#installation)
- [Reproducibility](#reproducibility)
- [Model Architecture](#model-architecture)
- [Experiment Tracking](#experiment-tracking)
- [Results](#results)
- [Report](#report)
- [References](#references)
- [Acknowledgments](#acknowledgments)

---

## Project Overview

This project implements a multimodal GCN to classify brain layers in the DLPFC using the 10x Genomics Visium spatial transcriptomics dataset and corresponding histological images. Each Visium spot is modeled as a node in a graph, with features derived from both gene expression and histology. The model integrates molecular, spatial, and morphological information to achieve robust layer classification.

---

## Data

- **Source:** [spatialDLPFC](https://github.com/LieberInstitute/spatialDLPFC) project, [Lieber Institute for Brain Development](https://www.libd.org/)
- **Download:** The dataset includes preprocessed AnnData objects and high-resolution histology images. Download scripts are provided.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd GNNCellClassification
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install DVC for pipeline management:**
   ```bash
   pip install dvc
   ```
---

## Reproducibility

- All steps are tracked with DVC and can be reproduced with `dvc repro`.
- Random seeds are set for data splits and PCA.
- All dependencies are listed in `requirements.txt`.

### Docker

A Dockerfile is provided for reproducing builds. Currently, the Dockerfile only supports the use of
CPU, but it should be easily extendable to support GPU as well.

  ```bash
  docker build classifier .
  docker run -it -v "$(pwd)":/work classifier
  . /opt/venv/bin/activate
  dvc repro
  ```

### Pipeline Stages

As shown above, the pipeline can be run end-to-end using [DVC](https://dvc.org/):

```bash
dvc repro
```

Stages defined in `dvc.yaml`:
- `getdata`: Downloads and unpacks the dataset.
- `preprocess`: Constructs spatial graphs and adjacency matrices.
- `featurize`: Extracts and saves node/edge features and labels.
- `train`: Trains the GNN and logs metrics/artifacts.
- `test`: Tests the GNN and logs metrics/artifacts.

---

#### Python implementation of the Pipeline stages

The pipeline can also be controlled via a command-line interface:

```bash
python3 src/main.py <stage> [--options]
```

### Main stages:

- **getdata:** Download raw data and images.
  ```bash
  python3 src/main.py getdata --dataset_dir dataset
  ```

- **preprocess:** Build adjacency graphs from raw data.
  ```bash
  python3 src/main.py preprocess --dataset_dir dataset --graph_dir out/graphs
  ```

- **featurize:** Generate feature tensors for GNN input.
  ```bash
  python3 src/main.py featurize --dataset_dir dataset --graph_dir out/graphs --tensors_dir dataset/tensors --params_file params.yaml
  ```

- **train:** Train the GNN model.
  ```bash
  python3 src/main.py train --tensors_dir dataset/tensors --params_file params.yaml
  ```

- **test:** Test the GNN model.
  ```bash
  python3 src/main.py test --tensors_dir dataset/tensors --params_file params.yaml
  ```
---

## Model Architecture

- **Node Features:** PCA-reduced gene expression (50 dims) + normalized histology color (1 dim)
- **Edge Features:** Pixel distances between spatially adjacent spots
- **GCN:** 3-layer GCN with ReLU and dropout, followed by a linear classifier
- **Loss:** Weighted cross-entropy with label smoothing to address class imbalance

**Default hyperparameters** (see `params.yaml`):
- Learning rate: 0.001
- Epochs: 400
- Batch size: 32
- Hidden layers: 208, 102, 40
- Label smoothing: 0.002

---

## Experiment Tracking

- **MLflow** is used for experiment tracking and artifact logging.
- The server is started automatically during training and logs are saved in the `artifacts/` directory.

---

## Results

- **Accuracy:** 81% multiclass accuracy on the held-out test set
- **Per-layer F1:** 0.82 when not taking into account the "Unknown" class

See the [report](#report) for detailed results and analysis.


---

## Report


---

## References

- [spatialDLPFC dataset](https://github.com/LieberInstitute/spatialDLPFC)
- [10x Genomics Visium](https://www.10xgenomics.com/products/spatial-gene-expression)
- Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional Networks", ICLR 2017

---

## Acknowledgments

We thank the Allen Institute for Brain Science and the Lieber Institute for Brain Development for providing the DLPFC dataset and associated histological images.

---

**Author:** Tomás Agustín González Orlando (taomasgonzalez@gmail.com)
