Create or update notebooks/colab_training.ipynb to be fully self-contained for Google Colab Pro with an H100 GPU.

The notebook must have these cells in order:
1. GPU verification: check torch.cuda.is_available(), print GPU name and memory
2. Install dependencies: pip install torch torch-geometric rdkit-pypi scanpy gseapy wandb omegaconf tqdm scikit-learn scipy matplotlib seaborn umap-learn
3. Clone the repo or upload src/ directory
4. Download Rosetta data (using src/data/download.py)
5. Preprocess data (using src/data/preprocess.py)
6. wandb login cell
7. Load config from configs/default.yaml (or whichever config is specified)
8. Training run with progress display
9. Evaluation on test set with results display
10. Save best checkpoint and download to local machine
11. Print peak GPU memory usage

The notebook should work top-to-bottom with no manual intervention after the wandb login.
