## GeoBind: nucleic acid binding sites segmentation on protein surface using geometric deep learning.
## Standard alone Software prerequisites
* [Conda*](https://docs.conda.io/en/latest/miniconda.html) Conda is recommended for environment management.
* [Python*](https://www.python.org/) (v3.7.12).
* [reduce*](https://github.com/rlabduke/reduce) (3.23). Add protons to proteins.
* [MSMS*](https://ccsb.scripps.edu/msms/downloads/) (2.6.1). Compute the surface of proteins.
* [hhblits](https://github.com/soedinglab/hh-suite) (3.3.0) Generate HMM profile. GeoBind can be trained without HMM profile.
## Python packages.
* [Pymesh*](https://pymesh.readthedocs.io/en/latest/insta llation.html) (v0.3). For mesh of protein management and downsampling.
* [BioPython*](https://github.com/biopython/biopython) (v1.78). To parse PDB files.
* [Pytorch*](https://pytorch.org/) (v1.10.1). Pytorch with GPU version. Use to model, train, and evaluate the actual neural networks.
* [pykeops*](https://www.kernel-operations.io/keops/index.html) (v2.1). For computation of all point interactions of a protein surface.
* [Pytorch-geometric*](https://pytorch-geometric.readthedocs.io/en/latest/index.html) (v2.0.4). For geometric neural networks.
* [scikit-learn*](https://scikit-learn.org/) (v0.24.1). For point cloud space searching and model evaluation.

## Specific usage
### 1. Download and install the standard alone software listed above.
Change the paths of these executable files (_reduce_, _msms_) at default_config/bin_path.py.

### 2. PDB files preparing.
a. All PDB files of seven kinds of ligands-binding proteins and corresponding ligands (DNA, RNA, ATP, HEM, Ca, Mn, Mg) are uploaded to (https://doi.org/10.5281/zenodo.7045931).
    Download the PDBs.zip, move it to Dataset/ and then unzip. Or change the dir_opts['raw_pdb_dir'](in default_config/dir_opts.py) and dir_opts['ligand_dir'] respectively to where the "receptor" and "ligand" folders locate.
```
mkdir Dataset && cd Dataset
wget https://zenodo.org/record/7045931/files/PDBs.zip
unzip PDBs.zip
```
b. The calulation of MSA information is time-consuming, we have uploaded hmm files of all ligand binding proteins to (https://doi.org/10.5281/zenodo.7045931).
   Download the hmm.zip, move it to Dataset/ and then unzip.
   Or replace the dir_opts['hhm_dir'] to the folder "hmm".
```
wget https://zenodo.org/record/7045931/files/hmm.zip
unzip hmm.zip
```
c. Preprocessing. Dataset_lists contains the lists of seven types of ligands binding proteins and their binding information.
For example, preparing the featured point cloud and binding interface/sites of DNA or RNA:
```
cd ..
python prepare_one.py --ligand RNA
python prepare_one.py --ligand DNA
```
### 3. Training from scratch. Change the ligand type and pre-trained model path (checkpoints_dir) in predict.sh and run:
```
sh train.sh
```
### 4. Predicting
Predicting the test set of each ligand binding protein, change the ligand type and pre-trained model path (checkpoints_dir) in train.sh and run:

```
sh predict.sh
```
For metal ion prediction task, the pretrained model needs input features of hmm and chemical features. For metal ion prediction task, the pretrained model needs input features of hmm, chemi and geo features.

### 5. Predicting binding sites of proteins not existing in Dataset_lists.
There is an easy using webserver www.zpliulab.cn/GeoBind. For large-scale predicting, we will upload a GeoBindProcessor without sparse the ligand structure (on going).

## License
GeoBind is released under an [MIT License](LICENSE).