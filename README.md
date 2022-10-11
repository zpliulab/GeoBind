##  GeoBind: nucleic acid binding sites segmentation on protein surface using geometric deep learning.
## Standard alone Software prerequisites
* [Conda](https://docs.conda.io/en/latest/miniconda.html) Conda is recommended for environment management.
* [Python](https://www.python.org/) (3.6).
* [reduce](http://kinemage.biochem.duke.edu/software/reduce.php) (3.23). To add protons to proteins.
* [MSMS](http://mgltools.scripps.edu/packages/MSMS/) (2.6.1). To compute the surface of proteins.
* [hhblits](https://github.com/soedinglab/hh-suite) (3.3.0) To generate HMM.
## Python packages.
* [Pymesh](https://pymesh.readthedocs.io/en/latest/installation.html) (v0.3). For mesh of protein management and downsampling.
* [BioPython](https://github.com/biopython/biopython) (1.78). To parse PDB files.
* [Pytorch](https://pytorch.org/) (1.10.1). Pytorch with GPU version. Use to model, train, and evaluate the actual neural networks.
* [pykeops](https://www.kernel-operations.io/keops/index.html) (2.1). For computation of all point interactions computation of a protein surface.
* [Pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html). For geometric neural networks.
* [scikit-learn](https://scikit-learn.org/) (0.24.1). For point cloud space searching and some evaluation metrics.

## Specific usage

### 1. Download and install the standard alone software listed above.
Change the paths of these executable file at default_config/bin_path.py.

### 2. PDB files preparing.
a. All PDB files of seven kinds of ligands-binding proteins and corresponding ligands (DNA, RNA, ATP, HEM, Ca, Mn, Mg) are uploaded to (https://doi.org/10.5281/zenodo.7045931) which are sourced from BioLip.Download the PDBs.zip, move it to Dataset/ and then unzip. Or change the dir_opts['raw_pdb_dir'](in default_config/dir_opts.py) and dir_opts['ligand_dir'] respectively to where the "receptor" and "ligand" folders locate.

b. The calulation of MSA information is time-consuming, we have uploaded hmm files of all ligand binding proteins to (https://doi.org/10.5281/zenodo.7045931).
   Download the hmm.zip, move it to Dataset/ and then unzip.
   Or replace the dir_opts['hhm_dir'] to the folder "hmm".

c. Preprocessing. Dataset_lists contains the lists of seven types of ligands binding proteins and their binding information.
For example, preparing the featured point cloud and binding interface/sites of DNA or RNA:
```
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
For metal ion prediction task, the pretrained model needs input features of hmm and chemi features. For metal ion prediction task, the pretrained model needs input features of hmm, chemi and geo features.

### 5. Predicting binding sites of proteins not existing in Dataset_lists.
There is an easy using webserver www.zpliulab.cn/GeoBind. For large-scale predicting, we will upload a GeoBindProcessor without sparse the ligand structure (on going).

### TODO
The source code of GeoBind models and Protein Processor will be released as soon as the paper is published.
## License
GeoBind is released under an [MIT License](LICENSE).