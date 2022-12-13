BioBrigit
===============

BioBrigit is a computational tool designed for the prediction of metal mobility pathways through a protein. It uses a novel scoring function that combines deep learning and previous domain knowledge regarding bioinorganic interactions as described in [Sánchez-Aparicio et al. (2017)](https://chemrxiv.org/engage/chemrxiv/article-details/60c74de1469df46a86f44378). The deep learning part of our hybrid approach consists on a 3D Convolutional Neural Network trained to interpret the biochemical environment to distinguish between metal-binding and non-binding protein regions.

<img src="./docs/figures/BioBrigit_light.png" width="820" class="center"/>

Features
--------
**Diferent options for customizing the search:**

* Search for the binding sites of specific metals.
* Provide a score that indicates how suitable a certain region will be for binding.
* Scan the whole protein or only a region (in PDB format).

**Possible applications:**

* Identification of probable metal diffusion pathways through a protein.
* Identification of conformational changes that alter the formation of such paths.
* Metalloenzyme and metallodrug design.
* Molecular physiopathology analysis.

**Modular design:**

* The modular design of this package allows for its use as a command-line application or to be integrated into a larger Python program or pipeline.

Installation
------------
The recommended environment for using BioBrigit takes advatange of GPU acceleration. However, it is possible to use the tool in CPU-only environments.

The first step is to create an environment with the necessary libraries. Some will bee directly installed from source.

```bash
> conda create -n {name} python=3.9
> conda activate {name}
> pip install git+https://github.com/Acellera/moleculekit
> pip install git+https://github.com/RaulFD-creator/biobrigit
> conda install pdb2pqr -c acellera -c conda-forge
> pip install scikit-learn
```

**2.1. Environment set-up with CUDA acelleration**

The last step is to install the deep learning framework:

```bash
> conda install pytorch pytorch-cuda -c pytorch -c nvidia
> conda install pytorch-lightning tensorboard torchmetrics -c conda-forge
```

**2.2. Environment set-up without CUDA acelleration**

If no CUDA device is available, the recommended installation of the deep learning framework is:

```bash
> conda install pytorch
> conda install pytorch-lightning tensorboard torchmetrics -c conda-forge
```

Usage
-----
Once the environment is properly set-up the use of the program is relatively simple. The easiest example is:

```bash
> biobrigit target metal
```

There are many parameters that can be also tuned, though default use is reccomended.

* `--model`: Which CNN model is to be used. Two options available `BrigitCNN` which is the default model with highest performance and `TinyBrigit`, which is a smaller model for improved computational efficiency, though it has demonstrated lower performance.
* `--device`: Whether to use GPU acceleration (`cuda`) or not (`cpu`). By default, it uses GPU if available.
* `--device_id`: which of the available GPU devices should be used for the calculations. In case a given system has more than one GPU available. By default, it uses the device labelled as 0.
* `--outputfile`: Name of the outputfiles. The file extensions (`.txt` and `.pdb`) will be added automatically.
* `--max_coordinators`: Number of maximum coordinators expected. By default, 2. It only affects the range of values assigned to the probes.
* `--residues`: Number of most likely coordinating residues. By default, 10.
* `--stride`: Step at which the voxelized representation of the protein should be parsed. By default, 1. The greater the stride, the greater the computational efficiency; however, the resolution of the predictions will be affected.
* `--pH`: pH of the medium at which the structure is to be evaluated. By default, 7.4.
* `--cluster_radius`: Radius of the clusters to be generated in armstrongs. By default, 5.
* `--cnn_threshold`: Threshold for considering CNN points as possible coordinations. Lower values will impact computational efficiency; greater values, may hide possible coordinating regions. By default, 0.5. Values should be within the range [0, 1].
* `--combined_threshold`: Threshold for considering predictions combining BioMetAll and CNN scores as positive. By default, 0.5. Values should be within the range [0, 1].
* `--voxelsize`: Resolution of the 3D representation. In Arnstrongs. By default, 1 A.
* `--cnn_weight`: Importance of the CNN score in the final score in relations to the BioMetAll score. By default, 0.5. Values should be within the range [0, 1].
* `--verbose`: Information that will be displayed. 0: Only Moleculekit, 1: All. By default, 1.
* `--residue score`: Scoring function for residue coordination analysis. Can be either `discrete`, that only considers how likely is a residue to bind to a certain metal (more computationally efficient); or `gaussian`, that also considers the fitness of the geometrical descriptors for a certain residue and metal. By default, `gaussian`.
* `--threads`: Number of threads available for multithreading calculation. By default it will create 2 threads per physical core.

**Examples:**

Searching for copper.

```bash
> biobrigit 1dhy Cu
```

Searching with generic metal.

```bash
> biobrigit 1dhy General --outputfile 1dhy_general
````

Fast preliminar exploration for binding sites with 4 coordinations, no GPU, and only considering the 4 most likely coordinating residues.

```bash
> biobrigit 1dhy Cu --stride 2 --model TinyBrigit --max_coordinators 4 --device cpu --residues 4
```

Search for small clusters at acidic pH (5.2).

```bash
> biobrigit 1dhy Cu --cluster_radius 3 --pH 5.2
```

License
-------
BioBrigit is an open-source software licensed under the BSD-3 Clause License. Check the details in the [LICENSE](https://github.com/raulfd-creator/biobrigit/blob/master/LICENSE) file.

History of versions
-------------------
* **v.0.1:** First operative release version.

OS Compatibility
----------------
BioBrigit is currently only compatible with Linux, due to some of its dependencies.

If you find some dificulties when installing it in a concrete distribution, please use the issues page to report them.

Credits
-------

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage) project template. 

Special thanks to [Silvia González López](https://www.linkedin.com/in/silvia-gonz%C3%A1lez-l%C3%B3pez-717558221/) for designing the BioBrigit logo.
