BioBrigit
===============

BioBrigit is a computational tool designed for the prediction of protein-metal
binding sites in proteins. It uses a novel scoring function powered by
a deep learning model and previous domain knowledge regarding bioinorganic
interactions.

The deep learning model used is a 3D Convolutional Neural Network (CNN) that
interprets the physico-chemical environment. The previous domain knowledge is 
based on the works by Sánchez-Aparicio et al. (2017), and translates into the 
use of statistics to score the suitability of a point of space based on its 
relative position to certain atoms in the protein backbone.

Features
--------
**Diferent options for customizing the search:**

* Search for binding sites of specific metals.
* Filter results according to how likely they are to bind metals.
* Takes into account binding with backbone nitrogens and oxygens.
* Scanning the whole protein, though, a region can also be provided as input (in PDB format).

**Modular design:**

* The modular design of this package allows for its use as a command-line tool or to be integrated into a larger Python program or pipeline.

**Possible applications:**

* Screening of a pool of `.pdb` structures.
* Identification of conformational changes that alter the formation of metal-binding sites.
* Identification of probable paths that the metals might have to traverse to transitorily binding regions before reaching the more stable, final, binding site.
* Metalloenzyme design.
* Metallodrug design.

Installation
------------
The recommended environment for using BioBrigit takes advatange of GPU acceleration. However, it is possible to use the tool in CPU-only environments.

The first step is to create an environment with the necessary libraries. Some will directly installed from source.

```bash
> conda create -n {name} python=3.9
> conda activate {name}
> pip install git+https://github.com/Acellera/moleculekit
> pip install git+https://github.com/RaulFD-creator/brigit
> conda install pdb2pqr -c acellera -c conda-forge
> pip install scikit-learn
```

### 2.1. Environment set-up with CUDA acelleration

The last step is to install the deep learning framework:

```bash
> conda install pytorch pytorch-cuda -c pytorch -c nvidia
> conda install pytorch-lightning tensorboard torchmetrics -c conda-forge
```

### 2.2. Environment set-up without CUDA acelleration

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



License
-------
BioBrigit is an open-source software licensed under the BSD-3 Clause License. Check the details in the [LICENSE](https://github.com/raulfd-creator/biobrigit/blob/master/LICENSE) file.

TODO
----

* Modify clustering or check clustering functions to identify specific motifs.
* Modify clustering or check clustering to propose mutations.

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
