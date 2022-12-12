===============
BioBrigit
===============

.. image:: https://img.shields.io/pypi/v/biobrigit.svg
        :target: https://pypi.python.org/pypi/biobrigit

.. image:: https://img.shields.io/travis/RaulFD-creator/biobrigit.svg
        :target: https://travis-ci.com/RaulFD-creator/biobrigit

.. image:: https://readthedocs.org/projects/biobrigit/badge/?version=latest
        :target: https://biobrigit.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status



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

License
-------
Brigit is an open-source software licensed under the BSD-3 Clause License. Check the details in the `LICENSE <https://github.com/raulfd-creator/biobrigit/blob/master/LICENSE>`_ file.

TODO
----

* Modify clustering or check clustering functions to identify specific motifs.
* Modify clustering or check clustering to propose mutations.

History of versions
-------------------
* **v.0.1:** First operative release version.

OS Compatibility
----------------
Brigit is currently only compatible with Linux, due to some of its dependencies.

If you find some dificulties when installing it in a concrete distribution, please use the issues page to report them.


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
