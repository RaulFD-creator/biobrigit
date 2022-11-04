===============
BrigitMetalPredictor
===============


.. image:: https://img.shields.io/pypi/v/brigit.svg
        :target: https://pypi.python.org/pypi/brigit

.. image:: https://img.shields.io/travis/RaulFD-creator/brigit.svg
        :target: https://travis-ci.com/RaulFD-creator/brigit

.. image:: https://readthedocs.org/projects/brigit/badge/?version=latest
        :target: https://brigit.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status



Brigit is a computational tool designed for the prediction of protein-metal
binding sites in proteins. It uses a novel scoring function powered by
a deep learning model and previous domain knowledge regarding bioinorganic
interactions.

The deep learning model used is a 3D Convolutional Neural Network (CNN) that
interprets the physico-chemical environment. The previous domain knowledge is 
based on the works by Sánchez-Aparicio et al. (2017), and translates into the 
use of statistics to score the suitability of a point of space based on its 
relative position to certain atoms in the protein backbone.

More information on: 

Software specifications:

* Free software: BSD license
* Documentation: https://brigit.readthedocs.io.

Features
--------


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
