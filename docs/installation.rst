.. highlight:: shell

============
Installation
============


Stable release
--------------

To install Brigit, run this command in your terminal:

*Still not implemented*

Requirements
------------

There are quite a few requirements that have to be installed 
in order for this program to work, so it is recommended to 
initialize a specific conda environment. Further, some of 
this commands will generate directories within your desktop
so it is recommended to create a directory from where to
install the program.

.. code-block:: console

    $ mkdir brigit_install

    $ cd brigit_install

    $ wget https://raw.githubusercontent.com/RaulFD-creator/brigit/master/environment.yml

    $ conda env create -f environment.yml

    $ conda activate brigit

    $ git clone https://github.com/Acellera/moleculekit

    $ pip install moleculekit

The installation of CUDA depends on the version of CUDA available in your machine, by default this is the recommended setup. However, if any problem arises, the version of pytorch-cuda should be changed. 

.. code-block:: console

    $ conda install pytorch pytorch-cuda=11.6 pytorch-lightning torchmetrics -c pytorch -c nvidia -c conda-forge

If you do not have a GPU with CUDA available, instead install the following:

.. code-block:: console

    $ conda install pytorch pytorch-lightning torchmetrics -c conda-forge -c pytorch

From sources
------------

The sources for Brigit can be downloaded from the `Github repo`_.

The first step will be to clone the repository. This process will create a
directory in whichever location you are currently at. It is recommended to
continue in the same directory from the previous step.

.. code-block:: console

    $ git clone https://github.com/RaulFD-creator/brigit

Once you have a copy of the source, without moving from the directory:

.. code-block:: console

    $ cd brigit
    $ pip install .

Or alternatively,

.. code-block:: console

    $ cd brigit
    $ python setup.py install


.. _Github repo: https://github.com/RaulFD-creator/brigit

Solving problems with dependencies
-----------------------------------

