.. highlight:: shell

============
Installation
============


Stable release
--------------

To install Brigit, run this command in your terminal:

.. code-block:: console

    $ pip install brigit

This is the preferred method to install Brigit, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for Brigit can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/RaulFD-creator/brigit

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/RaulFD-creator/brigit/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ pip install brigit

Or alternatively,

.. code-block:: console

    $ python setup.py install


.. _Github repo: https://github.com/RaulFD-creator/brigit
.. _tarball: https://github.com/RaulFD-creator/brigit/tarball/master

Requirements
------------

There are quite a few requirements that have to be installed 
in order for this program to work, so it is recommended to 
initialize a specific conda environment.

.. code-block:: console

    $ conda create -n brigit python=3.8

    $ conda activate brigit

    $ conda install moleculekit -c acellera

    $ wget https://raw.githubusercontent.com/Acellera/moleculekit/master/extra_requirements.txt

    $ conda install --file extra_requirements.txt -c acellera -c conda-forge


For CUDA support,

.. code-block:: console

    $ conda install pytorch torchvision pytorch-cuda=11.6 -c pytorch -c nvidia

    $ conda install pytorch-lightning -c conda-forge

If you do not have a GPU in your system,

.. code-block:: console

    $ conda install pytorch pytorch-lightning -c conda-forge