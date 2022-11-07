=====
Usage
=====

Console use
------------

The default use of Brigit is simple

.. code-block:: console

    $ brigit {target} {metal}

For example: 

.. code-block:: console

    $ brigit 1dhy Fe

An output file name can be set: 

.. code-block:: console

    $ brigit 1dhy Fe --outputfile output_name

Furthermore, the maximum number of coordinators to consider 
(`--max_coordinators`) or the threshold score (`--combined_threshold`), can
also be modified:

.. code-block:: console

    $ brigit 1dhy Fe --max_coordinators 4 --combined_threshold 0.75

Finally, the importance of each of the elements of the hybrid scoring
function can be tuned (`--cnn_weight`), as well as the scoring function used for
the coordination analysis (`--residue_score`):

.. code-block:: console

    $ brigit 1dhy Fe --cnn_weight 0.3 --residue_score gaussian

Integration in other projects
-----------------------------

To use Brigit in a project::

    import brigit
    predictor = brigit.Brigit()
    Brigit.run(*args)
