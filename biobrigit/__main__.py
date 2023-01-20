"""
Execution BioBrigit module.

Copyrigth by Raúl Fernández Díaz
"""
import argparse
import multiprocessing

import biobrigit


def parse_cli() -> dict:
    """
    Console script for brigit.
    """
    p = argparse.ArgumentParser()
    p.add_argument("target", type=str,
                   help="Molecule PDB file to be analysed.")
    p.add_argument("metals", type=str,
                   help="Symbol of the metal that is to be located. More than one metal can be used by separting them with a comma: METAL1,METAL2")
    p.add_argument("--model", type=str, default='BrigitCNN',
                   help="Name of the model to be used.")
    p.add_argument("--device", type=str, default='cuda',
                   help="Device in which calculations will be run.")
    p.add_argument("--device_id", type=int, default=0,
                   help="GPU ID in which the calculations will be run.")
    p.add_argument("--outputfile", type=str, default=None,
                   help='Path where the output should be written.')
    p.add_argument("--max_coordinators", type=int, default=2,
                   help="Number of residues that need to be coordinating a\
                   given metal.")
    p.add_argument("--residues", type=int, default=10,
                   help="Number of most likely residues to consider.")
    p.add_argument(
        "--stride", type=int, default=1,
        help="Step of the sliding window when evaluating the protein."
    )
    p.add_argument("--pH", type=float, default=7.4,
                   help='pH where the structure is to be evaluated.')
    p.add_argument("--cluster_radius", type=float, default=5.0,
                   help="Threshold used for the Birch clustering algorithm.")
    p.add_argument("--cnn_threshold", type=float, default=0.5,
                   help="Threshold for considering CNN points as possible\
                    coordinations.")
    p.add_argument("--combined_threshold", type=float, default=0.5,
                   help="Threshold for considering predictions positive\
                    combining BioMetAll and CNN scores.")
    p.add_argument("--voxelsize", type=float, default=1.0,
                   help="Resolution of the 3D representation. In Arnstrongs.")
    p.add_argument("--cnn_weight", type=float, default=0.5,
                   help='Weight assigned to CNN.')
    p.add_argument(
        "--verbose", type=int, default=1,
        help="Information that will be displayed. 0: Only Moleculekit, 1: All."
    )
    p.add_argument("--residue_score", type=str, default='gaussian',
                   help='Scoring function for residue coordination analysis.')
    p.add_argument("--threads", type=int, default=0,
                   help='Number of threads available for multithreading\
                    calculation. By default it will create 2 threads per\
                    physical core.')
    args = vars(p.parse_args())
    args['metals'] = args['metals'].split(',')
    print()
    return args


def welcome() -> None:
    """
    Print a welcoming message when the program is executed.
    """
    message = "Using BioBrigit by Raúl Fernández-Díaz"
    print("-" * (len(message) + 4))
    print("| " + message + " |")
    print("-" * (len(message) + 4))
    print()


def main():
    """
    Execute the program.
    """
    multiprocessing.freeze_support()
    welcome()
    args = parse_cli()
    biobrigit.run(args)


if __name__ == '__main__':
    main()
