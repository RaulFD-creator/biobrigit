import argparse
import multiprocessing
import deepbiometall


def parse_cli() -> dict:
    """Console script for deepbiometall."""
    p = argparse.ArgumentParser()
    p.add_argument("target", type=str,
                   help="Molecule PDB file to be analysed.")
    p.add_argument("metal", type=str,
                   help="Symbol of the metal that is to be located.")
    p.add_argument("--model", type=str, default='NewBrigit_2.ckpt',
                   help="Name of the model to be used.")
    p.add_argument("--device", type=str, default='cuda',
                   help="Device in which calculations will be run.")
    p.add_argument("--device_id", type=int, default=0,
                   help="GPU ID in which the calculations will be run.")
    p.add_argument("--outputfile", type=str, default=None,
                   help='Path where the output should be written.')
    p.add_argument("--max_coordinators", type=int, default=4,
                   help="Number of residues that need to be coordinating a\
                   given metal.")
    p.add_argument("--num_possible_residues", type=int, default=15,
                   help="Number of most likely residues to consider.")
    p.add_argument(
        "--stride", type=int, default=1,
        help="Step of the sliding window when evaluating the protein."
    )
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
    args = p.parse_args()

    print()
    return vars(args)


def welcome() -> None:
    message = "Using DeepBioMetAll by Raúl Fernández-Díaz"
    print("-" * (len(message) + 4))
    print("| " + message + " |")
    print("-" * (len(message) + 4))
    print()


def main():
    multiprocessing.freeze_support()
    welcome()
    args = parse_cli()
    deepbiometall.run(args)


if __name__ == '__main__':
    main()
