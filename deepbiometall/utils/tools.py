"""
Various utility functions within the DeepBioMetAll package.

Contains 4 functions:
    - download_pdb
    - get_undesired_channels
    - select_desired_channels
    - voxelize

2 classes:
    - SimpleMetalPDB

and 1 Data set:
    - METAL_RESNAMES
    - CHANNELS
    - ALL_METALS

Includes functions and data sets that are required in different
files throughout the project, thus, preventing redundancies.

Copyright by Raúl Fernández Díaz
"""

import os
import sys
import torch
import numpy as np
import urllib.request
from moleculekit.molecule import Molecule
from moleculekit.tools.atomtyper import prepareProteinForAtomtyping
from moleculekit.tools.voxeldescriptors import getCenters, getVoxelDescriptors
from .data import CHANNELS_DICT, read_stats


class ordered_list():
    def __init__(self):
        self.list = []
        self.dict = {}

    def add(self, other, counts):
        for idx, item in enumerate(self.list):
            if self.dict[item] < counts and counts > 0.0:
                self.list.insert(idx, other)
                self.dict[other] = counts
                return

        self.list.append(other)
        self.dict[other] = counts

    def pop(self, idx):
        del self.dict[self.list[idx]]
        self.list.pop(idx)

    def counts(self, idx):
        return self.dict[self.list[idx]]

    def to_list(self):
        return self.list

    def __len__(self):
        return len(self.list)

    def __str__(self):
        return str(self.list)

    def __repr__(self):
        return str(self.list)


def download_pdb(pdbcode: str, datadir: str,
                 downloadurl: str = 'https://files.rcsb.org/download/',
                 verbose: bool = True):
    """
    Downloads a PDB file from the RCSB-PDB and saves it in a directory.

    Args:
        pdbcode (str): The standard PDB ID e.g. '3ICB' or '3icb'.
        datadir (str): The directory where the downloaded file will be saved.
        downloadurl (str, optional): The base PDB download URL. Defaults to
        'https://files.rcsb.org/download/'.

    Returns:
        output_filename (str): Path of the PDB file.
    """
    pdb_filename = pdbcode + ".pdb"
    url = downloadurl + pdb_filename
    output_filename = os.path.join(datadir, pdb_filename)

    try:
        urllib.request.urlretrieve(url, output_filename)
    except Exception as err:
        if verbose:
            print(str(err), file=sys.stderr)
        return None

    return output_filename


def get_undesired_channels(uchannels, channels):
    """
    Helper function to voxelize() that reads the desired channels for
    voxelization and determines what channels should be turned off.
    It outputs a list of ints with the indexes of the undesired channels.
    """
    global CHANNELS
    working_channels = []
    undesired_channels = []

    for channel in channels:
        if channel not in CHANNELS_DICT.keys():
            raise Exception(
                f"Unsupported channel: {channel}"
                + f"Supported channels: {CHANNELS_DICT.keys()}."
                )
        working_channels.append(CHANNELS_DICT[channel])

    for i in range(8):
        if i not in working_channels:
            undesired_channels.append(i)
    for channel in undesired_channels:
        uchannels[:, channel] = 0

    return uchannels, undesired_channels


def select_desired_channels(protein_vox: np.array, undesired_channels: list,
                            length_centers: int) -> np.array:
    """
    From the moleculekit protein_vox extract only the desired channels.

    Args:
        protein_vox (np.array): Protein voxelized representation.
        undesired_channels (list): List of channels to remove.
        length_centers (int): Number of voxels in the representation.

    Returns:
        np.array: Protein voxelized representation without the undesired
                channels.
    """
    new_protein_vox = np.zeros(
        (length_centers, 8-len(undesired_channels))
    )
    j = 0

    for i in range(8):
        if i not in undesired_channels:
            new_protein_vox[:, j] = protein_vox[:, i]
            j += 1

    return new_protein_vox


def distribute(list, threads):
    length = len(list)
    sublist = [[] for idx in range(threads)]
    i = 0
    j = 0
    while i < length:
        if j == threads:
            j = 0
        sublist[j].append(list[i])
        j += 1
        i += 1
    return sublist


def voxelize(
    path: str,
    site_center: list,
    channels: list,
    buffer: int,
    voxel_resolution: float,
    radius: int,
    protein_atoms: bool = True
):
    """
    Generate voxel representation of a given MFS.

    Args:
        path (str): Path where the PDB file storing the MFS.
        metal_center (list): Coordinates of the metal center.
        channels (list): Channels to be computed during voxelization.
        buffer (int): Number of null voxels to add in every dimension.
        voxel_resolution (float): Resolution of the voxelization. Units in
                                    Arngstrons.
        radius (float): Dimension of the boxes.

    Returns:
        voxel_MFS (torch.Tensor): Voxel representation of the MFS as a Tensor.
    """

    protein = Molecule(path, validateElements=False)
    protein.remove('not protein')

    if protein.numAtoms < 1 and protein_atoms:
        raise RuntimeError('There are no protein atoms involved in the MFS.')

    protein = prepareProteinForAtomtyping(protein, verbose=False)

    centers = getCenters(protein)
    uchannels = np.ones((len(centers[0]), 8))
    uchannels, undesired_channels = get_undesired_channels(uchannels, channels)

    voxels, centers, dimns = getVoxelDescriptors(
                                                protein,
                                                voxelsize=voxel_resolution,
                                                buffer=buffer,
                                                validitychecks=False,
                                                userchannels=uchannels
                                                )

    new_voxels = select_desired_channels(
        voxels, undesired_channels, len(centers)
    )

    if len(new_voxels) == 0 or np.sum(new_voxels) == 0:
        raise RuntimeError("New voxels = 0")

    # From the 2D output create the proper 3D output
    nchannels = new_voxels.shape[1]
    voxels_t = new_voxels.transpose().reshape(
        [1, nchannels, dimns[0], dimns[1], dimns[2]]
        )
    voxels_t = np.array(voxels_t, dtype='float32')
    center_voxel = [i//2 for i in voxels_t.shape]

    # Ensuring radius is correct type
    radius = int(radius)
    voxelized = torch.empty(
        (1, len(channels), int(radius)*2, int(radius)*2, int(radius)*2)
    )
    voxelized[:, :, :, :, :] = torch.from_numpy(
        voxels_t[:, :, center_voxel[2]-radius:center_voxel[2]+radius,
                 center_voxel[3]-radius:center_voxel[3]+radius,
                 center_voxel[4]-radius:center_voxel[4]+radius]
        )
    return voxelized


def find_most_likely_coordinators(metal, num_residues: int = 20):
    stats = read_stats()
    metal_stats = stats[metal]
    residues = ordered_list()
    for residue, res_stats in metal_stats.items():
        if len(residues) < num_residues:
            residues.add(residue, res_stats['fitness'])

        elif (
            res_stats['counts'] > residues.counts(-1) and
            len(residues) == num_residues
        ):
            residues.pop(-1)
            residues.add(residue, res_stats['fitness'])
    return residues.to_list(), metal_stats


def geometric_relations(alphas, betas):
    alpha_distances = np.linalg.norm(alphas, axis=1)
    beta_distances = np.linalg.norm(betas, axis=1)
    alpha_beta_distances = np.linalg.norm(alphas - betas, axis=1)
    ab_angles = np.arccos(
        (
            np.square(alpha_distances) + np.square(alpha_beta_distances) -
            np.square(beta_distances)
        ) /
        (2 * alpha_distances * alpha_beta_distances)
    )
    return alpha_distances, beta_distances, ab_angles


if __name__ == '__main__':
    help(download_pdb)
    help(get_undesired_channels)
    help(select_desired_channels)
    help(voxelize)
