"""
Various utility functions within the Brigit package.

Contains 9 functions:
    - download_pdb
    - get_undesired_channels
    - select_desired_channels
    - voxelize
    - distribute
    - find_coordinators
    - geometry
    - set_up_cuda
    - load_model

1 class:
    - ordered_list

Includes functions and data sets that are required in different
files throughout the project, thus, preventing redundancies.

Copyright by Raúl Fernández Díaz
"""

import os
import sys
import urllib.request

import numpy as np
import torch
from moleculekit.molecule import Molecule
from moleculekit.tools.atomtyper import prepareProteinForAtomtyping
from moleculekit.tools.voxeldescriptors import getCenters, getVoxelDescriptors

from .data import CHANNELS_DICT, read_stats
from .models import BaseModel, BrigitCNN


class ordered_list():
    def __init__(self):
        self.list = []
        self.dict = {}

    def add(self, other, counts):
        for idx, item in enumerate(self.list):
            if self.counts(idx) < counts and counts > 0.0:
                self.list.insert(idx, other)
                self.dict[str(other)] = counts
                return

        self.list.append(other)
        self.dict[str(other)] = counts

    def pop(self, idx):
        del self.dict[str(self.list[idx])]
        self.list.pop(idx)

    def counts(self, idx):
        return self.dict[str(self.list[idx])]

    def to_list(self):
        return self.list

    def __len__(self):
        return len(self.list)

    def __str__(self):
        return str(self.list)

    def __repr__(self):
        return str(self.list)

    def __iter__(self):
        self.n = 0
        self.max = len(self.list)
        return self

    def __next__(self):
        if self.n < self.max:
            result = self.list[self.n]
            self.n += 1
        else:
            raise StopIteration

        return result


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


def find_coordinators(metal: str, num_residues: int = 20):
    stats, gaussian_stats = read_stats()
    stats = stats[metal]
    gaussian_stats = gaussian_stats[metal]
    residues = ordered_list()

    for residue, res_stats in stats.items():
        if residue not in gaussian_stats.keys():
            continue
        if len(residues) < num_residues:
            if res_stats['fitness'] > 0.0:
                residues.add(residue, res_stats['fitness'])
        elif res_stats['fitness'] > residues.counts(-1):
            residues.pop(-1)
            residues.add(residue, res_stats['fitness'])

    coordinators = residues.to_list()

    return coordinators, stats, gaussian_stats


def load_stats(metal: str, residues: dict, backbone: bool):
    stats, gaussian_stats = read_stats()
    stats = stats[metal]
    gaussian_stats = gaussian_stats[metal]
    if backbone:
        coordinators = {
            'residue': residues,
            'backbone_o': residues,
            'backbone_n': residues
        }
    else:
        coordinators = {
            'residue': residues,
            'backbone_o': {'mandatory': {}, 'either': [[]]},
            'backbone_n': {'mandatory': {}, 'either': [[]]}

        }

    return coordinators, stats, gaussian_stats


def geometry(v1, v2):
    v1_distances = np.linalg.norm(v1, axis=1)
    v2_distances = np.linalg.norm(v2, axis=1)
    v1_v2_distances = np.linalg.norm(v1 - v2, axis=1)
    if np.any(v1_v2_distances[:] == 0.0):
        return None, None, None
    v1v2_angles = np.arccos(
        (
            np.square(v1_distances) + np.square(v1_v2_distances) -
            np.square(v2_distances)
        ) /
        (2 * v1_distances * v1_v2_distances)
    )
    return v1_distances, v2_distances, v1v2_angles


def set_up_cuda(device_id: int) -> None:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.fastest = True
    torch.cuda.set_device(device_id)


def load_model(model: str, device: str, **kwargs) -> BaseModel:
    path = os.path.join(
        os.path.dirname(__file__), "trained_models", f'{model}.ckpt'
    )
    if model.lower() == 'brigitcnn':
        model = BrigitCNN.load_from_checkpoint(
            path,
            map_location=device,
            learning_rate=2e-4,
            neurons_layer=64,
            size=12,
            num_dimns=6
        )
    elif model.lower() == 'tinybrigit':
        model = BrigitCNN.load_from_checkpoint(
            path,
            map_location=device,
            learning_rate=2e-4,
            neurons_layer=16,
            size=12,
            num_dimns=6
        )
    else:
        message = f'Model: {model} is not available.'
        message += ' Please select either BrigitCNN or TinyBrigit.'
        raise RuntimeError(message)
    model.to(device)
    model.eval()
    return model


if __name__ == '__main__':
    help(download_pdb)
    help(get_undesired_channels)
    help(select_desired_channels)
    help(voxelize)
    help(geometry)
    help(set_up_cuda)
