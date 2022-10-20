"""Main module."""
import os
import time
import torch
import numpy as np
from sklearn.cluster import Birch
from moleculekit.molecule import Molecule
from moleculekit.tools.voxeldescriptors import getVoxelDescriptors, getCenters
from moleculekit.tools.atomtyper import prepareProteinForAtomtyping
from .utils.models import BaseModel, BrigitCNN, DeepSite
from .utils.data import CHANNELS
from .utils.tools import (
    get_undesired_channels,
    select_desired_channels,
    find_most_likely_coordinators
)
from .utils.pdb_parser import protein


class DeepBioMetAll():
    def __init__(
        self,
        device: str,
        device_id: int,
        **kwargs
    ):
        if device == 'cuda':
            self.device = f'{device}:{device_id}'
            set_up_cuda(device_id)
        else:
            self.device = device
        self.model = load_model(kwargs['model'], device)

    def create_PDB(
        self,
        target: str,
        outputfile: str,
        scores: np.array,
        threshold: float,
        centers: np.array,
        molecule: protein,
        **kwargs
    ) -> None:
        """
        Generate an output file in PDB format so that it can be processed with
        the corresponding visualization tools.

        Arguments
        ---------
        output_dir : str
            Path to the directory where output files should be stored.

        target_name : str
            Name of the output file.

        protein_scores : np.array
            Scores obtained from self.evaluate method.

        threshold : float, optional
            Minimum value to consider a prediction as positive.
        """
        outputfile = f'{outputfile}_brigit.pdb'
        with open(outputfile, "w") as fo:
            num_at = 0
            num_res = 0
            for entry in scores:
                close2protein = False
                for atom in molecule.atoms:
                    if np.linalg.norm(atom.coordinates - entry[:3]) < 3.5:
                        close2protein = True

                if not close2protein:
                    continue

                if entry[3] > threshold:
                    num_at += 1
                    num_res = 1
                    ch = "A"
                    prb_str = ""

                    for idx in range(3):
                        number = str(round(float(entry[idx]), 3))
                        prb_center = "{:.8s}".format(number)
                        if len(prb_center) < 8:
                            prb_center = " "*(8-len(prb_center)) + prb_center
                            prb_str += prb_center
                        else:
                            prb_str += prb_center

                    atom = "HE"
                    blank = " "*(7-len(str(num_at)))
                    fo.write("ATOM" + blank + "%s  %s  SLN %s" %
                             (num_at, atom, ch))
                    blank = " "*(3-len(str(num_res)))
                    score = str(round(entry[3], 2))
                    score = score if len(score) == 4 else score + '0'
                    fo.write(blank + "%s     %s  1.00  %s          %s\n" %
                             (num_res, prb_str, score, atom))

            for entry in centers.values():
                num_at += 1
                num_res = 1
                ch = "A"
                prb_str = ""

                for idx in range(3):
                    number = str(round(float(entry['center_coors'][idx]), 3))
                    prb_center = "{:.8s}".format(number)
                    if len(prb_center) < 8:
                        prb_center = " "*(8-len(prb_center)) + prb_center
                        prb_str += prb_center
                    else:
                        prb_str += prb_center

                atom = "AR"
                blank = " "*(7-len(str(num_at)))
                fo.write("ATOM" + blank + "%s  %s  SLN %s" %
                         (num_at, atom, ch))
                blank = " "*(3-len(str(num_res)))
                score = str(entry['score'])
                score = score if len(score) == 4 else score + '0'
                fo.write(blank + "%s     %s  1.00  %s          %s\n" %
                         (num_res, prb_str, score, atom))

    def voxelize(
        self,
        target: str,
        voxelsize: float = 1.0,
        buffer: int = 6,
        validitychecks: bool = False,
        channels: list = CHANNELS,
        **kwargs
    ):
        if isinstance(target, str):
            protein = Molecule(target)
        elif isinstance(target, Molecule):
            protein = target
        else:
            raise TypeError(
                'Target input has to be either: a) PDB ID, b) path to local\
                    PDB file, c) Molecule object.'
            )

        protein.remove("not protein")
        protein = prepareProteinForAtomtyping(protein, verbose=0)
        centers = getCenters(protein)
        uchannels = np.ones((len(centers[0]), 8))
        uchannels, undesired_channels = get_undesired_channels(
            uchannels, channels
        )

        vox, p_centers, p_N = getVoxelDescriptors(
            protein,
            voxelsize=voxelsize,
            buffer=buffer,
            validitychecks=validitychecks,
            userchannels=uchannels
        )

        new_vox = select_desired_channels(
            vox, undesired_channels, len(p_centers)
        )

        # From the 2D output create the proper 3D output
        nchannels = new_vox.shape[1]
        final_vox = new_vox.transpose().reshape(
            [1, nchannels, p_N[0], p_N[1], p_N[2]]
        )
        nvoxels = np.array([p_N[0], p_N[1], p_N[2]])

        return final_vox, p_centers, nvoxels

    def evaluate(
        self,
        vox: np.array,
        p_centers: np.array,
        voxelsize: float = 1.0,
        region_size: int = 12,
        stride: int = 1,
        occupancy_restrictions: float = 0.4,
        **kwargs
    ) -> np.array:
        border = region_size // 2
        size_x, size_y, size_z = vox.shape[2], vox.shape[3], vox.shape[4]
        max_x, max_y, max_z = size_x-border, size_y-border, size_z-border

        # Create an empty representation of the protein to be filled
        # with the metal-bindingness prediction
        scores = np.zeros((size_x, size_y, size_z))
        scores = torch.Tensor(scores).to('cpu')
        vox = torch.tensor(vox, device=self.device).float()

        x_dim = len(range(border, max_x, stride))
        y_dim = len(range(border, max_y, stride))
        z_dim = len(range(border, max_z, stride))
        num_points = x_dim * y_dim * z_dim
        counter = 0

        for x in range(border, max_x, stride):
            for y in range(border, max_y, stride):
                for z in range(border, max_z, stride):

                    if vox[:, 5, x, y, z] > occupancy_restrictions:
                        continue

                    if (counter % (num_points // 30) == 0):
                        print(f'{round((counter/num_points)*100, 2)}%')
                    x1, x2 = x - border, x + border
                    y1, y2 = y - border, y + border
                    z1, z2 = z - border, z + border
                    output = self.model(
                        vox[:, :, x1: x2, y1: y2, z1: z2]
                        ).detach().cpu()
                    scores[x, y, z] = output

                    counter += 1
        print()
        scores = scores.reshape(-1)
        scores = scores.detach().cpu().numpy()
        scores = np.c_[p_centers, scores]

        return scores

    def predict(
        self,
        target: str,
        metal: str,
        max_coordinators: int = 4,
        outputfile: str = None,
        cnn_threshold: float = 0.5,
        combined_threshold: float = 0.5,
        verbose: int = 1,
        clustering_threshold: float = 4.0,
        **kwargs
    ) -> np.array:
        start = time.time()
        if not isinstance(target, str):
            message = 'Target has to be:\n  a) string with PDB ID or\n'
            message += '  b) path to local PDB file.'
            raise TypeError(message)

        if combined_threshold > cnn_threshold:
            cnn_threshold = combined_threshold

        if verbose == 1:
            print(f'Voxelizing target: {target}', end='\n\n')

        vox, p_centers, nvoxels = self.voxelize(target, **kwargs)

        if verbose == 1:
            print(f'\n\nEvaluating target: {target}', end='\n\n')

        scores = self.evaluate(vox, p_centers, **kwargs)
        scores, molecule = self.coordination_analysis(
            target, max_coordinators, metal, scores, cnn_threshold,
            **kwargs
        )
        best_scores = np.argwhere(scores[:, 3] > combined_threshold)
        new_scores = np.zeros((len(best_scores), 4))
        for idx, idx_score in enumerate(best_scores):
            new_scores[idx, :] = scores[idx_score, :]
        centers = self.clusterize(
            new_scores, molecule, clustering_threshold
        )

        if outputfile is None:
            if target.endswith('.pdb'):
                outputfile = target.split('.')[0]
            else:
                outputfile = target
        self.create_PDB(
            target, outputfile, new_scores, combined_threshold, centers,
            molecule
        )
        self.check_clusters(centers, molecule, outputfile, kwargs['args'])
        end = time.time()
        print(f'Computation took {end-start} s.', end='\n\n')
        return scores

    def coordination_analysis(
        self, target, max_coordinators, metal, scores, threshold, cnn_weight,
        **kwargs
    ):
        molecule = protein(target, True)
        residues, o_residues, n_residues, metal_stats = (
            find_most_likely_coordinators(metal, kwargs['residues'])
        )
        molecule.set_stats(metal_stats, max_coordinators)
        molecule.parse_residues(residues, o_residues, n_residues)

        for idx, probe in enumerate(scores):
            if probe[3] > threshold:
                scores[idx, 3] = molecule.coordination_score(probe, cnn_weight)
        return scores, molecule

    def clusterize(
        self, scores, molecule, clustering_threshold
    ):
        clustering = Birch(threshold=clustering_threshold, n_clusters=None)
        try:
            labels = clustering.fit_predict(scores[:, :3])
        except ValueError:
            print('There are not enough predicted points as to properly\
                  \nclusterize.')
            exit()
        clusters = {}
        result = {}
        for idx, score in enumerate(scores):
            try:
                clusters[labels[idx]].append(score)
            except KeyError:
                clusters[labels[idx]] = []

        for name, cluster in clusters.items():
            cluster = np.array(cluster)
            if len(cluster) == 0:
                continue
            cluster_mean = np.average(
                cluster[:, :3], axis=0, weights=cluster[:, 3]
            )
            score_mean = np.average(cluster[:, 3])
            result[name] = {
                'center_coors': cluster_mean,
                'score': score_mean
            }

        return result

    def check_clusters(self, clusters, molecule, outputfile, args):
        outputfile_name = outputfile + '.clusters'
        with open(outputfile_name, 'w') as writer:
            writer.write(f'{args}\n')
            writer.write('cluster,coordinators,score\n')
            for name, center in clusters.items():
                coordinator_found = False
                for residue in molecule.residues:
                    for atom in residue.atoms:
                        if atom.element not in ['N', 'O']:
                            continue
                        if (
                            np.linalg.norm(
                                clusters[name]['center_coors'] -
                                atom.coordinates
                            ) < 3.5
                        ):
                            if not coordinator_found:
                                writer.write(f'{name},')
                            coordinator_found = True
                            writer.write(f'{atom.name}_{residue.id};')
                if coordinator_found:
                    writer.write(f",{clusters[name]['score']}\n")


def load_model(model: str, device: str, **kwargs) -> BaseModel:
    path = os.path.join(
        os.path.dirname(__file__), "trained_models", f'{model}.ckpt'
    )
    if model == 'NewBrigit_2':
        model = BrigitCNN.load_from_checkpoint(
            path,
            map_location=device,
            learning_rate=2e-4,
            neurons_layer=64,
            size=12,
            num_dimns=6
        )
    elif model == 'DeepSite':
        model = DeepSite.load_from_checkpoint(
            path,
            map_location=device,
            learning_rate=2e-4
        )

    model.to(device)
    model.eval()
    return model


def set_up_cuda(device_id: int) -> None:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.fastest = True
    torch.cuda.set_device(device_id)


def run(args: dict):
    print(args, end="\n\n")
    args['args'] = args
    deepbiometall = DeepBioMetAll(**args)
    deepbiometall.predict(**args)


"""
# Might be a useful solution for silencing all
# annoying outputs
from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
"""

if __name__ == '__main__':
    help(DeepBioMetAll)
