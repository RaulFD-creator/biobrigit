"""Main module."""
import os
import time
import torch
import numpy as np
from moleculekit.molecule import Molecule
from moleculekit.tools.voxeldescriptors import getVoxelDescriptors, getCenters
from moleculekit.tools.atomtyper import prepareProteinForAtomtyping
from deepbiometall.utils.models import BaseModel, BrigitCNN
from deepbiometall.utils.data import CHANNELS
from deepbiometall.utils.tools import (
    get_undesired_channels, select_desired_channels
)


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
        outputdir: str,
        scores: np.array,
        threshold: float = 0.5,
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
        output_path = os.path.join(outputdir, target+'_brigit_results.pdb')
        with open(output_path, "w") as fo:
            num_at = 0
            num_res = 0
            for entry in scores:
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
                    atom = "NE"
                    blank = " "*(7-len(str(num_at)))
                    fo.write("ATOM" + blank + "%s  %s  SLN %s" %
                             (num_at, atom, ch))
                    blank = " "*(3-len(str(num_res)))
                    fo.write(blank + "%s     %s  1.00  0.00          %s\n" %
                             (num_res, prb_str, atom))

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
                    x1, x2 = x-border, x+border
                    y1, y2 = y-border, y+border
                    z1, z2 = z-border, z+border
                    output = self.model(
                        vox[:, :, x1:x2, y1:y2, z1:z2]
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
        outputdir: str = '.',
        threshold: float = 0.5,
        verbose: int = 1,
        **kwargs
    ) -> np.array:
        start = time.time()
        if not isinstance(target, str):
            message = 'Target has to be:\n  a) string with PDB ID or\n'
            message += '  b) path to local PDB file.'
            raise TypeError(message)

        if verbose == 1:
            print(f'Voxelizing target: {target}', end='\n\n')

        vox, p_centers, nvoxels = self.voxelize(target, **kwargs)

        if verbose == 1:
            print(f'\n\nEvaluating target: {target}', end='\n\n')

        scores = self.evaluate(vox, p_centers, **kwargs)
        self.create_PDB(target, outputdir, scores, threshold)

        end = time.time()
        print(f'Computation took {end-start} s.', end='\n\n')
        return scores


def load_model(model: str, device: str, **kwargs) -> BaseModel:
    path = os.path.join(os.path.dirname(__file__), "trained_models", model)
    model = BrigitCNN.load_from_checkpoint(
        path,
        map_location=device,
        learning_rate=2e-4,
        neurons_layer=32,
        size=12,
        num_dimns=6
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
    deepbiometall = DeepBioMetAll(**args)
    deepbiometall.predict(**args)


if __name__ == '__main__':
    help(DeepBioMetAll)
