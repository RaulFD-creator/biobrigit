"""
Main BrigitMetalPredictor module.

Contains 1 class:
    - Brigit

Copyright by Raúl Fernández Díaz
"""
import time
import multiprocessing
import warnings
import torch
import numpy as np
from sklearn.cluster import Birch
from moleculekit.molecule import Molecule
from moleculekit.tools.voxeldescriptors import getVoxelDescriptors, getCenters
from moleculekit.tools.atomtyper import prepareProteinForAtomtyping
from .utils.data import CHANNELS
from .utils.tools import (
    get_undesired_channels,
    select_desired_channels,
    find_coordinators,
    ordered_list,
    set_up_cuda,
    load_model,
    load_stats,
    distribute
)
from .utils.pdb_parser import protein
from .utils.scoring import (
    parse_residues,
    parse_residues_motif,
    coordination_scorer,
    motif_scorer,
    discrete_score,
    gaussian_score
)


class Brigit():
    """
    Main class encapsulating the necessary attributes and methods
    for predicting protein-metal binding regions.
    """
    def __init__(
        self,
        device: str,
        device_id: int,
        model: str,
        **kwargs
    ):
        """
        Initialize a new instance of the class.
        Set CUDA up if any GPU device is available.
        """
        # TODO: Verify that cuda device is available before setting it
        # up.
        if device == 'cuda':
            if torch.cuda.is_available():
                self.device = f'{device}:{device_id}'
                set_up_cuda(device_id)
            else:
                warnings.warn('Warning: No CUDA compatible GPU was available,\
                    defaulting to CPU.')
                self.device = 'cpu'
        else:
            self.device = device
            if torch.cuda.is_available():
                warnings.warn('Using CPU for caluclations, although there is\
                    a CUDA compatible GPU available. Using the GPU would\
                    accelerate significantly the calculations.')
        self.model = load_model(model, device)

    def predict(
        self,
        target: str,
        metal: str,
        max_coordinators: int = 4,
        outputfile: str = None,
        cnn_threshold: float = 0.5,
        combined_threshold: float = 0.5,
        verbose: int = 1,
        cluster_radius: float = 5.0,
        motif: str = None,
        **kwargs
    ) -> np.array:
        """
        Main method of the class. It takes a protein target and it
        predicts the suitability of its regions for coordinating metals.

        First, the protein is voxelized using the Moleculekit library. 6
        channels are computed: hidrophobicity, H-bond acceptance, H-bond
        reception, positive ionizability, negative ionizability, excluded
        volume. The voxelized representation is parsed with the CNN and each
        point is evaluated. Then, all points with scores superior to the
        threshold are further analysed. In this case, their relative position
        to key atoms in the protein backbone will provide a statistical score
        that serves as indicator as to how well coordinated a hypothetical
        metal ion will be.

        The resulting points will be clustered attending to their spatial
        coordinates and the centers of the clusters will be assigned as
        a weighted mean of the coordinates of all points within the cluster.
        The weights will be the square of the scores. So, that the center is
        closer to highly coordinable regions.

        Finally, two output files will be generated:
            1. A '_brigit.pdb' file which contains all probes with a score
                superior to `combined_threshold`.
            2. A '.clusters' file which contains all residues within
                `cluster_radius` of every cluster center. Thus, summarising
                the most likely coordinating residues in the protein. Each
                cluster will have assigned a global score which will be a sum
                of the scores of all the probes that comprise it.

        Args:
            target (str): Path to the PDB file or PDB code of the target
                protein.
            metal (str): IUPAC symbol of the metal to be evaluated. There is a
                mode, 'General', that uses statistics built from all metals
                simultaneously.
            max_coordinators (int, optional): Maximum number of coordinators
                expected. Value affects the sensibility of the coordination
                analysis. Defaults to 4.
            outputfile (str, optional): Path where the output files are to
                be stored. It is recommended that it does not contain tags,
                e.g., '.pdb', as the program itself will asign them.
                Defaults to None.
            cnn_threshold (float, optional): Score threshold for the output of
                the CNN model to be further analysed. Defaults to 0.5.
            combined_threshold (float, optional): Final threshold to consider a
                region. Defaults to 0.5.
            verbose (int, optional): Ammount of information to be displayed
                during computation. Defaults to 1.
            cluster_radius (float, optional): Radius of the clusters used to
                organise the results. Affects mainly the final summary.
                Defaults to 5.0.

        Raises:
            TypeError: In case the target is neither a PDB code nor a Path to a
                local PDB file.

        Returns:
            np.array: Set of coordinates and their score.
        """
        start = time.time()
        verbose = bool(verbose)

        # Check that target is either a PDB file or PDB code
        if not isinstance(target, str):
            message = 'Target has to be:\n  a) string with PDB ID or\n'
            message += '  b) path to local PDB file.'
            raise TypeError(message)

        # Verify that the final threshold is not greater than
        # CNN threshold
        if combined_threshold > cnn_threshold:
            cnn_threshold = combined_threshold

        # Verify and prepare system for motif search
        if motif is not None:
            residues = {'mandatory': {}, 'either': [[]]}
            new_residues = motif.split(',')

            for residue in new_residues:
                if '/' in residue:
                    residues['either'].append(residue.split('/'))
                else:
                    try:
                        residues['mandatory'][residue] += 1
                    except KeyError:
                        residues['mandatory'][residue] = 1

            del new_residues
            kwargs['residues'] = residues

        # Voxelize the protein
        if verbose:
            print(f'Voxelizing target: {target}', end='\n\n')

        vox, p_centers, nvoxels = self.voxelize(target, **kwargs)

        # CNN evaluation
        if verbose:
            print(f'\nCNN evaluation of target: {target}', end='\n\n')

        scores = self.evaluate(vox, p_centers, **kwargs)

        # Coordination analysis
        if verbose:
            print(f'\nCoordination analysis of target: {target}', end='\n\n')

        scores, molecule, coordinators = self.coordination_analysis(
            target, max_coordinators, metal, scores, cnn_threshold,
            verbose, **kwargs
        )

        if verbose:
            print('Clusterizing results', end='\n\n')
        # Selection of best positions
        best_scores = np.argwhere(scores[:, 3] > combined_threshold)
        new_scores = np.zeros((len(best_scores), 4))
        for idx, idx_score in enumerate(best_scores):
            new_scores[idx, :] = scores[idx_score, :]

        # Spatial clusterization of best positions
        centers, cluster_scores = self.clusterize(
            new_scores, molecule, cluster_radius
        )

        if verbose:
            print('Preparing and writing output files', end='\n\n')

        # Preparing and writing output files
        if outputfile is None:
            if target.endswith('.pdb'):
                outputfile = target.split('.')[0]
            else:
                outputfile = target
        self.create_PDB(
            target, outputfile, new_scores, combined_threshold, cluster_scores,
            molecule
        )
        self.check_clusters(
            centers, molecule, outputfile, coordinators, cluster_radius,
            kwargs['args']
        )
        end = time.time()

        if verbose:
            print(f'Computation took {round(end-start, 2)} s.', end='\n\n')
        return scores

    def coordination_analysis(
        self,
        target: protein,
        max_coordinators: int,
        metal: str,
        scores: np.array,
        threshold: float,
        verbose: bool,
        residue_score: str,
        backbone_score: str,
        cnn_weight: float,
        residues: int,
        threads: int,
        motif_backbone: bool,
        **kwargs
    ) -> tuple:
        """
        It uses a series of statistical values regarding backbone
        preorganization of metal-binding amino acids, to evaluate
        how well coordinated each probe would be.

        There are 2 main functions that can be used for this scoring:
            1. 'discrete': which uses the median values of the statistics
                and evaluates whether a certain geometric requirements are
                fulfilled or not.
            2. 'gaussian': which instead of using a boolean argument, scores
                more positively those regions as they are closer to the actual
                central values in a continuos fashion.

        More details on these functions can be found in the Docstrings for
        the module `utils/scoring`.

        Args:
            target (protein): Path to the PDB file or PDB code of the target
                protein.
            max_coordinators (int): Maximum number of coordinators
                expected. Value affects the sensibility of the coordination
                analysis.
            metal (str): IUPAC symbol of the metal to be evaluated. There is a
                mode, 'General', that uses statistics built from all metals
                simultaneously.
            scores (np.array): Array with dimensions (len(probes), 4) where the
                first 3 dimensions describe the cartesian coordinates occupied
                by the probe, and the fourth is its score.
            threshold (float): (CNN) score value that determines whether the
                coordination of a certain probe will be analysed or not.
            verbose (bool): Ammount of information to be displayed
                during computation.
            residue_score (str): Scoring function used to evaluate the
                coordination through side chains.
            backbone_score (str): Scoring function used to evaluate the
                coordination through backbone atoms.
            cnn_weight (float): Proportion of the final score that will be
                informed by the CNN. Its complementary (1 - cnn_weight)
                corresponds to the proportion of the final score informed
                by the coordination analysis.
            residues (int): Number of the most likely residue coordinators for
                any specific metal.
            threads (int): Number of threads to be used during the
                voxelization.

        Raises:
            IndexError: Occurs in the event that the residue coordination
                scoring function selected is not implemented.
            IndexError: Occurs in the event that the backbone coordination
                scoring function selected is not implemented.

        Returns:
            tuple: (np.array, protein) the np.array corresponds to
                the updated scores array with the cartesian coordiantes
                of the probes and their scores.
        """
        zeros = np.zeros(np.shape(scores[:, 3]))
        molecule = protein(target, True)
        if isinstance(residues, dict):
            coordinators, stats, gaussian_stats = load_stats(
                metal, residues, motif_backbone
            )
            molecule_info = parse_residues_motif(molecule, coordinators)
            mode = 'motif_detection'
        else:
            coordinators, stats, gaussian_stats = find_coordinators(
                metal, residues
            )
            molecule_info = parse_residues(molecule, coordinators)
            mode = 'normal_evaluation'

        if residue_score == 'discrete':
            residue_score = discrete_score
        elif residue_score == 'gaussian':
            residue_score = gaussian_score
        else:
            raise IndexError(
                f'Residue coordination scoring function: {residue_score} not\
                    currently implemented.'
            )

        if backbone_score == 'discrete':
            backbone_score = discrete_score
        else:
            raise IndexError(
                f'Backbone coordination scoring function: {backbone_score} not\
                    currently implemented.'
            )
        chunks = distribute(scores, threads)
        args = [(molecule, chunks[idx], stats, gaussian_stats, molecule_info,
                coordinators, residue_score, backbone_score, max_coordinators,
                threshold, cnn_weight, mode) for idx in range(threads)]
        pool = multiprocessing.Pool(threads)
        zeros = pool.starmap(analyse_probes, args)
        new_zeros = []
        for element in zeros:
            for subelement in element:
                new_zeros.append(subelement)

        new_zeros = np.array(new_zeros)

        return new_zeros, molecule, coordinators

    def voxelize(
        self,
        target: str,
        voxelsize: float = 1.0,
        buffer: int = 6,
        validitychecks: bool = False,
        channels: list = CHANNELS,
        **kwargs
    ) -> tuple:
        """
        Creates a voxelized representation of target that contains structural
        information regarding the spatial distribution of its physico-chemical
        properties. It is powered by the Moleculekit library.

        Args:
            target (str): Path to the PDB file or PDB code of the target
                protein.
            voxelsize (float, optional): Distance between the different points
                in the voxelization. Defaults to 1.0.
            buffer (int, optional): Space generated around the protein so that
                the model might analyse its borders. Its value should be as big
                as the outmost layer of the CNN model. Defaults to 6.
            validitychecks (bool, optional): Whether the voxelization program
                will check the integrity of the PDB file. Its best to set it to
                False, specially in the case of screening. Defaults to False.
            channels (list, optional): List of channels allowed for
                voxelization. In any standard use the channels are limited by
                the training of the CNN model and should not be changed.
                However, in case anyone has developed their own model they can
                be changed. Defaults to CHANNELS.

        Raises:
            TypeError: In case the target is neither a PDB code nor a Path to a
                local PDB file.

        Returns:
            tuple: (final_vox, p_centers, nvoxels) First element is the
                voxelized representation of the protein; second element, the
                real cartesian coordinates of each of the voxels in the first
                element; the third element, is the number of such voxels in
                the image.
        """
        if isinstance(target, str):
            protein = Molecule(target, validateElements=False)
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
        # with the coordination score prediction
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
                    if (counter % (num_points // 30) == 0):
                        print(f'{round((counter/num_points)*100, 2)}%')
                    counter += 1

                    if vox[:, 5, x, y, z] > occupancy_restrictions:
                        continue

                    x1, x2 = x - border, x + border
                    y1, y2 = y - border, y + border
                    z1, z2 = z - border, z + border
                    output = self.model(
                        vox[:, :, x1: x2, y1: y2, z1: z2]
                        ).detach().cpu()
                    scores[x, y, z] = output

        print()
        scores = scores.reshape(-1)
        scores = scores.detach().cpu().numpy()
        scores = np.c_[p_centers, scores]

        return scores

    def clusterize(
        self, scores, molecule, cluster_radius
    ):
        clustering = Birch(threshold=cluster_radius, n_clusters=None)
        try:
            labels = clustering.fit_predict(scores[:, :3])
        except ValueError:
            raise ValueError('There are not enough predicted points as to\
                properly clusterize.')

        clusters = {}
        mean_clusters = {}
        result = ordered_list()
        for idx, score in enumerate(scores):
            try:
                clusters[labels[idx]].append(score)
            except KeyError:
                clusters[labels[idx]] = []

        for name, cluster in clusters.items():
            cluster = np.array(cluster)
            if len(cluster) == 0:
                continue
            try:
                cluster_mean = np.average(
                    cluster[:, :3], axis=0, weights=cluster[:, 3] ** 2
                )
            except ZeroDivisionError:
                cluster_mean = np.average(cluster[:, :3], axis=0)
            score_sum = np.sum(cluster[:, 3])
            score_mean = np.mean(cluster[:, 3])
            result.add(cluster_mean, score_sum)
            str_mean = ",".join(
                str(round(float(cluster_mean[i]), 3)) for i in range(3)
            )
            mean_clusters[str_mean] = score_mean
        return result, mean_clusters

    def check_clusters(
        self,
        clusters,
        molecule,
        outputfile,
        coordinators,
        cluster_radius,
        args
    ):
        outputfile_name = f'{outputfile}.clusters'
        with open(outputfile_name, 'w') as writer:
            writer.write(f'{args}\n')
            writer.write(f'{coordinators}\n')
            writer.write('cluster,coordinators,score\n')
            for name, center in enumerate(clusters):
                coordinator_found = False
                line = []
                for residue in molecule.residues:
                    for atom in residue.atoms:
                        if atom.element not in ['N', 'O', 'S']:
                            continue
                        if (
                            np.linalg.norm(
                                atom - center
                            ) < cluster_radius
                        ):
                            if not coordinator_found:
                                writer.write(f'{name},')
                            coordinator_found = True
                            res = residue.id.split('_')
                            res2 = res[1].split('(')
                            res2[1] = res2[1].strip(')')
                            message = f"{res[0]}:{res2[0]}:{res2[1]}"
                            if message not in line:
                                line.append(message)
                if coordinator_found:
                    writer.write(';'.join(res for res in line))
                    writer.write(f",{round(clusters.counts(name), 2)}\n")

    def create_PDB(
        self,
        target: str,
        outputfile: str,
        scores: np.array,
        threshold: float,
        cluster_scores: dict,
        molecule: protein,
        **kwargs
    ) -> None:
        """
        Generate a PDB-style file with the coordinates of the probes
        with a score superior to `threshold`. The probes will be
        stored as `HE` atoms and the cluster centers as `AR` atoms. The
        `b-factor` will be used to store the score each probe has obtained.

        Before saving any probe coordinate, it will verify whether the probe
        is at a reasonable distance from any relevant protein atom to mitigate
        possible noise.

        Args:
            target (str): Name of the protein used for the computation.
            outputfile (str): Name of the output file, will be completed with
                the tag `_brigit.pdb` to differenciate it from other output
                files.
            scores (dict): Dict with probe coordinates and their
                coordination scores. Dimensions will be (len(probes), 4).
            threshold (float): Coordination score value below which probes will
                be discarded.
            centers (np.array): Array with cluster center coordinates and their
                coordination scores. Similar to `scores`, its dimensions will
                be (len(cluster_centers), 4).
            molecule (protein): Protein object.
        """
        outputfile = f'{outputfile}_brigit.pdb'
        with open(outputfile, "w") as fo:
            num_at = 0
            num_res = 0
            for entry in scores:
                close2protein = False
                for atom in molecule.atoms:
                    if (
                        np.linalg.norm(atom - entry[:3]) < 3. and
                        atom.element not in ['H']
                    ):
                        close2protein = True
                        break

                if not close2protein:
                    continue

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

            for name, entry in enumerate(cluster_scores.keys()):
                num_at += 1
                num_res = 1
                ch = "A"
                prb_str = ""
                list_entry = entry.split(',')

                for idx in range(3):
                    number = str(round(float(list_entry[idx]), 3))
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
                score = str(cluster_scores[entry])
                score = score if len(score) == 4 else score + '0'
                fo.write(blank + "%s     %s  1.00  %s          %s\n" %
                         (num_res, prb_str, score, atom))


def analyse_probes(
    molecule, probes, stats, gaussian_stats, molecule_info,
    coordinators, residue_score, backbone_score, max_coordinators,
    threshold, cnn_weight, mode
):
    zeros = np.zeros((len(probes), 4))
    coor_weight = (1 - cnn_weight)

    if mode == 'motif_detection':
        scorer = motif_scorer
    elif mode == 'normal_evaluation':
        scorer = coordination_scorer
    for idx, probe in enumerate(probes):
        if probe[3] > threshold:
            zeros[idx, :] = probe
            zeros[idx, 3] *= cnn_weight
            zeros[idx, 3] += scorer(
                molecule=molecule,
                probe=probe,
                stats=stats,
                gaussian_stats=gaussian_stats,
                molecule_info=molecule_info,
                coordinators=coordinators,
                residue_scoring=gaussian_score,
                backbone_scoring=discrete_score,
                max_coordinators=max_coordinators
            ) * coor_weight
    return zeros


def run(args: dict):
    print(args, end="\n\n")
    args['args'] = args
    if args['threads'] == 0:
        args['threads'] = multiprocessing.cpu_count() * 2
    brigit = Brigit(**args)
    brigit.predict(**args)


if __name__ == '__main__':
    help(Brigit)
