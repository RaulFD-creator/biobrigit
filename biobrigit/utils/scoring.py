"""
Brigit module that contains the main functions
for evaluating coordination probes.

Contains 5 functions:

    - parse_residues
    - coordination_scorer
    - discrete_score
    - gaussian_score
    - double_gaussian

Copyright by Raúl Fernández Díaz
"""

import numpy as np

from .pdb_parser import protein
from .tools import geometry


def parse_residues(
    molecule: protein,
    coordinators: dict,
) -> dict:
    """
    Create a dictionary that collects the coordinates of the
    atoms of a given protein that belong to the most likely
    coordinators. Depending on the coordination configuration
    (i.e., residue, backbone O, or backbone N), the atoms of interest
    will vary (e.g., for residue atoms of interest are alpha C and beta
    C).

    Args:
        molecule (protein): Protein object.
        coordinators (dict): Dictionary summarising the residues that
            are the most likely coordinators for a given metal for
            the different configurations.

    Returns:
        info (dict): Collection of the atoms of interest for evaluating
            coordination, grouped by the coordination configuration.
    """
    alphas = {residue: [] for residue in coordinators}
    betas = {residue: [] for residue in coordinators}

    for idx, residue in enumerate(molecule.residues):
        try:
            if residue.name in coordinators:
                alphas[residue.name].append(residue.alpha.coordinates)
                betas[residue.name].append(residue.beta.coordinates)
        except AttributeError:
            continue

    alphas = {
        residue: np.array(alphas[residue]).reshape(len(alphas[residue]), 3)
        for residue in coordinators
    }
    betas = {
        residue: np.array(betas[residue]).reshape(len(betas[residue]), 3)
        for residue in coordinators
    }

    info = {
        'residue_alphas': alphas,
        'residue_2nd_atom': betas,
    }
    return info


def coordination_scorer(
    molecule: protein,
    probe: np.array,
    stats: dict,
    gaussian_stats: dict,
    molecule_info: dict,
    coordinators: dict,
    residue_scoring,
    max_coordinators,
    **kwargs
) -> float:
    """
    Compute how strongly a hypothetical metal would be coordinated
    if it where located at the coordinates of `probe`.

    It can use different scoring functions for residue or backbone
    coordinations. It iterates through all coordinating residues and
    calculates the distances between the probe and the alpha C and a
    second atom (beta C for residue coordination, O or N for the corresponding
    backbone coordination), and the angles between both atoms with regards
    to the probe. Then it uses this information and the appropriate scoring
    function to assign a coordination score to the probe.

    Finally, the cummulative score is multiplied by a factor that depends on
    how many coordinating residues were located and how many were expected.
    Final value will be at maximum of 1.0.

    Args:
        molecule (protein): Protein object.
        probe (np.array): 3D array with the spatial coordinates of a
            hypothetical metal.
        stats (dict): Dictionary containing statistical values describing
            the distances a metallic ion has to keep with regards to different
            backbone atoms in order to properly be coordinated.
        gaussian_stats (dict): Similar to `stats`, but these statistics have
            been calculated to fit the combination of 2 gaussian curves.
        molecule_info (dict): Dictionary containing a collection of the atoms
            of interest for evaluating coordination, grouped by the
            coordination configuration. Output of `parse_residues` function.
        coordinators (dict): Dictionary summarising the residues that
            are the most likely coordinators for a given metal for
            the different configurations.
        residue_scoring (function): Function indicating how to score the
            coordination through residues.
        backbone_scoring (function): Function indicating how to score the
            coordination through backbone atoms.

    Returns:
        score (float): Score associated to the point in space occupied by the
            `probe`.
    """
    fitness = 0
    possible_coordinators = 0

    kwargs['gaussian_stats'] = gaussian_stats
    for res in coordinators:
        alphas = probe[:3] - molecule_info['residue_alphas'][res]
        atm2 = probe[:3] - molecule_info['residue_2nd_atom'][res]
        dist_1, dist_2, angles = geometry(alphas, atm2)
        fitness_res, coors_res = (
            residue_scoring(
                dist_1, dist_2, angles, res, stats, **kwargs
            )
        )
        fitness += fitness_res
        possible_coordinators += coors_res

    fitness *= (possible_coordinators / max_coordinators)
    return fitness if fitness < 1.0 else 1.0


def discrete_score(
    dist_1: np.array,
    dist_2: np.array,
    angles: np.array,
    residue: str,
    stats: dict,
    coor_type: str,
    **kwargs
) -> tuple:
    """
    Scoring function to be used by the `coordination_scorer` function
    to compute the relative strength or suitability of the spatial position
    represented by the probe for coordinating a given metal ion.

    This particular scoring function uses a set of statistical values
    calculated as the median of the distribution of distances between the
    metallic moietie, the alpha C and the appropriate second atom, and the
    angles between them. The ranges are set at the median +- 3 times the
    standard deviation.

    The algorithm first finds all positions that comply with all 3 requirements
    simultaneously and scores them according to the relative abundance of the
    specific metal being coordinating by the specific residue.

    Args:
        dist_1 (np.array): Distances from `probe` to the alpha C.
        dist_2 (np.array): Distances from `probe` to the second atom which
            could be beta C (residue), backbone O, or backbone N.
        angles (np.array): Angles formed by the vectors defined by alpha C and
            `probe` and 2nd atom and `probe`.
        residue (str): Name of the residue currently being evaluated.
        stats (dict):  Dictionary containing statistical values describing
            the distances a metallic ion has to keep with regards to different
            backbone atoms in order to properly be coordinated.
        coor_type (str): Whether the coordination is with the residue, with a
            backbone O, or a backbone N. The main difference is the set of
            statistics used for the evaluation.

    Returns:
        tuple: Contains two different values, i.e., `fitness` which is the
            score obtained for a given probe and a given type of residue and
            `possible_coordinators` which is the number of such residues that
            fulfill the geometric requirements.
    """
    if coor_type == 'residue':
        a, b, c, d, e, f = 'amin', 'amax', 'bmin', 'bmax', 'abmin', 'abmax'
        g = 'fitness'
    elif coor_type == 'backbone_o':
        a, b, c, d, e, f = 'aomin', 'aomax', 'omin', 'omax', 'maomin', 'maomax'
        g = 'o_fitness'
    elif coor_type == 'backbone_n':
        a, b, c, d, e, f = 'anmin', 'anmax', 'nmin', 'nmax', 'manmin', 'manmax'
        g = 'n_fitness'

    fitness = 0
    possible_coordinators = 0
    alpha_trues = np.argwhere(
        (dist_1 > stats[residue][a]) &
        (dist_1 < stats[residue][b])
    )
    beta_trues = np.argwhere(
        (dist_2 > stats[residue][c]) &
        (dist_2 < stats[residue][d])
    )
    ab_trues = np.argwhere(
        (angles > stats[residue][e]) &
        (angles < stats[residue][f])
    )
    for true in alpha_trues:
        if true in beta_trues and true in ab_trues:
            fitness += stats[residue][g]
            possible_coordinators += 1

    return fitness, possible_coordinators


def gaussian_score(
    dist_1: np.array,
    dist_2: np.array,
    angles: np.array,
    residue: str,
    stats: dict,
    gaussian_stats: dict,
    **kwargs
) -> tuple:
    """
    Scoring function to be used by the `coordination_scorer` function
    to compute the relative strength or suitability of the spatial position
    represented by the probe for coordinating a given metal ion.

    This particular scoring function uses a set of statistical values
    calculated as the parameters of the combination of 2 gaussian curves
    that best fit the distribution of the data. Currently, this function
    can only be used for evaluating residue coordination and not backbone
    O or N.

    The algorithm first finds all positions that comply with all 3 requirements
    simultaneously and scores them according to the relative abundance of the
    specific metal being coordinating by the specific residue and to their
    position within the gaussian curves.


    Args:
        dist_1 (np.array): Distances from `probe` to the alpha C.
        dist_2 (np.array): Distances from `probe` to the second atom which
            could be beta C (residue), backbone O, or backbone N.
        angles (np.array): Angles formed by the vectors defined by alpha C and
            `probe` and 2nd atom and `probe`.
        residue (str): Name of the residue currently being evaluated.
        stats (dict):  Dictionary containing statistical values describing
            the distances a metallic ion has to keep with regards to different
            backbone atoms in order to properly be coordinated.
        gaussian_stats (dict):  Similar to `stats`, but these statistics have
            been calculated to fit the combination of 2 gaussian curves.

    Returns:
        tuple: _description_
    """
    fitness = 0
    possible_coordinators = 0

    score_1 = bimodal(dist_1, *gaussian_stats[residue]['alpha'])
    score_2 = bimodal(dist_2, *gaussian_stats[residue]['beta'])
    score_3 = bimodal(angles, *gaussian_stats[residue]['MAB'])

    # BioMetAll v1.0 filter
    alpha_trues = np.argwhere(score_1 > 0.01)
    beta_trues = np.argwhere(score_2 > 0.01)
    ab_trues = np.argwhere(score_3 > 0.01)
    true_indexes = []

    for true in alpha_trues:
        if true in beta_trues and true in ab_trues:
            true_indexes.append(true)

    if len(true_indexes) < 1:
        return 0, 0

    # Gaussian score
    true_indexes = np.array(true_indexes)

    fitness = np.sum(
        (score_1[true_indexes] + score_2[true_indexes] + score_3[true_indexes])
        / 3
    )
    fitness = float(fitness)
    fitness *= stats[residue]['fitness']
    possible_coordinators += len(true_indexes)

    return fitness, possible_coordinators


def bimodal(
    x: np.array,
    chi1: float,
    nu1: float,
    sigma1: float,
    chi2: float,
    nu2: float,
    sigma2: float
) -> np.array or float:
    """
    Helper function for the `gaussian_score` function that computes
    the score associated to a certain set of parameters for the input
    `x`.

    Args:
        x (np.array): Set of input values to evaluate.
        prop (float): Factor describing the contribution of each of
            the gaussians to the final result.
        chi1 (float): Height of the first gaussian.
        nu1 (float): Average value for the first gaussian.
        sigma1 (float): Standard deviation of the first gaussian.
        chi2 (float): Height of the second gaussian.
        nu2 (float): Average value for the second gaussian.
        sigma2 (float): Standard deviation of the second gaussian.

    Returns:
        result (np.array or float): Value or array of values with len(x)
            with the associated probability.
    """
    first_gaussian = _normpdf(x, chi1, nu1, sigma1)
    second_gaussian = _normpdf(x, chi2, nu2, sigma2)
    return first_gaussian + second_gaussian


def _normpdf(x: np.array or float, chi: float, nu: float, std: float):
    """
    Helper function for `double_gaussian`, computes the PDF of a
    gaussian function.

    Args:
        x (np.array or float): Set of input values to evaluate.
        chi (float): Height of the gaussian.
        nu (float): Average value for the gaussian.
        std (float): Standard deviation of the gaussian.

    Returns:
        result (float): PDF value.
    """

    var = std ** 2
    num = np.exp(- (x - nu) ** 2 / (2 * var))
    return chi * num


if __name__ == '__main__':
    help(parse_residues)
    help(coordination_scorer)
    help(discrete_score)
    help(gaussian_score)
    help(bimodal)
