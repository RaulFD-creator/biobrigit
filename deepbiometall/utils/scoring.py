import numpy as np
import scipy.stats as stats
from .tools import geometry
from .pdb_parser import protein


def parse_residues(
    molecule: protein,
    coordinators: list,
):
    alphas = {residue: [] for residue in coordinators['residue']}
    betas = {residue: [] for residue in coordinators['residue']}
    o_alphas = {residue: [] for residue in coordinators['backbone_o']}
    o = {residue: [] for residue in coordinators['backbone_o']}
    n_alphas = {residue: [] for residue in coordinators['backbone_n']}
    n = {residue: [] for residue in coordinators['backbone_n']}

    for idx, residue in enumerate(molecule.residues):
        try:
            if residue.name in coordinators['residue']:
                alphas[residue.name].append(residue.alpha.coordinates)
                betas[residue.name].append(residue.beta.coordinates)
            elif residue.name in coordinators['backbone_o']:
                o_alphas[residue.name].append(residue.alpha.coordinates)
                o[residue.name].append(residue.o.coordinates)
            elif residue.name in coordinators['backbone_n']:
                n_alphas[residue.name].append(residue.alpha.coordinates)
                n[residue.name].append(residue.n.coordinates)
        except AttributeError:
            continue

    alphas = {
        residue: np.array(alphas[residue]).reshape(len(alphas[residue]), 3)
        for residue in coordinators['residue']
    }
    betas = {
        residue: np.array(betas[residue]).reshape(len(betas[residue]), 3)
        for residue in coordinators['residue']
    }
    o_alphas = {
        residue: np.array(o_alphas[residue]).reshape(
            len(o_alphas[residue]), 3
        ) for residue in coordinators['backbone_o']
    }
    o = {
        residue: np.array(o[residue]).reshape(len(o[residue]), 3)
        for residue in coordinators['backbone_o']
    }
    n_alphas = {
        residue: np.array(n_alphas[residue]).reshape(
            len(n_alphas[residue]), 3
        ) for residue in coordinators['backbone_n']
    }
    n = {
        residue: np.array(n[residue]).reshape(
            len(n[residue]), 3
        ) for residue in coordinators['backbone_n']
    }
    info = {
        'residue_alphas': alphas,
        'residue_2nd_atom': betas,
        'backbone_o_alphas': o_alphas,
        'backbone_o_2nd_atom': o,
        'backbone_n_alphas': n_alphas,
        'backbone_n_2nd_atom': n
    }
    return info


def coordination_score(
    molecule: protein,
    probe: np.array,
    stats: dict,
    gaussian_stats: dict,
    molecule_info: dict,
    coordinators: dict,
    residue_scoring,
    backbone_scoring,
    **kwargs
):
    fitness = 0
    coordination_types = ['residue', 'backbone_o', 'backbone_n']
    for coor_type in coordination_types:
        kwargs['coor_type'] = coor_type
        kwargs['gaussian_stats'] = gaussian_stats
        for res in coordinators[coor_type]:
            alphas = probe[:3] - molecule_info[f'{coor_type}_alphas'][res]
            atm2 = probe[:3] - molecule_info[f'{coor_type}_2nd_atom'][res]
            dist_1, dist_2, angles = geometry(alphas, atm2)
            fitness += (
                residue_scoring(
                    dist_1, dist_2, angles, res, stats, **kwargs
                )
                if coor_type == 'residue' else
                backbone_scoring(
                    dist_1, dist_2, angles, res, stats, **kwargs
                )
            )
    return fitness


def discrete_score(
    dist_1: np.array,
    dist_2: np.array,
    angles: np.array,
    residue: str,
    stats: dict,
    max_coordinators: int,
    coor_type: str,
    **kwargs
) -> float:
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

    if possible_coordinators > max_coordinators:
        possible_coordinators = max_coordinators

    fitness *= (possible_coordinators / max_coordinators)
    if fitness > 1.0:
        fitness = 1.0

    return fitness


def gaussian_score(
    dist_1: np.array,
    dist_2: np.array,
    angles: np.array,
    residue: str,
    stats: dict,
    gaussian_stats: dict,
    **kwargs
) -> float:
    fitness = 0

    alpha_scores = double_gaussian(dist_1, *gaussian_stats[residue]['alpha'])
    beta_scores = double_gaussian(dist_2, *gaussian_stats[residue]['beta'])
    angle_scores = double_gaussian(angles, *gaussian_stats[residue]['MAB'])

    alpha_trues = np.argwhere(alpha_scores > 0.2)
    beta_trues = np.argwhere(beta_scores > 0.2)
    angle_trues = np.argwhere(angle_scores > 0.2)

    for true in alpha_trues:
        if true in beta_trues and true in angle_trues:
            score_1 = alpha_scores[true]
            score_2 = beta_scores[true]
            score_angles = angle_scores[true]
            fitness += (score_1 + score_2 + score_angles) / 3
            fitness *= stats[residue]['fitness']

    if fitness > 1.0:
        fitness = 1.0
    return fitness


def double_gaussian(x, prop, nu1, sigma1, nu2, sigma2, *args):
    first_gaussian = stats.norm(nu1, sigma1).pdf(x)
    second_gaussian = stats.norm(nu2, sigma2).pdf(x)
    return prop * first_gaussian + (1-prop) * second_gaussian
