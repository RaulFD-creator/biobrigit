"""
PDB Parser utilities within the DeepBioMetAll tool.

Contains 3 classes:
    - protein
    - residue
    - atom

Contains 1 function:
    - write_site

Simplifies the process of parsing PDB files and
solves some of the problems with BioPython.

Copyrigth by Raúl Fernández Díaz
"""
import os
import numpy as np
from .data import ATOMIC_MASSES, METAL_RESNAMES, NEW_METALS
from .tools import download_pdb, geometric_relations


def res2letter(residue: str):
    global RESNAME2LETTER_DICT
    try:
        result = RESNAME2LETTER_DICT[residue]
    except KeyError:
        result = None
    return result


class atom():
    """
    Atom class that stores all significant information from a
    protein atom.

    Attributes:
        entry_type (str): ATOM or HETATM.
        id (int): Atom serial number.
        name (str): Role that the atom plays in the protein.
        element (str): IUPAQ symbol of the atom.
        resname (str): Name of the residue to which it
            belongs.
        res_ID (str): Has been adapted to accomodate the MetalPDB format.
            Example: LYS_457(B).
        coordinates (np.array): 3D coordinates of the atom.
    """

    def __init__(self, line: str):
        """
        Instanciate new atom class.

        Args:
            line (str): PDB line describing the atom
        """
        self.entry_type = 'ATOM' if line[:4] == 'ATOM' else 'HETATM'
        self.id = int(line[6:11])
        self.name = _remove_blank_spaces(line[12:16])
        self.element = _remove_blank_spaces(line[76:80])
        self.resname = _remove_blank_spaces(line[17:20])
        self.res_ID = f'{self.resname}_{int(line[22:26])}({line[21]})'
        self.coordinates = np.array(
            [[float(line[30:38]), float(line[38:46]), float(line[48:54])]]
        )
        self.companions = [self.element]

    def __str__(self):
        return f'{self.id}_{self.name}'

    def __repr__(self):
        return f'{self.id}_{self.name}'

    def __sub__(self, other):
        return self.coordinates - other.coordinates


class residue():
    """
    Residue class that stores all significant information from a protein
    residue.
    """

    def __init__(self, atoms: list):
        self.name = atoms[0].resname
        self.id = atoms[0].res_ID
        self.atoms = atoms
        alpha, beta, o, n = self.coordinates()
        self.alpha = alpha
        self.beta = beta
        self.o = o
        self.n = n

    def coordinates(self):
        """
        Calculates the center of mass of the residue.

        Uses a weighted average function from the numpy library. Results
        are then rounded as per the significant numbers. Takes advantage of a
        global variable, i.e., `ATOMIC_MASSES` that contains the atomic masses
        of the main atoms that might comprise in the residue.
        """
        global ATOMIC_MASSES
        alpha = None
        beta = None
        o = None
        n = None

        for atom in self.atoms:
            if atom.name == 'CA':
                alpha = atom
            elif (atom.name == 'CB'):
                beta = atom
            elif atom.name == 'O':
                o = atom
            elif atom.name == 'N':
                n = atom

        return alpha, beta, o, n

    def __str__(self):
        return f'{self.name}'

    def __repr__(self):
        return f'{self.name}'

    def __sub__(self, other):
        return self.coordinates - other.coordinates


class protein():
    """
    Protein class that stores all relevant information from
    a PDB file.

    Attributes:
        name (str): PDB ID.
        atoms (list): List of atoms comprising the protein.
        hetatms (list): List of hetatms comprising the
            protein.

    Methods:
        write_MFS (path, center, radius): Writes a PDB
            file with the region surrounding a given
            HETATM.
    """

    def __init__(self, pdb_code: str, residues: bool = False):
        """
        Instanciates a new protein class.

        Parses a PDB file and collects all its atoms
        and hetatms for further processing. If the PDB
        file is not available locally, it downloads it from
        the appropriate server.

        Args:
            pdb_code (str): Path to a PDB file.
        """
        self.name = pdb_code
        self.atoms = []
        self.hetatms = []
        self.residues = []

        self._calculate_atoms(pdb_code)
        self._calculate_residues(residues)

    def _calculate_residues(self, residues):
        if not residues:
            return None

        try:
            residue_ids = {self.atoms[0].res_ID}
        except IndexError:
            self.residues = None
            return None

        tmp_id = self.atoms[0].res_ID
        tmp_atoms = []

        for atom in self.atoms:
            if (atom.res_ID not in residue_ids):
                tmp_id = atom.res_ID
                residue_ids.add(atom.res_ID)
                self.residues.append(residue(tmp_atoms))
                tmp_atoms = [atom]

            elif atom.res_ID == tmp_id:
                tmp_atoms.append(atom)

    def _calculate_atoms(self, pdb_code):
        try:
            self._read_pdb(pdb_code)

        except FileNotFoundError:
            filename = download_pdb(pdb_code, '.')
            self._read_pdb(filename)
            os.remove(filename)

    def _read_pdb(self, pdb_code):
        with open(pdb_code) as pdbfile:
            for line in pdbfile:
                if line[:4] == 'ATOM':
                    self.atoms.append(atom(line))
                elif line[:6] == 'HETATM':
                    self.hetatms.append(atom(line))

    def find_metals(self):
        to_remove = []
        metals = []
        self.metals = []

        for hetatm in self.hetatms:
            if (
                hetatm.atom_type in NEW_METALS and
                hetatm.resname in METAL_RESNAMES
            ):
                metals.append(hetatm)

        for idx, metal in enumerate(metals):
            if idx in to_remove:
                continue
            distances = np.ones(len(metals))
            distances[:] = 5.0
            for idx2, metal2 in enumerate(metals):
                if idx2 in to_remove or idx == idx2:
                    continue
                if np.linalg.norm(metal - metal2) < 5.0:
                    metal.companions.append(metal2.atom_type)
                    to_remove.append(idx2)
            self.metals.append(metal)

    def parse_residues(self, coordinators: list):
        self.coordinators = coordinators
        indexes = {residue: [] for residue in coordinators}
        alphas = {residue: [] for residue in coordinators}
        betas = {residue: [] for residue in coordinators}
        # center = {residue: [] for residue in coordinators}
        # o = {residue: [] for residue in coordinators}
        # n = {residue: [] for residue in coordinators}

        for idx, residue in enumerate(self.residues):
            if residue.name not in self.coordinators:
                continue
            indexes[residue.name].append(idx)
            alphas[residue.name].append(residue.alpha.coordinates)
            # o[residue].append(residue.o.coordinates)
            # n[residue].append(residue.n.coordinates)
            if residue.name == 'GLY':
                continue
            betas[residue.name].append(residue.beta.coordinates)
            # center[residue].append(residue.center)

        alphas = {
            residue: np.array(alphas[residue]).reshape(len(alphas[residue]), 3)
            for residue in coordinators
        }
        betas = {
            residue: np.array(betas[residue]).reshape(len(betas[residue]), 3)
            for residue in coordinators
        }
        """center = {
            residue: np.array(center[residue]).reshape(len(center[residue]), 3)
            for residue in coordinators
        }
        o = {
            residue: np.array(o[residue]).reshape(len(o[residue]), 3)
            for residue in coordinators
        }
        n = {
            residue: np.array(n[residue]).reshape(len(n[residue]), 3)
            for residue in coordinators
        }"""
        self.info = {
            'indexes': indexes,
            'alphas': alphas,
            'betas': betas,
            # 'centers': center,
            # 'backbone_O': o,
            # 'backbone_N': n
        }

    def set_stats(self, stats: dict, max_coordinators: int):
        self.stats = stats
        self.max_coordinators = max_coordinators

    def coordination_score(
        self,
        probe,
        cnn_weight: float = 0.5
    ):
        cnn_score = probe[3] * cnn_weight
        possible_coordinators = 0
        biometall_weight = 1 - cnn_weight
        fitness_score = 0

        # TODO: There will be three tiers of coordination
        # highest_tier = (3/4) * np.pi * 3**3 * 6

        for residue in self.coordinators:
            alphas = probe[:3] - self.info['alphas'][residue]
            betas = probe[:3] - self.info['betas'][residue]
            alpha_dists, beta_dists, ab_angles = geometric_relations(
                alphas, betas
            )
            alpha_trues = np.argwhere(
                (alpha_dists > self.stats[residue]['amin']) &
                (alpha_dists < self.stats[residue]['amax'])
            )
            beta_trues = np.argwhere(
                (beta_dists > self.stats[residue]['bmin']) &
                (beta_dists < self.stats[residue]['bmax'])
            )
            ab_trues = np.argwhere(
                (ab_angles > self.stats[residue]['abmin']) &
                (ab_angles < self.stats[residue]['abmax'])
            )
            for true in alpha_trues:
                if true in beta_trues and true in ab_trues:
                    b_score = self.stats[residue]['fitness'] * biometall_weight
                    fitness_score += b_score
                    possible_coordinators += 1
        fitness_score = (fitness_score + cnn_score) * (
            possible_coordinators / self.max_coordinators
        )
        return fitness_score

    def __str__(self):
        return f'Protein Structure of PDB ID: {self.name}'

    def __repr__(self):
        return str(self)


def write_site(
    structure: protein,
    output_path: str,
    center: np.array,
    radius: float,
    protein_atoms: bool = True
):
    """
    Generate a PDB with a all residues that contain at least one
    atom within a certain radius of a set of coordinates.

    Args:
        output_path (str): Path where the new PDB
            file will be written.
        center (np.array): Coordinates for the center of the site with
            shape (1, 3)
        radius (float): Radius of the site to be
            computed.
    """
    mfs_residues = set()

    for atom in structure.atoms:
        if np.linalg.norm(atom.coordinates - center) <= radius:
            mfs_residues.add(atom.res_ID)

    control_num_atoms = []
    atoms_in_trp = 27
    threshold = (1/3) * (radius ** 2) * atoms_in_trp * np.pi

    with open(output_path, 'w') as fo:
        for atom in structure.atoms:
            if atom.res_ID in mfs_residues:
                fo.write(atom.line)
                control_num_atoms.append(atom)
        fo.write('END')

    if protein_atoms:
        if len(control_num_atoms) > threshold:
            try:
                os.remove(output_path)
            except FileNotFoundError:
                pass
            raise RuntimeError(f"Too Many Atoms: {len(control_num_atoms)}")

        elif len(control_num_atoms) == 0:
            try:
                os.remove(output_path)
            except FileNotFoundError:
                pass
            raise RuntimeError(f"Not Enough Atoms: {len(control_num_atoms)}")


def _remove_blank_spaces(sentence):
    atom_type = ""
    for char in sentence:
        if char not in [" ", "\t", "\n"]:
            atom_type += char
    return atom_type


if __name__ == '__main__':
    help(protein)
    help(residue)
    help(atom)
    help(write_site)
