"""
Data module within the Brigit package.

Contains several different data structures that refer to relevant
information required by other modules within the package.

Copyright by Raúl Fernández Díaz
"""
import os
import json


def read_stats() -> dict:
    # Define paths to stats
    current_dir = os.path.dirname(__file__)
    stats_path = os.path.join(current_dir, 'stats')
    residues_path = os.path.join(stats_path, 'residue_statistics.json')
    gaussian_path = os.path.join(stats_path, 'gaussian_statistics.json')

    # Read stats
    with open(residues_path) as reader_1:
        stats = json.load(reader_1)
    with open(gaussian_path) as reader_2:
        gaussian_stats = json.load(reader_2)

    return stats, gaussian_stats


ALL_CHANNELS = ['hydrophobic', 'aromatic', 'hbond_acceptor', 'hbond_donor',
                'positive_ionizable', 'negative_ionizable', 'metal',
                'excluded_volume']
ALL_METALS = {
    'LI', 'BE',
    'NA', 'MG', 'V', 'CR', 'MN', 'FE', 'CO', 'NI', 'CU', 'ZN',
    'K',  'CA',      'MO',       'RU', 'RH', 'PD', 'AG', 'CD',
    'RB', 'SR',                        'IR', 'PT', 'AU', 'HG',
    'CS', 'BA'
}
ATOMIC_MASSES = {
    'H': 1.01,
    'D': 2.01,
    'C': 12.01,
    'N': 14.01,
    'N1+': 14.01,
    'N1-': 14.01,
    'O': 16.0,
    'O1+': 16.0,
    'O1-': 16.0,
    'O-1': 16.0,
    'O2-': 16.0,
    'S': 32.06,
    'S2-': 32.06,
    'P': 30.97,
    'P1+': 30.97,
    'SE': 78.96,
}
CHANNELS = ['hydrophobic', 'hbond_acceptor', 'hbond_donor',
            'positive_ionizable', 'negative_ionizable', 'excluded_volume']


CHANNELS_DICT = {channel: idx for idx, channel in enumerate(ALL_CHANNELS)}

METAL_RESNAMES = {
    'DDH', 'BEF', 'H79', '1PT', 'AV2', 'P9G', 'SNF', 'CFC', 'HCB', 'BF4',
    'CAS', 'TIL', 'A72', 'IR3', 'HDE', '2J0', 'RHX', 'MYQ', 'PBM', 'SQ1',
    '72B', 'HEG', 'CON', '0UE', 'CPO', 'ALB', 'RKP', 'VER', 'F4S', 'FMI',
    'DRB', '118', 'HEM', '3UQ', 'COB', 'PLL', 'ZPT', 'S31', 'CPT', 'CNC',
    'HEB', 'IUM', 'JSE', 'B1M', 'PCU', 'HIF', 'S32', 'KSB', 'CVC', 'HRU',
    'PCL', '1MK', 'RHD', 'AC9', 'WO3', 'HEO', 'V', 'PT4', 'H58', 'F3S',
    'AM', 'ALF', 'DTZ', '31Q', 'HCO', 'M43', '0JC', 'WO5', 'PT', 'FCE',
    'CU1', 'R7U', 'CU', 'AMW', '0TE', '8WV', 'DVW', 'B9F', 'A71', 'I83',
    'PC4', 'CBY', 'RKL', 'CUL', 'EU3', 'DHE', 'TTO', 'MD9', 'PC3', 'RKM',
    '51O', 'ELJ', 'CL7', 'WO2', 'AF3', '6CQ', 'BVA', 'N7H', 'RH3', 'CUS',
    'JSD', 'PD', 'RE', 'HEV', 'TBY', 'C4R', 'QHL', 'XAX', '2GO', 'MN',
    'DAZ', 'MM1', 'CS', 'MM4', 'HES', 'OS', '73M', 'MM6', 'CAC', 'HBF',
    'SM', 'NCP', 'CA', 'MOS', 'BVQ', 'HG', '7G4', 'HGB', 'IRI', 'ART',
    'ZEM', 'APW', 'PR', 'RUA', 'R6A', 'HNN', 'FES', 'FDD', '35N', 'DW2',
    '5IR', 'GBF', '522', 'OBV', 'H57', 'U1', 'CL1', 'PMR', 'CM2', 'PB',
    'C7P', 'NI', 'HAS', 'EMT', 'A6R', 'GB0', 'ZN9', '3T3', 'MGF', 'VN4',
    'REJ', 'CE', 'IMF', 'YT3', 'COY', 'CSR', 'BE7', 'BCB', '4IR', 'FEC',
    'CAD', 'UVC', 'PHG', '2FH', 'CMH', 'GA', '0TN', 'UFE', 'AUF', 'BTF',
    'TB', 'LOS', 'RIR', '0H2', 'AU3', 'NTE', '6CO', 'CLA', 'BA', 'BAZ',
    'CL0', 'SR', 'AIV', 'MBO', '89R', 'OMO', 'OS1', 'SF3', 'FLL', '4A6',
    'LA', 'IR', 'TPT', 'DW5', '3CG', 'PA0', 'DY', 'HE6', 'EFE', '6B6',
    '9RU', 'ZN7', 'HDD', '5AU', 'CCH', 'RU1', 'CO', 'HEA', 'COH', 'AUC',
    '4KV', 'F43', 'SI4', 'VN3', 'HEC', 'RPS', 'I42', 'ME3', 'HDM', 'R1Z',
    'HME', 'FC6', 'IME', 'J1R', 'GD3', '2PT', '3WB', '6BP', 'RSW', 'POR',
    '9QB', 'RUX', 'ZNH', 'AG1', 'OSV', 'OFE', 'CR', '4MO', 'DWC', 'CAF',
    'PFC', '9D7', '8AR', 'FE', 'MNR', 'OS4', 'B13', 'NA', '3NI', 'RHE',
    'RTA', 'WO4', 'GIX', 'B12', 'CD', 'ZCM', 'CM1', 'RU8', 'ARS', 'BOZ',
    'CUP', '4HE', 'V5A', '9ZQ', 'C5L', '08T', 'SIR', 'YPT', 'YB2', 'MOO',
    'OHX', '7GE', 'AVC', 'MP1', 'LI', 'SF4', 'B22', 'RFB', 'FCI', 'ER3',
    'BF2', 'RUR', 'B30', 'MTQ', 'RU2', '6ZJ', 'GD', 'DEF', '3Q7', 'DW1',
    'MM5', 'YBT', 'YOK', 'BCL', 'ASR', 'I2A', 'SI7', 'Y1', 'CHL', 'RH',
    '6O0', 'SI8', 'N2R', 'PTN', 'REI', 'B1R', 'ITM', 'MOM', 'SI0', 'SVP',
    'RMD', 'MG', 'T0M', 'FE9', '76R', 'RHM', '2MO', 'FS1', 'SI9', '4PU',
    'VVO', '4TI', 'FE2', 'TL', 'UNL', 'CFQ', '2T8', 'RBN', 'MH2', 'VEA',
    'NOB', 'T1A', 'RU', 'MMC', '8CY', 'RKF', 'MTV', 'PL1', '6HE', 'ISW',
    'HE5', 'PMB', 'SB', 'RU7', 'REP', 'RTB', 'ZN6', 'GCR', 'REQ', 'L2D',
    'AL', '9Q8', '1Y8', 'HFM', 'RTC', 'R1C', 'CF', 'MH0', 'CSB', 'L4D',
    'L2M', 'AU', 'AOH', 'AG', 'ZND', 'DRU', 'KYT', 'TTA', 'CX8', 'EU',
    'ATS', '6WO', 'NRU', 'HG2', 'TCN', 'REO', 'QPT', 'SMO', 'N2W', '3ZZ',
    'DEU', 'YOL', 'KYS', 'NXC', 'LU', 'SBO', 'ZN8', 'EMC', 'HKL', 'R4A',
    'MM2', '7BU', '0OD', 'JM1', '6MO', 'YXX', 'JSC', 'HGD', 'SKZ', 'BS3',
    'FEL', 'PNI', 'HB1', 'RHL', 'RML', 'ZN0', 'TH', 'ZN', 'MSS', 'YB',
    'TA0', 'N2N', 'RBU', 'DOZ', 'NCO', 'K', 'RXO', 'YOM', 'FDE', 'LPT',
    '188', 'RAX', 'BJ5', 'CU6', 'RUI', 'IN', 'SXC', 'OEY', 'LSI', 'CX3',
    'PCD', 'RFA', 'ICA', 'B6F', 'MN3', 'MNH', 'RUC', 'M10', 'MOW', '11R',
    'HNI', '7MT', 'W', 'HGI', '1FH', 'WPC', 'MAP', 'R9A', 'ZN5', '68G',
    'RUH', 'RCZ', 'E52', 'HO3', 'MO', '7HE', 'VO4', 'FEM', 'T9T', 'CB5',
    'HO', '6WF', '3CO', 'CLN', '5LN', 'RB', 'TAS', 'CQ4', 'B1Z'
}

NEW_METALS = {f'{metal}{idx}' for idx in range(1, 4) for metal in ALL_METALS}
NEW_METALS = NEW_METALS.union(ALL_METALS)

RESNAME2LETTER_DICT = {
    'VAL': 'V', 'ILE': 'I', 'LEU': 'L', 'GLU': 'E', 'GLN': 'Q',
    'ASP': 'D', 'ASN': 'N', 'HIS': 'H', 'TRP': 'W', 'PHE': 'F',
    'TYR': 'Y', 'ARG': 'R', 'LYS': 'K', 'SER': 'S', 'THR': 'T',
    'MET': 'M', 'ALA': 'A', 'GLY': 'G', 'PRO': 'P', 'CYS': 'C'
}


if __name__ == '__main__':
    print('File with different data stored to be used as global variables.')
