import h5py
class Conformer:
    def __init__(self):
        self.label = ''
        self.coords = []
        self.energy = 0.0

class Molecule:
    def __init__(self):
        self.label = ''
        self.inchi = ''
        self.species = []
        self.confs = []

def read_h5(file_path):
    with h5py.File(file_path, 'r') as file:
        mols = []
        for mol_id in file.keys():
            mol_group = file[mol_id]
            mol = Molecule()
            mol.label = mol_id
            mol.inchi = mol_group.attrs['inchi']
            mol.species = mol_group.attrs['species']

            for conf_id in mol_group.keys():
                conf_group = mol_group[conf_id]
                conf = Conformer()
                conf.label = conf_id
                conf.energy = conf_group.attrs['energy']

                xyz = conf_group['atXYZ'][:]
                conf.coords = [(xyz[i], xyz[i + 1], xyz[i + 2]) for i in range(0, len(xyz), 3)]
                mol.confs.append(conf)
            mols.append(mol)
        return mols
