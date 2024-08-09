import h5py
import torch
import math

# Example program to pull conformer samples from test.h5 and compare model predictions with theory
# When using this on your own molecules, ensure
# 1) Geometries are calculated with MMFF
# 2) The model is only used to compare delta energies between two (or more) conformers, as illustrated. 
#    Predictions are not accurate for absolute energies
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

if __name__ == "__main__":
    hart2kj = 2625.5
    model = torch.jit.load('DLFF03.pt')
    print(f"MLFF correction model version: {model.version} Copyright 2024 Wavefunction, Inc.")
    mols = read_h5('test.h5')[0:5] # grab a few molecules
 
    for mol in mols:
        mse = 0
        nconfs = len(mol.confs)
        if nconfs == 1:
            print (f"{mol.label}: Need at least two conformers")
            continue

        Z = torch.tensor(mol.species,dtype=torch.int64)
        R = []
        for conf in mol.confs:
            R.append(torch.tensor(conf.coords,dtype = torch.float32))
        R = torch.stack(R,axis=0) # batch along first dim
        pred_energy = model(Z,R) # our model processes one unique molecule at a time
        # show the delta-energies and compare with truth
        conf0 = mol.confs[0]
        
        print (f"{mol.label}: {nconfs} conformers: Deltas vs {conf0.label}")
        print (f"{'Conf':<10}{'Truth':>10}{'Pred':>10}{'Error':>10}")
        for i in range(1,nconfs):
            conf = mol.confs[i]
            dtruth = hart2kj*(conf.energy-conf0.energy)
            dpred = hart2kj*(pred_energy[i]-pred_energy[0])
            err = abs(dtruth-dpred)
            mse += err*err
            print (f"{conf.label:<10}{dtruth:>10.4f}{dpred:>10.4f}{err:>10.4f}")
        print(f"RMSE: {math.sqrt(mse/nconfs):.4f} KJ/mol\n")
        
