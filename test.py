import torch
import math
from read_h5 import read_h5

# Example program to pull conformer samples from test.h5 and compare model predictions with theory
if __name__ == "__main__":
    hart2kcal = 627.509
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
            dtruth = hart2kcal*(conf.energy-conf0.energy)
            dpred = hart2kcal*(pred_energy[i]-pred_energy[0])
            err = abs(dtruth-dpred)
            mse += err*err
            print (f"{conf.label:<10}{dtruth:>10.4f}{dpred:>10.4f}{err:>10.4f}")
        print(f"RMSE: {math.sqrt(mse/nconfs):.4f} kcal/mol\n")
        
