import torch
from read_xyz import read_xyz

# When using this on your own molecules, it is recommended that
# 1) Geometries are calculated with MMFF
# 2) The model is only used to compare delta energies between two (or more) conformers of the same molecule.
#    Predictions are not accurate for absolute energies
if __name__ == "__main__":
    hart2kcal = 627.509
    model = torch.jit.load('DLFF03.pt')
    print(f"MLFF correction model version: {model.version} Copyright 2024 Wavefunction, Inc.")
    mols = read_xyz('input.xyz')

    for mol in mols:
        print(f"Read {mol.label}")
        Z = torch.tensor(mol.species,dtype=torch.int64)
        R = torch.tensor(mol.coords,dtype=torch.float32).unsqueeze(0) # add batch dimension
        res = model(Z,R) # our model processes one unique molecule at a time
        energy = hart2kcal*res[0]
        print(f"Rel energy (kcal/mol): {energy:.3f}")
