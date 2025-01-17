class Molecule:
    def __init__(self):
        self.label = ""
        self.species = []
        self.coords = []

def read_xyz(path): 
    periodic_table = {'H':1, 'B':5, 'C':6, 'N':7, 'O':8, 'F':9, 'Si':14, 'P':15, 'S':16, 'Cl':17, 'Br':35}
    mols = []
    with open(path, "r") as fh:
        while True:
            # read to first number
            n = ''
            while True:
                n = fh.readline() #number species
                if n == '': # EOF
                    break
                try:
                    n = int(n)
                    break
                except ValueError:
                    pass
            if n == '':
                break
            label = fh.readline().strip()
            mol = Molecule()
            mol.label = label
            for i in range(0,n):
                line = fh.readline()
                res = line.split()
                ch = res[0]
                if ch.isalpha():
                    mol.species.append(periodic_table[res[0]])
                else:
                    mol.species.append(int(res[0]))
                mol.coords.append((float(res[1]),float(res[2]),float(res[3])))
            mols.append(mol)
    return mols
