import rdkit.Chem.AllChem as Chem
from torsionnet.utils import get_conformer_energies


if __name__ == '__main__':

    lignin = Chem.MolFromMolFile('8_0.mol')
    lignin = Chem.AddHs(lignin)

    Chem.EmbedMultipleConfs(lignin, numConfs=20, numThreads=-1)
    import pdb
    pdb.set_trace()
    Chem.MMFFOptimizeMoleculeConfs(lignin, numThreads=-1)
    energys = get_conformer_energies(lignin)
    print(energys)