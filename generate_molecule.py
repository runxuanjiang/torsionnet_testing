from torsionnet.generate_molecule.alkane.generate_branched_alkane import generate_branched_alkane
from torsionnet.generate_molecule.lignin.generate_lignin import generate_lignin
from rdkit import Chem
from torsionnet.utils.chem_utils import calculate_normalizers
mol = generate_lignin(4)
print(Chem.MolToMolBlock(mol))
E0, Z0 = calculate_normalizers(mol)
print(E0, Z0)