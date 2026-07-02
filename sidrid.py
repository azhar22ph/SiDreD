from rdkit import Chem
from rdkit.Chem import Descriptors, Draw

# Ibuprofen, written as a SMILES string — a text notation for molecular structure
ibuprofen_smiles = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"

# Parse the SMILES string into an RDKit molecule object
molecule = Chem.MolFromSmiles(ibuprofen_smiles)

if molecule is None:
    print("Failed to parse molecule — check the SMILES string")
else:
    print("Molecule loaded successfully!")
    print(f"Number of atoms: {molecule.GetNumAtoms()}")
    print(f"Molecular weight: {Descriptors.MolWt(molecule):.2f} g/mol") 
    print(f"LogP (lipophilicity): {Descriptors.MolLogP(molecule):.2f}")
    print(f"H-bond donors: {Descriptors.NumHDonors(molecule)}")
    print(f"H-bond acceptors: {Descriptors.NumHAcceptors(molecule)}")

    # Save a 2D image of the molecule so you can see it
    Draw.MolToFile(molecule, "ibuprofen.png", size=(400, 400))
    print("Saved image as ibuprofen.png — check your project folder!")