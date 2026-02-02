#!/usr/bin/env python3
"""
Generate optimized 3D molecular structures using RDKit.
Outputs SDF format strings that can be embedded in molecular.js.
"""

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import json

# Molecules to generate with correct SMILES
MOLECULES = {
    "caffeine": "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "benzene": "c1ccccc1",
    "ethanol": "CCO",
    "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "glucose": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
}


def generate_sdf(name: str, smiles: str) -> str:
    """Generate an optimized 3D SDF structure from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES for {name}: {smiles}")

    # Add hydrogens
    mol = Chem.AddHs(mol)

    # Generate 3D coordinates using ETKDG (better initial geometry)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42  # Reproducible
    result = AllChem.EmbedMolecule(mol, params)
    if result != 0:
        # Fallback to default embedding
        AllChem.EmbedMolecule(mol)

    # Optimize with MMFF94 force field (better than UFF for organics)
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        # Fallback to UFF if MMFF fails
        AllChem.UFFOptimizeMolecule(mol, maxIters=500)

    # Set molecule name
    mol.SetProp("_Name", name.capitalize())

    # Generate SDF block
    sdf = Chem.MolToMolBlock(mol)
    return sdf


def format_for_js(name: str, sdf: str) -> str:
    """Format SDF for embedding in JavaScript template literal."""
    # Escape backticks and ${} if any
    escaped = sdf.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
    return f'  {name}: `{escaped}`'


def main():
    print("Generating optimized molecular structures with RDKit...\n")

    results = {}
    for name, smiles in MOLECULES.items():
        try:
            sdf = generate_sdf(name, smiles)
            results[name] = sdf
            print(f"Generated {name}")
        except Exception as e:
            print(f"Error generating {name}: {e}")

    print("\n// Paste this into BUILTIN_MOLECULES in molecular.js:\n")
    print("export const BUILTIN_MOLECULES = {")

    entries = []
    for name, sdf in results.items():
        entries.append(format_for_js(name, sdf))

    print(",\n\n".join(entries))
    print("\n};")


if __name__ == "__main__":
    main()
