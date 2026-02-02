import test from "node:test";
import assert from "node:assert/strict";
import { parsePDB, splitMolDataByHetatm } from "../src/molecular.js";

test("parsePDB marks HETATM and splitMolDataByHetatm partitions atoms/bonds", () => {
  const pdb = [
    "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N",
    "HETATM    2  C1  LIG A   2       1.000   0.000   0.000  1.00  0.00           C",
    "HETATM    3  O1  LIG A   2       1.500   0.000   0.000  1.00  0.00           O",
    "CONECT    2    3",
    ""
  ].join("\n");

  const mol = parsePDB(pdb);
  assert.equal(mol.atoms.length, 3, "Should parse 3 atoms");
  assert.equal(mol.atoms.filter((a) => a.isHet).length, 2, "Should mark 2 HETATM atoms");

  const split = splitMolDataByHetatm(mol);
  assert.equal(split.standard.atoms.length, 1, "Should have 1 standard atom");
  assert.equal(split.hetero.atoms.length, 2, "Should have 2 hetero atoms");
  assert.equal(split.hetero.bonds.length, 1, "Should keep hetero bonds");
  assert.equal(split.standard.bonds.length, 0, "Should have no standard bonds");
});
