/**
 * Molecular file parser and converter for PDB and SDF formats.
 * Converts molecular structures to spheres (atoms) and cylinders (bonds).
 */

// CPK/Jmol-style element colors
const ELEMENT_COLORS = {
  H:  [1.000, 1.000, 1.000],  // White
  C:  [0.565, 0.565, 0.565],  // Dark gray
  N:  [0.188, 0.314, 0.973],  // Blue
  O:  [1.000, 0.051, 0.051],  // Red
  F:  [0.565, 0.878, 0.314],  // Green
  Cl: [0.122, 0.941, 0.122],  // Green
  Br: [0.651, 0.161, 0.161],  // Brown
  I:  [0.580, 0.000, 0.580],  // Purple
  S:  [1.000, 0.784, 0.196],  // Yellow
  P:  [1.000, 0.502, 0.000],  // Orange
  Fe: [0.878, 0.400, 0.200],  // Orange-brown
  Zn: [0.490, 0.502, 0.690],  // Blue-gray
  Cu: [0.784, 0.502, 0.200],  // Copper
  Mg: [0.541, 1.000, 0.000],  // Green
  Ca: [0.239, 1.000, 0.000],  // Green
  Na: [0.671, 0.361, 0.949],  // Purple
  K:  [0.561, 0.251, 0.831],  // Purple
  // Default for unknown elements
  DEFAULT: [0.800, 0.400, 0.800]  // Pink
};

// Van der Waals radii in Angstroms (Bondi/commonly used approximations)
const ELEMENT_RADII = {
  H:  1.20,
  C:  1.70,
  N:  1.55,
  O:  1.52,
  F:  1.47,
  Cl: 1.75,
  Br: 1.85,
  I:  1.98,
  S:  1.80,
  P:  1.80,
  Fe: 1.80,
  Zn: 1.39,
  Cu: 1.40,
  Mg: 1.73,
  Ca: 2.31,
  Na: 2.27,
  K:  2.75,
  DEFAULT: 1.70
};

// Bond display radius
const BOND_RADIUS = 0.15;
const BOND_COLOR = [0.9, 0.9, 0.9];  // White/light gray

/**
 * Parse a PDB file and extract atoms and bonds.
 * @param {string} text - PDB file content
 * @returns {{atoms: Array, bonds: Array}}
 */
export function parsePDB(text) {
  const atoms = [];
  const bonds = [];
  const lines = text.split('\n');

  // Map from atom serial number to index
  const atomIndexMap = new Map();

  for (const line of lines) {
    const recordType = line.substring(0, 6).trim();

    if (recordType === 'ATOM' || recordType === 'HETATM') {
      const serial = parseInt(line.substring(6, 11).trim(), 10);
      const name = line.substring(12, 16).trim();
      const x = parseFloat(line.substring(30, 38).trim());
      const y = parseFloat(line.substring(38, 46).trim());
      const z = parseFloat(line.substring(46, 54).trim());

      // Element symbol - try column 77-78 first, then extract from atom name
      let element = line.substring(76, 78).trim();
      if (!element) {
        // Extract from atom name (first 1-2 letters)
        element = name.replace(/[0-9]/g, '').substring(0, 2).trim();
        if (element.length > 1) {
          element = element[0].toUpperCase() + element[1].toLowerCase();
        }
      }
      element = element.toUpperCase();
      if (element.length === 2 && !ELEMENT_RADII[element]) {
        element = element[0]; // Try single letter if two-letter not found
      }

      atomIndexMap.set(serial, atoms.length);
      atoms.push({
        serial,
        name,
        element,
        position: [x, y, z],
        isHet: recordType === 'HETATM'
      });
    } else if (recordType === 'CONECT') {
      // Parse connectivity records
      const serial = parseInt(line.substring(6, 11).trim(), 10);
      const fromIndex = atomIndexMap.get(serial);
      if (fromIndex === undefined) continue;

      // Parse bonded atoms (columns 11-16, 16-21, 21-26, 26-31)
      for (let col = 11; col < 31; col += 5) {
        const bondedStr = line.substring(col, col + 5).trim();
        if (!bondedStr) continue;
        const bondedSerial = parseInt(bondedStr, 10);
        const toIndex = atomIndexMap.get(bondedSerial);
        if (toIndex !== undefined && fromIndex < toIndex) {
          // Only add bond once (from lower to higher index)
          bonds.push([fromIndex, toIndex]);
        }
      }
    }
  }

  // If no CONECT records, generate bonds based on distance
  if (bonds.length === 0 && atoms.length > 1) {
    generateBondsFromDistance(atoms, bonds);
  }

  return { atoms, bonds };
}

/**
 * Parse an SDF/MOL file and extract atoms and bonds.
 * @param {string} text - SDF/MOL file content
 * @returns {{atoms: Array, bonds: Array}}
 */
export function parseSDF(text) {
  const atoms = [];
  const bonds = [];
  const lines = text.split('\n');

  // Find counts line (line 4, 0-indexed line 3)
  // Format: aaabbblllfffcccsssxxxrrrpppiiimmmvvvvvv
  // aaa = number of atoms, bbb = number of bonds
  let lineIndex = 3;
  if (lines.length < 4) {
    throw new Error('Invalid SDF file: too few lines');
  }

  const countsLine = lines[lineIndex];
  const atomCount = parseInt(countsLine.substring(0, 3).trim(), 10);
  const bondCount = parseInt(countsLine.substring(3, 6).trim(), 10);

  if (isNaN(atomCount) || isNaN(bondCount)) {
    throw new Error('Invalid SDF file: cannot parse atom/bond counts');
  }

  // Parse atoms (starts at line 5, 0-indexed line 4)
  lineIndex = 4;
  for (let i = 0; i < atomCount; i++) {
    const line = lines[lineIndex + i];
    if (!line) continue;

    // Format: x y z symbol ...
    const x = parseFloat(line.substring(0, 10).trim());
    const y = parseFloat(line.substring(10, 20).trim());
    const z = parseFloat(line.substring(20, 30).trim());
    const element = line.substring(31, 34).trim().toUpperCase();

    atoms.push({
      serial: i + 1,
      name: element + (i + 1),
      element: element.length === 2 ? element[0] + element[1].toLowerCase() : element,
      position: [x, y, z],
      isHet: false
    });
  }

  // Parse bonds
  lineIndex = 4 + atomCount;
  for (let i = 0; i < bondCount; i++) {
    const line = lines[lineIndex + i];
    if (!line) continue;

    // Format: 111222tttsssxxxrrrccc
    // 111 = first atom, 222 = second atom, ttt = bond type
    const atom1 = parseInt(line.substring(0, 3).trim(), 10) - 1;
    const atom2 = parseInt(line.substring(3, 6).trim(), 10) - 1;

    if (atom1 >= 0 && atom2 >= 0 && atom1 < atomCount && atom2 < atomCount) {
      bonds.push([atom1, atom2]);
    }
  }

  return { atoms, bonds };
}

// Covalent radii for bond detection (slightly larger than VdW for bonding)
const COVALENT_RADII = {
  H:  0.31, C:  0.76, N:  0.71, O:  0.66, S:  1.05, P:  1.07,
  F:  0.57, Cl: 1.02, Br: 1.20, I:  1.39, Fe: 1.32, Zn: 1.22,
  Ca: 1.76, Mg: 1.41, Na: 1.66, K:  2.03, DEFAULT: 0.80
};

/**
 * Generate bonds based on interatomic distances.
 * Uses covalent radii with tolerance to determine bonding.
 * Also handles backbone connectivity for proteins.
 */
function generateBondsFromDistance(atoms, bonds) {
  const tolerance = 0.45; // Angstroms tolerance for bonds
  const minDist = 0.4;    // Minimum distance (avoid self-bonds)

  // Build spatial hash for faster neighbor lookup
  const cellSize = 2.5; // Angstroms - larger than max bond length
  const cells = new Map();

  for (let i = 0; i < atoms.length; i++) {
    const a = atoms[i];
    const cx = Math.floor(a.position[0] / cellSize);
    const cy = Math.floor(a.position[1] / cellSize);
    const cz = Math.floor(a.position[2] / cellSize);
    const key = `${cx},${cy},${cz}`;
    if (!cells.has(key)) cells.set(key, []);
    cells.get(key).push(i);
  }

  // Check each atom against neighbors in adjacent cells
  const checkedPairs = new Set();

  for (let i = 0; i < atoms.length; i++) {
    const a1 = atoms[i];
    const r1 = COVALENT_RADII[a1.element] || COVALENT_RADII.DEFAULT;
    const cx = Math.floor(a1.position[0] / cellSize);
    const cy = Math.floor(a1.position[1] / cellSize);
    const cz = Math.floor(a1.position[2] / cellSize);

    // Check 27 neighboring cells (including self)
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        for (let dz = -1; dz <= 1; dz++) {
          const key = `${cx + dx},${cy + dy},${cz + dz}`;
          const cell = cells.get(key);
          if (!cell) continue;

          for (const j of cell) {
            if (j <= i) continue; // Only check pairs once
            const pairKey = `${i},${j}`;
            if (checkedPairs.has(pairKey)) continue;
            checkedPairs.add(pairKey);

            const a2 = atoms[j];
            const r2 = COVALENT_RADII[a2.element] || COVALENT_RADII.DEFAULT;

            const px = a1.position[0] - a2.position[0];
            const py = a1.position[1] - a2.position[1];
            const pz = a1.position[2] - a2.position[2];
            const dist = Math.sqrt(px * px + py * py + pz * pz);

            // Bond if distance is within covalent radii sum + tolerance
            const maxDist = r1 + r2 + tolerance;
            if (dist >= minDist && dist <= maxDist) {
              bonds.push([i, j]);
            }
          }
        }
      }
    }
  }
}

/**
 * Convert parsed molecular data to spheres and cylinders for rendering.
 * @param {{atoms: Array, bonds: Array}} molData - Parsed molecular data
 * @param {Object} options - Conversion options
 * @returns {{spheres: Array, cylinders: Array}}
 */
export function moleculeToGeometry(molData, options = {}) {
  const radiusScale = options.radiusScale ?? 0.4;  // Scale down VdW radii for ball-and-stick
  const bondRadius = options.bondRadius ?? BOND_RADIUS;
  const bondColor = options.bondColor || BOND_COLOR;
  const showBonds = options.showBonds !== false;  // Default to true

  const spheres = [];
  const cylinders = [];

  // Create spheres for atoms
  for (const atom of molData.atoms) {
    const element = atom.element;
    const radius = (ELEMENT_RADII[element] || ELEMENT_RADII.DEFAULT) * radiusScale;
    const color = ELEMENT_COLORS[element] || ELEMENT_COLORS.DEFAULT;

    spheres.push({
      center: atom.position,
      radius,
      color
    });
  }

  // Create cylinders for bonds
  if (showBonds && bondRadius > 0) {
    for (const [i, j] of molData.bonds) {
      const a1 = molData.atoms[i];
      const a2 = molData.atoms[j];

      cylinders.push({
        p1: a1.position,
        p2: a2.position,
        radius: bondRadius,
        color: bondColor
      });
    }
  }

  return { spheres, cylinders };
}

export function splitMolDataByHetatm(molData) {
  const standardAtoms = [];
  const heteroAtoms = [];
  const standardMap = new Map();
  const heteroMap = new Map();

  molData.atoms.forEach((atom, idx) => {
    if (atom.isHet) {
      heteroMap.set(idx, heteroAtoms.length);
      heteroAtoms.push(atom);
    } else {
      standardMap.set(idx, standardAtoms.length);
      standardAtoms.push(atom);
    }
  });

  const standardBonds = [];
  const heteroBonds = [];
  for (const [i, j] of molData.bonds) {
    const iHet = heteroMap.has(i);
    const jHet = heteroMap.has(j);
    if (iHet && jHet) {
      heteroBonds.push([heteroMap.get(i), heteroMap.get(j)]);
    } else if (!iHet && !jHet) {
      standardBonds.push([standardMap.get(i), standardMap.get(j)]);
    }
  }

  return {
    standard: { atoms: standardAtoms, bonds: standardBonds },
    hetero: { atoms: heteroAtoms, bonds: heteroBonds }
  };
}

/**
 * Fetch and parse a PDB file from RCSB.
 * @param {string} pdbId - 4-letter PDB ID
 * @returns {Promise<{atoms: Array, bonds: Array}>}
 */
export async function fetchPDB(pdbId) {
  const url = `https://files.rcsb.org/download/${pdbId.toUpperCase()}.pdb`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch PDB ${pdbId}: ${response.statusText}`);
  }
  const text = await response.text();
  return parsePDB(text);
}

/**
 * Detect file type from content and parse accordingly.
 * @param {string} text - File content
 * @param {string} filename - Original filename for type detection
 * @returns {{atoms: Array, bonds: Array}}
 */
export function parseAutoDetect(text, filename = '') {
  const ext = filename.toLowerCase().split('.').pop();

  if (ext === 'pdb' || text.includes('ATOM  ') || text.includes('HETATM')) {
    return parsePDB(text);
  } else if (ext === 'sdf' || ext === 'mol' || text.includes('V2000') || text.includes('V3000')) {
    return parseSDF(text);
  } else {
    // Try PDB first, then SDF
    try {
      return parsePDB(text);
    } catch {
      return parseSDF(text);
    }
  }
}

/**
 * Built-in small molecule SDF data.
 * Generated from SMILES using RDKit ETKDGv3 + MMFF94 optimization.
 */
export const BUILTIN_MOLECULES = {
  caffeine: `Caffeine
     RDKit          3D

 24 25  0  0  0  0  0  0  0  0999 V2000
    3.2658    0.6194   -0.2409 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.1249   -0.2595   -0.2859 N   0  0  0  0  0  0  0  0  0  0  0  0
    2.1488   -1.6034   -0.5505 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.9392   -2.1268   -0.5185 N   0  0  0  0  0  0  0  0  0  0  0  0
    0.1233   -1.0766   -0.2235 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.8204    0.0837   -0.0745 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.2062    1.3271    0.2358 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.8439    2.3674    0.3680 O   0  0  0  0  0  0  0  0  0  0  0  0
   -1.1817    1.2264    0.3699 N   0  0  0  0  0  0  0  0  0  0  0  0
   -1.9388    0.0449    0.2235 C   0  0  0  0  0  0  0  0  0  0  0  0
   -3.1657    0.0569    0.3617 O   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2423   -1.1249   -0.0817 N   0  0  0  0  0  0  0  0  0  0  0  0
   -1.9467   -2.3829   -0.2489 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.9266    2.4291    0.6868 C   0  0  0  0  0  0  0  0  0  0  0  0
    4.1739    0.0491   -0.4527 H   0  0  0  0  0  0  0  0  0  0  0  0
    3.1354    1.3980   -0.9966 H   0  0  0  0  0  0  0  0  0  0  0  0
    3.3298    1.0558    0.7589 H   0  0  0  0  0  0  0  0  0  0  0  0
    3.0626   -2.1463   -0.7575 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.5722   -3.1026    0.4860 H   0  0  0  0  0  0  0  0  0  0  0  0
   -3.0246   -2.2689   -0.1073 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.7655   -2.7624   -1.2595 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.4668    2.2755    1.6268 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.6587    2.6133   -0.1063 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2845    3.3077    0.7869 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  1  0
  3  4  2  0
  4  5  1  0
  5  6  2  0
  6  7  1  0
  7  8  2  0
  7  9  1  0
  9 10  1  0
 10 11  2  0
 10 12  1  0
 12 13  1  0
  9 14  1  0
  6  2  1  0
 12  5  1  0
  1 15  1  0
  1 16  1  0
  1 17  1  0
  3 18  1  0
 13 19  1  0
 13 20  1  0
 13 21  1  0
 14 22  1  0
 14 23  1  0
 14 24  1  0
M  END`,

  aspirin: `Aspirin
     RDKit          3D

 21 21  0  0  0  0  0  0  0  0999 V2000
   -2.5960   -2.2696    0.0648 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.7187   -1.1363   -0.3760 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.5694   -0.7889   -1.5405 O   0  0  0  0  0  0  0  0  0  0  0  0
   -1.1424   -0.5510    0.7460 O   0  0  0  0  0  0  0  0  0  0  0  0
   -0.2349    0.4604    0.4178 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.6980    1.7768    0.5130 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.1642    2.8335    0.2273 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.4846    2.5754   -0.1395 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.9497    1.2583   -0.2153 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.0950    0.1821    0.0702 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.5903   -1.2187    0.0199 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.9771   -2.2262    0.3208 O   0  0  0  0  0  0  0  0  0  0  0  0
    2.8642   -1.3007   -0.4107 O   0  0  0  0  0  0  0  0  0  0  0  0
   -3.0065   -2.7726   -0.8155 H   0  0  0  0  0  0  0  0  0  0  0  0
   -3.4223   -1.8851    0.6675 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.0086   -2.9949    0.6336 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.7266    1.9769    0.7996 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.1930    3.8587    0.2884 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.1566    3.4003   -0.3651 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.9861    1.0849   -0.4964 H   0  0  0  0  0  0  0  0  0  0  0  0
    3.0487   -2.2633   -0.4100 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  2  0
  2  4  1  0
  4  5  1  0
  5  6  2  0
  6  7  1  0
  7  8  2  0
  8  9  1  0
  9 10  2  0
 10 11  1  0
 11 12  2  0
 11 13  1  0
 10  5  1  0
  1 14  1  0
  1 15  1  0
  1 16  1  0
  6 17  1  0
  7 18  1  0
  8 19  1  0
  9 20  1  0
 13 21  1  0
M  END`,

  benzene: `Benzene
     RDKit          3D

 12 12  0  0  0  0  0  0  0  0999 V2000
    0.8035   -1.1401   -0.0082 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.3889    0.1258   -0.0281 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.5853    1.2659   -0.0199 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.8035    1.1401    0.0083 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.3889   -0.1258    0.0281 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.5853   -1.2659    0.0199 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.4295   -2.0284   -0.0147 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.4709    0.2238   -0.0501 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.0414    2.2522   -0.0354 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.4295    2.0284    0.0147 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.4709   -0.2238    0.0500 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.0414   -2.2522    0.0354 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  2  0
  2  3  1  0
  3  4  2  0
  4  5  1  0
  5  6  2  0
  6  1  1  0
  1  7  1  0
  2  8  1  0
  3  9  1  0
  4 10  1  0
  5 11  1  0
  6 12  1  0
M  END`,

  ethanol: `Ethanol
     RDKit          3D

  9  8  0  0  0  0  0  0  0  0999 V2000
   -0.8883    0.1670   -0.0273 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.4658   -0.5116   -0.0368 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.4311    0.3229    0.5867 O   0  0  0  0  0  0  0  0  0  0  0  0
   -0.8487    1.1175   -0.5695 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.6471   -0.4704   -0.4896 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.1964    0.3978    0.9977 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.7920   -0.7224   -1.0597 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.4246   -1.4559    0.5138 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.4671    1.1550    0.0848 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  1  0
  1  4  1  0
  1  5  1  0
  1  6  1  0
  2  7  1  0
  2  8  1  0
  3  9  1  0
M  END`,

  ibuprofen: `Ibuprofen
     RDKit          3D

 33 33  0  0  0  0  0  0  0  0999 V2000
   -4.7223   -0.5937   -0.0963 C   0  0  0  0  0  0  0  0  0  0  0  0
   -3.2926   -0.0634   -0.2299 C   0  0  0  0  0  0  0  0  0  0  0  0
   -3.2135    1.3545    0.3419 C   0  0  0  0  0  0  0  0  0  0  0  0
   -2.3120   -1.0244    0.4705 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.8585   -0.7055    0.2147 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.0616   -0.1539    1.2235 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.2826    0.1444    0.9850 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.8629   -0.1044   -0.2672 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.0606   -0.6609   -1.2745 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.2841   -0.9575   -1.0363 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.3203    0.2221   -0.5548 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.6134    1.7132   -0.3864 C   0  0  0  0  0  0  0  0  0  0  0  0
    4.2969   -0.5799    0.2886 C   0  0  0  0  0  0  0  0  0  0  0  0
    4.1498   -0.9950    1.4262 O   0  0  0  0  0  0  0  0  0  0  0  0
    5.4727   -0.7833   -0.3475 O   0  0  0  0  0  0  0  0  0  0  0  0
   -5.0250   -0.6615    0.9542 H   0  0  0  0  0  0  0  0  0  0  0  0
   -4.8102   -1.5911   -0.5401 H   0  0  0  0  0  0  0  0  0  0  0  0
   -5.4310    0.0631   -0.6117 H   0  0  0  0  0  0  0  0  0  0  0  0
   -3.0546   -0.0123   -1.2999 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.2266    1.7970    0.1741 H   0  0  0  0  0  0  0  0  0  0  0  0
   -3.4111    1.3602    1.4192 H   0  0  0  0  0  0  0  0  0  0  0  0
   -3.9494    2.0078   -0.1393 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.5074   -1.0352    1.5509 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.4921   -2.0526    0.1289 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.4801    0.0479    2.2070 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.8762    0.5619    1.7961 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.4734   -0.8686   -2.2594 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.8819   -1.3895   -1.8358 H   0  0  0  0  0  0  0  0  0  0  0  0
    3.5204   -0.0441   -1.6018 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.9371    2.3199   -0.9986 H   0  0  0  0  0  0  0  0  0  0  0  0
    4.6410    1.9430   -0.6895 H   0  0  0  0  0  0  0  0  0  0  0  0
    3.5062    2.0350    0.6556 H   0  0  0  0  0  0  0  0  0  0  0  0
    6.0005   -1.2931    0.3031 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  1  0
  2  4  1  0
  4  5  1  0
  5  6  2  0
  6  7  1  0
  7  8  2  0
  8  9  1  0
  9 10  2  0
  8 11  1  0
 11 12  1  0
 11 13  1  0
 13 14  2  0
 13 15  1  0
 10  5  1  0
  1 16  1  0
  1 17  1  0
  1 18  1  0
  2 19  1  0
  3 20  1  0
  3 21  1  0
  3 22  1  0
  4 23  1  0
  4 24  1  0
  6 25  1  0
  7 26  1  0
  9 27  1  0
 10 28  1  0
 11 29  1  0
 12 30  1  0
 12 31  1  0
 12 32  1  0
 15 33  1  0
M  END`,

  glucose: `Glucose
     RDKit          3D

 24 24  0  0  0  0  0  0  0  0999 V2000
    3.2051    0.8524   -0.2701 O   0  0  0  0  0  0  0  0  0  0  0  0
    2.4883   -0.2408    0.2960 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.0329   -0.2801   -0.2001 C   0  0  1  0  0  0  0  0  0  0  0  0
    0.4001   -1.3999    0.4332 O   0  0  0  0  0  0  0  0  0  0  0  0
   -0.9615   -1.5571    0.0711 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.0784   -1.8385   -1.3183 O   0  0  0  0  0  0  0  0  0  0  0  0
   -1.7909   -0.3152    0.4191 C   0  0  2  0  0  0  0  0  0  0  0  0
   -3.1481   -0.4610   -0.0363 O   0  0  0  0  0  0  0  0  0  0  0  0
   -1.1893    0.9402   -0.2154 C   0  0  1  0  0  0  0  0  0  0  0  0
   -1.8920    2.0840    0.3063 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.3112    1.0548    0.0838 C   0  0  2  0  0  0  0  0  0  0  0  0
    0.8681    2.1054   -0.7198 O   0  0  0  0  0  0  0  0  0  0  0  0
    4.1300    0.7641    0.0260 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.5196   -0.1531    1.3879 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.9982   -1.1735    0.0297 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.0645   -0.4428   -1.2859 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.3686   -2.4193    0.6109 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7640   -2.7541   -1.4189 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.8276   -0.1846    1.5072 H   0  0  0  0  0  0  0  0  0  0  0  0
   -3.0812   -0.7998   -0.9521 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.3619    0.9539   -1.2986 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.8405    1.8910    0.1810 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.4427    1.3443    1.1338 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.8432    2.0297   -0.6450 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  1  0
  3  4  1  0
  4  5  1  0
  5  6  1  0
  5  7  1  0
  7  8  1  0
  7  9  1  0
  9 10  1  0
  9 11  1  0
 11 12  1  0
 11  3  1  0
  1 13  1  0
  2 14  1  0
  2 15  1  0
  3 16  1  6
  5 17  1  0
  6 18  1  0
  7 19  1  1
  8 20  1  0
  9 21  1  6
 10 22  1  0
 11 23  1  1
 12 24  1  0
M  END`
};

/**
 * Get a built-in molecule by name.
 * @param {string} name - Molecule name (e.g., 'caffeine', 'aspirin')
 * @returns {{atoms: Array, bonds: Array}}
 */
export function getBuiltinMolecule(name) {
  const sdf = BUILTIN_MOLECULES[name.toLowerCase()];
  if (!sdf) {
    throw new Error(`Unknown molecule: ${name}. Available: ${Object.keys(BUILTIN_MOLECULES).join(', ')}`);
  }
  return parseSDF(sdf);
}
