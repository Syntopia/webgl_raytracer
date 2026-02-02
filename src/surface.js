/**
 * Solvent Excluded Surface (SES) computation using distance grids.
 *
 * Algorithm:
 * 1. Create distance grid for atom spheres expanded by probe radius (SAS)
 * 2. Extract SAS contour surface using marching cubes at level 0
 * 3. Place probe spheres at each SAS vertex, create new distance grid
 * 4. Extract SES contour surface using marching cubes
 * 5. Filter out extra surfaces by checking distance to original atoms
 */

// Marching cubes edge table - which edges are intersected for each case
const EDGE_TABLE = new Uint16Array([
  0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
  0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
  0x190, 0x99, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
  0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
  0x230, 0x339, 0x33, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
  0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
  0x3a0, 0x2a9, 0x1a3, 0xaa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
  0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
  0x460, 0x569, 0x663, 0x76a, 0x66, 0x16f, 0x265, 0x36c,
  0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
  0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff, 0x3f5, 0x2fc,
  0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
  0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55, 0x15c,
  0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
  0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc,
  0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
  0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
  0xcc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
  0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
  0x15c, 0x55, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
  0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
  0x2fc, 0x3f5, 0xff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
  0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
  0x36c, 0x265, 0x16f, 0x66, 0x76a, 0x663, 0x569, 0x460,
  0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
  0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa, 0x1a3, 0x2a9, 0x3a0,
  0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
  0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33, 0x339, 0x230,
  0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
  0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99, 0x190,
  0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
  0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0
]);

// Triangle table - which vertices form triangles for each case
// -1 marks end of list
const TRI_TABLE = [
  [-1],
  [0, 8, 3, -1],
  [0, 1, 9, -1],
  [1, 8, 3, 9, 8, 1, -1],
  [1, 2, 10, -1],
  [0, 8, 3, 1, 2, 10, -1],
  [9, 2, 10, 0, 2, 9, -1],
  [2, 8, 3, 2, 10, 8, 10, 9, 8, -1],
  [3, 11, 2, -1],
  [0, 11, 2, 8, 11, 0, -1],
  [1, 9, 0, 2, 3, 11, -1],
  [1, 11, 2, 1, 9, 11, 9, 8, 11, -1],
  [3, 10, 1, 11, 10, 3, -1],
  [0, 10, 1, 0, 8, 10, 8, 11, 10, -1],
  [3, 9, 0, 3, 11, 9, 11, 10, 9, -1],
  [9, 8, 10, 10, 8, 11, -1],
  [4, 7, 8, -1],
  [4, 3, 0, 7, 3, 4, -1],
  [0, 1, 9, 8, 4, 7, -1],
  [4, 1, 9, 4, 7, 1, 7, 3, 1, -1],
  [1, 2, 10, 8, 4, 7, -1],
  [3, 4, 7, 3, 0, 4, 1, 2, 10, -1],
  [9, 2, 10, 9, 0, 2, 8, 4, 7, -1],
  [2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1],
  [8, 4, 7, 3, 11, 2, -1],
  [11, 4, 7, 11, 2, 4, 2, 0, 4, -1],
  [9, 0, 1, 8, 4, 7, 2, 3, 11, -1],
  [4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1],
  [3, 10, 1, 3, 11, 10, 7, 8, 4, -1],
  [1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1],
  [4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1],
  [4, 7, 11, 4, 11, 9, 9, 11, 10, -1],
  [9, 5, 4, -1],
  [9, 5, 4, 0, 8, 3, -1],
  [0, 5, 4, 1, 5, 0, -1],
  [8, 5, 4, 8, 3, 5, 3, 1, 5, -1],
  [1, 2, 10, 9, 5, 4, -1],
  [3, 0, 8, 1, 2, 10, 4, 9, 5, -1],
  [5, 2, 10, 5, 4, 2, 4, 0, 2, -1],
  [2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1],
  [9, 5, 4, 2, 3, 11, -1],
  [0, 11, 2, 0, 8, 11, 4, 9, 5, -1],
  [0, 5, 4, 0, 1, 5, 2, 3, 11, -1],
  [2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1],
  [10, 3, 11, 10, 1, 3, 9, 5, 4, -1],
  [4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1],
  [5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1],
  [5, 4, 8, 5, 8, 10, 10, 8, 11, -1],
  [9, 7, 8, 5, 7, 9, -1],
  [9, 3, 0, 9, 5, 3, 5, 7, 3, -1],
  [0, 7, 8, 0, 1, 7, 1, 5, 7, -1],
  [1, 5, 3, 3, 5, 7, -1],
  [9, 7, 8, 9, 5, 7, 10, 1, 2, -1],
  [10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1],
  [8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1],
  [2, 10, 5, 2, 5, 3, 3, 5, 7, -1],
  [7, 9, 5, 7, 8, 9, 3, 11, 2, -1],
  [9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1],
  [2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1],
  [11, 2, 1, 11, 1, 7, 7, 1, 5, -1],
  [9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1],
  [5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1],
  [11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1],
  [11, 10, 5, 7, 11, 5, -1],
  [10, 6, 5, -1],
  [0, 8, 3, 5, 10, 6, -1],
  [9, 0, 1, 5, 10, 6, -1],
  [1, 8, 3, 1, 9, 8, 5, 10, 6, -1],
  [1, 6, 5, 2, 6, 1, -1],
  [1, 6, 5, 1, 2, 6, 3, 0, 8, -1],
  [9, 6, 5, 9, 0, 6, 0, 2, 6, -1],
  [5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1],
  [2, 3, 11, 10, 6, 5, -1],
  [11, 0, 8, 11, 2, 0, 10, 6, 5, -1],
  [0, 1, 9, 2, 3, 11, 5, 10, 6, -1],
  [5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1],
  [6, 3, 11, 6, 5, 3, 5, 1, 3, -1],
  [0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1],
  [3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1],
  [6, 5, 9, 6, 9, 11, 11, 9, 8, -1],
  [5, 10, 6, 4, 7, 8, -1],
  [4, 3, 0, 4, 7, 3, 6, 5, 10, -1],
  [1, 9, 0, 5, 10, 6, 8, 4, 7, -1],
  [10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1],
  [6, 1, 2, 6, 5, 1, 4, 7, 8, -1],
  [1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1],
  [8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1],
  [7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1],
  [3, 11, 2, 7, 8, 4, 10, 6, 5, -1],
  [5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1],
  [0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1],
  [9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1],
  [8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1],
  [5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1],
  [0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1],
  [6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1],
  [10, 4, 9, 6, 4, 10, -1],
  [4, 10, 6, 4, 9, 10, 0, 8, 3, -1],
  [10, 0, 1, 10, 6, 0, 6, 4, 0, -1],
  [8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1],
  [1, 4, 9, 1, 2, 4, 2, 6, 4, -1],
  [3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1],
  [0, 2, 4, 4, 2, 6, -1],
  [8, 3, 2, 8, 2, 4, 4, 2, 6, -1],
  [10, 4, 9, 10, 6, 4, 11, 2, 3, -1],
  [0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1],
  [3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1],
  [6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1],
  [9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1],
  [8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1],
  [3, 11, 6, 3, 6, 0, 0, 6, 4, -1],
  [6, 4, 8, 11, 6, 8, -1],
  [7, 10, 6, 7, 8, 10, 8, 9, 10, -1],
  [0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1],
  [10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1],
  [10, 6, 7, 10, 7, 1, 1, 7, 3, -1],
  [1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1],
  [2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1],
  [7, 8, 0, 7, 0, 6, 6, 0, 2, -1],
  [7, 3, 2, 6, 7, 2, -1],
  [2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1],
  [2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1],
  [1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1],
  [11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1],
  [8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1],
  [0, 9, 1, 11, 6, 7, -1],
  [7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1],
  [7, 11, 6, -1],
  [7, 6, 11, -1],
  [3, 0, 8, 11, 7, 6, -1],
  [0, 1, 9, 11, 7, 6, -1],
  [8, 1, 9, 8, 3, 1, 11, 7, 6, -1],
  [10, 1, 2, 6, 11, 7, -1],
  [1, 2, 10, 3, 0, 8, 6, 11, 7, -1],
  [2, 9, 0, 2, 10, 9, 6, 11, 7, -1],
  [6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1],
  [7, 2, 3, 6, 2, 7, -1],
  [7, 0, 8, 7, 6, 0, 6, 2, 0, -1],
  [2, 7, 6, 2, 3, 7, 0, 1, 9, -1],
  [1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1],
  [10, 7, 6, 10, 1, 7, 1, 3, 7, -1],
  [10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1],
  [0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1],
  [7, 6, 10, 7, 10, 8, 8, 10, 9, -1],
  [6, 8, 4, 11, 8, 6, -1],
  [3, 6, 11, 3, 0, 6, 0, 4, 6, -1],
  [8, 6, 11, 8, 4, 6, 9, 0, 1, -1],
  [9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1],
  [6, 8, 4, 6, 11, 8, 2, 10, 1, -1],
  [1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1],
  [4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1],
  [10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1],
  [8, 2, 3, 8, 4, 2, 4, 6, 2, -1],
  [0, 4, 2, 4, 6, 2, -1],
  [1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1],
  [1, 9, 4, 1, 4, 2, 2, 4, 6, -1],
  [8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1],
  [10, 1, 0, 10, 0, 6, 6, 0, 4, -1],
  [4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1],
  [10, 9, 4, 6, 10, 4, -1],
  [4, 9, 5, 7, 6, 11, -1],
  [0, 8, 3, 4, 9, 5, 11, 7, 6, -1],
  [5, 0, 1, 5, 4, 0, 7, 6, 11, -1],
  [11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1],
  [9, 5, 4, 10, 1, 2, 7, 6, 11, -1],
  [6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1],
  [7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1],
  [3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1],
  [7, 2, 3, 7, 6, 2, 5, 4, 9, -1],
  [9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1],
  [3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1],
  [6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1],
  [9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1],
  [1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1],
  [4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1],
  [7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1],
  [6, 9, 5, 6, 11, 9, 11, 8, 9, -1],
  [3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1],
  [0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1],
  [6, 11, 3, 6, 3, 5, 5, 3, 1, -1],
  [1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1],
  [0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1],
  [11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1],
  [6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1],
  [5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1],
  [9, 5, 6, 9, 6, 0, 0, 6, 2, -1],
  [1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1],
  [1, 5, 6, 2, 1, 6, -1],
  [1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1],
  [10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1],
  [0, 3, 8, 5, 6, 10, -1],
  [10, 5, 6, -1],
  [11, 5, 10, 7, 5, 11, -1],
  [11, 5, 10, 11, 7, 5, 8, 3, 0, -1],
  [5, 11, 7, 5, 10, 11, 1, 9, 0, -1],
  [10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1],
  [11, 1, 2, 11, 7, 1, 7, 5, 1, -1],
  [0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1],
  [9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1],
  [7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1],
  [2, 5, 10, 2, 3, 5, 3, 7, 5, -1],
  [8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1],
  [9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1],
  [9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1],
  [1, 3, 5, 3, 7, 5, -1],
  [0, 8, 7, 0, 7, 1, 1, 7, 5, -1],
  [9, 0, 3, 9, 3, 5, 5, 3, 7, -1],
  [9, 8, 7, 5, 9, 7, -1],
  [5, 8, 4, 5, 10, 8, 10, 11, 8, -1],
  [5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1],
  [0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1],
  [10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1],
  [2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1],
  [0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1],
  [0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1],
  [9, 4, 5, 2, 11, 3, -1],
  [2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1],
  [5, 10, 2, 5, 2, 4, 4, 2, 0, -1],
  [3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1],
  [5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1],
  [8, 4, 5, 8, 5, 3, 3, 5, 1, -1],
  [0, 4, 5, 1, 0, 5, -1],
  [8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1],
  [9, 4, 5, -1],
  [4, 11, 7, 4, 9, 11, 9, 10, 11, -1],
  [0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1],
  [1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1],
  [3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1],
  [4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1],
  [9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1],
  [11, 7, 4, 11, 4, 2, 2, 4, 0, -1],
  [11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1],
  [2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1],
  [9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1],
  [3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1],
  [1, 10, 2, 8, 7, 4, -1],
  [4, 9, 1, 4, 1, 7, 7, 1, 3, -1],
  [4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1],
  [4, 0, 3, 7, 4, 3, -1],
  [4, 8, 7, -1],
  [9, 10, 8, 10, 11, 8, -1],
  [3, 0, 9, 3, 9, 11, 11, 9, 10, -1],
  [0, 1, 10, 0, 10, 8, 8, 10, 11, -1],
  [3, 1, 10, 11, 3, 10, -1],
  [1, 2, 11, 1, 11, 9, 9, 11, 8, -1],
  [3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1],
  [0, 2, 11, 8, 0, 11, -1],
  [3, 2, 11, -1],
  [2, 3, 8, 2, 8, 10, 10, 8, 9, -1],
  [9, 10, 2, 0, 9, 2, -1],
  [2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1],
  [1, 10, 2, -1],
  [1, 3, 8, 9, 1, 8, -1],
  [0, 9, 1, -1],
  [0, 3, 8, -1],
  [-1]
];

/**
 * Create a 3D grid for distance field computation.
 */
class DistanceGrid {
  constructor(bounds, resolution, options = {}) {
    const padding = options.padding ?? 3.0; // Extra padding around bounds
    const maxIndexRange = options.maxIndexRange ?? 2;

    this.resolution = resolution;
    this.maxDist = maxIndexRange * resolution;
    this.defaultValue = this.maxDist;
    this.min = [
      bounds.minX - padding,
      bounds.minY - padding,
      bounds.minZ - padding
    ];
    this.max = [
      bounds.maxX + padding,
      bounds.maxY + padding,
      bounds.maxZ + padding
    ];

    const size = [
      this.max[0] - this.min[0],
      this.max[1] - this.min[1],
      this.max[2] - this.min[2]
    ];

    this.nx = Math.ceil(size[0] / resolution) + 1;
    this.ny = Math.ceil(size[1] / resolution) + 1;
    this.nz = Math.ceil(size[2] / resolution) + 1;

    this.data = new Float32Array(this.nx * this.ny * this.nz);
    this.data.fill(this.defaultValue); // Initialize with large positive values (outside)
  }

  index(ix, iy, iz) {
    return ix + iy * this.nx + iz * this.nx * this.ny;
  }

  get(ix, iy, iz) {
    if (ix < 0 || ix >= this.nx || iy < 0 || iy >= this.ny || iz < 0 || iz >= this.nz) {
      return this.defaultValue;
    }
    return this.data[this.index(ix, iy, iz)];
  }

  set(ix, iy, iz, value) {
    if (ix >= 0 && ix < this.nx && iy >= 0 && iy < this.ny && iz >= 0 && iz < this.nz) {
      this.data[this.index(ix, iy, iz)] = value;
    }
  }

  // Get world position from grid indices
  worldPos(ix, iy, iz) {
    return [
      this.min[0] + ix * this.resolution,
      this.min[1] + iy * this.resolution,
      this.min[2] + iz * this.resolution
    ];
  }

  // Get grid indices from world position
  gridIndices(x, y, z) {
    return [
      Math.floor((x - this.min[0]) / this.resolution),
      Math.floor((y - this.min[1]) / this.resolution),
      Math.floor((z - this.min[2]) / this.resolution)
    ];
  }

  /**
   * Add a sphere to the distance field.
   * Updates grid values to minimum of current value and distance to sphere surface.
   */
  addSphere(center, radius) {
    // Compute bounding box of sphere influence in grid coordinates
    const margin = radius + this.maxDist;
    const [gx0, gy0, gz0] = this.gridIndices(
      center[0] - margin, center[1] - margin, center[2] - margin
    );
    const [gx1, gy1, gz1] = this.gridIndices(
      center[0] + margin, center[1] + margin, center[2] + margin
    );

    for (let iz = Math.max(0, gz0); iz <= Math.min(this.nz - 1, gz1); iz++) {
      for (let iy = Math.max(0, gy0); iy <= Math.min(this.ny - 1, gy1); iy++) {
        for (let ix = Math.max(0, gx0); ix <= Math.min(this.nx - 1, gx1); ix++) {
          const [wx, wy, wz] = this.worldPos(ix, iy, iz);
          const dx = wx - center[0];
          const dy = wy - center[1];
          const dz = wz - center[2];
          let dist = Math.sqrt(dx * dx + dy * dy + dz * dz) - radius;
          if (dist > this.maxDist) dist = this.maxDist;
          else if (dist < -this.maxDist) dist = -this.maxDist;

          const idx = this.index(ix, iy, iz);
          if (dist < this.data[idx]) {
            this.data[idx] = dist;
          }
        }
      }
    }
  }

  /**
   * Clear the grid (reset to Infinity).
   */
  clear() {
    this.data.fill(this.defaultValue);
  }
}

/**
 * Extract isosurface using marching cubes algorithm.
 * @param {DistanceGrid} grid - The distance grid
 * @param {number} isovalue - The contour level (0 for surface)
 * @returns {{vertices: Float32Array, normals: Float32Array, indices: Uint32Array}}
 */
function marchingCubes(grid, isovalue = 0, options = {}) {
  const vertices = [];
  const normals = [];
  const indices = [];
  const smooth = options.smoothNormals ?? false;

  // Edge vertex cache to avoid duplicates - key is "ix,iy,iz,edgeDir"
  const edgeVertexCache = new Map();

  function interpolateVertex(p1, p2, v1, v2) {
    if (Math.abs(isovalue - v1) < 0.00001) return [...p1];
    if (Math.abs(isovalue - v2) < 0.00001) return [...p2];
    if (Math.abs(v1 - v2) < 0.00001) return [...p1];

    const t = (isovalue - v1) / (v2 - v1);
    return [
      p1[0] + t * (p2[0] - p1[0]),
      p1[1] + t * (p2[1] - p1[1]),
      p1[2] + t * (p2[2] - p1[2])
    ];
  }

  function computeNormal(pos) {
    // Compute gradient using trilinear interpolation at the exact vertex position
    // This produces much smoother normals than sampling at grid cell centers
    const h = grid.resolution;

    // Sample gradient at offset positions and use central differences
    // For each axis, we sample the distance field slightly offset and compute the gradient
    const gx = sampleTrilinear(pos[0] + h, pos[1], pos[2]) - sampleTrilinear(pos[0] - h, pos[1], pos[2]);
    const gy = sampleTrilinear(pos[0], pos[1] + h, pos[2]) - sampleTrilinear(pos[0], pos[1] - h, pos[2]);
    const gz = sampleTrilinear(pos[0], pos[1], pos[2] + h) - sampleTrilinear(pos[0], pos[1], pos[2] - h);

    const len = Math.sqrt(gx * gx + gy * gy + gz * gz);
    if (len > 0.0001) {
      return [gx / len, gy / len, gz / len];
    }
    return [0, 1, 0];
  }

  // Trilinear interpolation of the distance field at world position
  function sampleTrilinear(wx, wy, wz) {
    // Convert to continuous grid coordinates
    const fx = (wx - grid.min[0]) / grid.resolution;
    const fy = (wy - grid.min[1]) / grid.resolution;
    const fz = (wz - grid.min[2]) / grid.resolution;

    const ix = Math.floor(fx);
    const iy = Math.floor(fy);
    const iz = Math.floor(fz);

    // Fractional part
    const tx = fx - ix;
    const ty = fy - iy;
    const tz = fz - iz;

    // Sample 8 corners of the cell
    const v000 = grid.get(ix, iy, iz);
    const v100 = grid.get(ix + 1, iy, iz);
    const v010 = grid.get(ix, iy + 1, iz);
    const v110 = grid.get(ix + 1, iy + 1, iz);
    const v001 = grid.get(ix, iy, iz + 1);
    const v101 = grid.get(ix + 1, iy, iz + 1);
    const v011 = grid.get(ix, iy + 1, iz + 1);
    const v111 = grid.get(ix + 1, iy + 1, iz + 1);

    // Handle Infinity values - clamp to grid max distance for interpolation
    const clamp = (v) => (v === Infinity) ? grid.maxDist : ((v === -Infinity) ? -grid.maxDist : v);
    const c000 = clamp(v000), c100 = clamp(v100), c010 = clamp(v010), c110 = clamp(v110);
    const c001 = clamp(v001), c101 = clamp(v101), c011 = clamp(v011), c111 = clamp(v111);

    // Trilinear interpolation
    const c00 = c000 * (1 - tx) + c100 * tx;
    const c10 = c010 * (1 - tx) + c110 * tx;
    const c01 = c001 * (1 - tx) + c101 * tx;
    const c11 = c011 * (1 - tx) + c111 * tx;

    const c0 = c00 * (1 - ty) + c10 * ty;
    const c1 = c01 * (1 - ty) + c11 * ty;

    return c0 * (1 - tz) + c1 * tz;
  }

  function addVertex(pos) {
    const idx = vertices.length / 3;
    vertices.push(pos[0], pos[1], pos[2]);
    const n = computeNormal(pos);
    normals.push(n[0], n[1], n[2]);
    return idx;
  }

  // Standard marching cubes edge definition
  // Edge connects two corners - indices into the corner array
  // Corner order: 0=(0,0,0), 1=(1,0,0), 2=(1,1,0), 3=(0,1,0),
  //               4=(0,0,1), 5=(1,0,1), 6=(1,1,1), 7=(0,1,1)
  const edgeCorners = [
    [0, 1], [1, 2], [2, 3], [3, 0],  // Bottom face edges (z=0)
    [4, 5], [5, 6], [6, 7], [7, 4],  // Top face edges (z=1)
    [0, 4], [1, 5], [2, 6], [3, 7]   // Vertical edges
  ];

  // Corner offsets from cell origin (ix, iy, iz)
  const cornerOffsets = [
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
  ];

  // Edge cache keys: edge direction (0=x, 1=y, 2=z) and position
  function getEdgeCacheKey(ix, iy, iz, edgeIdx) {
    const [c0, c1] = edgeCorners[edgeIdx];
    const o0 = cornerOffsets[c0];
    const o1 = cornerOffsets[c1];
    // Determine edge direction and canonical position
    if (o0[0] !== o1[0]) { // X edge
      return `x,${ix + Math.min(o0[0], o1[0])},${iy + o0[1]},${iz + o0[2]}`;
    } else if (o0[1] !== o1[1]) { // Y edge
      return `y,${ix + o0[0]},${iy + Math.min(o0[1], o1[1])},${iz + o0[2]}`;
    } else { // Z edge
      return `z,${ix + o0[0]},${iy + o0[1]},${iz + Math.min(o0[2], o1[2])}`;
    }
  }

  // Process each cell
  for (let iz = 0; iz < grid.nz - 1; iz++) {
    for (let iy = 0; iy < grid.ny - 1; iy++) {
      for (let ix = 0; ix < grid.nx - 1; ix++) {
        // Get values at cube corners
        const v = new Array(8);
        const corners = new Array(8);
        for (let i = 0; i < 8; i++) {
          const o = cornerOffsets[i];
          v[i] = grid.get(ix + o[0], iy + o[1], iz + o[2]);
          corners[i] = grid.worldPos(ix + o[0], iy + o[1], iz + o[2]);
        }

        // Compute cube index (which corners are inside the surface)
        let cubeIndex = 0;
        for (let i = 0; i < 8; i++) {
          if (v[i] < isovalue) cubeIndex |= (1 << i);
        }

        // Skip if cube is entirely inside or outside
        if (EDGE_TABLE[cubeIndex] === 0) continue;

        // Compute edge vertices
        const edgeVerts = new Array(12).fill(-1);
        for (let e = 0; e < 12; e++) {
          if (EDGE_TABLE[cubeIndex] & (1 << e)) {
            const key = getEdgeCacheKey(ix, iy, iz, e);

            if (edgeVertexCache.has(key)) {
              edgeVerts[e] = edgeVertexCache.get(key);
            } else {
              const [c0, c1] = edgeCorners[e];
              const pos = interpolateVertex(corners[c0], corners[c1], v[c0], v[c1]);
              edgeVerts[e] = addVertex(pos);
              edgeVertexCache.set(key, edgeVerts[e]);
            }
          }
        }

        // Generate triangles from lookup table
        const tris = TRI_TABLE[cubeIndex];
        for (let i = 0; tris[i] !== -1; i += 3) {
          const i0 = edgeVerts[tris[i]];
          const i1 = edgeVerts[tris[i + 1]];
          const i2 = edgeVerts[tris[i + 2]];
          if (i0 >= 0 && i1 >= 0 && i2 >= 0) {
            indices.push(i0, i1, i2);
          }
        }
      }
    }
  }

  // Smooth normals by averaging at each vertex based on adjacent triangles
  const smoothedNormals = smooth ? smoothNormals(vertices, normals, indices) : new Float32Array(normals);

  return {
    vertices: new Float32Array(vertices),
    normals: smoothedNormals,
    indices: new Uint32Array(indices)
  };
}

/**
 * Smooth normals by computing area-weighted average of face normals at each vertex.
 * This produces smoother shading for marching cubes output.
 */
function smoothNormals(vertices, perVertexNormals, indices) {
  const vertexCount = vertices.length / 3;
  const accumulated = new Float32Array(vertexCount * 3);

  // For each triangle, compute face normal and accumulate weighted by area
  for (let i = 0; i < indices.length; i += 3) {
    const i0 = indices[i];
    const i1 = indices[i + 1];
    const i2 = indices[i + 2];

    // Get vertex positions
    const v0x = vertices[i0 * 3], v0y = vertices[i0 * 3 + 1], v0z = vertices[i0 * 3 + 2];
    const v1x = vertices[i1 * 3], v1y = vertices[i1 * 3 + 1], v1z = vertices[i1 * 3 + 2];
    const v2x = vertices[i2 * 3], v2y = vertices[i2 * 3 + 1], v2z = vertices[i2 * 3 + 2];

    // Edge vectors
    const e1x = v1x - v0x, e1y = v1y - v0y, e1z = v1z - v0z;
    const e2x = v2x - v0x, e2y = v2y - v0y, e2z = v2z - v0z;

    // Cross product (not normalized - magnitude is 2x triangle area)
    const nx = e1y * e2z - e1z * e2y;
    const ny = e1z * e2x - e1x * e2z;
    const nz = e1x * e2y - e1y * e2x;

    // Accumulate to all three vertices (area-weighted)
    accumulated[i0 * 3] += nx; accumulated[i0 * 3 + 1] += ny; accumulated[i0 * 3 + 2] += nz;
    accumulated[i1 * 3] += nx; accumulated[i1 * 3 + 1] += ny; accumulated[i1 * 3 + 2] += nz;
    accumulated[i2 * 3] += nx; accumulated[i2 * 3 + 1] += ny; accumulated[i2 * 3 + 2] += nz;
  }

  // Normalize accumulated normals
  const result = new Float32Array(vertexCount * 3);
  for (let i = 0; i < vertexCount; i++) {
    const ax = accumulated[i * 3];
    const ay = accumulated[i * 3 + 1];
    const az = accumulated[i * 3 + 2];
    const len = Math.sqrt(ax * ax + ay * ay + az * az);

    if (len > 0.0001) {
      result[i * 3] = ax / len;
      result[i * 3 + 1] = ay / len;
      result[i * 3 + 2] = az / len;
    } else {
      // Fall back to the per-vertex gradient normal
      result[i * 3] = perVertexNormals[i * 3];
      result[i * 3 + 1] = perVertexNormals[i * 3 + 1];
      result[i * 3 + 2] = perVertexNormals[i * 3 + 2];
    }
  }

  return result;
}

/**
 * Compute connected components of a triangle mesh.
 * Returns array of component indices for each vertex.
 */
function findConnectedComponents(vertices, indices) {
  const vertexCount = vertices.length / 3;
  const parent = new Int32Array(vertexCount);
  for (let i = 0; i < vertexCount; i++) parent[i] = i;

  function find(x) {
    if (parent[x] !== x) parent[x] = find(parent[x]);
    return parent[x];
  }

  function union(x, y) {
    const px = find(x);
    const py = find(y);
    if (px !== py) parent[px] = py;
  }

  // Union vertices connected by edges
  for (let i = 0; i < indices.length; i += 3) {
    union(indices[i], indices[i + 1]);
    union(indices[i + 1], indices[i + 2]);
  }

  // Flatten and renumber components
  const componentMap = new Map();
  const components = new Int32Array(vertexCount);
  let nextComponent = 0;

  for (let i = 0; i < vertexCount; i++) {
    const root = find(i);
    if (!componentMap.has(root)) {
      componentMap.set(root, nextComponent++);
    }
    components[i] = componentMap.get(root);
  }

  return { components, count: nextComponent };
}

function buildAtomHash(atoms, cellSize) {
  if (!Number.isFinite(cellSize) || cellSize <= 0) {
    return null;
  }

  const cells = new Map();
  for (let i = 0; i < atoms.length; i++) {
    const c = atoms[i].center;
    const ix = Math.floor(c[0] / cellSize);
    const iy = Math.floor(c[1] / cellSize);
    const iz = Math.floor(c[2] / cellSize);
    const key = `${ix},${iy},${iz}`;
    const bucket = cells.get(key);
    if (bucket) {
      bucket.push(i);
    } else {
      cells.set(key, [i]);
    }
  }

  return { cells, cellSize };
}

function queryAtomHash(atomHash, x, y, z) {
  if (!atomHash) return null;
  const { cells, cellSize } = atomHash;
  const ix = Math.floor(x / cellSize);
  const iy = Math.floor(y / cellSize);
  const iz = Math.floor(z / cellSize);
  const indices = [];

  for (let dx = -1; dx <= 1; dx++) {
    for (let dy = -1; dy <= 1; dy++) {
      for (let dz = -1; dz <= 1; dz++) {
        const key = `${ix + dx},${iy + dy},${iz + dz}`;
        const bucket = cells.get(key);
        if (bucket) {
          for (let i = 0; i < bucket.length; i++) {
            indices.push(bucket[i]);
          }
        }
      }
    }
  }

  return indices;
}

/**
 * Filter mesh to keep only components close to original atoms.
 */
function filterSESComponents(mesh, atoms, probeRadius, maxAtomRadius) {
  const { components, count } = findConnectedComponents(mesh.vertices, mesh.indices);

  if (count <= 1) return mesh;

  // Check one vertex from each component
  const componentValid = new Array(count).fill(false);
  const checked = new Array(count).fill(false);
  const threshold = probeRadius * 1.5;
  const maxCenterDistance = (maxAtomRadius ?? 0) + threshold;
  const atomHash = buildAtomHash(atoms, maxCenterDistance);

  for (let i = 0; i < mesh.vertices.length / 3; i++) {
    const comp = components[i];
    if (checked[comp]) continue;
    checked[comp] = true;

    const vx = mesh.vertices[i * 3];
    const vy = mesh.vertices[i * 3 + 1];
    const vz = mesh.vertices[i * 3 + 2];

    if (atomHash) {
      const candidates = queryAtomHash(atomHash, vx, vy, vz);
      if (!candidates || candidates.length === 0) continue;

      for (let c = 0; c < candidates.length; c++) {
        const atom = atoms[candidates[c]];
        const dx = vx - atom.center[0];
        const dy = vy - atom.center[1];
        const dz = vz - atom.center[2];
        const centerDist = Math.sqrt(dx * dx + dy * dy + dz * dz);

        if (centerDist > maxCenterDistance) continue;
        const dist = centerDist - atom.radius;

        if (dist < threshold) {
          componentValid[comp] = true;
          break;
        }
      }
    } else {
      for (const atom of atoms) {
        const dx = vx - atom.center[0];
        const dy = vy - atom.center[1];
        const dz = vz - atom.center[2];
        const centerDist = Math.sqrt(dx * dx + dy * dy + dz * dz);
        const dist = centerDist - atom.radius;

        if (dist < threshold) {
          componentValid[comp] = true;
          break;
        }
      }
    }
  }

  // Filter triangles
  const newIndices = [];
  for (let i = 0; i < mesh.indices.length; i += 3) {
    const comp = components[mesh.indices[i]];
    if (componentValid[comp]) {
      newIndices.push(mesh.indices[i], mesh.indices[i + 1], mesh.indices[i + 2]);
    }
  }

  // Reindex vertices
  const usedVertices = new Set(newIndices);
  const vertexMap = new Map();
  const newVertices = [];
  const newNormals = [];

  for (const oldIdx of usedVertices) {
    vertexMap.set(oldIdx, newVertices.length / 3);
    newVertices.push(
      mesh.vertices[oldIdx * 3],
      mesh.vertices[oldIdx * 3 + 1],
      mesh.vertices[oldIdx * 3 + 2]
    );
    newNormals.push(
      mesh.normals[oldIdx * 3],
      mesh.normals[oldIdx * 3 + 1],
      mesh.normals[oldIdx * 3 + 2]
    );
  }

  const remappedIndices = newIndices.map(i => vertexMap.get(i));

  return {
    vertices: new Float32Array(newVertices),
    normals: new Float32Array(newNormals),
    indices: new Uint32Array(remappedIndices)
  };
}

/**
 * Compute Solvent Excluded Surface (SES) for a molecule.
 *
 * @param {Array} atoms - Array of {center: [x,y,z], radius: number}
 * @param {Object} options - Options
 * @param {number} options.probeRadius - Probe sphere radius (default 1.4 Å for water)
 * @param {number} options.resolution - Grid resolution in Å (default 0.25)
 * @param {boolean} options.sas - If true, return solvent accessible surface (SAS)
 * @param {boolean} options.smoothNormals - If true, smooth normals using face averages
 * @returns {{vertices: Float32Array, normals: Float32Array, indices: Uint32Array}}
 */
export function computeSES(atoms, options = {}) {
  const probeRadius = options.probeRadius ?? 1.4;
  const resolution = options.resolution ?? 0.25;
  const returnSAS = options.sas ?? false;
  const smoothNormals = options.smoothNormals ?? false;

  if (atoms.length === 0) {
    return { vertices: new Float32Array(0), normals: new Float32Array(0), indices: new Uint32Array(0) };
  }

  // Compute bounds
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  let maxAtomRadius = 0;

  for (const atom of atoms) {
    minX = Math.min(minX, atom.center[0]);
    minY = Math.min(minY, atom.center[1]);
    minZ = Math.min(minZ, atom.center[2]);
    maxX = Math.max(maxX, atom.center[0]);
    maxY = Math.max(maxY, atom.center[1]);
    maxZ = Math.max(maxZ, atom.center[2]);
    if (atom.radius > maxAtomRadius) maxAtomRadius = atom.radius;
  }

  const bounds = { minX, minY, minZ, maxX, maxY, maxZ };
  const padding = 2 * probeRadius + maxAtomRadius + resolution;

  // Step 1: Create distance grid for expanded spheres (SAS)
  const grid = new DistanceGrid(bounds, resolution, { padding, maxIndexRange: 2 });

  for (const atom of atoms) {
    grid.addSphere(atom.center, atom.radius + probeRadius);
  }

  // Step 2: Extract SAS surface
  const sasMesh = marchingCubes(grid, 0, { smoothNormals });

  if (sasMesh.vertices.length === 0) {
    return sasMesh;
  }
  if (returnSAS) {
    return sasMesh;
  }

  // Step 3: Place probe spheres at SAS vertices and create new distance grid
  grid.clear();

  const sasVertexCount = sasMesh.vertices.length / 3;
  for (let i = 0; i < sasVertexCount; i++) {
    const cx = sasMesh.vertices[i * 3];
    const cy = sasMesh.vertices[i * 3 + 1];
    const cz = sasMesh.vertices[i * 3 + 2];
    grid.addSphere([cx, cy, cz], probeRadius);
  }

  // Step 4: Extract SES surface
  let sesMesh = marchingCubes(grid, 0, { smoothNormals });

  // Step 5: Filter out extra surface components
  sesMesh = filterSESComponents(sesMesh, atoms, probeRadius, maxAtomRadius);

  return sesMesh;
}

/**
 * Convert SES mesh to triangles for the raytracer.
 * @param {Object} mesh - The SES mesh from computeSES
 * @param {Array} color - RGB color [r, g, b]
 * @returns {{positions: Float32Array, indices: Uint32Array, normals: Float32Array, triColors: Float32Array}}
 */
export function sesToTriangles(mesh, color = [0.8, 0.8, 0.9]) {
  const triCount = mesh.indices.length / 3;
  const triColors = new Float32Array(triCount * 3);

  for (let i = 0; i < triCount; i++) {
    triColors[i * 3] = color[0];
    triColors[i * 3 + 1] = color[1];
    triColors[i * 3 + 2] = color[2];
  }

  return {
    positions: mesh.vertices,
    indices: mesh.indices,
    normals: mesh.normals,
    triColors
  };
}
