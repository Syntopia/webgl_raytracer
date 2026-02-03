/**
 * GPU-accelerated Solvent Excluded Surface (SES) computation using WebGL2.
 *
 * Strategy:
 * 1. Compute distance field on GPU using fragment shaders with MIN blending
 * 2. Read back the distance field to CPU
 * 3. Run marching cubes on CPU (with the optimized JS implementation)
 *
 * This provides significant speedup for the distance field computation,
 * which is the main bottleneck (O(atoms * grid_volume)).
 */

// Marching cubes tables (same as in surface.js)
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

let glContext = null;
let distanceProgram = null;
let quadVAO = null;
let instanceBuffer = null;
let gridFramebuffer = null;
let gridTexture = null;
let gridTexWidth = 0;
let gridTexHeight = 0;

const DISTANCE_VS = `#version 300 es
in vec2 a_position;
in vec4 a_rect;         // pixel-space rect (x0, y0, x1, y1) in grid coordinates
in vec2 a_sliceRadius;  // iz, sphere radius
in vec3 a_center;

uniform ivec3 u_gridDim;
uniform ivec2 u_texDim;
uniform int u_slicesPerRow;

out vec3 v_center;
out float v_radius;
out float v_slice;

void main() {
  float ix = mix(a_rect.x, a_rect.z, a_position.x);
  float iy = mix(a_rect.y, a_rect.w, a_position.y);
  float iz = a_sliceRadius.x;

  float tileX = mod(iz, float(u_slicesPerRow));
  float tileY = floor(iz / float(u_slicesPerRow));

  float px = tileX * float(u_gridDim.x) + ix;
  float py = tileY * float(u_gridDim.y) + iy;

  vec2 ndc = vec2(
    (px / float(u_texDim.x)) * 2.0 - 1.0,
    (py / float(u_texDim.y)) * 2.0 - 1.0
  );
  gl_Position = vec4(ndc, 0.0, 1.0);

  v_center = a_center;
  v_radius = a_sliceRadius.y;
  v_slice = iz;
}
`;

const DISTANCE_FS = `#version 300 es
precision highp float;
precision highp int;

uniform vec3 u_gridMin;
uniform float u_resolution;
uniform ivec3 u_gridDim;
uniform ivec2 u_texDim;
uniform int u_slicesPerRow;
uniform float u_maxDist;

in vec3 v_center;
in float v_radius;
in float v_slice;
out vec4 fragColor;

// Convert pixel coordinate to grid index
ivec3 texCoordToGrid(ivec2 px) {
  int tileX = px.x / u_gridDim.x;
  int tileY = px.y / u_gridDim.y;
  int iz = tileY * u_slicesPerRow + tileX;
  int ix = px.x - tileX * u_gridDim.x;
  int iy = px.y - tileY * u_gridDim.y;
  return ivec3(ix, iy, iz);
}

vec3 gridToWorld(ivec3 idx) {
  return u_gridMin + vec3(idx) * u_resolution;
}

void main() {
  ivec2 px = ivec2(gl_FragCoord.xy);
  ivec3 gridIdx = texCoordToGrid(px);

  if (gridIdx.x < 0 || gridIdx.y < 0 || gridIdx.z < 0 ||
      gridIdx.x >= u_gridDim.x || gridIdx.y >= u_gridDim.y || gridIdx.z >= u_gridDim.z) {
    fragColor = vec4(u_maxDist, 0.0, 0.0, 1.0);
    return;
  }

  vec3 worldPos = gridToWorld(gridIdx);
  float dist = length(worldPos - v_center) - v_radius;
  dist = clamp(dist, -u_maxDist, u_maxDist);
  fragColor = vec4(dist, 0.0, 0.0, 1.0);
}
`;

function initWebGLSurface() {
  if (glContext) return glContext;

  const canvas = document.createElement('canvas');
  canvas.width = 1;
  canvas.height = 1;

  glContext = canvas.getContext('webgl2', {
    antialias: false,
    depth: false,
    stencil: false,
    preserveDrawingBuffer: false
  });

  if (!glContext) {
    throw new Error('WebGL2 not supported for surface computation');
  }

  const gl = glContext;

  // Check for required extensions
  const floatExt = gl.getExtension('EXT_color_buffer_float');
  if (!floatExt) {
    throw new Error('EXT_color_buffer_float not supported');
  }

  // Compile shaders
  const vs = gl.createShader(gl.VERTEX_SHADER);
  gl.shaderSource(vs, DISTANCE_VS);
  gl.compileShader(vs);
  if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
    throw new Error('Vertex shader error: ' + gl.getShaderInfoLog(vs));
  }

  const fs = gl.createShader(gl.FRAGMENT_SHADER);
  gl.shaderSource(fs, DISTANCE_FS);
  gl.compileShader(fs);
  if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
    throw new Error('Fragment shader error: ' + gl.getShaderInfoLog(fs));
  }

  distanceProgram = gl.createProgram();
  gl.attachShader(distanceProgram, vs);
  gl.attachShader(distanceProgram, fs);
  gl.linkProgram(distanceProgram);
  if (!gl.getProgramParameter(distanceProgram, gl.LINK_STATUS)) {
    throw new Error('Program link error: ' + gl.getProgramInfoLog(distanceProgram));
  }

  // Create quad VAO
  quadVAO = gl.createVertexArray();
  gl.bindVertexArray(quadVAO);

  const quadBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
    0, 0,
    1, 0,
    0, 1,
    1, 1
  ]), gl.STATIC_DRAW);

  const posLoc = gl.getAttribLocation(distanceProgram, 'a_position');
  gl.enableVertexAttribArray(posLoc);
  gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
  gl.vertexAttribDivisor(posLoc, 0);

  instanceBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, instanceBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, 0, gl.DYNAMIC_DRAW);

  const rectLoc = gl.getAttribLocation(distanceProgram, 'a_rect');
  const sliceLoc = gl.getAttribLocation(distanceProgram, 'a_sliceRadius');
  const centerLoc = gl.getAttribLocation(distanceProgram, 'a_center');
  const stride = 9 * 4;
  gl.enableVertexAttribArray(rectLoc);
  gl.vertexAttribPointer(rectLoc, 4, gl.FLOAT, false, stride, 0);
  gl.vertexAttribDivisor(rectLoc, 1);

  gl.enableVertexAttribArray(sliceLoc);
  gl.vertexAttribPointer(sliceLoc, 2, gl.FLOAT, false, stride, 4 * 4);
  gl.vertexAttribDivisor(sliceLoc, 1);

  gl.enableVertexAttribArray(centerLoc);
  gl.vertexAttribPointer(centerLoc, 3, gl.FLOAT, false, stride, 6 * 4);
  gl.vertexAttribDivisor(centerLoc, 1);

  gl.bindVertexArray(null);

  return gl;
}

function computeDistanceFieldGPU(atoms, bounds, resolution, probeRadius, maxDist, paddingOverride = null) {
  const gl = initWebGLSurface();

  const padding = paddingOverride ?? (2 * probeRadius + resolution);
  const gridMin = [
    bounds.minX - padding,
    bounds.minY - padding,
    bounds.minZ - padding
  ];
  const gridMax = [
    bounds.maxX + padding,
    bounds.maxY + padding,
    bounds.maxZ + padding
  ];
  const gridSize = [
    gridMax[0] - gridMin[0],
    gridMax[1] - gridMin[1],
    gridMax[2] - gridMin[2]
  ];

  const nx = Math.ceil(gridSize[0] / resolution) + 1;
  const ny = Math.ceil(gridSize[1] / resolution) + 1;
  const nz = Math.ceil(gridSize[2] / resolution) + 1;

  // Check max texture size
  const maxTexSize = gl.getParameter(gl.MAX_TEXTURE_SIZE);

  // Calculate tiled 2D layout that fits within texture limits
  // Tile Z slices in a grid pattern
  const slicesPerRow = Math.min(nz, Math.floor(maxTexSize / nx));
  if (slicesPerRow === 0) {
    throw new Error(`Grid X dimension (${nx}) exceeds max texture size (${maxTexSize})`);
  }
  const numRows = Math.ceil(nz / slicesPerRow);

  const texWidth = slicesPerRow * nx;
  const texHeight = numRows * ny;

  if (texWidth > maxTexSize || texHeight > maxTexSize) {
    throw new Error(`Grid too large for GPU: ${texWidth}x${texHeight} exceeds ${maxTexSize}. Grid: ${nx}x${ny}x${nz}`);
  }

  // Resize canvas to match texture
  gl.canvas.width = texWidth;
  gl.canvas.height = texHeight;
  gl.viewport(0, 0, texWidth, texHeight);

  if (!gridTexture || gridTexWidth !== texWidth || gridTexHeight !== texHeight) {
    if (gridTexture) {
      gl.deleteTexture(gridTexture);
    }
    if (gridFramebuffer) {
      gl.deleteFramebuffer(gridFramebuffer);
    }

    gridTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, gridTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, texWidth, texHeight, 0, gl.RED, gl.FLOAT, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    gridFramebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, gridFramebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, gridTexture, 0);

    gridTexWidth = texWidth;
    gridTexHeight = texHeight;
  } else {
    gl.bindTexture(gl.TEXTURE_2D, gridTexture);
    gl.bindFramebuffer(gl.FRAMEBUFFER, gridFramebuffer);
  }

  // Setup program
  gl.useProgram(distanceProgram);
  gl.bindVertexArray(quadVAO);

  // Set static uniforms
  gl.uniform3fv(gl.getUniformLocation(distanceProgram, 'u_gridMin'), gridMin);
  gl.uniform1f(gl.getUniformLocation(distanceProgram, 'u_resolution'), resolution);
  gl.uniform3iv(gl.getUniformLocation(distanceProgram, 'u_gridDim'), [nx, ny, nz]);
  gl.uniform2iv(gl.getUniformLocation(distanceProgram, 'u_texDim'), [texWidth, texHeight]);
  gl.uniform1i(gl.getUniformLocation(distanceProgram, 'u_slicesPerRow'), slicesPerRow);
  gl.uniform1f(gl.getUniformLocation(distanceProgram, 'u_maxDist'), maxDist);

  // Build per-slice instances for each sphere
  const instanceData = [];
  for (let i = 0; i < atoms.length; i++) {
    const atom = atoms[i];
    const sphereRadius = atom.radius + probeRadius;
    const influence = sphereRadius + maxDist;

    const ix0 = Math.floor((atom.center[0] - influence - gridMin[0]) / resolution);
    const ix1 = Math.floor((atom.center[0] + influence - gridMin[0]) / resolution);
    const iy0 = Math.floor((atom.center[1] - influence - gridMin[1]) / resolution);
    const iy1 = Math.floor((atom.center[1] + influence - gridMin[1]) / resolution);
    const iz0 = Math.floor((atom.center[2] - influence - gridMin[2]) / resolution);
    const iz1 = Math.floor((atom.center[2] + influence - gridMin[2]) / resolution);

    const zStart = Math.max(0, iz0);
    const zEnd = Math.min(nz - 1, iz1);

    for (let iz = zStart; iz <= zEnd; iz++) {
      const z = gridMin[2] + iz * resolution;
      const dz = z - atom.center[2];
      const sliceRadiusSq = influence * influence - dz * dz;
      if (sliceRadiusSq <= 0) {
        continue;
      }
      const sliceRadius = Math.sqrt(sliceRadiusSq);
      const sx0 = Math.floor((atom.center[0] - sliceRadius - gridMin[0]) / resolution);
      const sx1 = Math.floor((atom.center[0] + sliceRadius - gridMin[0]) / resolution);
      const sy0 = Math.floor((atom.center[1] - sliceRadius - gridMin[1]) / resolution);
      const sy1 = Math.floor((atom.center[1] + sliceRadius - gridMin[1]) / resolution);

      const fx0 = Math.max(0, Math.min(nx - 1, sx0));
      const fx1 = Math.max(0, Math.min(nx - 1, sx1));
      const fy0 = Math.max(0, Math.min(ny - 1, sy0));
      const fy1 = Math.max(0, Math.min(ny - 1, sy1));

      if (fx1 < fx0 || fy1 < fy0) {
        continue;
      }

      const rx0 = fx0;
      const ry0 = fy0;
      const rx1 = Math.min(nx, fx1 + 1);
      const ry1 = Math.min(ny, fy1 + 1);

      instanceData.push(
        rx0, ry0, rx1, ry1,
        iz, sphereRadius,
        atom.center[0], atom.center[1], atom.center[2]
      );
    }
  }

  const instanceCount = instanceData.length / 9;

  // Clear texture to max distance
  gl.bindFramebuffer(gl.FRAMEBUFFER, gridFramebuffer);
  gl.clearColor(maxDist, 0.0, 0.0, 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT);

  gl.enable(gl.BLEND);
  gl.blendEquation(gl.MIN);
  gl.blendFunc(gl.ONE, gl.ONE);

  gl.bindBuffer(gl.ARRAY_BUFFER, instanceBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(instanceData), gl.DYNAMIC_DRAW);

  if (instanceCount > 0) {
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, instanceCount);
  }

  gl.disable(gl.BLEND);

  // Read back final result
  gl.bindFramebuffer(gl.FRAMEBUFFER, gridFramebuffer);
  const result = new Float32Array(texWidth * texHeight);
  gl.readPixels(0, 0, texWidth, texHeight, gl.RED, gl.FLOAT, result);

  // Convert tiled 2D texture layout back to 3D grid
  const gridData = new Float32Array(nx * ny * nz);
  for (let iz = 0; iz < nz; iz++) {
    // Which tile contains this Z slice?
    const tileX = iz % slicesPerRow;
    const tileY = Math.floor(iz / slicesPerRow);

    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        const texX = tileX * nx + ix;
        const texY = tileY * ny + iy;
        const texIdx = texY * texWidth + texX;
        const gridIdx = ix + iy * nx + iz * nx * ny;
        gridData[gridIdx] = result[texIdx];
      }
    }
  }

  return {
    data: gridData,
    nx, ny, nz,
    min: gridMin,
    resolution,
    maxDist
  };
}

// CPU marching cubes on GPU-computed distance field
function marchingCubesGPU(grid, isovalue, options = {}) {
  const { nx, ny, nz, min, resolution, maxDist, data } = grid;
  const smooth = options.smoothNormals ?? false;

  const vertices = [];
  const normals = [];
  const indices = [];
  const edgeVertexCache = new Map();

  function getVal(ix, iy, iz) {
    if (ix < 0 || ix >= nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz) {
      return maxDist;
    }
    return data[ix + iy * nx + iz * nx * ny];
  }

  function worldPos(ix, iy, iz) {
    return [
      min[0] + ix * resolution,
      min[1] + iy * resolution,
      min[2] + iz * resolution
    ];
  }

  function sampleTrilinear(wx, wy, wz) {
    const fx = (wx - min[0]) / resolution;
    const fy = (wy - min[1]) / resolution;
    const fz = (wz - min[2]) / resolution;

    const ix = Math.floor(fx);
    const iy = Math.floor(fy);
    const iz = Math.floor(fz);

    const tx = fx - ix;
    const ty = fy - iy;
    const tz = fz - iz;

    const clamp = (v) => (v === Infinity) ? maxDist : ((v === -Infinity) ? -maxDist : v);

    const c000 = clamp(getVal(ix, iy, iz));
    const c100 = clamp(getVal(ix + 1, iy, iz));
    const c010 = clamp(getVal(ix, iy + 1, iz));
    const c110 = clamp(getVal(ix + 1, iy + 1, iz));
    const c001 = clamp(getVal(ix, iy, iz + 1));
    const c101 = clamp(getVal(ix + 1, iy, iz + 1));
    const c011 = clamp(getVal(ix, iy + 1, iz + 1));
    const c111 = clamp(getVal(ix + 1, iy + 1, iz + 1));

    const c00 = c000 * (1 - tx) + c100 * tx;
    const c10 = c010 * (1 - tx) + c110 * tx;
    const c01 = c001 * (1 - tx) + c101 * tx;
    const c11 = c011 * (1 - tx) + c111 * tx;

    const c0 = c00 * (1 - ty) + c10 * ty;
    const c1 = c01 * (1 - ty) + c11 * ty;

    return c0 * (1 - tz) + c1 * tz;
  }

  function computeNormal(pos) {
    const h = resolution;
    const gx = sampleTrilinear(pos[0] + h, pos[1], pos[2]) - sampleTrilinear(pos[0] - h, pos[1], pos[2]);
    const gy = sampleTrilinear(pos[0], pos[1] + h, pos[2]) - sampleTrilinear(pos[0], pos[1] - h, pos[2]);
    const gz = sampleTrilinear(pos[0], pos[1], pos[2] + h) - sampleTrilinear(pos[0], pos[1], pos[2] - h);

    const len = Math.sqrt(gx * gx + gy * gy + gz * gz);
    if (len > 0.0001) {
      return [gx / len, gy / len, gz / len];
    }
    return [0, 1, 0];
  }

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

  function addVertex(pos) {
    const idx = vertices.length / 3;
    vertices.push(pos[0], pos[1], pos[2]);
    const n = computeNormal(pos);
    normals.push(n[0], n[1], n[2]);
    return idx;
  }

  const edgeCorners = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7]
  ];

  const cornerOffsets = [
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
  ];

  function getEdgeCacheKey(ix, iy, iz, edgeIdx) {
    const [c0, c1] = edgeCorners[edgeIdx];
    const o0 = cornerOffsets[c0];
    const o1 = cornerOffsets[c1];
    if (o0[0] !== o1[0]) {
      return `x,${ix + Math.min(o0[0], o1[0])},${iy + o0[1]},${iz + o0[2]}`;
    } else if (o0[1] !== o1[1]) {
      return `y,${ix + o0[0]},${iy + Math.min(o0[1], o1[1])},${iz + o0[2]}`;
    } else {
      return `z,${ix + o0[0]},${iy + o0[1]},${iz + Math.min(o0[2], o1[2])}`;
    }
  }

  for (let iz = 0; iz < nz - 1; iz++) {
    for (let iy = 0; iy < ny - 1; iy++) {
      for (let ix = 0; ix < nx - 1; ix++) {
        const v = new Array(8);
        const corners = new Array(8);
        for (let i = 0; i < 8; i++) {
          const o = cornerOffsets[i];
          v[i] = getVal(ix + o[0], iy + o[1], iz + o[2]);
          corners[i] = worldPos(ix + o[0], iy + o[1], iz + o[2]);
        }

        let cubeIndex = 0;
        for (let i = 0; i < 8; i++) {
          if (v[i] < isovalue) cubeIndex |= (1 << i);
        }

        if (EDGE_TABLE[cubeIndex] === 0) continue;

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

  const smoothedNormals = smooth ? smoothNormalsGPU(vertices, normals, indices) : new Float32Array(normals);

  return {
    vertices: new Float32Array(vertices),
    normals: smoothedNormals,
    indices: new Uint32Array(indices)
  };
}

function smoothNormalsGPU(vertices, perVertexNormals, indices) {
  const vertexCount = vertices.length / 3;
  const accumulated = new Float32Array(vertexCount * 3);

  for (let i = 0; i < indices.length; i += 3) {
    const i0 = indices[i];
    const i1 = indices[i + 1];
    const i2 = indices[i + 2];

    const v0x = vertices[i0 * 3], v0y = vertices[i0 * 3 + 1], v0z = vertices[i0 * 3 + 2];
    const v1x = vertices[i1 * 3], v1y = vertices[i1 * 3 + 1], v1z = vertices[i1 * 3 + 2];
    const v2x = vertices[i2 * 3], v2y = vertices[i2 * 3 + 1], v2z = vertices[i2 * 3 + 2];

    const e1x = v1x - v0x, e1y = v1y - v0y, e1z = v1z - v0z;
    const e2x = v2x - v0x, e2y = v2y - v0y, e2z = v2z - v0z;

    const nx = e1y * e2z - e1z * e2y;
    const ny = e1z * e2x - e1x * e2z;
    const nz = e1x * e2y - e1y * e2x;

    accumulated[i0 * 3] += nx; accumulated[i0 * 3 + 1] += ny; accumulated[i0 * 3 + 2] += nz;
    accumulated[i1 * 3] += nx; accumulated[i1 * 3 + 1] += ny; accumulated[i1 * 3 + 2] += nz;
    accumulated[i2 * 3] += nx; accumulated[i2 * 3 + 1] += ny; accumulated[i2 * 3 + 2] += nz;
  }

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
      result[i * 3] = perVertexNormals[i * 3];
      result[i * 3 + 1] = perVertexNormals[i * 3 + 1];
      result[i * 3 + 2] = perVertexNormals[i * 3 + 2];
    }
  }

  return result;
}

function findConnectedComponentsGPU(vertices, indices) {
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

  for (let i = 0; i < indices.length; i += 3) {
    union(indices[i], indices[i + 1]);
    union(indices[i + 1], indices[i + 2]);
  }

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

function filterSESComponentsGPU(mesh, atoms, probeRadius, maxAtomRadius) {
  const { components, count } = findConnectedComponentsGPU(mesh.vertices, mesh.indices);

  if (count <= 1) return mesh;

  const componentValid = new Array(count).fill(false);
  const checked = new Array(count).fill(false);
  const threshold = probeRadius * 1.5;

  for (let i = 0; i < mesh.vertices.length / 3; i++) {
    const comp = components[i];
    if (checked[comp]) continue;
    checked[comp] = true;

    const vx = mesh.vertices[i * 3];
    const vy = mesh.vertices[i * 3 + 1];
    const vz = mesh.vertices[i * 3 + 2];

    for (const atom of atoms) {
      const dx = vx - atom.center[0];
      const dy = vy - atom.center[1];
      const dz = vz - atom.center[2];
      const distSq = dx * dx + dy * dy + dz * dz;
      const thresholdForAtom = atom.radius + threshold;

      if (distSq < thresholdForAtom * thresholdForAtom) {
        componentValid[comp] = true;
        break;
      }
    }
  }

  const newIndices = [];
  for (let i = 0; i < mesh.indices.length; i += 3) {
    const comp = components[mesh.indices[i]];
    if (componentValid[comp]) {
      newIndices.push(mesh.indices[i], mesh.indices[i + 1], mesh.indices[i + 2]);
    }
  }

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
 * Compute SES using WebGL GPU acceleration.
 */
export function computeSESWebGL(atoms, options = {}) {
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
  const maxDist = 2 * resolution;
  const padding = 2 * probeRadius + maxAtomRadius + resolution;

  // Step 1: Compute SAS distance field on GPU
  const sasGrid = computeDistanceFieldGPU(atoms, bounds, resolution, probeRadius, maxDist, padding);

  // Step 2: Extract SAS mesh
  const sasMesh = marchingCubesGPU(sasGrid, 0, { smoothNormals });

  if (sasMesh.vertices.length === 0) {
    return sasMesh;
  }
  if (returnSAS) {
    return sasMesh;
  }

  // Step 3: Build SES distance field from SAS vertices
  const sasVertexCount = sasMesh.vertices.length / 3;
  const sasAtoms = [];

  // Deduplicate vertices spatially
  const dedupCellSize = resolution * 0.5;
  const invCell = 1.0 / dedupCellSize;
  const cells = new Map();

  for (let i = 0; i < sasVertexCount; i++) {
    const x = sasMesh.vertices[i * 3];
    const y = sasMesh.vertices[i * 3 + 1];
    const z = sasMesh.vertices[i * 3 + 2];

    const cx = Math.floor(x * invCell);
    const cy = Math.floor(y * invCell);
    const cz = Math.floor(z * invCell);
    const key = `${cx},${cy},${cz}`;

    if (!cells.has(key)) {
      cells.set(key, { center: [x, y, z], radius: 0 });
      sasAtoms.push({ center: [x, y, z], radius: 0 });
    }
  }

  // Compute SES distance field on GPU (probe spheres at SAS vertices)
  const sesGrid = computeDistanceFieldGPU(sasAtoms, bounds, resolution, probeRadius, maxDist, padding);

  // Step 4: Extract SES mesh
  let sesMesh = marchingCubesGPU(sesGrid, 0, { smoothNormals });

  // Step 5: Filter components
  sesMesh = filterSESComponentsGPU(sesMesh, atoms, probeRadius, maxAtomRadius);

  // Flip normals
  for (let i = 0; i < sesMesh.normals.length; i++) {
    sesMesh.normals[i] = -sesMesh.normals[i];
  }

  return sesMesh;
}

/**
 * Check if WebGL surface computation is available.
 */
export function webglSurfaceAvailable() {
  try {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2');
    if (!gl) return false;
    const floatExt = gl.getExtension('EXT_color_buffer_float');
    return !!floatExt;
  } catch {
    return false;
  }
}
