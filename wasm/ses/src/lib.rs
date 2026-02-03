mod tables;

use js_sys::{Float32Array, Object, Reflect, Uint32Array};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

use tables::{EDGE_TABLE, TRI_TABLE};

/// Fast approximate square root using the famous "fast inverse sqrt" trick
/// followed by multiplication. Accuracy is within ~0.2% which is sufficient
/// for distance field calculations.
#[inline(always)]
fn fast_sqrt(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    // One iteration of Newton-Raphson on inverse sqrt, then multiply
    let half = 0.5 * x;
    let i = x.to_bits();
    let i = 0x5f375a86 - (i >> 1); // Magic constant for f32
    let y = f32::from_bits(i);
    let y = y * (1.5 - half * y * y); // One Newton-Raphson iteration
    x * y
}

#[derive(Clone, Copy)]
struct Atom {
    center: [f32; 3],
    radius: f32,
}

struct DistanceGrid {
    resolution: f32,
    max_dist: f32,
    default_value: f32,
    min: [f32; 3],
    nx: usize,
    ny: usize,
    nz: usize,
    data: Vec<f32>,
}

impl DistanceGrid {
    fn new(bounds_min: [f32; 3], bounds_max: [f32; 3], resolution: f32, padding: f32, max_index_range: f32) -> Result<Self, String> {
        let max_dist = max_index_range * resolution;
        let default_value = max_dist;
        let min = [
            bounds_min[0] - padding,
            bounds_min[1] - padding,
            bounds_min[2] - padding,
        ];
        let _max = [
            bounds_max[0] + padding,
            bounds_max[1] + padding,
            bounds_max[2] + padding,
        ];
        let size = [
            _max[0] - min[0],
            _max[1] - min[1],
            _max[2] - min[2],
        ];
        let nx = (size[0] / resolution).ceil() as usize + 1;
        let ny = (size[1] / resolution).ceil() as usize + 1;
        let nz = (size[2] / resolution).ceil() as usize + 1;
        let total = nx
            .checked_mul(ny)
            .and_then(|v| v.checked_mul(nz))
            .ok_or_else(|| "Grid dimensions too large".to_string())?;
        let data = vec![default_value; total];
        Ok(Self {
            resolution,
            max_dist,
            default_value,
            min,
            nx,
            ny,
            nz,
            data,
        })
    }

    #[inline]
    fn index(&self, ix: usize, iy: usize, iz: usize) -> usize {
        ix + iy * self.nx + iz * self.nx * self.ny
    }

    #[inline]
    fn get(&self, ix: i32, iy: i32, iz: i32) -> f32 {
        if ix < 0 || iy < 0 || iz < 0 {
            return self.default_value;
        }
        let (ix, iy, iz) = (ix as usize, iy as usize, iz as usize);
        if ix >= self.nx || iy >= self.ny || iz >= self.nz {
            return self.default_value;
        }
        self.data[self.index(ix, iy, iz)]
    }

    #[inline]
    fn world_pos(&self, ix: i32, iy: i32, iz: i32) -> [f32; 3] {
        [
            self.min[0] + ix as f32 * self.resolution,
            self.min[1] + iy as f32 * self.resolution,
            self.min[2] + iz as f32 * self.resolution,
        ]
    }

    #[inline]
    fn grid_indices(&self, x: f32, y: f32, z: f32) -> (i32, i32, i32) {
        (
            ((x - self.min[0]) / self.resolution).floor() as i32,
            ((y - self.min[1]) / self.resolution).floor() as i32,
            ((z - self.min[2]) / self.resolution).floor() as i32,
        )
    }

    fn add_sphere(&mut self, center: [f32; 3], radius: f32) {
        let margin = radius + self.max_dist;
        let (gx0, gy0, gz0) = self.grid_indices(center[0] - margin, center[1] - margin, center[2] - margin);
        let (gx1, gy1, gz1) = self.grid_indices(center[0] + margin, center[1] + margin, center[2] + margin);

        let ix0 = gx0.max(0) as usize;
        let iy0 = gy0.max(0) as usize;
        let iz0 = gz0.max(0) as usize;
        let ix1 = gx1.min(self.nx as i32 - 1).max(0) as usize;
        let iy1 = gy1.min(self.ny as i32 - 1).max(0) as usize;
        let iz1 = gz1.min(self.nz as i32 - 1).max(0) as usize;

        // Pre-compute squared thresholds for early rejection
        let max_dist_plus_r = self.max_dist + radius;
        let max_dist_plus_r_sq = max_dist_plus_r * max_dist_plus_r;
        let neg_max_dist_plus_r = radius - self.max_dist;
        let neg_threshold_sq = if neg_max_dist_plus_r > 0.0 {
            neg_max_dist_plus_r * neg_max_dist_plus_r
        } else {
            0.0
        };

        let stride_y = self.nx;
        let stride_z = self.nx * self.ny;

        for iz in iz0..=iz1 {
            let wz = self.min[2] + iz as f32 * self.resolution;
            let dz = wz - center[2];
            let dz2 = dz * dz;
            // Early Z rejection: if dz² alone exceeds threshold, skip entire Z-slice
            if dz2 > max_dist_plus_r_sq {
                continue;
            }
            let base_z = iz * stride_z;

            for iy in iy0..=iy1 {
                let wy = self.min[1] + iy as f32 * self.resolution;
                let dy = wy - center[1];
                let dy2 = dy * dy;
                let dyz2 = dy2 + dz2;
                // Early Y rejection: if dy² + dz² alone exceeds threshold, skip entire row
                if dyz2 > max_dist_plus_r_sq {
                    continue;
                }
                let base_yz = base_z + iy * stride_y;

                for ix in ix0..=ix1 {
                    let wx = self.min[0] + ix as f32 * self.resolution;
                    let dx = wx - center[0];
                    let dist_sq = dx * dx + dyz2;

                    // Early rejection: squared distance outside influence range
                    if dist_sq > max_dist_plus_r_sq {
                        continue;
                    }

                    let idx = base_yz + ix;
                    let current = unsafe { *self.data.get_unchecked(idx) };

                    // Skip if this cell is already deeply inside another sphere
                    // and our sphere can't possibly improve it
                    if current <= -self.max_dist && dist_sq >= neg_threshold_sq {
                        continue;
                    }

                    // Only compute sqrt when we might update the cell
                    let dist_raw = fast_sqrt(dist_sq) - radius;
                    let dist = dist_raw.clamp(-self.max_dist, self.max_dist);

                    if dist < current {
                        unsafe { *self.data.get_unchecked_mut(idx) = dist; }
                    }
                }
            }
        }
    }

    fn clear(&mut self) {
        self.data.fill(self.default_value);
    }
}

struct Mesh {
    vertices: Vec<f32>,
    normals: Vec<f32>,
    indices: Vec<u32>,
}

fn interpolate_vertex(p1: [f32; 3], p2: [f32; 3], v1: f32, v2: f32, isovalue: f32) -> [f32; 3] {
    if (isovalue - v1).abs() < 1e-5 {
        return p1;
    }
    if (isovalue - v2).abs() < 1e-5 {
        return p2;
    }
    if (v1 - v2).abs() < 1e-5 {
        return p1;
    }
    let t = (isovalue - v1) / (v2 - v1);
    [
        p1[0] + t * (p2[0] - p1[0]),
        p1[1] + t * (p2[1] - p1[1]),
        p1[2] + t * (p2[2] - p1[2]),
    ]
}

/// Trilinear interpolation of distance field for smooth normal computation.
#[inline]
fn sample_trilinear(grid: &DistanceGrid, wx: f32, wy: f32, wz: f32) -> f32 {
    let fx = (wx - grid.min[0]) / grid.resolution;
    let fy = (wy - grid.min[1]) / grid.resolution;
    let fz = (wz - grid.min[2]) / grid.resolution;

    let ix = fx.floor() as i32;
    let iy = fy.floor() as i32;
    let iz = fz.floor() as i32;

    let tx = fx - ix as f32;
    let ty = fy - iy as f32;
    let tz = fz - iz as f32;

    let v000 = grid.get(ix, iy, iz);
    let v100 = grid.get(ix + 1, iy, iz);
    let v010 = grid.get(ix, iy + 1, iz);
    let v110 = grid.get(ix + 1, iy + 1, iz);
    let v001 = grid.get(ix, iy, iz + 1);
    let v101 = grid.get(ix + 1, iy, iz + 1);
    let v011 = grid.get(ix, iy + 1, iz + 1);
    let v111 = grid.get(ix + 1, iy + 1, iz + 1);

    let clamp = |v: f32| {
        if v.is_infinite() {
            if v.is_sign_negative() { -grid.max_dist } else { grid.max_dist }
        } else {
            v
        }
    };
    let c000 = clamp(v000);
    let c100 = clamp(v100);
    let c010 = clamp(v010);
    let c110 = clamp(v110);
    let c001 = clamp(v001);
    let c101 = clamp(v101);
    let c011 = clamp(v011);
    let c111 = clamp(v111);

    let c00 = c000 * (1.0 - tx) + c100 * tx;
    let c10 = c010 * (1.0 - tx) + c110 * tx;
    let c01 = c001 * (1.0 - tx) + c101 * tx;
    let c11 = c011 * (1.0 - tx) + c111 * tx;

    let c0 = c00 * (1.0 - ty) + c10 * ty;
    let c1 = c01 * (1.0 - ty) + c11 * ty;

    c0 * (1.0 - tz) + c1 * tz
}

/// Compute normal using direct grid cell lookups (faster but lower quality).
/// Uses central difference on the 6 neighboring grid cells.
#[allow(dead_code)]
#[inline]
fn compute_normal_fast(grid: &DistanceGrid, ix: i32, iy: i32, iz: i32) -> [f32; 3] {
    // Central difference using direct grid samples (much faster than trilinear)
    let gx = grid.get(ix + 1, iy, iz) - grid.get(ix - 1, iy, iz);
    let gy = grid.get(ix, iy + 1, iz) - grid.get(ix, iy - 1, iz);
    let gz = grid.get(ix, iy, iz + 1) - grid.get(ix, iy, iz - 1);
    let len_sq = gx * gx + gy * gy + gz * gz;
    if len_sq > 1e-8 {
        let inv_len = fast_inv_sqrt(len_sq);
        [gx * inv_len, gy * inv_len, gz * inv_len]
    } else {
        [0.0, 1.0, 0.0]
    }
}

/// Compute normal using trilinear interpolation for smooth gradients.
#[inline]
fn compute_normal(grid: &DistanceGrid, pos: [f32; 3]) -> [f32; 3] {
    let h = grid.resolution;
    let gx = sample_trilinear(grid, pos[0] + h, pos[1], pos[2]) - sample_trilinear(grid, pos[0] - h, pos[1], pos[2]);
    let gy = sample_trilinear(grid, pos[0], pos[1] + h, pos[2]) - sample_trilinear(grid, pos[0], pos[1] - h, pos[2]);
    let gz = sample_trilinear(grid, pos[0], pos[1], pos[2] + h) - sample_trilinear(grid, pos[0], pos[1], pos[2] - h);
    let len_sq = gx * gx + gy * gy + gz * gz;
    if len_sq > 1e-8 {
        let inv_len = fast_inv_sqrt(len_sq);
        [gx * inv_len, gy * inv_len, gz * inv_len]
    } else {
        [0.0, 1.0, 0.0]
    }
}

/// Fast inverse square root (Quake III style)
#[inline(always)]
fn fast_inv_sqrt(x: f32) -> f32 {
    let half = 0.5 * x;
    let i = x.to_bits();
    let i = 0x5f375a86 - (i >> 1);
    let y = f32::from_bits(i);
    y * (1.5 - half * y * y)
}

/// Optimized edge cache using a flat array with spatial indexing.
/// Much faster than HashMap for dense grid traversal.
struct EdgeCache {
    // Store vertex index for each edge. Layout: [axis][z][y][x]
    // axis 0 = X edges, axis 1 = Y edges, axis 2 = Z edges
    data: Vec<u32>,
    nx: usize,
    ny: usize,
    nz: usize,
}

impl EdgeCache {
    const EMPTY: u32 = u32::MAX;

    fn new(nx: usize, ny: usize, nz: usize) -> Self {
        // Each axis needs (nx * ny * nz) slots
        let size = 3 * nx * ny * nz;
        Self {
            data: vec![Self::EMPTY; size],
            nx, ny, nz,
        }
    }

    #[inline(always)]
    fn index(&self, axis: u8, ix: i32, iy: i32, iz: i32) -> usize {
        let base = (axis as usize) * self.nx * self.ny * self.nz;
        base + (iz as usize) * self.nx * self.ny + (iy as usize) * self.nx + (ix as usize)
    }

    #[inline(always)]
    fn get(&self, axis: u8, ix: i32, iy: i32, iz: i32) -> Option<u32> {
        let idx = self.index(axis, ix, iy, iz);
        let val = unsafe { *self.data.get_unchecked(idx) };
        if val == Self::EMPTY { None } else { Some(val) }
    }

    #[inline(always)]
    fn set(&mut self, axis: u8, ix: i32, iy: i32, iz: i32, vertex_idx: u32) {
        let idx = self.index(axis, ix, iy, iz);
        unsafe { *self.data.get_unchecked_mut(idx) = vertex_idx; }
    }
}

fn marching_cubes(grid: &DistanceGrid, isovalue: f32, smooth_normals: bool) -> Mesh {
    // Pre-allocate with estimated capacity (reduces reallocations)
    let estimated_vertices = (grid.nx * grid.ny * grid.nz) / 20;
    let mut vertices: Vec<f32> = Vec::with_capacity(estimated_vertices * 3);
    let mut normals: Vec<f32> = Vec::with_capacity(estimated_vertices * 3);
    let mut indices: Vec<u32> = Vec::with_capacity(estimated_vertices * 2);

    // Use flat array edge cache instead of HashMap
    let mut edge_cache = EdgeCache::new(grid.nx, grid.ny, grid.nz);

    let edge_corners: [[usize; 2]; 12] = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ];
    let corner_offsets: [[i32; 3]; 8] = [
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ];

    // Inline vertex addition with trilinear normal computation
    let mut add_vertex = |pos: [f32; 3]| -> u32 {
        let idx = (vertices.len() / 3) as u32;
        vertices.extend_from_slice(&pos);
        let n = compute_normal(grid, pos);
        normals.extend_from_slice(&n);
        idx
    };

    let nz_limit = (grid.nz.saturating_sub(1)) as i32;
    let ny_limit = (grid.ny.saturating_sub(1)) as i32;
    let nx_limit = (grid.nx.saturating_sub(1)) as i32;

    for iz in 0..nz_limit {
        for iy in 0..ny_limit {
            for ix in 0..nx_limit {
                // Load all 8 corner values (likely cache-friendly access pattern)
                let mut v = [0.0f32; 8];
                for i in 0..8 {
                    let o = corner_offsets[i];
                    v[i] = grid.get(ix + o[0], iy + o[1], iz + o[2]);
                }

                // Build cube index
                let mut cube_index = 0u16;
                for i in 0..8 {
                    if v[i] < isovalue {
                        cube_index |= 1 << i;
                    }
                }

                let edge_bits = EDGE_TABLE[cube_index as usize];
                if edge_bits == 0 {
                    continue;
                }

                // Compute corner world positions only for active cells
                let mut corners = [[0.0f32; 3]; 8];
                for i in 0..8 {
                    let o = corner_offsets[i];
                    corners[i] = grid.world_pos(ix + o[0], iy + o[1], iz + o[2]);
                }

                let mut edge_verts: [i32; 12] = [-1; 12];
                for e in 0..12 {
                    if (edge_bits & (1 << e)) != 0 {
                        let c0 = edge_corners[e][0];
                        let c1 = edge_corners[e][1];
                        let o0 = corner_offsets[c0];
                        let o1 = corner_offsets[c1];
                        let (axis, kx, ky, kz) = if o0[0] != o1[0] {
                            (0u8, ix + o0[0].min(o1[0]), iy + o0[1], iz + o0[2])
                        } else if o0[1] != o1[1] {
                            (1u8, ix + o0[0], iy + o0[1].min(o1[1]), iz + o0[2])
                        } else {
                            (2u8, ix + o0[0], iy + o0[1], iz + o0[2].min(o1[2]))
                        };

                        if let Some(cached_idx) = edge_cache.get(axis, kx, ky, kz) {
                            edge_verts[e] = cached_idx as i32;
                        } else {
                            let pos = interpolate_vertex(corners[c0], corners[c1], v[c0], v[c1], isovalue);
                            let idx = add_vertex(pos);
                            edge_cache.set(axis, kx, ky, kz, idx);
                            edge_verts[e] = idx as i32;
                        }
                    }
                }

                let tri_row = &TRI_TABLE[cube_index as usize];
                let mut t = 0usize;
                while t + 2 < tri_row.len() && tri_row[t] != -1 {
                    let i0 = edge_verts[tri_row[t] as usize];
                    let i1 = edge_verts[tri_row[t + 1] as usize];
                    let i2 = edge_verts[tri_row[t + 2] as usize];
                    if i0 >= 0 && i1 >= 0 && i2 >= 0 {
                        indices.push(i0 as u32);
                        indices.push(i1 as u32);
                        indices.push(i2 as u32);
                    }
                    t += 3;
                }
            }
        }
    }

    let normals_out = if smooth_normals {
        smooth_normals_fn(&vertices, &normals, &indices)
    } else {
        normals
    };

    Mesh {
        vertices,
        normals: normals_out,
        indices,
    }
}

fn smooth_normals_fn(vertices: &[f32], per_vertex_normals: &[f32], indices: &[u32]) -> Vec<f32> {
    let vertex_count = vertices.len() / 3;
    // Reuse a single buffer - accumulate then normalize in-place
    let mut result = vec![0.0f32; vertex_count * 3];

    for tri in indices.chunks_exact(3) {
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;

        // Compute face normal via cross product
        let v0x = vertices[i0 * 3];
        let v0y = vertices[i0 * 3 + 1];
        let v0z = vertices[i0 * 3 + 2];

        let e1x = vertices[i1 * 3] - v0x;
        let e1y = vertices[i1 * 3 + 1] - v0y;
        let e1z = vertices[i1 * 3 + 2] - v0z;

        let e2x = vertices[i2 * 3] - v0x;
        let e2y = vertices[i2 * 3 + 1] - v0y;
        let e2z = vertices[i2 * 3 + 2] - v0z;

        let nx = e1y * e2z - e1z * e2y;
        let ny = e1z * e2x - e1x * e2z;
        let nz = e1x * e2y - e1y * e2x;

        // Accumulate to all three vertices
        result[i0 * 3] += nx;
        result[i0 * 3 + 1] += ny;
        result[i0 * 3 + 2] += nz;
        result[i1 * 3] += nx;
        result[i1 * 3 + 1] += ny;
        result[i1 * 3 + 2] += nz;
        result[i2 * 3] += nx;
        result[i2 * 3 + 1] += ny;
        result[i2 * 3 + 2] += nz;
    }

    // Normalize in-place using fast inverse sqrt
    for i in 0..vertex_count {
        let ax = result[i * 3];
        let ay = result[i * 3 + 1];
        let az = result[i * 3 + 2];
        let len_sq = ax * ax + ay * ay + az * az;
        if len_sq > 1e-8 {
            let inv_len = fast_inv_sqrt(len_sq);
            result[i * 3] = ax * inv_len;
            result[i * 3 + 1] = ay * inv_len;
            result[i * 3 + 2] = az * inv_len;
        } else {
            // Fallback to per-vertex normals
            result[i * 3] = per_vertex_normals[i * 3];
            result[i * 3 + 1] = per_vertex_normals[i * 3 + 1];
            result[i * 3 + 2] = per_vertex_normals[i * 3 + 2];
        }
    }
    result
}

fn find_connected_components(vertex_count: usize, indices: &[u32]) -> (Vec<usize>, usize) {
    let mut parent: Vec<usize> = (0..vertex_count).collect();

    fn find(parent: &mut [usize], x: usize) -> usize {
        if parent[x] != x {
            let root = find(parent, parent[x]);
            parent[x] = root;
        }
        parent[x]
    }

    fn union(parent: &mut [usize], x: usize, y: usize) {
        let px = find(parent, x);
        let py = find(parent, y);
        if px != py {
            parent[px] = py;
        }
    }

    for tri in indices.chunks(3) {
        if tri.len() < 3 {
            continue;
        }
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;
        union(&mut parent, i0, i1);
        union(&mut parent, i1, i2);
    }

    let mut component_map: HashMap<usize, usize> = HashMap::new();
    let mut components = vec![0usize; vertex_count];
    let mut next_component = 0usize;

    for i in 0..vertex_count {
        let root = find(&mut parent, i);
        let entry = component_map.entry(root).or_insert_with(|| {
            let c = next_component;
            next_component += 1;
            c
        });
        components[i] = *entry;
    }

    (components, next_component)
}

struct AtomHash {
    cells: HashMap<(i32, i32, i32), Vec<usize>>,
    cell_size: f32,
}

fn build_atom_hash(atoms: &[Atom], cell_size: f32) -> Option<AtomHash> {
    if !cell_size.is_finite() || cell_size <= 0.0 {
        return None;
    }
    let mut cells: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
    for (i, atom) in atoms.iter().enumerate() {
        let ix = (atom.center[0] / cell_size).floor() as i32;
        let iy = (atom.center[1] / cell_size).floor() as i32;
        let iz = (atom.center[2] / cell_size).floor() as i32;
        cells.entry((ix, iy, iz)).or_insert_with(Vec::new).push(i);
    }
    Some(AtomHash { cells, cell_size })
}

fn query_atom_hash(atom_hash: &AtomHash, x: f32, y: f32, z: f32) -> Vec<usize> {
    let ix = (x / atom_hash.cell_size).floor() as i32;
    let iy = (y / atom_hash.cell_size).floor() as i32;
    let iz = (z / atom_hash.cell_size).floor() as i32;
    let mut indices = Vec::new();
    for dx in -1..=1 {
        for dy in -1..=1 {
            for dz in -1..=1 {
                if let Some(bucket) = atom_hash.cells.get(&(ix + dx, iy + dy, iz + dz)) {
                    indices.extend_from_slice(bucket);
                }
            }
        }
    }
    indices
}

fn filter_ses_components(mesh: Mesh, atoms: &[Atom], probe_radius: f32, max_atom_radius: f32) -> Mesh {
    let vertex_count = mesh.vertices.len() / 3;
    let (components, count) = find_connected_components(vertex_count, &mesh.indices);
    if count <= 1 {
        return mesh;
    }

    let mut component_valid = vec![false; count];
    let mut checked = vec![false; count];
    let threshold = probe_radius * 1.5;
    let max_center_distance = max_atom_radius + threshold;
    let max_center_dist_sq = max_center_distance * max_center_distance;
    let atom_hash = build_atom_hash(atoms, max_center_distance);

    for i in 0..vertex_count {
        let comp = components[i];
        if checked[comp] {
            continue;
        }
        checked[comp] = true;
        let vx = mesh.vertices[i * 3];
        let vy = mesh.vertices[i * 3 + 1];
        let vz = mesh.vertices[i * 3 + 2];

        if let Some(ref hash) = atom_hash {
            let candidates = query_atom_hash(hash, vx, vy, vz);
            if candidates.is_empty() {
                continue;
            }
            for idx in candidates {
                let atom = atoms[idx];
                let dx = vx - atom.center[0];
                let dy = vy - atom.center[1];
                let dz = vz - atom.center[2];
                let dist_sq = dx * dx + dy * dy + dz * dz;
                // Early rejection using squared distance
                if dist_sq > max_center_dist_sq {
                    continue;
                }
                // Only compute sqrt when we need the actual distance
                let threshold_for_atom = atom.radius + threshold;
                if dist_sq < threshold_for_atom * threshold_for_atom {
                    component_valid[comp] = true;
                    break;
                }
            }
        } else {
            for atom in atoms {
                let dx = vx - atom.center[0];
                let dy = vy - atom.center[1];
                let dz = vz - atom.center[2];
                let dist_sq = dx * dx + dy * dy + dz * dz;
                let threshold_for_atom = atom.radius + threshold;
                // Use squared distance comparison to avoid sqrt
                if dist_sq < threshold_for_atom * threshold_for_atom {
                    component_valid[comp] = true;
                    break;
                }
            }
        }
    }

    let mut new_indices: Vec<usize> = Vec::new();
    for tri in mesh.indices.chunks(3) {
        if tri.len() < 3 {
            continue;
        }
        let comp = components[tri[0] as usize];
        if component_valid[comp] {
            new_indices.push(tri[0] as usize);
            new_indices.push(tri[1] as usize);
            new_indices.push(tri[2] as usize);
        }
    }

    let mut vertex_map: HashMap<usize, usize> = HashMap::new();
    let mut new_vertices: Vec<f32> = Vec::new();
    let mut new_normals: Vec<f32> = Vec::new();
    let mut remapped_indices: Vec<u32> = Vec::with_capacity(new_indices.len());

    for &old_idx in &new_indices {
        let entry = *vertex_map.entry(old_idx).or_insert_with(|| {
            let new_idx = new_vertices.len() / 3;
            new_vertices.push(mesh.vertices[old_idx * 3]);
            new_vertices.push(mesh.vertices[old_idx * 3 + 1]);
            new_vertices.push(mesh.vertices[old_idx * 3 + 2]);
            new_normals.push(mesh.normals[old_idx * 3]);
            new_normals.push(mesh.normals[old_idx * 3 + 1]);
            new_normals.push(mesh.normals[old_idx * 3 + 2]);
            new_idx
        });
        remapped_indices.push(entry as u32);
    }

    Mesh {
        vertices: new_vertices,
        normals: new_normals,
        indices: remapped_indices,
    }
}

fn flip_normals(mesh: &mut Mesh) {
    for n in mesh.normals.iter_mut() {
        *n = -*n;
    }
}

/// Deduplicate vertices using spatial hashing.
/// Returns representative points from each occupied cell.
fn deduplicate_vertices(vertices: &[f32], cell_size: f32) -> Vec<[f32; 3]> {
    let vertex_count = vertices.len() / 3;
    if vertex_count == 0 {
        return Vec::new();
    }

    // Use a HashMap to track occupied cells and their representative vertex
    let inv_cell = 1.0 / cell_size;
    let mut cells: HashMap<(i32, i32, i32), [f32; 3]> = HashMap::with_capacity(vertex_count / 4);

    for i in 0..vertex_count {
        let x = vertices[i * 3];
        let y = vertices[i * 3 + 1];
        let z = vertices[i * 3 + 2];

        let cx = (x * inv_cell).floor() as i32;
        let cy = (y * inv_cell).floor() as i32;
        let cz = (z * inv_cell).floor() as i32;

        // Only keep first vertex in each cell (or could average them)
        cells.entry((cx, cy, cz)).or_insert([x, y, z]);
    }

    cells.into_values().collect()
}

fn mesh_to_js(mesh: Mesh) -> Result<JsValue, JsValue> {
    let obj = Object::new();
    let vertices = Float32Array::from(mesh.vertices.as_slice());
    let normals = Float32Array::from(mesh.normals.as_slice());
    let indices = Uint32Array::from(mesh.indices.as_slice());
    Reflect::set(&obj, &JsValue::from_str("vertices"), &vertices)?;
    Reflect::set(&obj, &JsValue::from_str("normals"), &normals)?;
    Reflect::set(&obj, &JsValue::from_str("indices"), &indices)?;
    Ok(obj.into())
}

#[wasm_bindgen]
pub fn compute_ses(
    centers: &[f32],
    radii: &[f32],
    probe_radius: f32,
    resolution: f32,
    return_sas: bool,
    smooth_normals: bool,
) -> Result<JsValue, JsValue> {
    if centers.len() % 3 != 0 {
        return Err(JsValue::from_str("centers length must be multiple of 3"));
    }
    let atom_count = centers.len() / 3;
    if radii.len() != atom_count {
        return Err(JsValue::from_str("radii length must match centers"));
    }
    if atom_count == 0 {
        return mesh_to_js(Mesh { vertices: Vec::new(), normals: Vec::new(), indices: Vec::new() });
    }

    let mut atoms: Vec<Atom> = Vec::with_capacity(atom_count);
    let mut bounds_min = [f32::INFINITY; 3];
    let mut bounds_max = [f32::NEG_INFINITY; 3];
    let mut max_atom_radius = 0.0f32;

    for i in 0..atom_count {
        let center = [centers[i * 3], centers[i * 3 + 1], centers[i * 3 + 2]];
        let radius = radii[i];
        atoms.push(Atom { center, radius });
        for a in 0..3 {
            bounds_min[a] = bounds_min[a].min(center[a]);
            bounds_max[a] = bounds_max[a].max(center[a]);
        }
        if radius > max_atom_radius {
            max_atom_radius = radius;
        }
    }

    let padding = 2.0 * probe_radius + max_atom_radius + resolution;
    let mut grid = DistanceGrid::new(bounds_min, bounds_max, resolution, padding, 2.0)
        .map_err(|e| JsValue::from_str(&e))?;

    for atom in &atoms {
        grid.add_sphere(atom.center, atom.radius + probe_radius);
    }

    let sas_mesh = marching_cubes(&grid, 0.0, smooth_normals);
    if sas_mesh.vertices.is_empty() {
        return mesh_to_js(sas_mesh);
    }
    if return_sas {
        return mesh_to_js(sas_mesh);
    }

    grid.clear();

    // Deduplicate SAS vertices spatially to reduce redundant probe sphere calculations.
    // Vertices within half resolution distance contribute nearly identical probe spheres.
    let dedup_cell_size = resolution * 0.5;
    let deduped_centers = deduplicate_vertices(&sas_mesh.vertices, dedup_cell_size);

    for center in &deduped_centers {
        grid.add_sphere(*center, probe_radius);
    }

    let ses_mesh = marching_cubes(&grid, 0.0, smooth_normals);
    let mut filtered = filter_ses_components(ses_mesh, &atoms, probe_radius, max_atom_radius);
    flip_normals(&mut filtered);
    mesh_to_js(filtered)
}
