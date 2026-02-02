import test from "node:test";
import assert from "node:assert/strict";
import { computeSES } from "../src/surface.js";

test("computeSES excludes outer shell for a single atom", () => {
  const atomRadius = 1.5;
  const probeRadius = 1.4;
  const resolution = 0.25;
  const atoms = [{ center: [0, 0, 0], radius: atomRadius }];

  const mesh = computeSES(atoms, { probeRadius, resolution });

  assert.ok(mesh.vertices.length > 0, "SES mesh should not be empty");

  let maxDist = 0;
  for (let i = 0; i < mesh.vertices.length; i += 3) {
    const x = mesh.vertices[i];
    const y = mesh.vertices[i + 1];
    const z = mesh.vertices[i + 2];
    const d = Math.hypot(x, y, z);
    if (d > maxDist) maxDist = d;
  }

  const outerShellRadius = atomRadius + 2 * probeRadius;
  assert.ok(
    maxDist < outerShellRadius - probeRadius * 0.5,
    `SES should not include outer shell (maxDist=${maxDist.toFixed(2)}Å)`
  );
});

test("computeSES normals are unit vectors pointing outward", () => {
  const atoms = [{ center: [0, 0, 0], radius: 1.5 }];
  const mesh = computeSES(atoms, { probeRadius: 1.4, resolution: 0.25 });

  assert.ok(mesh.normals.length === mesh.vertices.length, "Should have one normal per vertex");

  const vertexCount = mesh.vertices.length / 3;
  let outwardCount = 0;

  for (let i = 0; i < vertexCount; i++) {
    const nx = mesh.normals[i * 3];
    const ny = mesh.normals[i * 3 + 1];
    const nz = mesh.normals[i * 3 + 2];

    // Check unit length
    const len = Math.hypot(nx, ny, nz);
    assert.ok(Math.abs(len - 1.0) < 0.01, `Normal should be unit length (got ${len})`);

    // For a single sphere centered at origin, normals should point outward (same direction as vertex)
    const vx = mesh.vertices[i * 3];
    const vy = mesh.vertices[i * 3 + 1];
    const vz = mesh.vertices[i * 3 + 2];
    const vlen = Math.hypot(vx, vy, vz);

    if (vlen > 0.01) {
      const dot = (nx * vx + ny * vy + nz * vz) / vlen;
      if (dot > 0.5) outwardCount++;
    }
  }

  // Most normals should point outward for a single atom SES
  const outwardRatio = outwardCount / vertexCount;
  assert.ok(outwardRatio > 0.9, `Most normals should point outward (got ${(outwardRatio * 100).toFixed(1)}%)`);
});

test("computeSES default resolution matches explicit 0.25", () => {
  const atoms = [
    { center: [0, 0, 0], radius: 1.5 },
    { center: [2.5, 0, 0], radius: 1.5 }
  ];

  const meshDefault = computeSES(atoms, { probeRadius: 1.4 });
  const meshExplicit = computeSES(atoms, { probeRadius: 1.4, resolution: 0.25 });

  assert.equal(meshDefault.vertices.length, meshExplicit.vertices.length, "Vertices length should match");
  assert.equal(meshDefault.indices.length, meshExplicit.indices.length, "Indices length should match");
  assert.equal(meshDefault.normals.length, meshExplicit.normals.length, "Normals length should match");
});

test("computeSES can return SAS surface", () => {
  const atoms = [{ center: [0, 0, 0], radius: 1.5 }];
  const sasMesh = computeSES(atoms, { probeRadius: 1.4, resolution: 0.5, sas: true });

  assert.ok(sasMesh.vertices.length > 0, "SAS mesh should not be empty");
  assert.ok(sasMesh.indices.length > 0, "SAS mesh should have triangles");
});
