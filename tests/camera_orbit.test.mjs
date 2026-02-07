import test from "node:test";
import assert from "node:assert/strict";
import { applyOrbitDragToRotation, quatRotateVec, resolveRotationLock } from "../src/camera_orbit.js";

function absDot(a, b) {
  return Math.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
}

test("horizontal drag yaws around world vertical axis", () => {
  const start = [0, 0, 0, 1];
  const rotated = applyOrbitDragToRotation(start, 120, 0);
  const forward = quatRotateVec(rotated, [0, 0, 1]);

  assert(Math.abs(forward[1]) < 1e-6, "Yaw-only drag should keep forward Y unchanged");
  assert(Math.abs(forward[0]) > 1e-3, "Yaw-only drag should change horizontal heading");
});

test("vertical drag pitches around camera horizontal axis", () => {
  const start = [0, 0, 0, 1];
  const rotated = applyOrbitDragToRotation(start, 0, -120);
  const forward = quatRotateVec(rotated, [0, 0, 1]);

  assert(Math.abs(forward[1]) > 1e-3, "Pitch drag should tilt camera vertically");
  assert(Math.abs(forward[0]) < 1e-6, "Pitch-only drag should not introduce yaw");
});

test("pitch is clamped near poles to keep orbit stable", () => {
  let rotation = [0, 0, 0, 1];
  for (let i = 0; i < 200; i += 1) {
    rotation = applyOrbitDragToRotation(rotation, 0, -20);
  }
  const forward = quatRotateVec(rotation, [0, 0, 1]);

  assert(Math.abs(forward[1]) < 0.995, "Forward vector Y should stay below pole clamp");
});

test("combined drag still produces a valid orientation", () => {
  const rotation = applyOrbitDragToRotation([0, 0, 0, 1], 140, -90);
  const right = quatRotateVec(rotation, [1, 0, 0]);
  const up = quatRotateVec(rotation, [0, 1, 0]);
  const forward = quatRotateVec(rotation, [0, 0, 1]);

  assert(absDot(right, up) < 1e-6, "Right and up should remain orthogonal");
  assert(absDot(right, forward) < 1e-6, "Right and forward should remain orthogonal");
  assert(absDot(up, forward) < 1e-6, "Up and forward should remain orthogonal");
});

test("rotation lock chooses dominant drag axis and then stays fixed", () => {
  const threshold = 2.0;
  const nearZeroLock = resolveRotationLock(null, 1.0, 1.2, threshold);
  assert.equal(nearZeroLock, null, "Small movement should not lock axis");

  const yawLock = resolveRotationLock(null, 8.0, 3.0, threshold);
  assert.equal(yawLock, "yaw", "Horizontal-dominant drag should lock to yaw");

  const persistent = resolveRotationLock("yaw", 1.0, 20.0, threshold);
  assert.equal(persistent, "yaw", "Existing lock should persist for the gesture");

  const pitchLock = resolveRotationLock(null, 2.0, 5.0, threshold);
  assert.equal(pitchLock, "pitch", "Vertical-dominant drag should lock to pitch");
});
