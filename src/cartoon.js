const DEFAULT_OPTIONS = {
  helixRadius: 0.35,
  loopRadius: 0.2,
  sheetWidth: 2.9,
  helixWidth: 2.7,
  helixEdgeWidthScale: 0.72,
  helixCrossSectionSegments: 3,
  helixThickness: 0.25,
  sheetThickness: 0.25,
  helixSides: 20,
  loopSides: 16,
  helixSubdivisions: 8,
  loopSubdivisions: 7,
  sheetSubdivisions: 6,
  maxGap: 4.8,
  arrowBaseScale: 1.8,
  arrowLength: 2.3,
  hbondDistance: 4.0,
  hbondEnergyCutoff: -0.5,
  colors: {
    helixFront: [0.85, 0.1, 0.1],
    helixBack: [0.95, 0.95, 0.95],
    sheet: [0.2, 0.7, 0.2],
    loop: [0.65, 0.65, 0.65]
  }
};


function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function vec3Add(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function vec3Sub(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function vec3Scale(a, s) {
  return [a[0] * s, a[1] * s, a[2] * s];
}

function vec3Dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function vec3Cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0]
  ];
}

function vec3Length(a) {
  return Math.sqrt(vec3Dot(a, a));
}

function vec3Normalize(a) {
  const len = vec3Length(a);
  if (len <= 1e-8) return [0, 0, 0];
  return [a[0] / len, a[1] / len, a[2] / len];
}

function vec3ProjectOut(a, n) {
  const d = vec3Dot(a, n);
  return [a[0] - n[0] * d, a[1] - n[1] * d, a[2] - n[2] * d];
}

function vec3Blend(a, b, t) {
  const s = 1 - t;
  return [a[0] * s + b[0] * t, a[1] * s + b[1] * t, a[2] * s + b[2] * t];
}

function smoothstep(t) {
  const x = clamp(t, 0, 1);
  return x * x * (3 - 2 * x);
}

function computeRibbonHalfWidths(width, edgeWidthScale = 1.0) {
  const halfW = width * 0.5;
  const edgeHalfW = halfW * clamp(edgeWidthScale, 0.2, 1.0);
  return { halfW, edgeHalfW };
}

function makeTaperedWidths(count, fullWidth, endWidth, fraction = 0.2) {
  if (count <= 1) return [fullWidth];
  const rampCount = Math.max(1, Math.floor(count * clamp(fraction, 0.05, 0.45)));
  const widths = new Array(count);
  for (let i = 0; i < count; i += 1) {
    const d = Math.min(i, count - 1 - i);
    if (d >= rampCount) {
      widths[i] = fullWidth;
      continue;
    }
    const t = smoothstep(d / rampCount);
    widths[i] = endWidth * (1 - t) + fullWidth * t;
  }
  return widths;
}

function smoothPointsFixedEndpoints(points, iterations = 1, weight = 0.5) {
  if (points.length < 3 || iterations <= 0) return points.slice();
  let current = points.slice();
  const w = clamp(weight, 0, 1);
  for (let iter = 0; iter < iterations; iter += 1) {
    const next = new Array(current.length);
    next[0] = current[0];
    for (let i = 1; i < current.length - 1; i += 1) {
      const avg = vec3Blend(current[i - 1], current[i + 1], 0.5);
      next[i] = vec3Blend(current[i], avg, w);
    }
    next[current.length - 1] = current[current.length - 1];
    current = next;
  }
  return current;
}

function rotateAroundAxis(v, axis, angle) {
  const cosA = Math.cos(angle);
  const sinA = Math.sin(angle);
  const dot = vec3Dot(axis, v);
  const cross = vec3Cross(axis, v);
  return [
    v[0] * cosA + cross[0] * sinA + axis[0] * dot * (1 - cosA),
    v[1] * cosA + cross[1] * sinA + axis[1] * dot * (1 - cosA),
    v[2] * cosA + cross[2] * sinA + axis[2] * dot * (1 - cosA)
  ];
}

function computeTangents(points) {
  const tangents = new Array(points.length);
  for (let i = 0; i < points.length; i += 1) {
    let t;
    if (i === 0) {
      t = vec3Sub(points[1], points[0]);
    } else if (i === points.length - 1) {
      t = vec3Sub(points[i], points[i - 1]);
    } else {
      const forward = vec3Sub(points[i + 1], points[i]);
      const backward = vec3Sub(points[i], points[i - 1]);
      t = vec3Add(forward, backward);
    }
    tangents[i] = vec3Normalize(t);
  }
  return tangents;
}

function pickPerpendicular(tangent) {
  const up = Math.abs(tangent[1]) < 0.8 ? [0, 1, 0] : [1, 0, 0];
  const n = vec3Cross(up, tangent);
  return vec3Normalize(n);
}

function computeFrames(points, referenceNormals = null) {
  const tangents = computeTangents(points);
  const normals = new Array(points.length);
  const binormals = new Array(points.length);

  for (let i = 0; i < points.length; i += 1) {
    const t = tangents[i];
    let n = null;

    if (referenceNormals && referenceNormals[i]) {
      const ref = referenceNormals[i];
      const projected = vec3ProjectOut(ref, t);
      const projectedLen = vec3Length(projected);
      if (projectedLen > 1e-6) {
        n = vec3Normalize(projected);
      }
    }

    if (!n) {
      if (i === 0) {
        n = pickPerpendicular(t);
      } else {
        const prevT = tangents[i - 1];
        const prevN = normals[i - 1];
        const axis = vec3Cross(prevT, t);
        const axisLen = vec3Length(axis);
        if (axisLen > 1e-6) {
          const axisNorm = vec3Scale(axis, 1 / axisLen);
          const angle = Math.acos(clamp(vec3Dot(prevT, t), -1, 1));
          n = rotateAroundAxis(prevN, axisNorm, angle);
        } else {
          n = prevN;
        }
      }
    }

    if (i > 0 && vec3Dot(n, normals[i - 1]) < 0) {
      n = vec3Scale(n, -1);
    }

    const b = vec3Normalize(vec3Cross(t, n));
    const nOrtho = vec3Normalize(vec3Cross(b, t));

    normals[i] = nOrtho;
    binormals[i] = b;
  }

  return { tangents, normals, binormals };
}

function resampleCatmullRom(points, subdivisions) {
  if (subdivisions <= 1 || points.length < 2) {
    return points.slice();
  }

  const out = [];
  for (let i = 0; i < points.length - 1; i += 1) {
    const p0 = points[i - 1] || points[i];
    const p1 = points[i];
    const p2 = points[i + 1];
    const p3 = points[i + 2] || points[i + 1];

    for (let s = 0; s < subdivisions; s += 1) {
      const t = s / subdivisions;
      const t2 = t * t;
      const t3 = t2 * t;
      const m0 = -0.5 * t3 + t2 - 0.5 * t;
      const m1 = 1.5 * t3 - 2.5 * t2 + 1.0;
      const m2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t;
      const m3 = 0.5 * t3 - 0.5 * t2;

      out.push([
        p0[0] * m0 + p1[0] * m1 + p2[0] * m2 + p3[0] * m3,
        p0[1] * m0 + p1[1] * m1 + p2[1] * m2 + p3[1] * m3,
        p0[2] * m0 + p1[2] * m1 + p2[2] * m2 + p3[2] * m3
      ]);
    }
  }
  out.push(points[points.length - 1]);
  return out;
}

function computeRibbonFrames(points, normalHint) {
  const tangents = computeTangents(points);
  const normals = new Array(points.length);
  const binormals = new Array(points.length);

  let n0 = normalHint ? vec3ProjectOut(normalHint, tangents[0]) : null;
  if (!n0 || vec3Length(n0) < 1e-6) {
    n0 = pickPerpendicular(tangents[0]);
  } else {
    n0 = vec3Normalize(n0);
  }
  let b0 = vec3Normalize(vec3Cross(tangents[0], n0));
  n0 = vec3Normalize(vec3Cross(b0, tangents[0]));
  normals[0] = n0;
  binormals[0] = b0;

  for (let i = 1; i < points.length; i += 1) {
    const prevT = tangents[i - 1];
    const t = tangents[i];
    let n = normals[i - 1];
    const axis = vec3Cross(prevT, t);
    const axisLen = vec3Length(axis);
    if (axisLen > 1e-6) {
      const axisNorm = vec3Scale(axis, 1 / axisLen);
      const angle = Math.acos(clamp(vec3Dot(prevT, t), -1, 1));
      n = rotateAroundAxis(n, axisNorm, angle);
    }
    if (vec3Dot(n, normals[i - 1]) < 0) {
      n = vec3Scale(n, -1);
    }
    const b = vec3Normalize(vec3Cross(t, n));
    const nOrtho = vec3Normalize(vec3Cross(b, t));
    normals[i] = nOrtho;
    binormals[i] = b;
  }

  return { tangents, normals, binormals };
}

function computeFlatSheetFrames(points, sheetNormal) {
  // For flat beta sheets: ensure H-bond direction (binormal) is orthogonal to sheet normal
  const tangents = computeTangents(points);
  const normals = new Array(points.length);
  const binormals = new Array(points.length);
  const sn = vec3Normalize(sheetNormal);

  for (let i = 0; i < points.length; i += 1) {
    const t = tangents[i];
    // Binormal = cross(tangent, sheetNormal) is orthogonal to both
    // This ensures H-bond direction ⊥ sheet normal
    let b = vec3Cross(t, sn);
    if (vec3Length(b) < 1e-6) {
      b = pickPerpendicular(t);
    } else {
      b = vec3Normalize(b);
    }
    // Ensure consistent binormal direction along the strand
    if (i > 0 && vec3Dot(b, binormals[i - 1]) < 0) {
      b = vec3Scale(b, -1);
    }
    // Normal completes the orthonormal frame
    const n = vec3Normalize(vec3Cross(b, t));
    normals[i] = n;
    binormals[i] = b;
  }

  return { tangents, normals, binormals };
}

function computeRibbonFramesWithTargets(points, targetNormals, weight = 0.7) {
  const tangents = computeTangents(points);
  const normals = new Array(points.length);
  const binormals = new Array(points.length);
  let prevTarget = null;

  let n0 = targetNormals[0] ? vec3ProjectOut(targetNormals[0], tangents[0]) : null;
  if (!n0 || vec3Length(n0) < 1e-6) {
    n0 = pickPerpendicular(tangents[0]);
  } else {
    n0 = vec3Normalize(n0);
  }
  let b0 = vec3Normalize(vec3Cross(tangents[0], n0));
  n0 = vec3Normalize(vec3Cross(b0, tangents[0]));
  normals[0] = n0;
  binormals[0] = b0;
  prevTarget = n0;

  for (let i = 1; i < points.length; i += 1) {
    const prevT = tangents[i - 1];
    const t = tangents[i];
    let n = normals[i - 1];
    const axis = vec3Cross(prevT, t);
    const axisLen = vec3Length(axis);
    if (axisLen > 1e-6) {
      const axisNorm = vec3Scale(axis, 1 / axisLen);
      const angle = Math.acos(clamp(vec3Dot(prevT, t), -1, 1));
      n = rotateAroundAxis(n, axisNorm, angle);
    }

    let target = targetNormals[i] ? vec3ProjectOut(targetNormals[i], t) : null;
    if (!target || vec3Length(target) < 1e-6) {
      target = n;
    } else {
      target = vec3Normalize(target);
    }

    if (prevTarget && vec3Dot(target, prevTarget) < 0) {
      target = vec3Scale(target, -1);
    }

    if (vec3Dot(target, n) < 0) {
      target = vec3Scale(target, -1);
    }

    let blended = vec3Blend(n, target, weight);
    if (vec3Length(blended) < 1e-6) {
      blended = target;
    }
    blended = vec3Normalize(blended);
    if (vec3Dot(blended, target) < 0) {
      blended = vec3Scale(blended, -1);
    }

    const b = vec3Normalize(vec3Cross(t, blended));
    const nOrtho = vec3Normalize(vec3Cross(b, t));
    normals[i] = nOrtho;
    binormals[i] = b;
    prevTarget = target;
  }

  return { tangents, normals, binormals };
}

function appendStrip(out, pointsA, pointsB, normalsA, normalsB, color, flip) {
  if (pointsA.length < 2 || pointsA.length !== pointsB.length) return;

  const baseIndex = out.positions.length / 3;

  for (let i = 0; i < pointsA.length; i += 1) {
    const a = pointsA[i];
    const b = pointsB[i];
    const na = normalsA[i];
    const nb = normalsB[i];
    out.positions.push(a[0], a[1], a[2]);
    out.positions.push(b[0], b[1], b[2]);
    out.normals.push(na[0], na[1], na[2]);
    out.normals.push(nb[0], nb[1], nb[2]);
  }

  for (let i = 0; i < pointsA.length - 1; i += 1) {
    const i0 = baseIndex + i * 2;
    const i1 = baseIndex + i * 2 + 1;
    const i2 = baseIndex + (i + 1) * 2 + 1;
    const i3 = baseIndex + (i + 1) * 2;

    if (flip) {
      out.indices.push(i0, i1, i2);
      out.indices.push(i0, i2, i3);
    } else {
      out.indices.push(i0, i2, i1);
      out.indices.push(i0, i3, i2);
    }
    out.triColors.push(color[0], color[1], color[2]);
    out.triColors.push(color[0], color[1], color[2]);
  }
}

function appendTube(out, points, radius, sides, color, referenceNormal = null) {
  if (points.length < 2) return;

  const frames = computeRibbonFrames(points, referenceNormal);
  const baseIndex = out.positions.length / 3;

  for (let i = 0; i < points.length; i += 1) {
    const p = points[i];
    const n = frames.normals[i];
    const b = frames.binormals[i];
    for (let k = 0; k < sides; k += 1) {
      const angle = (Math.PI * 2 * k) / sides;
      const radial = vec3Add(vec3Scale(n, Math.cos(angle)), vec3Scale(b, Math.sin(angle)));
      const pos = vec3Add(p, vec3Scale(radial, radius));
      out.positions.push(pos[0], pos[1], pos[2]);
      out.normals.push(radial[0], radial[1], radial[2]);
    }
  }

  for (let i = 0; i < points.length - 1; i += 1) {
    for (let k = 0; k < sides; k += 1) {
      const kNext = (k + 1) % sides;
      const i0 = baseIndex + i * sides + k;
      const i1 = baseIndex + (i + 1) * sides + k;
      const i2 = baseIndex + (i + 1) * sides + kNext;
      const i3 = baseIndex + i * sides + kNext;

      out.indices.push(i0, i1, i2);
      out.indices.push(i0, i2, i3);
      out.triColors.push(color[0], color[1], color[2]);
      out.triColors.push(color[0], color[1], color[2]);
    }
  }
}

function appendTubeMasked(out, points, radius, sides, color, segmentMask, referenceNormal = null) {
  if (points.length < 2) return;

  const frames = computeRibbonFrames(points, referenceNormal);
  const baseIndex = out.positions.length / 3;

  for (let i = 0; i < points.length; i += 1) {
    const p = points[i];
    const n = frames.normals[i];
    const b = frames.binormals[i];
    for (let k = 0; k < sides; k += 1) {
      const angle = (Math.PI * 2 * k) / sides;
      const radial = vec3Add(vec3Scale(n, Math.cos(angle)), vec3Scale(b, Math.sin(angle)));
      const pos = vec3Add(p, vec3Scale(radial, radius));
      out.positions.push(pos[0], pos[1], pos[2]);
      out.normals.push(radial[0], radial[1], radial[2]);
    }
  }

  for (let i = 0; i < points.length - 1; i += 1) {
    if (!segmentMask[i]) continue;
    for (let k = 0; k < sides; k += 1) {
      const kNext = (k + 1) % sides;
      const i0 = baseIndex + i * sides + k;
      const i1 = baseIndex + (i + 1) * sides + k;
      const i2 = baseIndex + (i + 1) * sides + kNext;
      const i3 = baseIndex + i * sides + kNext;

      out.indices.push(i0, i1, i2);
      out.indices.push(i0, i2, i3);
      out.triColors.push(color[0], color[1], color[2]);
      out.triColors.push(color[0], color[1], color[2]);
    }
  }
}

function appendRibbon(out, points, normalHint, widths, color) {
  if (points.length < 2) return;

  const frames = computeRibbonFrames(points, normalHint);

  const baseIndex = out.positions.length / 3;

  for (let i = 0; i < points.length; i += 1) {
    const p = points[i];
    const n = frames.normals[i];
    const b = frames.binormals[i];
    const halfWidth = widths[i] * 0.5;
    const left = vec3Sub(p, vec3Scale(b, halfWidth));
    const right = vec3Add(p, vec3Scale(b, halfWidth));

    out.positions.push(left[0], left[1], left[2]);
    out.positions.push(right[0], right[1], right[2]);
    out.normals.push(n[0], n[1], n[2]);
    out.normals.push(n[0], n[1], n[2]);
  }

  for (let i = 0; i < points.length - 1; i += 1) {
    const i0 = baseIndex + i * 2;
    const i1 = baseIndex + i * 2 + 1;
    const i2 = baseIndex + (i + 1) * 2 + 1;
    const i3 = baseIndex + (i + 1) * 2;

    out.indices.push(i0, i1, i2);
    out.indices.push(i0, i2, i3);
    out.triColors.push(color[0], color[1], color[2]);
    out.triColors.push(color[0], color[1], color[2]);
  }
}

function appendRibbonDoubleSided(out, points, normalHint, widths, frontColor, backColor) {
  if (points.length < 2) return;

  const frames = computeRibbonFrames(points, normalHint);

  const baseIndex = out.positions.length / 3;

  for (let i = 0; i < points.length; i += 1) {
    const p = points[i];
    const n = frames.normals[i];
    const b = frames.binormals[i];
    const halfWidth = widths[i] * 0.5;
    const left = vec3Sub(p, vec3Scale(b, halfWidth));
    const right = vec3Add(p, vec3Scale(b, halfWidth));

    out.positions.push(left[0], left[1], left[2]);
    out.positions.push(right[0], right[1], right[2]);
    out.normals.push(n[0], n[1], n[2]);
    out.normals.push(n[0], n[1], n[2]);
  }

  for (let i = 0; i < points.length - 1; i += 1) {
    const i0 = baseIndex + i * 2;
    const i1 = baseIndex + i * 2 + 1;
    const i2 = baseIndex + (i + 1) * 2 + 1;
    const i3 = baseIndex + (i + 1) * 2;

    out.indices.push(i0, i1, i2);
    out.indices.push(i0, i2, i3);
    out.triColors.push(frontColor[0], frontColor[1], frontColor[2]);
    out.triColors.push(frontColor[0], frontColor[1], frontColor[2]);

    out.indices.push(i0, i2, i1);
    out.indices.push(i0, i3, i2);
    out.triColors.push(backColor[0], backColor[1], backColor[2]);
    out.triColors.push(backColor[0], backColor[1], backColor[2]);
  }
}

function appendRibbonVolume(out, points, normalHint, widths, thickness, colors, targetNormals = null, targetWeight = 0.7, flatSheetNormal = null) {
  if (points.length < 2) return;

  let frames;
  if (flatSheetNormal) {
    // For flat beta sheets: use constant normal direction
    frames = computeFlatSheetFrames(points, flatSheetNormal);
  } else if (targetNormals) {
    frames = computeRibbonFramesWithTargets(points, targetNormals, targetWeight);
  } else {
    frames = computeRibbonFrames(points, normalHint);
  }
  const halfT = thickness * 0.5;
  const edgeWidthScale = colors.edgeWidthScale ?? 1.0;
  const edgeProfileSegments = Math.max(1, Math.floor(colors.edgeProfileSegments ?? 1));
  const profileRows = edgeProfileSegments * 2 + 1;

  const leftProfiles = Array.from({ length: profileRows }, () => []);
  const rightProfiles = Array.from({ length: profileRows }, () => []);
  const leftProfileNormals = Array.from({ length: profileRows }, () => []);
  const rightProfileNormals = Array.from({ length: profileRows }, () => []);

  for (let i = 0; i < points.length; i += 1) {
    const p = points[i];
    const n = frames.normals[i];
    const b = frames.binormals[i];
    const { halfW, edgeHalfW } = computeRibbonHalfWidths(widths[i], edgeWidthScale);
    for (let row = 0; row < profileRows; row += 1) {
      const t = row / (profileRows - 1);
      const theta = (0.5 - t) * Math.PI;
      const cosTheta = Math.cos(theta);
      const sinTheta = Math.sin(theta);
      const sideHalf = edgeHalfW + (halfW - edgeHalfW) * (cosTheta * cosTheta);
      const vertical = halfT * sinTheta;

      const leftPoint = vec3Add(vec3Sub(p, vec3Scale(b, sideHalf)), vec3Scale(n, vertical));
      const rightPoint = vec3Add(vec3Add(p, vec3Scale(b, sideHalf)), vec3Scale(n, vertical));
      leftProfiles[row].push(leftPoint);
      rightProfiles[row].push(rightPoint);

      const leftNormal = vec3Normalize(vec3Add(vec3Scale(b, -cosTheta), vec3Scale(n, sinTheta)));
      const rightNormal = vec3Normalize(vec3Add(vec3Scale(b, cosTheta), vec3Scale(n, sinTheta)));
      leftProfileNormals[row].push(leftNormal);
      rightProfileNormals[row].push(rightNormal);
    }
  }

  const topColor = colors.top || colors.side;
  const bottomColor = colors.bottom || colors.side;
  const sideColor = colors.side || colors.top || colors.bottom;
  appendStrip(
    out,
    leftProfiles[0],
    rightProfiles[0],
    leftProfileNormals[0],
    rightProfileNormals[0],
    topColor,
    false
  );
  appendStrip(
    out,
    leftProfiles[profileRows - 1],
    rightProfiles[profileRows - 1],
    leftProfileNormals[profileRows - 1],
    rightProfileNormals[profileRows - 1],
    bottomColor,
    true
  );

  for (let row = 0; row < profileRows - 1; row += 1) {
    appendStrip(
      out,
      leftProfiles[row],
      leftProfiles[row + 1],
      leftProfileNormals[row],
      leftProfileNormals[row + 1],
      sideColor,
      true
    );
    appendStrip(
      out,
      rightProfiles[row],
      rightProfiles[row + 1],
      rightProfileNormals[row],
      rightProfileNormals[row + 1],
      sideColor,
      false
    );
  }
}

function polylineLength(points) {
  let total = 0;
  for (let i = 0; i < points.length - 1; i += 1) {
    total += vec3Length(vec3Sub(points[i + 1], points[i]));
  }
  return total;
}

function trimPolylineTail(points, vectors, tailLength, minBodyLength = 0.6) {
  if (points.length < 2) {
    return null;
  }
  if (vectors && vectors.length !== points.length) {
    throw new Error("trimPolylineTail vectors length must match points length.");
  }

  const totalLength = polylineLength(points);
  if (totalLength <= 1e-8) {
    return null;
  }

  const maxTailLength = Math.max(0, totalLength - minBodyLength);
  const clampedTailLength = clamp(tailLength, 0, maxTailLength);
  if (clampedTailLength <= 1e-6) {
    return null;
  }

  let remainingTail = clampedTailLength;
  for (let i = points.length - 1; i >= 1; i -= 1) {
    const p0 = points[i - 1];
    const p1 = points[i];
    const seg = vec3Sub(p1, p0);
    const segLen = vec3Length(seg);
    if (segLen <= 1e-8) {
      continue;
    }
    if (remainingTail > segLen) {
      remainingTail -= segLen;
      continue;
    }

    const t = (segLen - remainingTail) / segLen;
    const basePoint = vec3Blend(p0, p1, t);
    const baseTangent = vec3Scale(seg, 1 / segLen);
    const bodyPoints = points.slice(0, i);
    bodyPoints.push(basePoint);

    let bodyVectors = null;
    if (vectors) {
      bodyVectors = vectors.slice(0, i);
      const v0 = vectors[i - 1] || vectors[0];
      const v1 = vectors[i] || v0;
      let vSplit = vec3Blend(v0, v1, t);
      if (vec3Length(vSplit) <= 1e-8) {
        vSplit = v0;
      }
      vSplit = vec3Normalize(vSplit);
      if (bodyVectors.length > 0 && vec3Dot(vSplit, bodyVectors[bodyVectors.length - 1]) < 0) {
        vSplit = vec3Scale(vSplit, -1);
      }
      bodyVectors.push(vSplit);
    }

    return {
      bodyPoints,
      bodyVectors,
      basePoint,
      baseTangent,
      arrowLength: clampedTailLength
    };
  }

  return null;
}

function appendTrianglePrism(out, a, b, c, normal, thickness, color) {
  const halfT = thickness * 0.5;
  const n = vec3Normalize(normal);
  const topOffset = vec3Scale(n, halfT);
  const bottomOffset = vec3Scale(n, -halfT);

  const aTop = vec3Add(a, topOffset);
  const bTop = vec3Add(b, topOffset);
  const cTop = vec3Add(c, topOffset);
  const aBot = vec3Add(a, bottomOffset);
  const bBot = vec3Add(b, bottomOffset);
  const cBot = vec3Add(c, bottomOffset);

  const baseIndex = out.positions.length / 3;
  const verts = [aTop, bTop, cTop, aBot, bBot, cBot];
  for (const v of verts) {
    out.positions.push(v[0], v[1], v[2]);
  }

  // Top face
  out.indices.push(baseIndex + 0, baseIndex + 1, baseIndex + 2);
  out.triColors.push(color[0], color[1], color[2]);
  // Bottom face
  out.indices.push(baseIndex + 5, baseIndex + 4, baseIndex + 3);
  out.triColors.push(color[0], color[1], color[2]);

  const edges = [
    [aTop, bTop, bBot, aBot],
    [bTop, cTop, cBot, bBot],
    [cTop, aTop, aBot, cBot]
  ];
  for (let e = 0; e < edges.length; e += 1) {
    const idx = baseIndex + 6 + e * 4;
    const quad = edges[e];
    for (const v of quad) {
      out.positions.push(v[0], v[1], v[2]);
    }
    out.indices.push(idx + 0, idx + 1, idx + 2);
    out.indices.push(idx + 0, idx + 2, idx + 3);
    out.triColors.push(color[0], color[1], color[2]);
    out.triColors.push(color[0], color[1], color[2]);
  }

  const normals = [];
  normals.push(n, n, n, vec3Scale(n, -1), vec3Scale(n, -1), vec3Scale(n, -1));
  for (let e = 0; e < edges.length; e += 1) {
    const quad = edges[e];
    const e1 = vec3Sub(quad[1], quad[0]);
    const e2 = vec3Sub(quad[3], quad[0]);
    let nn = vec3Normalize(vec3Cross(e1, e2));
    if (vec3Length(nn) < 1e-6) {
      nn = [0, 1, 0];
    }
    normals.push(nn, nn, nn, nn);
  }

  for (const nn of normals) {
    out.normals.push(nn[0], nn[1], nn[2]);
  }
}

function pickAtom(existing, candidate) {
  if (!existing) return candidate;
  const prefer = (atom) => !atom.altLoc || atom.altLoc === "A";
  if (prefer(candidate) && !prefer(existing)) return candidate;
  if (prefer(existing) && !prefer(candidate)) return existing;
  const occA = Number.isFinite(candidate.occupancy) ? candidate.occupancy : 0;
  const occB = Number.isFinite(existing.occupancy) ? existing.occupancy : 0;
  if (occA > occB) return candidate;
  return existing;
}

function buildResidues(atoms) {
  const residues = new Map();
  let orderCounter = 0;

  for (const atom of atoms) {
    if (atom.isHet) continue;
    if (atom.resSeq === undefined || atom.resSeq === null) continue;
    if (atom.chainId === undefined || atom.chainId === null) continue;

    const chainId = atom.chainId;
    const iCode = atom.iCode || "";
    const key = `${chainId}:${atom.resSeq}:${iCode}`;
    let residue = residues.get(key);
    if (!residue) {
      residue = {
        key,
        chainId,
        resSeq: atom.resSeq,
        iCode,
        resName: atom.resName || "",
        atoms: {},
        order: orderCounter += 1
      };
      residues.set(key, residue);
    }

    const name = atom.name ? atom.name.trim().toUpperCase() : "";
    let canonical = name;
    if (canonical === "OXT") {
      canonical = "O";
    }
    if (canonical === "N" || canonical === "CA" || canonical === "C" || canonical === "O") {
      const existing = residue.atoms[canonical];
      residue.atoms[canonical] = pickAtom(existing, atom);
    }
  }

  const out = Array.from(residues.values());
  out.sort((a, b) => a.order - b.order);

  for (const residue of out) {
    const n = residue.atoms.N;
    const ca = residue.atoms.CA;
    const c = residue.atoms.C;
    const o = residue.atoms.O;
    residue.complete = Boolean(n && ca && c && o);
    residue.planeNormal = null;
    if (residue.complete) {
      const v1 = vec3Sub(c.position, ca.position);
      const v2 = vec3Sub(n.position, ca.position);
      const normal = vec3Normalize(vec3Cross(v1, v2));
      residue.planeNormal = normal;
    }
  }

  return out;
}

function hbondEnergy(resA, resB, resBPrev) {
  if (!resBPrev || resBPrev.chainId !== resB.chainId) return 0;
  const o = resA.atoms.O.position;
  const c = resA.atoms.C.position;
  const n = resB.atoms.N.position;
  const cPrev = resBPrev.atoms.C.position;
  const hn = [n[0] - cPrev[0], n[1] - cPrev[1], n[2] - cPrev[2]];
  const hnLen = Math.sqrt(hn[0] * hn[0] + hn[1] * hn[1] + hn[2] * hn[2]);
  if (hnLen < 1e-6) return 0;
  const h = [
    n[0] + (hn[0] / hnLen),
    n[1] + (hn[1] / hnLen),
    n[2] + (hn[2] / hnLen)
  ];

  const dxON = o[0] - n[0];
  const dyON = o[1] - n[1];
  const dzON = o[2] - n[2];
  const rON = Math.sqrt(dxON * dxON + dyON * dyON + dzON * dzON);

  const dxCH = c[0] - h[0];
  const dyCH = c[1] - h[1];
  const dzCH = c[2] - h[2];
  const rCH = Math.sqrt(dxCH * dxCH + dyCH * dyCH + dzCH * dzCH);

  const dxOH = o[0] - h[0];
  const dyOH = o[1] - h[1];
  const dzOH = o[2] - h[2];
  const rOH = Math.sqrt(dxOH * dxOH + dyOH * dyOH + dzOH * dzOH);

  const dxCN = c[0] - n[0];
  const dyCN = c[1] - n[1];
  const dzCN = c[2] - n[2];
  const rCN = Math.sqrt(dxCN * dxCN + dyCN * dyCN + dzCN * dzCN);

  if (rON < 1e-6 || rCH < 1e-6 || rOH < 1e-6 || rCN < 1e-6) return 0;

  const energy = 0.084 * 332 * (1 / rON + 1 / rCH - 1 / rOH - 1 / rCN);
  return energy;
}

function computeHBonds(residues, options) {
  const hbonds = Array.from({ length: residues.length }, () => new Set());
  const maxDist = options.hbondDistance;

  for (let i = 0; i < residues.length; i += 1) {
    const resA = residues[i];
    if (!resA.complete) continue;
    const o = resA.atoms.O.position;

    for (let j = 0; j < residues.length; j += 1) {
      if (i === j) continue;
      const resB = residues[j];
      if (!resB.complete) continue;
      const prev = j > 0 && residues[j - 1].chainId === resB.chainId ? residues[j - 1] : null;
      if (!prev) continue;

      const n = resB.atoms.N.position;
      const dx = o[0] - n[0];
      const dy = o[1] - n[1];
      const dz = o[2] - n[2];
      const rON = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (rON > maxDist) continue;

      const energy = hbondEnergy(resA, resB, prev);
      if (energy < options.hbondEnergyCutoff) {
        hbonds[i].add(j);
      }
    }
  }

  return hbonds;
}

function assignSecondaryStructure(residues, options, hbonds = null) {
  const ss = new Array(residues.length).fill("C");
  const hbondMap = hbonds || computeHBonds(residues, options);

  for (let i = 0; i < residues.length; i += 1) {
    for (const offset of [3, 4, 5]) {
      const j = i + offset;
      if (j < residues.length && residues[i].chainId === residues[j].chainId && hbondMap[i].has(j)) {
        for (let k = i + 1; k <= j; k += 1) {
          ss[k] = "H";
        }
      }
    }
  }

  for (let i = 0; i < residues.length; i += 1) {
    for (let j = i + 4; j < residues.length; j += 1) {
      if (ss[i] === "H" || ss[j] === "H") continue;
      if (residues[i].chainId !== residues[j].chainId) continue;
      if (hbondMap[i].has(j) && hbondMap[j].has(i)) {
        ss[i] = "E";
        ss[j] = "E";
      }
    }
  }

  return ss;
}

function adjustShortSegments(residues, ss) {
  const adjusted = ss.slice();
  const minHelix = 4;
  const minSheet = 2;

  let start = 0;
  while (start < residues.length) {
    const chainId = residues[start].chainId;
    let end = start + 1;
    while (end < residues.length && residues[end].chainId === chainId) {
      end += 1;
    }

    let segStart = start;
    for (let i = start + 1; i <= end; i += 1) {
      const isBreak = i === end || adjusted[i] !== adjusted[segStart];
      if (isBreak) {
        const len = i - segStart;
        const type = adjusted[segStart];
        if (type === "H" && len < minHelix) {
          for (let k = segStart; k < i; k += 1) {
            adjusted[k] = "C";
          }
        }
        if (type === "E" && len < minSheet) {
          for (let k = segStart; k < i; k += 1) {
            adjusted[k] = "C";
          }
        }
        segStart = i;
      }
    }

    start = end;
  }

  return adjusted;
}

function residueTangent(index, residues) {
  const curr = residues[index];
  const prev = index > 0 && residues[index - 1].chainId === curr.chainId ? residues[index - 1] : null;
  const next = index < residues.length - 1 && residues[index + 1].chainId === curr.chainId ? residues[index + 1] : null;
  let t = null;
  if (prev && next) {
    const forward = vec3Sub(next.atoms.CA.position, curr.atoms.CA.position);
    const backward = vec3Sub(curr.atoms.CA.position, prev.atoms.CA.position);
    t = vec3Add(forward, backward);
  } else if (next) {
    t = vec3Sub(next.atoms.CA.position, curr.atoms.CA.position);
  } else if (prev) {
    t = vec3Sub(curr.atoms.CA.position, prev.atoms.CA.position);
  } else {
    t = [1, 0, 0];
  }
  return vec3Normalize(t);
}

function pickHelixNormal(segment, residues, hbonds) {
  let accum = [0, 0, 0];
  let ref = null;
  let count = 0;
  const preferOffsets = [4, 3, 5];

  for (const residue of segment.residues) {
    const i = residue.index;
    const partners = hbonds[i];
    if (!partners || partners.size === 0) continue;

    let target = null;
    for (const offset of preferOffsets) {
      const j = i + offset;
      if (j < residues.length && partners.has(j)) {
        target = j;
        break;
      }
    }
    if (target === null) {
      for (const j of partners) {
        if (residues[j] && residues[j].chainId === residue.chainId) {
          target = j;
          break;
        }
      }
    }
    if (target === null) continue;

    const o = residue.atoms.O.position;
    const n = residues[target].atoms.N.position;
    const hbondDir = vec3Normalize(vec3Sub(n, o));
    const t = residueTangent(i, residues);
    let nvec = vec3Cross(t, hbondDir);
    if (vec3Length(nvec) < 1e-6) continue;
    nvec = vec3Normalize(nvec);
    if (ref && vec3Dot(nvec, ref) < 0) {
      nvec = vec3Scale(nvec, -1);
    }
    if (!ref) ref = nvec;
    accum = vec3Add(accum, nvec);
    count += 1;
  }

  if (count > 0) {
    return vec3Normalize(accum);
  }

  let avgNormal = [0, 0, 0];
  let refPlane = null;
  for (const residue of segment.residues) {
    if (!residue.planeNormal) continue;
    let n = residue.planeNormal;
    if (refPlane && vec3Dot(n, refPlane) < 0) {
      n = vec3Scale(n, -1);
    }
    if (!refPlane) refPlane = n;
    avgNormal = vec3Add(avgNormal, n);
  }
  avgNormal = vec3Normalize(avgNormal);
  if (vec3Length(avgNormal) < 1e-6) {
    return [0, 1, 0];
  }
  return avgNormal;
}

function computeHelixAxis(points) {
  if (points.length < 2) return [1, 0, 0];
  const dir = vec3Sub(points[points.length - 1], points[0]);
  const len = vec3Length(dir);
  if (len < 1e-6) return [1, 0, 0];
  return vec3Scale(dir, 1 / len);
}

function computeCentroid(points) {
  let sum = [0, 0, 0];
  for (const p of points) {
    sum = vec3Add(sum, p);
  }
  return vec3Scale(sum, 1 / Math.max(1, points.length));
}

function flattenPointsToPlane(points, normal) {
  if (points.length < 2) return points.slice();
  const centroid = computeCentroid(points);
  const n = vec3Normalize(normal);
  return points.map((p) => {
    const rel = vec3Sub(p, centroid);
    const dist = vec3Dot(rel, n);
    return vec3Sub(p, vec3Scale(n, dist));
  });
}

function computePeptideMidpoints(residues) {
  // Use midpoints between consecutive CA atoms - these lie closer to the sheet plane
  // and avoid the up-down oscillation of individual CA atoms
  const midpoints = [];
  for (let i = 0; i < residues.length; i += 1) {
    if (i === 0) {
      // First point: use CA position
      midpoints.push(residues[i].atoms.CA.position);
    } else if (i === residues.length - 1) {
      // Last point: use CA position
      midpoints.push(residues[i].atoms.CA.position);
    } else {
      // Middle points: use midpoint between current and next CA
      const curr = residues[i].atoms.CA.position;
      const next = residues[i + 1].atoms.CA.position;
      midpoints.push(vec3Blend(curr, next, 0.5));
    }
  }
  return midpoints;
}

function linearInterpolatePoints(points, subdivisions) {
  if (subdivisions <= 1 || points.length < 2) return points.slice();
  const out = [];
  for (let i = 0; i < points.length - 1; i += 1) {
    const p1 = points[i];
    const p2 = points[i + 1];
    for (let s = 0; s < subdivisions; s += 1) {
      const t = s / subdivisions;
      out.push(vec3Blend(p1, p2, t));
    }
  }
  out.push(points[points.length - 1]);
  return out;
}

function hermiteInterpolatePoints(points, subdivisions, tension = 0.5) {
  // Hermite spline with tension control - tension=1 gives linear, tension=0 gives Catmull-Rom
  if (subdivisions <= 1 || points.length < 2) return points.slice();

  const out = [];
  for (let i = 0; i < points.length - 1; i += 1) {
    const p0 = points[i - 1] || points[i];
    const p1 = points[i];
    const p2 = points[i + 1];
    const p3 = points[i + 2] || points[i + 1];

    // Compute tangents with tension
    const scale = (1 - tension) * 0.5;
    const m1 = vec3Scale(vec3Sub(p2, p0), scale);
    const m2 = vec3Scale(vec3Sub(p3, p1), scale);

    for (let s = 0; s < subdivisions; s += 1) {
      const t = s / subdivisions;
      const t2 = t * t;
      const t3 = t2 * t;

      // Hermite basis functions
      const h00 = 2 * t3 - 3 * t2 + 1;
      const h10 = t3 - 2 * t2 + t;
      const h01 = -2 * t3 + 3 * t2;
      const h11 = t3 - t2;

      out.push([
        p1[0] * h00 + m1[0] * h10 + p2[0] * h01 + m2[0] * h11,
        p1[1] * h00 + m1[1] * h10 + p2[1] * h01 + m2[1] * h11,
        p1[2] * h00 + m1[2] * h10 + p2[2] * h01 + m2[2] * h11
      ]);
    }
  }
  out.push(points[points.length - 1]);
  return out;
}

function splitByChain(residues) {
  const chains = new Map();
  for (const residue of residues) {
    const list = chains.get(residue.chainId) || [];
    list.push(residue);
    chains.set(residue.chainId, list);
  }
  return chains;
}

function fitPlaneNormal(points) {
  // Fit a plane to points using covariance matrix / PCA
  // Returns the normal of the best-fit plane (smallest eigenvalue direction)
  if (points.length < 3) return [0, 1, 0];

  // Compute centroid
  let cx = 0, cy = 0, cz = 0;
  for (const p of points) {
    cx += p[0]; cy += p[1]; cz += p[2];
  }
  cx /= points.length; cy /= points.length; cz /= points.length;

  // Compute covariance matrix elements
  let xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;
  for (const p of points) {
    const dx = p[0] - cx, dy = p[1] - cy, dz = p[2] - cz;
    xx += dx * dx; xy += dx * dy; xz += dx * dz;
    yy += dy * dy; yz += dy * dz; zz += dz * dz;
  }

  // Find the smallest eigenvector of the 3x3 symmetric matrix
  // Using inverse power iteration
  // Cov = [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]

  // Start with initial guess
  let v = [1, 0, 0];
  if (Math.abs(xx) < Math.abs(yy) && Math.abs(xx) < Math.abs(zz)) {
    v = [1, 0, 0];
  } else if (Math.abs(yy) < Math.abs(zz)) {
    v = [0, 1, 0];
  } else {
    v = [0, 0, 1];
  }

  // Power iteration to find dominant eigenvector (largest eigenvalue)
  for (let iter = 0; iter < 20; iter++) {
    const nx = xx * v[0] + xy * v[1] + xz * v[2];
    const ny = xy * v[0] + yy * v[1] + yz * v[2];
    const nz = xz * v[0] + yz * v[1] + zz * v[2];
    const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
    if (len < 1e-10) break;
    v = [nx / len, ny / len, nz / len];
  }

  // v is now the largest eigenvector (direction of most variance)
  // The normal to the plane should be perpendicular to this
  // Find a second eigenvector by deflating and iterating again
  const e1 = v;
  let e2 = Math.abs(e1[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  e2 = vec3Normalize(vec3Cross(e1, e2));

  // Power iteration on deflated matrix for second eigenvector
  for (let iter = 0; iter < 20; iter++) {
    const nx = xx * e2[0] + xy * e2[1] + xz * e2[2];
    const ny = xy * e2[0] + yy * e2[1] + yz * e2[2];
    const nz = xz * e2[0] + yz * e2[1] + zz * e2[2];
    // Project out e1 component
    const proj = nx * e1[0] + ny * e1[1] + nz * e1[2];
    const px = nx - proj * e1[0];
    const py = ny - proj * e1[1];
    const pz = nz - proj * e1[2];
    const len = Math.sqrt(px * px + py * py + pz * pz);
    if (len < 1e-10) break;
    e2 = [px / len, py / len, pz / len];
  }

  // The normal is perpendicular to both e1 and e2
  const normal = vec3Normalize(vec3Cross(e1, e2));
  return vec3Length(normal) > 1e-6 ? normal : [0, 1, 0];
}

function collectSheetDistanceContacts(residues, ss, options = {}) {
  const maxDist = options.hbondDistance ?? DEFAULT_OPTIONS.hbondDistance;
  const isSheet = (i) => ss[i] === "E";
  const directedContacts = [];
  const outgoingContacts = Array.from({ length: residues.length }, () => new Set());

  for (let i = 0; i < residues.length; i += 1) {
    if (!isSheet(i)) continue;
    for (let j = i + 1; j < residues.length; j += 1) {
      if (!isSheet(j)) continue;
      if (!isInterStrandPair(residues, i, j)) continue;

      const oi = residues[i].atoms.O.position;
      const nj = residues[j].atoms.N.position;
      const dirIJ = vec3Sub(nj, oi);
      const distIJ = vec3Length(dirIJ);
      if (distIJ <= maxDist && distIJ > 1e-6) {
        directedContacts.push({ from: i, to: j, direction: dirIJ });
        outgoingContacts[i].add(j);
      }

      const oj = residues[j].atoms.O.position;
      const ni = residues[i].atoms.N.position;
      const dirJI = vec3Sub(ni, oj);
      const distJI = vec3Length(dirJI);
      if (distJI <= maxDist && distJI > 1e-6) {
        directedContacts.push({ from: j, to: i, direction: dirJI });
        outgoingContacts[j].add(i);
      }
    }
  }

  return { directedContacts, outgoingContacts };
}

function computeSheetNormals(residues, ss, hbonds, options = {}) {
  const aligned = new Array(residues.length).fill(null);
  const isSheet = (i) => ss[i] === "E";
  const { outgoingContacts } = collectSheetDistanceContacts(residues, ss, options);
  const incomingHBonds = makeIncomingHBonds(outgoingContacts, residues.length);
  const maxGap = options.maxGap ?? DEFAULT_OPTIONS.maxGap;
  const strands = segmentByType(residues, ss, maxGap).filter((segment) => segment.type === "E");

  // Compute local normals along each strand from local inter-strand contacts.
  // This avoids forcing one global normal across curved/twisted beta strands.
  for (const strand of strands) {
    const members = strand.residues.map((residue) => residue.index);
    const localNormals = new Array(members.length).fill(null);
    let strandRef = null;

    for (let m = 0; m < members.length; m += 1) {
      const i = members[m];
      const t = residueTangent(i, residues);
      if (vec3Length(t) < 1e-6) continue;

      let hAccum = [0, 0, 0];
      let hRef = null;
      let hCount = 0;

      const accumulateHBondDir = (rawDir) => {
        let dir = rawDir;
        if (vec3Length(dir) < 1e-6) return;
        dir = vec3ProjectOut(dir, t);
        if (vec3Length(dir) < 1e-6) return;
        dir = vec3Normalize(dir);
        if (hRef && vec3Dot(dir, hRef) < 0) dir = vec3Scale(dir, -1);
        if (!hRef) hRef = dir;
        hAccum = vec3Add(hAccum, dir);
        hCount += 1;
      };

      const outgoing = outgoingContacts[i] || new Set();
      for (const j of outgoing) {
        if (!isSheet(j) || !isInterStrandPair(residues, i, j)) continue;
        const oi = residues[i].atoms.O.position;
        const nj = residues[j].atoms.N.position;
        accumulateHBondDir(vec3Sub(nj, oi));
      }

      const incoming = incomingHBonds[i] || new Set();
      for (const j of incoming) {
        if (!isSheet(j) || !isInterStrandPair(residues, i, j)) continue;
        const oj = residues[j].atoms.O.position;
        const ni = residues[i].atoms.N.position;
        accumulateHBondDir(vec3Sub(ni, oj));
      }

      if (hCount > 0 && vec3Length(hAccum) > 1e-6) {
        const hAvg = vec3Normalize(hAccum);
        let n = vec3Cross(t, hAvg);
        if (vec3Length(n) > 1e-6) {
          n = vec3Normalize(n);
          if (strandRef && vec3Dot(n, strandRef) < 0) n = vec3Scale(n, -1);
          if (!strandRef) strandRef = n;
          localNormals[m] = n;
        }
      }
    }

    const caPositions = members.map((i) => residues[i].atoms.CA.position);
    let fallback = fitPlaneNormal(caPositions);
    if (strandRef && vec3Dot(fallback, strandRef) < 0) {
      fallback = vec3Scale(fallback, -1);
    }

    // Fill gaps by propagating nearest known normal.
    let last = null;
    for (let m = 0; m < localNormals.length; m += 1) {
      if (localNormals[m]) {
        last = localNormals[m];
      } else if (last) {
        localNormals[m] = last;
      }
    }
    last = null;
    for (let m = localNormals.length - 1; m >= 0; m -= 1) {
      if (localNormals[m]) {
        last = localNormals[m];
      } else if (last) {
        localNormals[m] = last;
      }
    }
    for (let m = 0; m < localNormals.length; m += 1) {
      if (!localNormals[m]) {
        localNormals[m] = fallback;
      }
    }

    // Light smoothing while keeping orientation continuity.
    for (let iter = 0; iter < 2; iter += 1) {
      const next = localNormals.slice();
      for (let m = 1; m < localNormals.length - 1; m += 1) {
        let left = localNormals[m - 1];
        let center = localNormals[m];
        let right = localNormals[m + 1];
        if (vec3Dot(left, center) < 0) left = vec3Scale(left, -1);
        if (vec3Dot(right, center) < 0) right = vec3Scale(right, -1);
        const blended = vec3Normalize(vec3Add(vec3Add(vec3Scale(center, 2.0), left), right));
        if (vec3Length(blended) > 1e-6) {
          next[m] = blended;
        }
      }
      localNormals.splice(0, localNormals.length, ...next);
    }

    for (let m = 1; m < localNormals.length; m += 1) {
      if (vec3Dot(localNormals[m], localNormals[m - 1]) < 0) {
        localNormals[m] = vec3Scale(localNormals[m], -1);
      }
    }

    for (let m = 0; m < members.length; m += 1) {
      aligned[members[m]] = localNormals[m];
    }
  }

  return aligned;
}

function isInterStrandPair(residues, i, j) {
  return residues[i].chainId !== residues[j].chainId || Math.abs(j - i) > 2;
}

function makeIncomingHBonds(hbonds, residueCount) {
  const incoming = Array.from({ length: residueCount }, () => new Set());
  for (let i = 0; i < hbonds.length; i += 1) {
    const partners = hbonds[i];
    if (!partners) continue;
    for (const j of partners) {
      if (j >= 0 && j < residueCount) {
        incoming[j].add(i);
      }
    }
  }
  return incoming;
}

function angleBetweenNormalAndBondDeg(normal, direction) {
  const n = vec3Normalize(normal);
  const d = vec3Normalize(direction);
  if (vec3Length(n) < 1e-6 || vec3Length(d) < 1e-6) {
    return null;
  }
  const cosAbs = clamp(Math.abs(vec3Dot(n, d)), 0, 1);
  return Math.acos(cosAbs) * 180 / Math.PI;
}

function computeSheetStrandDiagnostics(residues, ss, hbonds, sheetNormals, segments, options = {}) {
  const diagnostics = [];
  const { directedContacts } = collectSheetDistanceContacts(residues, ss, options);

  const sheetSegments = segments.filter((segment) => segment.type === "E");
  for (let strandIndex = 0; strandIndex < sheetSegments.length; strandIndex += 1) {
    const segment = sheetSegments[strandIndex];
    const residueIndices = segment.residues.map((residue) => residue.index);
    const residueSet = new Set(residueIndices);
    const strandNormal = residueIndices
      .map((idx) => sheetNormals[idx])
      .find((n) => Boolean(n)) || null;

    let outgoingCount = 0;
    let incomingCount = 0;
    const partnerResidues = new Set();
    const angles = [];

    for (const contact of directedContacts) {
      const fromInside = residueSet.has(contact.from);
      const toInside = residueSet.has(contact.to);
      if (fromInside === toInside) continue;

      if (fromInside) {
        outgoingCount += 1;
        partnerResidues.add(contact.to);
      } else {
        incomingCount += 1;
        partnerResidues.add(contact.from);
      }

      const insideIdx = fromInside ? contact.from : contact.to;
      const normal = sheetNormals[insideIdx] || strandNormal;
      if (normal) {
        const angle = angleBetweenNormalAndBondDeg(normal, contact.direction);
        if (angle !== null) angles.push(angle);
      }
    }

    let angleMean = null;
    let angleMin = null;
    let angleMax = null;
    if (angles.length > 0) {
      angleMin = angles[0];
      angleMax = angles[0];
      let sum = 0;
      for (const angle of angles) {
        sum += angle;
        if (angle < angleMin) angleMin = angle;
        if (angle > angleMax) angleMax = angle;
      }
      angleMean = sum / angles.length;
    }

    diagnostics.push({
      strandIndex,
      chainId: segment.residues[0]?.chainId || "?",
      startSeq: segment.residues[0]?.resSeq ?? null,
      endSeq: segment.residues[segment.residues.length - 1]?.resSeq ?? null,
      residueCount: segment.residues.length,
      outgoingCount,
      incomingCount,
      totalCount: outgoingCount + incomingCount,
      partnerResidueCount: partnerResidues.size,
      angleCount: angles.length,
      angleMean,
      angleMin,
      angleMax
    });
  }

  return diagnostics;
}

function resampleSegmentNormals(segmentResidues, residueNormals, sampleCount) {
  const count = segmentResidues.length;
  if (count === 0 || sampleCount <= 0) return [];

  const base = segmentResidues.map((residue) => {
    const n = residueNormals[residue.index] || residue.planeNormal || [0, 1, 0];
    return vec3Normalize(n);
  });

  for (let i = 1; i < base.length; i += 1) {
    if (vec3Dot(base[i], base[i - 1]) < 0) {
      base[i] = vec3Scale(base[i], -1);
    }
  }

  if (sampleCount === 1) {
    return [base[0]];
  }
  if (sampleCount === base.length) {
    return base.slice();
  }

  const out = new Array(sampleCount);
  for (let s = 0; s < sampleCount; s += 1) {
    const u = (s / (sampleCount - 1)) * (base.length - 1);
    const i0 = Math.floor(u);
    const i1 = Math.min(base.length - 1, i0 + 1);
    const t = u - i0;

    const n0 = base[i0];
    let n1 = base[i1];
    if (vec3Dot(n0, n1) < 0) {
      n1 = vec3Scale(n1, -1);
    }

    let n = vec3Blend(n0, n1, t);
    if (vec3Length(n) < 1e-6) {
      n = n0;
    } else {
      n = vec3Normalize(n);
    }

    if (s > 0 && vec3Dot(n, out[s - 1]) < 0) {
      n = vec3Scale(n, -1);
    }
    out[s] = n;
  }

  return out;
}

function emitSheetDiagnostics(debugLog, diagnostics) {
  const logger = typeof debugLog === "function" ? debugLog : (msg) => console.log(msg);
  logger(`[cartoon] Sheet strand diagnostics (${diagnostics.length} strands)`);
  for (const diag of diagnostics) {
    const angleText = diag.angleCount > 0
      ? `angle(deg) mean=${diag.angleMean.toFixed(1)} min=${diag.angleMin.toFixed(1)} max=${diag.angleMax.toFixed(1)}`
      : "angle(deg) n/a";
    logger(
      `[cartoon] Strand ${diag.strandIndex + 1} ` +
      `${diag.chainId}:${diag.startSeq}-${diag.endSeq} residues=${diag.residueCount} ` +
      `hbonds total=${diag.totalCount} out=${diag.outgoingCount} in=${diag.incomingCount} ` +
      `partners=${diag.partnerResidueCount} source=distance ${angleText}`
    );
  }
}

function buildLoopTubes(residues, ss, options, out) {
  const chains = splitByChain(residues);
  const subdivisions = Math.max(1, options.loopSubdivisions);

  for (const chainResidues of chains.values()) {
    if (chainResidues.length < 2) continue;

    let start = 0;
    while (start < chainResidues.length - 1) {
      let end = start + 1;
      while (end < chainResidues.length) {
        const a = chainResidues[end - 1];
        const b = chainResidues[end];
        const gap = vec3Length(vec3Sub(b.atoms.CA.position, a.atoms.CA.position));
        if (gap > options.maxGap) break;
        end += 1;
      }

      const run = chainResidues.slice(start, end);
      if (run.length >= 2) {
        const points = run.map((r) => r.atoms.CA.position);
        const smoothPoints = resampleCatmullRom(points, subdivisions);
        const segmentMask = [];
        const runLabels = run.map((r) => ss[r.index]);

        const totalSegments = smoothPoints.length - 1;
        for (let seg = 0; seg < totalSegments; seg += 1) {
          const residueIndex = Math.min(run.length - 2, Math.floor(seg / subdivisions));
          const aLabel = runLabels[residueIndex];
          const bLabel = runLabels[residueIndex + 1];
          segmentMask.push(aLabel === "C" || bLabel === "C");
        }

        appendTubeMasked(out, smoothPoints, options.loopRadius, options.loopSides, options.colors.loop, segmentMask);
      }

      start = Math.max(end, start + 1);
    }
  }
}

function segmentByType(residues, ss, maxGap) {
  const segments = [];
  const chains = splitByChain(residues);

  for (const chainResidues of chains.values()) {
    let start = 0;
    while (start < chainResidues.length) {
      const type = ss[chainResidues[start].index];
      let end = start + 1;
      while (end < chainResidues.length) {
        const prev = chainResidues[end - 1];
        const curr = chainResidues[end];
        const gap = vec3Length(vec3Sub(curr.atoms.CA.position, prev.atoms.CA.position));
        if (gap > maxGap) break;
        if (ss[curr.index] !== type) break;
        end += 1;
      }
      segments.push({
        type,
        residues: chainResidues.slice(start, end)
      });
      start = end;
    }
  }

  return segments;
}

function applySecondaryRanges(residues, ss, ranges, type) {
  if (!ranges || ranges.length === 0) return 0;
  let count = 0;
  for (const range of ranges) {
    const chainId = range.chainId ?? " ";
    const endChain = range.endChainId ?? chainId;
    for (const residue of residues) {
      if (residue.chainId !== chainId && residue.chainId !== endChain) continue;
      if (residue.resSeq < range.startSeq || residue.resSeq > range.endSeq) continue;
      if (ss[residue.index] !== type) {
        ss[residue.index] = type;
        count += 1;
      }
    }
  }
  return count;
}

export function buildBackboneCartoon(molData, options = {}) {
  const opts = {
    ...DEFAULT_OPTIONS,
    ...options,
    colors: {
      ...DEFAULT_OPTIONS.colors,
      ...(options.colors || {})
    }
  };

  if (!molData || !Array.isArray(molData.atoms)) {
    throw new Error("Cartoon mode requires PDB atoms.");
  }

  const residuesAll = buildResidues(molData.atoms);
  if (residuesAll.length === 0) {
    throw new Error("Cartoon mode requires PDB residue data with backbone atoms.");
  }
  const incomplete = residuesAll.filter((r) => !r.complete);
  if (incomplete.length > 0) {
    throw new Error(`Cartoon mode requires complete backbone atoms (N/CA/C/O). Missing in ${incomplete.length} residues.`);
  }
  const residues = residuesAll;
  if (residues.length < 4) {
    throw new Error("Cartoon mode requires at least four residues with backbone atoms.");
  }

  residues.forEach((residue, index) => {
    residue.index = index;
  });

  const hbonds = computeHBonds(residues, opts);
  const ss = new Array(residues.length).fill("C");
  const secondary = molData.secondary || null;
  const hasPdbSecondary = Boolean(
    secondary && (secondary.helices?.length || 0) + (secondary.sheets?.length || 0) > 0
  );
  if (hasPdbSecondary) {
    applySecondaryRanges(residues, ss, secondary.helices, "H");
    applySecondaryRanges(residues, ss, secondary.sheets, "E");
  }

  const dssp = assignSecondaryStructure(residues, opts, hbonds);
  for (let i = 0; i < ss.length; i += 1) {
    if (ss[i] === "C" && dssp[i] !== "C") {
      ss[i] = dssp[i];
    }
  }

  const adjusted = hasPdbSecondary ? ss : adjustShortSegments(residues, ss);
  const sheetNormals = computeSheetNormals(residues, adjusted, hbonds, opts);
  const segments = segmentByType(residues, adjusted, opts.maxGap);
  const sheetDiagnostics = computeSheetStrandDiagnostics(residues, adjusted, hbonds, sheetNormals, segments, opts);

  if (opts.debugSheetOrientation) {
    emitSheetDiagnostics(opts.debugLog, sheetDiagnostics);
  }

  const out = {
    positions: [],
    normals: [],
    indices: [],
    triColors: []
  };

  for (const segment of segments) {
    const points = segment.residues.map((r) => r.atoms.CA.position);
    if (points.length < 2) continue;

    if (segment.type === "H") {
      const smoothPoints = resampleCatmullRom(points, opts.helixSubdivisions);
      const endWidth = Math.max(0.01, opts.loopRadius * 2.0);
      const widths = makeTaperedWidths(smoothPoints.length, opts.helixWidth, endWidth, 0.2);
      const avgNormal = pickHelixNormal(segment, residues, hbonds);
      const axisDir = computeHelixAxis(smoothPoints);
      const axisOrigin = computeCentroid(smoothPoints);
      const targetNormals = [];
      let prevRadial = null;
      for (const p of smoothPoints) {
        const rel = vec3Sub(p, axisOrigin);
        const proj = vec3Scale(axisDir, vec3Dot(rel, axisDir));
        let radial = vec3Sub(rel, proj);
        if (vec3Length(radial) < 1e-6) {
          radial = avgNormal;
        }
        radial = vec3Normalize(radial);
        if (prevRadial && vec3Dot(radial, prevRadial) < 0) {
          radial = vec3Scale(radial, -1);
        }
        prevRadial = radial;
        targetNormals.push(radial);
      }
      appendRibbonVolume(
        out,
        smoothPoints,
        avgNormal,
        widths,
        opts.helixThickness,
        {
          top: opts.colors.helixFront,
          bottom: opts.colors.helixBack,
          side: opts.colors.helixFront,
          edgeWidthScale: opts.helixEdgeWidthScale,
          edgeProfileSegments: opts.helixCrossSectionSegments
        },
        targetNormals,
        0.95
      );
    } else if (segment.type === "E") {
      // Use CA positions, smooth them (keep endpoints fixed), then resample
      let sheetPoints = smoothPointsFixedEndpoints(points, 2, 0.6);
      if (opts.sheetSubdivisions > 1) {
        sheetPoints = resampleCatmullRom(sheetPoints, opts.sheetSubdivisions);
      }

      const targetNormals = resampleSegmentNormals(segment.residues, sheetNormals, sheetPoints.length);
      const sheetNormal = targetNormals[0] || [0, 1, 0];
      const arrowTail = trimPolylineTail(sheetPoints, targetNormals, opts.arrowLength);
      const bodyPoints = arrowTail?.bodyPoints || sheetPoints;
      const bodyNormals = arrowTail?.bodyVectors || targetNormals;
      const bodyWidths = new Array(bodyPoints.length).fill(opts.sheetWidth);

      // Draw ribbon with local target normals derived from local inter-strand contact axes.
      appendRibbonVolume(
        out,
        bodyPoints,
        sheetNormal,
        bodyWidths,
        opts.sheetThickness,
        {
          top: opts.colors.sheet,
          bottom: opts.colors.sheet,
          side: opts.colors.sheet
        },
        bodyNormals,
        0.95
      );

      // Arrow at the end: strand body is truncated so the arrowhead is the terminal geometry.
      if (bodyPoints.length >= 2) {
        const baseCenter = arrowTail?.basePoint || bodyPoints[bodyPoints.length - 1];
        const lastT = arrowTail?.baseTangent || computeTangents(bodyPoints)[bodyPoints.length - 1];
        const arrowLength = arrowTail?.arrowLength || opts.arrowLength;
        const endNormal = bodyNormals[bodyNormals.length - 1] || sheetNormal;
        let binormal = vec3Cross(lastT, endNormal);
        if (vec3Length(binormal) < 1e-6) {
          binormal = pickPerpendicular(lastT);
        } else {
          binormal = vec3Normalize(binormal);
        }
        const tip = vec3Add(baseCenter, vec3Scale(lastT, arrowLength));
        const baseHalf = (opts.sheetWidth * opts.arrowBaseScale) * 0.5;
        const baseLeft = vec3Sub(baseCenter, vec3Scale(binormal, baseHalf));
        const baseRight = vec3Add(baseCenter, vec3Scale(binormal, baseHalf));
        appendTrianglePrism(out, baseLeft, baseRight, tip, endNormal, opts.sheetThickness, opts.colors.sheet);
      }
    }
  }

  buildLoopTubes(residues, adjusted, opts, out);

  const mesh = {
    positions: new Float32Array(out.positions),
    normals: new Float32Array(out.normals),
    indices: new Uint32Array(out.indices),
    triColors: new Float32Array(out.triColors)
  };

  if (opts.debugSheetOrientation) {
    mesh.sheetDiagnostics = sheetDiagnostics;
  }

  return mesh;
}

export function buildSheetHbondCylinders(molData, options = {}) {
  const opts = {
    ...DEFAULT_OPTIONS,
    ...options,
    hbondRadius: options.hbondRadius ?? 0.06,
    hbondColor: options.hbondColor ?? [0.2, 0.7, 1.0],
    normalRadius: options.normalRadius ?? 0.05,
    normalLength: options.normalLength ?? 1.5,
    normalColor: options.normalColor ?? [0.95, 0.75, 0.2]
  };

  if (!molData || !Array.isArray(molData.atoms)) {
    throw new Error("Cartoon mode requires PDB atoms.");
  }

  const residuesAll = buildResidues(molData.atoms);
  if (residuesAll.length === 0) {
    throw new Error("Cartoon mode requires PDB residue data with backbone atoms.");
  }
  const incomplete = residuesAll.filter((r) => !r.complete);
  if (incomplete.length > 0) {
    throw new Error(`Cartoon mode requires complete backbone atoms (N/CA/C/O). Missing in ${incomplete.length} residues.`);
  }

  const residues = residuesAll;
  residues.forEach((residue, index) => {
    residue.index = index;
  });

  const hbonds = computeHBonds(residues, opts);
  const ss = new Array(residues.length).fill("C");
  const secondary = molData.secondary || null;
  const hasPdbSecondary = Boolean(
    secondary && (secondary.helices?.length || 0) + (secondary.sheets?.length || 0) > 0
  );
  if (hasPdbSecondary) {
    applySecondaryRanges(residues, ss, secondary.helices, "H");
    applySecondaryRanges(residues, ss, secondary.sheets, "E");
  }

  const dssp = assignSecondaryStructure(residues, opts, hbonds);
  for (let i = 0; i < ss.length; i += 1) {
    if (ss[i] === "C" && dssp[i] !== "C") {
      ss[i] = dssp[i];
    }
  }

  const adjusted = hasPdbSecondary ? ss : adjustShortSegments(residues, ss);
  const isSheet = (i) => adjusted[i] === "E";
  const sheetNormals = computeSheetNormals(residues, adjusted, hbonds, opts);
  const { directedContacts } = collectSheetDistanceContacts(residues, adjusted, opts);

  const cylinders = [];
  for (const contact of directedContacts) {
    const from = residues[contact.from];
    const to = residues[contact.to];
    const o = from.atoms.O.position;
    const n = to.atoms.N.position;
    cylinders.push({
      p1: [o[0], o[1], o[2]],
      p2: [n[0], n[1], n[2]],
      radius: opts.hbondRadius,
      color: opts.hbondColor
    });
  }

  for (let i = 0; i < residues.length; i += 1) {
    if (!isSheet(i)) continue;
    const n = sheetNormals[i];
    if (!n) continue;
    const ca = residues[i].atoms.CA.position;
    const tip = vec3Add(ca, vec3Scale(n, opts.normalLength));
    cylinders.push({
      p1: [ca[0], ca[1], ca[2]],
      p2: [tip[0], tip[1], tip[2]],
      radius: opts.normalRadius,
      color: opts.normalColor
    });
  }

  return cylinders;
}

export function __test__buildResidues(atoms) {
  return buildResidues(atoms);
}

export function __test__assignSecondaryStructure(residues, options = {}) {
  return assignSecondaryStructure(residues, { ...DEFAULT_OPTIONS, ...options });
}

export function __test__computeSheetNormals(residues, ss, hbonds, options = {}) {
  return computeSheetNormals(residues, ss, hbonds, { ...DEFAULT_OPTIONS, ...options });
}

export function __test__computeSheetStrandDiagnostics(residues, ss, hbonds, sheetNormals, segments, options = {}) {
  return computeSheetStrandDiagnostics(residues, ss, hbonds, sheetNormals, segments, { ...DEFAULT_OPTIONS, ...options });
}

export function __test__trimPolylineTail(points, vectors, tailLength, minBodyLength = 0.6) {
  return trimPolylineTail(points, vectors, tailLength, minBodyLength);
}

export function __test__computeRibbonHalfWidths(width, edgeWidthScale = 1.0) {
  return computeRibbonHalfWidths(width, edgeWidthScale);
}
