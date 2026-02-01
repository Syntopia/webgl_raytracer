# WebGL2 Ray Tracer (SAH BVH + glTF)

This is a minimal WebGL2 ray tracer that builds a SAH BVH on the CPU, flattens it into a linear array, uploads everything into floating point textures, and performs traversal + triangle intersection in a fragment shader with accumulation.

## Features
- CPU SAH BVH build (flattened BVH layout)
- BVH + triangle data packed into RGBA32F textures
- glTF 2.0 loader for embedded buffer `.gltf`
- Fragment shader traversal with a fixed-size stack
- Ping-pong accumulation buffers

## Run
Serve the directory with a local server, for example:

```
python -m http.server
```

Then open `http://localhost:8000`.

## Example assets
Generate the embedded-buffer `.gltf` examples:

```
python tools/generate_examples.py
```

## Tests
```
npm test
```

## Notes
- The loader supports embedded `data:` buffers and external buffers when the glTF is served over HTTP.
- If you upload a local `.gltf` file with external buffers, it will fail (no path access from the browser).
- Materials are stubbed to a single ID per triangle.
- Controls: drag to orbit, mouse wheel to zoom, WASDQE to pan while rendering.
- Autorun: append `?autorun=1` to start rendering automatically, and `&example=assets/cube.gltf` to pick the example.
- If you hit GPU memory errors, lower the render scale in the UI.
- Traversal mode: switch between BVH and brute force in the UI (debugging).
