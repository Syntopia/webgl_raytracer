# WebGL2 Ray Tracer (Molecular)

This project is a WebGL2 path tracer focused on molecular scenes (PDB/SDF/MOL), with CPU BVH build + texture-packed traversal on GPU.

## Features
- Molecular import: PDB / SDF / MOL
- Cartoon / atom / SES surface rendering paths
- Analytic sky environment + HDR environment maps
- Directional and ambient lighting controls
- Depth-of-field controls and focus picking

## Run
Serve the project with uvicorn:

```bash
mamba run -n wave uvicorn server:app --reload
```

Then open `http://localhost:8000`.

## Optional: WebAssembly SES surface
Build the WASM module:

```bash
npm run build:wasm
```

## Assets
HDR environment maps can be downloaded with:

```bash
python tools/download_envs.py
```

## Tests
```bash
mamba run -n wave node --test
```
