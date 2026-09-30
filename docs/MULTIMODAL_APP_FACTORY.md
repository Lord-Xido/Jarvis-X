# Multimodal Application Factory

The Moagi Multimodal Application Factory is the application-generation layer for
Jarvis-X. It complements the existing bounded 3D multimodal runtime in
`src/jarvisx/dr_moagi_multimodal_loop.py`; it does not replace that runtime.

## Purpose

The factory converts a compact application specification into a runnable Python
software project:

```text
specification
    -> project scaffold
    -> FastAPI service
    -> multimodal endpoints
    -> optional Conv3D codec
    -> compile validation
    -> tests
    -> release archive
    -> deployable application
```

Generated applications support:

- text analysis
- image inspection and transformation
- WAV audio inspection
- FFprobe-assisted video inspection
- arbitrary-file ingestion and hashing
- NumPy 3D-volume analysis
- optional PyTorch 3D autoencoding/decoding
- REST/OpenAPI access
- browser-accessible service landing page
- pytest smoke tests
- Docker packaging
- ZIP release generation

## Factory

The repository entry point is:

```text
tools/moagi_multimodal_app_factory.py
```

The factory itself uses only the Python standard library. Runtime dependencies
belong to the generated application, keeping project generation portable.

## Generate an application

```bash
python tools/moagi_multimodal_app_factory.py all jarvis_media_studio \
  --output .generated/jarvis_media_studio \
  --title "Jarvis Media Studio"
```

The generated application contains approximately:

```text
jarvis_media_studio/
├── app_spec.json
├── Dockerfile
├── README.md
├── manage.py
├── requirements.txt
├── requirements-ai.txt
├── jarvis_media_studio/
│   ├── __init__.py
│   ├── app.py
│   └── autoencoder3d.py
├── tests/
│   └── test_smoke.py
├── checkpoints/
├── outputs/
└── uploads/
```

## Run the generated application

```bash
cd .generated/jarvis_media_studio
python manage.py install
python manage.py test
python manage.py run
```

Then open:

```text
http://127.0.0.1:8000
http://127.0.0.1:8000/docs
```

Install optional neural/multimedia dependencies with:

```bash
python manage.py install --ai
```

## 3D autoencoding path

The generated optional codec implements:

```text
X -> Conv3D encoder -> Z -> inward latent refinement -> ConvTranspose3D decoder -> X_hat
```

with refinement:

```text
Z(k+1) = (1-alpha) Z(k) + alpha E(D(Z(k)))
```

This is deliberately a small application-level neural codec. Jarvis-X's
existing `dr_moagi_multimodal_loop.py` remains the repository's bounded
reference runtime for the broader multimodal 3D loop.

## Validation contract

The factory performs Python compile validation before packaging. Repository CI
then goes further by generating a fresh application, installing its declared
base dependencies, executing its test suite, creating a release ZIP, and
uploading the result as a workflow artifact.

The engineering invariant is:

```text
Generate -> Validate -> Test -> Package -> Verify
```

A generated source tree is not considered operational merely because files were
written; CI must prove that the generated application imports, serves its
health/text/3D-volume interfaces, and packages successfully.

## CI

The dedicated workflow is:

```text
.github/workflows/multimodal-app-factory.yml
```

It is path-scoped to the factory, this document, and its own workflow definition
so unrelated repository changes do not trigger unnecessary application builds.

## Relationship to Jarvis-X

The layering is:

```text
Jarvis-X runtime / codec primitives
              ^
              |
generated multimodal application
              ^
              |
Moagi application factory
              ^
              |
application specification
```

This keeps the software-development concern separate from the lower-level
runtime concern while allowing generated applications to evolve toward deeper
Jarvis-X integration without duplicating or deleting existing runtime code.
