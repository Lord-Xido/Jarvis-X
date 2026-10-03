# Multimodal Evidence Retrieval Runtime

Jarvis-X treats retrieval as a closed-loop evidence-acquisition subsystem rather
than a one-shot text augmentation step.

## Canonical flow

```text
query / world state
        |
        v
multimodal candidate retrieval
        |
        v
segment localization
(time / 2D region / 3D region)
        |
        v
shared evidence embedding
        |
        v
3D latent state Z
        |
        v
inward fixed-point refinement
        |
        v
CTR verification
        |
        +--> accepted evidence bundle
        |
        +--> uncertainty remains
                  |
                  v
          target missing modality
                  |
                  +---------------------> retrieve again
```

The supported evidence domains are:

- text
- audio
- video
- image
- 3D
- code
- sensor streams
- structured records

The reference runtime uses deterministic local embeddings so its behavior can be
tested without network dependencies. Production systems should replace or
augment that reference embedding with modality-native encoders and external
retrieval adapters.

## Evidence address

One evidence unit is not merely a document. It is an addressable segment:

[
e_i =
(o_i,; m_i,; 	au_i,; r_i^{2D},; r_i^{3D},; c_i,; p_i,; z_i)
]

where:

- (o_i) is the source object,
- (m_i) is modality,
- (	au_i) is an optional time interval,
- (r_i^{2D}) is an optional image/video region,
- (r_i^{3D}) is an optional volumetric region,
- (c_i) is confidence,
- (p_i) is provenance,
- (z_i) is an optional native feature vector.

This allows a query to retrieve, for example, an audio interval rather than only
a catalogue page that describes the audio.

## Shared retrieval

For query (q) and evidence (e_i),

[
s_i = cos(E_q(q), E_m(e_i)).
]

Production adapters may implement different native encoders (E_m), provided
they expose evidence in a comparable retrieval space.

## Inward 3D reasoning

Retrieved evidence is compressed into a three-dimensional coordination state:

[
Z_0 = E_{3D}(q,{e_i}).
]

The reference inward loop is:

[
Z_{k+1}
=
(1-lambda)Z_k
+
lambda F(Z_k,{e_i}),
]

and terminates when:

[
lVert Z_{k+1}-Z_kVert < epsilon.
]

The 3D state is a coordination geometry, not a claim that all semantic
information can be losslessly represented by three scalars. Production models
can retain rich (d)-dimensional evidence features while projecting their
routing/control state into 3D.

## CTR verification

The runtime computes an evidence receipt across four signals:

[
C =
w_m C_{	ext{modality}}
+
w_p C_{	ext{provenance}}
+
w_a C_{	ext{agreement}}
+
w_c C_{	ext{confidence}}.
]

Uncertainty is:

[
U = 1-C.
]

The result is accepted only when required modalities are present and:

[
U le U_{max}.
]

The operating invariant remains:

[
	ext{Generate}
ightarrow
	ext{Contrast}
ightarrow
	ext{Reckon}
ightarrow
	ext{Verify}
ightarrow
	ext{Correct}.
]

If uncertainty remains, correction means acquiring new evidence rather than
inventing missing media content.

## Native media adapter boundary

The reference runtime does **not** claim that metadata equals media access.

Production adapters are responsible for operations such as:

- audio byte retrieval and decoding,
- ASR / singing-voice recognition,
- audio event and music embeddings,
- video frame and temporal indexing,
- image vision encoders and region localization,
- volumetric / point-cloud encoders,
- repository-aware code retrieval,
- sensor stream ingestion.

Adapters populate `EvidenceSegment` instances. The core then performs shared
retrieval, fusion, 3D inward refinement, provenance accounting, CTR verification,
and uncertainty-driven acquisition.

## Example

```bash
python -m jarvisx.multimodal_evidence_runtime \
  --output artifacts/multimodal-evidence-runtime.json
```

The bundled demo deliberately mirrors the retrieval failure mode that motivated
this layer: catalogue text alone is not treated as equivalent to the underlying
audio. The evidence bundle can carry a localized audio interval and an aligned
video interval separately from textual metadata.

## Claim boundary

Implemented and testable:

- multimodal evidence object model,
- temporal / 2D / 3D localization metadata,
- deterministic shared retrieval,
- 3D latent fixed-point refinement,
- provenance-aware CTR receipts,
- explicit required-modality coverage,
- uncertainty-triggered re-retrieval,
- deterministic JSON execution receipts.

Not implied by this reference implementation:

- universal access to external media,
- automatic copyright bypass,
- production-grade ASR/OCR/video/3D understanding,
- correctness of third-party evidence,
- AGI.

The purpose of this layer is to make those capabilities pluggable while keeping
the reasoning and verification contract explicit.
