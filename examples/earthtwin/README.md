# EarthTwin Meta-Runtime Demo

EarthTwin is a dependency-free browser demonstration of several Jarvis X runtime ideas:

- a compact binary city ROM with explicit encode/decode boundaries;
- usage-weighted ROM re-encoding for query locality;
- bounded query caching with invalidation on refinement;
- adaptive render level-of-detail driven by measured frame rate;
- observable query heat, cache efficiency, ROM size, and refinement count;
- a 3D globe interface backed by Three.js.

The demo is intentionally local and deterministic. Its chat surface is a small ROM query parser, not a general language model. The included city records are static demonstration data and should not be treated as a current demographic source.

## Run

ES modules and the import map require an HTTP origin. From the repository root:

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000/examples/earthtwin/
```

The page downloads Three.js modules and Earth textures from the public URLs declared in `index.html` and `app.js`, so first load requires network access.

## Query examples

```text
Where is Tokyo?
Time in Delhi
Show cities > 10M
Show cities < 5M
```

## Operational refinement

For city `c`, query heat is maintained as

```text
heat(c) <- heat(c) + 1
```

Every third direct query for a city triggers a deterministic ROM refinement:

```text
orderedCities <- sort(cities, key=(-heat, name))
ROM'          <- encodeROM(orderedCities)
cache         <- empty
```

Rendering quality uses an exponentially smoothed FPS measurement with hysteresis so it does not oscillate rapidly between quality levels.

## Security boundary

User and runtime messages are inserted with `textContent`; query text is never interpreted as HTML. ROM decoding performs explicit bounds and string-index validation. The content-security policy limits executable and image origins to the demo and its declared Three.js resources.
