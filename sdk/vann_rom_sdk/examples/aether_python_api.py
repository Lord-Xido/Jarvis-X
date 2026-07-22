from __future__ import annotations

import json

from vann_rom_sdk import AetherEngine, synthetic_aether_input


engine = AetherEngine()
result = engine.run(
    synthetic_aether_input(),
    adapt=True,
    optimize=True,
)

print(json.dumps(result.to_dict(), indent=2))
