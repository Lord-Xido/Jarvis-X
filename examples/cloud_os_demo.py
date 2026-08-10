"""Minimal local demonstration of the Dr Moagi Cloud OS."""

from jarvisx.cloud_os import DrMoagiCloudOS, Field3D


def main() -> None:
    runtime = DrMoagiCloudOS()
    runtime.register_node("demo-node", max_cells=4096, max_concurrency=2)

    field = Field3D.from_values(
        [float((x + y + z) % 5) for z in range(4) for y in range(4) for x in range(4)],
        (4, 4, 4),
    )
    job = runtime.auto_optimize(
        field,
        request_id="demo-001",
        complexity_weight=0.05,
    )

    snapshot = runtime.job_snapshot(job.job_id)
    result = snapshot["result"]
    assert isinstance(result, dict)
    print("job:", snapshot["job_id"])
    print("node:", snapshot["node_id"])
    print("latent shape:", result["selected_latent_shape"])
    print("mse:", result["mse"])
    print("compression ratio:", result["compression_ratio"])
    print("ledger valid:", runtime.ledger.verify())


if __name__ == "__main__":
    main()
