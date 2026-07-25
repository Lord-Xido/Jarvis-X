"""Runnable Dr Moagi M.M ROM Ω³ reference demonstration."""

from jarvisx.dr_moagi_3d import DrMoagiEngine, REQUIRED_VERIFICATION


def main() -> None:
    engine = DrMoagiEngine()
    source = b"JARVIS"
    decoded, committed, coordinate = engine.cycle(
        source,
        REQUIRED_VERIFICATION,
        candidate_loss=1,
        active_loss=2,
    )

    print("input:", source)
    print("decoded:", decoded)
    print("coordinate:", coordinate)
    print("cell address:", coordinate.cell_address)
    print("cube/local:", coordinate.cube_id, coordinate.local_address)
    print("committed:", committed)
    print("version:", engine.version)


if __name__ == "__main__":
    main()
