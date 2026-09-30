"""Smoke test for the recognition modules (face backend + plate OCR).

    python tests/smoke_recognition.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.recognition import face_recognizer as fr  # noqa: E402
from backend.recognition.plate_ocr import (  # noqa: E402
    _normalise_plate_text,
    fuzzy_match_plate,
)


def main() -> None:
    print("=== face backend ===")
    backend = fr.init_backend()
    print(f"active backend : {backend}")
    print(f"embedding dim  : {fr.embedding_dim()}")

    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    faces = fr.detect_faces(frame)
    print(f"detect_faces   : {len(faces)} on noise (0 expected)")

    crop = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
    emb = fr.extract_embedding(crop)
    print(f"embedding      : {None if emb is None else emb.shape}")
    if emb is not None:
        print(f"L2 norm        : {float(np.linalg.norm(emb)):.4f} (1.0 expected)")

    print("\n=== gallery matching ===")
    rng = np.random.default_rng(42)
    v1 = rng.normal(size=fr.embedding_dim()).astype(np.float32)
    v2 = rng.normal(size=fr.embedding_dim()).astype(np.float32)
    gallery = fr.FaceGallery([(1, v1), (2, v2)])
    print(f"gallery size   : {gallery.size}")

    # An exact copy of a gallery vector must match its owner.
    same = gallery.match(v1)
    print(f"self-match     : {same.status.value} owner={same.owner_id} sim={same.similarity:.3f}")
    assert same.owner_id == 1, "a vector must match itself"

    # An unrelated vector must not.
    other = gallery.match(rng.normal(size=fr.embedding_dim()).astype(np.float32))
    print(f"stranger       : {other.status.value} sim={other.similarity:.3f}")

    print("\n=== plate OCR post-processing ===")
    cases = [
        ("ABC-123XY", "ABC123XY"),
        ("abc 123 xy", "ABC123XY"),
        ("0BC123XY", "OBC123XY"),   # leading 0 -> letter O
        ("ABC12OXY", "ABC120XY"),   # trailing O -> digit 0
    ]
    for raw, expected in cases:
        got = _normalise_plate_text(raw)
        flag = "OK " if got == expected else "DIFF"
        print(f"  {flag} {raw!r:14} -> {got!r}")

    print("\n=== fuzzy plate matching ===")
    print(f"  exact          : {fuzzy_match_plate('ABC123XY', 'ABC-123XY')}")
    print(f"  one char off   : {fuzzy_match_plate('ABC123XZ', 'ABC123XY')}")
    print(f"  different plate: {fuzzy_match_plate('ZZZ999AA', 'ABC123XY')}")

    print("\nsmoke test PASSED")


if __name__ == "__main__":
    main()
