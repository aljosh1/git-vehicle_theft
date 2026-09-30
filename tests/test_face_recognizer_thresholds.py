from backend.recognition import face_recognizer as face


def test_match_threshold_uses_backend_default_when_override_unset(monkeypatch) -> None:
    monkeypatch.setattr(face, "init_backend", lambda: "opencv-yunet-sface")
    monkeypatch.setattr(face, "ACTIVE_BACKEND", "opencv-yunet-sface")
    monkeypatch.setattr(face.settings, "FACE_MATCH_THRESHOLD", None)

    assert face.match_threshold() == 0.363


def test_match_threshold_respects_global_override(monkeypatch) -> None:
    monkeypatch.setattr(face, "init_backend", lambda: "opencv-yunet-sface")
    monkeypatch.setattr(face, "ACTIVE_BACKEND", "opencv-yunet-sface")
    monkeypatch.setattr(face.settings, "FACE_MATCH_THRESHOLD", 0.77)

    assert face.match_threshold() == 0.77
