from backend.recognition.plate_ocr import _normalise_plate_text, fuzzy_match_plate


def test_plate_normalisation_keeps_digits_and_letters() -> None:
    assert _normalise_plate_text('ABC 123') == 'ABC123'
    assert _normalise_plate_text('lag-123-xy') == 'LAG123XY'


def test_plate_normalisation_strips_nigerian_state_banner_text() -> None:
    assert _normalise_plate_text('ABUJA ABC123DE') == 'ABC123DE'
    assert _normalise_plate_text('LAGOSXYZ987AA') == 'XYZ987AA'


def test_plate_normalisation_ignores_flag_prefix_m_noise() -> None:
    assert _normalise_plate_text('MABC123DE') == 'ABC123DE'


def test_fuzzy_plate_match_allows_small_ocr_noise() -> None:
    assert fuzzy_match_plate('ABC123X', 'ABC123XY', threshold=0.85) is True
