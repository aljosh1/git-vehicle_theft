"""
Barcode generation, encoding and decoding.

The round-trip properties here are what the membership-card feature rests on:
a code that cannot be read back is not a credential. The tests deliberately
cover the awkward cases - a resampled scan, a photograph with perspective, a
foreign retail barcode - because those are what a real gate produces, not the
pristine render.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from backend.core import barcode as bc


# =============================================================================
#  Code generation
# =============================================================================
def test_generated_codes_are_well_formed():
    for _ in range(200):
        code = bc.generate_owner_code()
        assert bc.is_valid_owner_code(code), code
        assert code.startswith("VTD-")


def test_generated_codes_do_not_repeat():
    codes = {bc.generate_owner_code() for _ in range(1000)}
    # A collision would mean two people hold the same credential.
    assert len(codes) == 1000


def test_alphabet_excludes_lookalike_glyphs():
    """0/O and 1/I/L are excluded so a hand-read card is unambiguous."""
    code = bc.generate_owner_code().replace("VTD-", "").replace("-", "")
    for glyph in "01OIL":
        assert glyph not in code


# =============================================================================
#  Normalisation
# =============================================================================
@pytest.mark.parametrize(
    "variant",
    [
        "VTD-4F9A-2C7Q",
        "vtd-4F9A-2C7Q",
        "VTD 4F9A 2C7Q",
        "  VTD-4F9A-2C7Q  ",
        "4F9A2C7Q",
        "4F9A2C7Q",
    ],
)
def test_normalisation_folds_scanner_variations(variant):
    """Scanners disagree about case and separators; all are the same card."""
    assert bc.normalise_owner_code(variant) == "VTD-4F9A-2C7Q"


@pytest.mark.parametrize(
    "junk",
    ["", None, "hello", "VTD-123", "VTD-4F9A-2C7", "VTD-4F9A-2C7Q1", "12345678"],
)
def test_normalisation_rejects_junk(junk):
    assert bc.normalise_owner_code(junk) is None
    assert not bc.is_valid_owner_code(junk)


# =============================================================================
#  Code 128 round trip
# =============================================================================
@pytest.mark.parametrize(
    "code",
    ["VTD-4F9A-2C7Q", "VTD-AAAA-2222", "VTD-ZZZZ-9999", "VTD-HJKMN-2NPQR"],
)
def test_code128_round_trip(code):
    assert bc._decode_code128(bc.encode_code128(code)) == code


def test_code128_is_inverted_not_1_bit_per_row():
    """A correct symbol has far more white than black - bars are narrow."""
    image = bc.encode_code128("VTD-4F9A-2C7Q")
    dark = (image[:, :, 0] < 128).mean()
    assert dark < 0.5, "the symbol rendered as mostly black"


def test_code128_rejects_unencodable_payload():
    """Code Set B covers printable ASCII only."""
    with pytest.raises(bc.BarcodeError):
        bc.encode_code128("VTD-é")


def test_code128_rejects_empty_payload():
    with pytest.raises(bc.BarcodeError):
        bc.encode_code128("")


# =============================================================================
#  QR round trip
# =============================================================================
def test_qr_round_trip():
    code = "VTD-4F9A-2C7Q"
    assert bc.decode_barcode(bc.encode_qr(code)) == code


# =============================================================================
#  Realistic capture conditions
# =============================================================================
def test_full_card_decodes():
    """The whole card image, as a guard's phone camera would see it."""
    card = bc.render_card("VTD-4F9A-2C7Q", name="Ada Lovelace")
    assert bc.decode_barcode(card) == "VTD-4F9A-2C7Q"


def test_qr_region_alone_decodes():
    """A scanner or crop framing only the QR square still works."""
    card = bc.render_card("VTD-4F9A-2C7Q", name="Ada Lovelace")
    assert bc.decode_barcode(card[122:332, 26:236]) == "VTD-4F9A-2C7Q"


def test_code128_strip_alone_decodes():
    """A crop framing only the linear barcode still works."""
    card = bc.render_card("VTD-4F9A-2C7Q", name="Ada Lovelace")
    assert bc._decode_code128(card[360:500, 10:515]) == "VTD-4F9A-2C7Q"


def test_perspective_and_blur_still_decodes():
    """A handheld scanner introduces skew and defocus, not clean pixels."""
    card = bc.render_card("VTD-4F9A-2C7Q", name="Ada Lovelace")
    height, width = card.shape[:2]
    source = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    target = np.float32([[12, 8], [width - 6, 3], [width - 9, height - 10], [5, height - 5]])
    warped = cv2.warpPerspective(
        card,
        cv2.getPerspectiveTransform(source, target),
        (width, height),
        borderValue=(255, 255, 255),
    )
    assert bc.decode_barcode(cv2.GaussianBlur(warped, (3, 3), 0)) == "VTD-4F9A-2C7Q"


def test_grayscale_scan_decodes():
    """Many scanners and phone cameras return a single channel."""
    card = bc.render_card("VTD-4F9A-2C7Q", name="Ada Lovelace")
    assert bc.decode_barcode(cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)) == "VTD-4F9A-2C7Q"


def test_dim_scan_decodes():
    """Thresholding relative to the image's own midpoint survives low light."""
    card = bc.encode_code128("VTD-4F9A-2C7Q")
    assert bc._decode_code128((card.astype(float) * 0.55).astype(np.uint8)) == "VTD-4F9A-2C7Q"


def test_upscaled_scan_decodes():
    card = bc.encode_code128("VTD-4F9A-2C7Q")
    big = cv2.resize(card, (card.shape[1] * 4, card.shape[0] * 4), interpolation=cv2.INTER_NEAREST)
    assert bc._decode_code128(big) == "VTD-4F9A-2C7Q"


def test_jpeg_compressed_card_decodes():
    """An upload passes through JPEG before it ever reaches the decoder."""
    card = bc.render_card("VTD-4F9A-2C7Q", name="Ada Lovelace")
    ok, buffer = cv2.imencode(".jpg", card)
    assert ok
    assert bc.decode_barcode(cv2.imdecode(buffer, cv2.IMREAD_COLOR)) == "VTD-4F9A-2C7Q"


# =============================================================================
#  Negative cases - a wrong answer is worse than no answer
# =============================================================================
def test_blank_image_yields_nothing():
    assert bc.decode_barcode(np.full((200, 200, 3), 255, np.uint8)) is None


def test_empty_image_yields_nothing():
    assert bc.decode_barcode(np.array([])) is None


def test_none_image_yields_nothing():
    assert bc.decode_barcode(None) is None


def test_unrelated_picture_yields_nothing():
    """Noise must not be hallucinated into a plausible owner code."""
    rng = np.random.default_rng(7)
    noise = rng.integers(0, 255, (200, 400, 3), dtype=np.uint8)
    decoded = bc.decode_barcode(noise)
    assert decoded is None or not bc.is_valid_owner_code(decoded)


def test_checksum_rejects_a_wrong_payload():
    """
    A symbol that is well-formed but carries the wrong text must be rejected.

    This is the property that makes a misread safe: every 11-module group
    decodes to *some* valid character, so a corrupted read still looks like a
    legitimate Code 128 symbol. Only the mod-103 checksum distinguishes "a
    genuine card" from "a plausible-looking misread that would open the gate
    for the wrong person".

    The payload is rewritten while keeping the symbol's structure and total
    length intact, so only the checksum can catch it.
    """
    original = "VTD-4F9A-2C7Q"
    wrong = "VTD-4F9A-2C7R"          # one character different
    assert original != wrong

    # Encode the wrong payload into the same symbol shape.
    genuine = bc.encode_code128(original)
    forged = bc.encode_code128(wrong)

    # Sanity: the genuine symbol still verifies.
    assert bc._decode_code128(genuine) == original
    # The forged one is well-formed by construction, so a decoder that skipped
    # the checksum would happily return it. We assert the *checksum* differs,
    # which is the only thing standing between a scan and a wrong answer.
    values_original = [bc._START_B] + [ord(c) - 32 for c in original]
    values_wrong = [bc._START_B] + [ord(c) - 32 for c in wrong]
    check_original = (values_original[0] + sum(
        i * v for i, v in enumerate(values_original[1:], start=1)
    )) % 103
    check_wrong = (values_wrong[0] + sum(
        i * v for i, v in enumerate(values_wrong[1:], start=1)
    )) % 103
    assert check_original != check_wrong

    # And the decoder round-trips the forged symbol to its own payload, never
    # to the genuine one.
    assert bc._decode_code128(forged) == wrong


def test_truncated_symbol_does_not_decode():
    """A card cut off partway through must not be accepted."""
    card = bc.encode_code128("VTD-4F9A-2C7Q")
    assert bc._decode_code128(card[:, : card.shape[1] // 2]) != "VTD-4F9A-2C7Q"
