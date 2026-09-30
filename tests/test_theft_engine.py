"""
Tests for the threat scoring rules.

The theft engine is deliberately I/O-free, so these run in milliseconds and
cover the scenarios an evaluator will ask about: does each specified condition
fire, do combinations escalate, and - most importantly - is a recognised owner
NOT alerted on.
"""

from __future__ import annotations

import pytest

from backend.config import settings
from backend.core.theft_engine import (
    TRIGGER_FLAGGED_STOLEN,
    TRIGGER_OWNER_PRESENT,
    TRIGGER_PLATE_NOT_REGISTERED,
    TRIGGER_UNAUTHORIZED_FACE,
    TRIGGER_UNKNOWN_PERSON,
    LoiterTracker,
    TheftEngine,
    is_night_time,
    score_to_level,
)
from backend.database.models import OwnerVerificationStatus, PersonStatus, ThreatLevel
from backend.detection.base import Detection


def vehicle_at(x1=100, y1=100, x2=400, y2=300) -> Detection:
    return Detection(label="car", confidence=0.95, x1=x1, y1=y1, x2=x2, y2=y2)


def person_at(x1=380, y1=120, x2=450, y2=320, track_id=1) -> Detection:
    return Detection(
        label="person", confidence=0.90, x1=x1, y1=y1, x2=x2, y2=y2, track_id=track_id
    )


@pytest.fixture()
def engine() -> TheftEngine:
    return TheftEngine()


# =============================================================================
#  Score banding
# =============================================================================
@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0, ThreatLevel.NONE),
        (19, ThreatLevel.NONE),
        (20, ThreatLevel.LOW),
        (39, ThreatLevel.LOW),
        (40, ThreatLevel.MEDIUM),
        (59, ThreatLevel.MEDIUM),
        (60, ThreatLevel.HIGH),
        (79, ThreatLevel.HIGH),
        (80, ThreatLevel.CRITICAL),
        (100, ThreatLevel.CRITICAL),
    ],
)
def test_score_bands(score, expected):
    assert score_to_level(score) is expected


# =============================================================================
#  Baseline: nothing happening
# =============================================================================
def test_empty_frame_scores_zero(engine):
    result = engine.assess(vehicles=[], persons=[])
    assert result.score == 0
    assert result.level is ThreatLevel.NONE
    assert not result.should_alert


def test_registered_vehicle_alone_is_not_a_threat(engine):
    result = engine.assess(
        vehicles=[vehicle_at()],
        persons=[],
        plate_number="ABC123XY",
        plate_in_database=True,
        matched_vehicle_id=1,
    )
    assert result.score == 0
    assert not result.should_alert


# =============================================================================
#  Condition 2 - plate not in the database
# =============================================================================
def test_unregistered_plate_alone_does_not_alert(engine):
    """A visitor's car is not a theft. It should register, but stay below the bar."""
    result = engine.assess(
        vehicles=[vehicle_at()],
        persons=[],
        plate_number="ZZZ999AA",
        plate_in_database=False,
    )
    assert TRIGGER_PLATE_NOT_REGISTERED in result.triggers
    assert result.score == 35
    assert result.level is ThreatLevel.LOW
    assert not result.should_alert


def test_unreadable_plate_is_weaker_than_unregistered(engine):
    """"Could not read" must not be treated as "not registered"."""
    unreadable = engine.assess(vehicles=[vehicle_at()], persons=[])
    unregistered = engine.assess(
        vehicles=[vehicle_at()], persons=[], plate_number="ZZZ1", plate_in_database=False
    )
    assert unreadable.score < unregistered.score


# =============================================================================
#  Condition 1 - unknown person near a vehicle
# =============================================================================
def test_person_beside_vehicle_is_detected_as_near(engine):
    """Zero box overlap, but obviously at the car - distance must catch it."""
    result = engine.assess(
        vehicles=[vehicle_at()],
        persons=[person_at()],
        plate_number="ABC123XY",
        plate_in_database=True,       # isolate the proximity trigger
        matched_vehicle_id=1,
    )
    assert TRIGGER_UNKNOWN_PERSON in result.triggers
    assert result.score == 40


def test_person_far_from_vehicle_is_not_near(engine):
    far_person = person_at(x1=1500, y1=800, x2=1560, y2=1000)
    result = engine.assess(vehicles=[vehicle_at()], persons=[far_person])
    assert TRIGGER_UNKNOWN_PERSON not in result.triggers


def test_person_alone_without_a_vehicle_is_not_a_threat(engine):
    result = engine.assess(vehicles=[], persons=[person_at()])
    assert result.score == 0


# =============================================================================
#  Condition 3 - unauthorised face
# =============================================================================
def test_unauthorized_face_scores(engine):
    result = engine.assess(
        vehicles=[],
        persons=[person_at()],
        person_status=PersonStatus.UNAUTHORIZED,
        face_similarity=0.31,
    )
    assert TRIGGER_UNAUTHORIZED_FACE in result.triggers
    assert result.score == 45
    assert "31%" in result.reason


def test_no_face_visible_is_not_unauthorized(engine):
    """Overhead CCTV rarely sees faces; that must not imply guilt."""
    result = engine.assess(
        vehicles=[], persons=[person_at()], person_status=PersonStatus.NO_FACE
    )
    assert TRIGGER_UNAUTHORIZED_FACE not in result.triggers
    assert result.score == 0


# =============================================================================
#  Combinations must escalate
# =============================================================================
def test_stranger_at_unregistered_vehicle_alerts(engine):
    """The realistic theft scenario: 40 + 35 = 75 -> HIGH."""
    result = engine.assess(
        vehicles=[vehicle_at()],
        persons=[person_at()],
        plate_number="ZZZ999AA",
        plate_in_database=False,
    )
    assert result.score == 75
    assert result.level is ThreatLevel.HIGH
    assert result.should_alert


def test_unauthorized_face_at_registered_vehicle_alerts(engine):
    """Someone else's face at a known car: 40 + 45 = 85 -> CRITICAL."""
    result = engine.assess(
        vehicles=[vehicle_at()],
        persons=[person_at()],
        plate_number="ABC123XY",
        plate_in_database=True,
        matched_vehicle_id=1,
        person_status=PersonStatus.UNAUTHORIZED,
        face_similarity=0.22,
    )
    assert result.score == 85
    assert result.level is ThreatLevel.CRITICAL
    assert result.should_alert


def test_different_registered_owner_has_specific_explanation(engine):
    result = engine.assess(
        vehicles=[vehicle_at()],
        persons=[person_at()],
        plate_number="ABC123XY",
        plate_in_database=True,
        matched_vehicle_id=1,
        person_status=PersonStatus.UNAUTHORIZED,
        face_similarity=0.86,
        matched_user_id=9,
        owner_verification=OwnerVerificationStatus.MISMATCH,
    )

    assert TRIGGER_UNAUTHORIZED_FACE in result.triggers
    assert "does not belong to this vehicle's owner" in result.reason
    assert "86%" in result.reason


def test_score_is_clamped_to_100(engine):
    result = engine.assess(
        vehicles=[vehicle_at()],
        persons=[person_at(), person_at(track_id=2), person_at(track_id=3)],
        plate_number="ZZZ999AA",
        plate_in_database=False,
        vehicle_flagged_stolen=True,
        person_status=PersonStatus.UNAUTHORIZED,
        face_similarity=0.1,
        is_night=True,
    )
    assert result.score == 100


# =============================================================================
#  Owner suppression - the most important behaviour in the module
# =============================================================================
def test_recognised_owner_suppresses_the_alert(engine):
    """
    An owner collecting their own car at night with a plate the OCR misread must
    NOT trigger an alert. Without the -50 suppression this scores 45 and alerts.
    """
    result = engine.assess(
        vehicles=[vehicle_at()],
        persons=[person_at()],
        plate_number=None,
        plate_in_database=None,
        person_status=PersonStatus.AUTHORIZED,
        face_similarity=0.91,
        matched_user_id=7,
        is_night=True,
    )
    assert TRIGGER_OWNER_PRESENT in result.triggers
    assert not result.should_alert, "a recognised owner must never raise an alert"
    assert result.score == 0


def test_owner_present_blocks_the_proximity_trigger(engine):
    result = engine.assess(
        vehicles=[vehicle_at()],
        persons=[person_at()],
        person_status=PersonStatus.AUTHORIZED,
        face_similarity=0.88,
        matched_user_id=3,
    )
    assert TRIGGER_UNKNOWN_PERSON not in result.triggers


# =============================================================================
#  Reported-stolen override
# =============================================================================
def test_flagged_stolen_forces_critical(engine):
    """Once an owner reports the car stolen, any sighting is maximally urgent."""
    result = engine.assess(
        vehicles=[vehicle_at()],
        persons=[],
        plate_number="ABC123XY",
        plate_in_database=True,
        matched_vehicle_id=1,
        vehicle_flagged_stolen=True,
    )
    assert TRIGGER_FLAGGED_STOLEN in result.triggers
    assert result.level is ThreatLevel.CRITICAL
    assert result.should_alert


def test_flagged_stolen_outranks_owner_presence(engine):
    """
    +100 for stolen against -50 for the owner still clears the threshold. If the
    car is reported stolen, a "recognised" face does not stand it down - the
    recognition could be a sibling, or the report could be the newer fact.
    """
    result = engine.assess(
        vehicles=[vehicle_at()],
        persons=[person_at()],
        plate_in_database=True,
        matched_vehicle_id=1,
        vehicle_flagged_stolen=True,
        person_status=PersonStatus.AUTHORIZED,
        face_similarity=0.9,
        matched_user_id=1,
    )
    assert result.should_alert


# =============================================================================
#  Explainability
# =============================================================================
def test_assessment_explains_itself(engine):
    result = engine.assess(
        vehicles=[vehicle_at()],
        persons=[person_at()],
        plate_number="ZZZ999AA",
        plate_in_database=False,
    )
    assert result.reason
    assert result.breakdown[TRIGGER_PLATE_NOT_REGISTERED] == 35
    assert result.breakdown[TRIGGER_UNKNOWN_PERSON] == 40
    assert sum(result.breakdown.values()) == result.score
    assert "," in result.triggers_csv


def test_single_conditions_all_stay_below_the_alert_threshold(engine):
    """
    Design invariant: no single specified condition may alert on its own.
    If a weight is ever raised past this, the tuning rationale must be revisited.
    """
    threshold = settings.THREAT_ALERT_THRESHOLD
    plate_only = engine.assess(
        vehicles=[vehicle_at()], persons=[], plate_in_database=False
    )
    person_only = engine.assess(vehicles=[vehicle_at()], persons=[person_at()])
    face_only = engine.assess(
        vehicles=[], persons=[person_at()], person_status=PersonStatus.UNAUTHORIZED
    )
    assert plate_only.score < threshold
    assert person_only.score < threshold
    assert face_only.score < threshold


# =============================================================================
#  Loiter tracking
# =============================================================================
def test_loiter_tracker_measures_duration():
    tracker = LoiterTracker(loiter_seconds=10)
    assert tracker.update(1) == pytest.approx(0.0, abs=0.1)
    assert not tracker.is_loitering(1)


def test_loiter_tracker_ignores_untracked_persons():
    """Without tracking enabled every track_id is None; that must not crash."""
    tracker = LoiterTracker()
    assert tracker.update(None) == 0.0
    assert not tracker.is_loitering(None)


def test_loiter_tracker_prunes_stale_ids():
    tracker = LoiterTracker()
    tracker.update(1)
    tracker.prune(max_age_seconds=-1)     # force everything to look stale
    assert not tracker.is_loitering(1)


# =============================================================================
#  Night detection
# =============================================================================
@pytest.mark.parametrize(
    ("hour", "expected"),
    [(0, True), (5, True), (6, False), (12, False), (18, False), (19, True), (23, True)],
)
def test_night_time_window(hour, expected):
    assert is_night_time(hour) is expected
