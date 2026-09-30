"""
Armed and masked intruder detection.

The specification's third trigger - an unauthorised person attempting vehicle
access - is only actionable if the system can tell that the person is *armed*
or is *hiding their face*. These tests pin the two properties that make that
work, and the one property that stops it crying wolf:

1. A weapon is decisive on its own, with no vehicle in shot.
2. A weapon defeats the owner-present suppression. This is the single most
   dangerous line of code in the engine: get it wrong and a carjacking scores
   50, never alerts, and the guard watches it happen.
3. Everyday objects - a crowbar, a dark corridor - must not read as an armed
   assault, or the system is switched off within a week.
"""

from __future__ import annotations

import pytest

from backend.core.theft_engine import (
    TRIGGER_FACE_NOT_VISIBLE,
    TRIGGER_FACE_UNMATCHED,
    TRIGGER_OCCLUDED_FACE,
    TRIGGER_FIREARM,
    TRIGGER_MASK,
    TRIGGER_OWNER_PRESENT,
    TRIGGER_THREATENING_TOOL,
    TRIGGER_WEAPON,
    TheftEngine,
    WEIGHT_FACE_NOT_VISIBLE,
    WEIGHT_FIREARM,
    WEIGHT_MASK,
    WEIGHT_OCCLUDED_FACE,
    WEIGHT_THREATENING_TOOL,
    WEIGHT_WEAPON,
)
from backend.database.models import PersonStatus
from backend.detection.base import Detection

# =============================================================================
#  Fixtures / helpers
# =============================================================================
@pytest.fixture()
def engine() -> TheftEngine:
    return TheftEngine()


def person(track_id: int | None = None, box=(100, 100, 200, 400)) -> Detection:
    x1, y1, x2, y2 = box
    return Detection(
        label="person", confidence=0.9,
        x1=x1, y1=y1, x2=x2, y2=y2, track_id=track_id,
    )


def car(box=(300, 200, 600, 400)) -> Detection:
    x1, y1, x2, y2 = box
    return Detection(
        label="car", confidence=0.9,
        x1=x1, y1=y1, x2=x2, y2=y2,
    )


def weapon(label="knife", conf=0.8) -> Detection:
    # Centred inside `person()`'s box, i.e. held in hand.
    return Detection(
        label=label, confidence=conf, x1=140, y1=250, x2=160, y2=300,
    )


def mask(label="mask", conf=0.8) -> Detection:
    return Detection(
        label=label, confidence=conf, x1=120, y1=110, x2=180, y2=180,
    )


def occluded(conf=0.5) -> Detection:
    return Detection(
        label="face_occluded", confidence=conf, x1=120, y1=110, x2=180, y2=180,
    )


# =============================================================================
#  1. A weapon is decisive on its own
# =============================================================================
def test_gun_alerts_with_nothing_else_in_frame(engine):
    """A firearm in an otherwise empty car park must still dispatch an alert."""
    result = engine.assess(vehicles=[], persons=[person()], weapons=[weapon("gun")])
    assert result.should_alert
    assert result.is_armed
    assert TRIGGER_FIREARM in result.triggers
    assert result.level.value == "critical"


def test_knife_alerts_with_no_vehicle_and_no_face_match(engine):
    """
    The specification's core scenario, minus every corroborating signal.

    There is no registered vehicle, no face match and no plate read. The only
    evidence is a knife in a person's hand, and it is enough.
    """
    result = engine.assess(vehicles=[], persons=[person()], weapons=[weapon()])
    assert result.should_alert
    assert TRIGGER_WEAPON in result.triggers
    assert result.armed_person_ids == []      # no track id on this person


def test_weapon_alerts_on_its_own_with_no_person_detected(engine):
    """
    A knife with no person box in frame is still a knife.

    The person detector and the weapon detector are separate networks; a weapon
    near the edge of frame may be found while the person is not. Discarding it
    because no person was detected would be a real blind spot.
    """
    result = engine.assess(vehicles=[], persons=[], weapons=[weapon()])
    assert result.should_alert
    assert result.is_armed


@pytest.mark.parametrize(
    "label,expected",
    [
        ("gun", WEIGHT_FIREARM),
        ("pistol", WEIGHT_FIREARM),
        ("revolver", WEIGHT_FIREARM),
        ("handgun", WEIGHT_FIREARM),
        ("rifle", WEIGHT_FIREARM),
        ("shotgun", WEIGHT_FIREARM),
        ("knife", WEIGHT_WEAPON),
        ("machete", WEIGHT_WEAPON),
        ("baton", WEIGHT_WEAPON),
        ("sword", WEIGHT_WEAPON),
        ("crowbar", WEIGHT_THREATENING_TOOL),
        ("screwdriver", WEIGHT_THREATENING_TOOL),
    ],
)
def test_every_weapon_class_scores_correctly(engine, label, expected):
    result = engine.assess(vehicles=[], persons=[person()], weapons=[weapon(label)])
    breakdown = result.breakdown
    assert expected in breakdown.values(), f"{label} scored {breakdown}"


def test_firearm_outranks_knife_in_the_same_frame(engine):
    """
    The most serious weapon sets the grade.

    A frame with both must read as a firearm, not as a summed 185 that
    flattens the distinction and destroys the alert's usefulness.
    """
    result = engine.assess(
        vehicles=[], persons=[person()], weapons=[weapon("gun"), weapon("knife")],
    )
    assert TRIGGER_FIREARM in result.triggers
    assert TRIGGER_WEAPON not in result.triggers
    # Not additive: the score is the firearm's weight alone.
    assert result.breakdown[TRIGGER_FIREARM] == WEIGHT_FIREARM
    assert len(result.weapons) == 2


def test_crowbar_alone_does_not_cross_the_threshold(engine):
    """
    A tool is suggestive, not an attack. It must combine with something else.

    If this fired on its own, every maintenance worker with a screwdriver would
    trigger a police-grade alert and the system would be disabled within a week.
    """
    result = engine.assess(vehicles=[], persons=[person()], weapons=[weapon("crowbar")])
    assert result.breakdown[TRIGGER_THREATENING_TOOL] == WEIGHT_THREATENING_TOOL
    assert not result.should_alert, "a crowbar alone must not dispatch an alert"


def test_crowbar_becomes_serious_when_combined_with_a_thief(engine):
    """
    The tool weight exists to combine, not to stand alone.

    A crowbar at an unregistered plate is a break-in in progress, and the two
    signals together must clear the threshold - that is the whole reason the tool
    is scored at 60 rather than lower.
    """
    result = engine.assess(
        vehicles=[car()],
        persons=[person()],
        weapons=[weapon("crowbar")],
        plate_in_database=False,
        plate_number="XYZ789AB",
    )
    assert result.should_alert


def test_screwdriver_does_not_alert_in_an_ordinary_scene(engine):
    """
    A workman's tool, at a registered car with the owner present, must be silent.

    The tool alone scores 45 - below the 60 threshold. It is *combined* with the
    stranger-near-vehicle rule that a screwdriver always picks up that this must
    not page anybody, because that combination is a mechanic doing their job.
    """
    result = engine.assess(
        vehicles=[car()],
        persons=[person()],
        weapons=[weapon("screwdriver")],
        plate_in_database=True,
        person_status=PersonStatus.AUTHORIZED,
    )
    assert not result.should_alert, result.reason


def test_screwdriver_alone_without_a_vehicle_is_silent(engine):
    result = engine.assess(
        vehicles=[], persons=[person()], weapons=[weapon("screwdriver")],
    )
    assert not result.should_alert


# =============================================================================
#  2. A weapon defeats owner-present suppression  (the critical property)
# =============================================================================
def test_weapon_is_not_silenced_by_owner_being_present(engine):
    """
    The most important test in this file.

    Owner-present suppression (-50) exists so that an authorised owner
    collecting their own car is not alarmed by bad plate OCR. If it also
    silenced a weapon, a carjacking where the owner *is* recognised would score
    85 - 50 = 35 and never alert, while the visually identical scene with an
    unrecognised owner would score 85 and alert. The suppression would be
    actively making the system blind in exactly the case it should be loudest.
    """
    result = engine.assess(
        vehicles=[car()],
        persons=[person()],
        weapons=[weapon()],
        person_status=PersonStatus.AUTHORIZED,
    )
    assert TRIGGER_OWNER_PRESENT not in result.triggers
    assert result.should_alert
    assert result.is_armed


def test_gun_is_not_silenced_by_owner_being_present(engine):
    result = engine.assess(
        vehicles=[car()],
        persons=[person()],
        weapons=[weapon("gun")],
        person_status=PersonStatus.AUTHORIZED,
    )
    assert TRIGGER_OWNER_PRESENT not in result.triggers
    assert result.score == WEIGHT_FIREARM
    assert result.level.value == "critical"


def test_owner_present_still_suppresses_when_no_weapon(engine):
    """
    The suppression must keep working normally.

    Without this, a fix for the test above could simply delete the rule and
    make every owner trip an alert - which is the original problem it solved.
    """
    result = engine.assess(
        vehicles=[car()],
        persons=[person()],
        plate_in_database=False,
        plate_number="ABC123XY",
        person_status=PersonStatus.AUTHORIZED,
    )
    assert TRIGGER_OWNER_PRESENT in result.triggers
    assert not result.should_alert


def test_mask_does_not_bypass_suppression_alone(engine):
    """
    Only a *weapon* bypasses suppression.

    A mask is strong evidence of intent but is far more often innocent - a
    cyclist in a cold climate, someone with flu - so it must not stop the
    system recognising that the owner is present.
    """
    result = engine.assess(
        vehicles=[car()],
        persons=[person()],
        masks=[mask()],
        plate_in_database=False,
        person_status=PersonStatus.AUTHORIZED,
    )
    assert TRIGGER_OWNER_PRESENT in result.triggers


# =============================================================================
#  3. Weapon attribution
# =============================================================================
def test_weapon_is_attributed_to_the_person_holding_it(engine):
    result = engine.assess(
        vehicles=[], persons=[person(track_id=7)], weapons=[weapon()],
    )
    assert result.armed_person_ids == [7]
    assert "#7" in result.reason


def test_weapon_outside_every_person_is_still_reported(engine):
    """A weapon nobody is holding is still a weapon in the scene."""
    loose = Detection(
        label="knife", confidence=0.8, x1=800, y1=600, x2=830, y2=640,
    )
    result = engine.assess(vehicles=[], persons=[person(track_id=1)], weapons=[loose])
    assert result.should_alert
    assert result.armed_person_ids == []


# =============================================================================
#  4. Masks and face concealment
# =============================================================================
def test_detected_mask_scores_and_flags_concealment(engine):
    result = engine.assess(vehicles=[car()], persons=[person()], masks=[mask()])
    assert TRIGGER_MASK in result.triggers
    assert result.breakdown[TRIGGER_MASK] == WEIGHT_MASK
    assert result.is_concealed
    assert not result.is_armed          # a mask is not a weapon


def test_mask_outscores_an_inferred_occlusion(engine):
    """
    Evidence quality must be reflected in the score.

    A mask the model actually saw is a stronger claim than a face it merely
    struggled with, and the alert history should be able to tell them apart.
    """
    detected = engine.assess(vehicles=[car()], persons=[person()], masks=[mask()])
    inferred = engine.assess(
        vehicles=[car()], persons=[person()], occluded_faces=[occluded()],
    )
    assert detected.score > inferred.score
    assert TRIGGER_MASK in detected.triggers
    assert TRIGGER_OCCLUDED_FACE in inferred.triggers


def test_mask_and_occlusion_are_not_double_counted(engine):
    """
    A masked face is *one* fact, not two.

    Both signals firing together is the normal case - the mask causes the poor
    detection score - and scoring both would report a single concealed face as
    two separate findings.
    """
    result = engine.assess(
        vehicles=[car()], persons=[person()], masks=[mask()], occluded_faces=[occluded()],
    )
    assert TRIGGER_MASK in result.triggers
    assert TRIGGER_OCCLUDED_FACE not in result.triggers


def test_mask_plus_stranger_alerts(engine):
    """Masked stranger at an unregistered plate: the classic theft shape."""
    result = engine.assess(
        vehicles=[car()],
        persons=[person()],
        masks=[mask()],
        plate_in_database=False,
        plate_number="XYZ789AB",
    )
    assert result.should_alert


def test_masked_owner_is_not_alerted_on_mask_alone(engine):
    """
    A mask alone with the owner present must stay quiet.

    Masks are common and usually innocent; this is the false-positive guard.
    """
    result = engine.assess(
        vehicles=[car()],
        persons=[person()],
        masks=[mask()],
        plate_in_database=True,
        person_status=PersonStatus.AUTHORIZED,
    )
    assert not result.should_alert


def test_inferred_occlusion_is_reported_distinctly(engine):
    result = engine.assess(
        vehicles=[car()], persons=[person()], occluded_faces=[occluded()],
    )
    assert result.breakdown[TRIGGER_OCCLUDED_FACE] == WEIGHT_OCCLUDED_FACE
    assert result.is_concealed
    assert "mask" in result.reason.lower()


# =============================================================================
#  5. False positives
# =============================================================================
def test_an_ordinary_stranger_near_a_car_does_not_alert(engine):
    """The baseline the weapon rules must not disturb."""
    result = engine.assess(vehicles=[car()], persons=[person()])
    assert not result.should_alert
    assert not result.is_armed
    assert not result.is_concealed


def test_an_empty_frame_scores_zero(engine):
    result = engine.assess(vehicles=[], persons=[])
    assert result.score == 0
    assert not result.is_armed
    assert not result.should_alert


def test_nothing_detected_is_not_armed_or_concealed(engine):
    result = engine.assess(
        vehicles=[car()], persons=[person()], weapons=[], masks=[], occluded_faces=[],
    )
    assert not result.is_armed
    assert not result.is_concealed


def test_score_is_capped_at_100_even_with_every_signal(engine):
    result = engine.assess(
        vehicles=[car(), car()],
        persons=[person(), person(), person(), person()],
        weapons=[weapon("gun")],
        masks=[mask()],
        plate_in_database=False,
        plate_number="ZZZ111ZZ",
        is_night=True,
    )
    assert result.score == 100
    assert 0 <= result.score <= 100


# =============================================================================
#  6. A thief reaching into a partially-captured vehicle
# =============================================================================
# The motivating case: a car clipped by the frame edge, a thief bent through the
# driver window with no face visible. Previously this scored 50 - ten points
# under the threshold - and stayed silent, because "I could not see their face"
# and "they concealed their face" contributed the same (nothing).
#
# Partially-captured cars are the hard part, and they work because proximity is
# judged against whatever vehicle box the detector actually saw, while the
# *person* box is the stable element.
def _partial_car() -> Detection:
    """A car clipped by the left frame edge - the visible part only."""
    return car(box=(0, 200, 500, 450))


def _window_thief() -> Detection:
    """A person leaning into the window of the car above."""
    return person(track_id=3, box=(430, 180, 520, 500))


def test_thief_at_window_of_a_partial_car_with_no_visible_face_alerts(engine):
    """The target case: 50 before this change, now over the threshold."""
    result = engine.assess(
        vehicles=[_partial_car()],
        persons=[_window_thief()],
        plate_in_database=None,
        person_status=PersonStatus.NO_FACE,
    )
    assert result.should_alert
    assert TRIGGER_FACE_NOT_VISIBLE in result.triggers
    assert result.breakdown[TRIGGER_FACE_NOT_VISIBLE] == WEIGHT_FACE_NOT_VISIBLE


def test_unmatchable_face_at_a_window_alerts(engine):
    """
    A face that was found but could not be matched also fires.

    Weaker than NO_FACE - the system did get to look - but a person leaning
    into a window whose face matches nobody is still a break-in attempt.
    """
    result = engine.assess(
        vehicles=[_partial_car()],
        persons=[_window_thief()],
        plate_in_database=None,
        person_status=PersonStatus.UNKNOWN,
    )
    assert result.should_alert
    assert TRIGGER_FACE_UNMATCHED in result.triggers


def test_no_visible_face_scores_higher_than_unmatchable_face(engine):
    """
    The two must stay distinguishable.

    No face at all is the classic concealed-intruder signature - the person is
    bent into the cabin. A face that simply did not match is weaker evidence.
    """
    hidden = engine.assess(
        vehicles=[_partial_car()], persons=[_window_thief()],
        plate_in_database=None, person_status=PersonStatus.NO_FACE,
    )
    unmatched = engine.assess(
        vehicles=[_partial_car()], persons=[_window_thief()],
        plate_in_database=None, person_status=PersonStatus.UNKNOWN,
    )
    assert hidden.score > unmatched.score


@pytest.mark.parametrize(
    "status",
    [PersonStatus.AUTHORIZED, PersonStatus.UNAUTHORIZED, None],
)
def test_unauthorised_face_is_not_double_counted(engine, status):
    """
    Only the two *unverifiable* cases add this score.

    UNAUTHORIZED already carries a much larger weight of its own; adding to it
    would inflate a clear-cut case, and AUTHORIZED must stay suppressed.
    """
    result = engine.assess(
        vehicles=[_partial_car()],
        persons=[_window_thief()],
        plate_in_database=None,
        person_status=status,
    )
    if status is PersonStatus.UNAUTHORIZED:
        assert TRIGGER_FACE_NOT_VISIBLE not in result.triggers
        assert TRIGGER_FACE_UNMATCHED not in result.triggers


# --- the innocent cases this must not disturb -------------------------------
def test_a_stranger_walking_past_a_car_does_not_alert(engine):
    """
    The false-positive guard for the new rule.

    A person near a car is a +40; the unverifiable-identity score must require
    them to be *reaching into* the vehicle, not merely close to it.
    """
    result = engine.assess(
        vehicles=[car(box=(300, 200, 700, 450))],
        persons=[person(box=(100, 180, 190, 500))],
        plate_in_database=None,
        person_status=PersonStatus.NO_FACE,
    )
    assert not result.should_alert
    assert TRIGGER_FACE_NOT_VISIBLE not in result.triggers


def test_a_hidden_face_at_a_registered_car_with_the_owner_home_is_quiet(engine):
    """
    An owner leaning into their own car with their face turned away is routine.

    The score is gated on the owner not being recognised, so this stays at 0.
    """
    result = engine.assess(
        vehicles=[_partial_car()],
        persons=[_window_thief()],
        plate_in_database=True,
        plate_number="ABC123XY",
        person_status=PersonStatus.AUTHORIZED,
    )
    assert result.score == 0
    assert not result.should_alert


def test_a_hidden_face_in_an_empty_car_park_scores_nothing(engine):
    """Nobody scores for an unseen face when no vehicle is involved."""
    result = engine.assess(
        vehicles=[], persons=[person()], person_status=PersonStatus.NO_FACE,
    )
    assert result.score == 0
    assert TRIGGER_FACE_NOT_VISIBLE not in result.triggers


# --- partial capture --------------------------------------------------------
def test_proximity_survives_a_heavily_clipped_vehicle(engine):
    """
    Even 20% of the car visible is enough to attribute the thief to it.

    A thief at the window of a car whose box is only 500 px wide still overlaps
    that box, so the strong proximity test holds and the case scores.
    """
    tiny_car = car(box=(0, 200, 120, 400))
    result = engine.assess(
        vehicles=[tiny_car],
        persons=[person(box=(60, 190, 160, 500))],
        plate_in_database=None,
        person_status=PersonStatus.NO_FACE,
    )
    assert result.should_alert
    assert TRIGGER_FACE_NOT_VISIBLE in result.triggers


def test_break_in_at_night_compounds(engine):
    """Night aggravates an already-suspicious frame; it does not create one."""
    day = engine.assess(
        vehicles=[_partial_car()], persons=[_window_thief()],
        plate_in_database=None, person_status=PersonStatus.NO_FACE,
    )
    night = engine.assess(
        vehicles=[_partial_car()], persons=[_window_thief()],
        plate_in_database=None, person_status=PersonStatus.NO_FACE, is_night=True,
    )
    assert night.score > day.score
    assert night.should_alert


def test_the_two_proximity_tests_disagree_as_designed(engine):
    """
    `_person_near_vehicle` is deliberately looser than `_person_at_vehicle`.

    This is the design contract between them: the loose test feeds the moderate
    "someone is at your car" signal, the strict one gates the claim that they
    are reaching inside. A person standing beside a parked car must satisfy the
    first and fail the second, or every passer-by becomes a break-in.
    """
    car_box = car(box=(600, 200, 1000, 450))     # 400x250, diagonal ~471 px
    # A clear gap: this person's centre is ~665 px from the car's, well beyond
    # the car diagonal that the loose test uses as its radius.
    bystander = person(box=(80, 180, 190, 500))

    assert engine._person_near_vehicle([bystander], [car_box]) is False
    assert engine._person_at_vehicle([bystander], [car_box]) is False

    # Leaning into the boot: the boxes genuinely overlap, so both tests agree.
    leaning = person(box=(960, 190, 1070, 500))
    assert engine._person_near_vehicle([leaning], [car_box]) is True
    assert engine._person_at_vehicle([leaning], [car_box]) is True

    # Standing directly *beside* a car, close enough to be "at" it but not
    # touching it: the loose test accepts, the strict one does not. This is the
    # overlap between the two that keeps a passer-by out of the break-in path.
    beside = person(box=(490, 180, 600, 500))
    assert engine._person_near_vehicle([beside], [car_box]) is True
    assert engine._person_at_vehicle([beside], [car_box]) is False
