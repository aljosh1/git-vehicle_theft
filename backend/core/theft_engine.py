"""
Theft detection logic - the system's decision-making core.

The specification lists three trigger conditions:

    1. Unknown person detected near a registered vehicle
    2. Vehicle license plate not found in the database
    3. Unauthorized face attempts vehicle access

Treating those as independent booleans would produce an alarm on every weak
signal.  Instead each condition contributes additively to a **0-100 threat
score**, and only the total crosses the alerting threshold.  The reasoning:

* A stranger merely walking past a car park (condition 1 alone, +40) is not a
  theft - it is Tuesday.  It should be logged, not escalated.
* An unregistered plate alone (+35) usually means a visitor, not a thief.
* A stranger *at* an unregistered vehicle (75) is genuinely worth waking
  somebody up for.

Armed and masked intruders
--------------------------
The three conditions above all describe *suspicion*. Two situations are not
suspicion at all - they are an attack in progress, and they are scored
accordingly:

* **A weapon** (+85 to +100). A firearm alone clears the threshold; a knife
  does too. This is the largest single contribution in the engine, and it is
  applied *before* the owner-present suppression, which a weapon bypasses
  entirely. That bypass is the whole point: the -50 suppression exists to stop
  an owner being alarmed by their own badly-OCR'd plate, and applying it to
  someone holding a crowbar would push a certain robbery below the threshold
  and never page anybody.
* **A concealed face** (+30 to +55). A mask defeats face recognition outright,
  so a masked intruder is otherwise indistinguishable from an unrecognisable
  one. A detected mask is scored higher than an *inferred* face occlusion,
  because the evidence quality differs and the alert should say which it was.

The single most important rule is the **owner-present suppression** (-50).
Without it, an authorised owner whose plate OCRs badly at night triggers an
alert every time they collect their own car - and an alerting system that cries
wolf gets switched off, which is the real failure mode of systems like this.

Every score carries a human-readable `reason` and a machine-readable `triggers`
list, so an operator reviewing an alert months later can see exactly which rules
fired and an evaluator can audit the decision.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from backend.config import settings
from backend.database.models import OwnerVerificationStatus, PersonStatus, ThreatLevel
from backend.detection.base import (
    THREATENING_CLASS_NAMES,
    WEAPON_LABELS,
    Detection,
)
from backend.utils.logger import get_logger

log = get_logger(__name__)


# =============================================================================
#  Rule weights - tuned so that any single condition stays below the default
#  alert threshold of 60, and realistic combinations clear it.
# =============================================================================
WEIGHT_PLATE_NOT_REGISTERED = 35
WEIGHT_UNKNOWN_PERSON_NEAR_VEHICLE = 40
WEIGHT_UNAUTHORIZED_FACE = 45
WEIGHT_LOITERING = 15
WEIGHT_NO_PLATE_READ = 10
WEIGHT_MULTIPLE_PERSONS = 10
WEIGHT_NIGHT_TIME = 5
BONUS_FLAGGED_STOLEN = 100
SUPPRESSION_OWNER_PRESENT = -50

# ---- Armed / masked intruder weights ---------------------------------------
# These are deliberately larger than every situational modifier combined. A
# stranger loitering is suspicious; a stranger loitering *holding a weapon* is an
# active robbery, and an operator who has to weigh that judgement is being made
# to do the system's job for it.
#
# The gradient encodes intent, not just presence:
#   firearm 100     - the object is the incident. Alerts on its own.
#   weapon  85      - a machete or knife is no less dangerous in a car park.
#   tool    45      - a crowbar is suggestive, not conclusive. Deliberately set
#                     *below* the 60 alerting threshold so it must combine with
#                     another signal (an unregistered plate, an unrecognised
#                     face) to fire. At 60 it would trip the `>=` comparison
#                     alone and turn every worker's screwdriver into a police
#                     alert - and a system that cries wolf gets switched off.
#   mask    55      - defeats face recognition entirely, so the system cannot
#                     do the one job it is best at. Below the threshold for the
#                     same reason as the tool, but for the opposite cause: a
#                     mask is a *reliable* signal that is usually innocent.
#   occluded face 30 - the weakest signal; an inference from a poor detection
#                     score, which bad lighting can also produce.
WEIGHT_FIREARM = 100
WEIGHT_WEAPON = 85
WEIGHT_THREATENING_TOOL = 45
WEIGHT_MASK = 55
WEIGHT_OCCLUDED_FACE = 30

# A person reaching into a car whose face is never visible. Deliberately small:
# 15 puts "stranger at a window, face unseen" (50) over the 60 threshold, while
# leaving a stranger *merely walking past* (40) well under it. It is applied only
# when a person is at a vehicle - a face that is never visible in an empty car
# park is nobody's problem.
#
# This exists because "I could not see their face" and "they concealed their
# face" were scoring identically, and in a break-in those are the same event.
# See `TRIGGER_FACE_NOT_VISIBLE`.
WEIGHT_FACE_NOT_VISIBLE = 15
# A face that is visible but could not be matched to anyone, at a vehicle.
# Set to land exactly on the default alerting threshold, so it fires on its own
# in the break-in case but is one more signal (night, loitering, a second
# person) away from firing in a marginal one. Below NO_FACE because the system
# did get to look and still found nothing, which is a weaker claim than never
# getting the chance to look.
WEIGHT_FACE_UNMATCHED = 10
# An armed person is not "more suspicious than" an unarmed one - the owner-present
# suppression must not silence a gun, so weapons bypass it entirely.
WEAPON_SUPPRESSION_BYPASS = True

# Machine-readable trigger identifiers, stored on `alerts.triggers`.
TRIGGER_PLATE_NOT_REGISTERED = "plate_not_registered"
TRIGGER_UNKNOWN_PERSON = "unknown_person_near_vehicle"
TRIGGER_UNAUTHORIZED_FACE = "unauthorized_face"
TRIGGER_LOITERING = "loitering"
TRIGGER_NO_PLATE = "no_plate_read"
TRIGGER_MULTIPLE_PERSONS = "multiple_persons"
TRIGGER_NIGHT = "night_time"
TRIGGER_FLAGGED_STOLEN = "vehicle_flagged_stolen"
TRIGGER_OWNER_PRESENT = "owner_present"
TRIGGER_FIREARM = "firearm_detected"
TRIGGER_WEAPON = "weapon_detected"
TRIGGER_THREATENING_TOOL = "threatening_tool"
TRIGGER_MASK = "mask_detected"
TRIGGER_OCCLUDED_FACE = "face_occluded"
TRIGGER_FACE_NOT_VISIBLE = "face_not_visible_at_vehicle"
TRIGGER_FACE_UNMATCHED = "face_unmatched_at_vehicle"



@dataclass(slots=True)
class ThreatAssessment:
    """The verdict for one frame."""

    score: int
    level: ThreatLevel
    reason: str
    triggers: list[str] = field(default_factory=list)
    # Contribution breakdown, for the dashboard's explainability panel.
    breakdown: dict[str, int] = field(default_factory=dict)

    matched_vehicle_id: int | None = None
    plate_number: str | None = None
    plate_in_database: bool | None = None
    person_status: PersonStatus | None = None
    face_similarity: float | None = None
    matched_user_id: int | None = None

    # Threat objects seen in this frame, for the dashboard's explainability panel.
    weapons: list[str] = field(default_factory=list)
    masks: list[str] = field(default_factory=list)
    occluded_faces: int = 0
    armed_person_ids: list[int] = field(default_factory=list)

    @property
    def should_alert(self) -> bool:
        return self.score >= settings.THREAT_ALERT_THRESHOLD

    @property
    def triggers_csv(self) -> str:
        return ",".join(self.triggers)

    @property
    def is_armed(self) -> bool:
        """
        Was a weapon or an attack-capable tool seen?

        `is_armed` in the plain sense - a gun or a knife - is a narrower thing
        than the engine can tell from an image. A crowbar is not a weapon, but a
        person carrying one in a car park is committing the same act, so this
        returns True for both and the `triggers` field records which it was.
        """
        return any(
            trigger in {TRIGGER_FIREARM, TRIGGER_WEAPON, TRIGGER_THREATENING_TOOL}
            for trigger in self.triggers
        )

    @property
    def is_concealed(self) -> bool:
        """Was the person's face hidden, defeating face recognition?"""
        return bool(self.masks) or self.occluded_faces > 0


def score_to_level(score: int) -> ThreatLevel:
    """
    Map a 0-100 score onto the five bands.

    Boundaries match the enum's documented ranges: 0-19 none, 20-39 low,
    40-59 medium, 60-79 high, 80-100 critical.
    """
    if score >= 80:
        return ThreatLevel.CRITICAL
    if score >= 60:
        return ThreatLevel.HIGH
    if score >= 40:
        return ThreatLevel.MEDIUM
    if score >= 20:
        return ThreatLevel.LOW
    return ThreatLevel.NONE


class LoiterTracker:
    """
    Remembers how long each tracked person has been in frame.

    A stranger who walks through in two seconds is uninteresting; one who stands
    beside the same car for thirty is not.  Duration is the cheapest available
    proxy for intent, and it needs only a dict of first-seen timestamps keyed by
    the tracker's id.
    """

    def __init__(self, loiter_seconds: int | None = None):
        self.loiter_seconds = loiter_seconds or settings.LOITER_SECONDS
        self._first_seen: dict[int, float] = {}
        self._last_seen: dict[int, float] = {}

    def update(self, track_id: int | None) -> float:
        """Record a sighting. Returns how long this id has been present, in seconds."""
        if track_id is None:
            return 0.0
        now = time.monotonic()
        self._first_seen.setdefault(track_id, now)
        self._last_seen[track_id] = now
        return now - self._first_seen[track_id]

    def is_loitering(self, track_id: int | None) -> bool:
        if track_id is None:
            return False
        duration = self._last_seen.get(track_id, 0.0) - self._first_seen.get(
            track_id, 0.0
        )
        return duration >= self.loiter_seconds

    def prune(self, max_age_seconds: float = 60.0) -> None:
        """Forget ids not seen recently, so the dicts cannot grow without bound."""
        now = time.monotonic()
        stale = [
            tid for tid, seen in self._last_seen.items() if now - seen > max_age_seconds
        ]
        for tid in stale:
            self._first_seen.pop(tid, None)
            self._last_seen.pop(tid, None)


class TheftEngine:
    """
    Applies the scoring rules to one frame's detections.

    Stateless with respect to the database - the caller resolves plates to
    vehicles and faces to owners, then hands the outcomes here.  That separation
    keeps the rules unit-testable without any I/O, which is why
    `tests/test_theft_engine.py` runs in milliseconds.
    """

    def __init__(self, loiter_tracker: LoiterTracker | None = None):
        self.loiter = loiter_tracker or LoiterTracker()

    def assess(
        self,
        *,
        vehicles: list[Detection],
        persons: list[Detection],
        plate_number: str | None = None,
        plate_in_database: bool | None = None,
        matched_vehicle_id: int | None = None,
        vehicle_flagged_stolen: bool = False,
        person_status: PersonStatus | None = None,
        face_similarity: float | None = None,
        matched_user_id: int | None = None,
        owner_verification: OwnerVerificationStatus | None = None,
        weapons: list[Detection] | None = None,
        masks: list[Detection] | None = None,
        occluded_faces: list[Detection] | None = None,
        is_night: bool = False,
    ) -> ThreatAssessment:
        """
        Score one frame.

        Args:
            vehicles: vehicle detections in this frame.
            persons: person detections in this frame. Weapons are attributed to
                these people, so an armed intruder in a car park is scored even
                when no vehicle is in shot.
            plate_number: normalised plate text, if OCR produced one.
            plate_in_database: True if it matched a registered vehicle, False if
                it did not, None if no plate was read at all. The three-way
                distinction matters: "unreadable" is not "unregistered".
            matched_vehicle_id: `vehicles.id` when the plate matched.
            vehicle_flagged_stolen: owner has reported this vehicle stolen.
            person_status: vehicle-specific authorization verdict.
            face_similarity: best cosine similarity observed.
            matched_user_id: `users.id` of a recognised enrolled person.
            owner_verification: comparison of that person with the plate owner.
            weapons: weapon detections (gun, knife, machete, ...).
            masks: mask detections.
            occluded_faces: faces detected but poorly scored, i.e. covered.
            is_night: outside daylight hours - a mild aggravating factor.

        Returns:
            A `ThreatAssessment` with the score, band, reasons and breakdown.
        """
        score = 0
        triggers: list[str] = []
        reasons: list[str] = []
        breakdown: dict[str, int] = {}

        def add(points: int, trigger: str, reason: str) -> None:
            nonlocal score
            score += points
            triggers.append(trigger)
            reasons.append(reason)
            breakdown[trigger] = points

        weapons = weapons or []
        masks = masks or []
        occluded_faces = occluded_faces or []

        has_vehicle = bool(vehicles)
        has_person = bool(persons)
        owner_present = person_status is PersonStatus.AUTHORIZED

        # --- Threat objects: evaluated FIRST -------------------------------
        # Before everything else, because a weapon changes what every other
        # signal means. A knife next to an owner's own car is not a parking
        # dispute, and the owner-present suppression below must not silence it.
        weapon_names, weapon_points, weapon_trigger, weapon_reason = (
            self._assess_weapons(weapons, persons)
        )
        if weapon_points:
            add(weapon_points, weapon_trigger, weapon_reason)

        # --- Face concealment ---------------------------------------------
        # Scored independently of weapons, because "I could not identify this
        # person" and "this person hid their face" are different facts and an
        # operator reviewing an alert months later needs to know which it was.
        mask_reason = self._assess_masks(masks, occluded_faces)
        if mask_reason is not None:
            points, trigger, text = mask_reason
            add(points, trigger, text)

        # --- Condition 2: plate not in the database --------------------------
        # Only meaningful when a vehicle is actually in frame.
        if has_vehicle:
            if plate_in_database is False:
                add(
                    WEIGHT_PLATE_NOT_REGISTERED,
                    TRIGGER_PLATE_NOT_REGISTERED,
                    f"License plate {plate_number or 'unknown'} is not registered",
                )
            elif plate_in_database is None and plate_number is None:
                # Could not read the plate. Weak signal - fog, angle, dirt and
                # night-time all cause this on legitimate vehicles.
                add(
                    WEIGHT_NO_PLATE_READ,
                    TRIGGER_NO_PLATE,
                    "Vehicle present but its plate could not be read",
                )

        # --- Condition 1: unknown person near a registered vehicle ----------
        at_vehicle = False
        if has_person and has_vehicle:
            at_vehicle = self._person_near_vehicle(persons, vehicles)
            if at_vehicle and not owner_present:
                add(
                    WEIGHT_UNKNOWN_PERSON_NEAR_VEHICLE,
                    TRIGGER_UNKNOWN_PERSON,
                    "Unidentified person detected near a vehicle",
                )

        # --- Condition 3: unauthorised face ---------------------------------
        if person_status is PersonStatus.UNAUTHORIZED:
            similarity_note = (
                f" (best match {face_similarity:.0%})"
                if face_similarity is not None
                else ""
            )
            if owner_verification is OwnerVerificationStatus.MISMATCH:
                reason = f"Recognised face does not belong to this vehicle's owner{similarity_note}"
            else:
                reason = f"Face does not match any registered owner{similarity_note}"
            add(
                WEIGHT_UNAUTHORIZED_FACE,
                TRIGGER_UNAUTHORIZED_FACE,
                reason,
            )

        # --- An identity that could not be established at a vehicle --------
        # The gap this closes: a thief bent through a car window has no visible
        # face, which previously scored exactly the same as a stranger merely
        # walking past a car. Both are "unidentified", but one of them is
        # reaching into a vehicle, and that difference is the whole point.
        #
        # Gated on `at_vehicle` and on the owner not being recognised, so the
        # common innocent cases are untouched: nobody scores for a hidden face
        # in an open car park, and an owner leaning into their own car is still
        # suppressed.
        # The STRONG proximity test gates this, not the loose one used above.
        # A person merely walking along a row of parked cars is near a vehicle
        # but is not reaching into one, and must not be scored as an
        # unverifiable identity at a car.
        reaching_into_vehicle = (
            self._person_at_vehicle(persons, vehicles)
            if has_person and has_vehicle
            else False
        )

        concealment = self._assess_unverifiable_identity(
            person_status=person_status,
            at_vehicle=reaching_into_vehicle,
            owner_present=owner_present,
        )
        if concealment is not None:
            points, trigger, text = concealment
            add(points, trigger, text)

        if has_person:
            loitering = any(
                self.loiter.is_loitering(p.track_id) for p in persons
            )
            if loitering and not owner_present:
                add(
                    WEIGHT_LOITERING,
                    TRIGGER_LOITERING,
                    f"Person lingering for over {self.loiter.loiter_seconds}s",
                )

            if len(persons) >= 3 and not owner_present:
                add(
                    WEIGHT_MULTIPLE_PERSONS,
                    TRIGGER_MULTIPLE_PERSONS,
                    f"{len(persons)} people gathered around the vehicle",
                )

        if is_night and score > 0:
            # Night only aggravates something already suspicious; darkness alone
            # is not evidence of anything.
            add(WEIGHT_NIGHT_TIME, TRIGGER_NIGHT, "Occurred outside daylight hours")

        # --- Owner present: strong suppression ------------------------------
        # A weapon bypasses this entirely. Suppression exists to stop an
        # authorised owner's own car alarming them at 3am when the plate OCRs
        # badly; it was never meant to excuse a firearm, and applying -50 to an
        # armed intruder would drop a certain robbery below the alerting
        # threshold. A knife is not excused by the owner being present either -
        # the owner does not carry a crowbar.
        armed = weapon_points > 0
        if owner_present and not (armed and WEAPON_SUPPRESSION_BYPASS):
            add(
                SUPPRESSION_OWNER_PRESENT,
                TRIGGER_OWNER_PRESENT,
                "A registered owner was recognised at the scene",
            )
        elif owner_present and armed:
            log.warning(
                "owner present but a weapon was detected (%s) - the "
                "owner-present suppression was NOT applied", weapon_names,
            )

        # --- Reported stolen: overrides everything --------------------------
        # Applied last so it dominates the final score. If the owner reported
        # the car stolen, any sighting alerts regardless of who is present.
        if vehicle_flagged_stolen:
            score = 100
            if TRIGGER_FLAGGED_STOLEN not in triggers:
                triggers.insert(0, TRIGGER_FLAGGED_STOLEN)
                reasons.insert(
                    0, "This vehicle has been reported STOLEN by its owner"
                )
                breakdown[TRIGGER_FLAGGED_STOLEN] = BONUS_FLAGGED_STOLEN

        score = max(0, min(100, score))
        level = score_to_level(score)

        if reasons:
            reason_text = "; ".join(reasons)
        elif has_vehicle or has_person:
            reason_text = "Routine activity - nothing suspicious detected"
        else:
            reason_text = "No vehicles or persons in frame"

        return ThreatAssessment(
            score=score,
            level=level,
            reason=reason_text,
            triggers=triggers,
            breakdown=breakdown,
            matched_vehicle_id=matched_vehicle_id,
            plate_number=plate_number,
            plate_in_database=plate_in_database,
            person_status=person_status,
            face_similarity=face_similarity,
            matched_user_id=matched_user_id,
            weapons=[w.label for w in weapons],
            masks=[m.label for m in masks],
            occluded_faces=len(occluded_faces),
            armed_person_ids=self._armed_person_ids(weapons, persons),
        )

    # ====================================================================
    #  Threat-object scoring
    # ====================================================================
    # A firearm is separated from the general weapon class because the two
    # should not be scored identically: a knife in a car park is a violent
    # theft, a gun in one is an active shooting.
    _FIREARM_LABELS = frozenset(
        {"gun", "pistol", "revolver", "handgun", "firearm", "rifle", "shotgun"}
    )

    def _assess_weapons(
        self, weapons: list[Detection], persons: list[Detection]
    ) -> tuple[list[str], int, str, str]:
        """
        Grade the most serious weapon present and return it as one trigger.

        A frame with a gun *and* a knife produces a single `firearm_detected`
        trigger, not two additive ones. Stacking them would push any frame
        containing a weapon to 100 and destroy the ability to distinguish a
        crowbar from a firearm in the alert history - which is exactly the
        distinction an operator reviewing a pile of alerts needs.
        """
        if not weapons:
            return [], 0, "", ""

        # The most serious weapon sets the grade, not the count.
        best = max(weapons, key=lambda w: self._weapon_weight(w.label))
        label = best.label

        if label in self._FIREARM_LABELS:
            points, trigger = WEIGHT_FIREARM, TRIGGER_FIREARM
            description = "a firearm"
        elif label in WEAPON_LABELS:
            points, trigger = WEIGHT_WEAPON, TRIGGER_WEAPON
            description = f"a {label.replace('_', ' ')}"
        elif label in THREATENING_CLASS_NAMES:
            # A tool, not a weapon. Suggestive of intent, not conclusive on its
            # own - scoring it as a weapon would make every gardener with a
            # pair of shears an armed intruder.
            points, trigger = WEIGHT_THREATENING_TOOL, TRIGGER_THREATENING_TOOL
            description = f"a {label.replace('_', ' ')}"
        else:
            points, trigger = WEIGHT_THREATENING_TOOL, TRIGGER_THREATENING_TOOL
            description = f"a {label.replace('_', ' ')}"

        # Attribute the weapon to a person if one is holding it, so the alert
        # can say *who* is armed rather than that something is in frame.
        holder = self._holder_of(best, persons)
        others = len(weapons) - 1
        extra = f" (and {others} more)" if others > 0 else ""

        if holder is not None:
            reason = f"{holder} is holding {description}{extra}"
        else:
            reason = f"{description.capitalize()} detected in the scene{extra}"

        return [w.label for w in weapons], points, trigger, reason

    def _weapon_weight(self, label: str) -> int:
        """Ranking used to pick the most serious weapon in a frame."""
        if label in self._FIREARM_LABELS:
            return 3
        if label in WEAPON_LABELS:
            return 2
        if label in THREATENING_CLASS_NAMES:
            return 1
        return 0

    def _holder_of(
        self, weapon: Detection, persons: list[Detection]
    ) -> str | None:
        """
        Which person is holding this weapon, described for an alert.

        A knife's centre lying inside a person box is the standard tracker
        convention for "in hand". It is a geometric inference, not proof, so it
        is only used to make the alert readable - never to change the score.
        """
        for person in persons:
            if person.contains(weapon):
                if person.track_id is not None:
                    return f"Person #{person.track_id}"
                return "A person in frame"
        return None

    def _armed_person_ids(self, weapons: list[Detection], persons: list[Detection]) -> list[int]:
        """Track ids of people holding a weapon, for the dashboard's highlight."""
        return sorted(
            {
                person.track_id
                for weapon in weapons
                for person in persons
                if person.track_id is not None and person.contains(weapon)
            }
        )

    def _assess_masks(
        self, masks: list[Detection], occluded_faces: list[Detection]
    ) -> tuple[int, str, str] | None:
        """
        Grade face concealment.

        A real mask detection outranks an occlusion inference, because a mask the
        model actually saw is a stronger claim than a face it merely struggled
        with. When both are present only the mask is reported - the occlusion is
        almost certainly the same face, and listing it twice would double-count.
        """
        if masks:
            labels = ", ".join(sorted({m.label.replace("_", " ") for m in masks}))
            return (
                WEIGHT_MASK,
                TRIGGER_MASK,
                f"Face concealed with {labels} - identity cannot be confirmed by face recognition",
            )

        if occluded_faces:
            strongest = max(det.confidence for det in occluded_faces)
            return (
                WEIGHT_OCCLUDED_FACE,
                TRIGGER_OCCLUDED_FACE,
                f"Face not visible enough to identify ({strongest:.0%} confidence) "
                "- a mask or covering is likely",
            )
        return None

    def _assess_unverifiable_identity(
        self,
        *,
        person_status: PersonStatus | None,
        at_vehicle: bool,
        owner_present: bool,
    ) -> tuple[int, str, str] | None:
        """
        Score a person at a vehicle whose identity could not be established.

        Three cases are kept apart on purpose, because they mean different
        things to a guard and a single "unverified" bucket would erase the
        distinction:

        * ``NO_FACE``  - a person was there and no face was visible at all. In
          a break-in the person is bent into the cabin with their face hidden,
          so this is the classic concealed-intruder signature. Scored highest of
          the three.
        * ``UNKNOWN``  - a face *was* found and located, but it could not be
          embedded or matched. The system tried and failed, which is weaker
          evidence than never getting the chance to look.
        * ``UNAUTHORIZED`` is not handled here: it already has its own, much
          larger weight above.

        Returns None when the person was at no vehicle, or the owner was
        recognised - both are the ordinary, non-suspicious case.
        """
        if not at_vehicle or owner_present:
            return None

        if person_status is PersonStatus.NO_FACE:
            return (
                WEIGHT_FACE_NOT_VISIBLE,
                TRIGGER_FACE_NOT_VISIBLE,
                "A person at the vehicle with no visible face - identity "
                "could not be confirmed",
            )

        if person_status is PersonStatus.UNKNOWN:
            return (
                WEIGHT_FACE_UNMATCHED,
                TRIGGER_FACE_UNMATCHED,
                "A person at the vehicle whose face could not be matched to "
                "any registered owner",
            )

        return None

    def _person_near_vehicle(
        self, persons: list[Detection], vehicles: list[Detection]
    ) -> bool:
        """
        Is any person close enough to a vehicle to be interacting with it?

        Two tests, because IoU alone is unreliable here: a person standing
        *beside* a car has near-zero box overlap yet is obviously at the car.
        So we also accept a centre-distance within roughly one vehicle-width.

        This is the LOOSE test, and it is deliberately so - it feeds the
        +40 "someone is at your car" signal, where over-inclusion only adds a
        moderate score. It must not be used to decide that someone is *reaching
        into* a vehicle; see `_person_at_vehicle` for that.
        """
        for person in persons:
            for vehicle in vehicles:
                if person.iou(vehicle) >= settings.PROXIMITY_IOU_THRESHOLD:
                    return True

                px, py = person.center
                vx, vy = vehicle.center
                distance = ((px - vx) ** 2 + (py - vy) ** 2) ** 0.5
                # Diagonal of the vehicle box, which is a rough proxy for its
                # physical extent. A person within that distance is "at the car".
                diagonal = (vehicle.width**2 + vehicle.height**2) ** 0.5
                if distance <= diagonal:
                    return True
        return False

    def _person_at_vehicle(
        self, persons: list[Detection], vehicles: list[Detection]
    ) -> bool:
        """
        Is any person actually touching or leaning into a vehicle?

        The STRONG test, and the one that gates the "identity could not be
        verified" score. The difference from `_person_near_vehicle` is the whole
        point: someone walking past a parked car is *near* it, but you cannot
        reach through a window without your box overlapping the car.

        Relying on the loose test here would have made a person merely walking
        along a row of parked cars score as a break-in attempt. The test is
        geometric overlap of the person box with the vehicle box, or a person
        small enough and close enough to be inside it.

        Partially-captured cars are handled: the vehicle box is whatever the
        detector actually saw, and a thief leaning into the visible portion
        still overlaps it.
        """
        for person in persons:
            for vehicle in vehicles:
                if person.iou(vehicle) >= settings.PROXIMITY_IOU_THRESHOLD:
                    return True

                # Any genuine box intersection counts. YOLO person and car boxes
                # overlap readily when someone is bent into a cabin, even
                # though the IoU stays small because a person box is much larger.
                if (
                    person.x1 < vehicle.x2
                    and person.x2 > vehicle.x1
                    and person.y1 < vehicle.y2
                    and person.y2 > vehicle.y1
                ):
                    return True

                # A person entirely inside the vehicle box - the detector found
                # the window area and the whole body reads as "in the car".
                if (
                    vehicle.x1 <= person.x1
                    and person.x2 <= vehicle.x2
                    and vehicle.y1 <= person.y1
                    and person.y2 <= vehicle.y2
                ):
                    return True
        return False

    def update_tracking(self, persons: list[Detection]) -> None:
        """Feed person tracks to the loiter tracker. Call once per frame."""
        for person in persons:
            self.loiter.update(person.track_id)
        self.loiter.prune()


def is_night_time(hour: int | None = None) -> bool:
    """
    Rough daylight test: 19:00-06:00 counts as night.

    A production system would use the site's latitude with a sunrise/sunset
    calculation; for a fixed-site prototype the fixed window is honest enough,
    and the factor it feeds only contributes 5 points.
    """
    from datetime import datetime

    current = hour if hour is not None else datetime.now().hour
    return current >= 19 or current < 6
