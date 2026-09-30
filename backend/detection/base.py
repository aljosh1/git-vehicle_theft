"""
Shared detection types.

`Detection` is the single vocabulary every detector speaks and every consumer
(theft engine, annotator, database logger) reads.  Defining it here - rather
than passing raw Ultralytics `Results` objects around - is what keeps the
YOLO dependency confined to `backend/detection/`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# COCO class ids emitted by the pretrained YOLOv8 weights.
COCO_PERSON = 0
COCO_BICYCLE = 1
COCO_CAR = 2
COCO_MOTORCYCLE = 3
COCO_BUS = 5
COCO_TRUCK = 7
COCO_BIRD = 14
COCO_CAT = 15
COCO_DOG = 16
COCO_HORSE = 17
COCO_SHEEP = 18
COCO_COW = 19
COCO_ELEPHANT = 20
COCO_BEAR = 21
COCO_ZEBRA = 22
COCO_GIRAFFE = 23

# The four vehicle categories the specification requires.
VEHICLE_CLASS_IDS: dict[int, str] = {
    COCO_CAR: "car",
    COCO_MOTORCYCLE: "motorcycle",
    COCO_BUS: "bus",
    COCO_TRUCK: "truck",
}

VEHICLE_CLASS_NAMES = frozenset(VEHICLE_CLASS_IDS.values())

ANIMAL_CLASS_IDS: dict[int, str] = {
    COCO_BIRD: "bird",
    COCO_CAT: "cat",
    COCO_DOG: "dog",
    COCO_HORSE: "horse",
    COCO_SHEEP: "sheep",
    COCO_COW: "cow",
    COCO_ELEPHANT: "elephant",
    COCO_BEAR: "bear",
    COCO_ZEBRA: "zebra",
    COCO_GIRAFFE: "giraffe",
}

ANIMAL_CLASS_NAMES = frozenset(ANIMAL_CLASS_IDS.values())

# =============================================================================
#  Threat objects - weapons
# =============================================================================
# COCO class 79 is `knife`, the only weapon in the COCO vocabulary. It is
# deliberately absent from `VEHICLE_CLASS_IDS` and from the general person
# filter: a knife in frame is the single most decisive theft indicator there is,
# and it must survive the detector's class filter to reach the threat engine.
COCO_KNIFE = 79

# Weapon classes keyed by *name* rather than by COCO id, because the classes
# that actually matter - gun, knife, scissors, machete - do not exist in COCO at
# all. A purpose-trained model (see `backend/detection/threat_detector.py`)
# emits these names; the detector normalises spelling so `Gun`, `gun` and
# `pistol` all reach the theft engine as one label.
WEAPON_CLASS_NAMES: frozenset[str] = frozenset(
    {
        "knife", "knive", "knifes",
        "gun", "pistol", "revolver", "handgun", "firearm", "rifle", "shotgun",
        "machete", "cleaver", "baton", "club", "hammer", "axe", "sword",
        "grenade", "bomb", "weapon", "firearm",
        "scissors", "blade",
    }
)

# Items that are not weapons in themselves but are strongly associated with
# violent intent when carried into a vehicle area. Scored below the weapons
# themselves so that a garden tool in a worker's hand does not by itself page
# a security guard.
THREATENING_CLASS_NAMES: frozenset[str] = frozenset(
    {"crowbar", "screwdriver", "crow_bar", "pliers", "wrench", "torch", "crowbar"}
)

MASK_CLASS_NAMES: frozenset[str] = frozenset(
    {"mask", "facemask", "face_mask", "surgical_mask", "n95", "balaclava",
     "helmet_mask", "masks"}
)

# A weapon held *in* a person's hand, or a mask worn *on* a face, is the normal
# case and is what the rules below assume.
WEAPON_LABELS: frozenset[str] = WEAPON_CLASS_NAMES
MASK_LABELS: frozenset[str] = MASK_CLASS_NAMES


@dataclass(slots=True)
class Detection:
    """
    One detected object in one frame.

    Coordinates are absolute pixels in the source frame, top-left origin,
    `x1 <= x2` and `y1 <= y2` guaranteed by the detectors.
    """

    label: str                       # "car", "person", "license_plate", ...
    confidence: float                # 0.0 - 1.0
    x1: int
    y1: int
    x2: int
    y2: int
    class_id: int | None = None
    track_id: int | None = None
    # Per-stage enrichment: OCR text, face match, threat contribution, ...
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------- geometry --
    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[int, int]:
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2

    @property
    def is_vehicle(self) -> bool:
        return self.label in VEHICLE_CLASS_NAMES

    @property
    def is_person(self) -> bool:
        return self.label == "person"

    @property
    def is_animal(self) -> bool:
        return self.label in ANIMAL_CLASS_NAMES

    @property
    def is_weapon(self) -> bool:
        """A gun, knife or bladed object. The most decisive theft signal."""
        return self.label in WEAPON_LABELS

    @property
    def is_mask(self) -> bool:
        """A face mask or balaclava. Conceals identity, so it defeats face ID."""
        return self.label in MASK_LABELS

    @property
    def is_threatening(self) -> bool:
        """A weapon, or an implement strongly associated with an attack."""
        return self.label in WEAPON_LABELS or self.label in THREATENING_CLASS_NAMES

    def contains(self, other: "Detection") -> bool:
        """
        Is `other`'s centre inside this box?

        Used to attribute a knife to the person holding it, which is what makes
        "armed intruder" a per-person verdict rather than a free-floating
        detection that cannot be attributed to anybody.

        Written as a property on the *container* so the call site reads as
        `person.contains(weapon)` - "does this person contain this weapon" -
        rather than `weapon.centre_in(person)`, which reads backwards.
        """
        cx, cy = other.center
        return self.x1 <= cx <= self.x2 and self.y1 <= cy <= self.y2

    def iou(self, other: "Detection") -> float:
        """Intersection-over-union - used to associate persons with vehicles."""
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        union = self.area + other.area - inter
        return inter / union if union else 0.0

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form for the API and the detection log."""
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "bbox": list(self.bbox),
            "class_id": self.class_id,
            "track_id": self.track_id,
            "meta": self.meta,
        }
