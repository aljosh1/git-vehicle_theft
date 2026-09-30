"""Tests for multi-vehicle association and legacy result compatibility."""

from backend.core.pipeline import FrameResult, Pipeline, VehicleAnalysis
from backend.core.theft_engine import ThreatAssessment
from backend.database.models import OwnerVerificationStatus, PersonStatus, ThreatLevel
from backend.detection.base import Detection


def detection(label: str, box: tuple[int, int, int, int], track_id: int) -> Detection:
    return Detection(
        label=label,
        confidence=0.9,
        x1=box[0],
        y1=box[1],
        x2=box[2],
        y2=box[3],
        class_id=0,
        track_id=track_id,
    )


def test_person_is_assigned_to_only_one_nearest_vehicle() -> None:
    left = detection("car", (0, 100, 200, 300), 1)
    right = detection("car", (220, 100, 420, 300), 2)
    person = detection("person", (160, 90, 230, 310), 10)

    assignments = Pipeline._assign_persons_to_vehicles([left, right], [person])

    assert sum(person in rows for rows in assignments.values()) == 1
    assert assignments[id(left)] == [person]
    assert assignments[id(right)] == []


def test_distant_person_is_not_associated_with_vehicle() -> None:
    vehicle = detection("car", (0, 0, 100, 100), 1)
    person = detection("person", (500, 500, 550, 650), 2)
    assignments = Pipeline._assign_persons_to_vehicles([vehicle], [person])
    assert assignments[id(vehicle)] == []


def test_file_frame_stride_five_keeps_first_and_every_fifth_frame() -> None:
    selected = [
        frame_number
        for frame_number in range(1, 18)
        if Pipeline._should_process_file_frame(frame_number, 5)
    ]

    assert selected == [1, 6, 11, 16]


def test_file_frame_stride_one_keeps_every_frame() -> None:
    assert all(
        Pipeline._should_process_file_frame(frame_number, 1)
        for frame_number in range(1, 18)
    )


def test_invalid_file_frame_stride_safely_keeps_every_frame() -> None:
    assert Pipeline._should_process_file_frame(2, 0) is True
    assert Pipeline._should_process_file_frame(2, -4) is True


def test_primary_analysis_preserves_highest_risk_vehicle_fields() -> None:
    vehicle = detection("car", (0, 0, 100, 100), 1)
    analysis = VehicleAnalysis(
        vehicle=vehicle,
        plate_text="LAG123XY",
        plate_confidence=0.91,
        matched_vehicle_id=7,
        plate_in_database=True,
        vehicle_flagged_stolen=True,
        person_status=PersonStatus.UNAUTHORIZED,
        face_similarity=0.22,
        matched_user_id=4,
        assessment=ThreatAssessment(
            score=95,
            level=ThreatLevel.CRITICAL,
            reason="reported stolen vehicle",
        ),
    )
    result = FrameResult(frame_number=1, timestamp=0)

    Pipeline._apply_primary_analysis(result, analysis)

    assert result.plate_text == "LAG123XY"
    assert result.matched_vehicle_id == 7
    assert result.vehicle_flagged_stolen is True
    assert result.person_status == PersonStatus.UNAUTHORIZED
    assert result.threat_score == 95
    assert result.threat_level_name == ThreatLevel.CRITICAL.value


def test_registered_vehicle_owner_is_verified() -> None:
    analysis = VehicleAnalysis(
        vehicle=detection("car", (0, 0, 100, 100), 1),
        plate_in_database=True,
        expected_owner_id=7,
        matched_user_id=7,
        person_status=PersonStatus.AUTHORIZED,
    )

    Pipeline._verify_vehicle_owner(analysis)

    assert analysis.owner_verification is OwnerVerificationStatus.VERIFIED
    assert analysis.person_status is PersonStatus.AUTHORIZED


def test_different_registered_owner_is_rejected_for_vehicle() -> None:
    analysis = VehicleAnalysis(
        vehicle=detection("car", (0, 0, 100, 100), 1),
        plate_in_database=True,
        expected_owner_id=7,
        matched_user_id=9,
        person_status=PersonStatus.AUTHORIZED,
    )

    Pipeline._verify_vehicle_owner(analysis)

    assert analysis.owner_verification is OwnerVerificationStatus.MISMATCH
    assert analysis.person_status is PersonStatus.UNAUTHORIZED


def test_registered_vehicle_without_face_match_is_not_recognized() -> None:
    analysis = VehicleAnalysis(
        vehicle=detection("car", (0, 0, 100, 100), 1),
        plate_in_database=True,
        expected_owner_id=7,
    )

    Pipeline._verify_vehicle_owner(analysis)

    assert analysis.owner_verification is OwnerVerificationStatus.NOT_RECOGNIZED


def test_unregistered_vehicle_owner_verification_is_not_applicable() -> None:
    analysis = VehicleAnalysis(
        vehicle=detection("car", (0, 0, 100, 100), 1),
        plate_in_database=False,
        matched_user_id=7,
    )

    Pipeline._verify_vehicle_owner(analysis)

    assert analysis.owner_verification is OwnerVerificationStatus.NOT_APPLICABLE
