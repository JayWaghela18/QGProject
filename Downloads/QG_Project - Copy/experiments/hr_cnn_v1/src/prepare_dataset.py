from pathlib import Path
import json

UBFC_ROOT = Path(
    r"G:\.shortcut-targets-by-id\1o0XU4gTIo46YfwaWjIgbtCncc-oF44Xk\UBFC_DATASET"
)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
MANIFEST_PATH = OUTPUT_DIR / "dataset_manifest.json"


def find_subjects(dataset_dir: Path) -> list[dict]:
    subjects = []

    if not dataset_dir.exists():
        print(f"Dataset folder not found: {dataset_dir}")
        return subjects

    for subject_dir in sorted(dataset_dir.iterdir()):
        if not subject_dir.is_dir():
            continue

        video = subject_dir / "vid.avi"
        ground_truth = subject_dir / "ground_truth.txt"

        if video.is_file() and ground_truth.is_file():
            subjects.append({
                "subject": subject_dir.name,
                "dataset": dataset_dir.name,
                "video_path": str(video),
                "ground_truth_path": str(ground_truth),
            })

    return subjects


def main():
    all_subjects = []

    for dataset_name in ("DATASET_1", "DATASET_2"):
        dataset_dir = UBFC_ROOT / dataset_name
        subjects = find_subjects(dataset_dir)
        all_subjects.extend(subjects)
        print(f"{dataset_name}: {len(subjects)} usable subject folders")

    if not all_subjects:
        raise RuntimeError(
            "No subject folders containing both vid.avi "
            "and ground_truth.txt were found."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(all_subjects, indent=2),
        encoding="utf-8",
    )

    print(f"\nTotal usable subjects: {len(all_subjects)}")
    print(f"Manifest saved to: {MANIFEST_PATH}")
    print("\nFirst five entries:")

    for item in all_subjects[:5]:
        print(f"  {item['dataset']}/{item['subject']}")


if __name__ == "__main__":
    main()