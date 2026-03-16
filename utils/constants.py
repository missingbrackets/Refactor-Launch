from zoneinfo import ZoneInfo
from pathlib import Path
# -------------------- Configuration --------------------
GROUPINGS = [
    "vehicle_type",
    "vehicle_family",
    "launch_provider",
    "vehicle_variant",         
    "vehicle_minor_variant",   
]

OUTPUT_DIR = Path("src/launch_analysis/outputs")

TZ = ZoneInfo("Europe/London")  # consistent, explicit timezone for filenames

ATTEMPT_COLS = [
    "lv_type_attempt_number",
    "lv_family_attempt_number",
    "lv_provider_attempt_number",
    "lv_variant_attempt_number",
    "lv_minor_variant_attempt_number",
]

IDENTITY_COLS = [
    "vehicle_type",
    "launch_type",
    "vehicle_family",
    "launch_provider",
    "vehicle_variant",
    "vehicle_minor_variant",
]

# Which identity columns to include for each grouping (your “twist”)
INCLUDE_BY_GROUPING = {
    # include up to this specificity, exclude more specific ones
    "launch_provider": ["launch_provider"],
    "vehicle_family": ["launch_provider", "vehicle_family"],
    "vehicle_type": ["launch_provider", "vehicle_family", "vehicle_type"],
    "vehicle_variant": ["launch_provider", "vehicle_family", "vehicle_type", "vehicle_variant"],
    "vehicle_minor_variant": [
        "launch_provider",
        "vehicle_family",
        "vehicle_type",
        "vehicle_variant",
        "vehicle_minor_variant",
    ],
    "launch_type": ["launch_type"],  # if you ever include this as a GROUPING
}

# Map the human/UX grouping label -> (file key used in filenames, column name in the file)
SPECIFIC_GROUPINGS: dict[str, tuple[str, str]] = {
    "Launch Provider": ("launch_provider", "launch_provider"),
    "Vehicle Family": ("vehicle_family", "vehicle_family"),
    "Vehicle Minor Variant": ("vehicle_minor_variant", "vehicle_minor_variant"),
    "Vehicle Type": ("vehicle_type", "vehicle_type"),
    "Vehicle Variant": ("vehicle_variant", "vehicle_variant"),
}

SPECIFIC_COLS = [
    "next_launch_number_specific",
    "total_failures_specific",
    "failure_rate_specific",
    "ci_lower_specific",
    "ci_upper_specific",
]
TYPE_COLS = [
    "next_launch_number_type",
    "total_failures_type",
    "failure_rate_type",
    "ci_lower_type",
    "ci_upper_type",
]

# --- Levels mapping (ordered) ---
LEVELS = [
    ("type",          "vehicle_type",         "lv_type_attempt_number"),
    ("family",        "vehicle_family",       "lv_family_attempt_number"),
    ("provider",      "launch_provider",      "lv_provider_attempt_number"),
    ("variant",       "vehicle_variant",      "lv_variant_attempt_number"),
    ("minor_variant", "vehicle_minor_variant","lv_minor_variant_attempt_number"),
]