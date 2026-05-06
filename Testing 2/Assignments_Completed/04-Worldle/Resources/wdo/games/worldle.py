"""Helpers for the notebook-based Worldle assignment."""

from __future__ import annotations

import base64
import html
import random
import unicodedata
from pathlib import Path

from wdo.geometry.bbox import bbox_from_feature, iter_coordinates
from wdo.geometry.bearing import bearing_to_compass, initial_bearing
from wdo.geometry.distance import haversine_km, haversine_miles


ARROWS = {
    "N": "↑",
    "NE": "↗",
    "E": "→",
    "SE": "↘",
    "S": "↓",
    "SW": "↙",
    "W": "←",
    "NW": "↖",
}

NAME_KEYS = ("ADMIN", "name", "NAME", "NAME_EN", "SOVEREIGNT")
ISO3_KEYS = ("ISO_A3", "ISO3166-1-Alpha-3", "ADM0_A3", "iso_a3")
ISO2_KEYS = ("ISO_A2", "ISO3166-1-Alpha-2", "iso_a2")

DEFAULT_ALIASES = {
    "bahamas": "bahamas",
    "bolivia": "bolivia",
    "brunei": "brunei",
    "cape verde": "cape verde",
    "congo": "republic of the congo",
    "democratic republic of the congo": "dr congo",
    "czech republic": "czechia",
    "ivory coast": "cote d'ivoire",
    "laos": "laos",
    "moldova": "moldova",
    "north korea": "north korea",
    "russia": "russia",
    "south korea": "south korea",
    "syria": "syria",
    "taiwan": "taiwan",
    "tanzania": "tanzania",
    "united kingdom": "united kingdom",
    "united states of america": "united states",
    "venezuela": "venezuela",
    "vietnam": "vietnam",
}


def _properties(feature):
    return feature.get("properties", {}) if isinstance(feature, dict) else {}


def _first_property(feature, keys, default=None):
    props = _properties(feature)
    for key in keys:
        value = props.get(key)
        if value not in (None, "", "-99"):
            return value
    return default


def country_name(feature, default="Unknown"):
    """Return the best available display name for a country feature."""
    return str(_first_property(feature, NAME_KEYS, default))


def iso3_code(feature):
    """Return a feature's ISO-3 code when one is present."""
    value = _first_property(feature, ISO3_KEYS)
    return str(value).upper() if value else None


def iso2_code(feature):
    """Return a feature's ISO-2 code when one is present."""
    value = _first_property(feature, ISO2_KEYS)
    return str(value).lower() if value else None


def normalize_name(name):
    """Normalize a country name for forgiving joins."""
    text = unicodedata.normalize("NFKD", str(name or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    keep = [ch.lower() if ch.isalnum() else " " for ch in text]
    return " ".join("".join(keep).split())


def choose_target(features, seed=None):
    """Pick one country feature at random, reproducibly when ``seed`` is given."""
    feature_list = list(features)
    if not feature_list:
        raise ValueError("choose_target requires at least one feature")
    return random.Random(seed).choice(feature_list)


def feature_center(feature, method="bbox"):
    """Return a representative ``(lat, lon)`` point for a Polygon/MultiPolygon."""
    if method == "bbox":
        min_lon, min_lat, max_lon, max_lat = bbox_from_feature(feature)
        return ((min_lat + max_lat) / 2, (min_lon + max_lon) / 2)

    if method == "mean":
        points = list(iter_coordinates(feature))
        if not points:
            raise ValueError("feature_center could not find coordinates")
        lon = sum(point[0] for point in points) / len(points)
        lat = sum(point[1] for point in points) / len(points)
        return (lat, lon)

    raise ValueError("method must be 'bbox' or 'mean'")


def _same_country(left, right):
    left_iso = iso3_code(left)
    right_iso = iso3_code(right)
    if left_iso and right_iso:
        return left_iso == right_iso
    return normalize_name(country_name(left)) == normalize_name(country_name(right))


def guess_feedback(guess_feature, target_feature):
    """Compare a guess to the target and return distance, bearing, compass, and arrow."""
    guess_center = feature_center(guess_feature)
    target_center = feature_center(target_feature)
    correct = _same_country(guess_feature, target_feature)

    distance_km = 0.0 if correct else haversine_km(guess_center, target_center)
    distance_miles = 0.0 if correct else haversine_miles(guess_center, target_center)
    bearing_deg = 0.0 if correct else initial_bearing(guess_center, target_center)
    compass = "HERE" if correct else bearing_to_compass(bearing_deg)
    arrow = "✓" if correct else ARROWS[compass]

    return {
        "correct": correct,
        "guess_name": country_name(guess_feature),
        "guess_iso3": iso3_code(guess_feature),
        "target_name": country_name(target_feature),
        "target_iso3": iso3_code(target_feature),
        "guess_center": guess_center,
        "target_center": target_center,
        "distance_km": distance_km,
        "distance_miles": distance_miles,
        "bearing_deg": bearing_deg,
        "compass": compass,
        "arrow": arrow,
    }


def format_feedback(result, units="km") -> str:
    """Pretty-print feedback for logging or plain-text testing."""
    if result["correct"]:
        return f"{result['guess_name']}: correct"

    if units == "miles":
        distance = result["distance_miles"]
        label = "mi"
    else:
        distance = result["distance_km"]
        label = "km"

    return (
        f"{result['guess_name']}: {distance:,.0f} {label} "
        f"{result['arrow']} {result['compass']}"
    )


def _flag_index_by_code(flag_index):
    if isinstance(flag_index, dict):
        values = flag_index.values()
    else:
        values = flag_index
    return {
        str(item.get("code", "")).lower(): item
        for item in values
        if isinstance(item, dict) and item.get("code")
    }


def build_country_lookup(countries_geojson, flag_index, aliases=None, report_misses=False):
    """Join country features to flag metadata and return an ISO-3 keyed lookup."""
    aliases = {**DEFAULT_ALIASES, **(aliases or {})}
    flags_by_code = _flag_index_by_code(flag_index)
    flags_by_name = {
        normalize_name(item.get("name")): item
        for item in flags_by_code.values()
        if item.get("name")
    }

    features = countries_geojson.get("features", [])
    lookup = {}
    misses = []

    for feature in features:
        iso3 = iso3_code(feature) or normalize_name(country_name(feature)).upper()
        name = country_name(feature)
        iso2 = iso2_code(feature)
        flag = flags_by_code.get(iso2) if iso2 else None

        if flag is None:
            normalized = normalize_name(name)
            alias = aliases.get(normalized, normalized)
            flag = flags_by_name.get(alias)

        if flag is None:
            misses.append(name)
            iso2 = None
            flag_path = None
        else:
            iso2 = str(flag.get("code", iso2 or "")).lower()
            flag_path = str(Path("Resources") / "Data" / "flag-icons" / flag["flag_4x3"])

        lookup[iso3] = {
            "name": name,
            "iso2": iso2,
            "flag_path": flag_path,
            "feature": feature,
        }

    if report_misses and misses:
        print("Flag join misses:", ", ".join(sorted(misses)))

    return lookup


def flag_to_data_uri(flag_path, base_dir=None):
    """Return an SVG flag as a data URI so notebook HTML stays portable."""
    if not flag_path:
        return None
    path = Path(flag_path)
    if base_dir is not None and not path.is_absolute():
        path = Path(base_dir) / path
    if not path.exists():
        return None
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/svg+xml;base64,{payload}"


def proximity_color(distance_km, correct=False):
    """Return a readable status color based on distance."""
    if correct or distance_km <= 1:
        return "#2a9d8f"
    if distance_km < 750:
        return "#7cb342"
    if distance_km < 2500:
        return "#f4a261"
    return "#e76f51"


def render_guess_row(country_name, flag_path, arrow, distance_km, correct=False):
    """Return an HTML row for the guess history."""
    safe_name = html.escape(str(country_name))
    color = proximity_color(distance_km, correct)
    if flag_path:
        src = html.escape(str(flag_path), quote=True)
        flag_html = (
            f'<img src="{src}" width="38" '
            'style="border-radius:3px;border:1px solid #d9e2e8">'
        )
    else:
        flag_html = '<span style="width:38px;text-align:center">--</span>'

    distance = "correct" if correct else f"{distance_km:,.0f} km"
    return f"""
    <div style="display:grid;grid-template-columns:48px 1fr 44px 96px;align-items:center;
                gap:10px;padding:8px 10px;border-bottom:1px solid #dfe7ea;
                font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif">
      <div>{flag_html}</div>
      <div style="font-weight:650;color:#17324d">{safe_name}</div>
      <div style="font-size:24px;text-align:center;color:{color}">{html.escape(str(arrow))}</div>
      <div style="font-weight:700;text-align:right;color:{color}">{distance}</div>
    </div>
    """


def share_result(history):
    """Build a compact text summary of a finished round."""
    lines = []
    for item in history:
        if item["correct"]:
            lines.append(f"✓ {item['guess_name']}")
        else:
            lines.append(
                f"{item['arrow']} {item['distance_km']:,.0f} km - {item['guess_name']}"
            )
    return "\n".join(lines)


__all__ = [
    "ARROWS",
    "build_country_lookup",
    "choose_target",
    "country_name",
    "feature_center",
    "flag_to_data_uri",
    "format_feedback",
    "guess_feedback",
    "iso2_code",
    "iso3_code",
    "normalize_name",
    "proximity_color",
    "render_guess_row",
    "share_result",
]
