"""Bounding-box helpers for GeoJSON geometries."""


def _is_position(value):
    return (
        isinstance(value, (list, tuple))
        and len(value) >= 2
        and isinstance(value[0], (int, float))
        and isinstance(value[1], (int, float))
    )


def iter_coordinates(geometry_or_feature):
    """Yield ``(lon, lat)`` pairs from GeoJSON geometry, feature, or collection data."""
    if not isinstance(geometry_or_feature, dict):
        return

    kind = geometry_or_feature.get("type")

    if kind == "Feature":
        yield from iter_coordinates(geometry_or_feature.get("geometry", {}))
        return

    if kind == "FeatureCollection":
        for feature in geometry_or_feature.get("features", []):
            yield from iter_coordinates(feature)
        return

    if kind == "GeometryCollection":
        for geometry in geometry_or_feature.get("geometries", []):
            yield from iter_coordinates(geometry)
        return

    coordinates = geometry_or_feature.get("coordinates")
    if coordinates is None:
        return

    stack = [coordinates]
    while stack:
        item = stack.pop()
        if _is_position(item):
            yield (float(item[0]), float(item[1]))
        elif isinstance(item, (list, tuple)):
            stack.extend(reversed(item))


def bbox_from_points(points):
    """Return bbox as ``(min_lon, min_lat, max_lon, max_lat)``."""
    clean_points = [(float(lon), float(lat)) for lon, lat in points]
    if not clean_points:
        raise ValueError("bbox_from_points requires at least one point")

    lons = [point[0] for point in clean_points]
    lats = [point[1] for point in clean_points]
    return (min(lons), min(lats), max(lons), max(lats))


def bbox_from_feature(feature):
    """Extract all coordinates from a feature and compute bbox."""
    return bbox_from_points(iter_coordinates(feature))


def bbox_from_features(features):
    """Compute bbox across multiple features."""
    if isinstance(features, dict) and features.get("type") == "FeatureCollection":
        features = features.get("features", [])

    points = []
    for feature in features:
        points.extend(iter_coordinates(feature))
    return bbox_from_points(points)


def bbox_to_polygon(bbox):
    """Convert bbox tuple into a closed polygon coordinate list."""
    min_lon, min_lat, max_lon, max_lat = bbox
    return [
        (min_lon, min_lat),
        (max_lon, min_lat),
        (max_lon, max_lat),
        (min_lon, max_lat),
        (min_lon, min_lat),
    ]
