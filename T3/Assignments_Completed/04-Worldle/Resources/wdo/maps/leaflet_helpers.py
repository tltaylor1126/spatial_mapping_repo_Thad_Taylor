"""Small ipyleaflet wrappers used by the mapping notebooks."""

from wdo.geometry.bbox import bbox_from_feature, bbox_from_features, bbox_to_polygon


def _leaflet():
    try:
        import ipyleaflet as leaflet
    except ImportError as exc:
        raise ImportError(
            "ipyleaflet is required for wdo.maps.leaflet_helpers. "
            "Install it in the notebook environment before running map cells."
        ) from exc
    return leaflet


def _basemap_from_name(leaflet, name):
    if name is None:
        return leaflet.basemaps.CartoDB.Positron
    if not isinstance(name, str):
        return name

    lookup = {
        "openstreetmap": leaflet.basemaps.OpenStreetMap.Mapnik,
        "osm": leaflet.basemaps.OpenStreetMap.Mapnik,
        "positron": leaflet.basemaps.CartoDB.Positron,
        "dark": leaflet.basemaps.CartoDB.DarkMatter,
        "satellite": leaflet.basemaps.Esri.WorldImagery,
    }
    return lookup.get(name.lower(), leaflet.basemaps.CartoDB.Positron)


def make_map(center=(0, 0), zoom=2, basemap=None, scroll_wheel_zoom=True, **kwargs):
    """Return an ipyleaflet map with sensible notebook defaults."""
    leaflet = _leaflet()
    return leaflet.Map(
        center=center,
        zoom=zoom,
        basemap=_basemap_from_name(leaflet, basemap),
        scroll_wheel_zoom=scroll_wheel_zoom,
        **kwargs,
    )


def add_basemap(map_obj, name="OpenStreetMap"):
    """Add/select a basemap layer."""
    leaflet = _leaflet()
    layer = leaflet.basemap_to_tiles(_basemap_from_name(leaflet, name))
    map_obj.add_layer(layer)
    return layer


def add_geojson(map_obj, data, name=None, style=None):
    """Add GeoJSON data to a map."""
    leaflet = _leaflet()
    layer = leaflet.GeoJSON(
        data=data,
        name=name or "GeoJSON",
        style=style
        or {
            "color": "#1f2937",
            "fillColor": "#2a9d8f",
            "weight": 2,
            "fillOpacity": 0.35,
        },
    )
    map_obj.add_layer(layer)
    return layer


def fit_map_to_geojson(map_obj, data):
    """Adjust map viewport to fit GeoJSON bounds."""
    if isinstance(data, dict) and data.get("type") == "Feature":
        bbox = bbox_from_feature(data)
    elif isinstance(data, dict) and data.get("type") == "FeatureCollection":
        bbox = bbox_from_features(data)
    else:
        bbox = bbox_from_features(data)

    min_lon, min_lat, max_lon, max_lat = bbox
    map_obj.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])
    return map_obj


def add_layer_control(map_obj):
    """Add layer control widget."""
    leaflet = _leaflet()
    control = leaflet.LayersControl(position="topright")
    map_obj.add_control(control)
    return control


def add_scale_control(map_obj):
    """Add scale control widget."""
    leaflet = _leaflet()
    control = leaflet.ScaleControl(position="bottomleft")
    map_obj.add_control(control)
    return control


def add_bbox(map_obj, bbox, **style):
    """Draw a bounding box on the map."""
    leaflet = _leaflet()
    ring = [(lat, lon) for lon, lat in bbox_to_polygon(bbox)]
    layer = leaflet.Polygon(
        locations=ring,
        color=style.get("color", "#264653"),
        fill_color=style.get("fillColor", style.get("fill_color", "#e9c46a")),
        fill_opacity=style.get("fillOpacity", style.get("fill_opacity", 0.12)),
        weight=style.get("weight", 2),
        name=style.get("name", "Bounding box"),
    )
    map_obj.add_layer(layer)
    return layer


def add_path(map_obj, coords, **style):
    """Add a path/polyline to the map."""
    leaflet = _leaflet()
    layer = leaflet.Polyline(
        locations=list(coords),
        color=style.get("color", "#e76f51"),
        weight=style.get("weight", 3),
        opacity=style.get("opacity", 0.85),
        name=style.get("name", "Path"),
    )
    map_obj.add_layer(layer)
    return layer
