"""Build a reproducible Maryino POI snapshot from OpenStreetMap API XML.

Run fetch_maryino_osm.sh first. Uses only Python's standard library.
"""

import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw_osm"
OUT = ROOT / "data" / "maryino"


def read_osm(path):
    nodes, ways, relations = {}, {}, {}
    for _, elem in ET.iterparse(path, events=("end",)):
        if elem.tag == "node":
            nodes[int(elem.attrib["id"])] = (
                float(elem.attrib["lon"]), float(elem.attrib["lat"]),
                {t.attrib["k"]: t.attrib["v"] for t in elem.findall("tag")},
            )
            elem.clear()
        elif elem.tag == "way":
            ways[int(elem.attrib["id"])] = (
                [int(n.attrib["ref"]) for n in elem.findall("nd")],
                {t.attrib["k"]: t.attrib["v"] for t in elem.findall("tag")},
            )
            elem.clear()
        elif elem.tag == "relation":
            relations[int(elem.attrib["id"])] = (
                [(m.attrib["type"], int(m.attrib["ref"]), m.attrib.get("role", ""))
                 for m in elem.findall("member")],
                {t.attrib["k"]: t.attrib["v"] for t in elem.findall("tag")},
            )
            elem.clear()
    return nodes, ways, relations


def rings_from_members(members, ways, nodes, role):
    segments = [ways[ref][0] for typ, ref, member_role in members
                if typ == "way" and member_role == role and ref in ways]
    rings = []
    while segments:
        ring = list(segments.pop())
        while ring[0] != ring[-1]:
            for i, segment in enumerate(segments):
                if ring[-1] == segment[0]:
                    ring.extend(segment[1:])
                elif ring[-1] == segment[-1]:
                    ring.extend(reversed(segment[:-1]))
                else:
                    continue
                segments.pop(i)
                break
            else:
                raise ValueError("Incomplete boundary ring")
        rings.append([nodes[ref][:2] for ref in ring])
    return rings


def inside_ring(point, ring):
    x, y = point
    inside = False
    for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def classify(tags):
    railway, station, public_transport = (tags.get(k, "") for k in
                                          ("railway", "station", "public_transport"))
    if (railway == "station" and station == "subway") or tags.get("subway") == "yes" and railway == "station":
        return "metro_station"
    if railway == "subway_entrance":
        return "metro_entrance"
    if public_transport == "platform" and tags.get("subway") == "yes":
        return "metro_platform"
    if tags.get("highway") == "bus_stop" or tags.get("trolleybus") == "yes" and public_transport == "platform":
        return "bus_stop"
    if tags.get("leisure") == "fitness_centre":
        return "fitness_club"
    if tags.get("leisure") == "fitness_station":
        return "outdoor_fitness"
    if tags.get("shop") == "mall":
        return "shopping_mall"
    if tags.get("shop") in {"supermarket", "department_store", "hypermarket"}:
        return "large_store"
    if tags.get("building") == "retail":
        return "retail_building"
    if tags.get("building") == "office":
        return "office_building"
    if tags.get("office") and not tags.get("shop"):
        return "office_service"
    if tags.get("building") == "commercial" or tags.get("landuse") == "commercial":
        return "commercial"
    if tags.get("building") in {"apartments", "residential", "dormitory", "house", "detached"}:
        return "residential_building"
    if tags.get("landuse") == "residential":
        return "residential_area"
    if tags.get("amenity") in {"school", "university", "college", "kindergarten"}:
        return "education"
    if tags.get("amenity") in {"clinic", "hospital", "doctors"}:
        return "healthcare"
    return None


def centroid(coords):
    if len(coords) < 3 or coords[0] != coords[-1]:
        return tuple(sum(p[i] for p in coords) / len(coords) for i in (0, 1))
    lat0 = math.radians(sum(p[1] for p in coords) / len(coords))
    scale = math.cos(lat0)
    origin_x, origin_y = coords[0][0] * scale, coords[0][1]
    area2 = cx = cy = 0.0
    for (lon1, lat1), (lon2, lat2) in zip(coords, coords[1:]):
        x1, x2 = lon1 * scale - origin_x, lon2 * scale - origin_x
        y1, y2 = lat1 - origin_y, lat2 - origin_y
        cross = x1 * y2 - x2 * y1
        area2 += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    if abs(area2) < 1e-16:
        return tuple(sum(p[i] for p in coords[:-1]) / (len(coords) - 1) for i in (0, 1))
    return (origin_x + cx / (3 * area2)) / scale, origin_y + cy / (3 * area2)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bnodes, bways, brelations = read_osm(RAW / "boundary.osm")
    members, boundary_tags = brelations[455222]
    outer = rings_from_members(members, bways, bnodes, "outer")
    inner = rings_from_members(members, bways, bnodes, "inner")
    def in_district(point):
        return any(inside_ring(point, ring) for ring in outer) and not any(
            inside_ring(point, ring) for ring in inner)

    nodes, ways, relations = {}, {}, {}
    for path in sorted(RAW.glob("tile_*.osm")):
        n, w, r = read_osm(path)
        nodes.update(n)
        ways.update(w)
        relations.update(r)
    if not nodes:
        raise ValueError("No tile files; run fetch_maryino_osm.sh")

    records = []
    for node_id, (lon, lat, tags) in nodes.items():
        category = classify(tags)
        if category and in_district((lon, lat)):
            records.append(("node", node_id, category, lon, lat, tags))
    for way_id, (refs, tags) in ways.items():
        category = classify(tags)
        coords = [nodes[ref][:2] for ref in refs if ref in nodes]
        if category and len(coords) == len(refs) and coords:
            lon, lat = centroid(coords)
            if in_district((lon, lat)):
                records.append(("way", way_id, category, lon, lat, tags))
    for relation_id, (members, tags) in relations.items():
        category = classify(tags)
        if not category:
            continue
        coords = [nodes[ref][:2] for typ, way_ref, role in members if typ == "way" and role == "outer"
                  for ref in ways.get(way_ref, ([], {}))[0] if ref in nodes]
        if coords:
            lon, lat = centroid(coords)
            if in_district((lon, lat)):
                records.append(("relation", relation_id, category, lon, lat, tags))

    records.sort(key=lambda r: (r[2], r[0], r[1]))
    fields = ["osm_type", "osm_id", "category", "name", "lon", "lat", "address", "brand", "operator", "osm_url"]
    with (OUT / "poi.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for typ, osm_id, category, lon, lat, tags in records:
            address = ", ".join(filter(None, [tags.get("addr:street"), tags.get("addr:housenumber")]))
            writer.writerow(dict(osm_type=typ, osm_id=osm_id, category=category,
                                 name=tags.get("name", ""), lon=f"{lon:.7f}", lat=f"{lat:.7f}",
                                 address=address, brand=tags.get("brand", ""),
                                 operator=tags.get("operator", ""),
                                 osm_url=f"https://www.openstreetmap.org/{typ}/{osm_id}"))
    features = [dict(type="Feature", geometry=dict(type="Point", coordinates=[lon, lat]),
                     properties=dict(osm_type=typ, osm_id=osm_id, category=category, **tags))
                for typ, osm_id, category, lon, lat, tags in records]
    (OUT / "poi.geojson").write_text(json.dumps(dict(type="FeatureCollection", features=features), ensure_ascii=False), encoding="utf-8")
    boundary = dict(type="FeatureCollection", features=[dict(type="Feature", properties=boundary_tags,
        geometry=dict(type="Polygon", coordinates=outer + inner))])
    (OUT / "boundary.geojson").write_text(json.dumps(boundary, ensure_ascii=False), encoding="utf-8")
    summary = dict(generated_utc=datetime.now(timezone.utc).isoformat(), source="OpenStreetMap API 0.6",
                   boundary_relation=455222, tiles=len(list(RAW.glob("tile_*.osm"))),
                   counts=dict(sorted(Counter(r[2] for r in records).items())), total=len(records))
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
