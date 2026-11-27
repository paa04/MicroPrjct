# External data acquisition — Moscow only (2026-11-27)

## Priority and provenance

| Source | Data | Access/status | Temporal use |
|---|---|---|---|
| ADG portfolio | 39 addresses from the 2014 acquisition | Curated CSV committed; geocoding still needs address verification | Portfolio membership known by 2014-12-15; does not imply building available or reconstructed |
| ADG DDX announcements | Five sites announced 2021-10-19, four 2024-02-06 | Curated CSV committed | Public pipeline forecast only, NOT blind location prediction |
| [ShopAndMall](https://shopandmall.ru/vse-torgovye-centry/moskva) | 355 listed shopping-center offers, address, GBA, GLA, opening date | Public catalog, full export not obtained; check license before collection | Validate dates and historical owner; current tenant lists can leak outcomes |
| [M2DATA](https://m2data.net/ru/api) | Commercial building/space area, rent, date and reason of listing withdrawal; provider reports 57,000+ histories across its markets | Commercial API; Moscow subset and large retail coverage require confirmation | Best prospective paid source for dated supply; withdrawal isn't necessarily transaction |
| [Historical stationary retail register](https://data.apicrafter.ru/packages/datamos-sto) | 42,852 records, operator, type, address, geometry | CC-BY-SA per catalog; 2021-10-23 archive, 3.4MB zip. Download blocked in this runtime | Valid from 2021-10-23, not for earlier forecasts |
| [Historical city lease auction register](https://data.apicrafter.ru/packages/datamos-objectsrealestate) | 2,611 records, stages of lease auctions | CC-BY-SA per catalog; 2021-10-23 archive, 259KB zip. Download blocked | City property only, not private mall supply |
| [Moscow auctions](https://torgi.mos.ru/) / [GIS Torgi](https://torgi.gov.ru/) | Dated public offers and auction outcomes | Public portal; bulk retrieval not verified | Filter Moscow, area, publication date, lease type; city-biased |
| [GIS ЖКХ dataset](https://tochno.st/datasets/gisgkh) | Building year, residential and total area, address | CC BY 4.0, CSV/XLSX/Parquet per source; Moscow subset not downloaded | Current building stock with year-built filter is retrospective, not true historical stock |
| [GHSL GHS-POP](https://doi.org/10.2905/JRC.CXKEDRR) | Population estimates at 100m, five-year epochs | Public raster; crop Moscow only. Not downloaded | R2023A is retrospective/reconstructed; 2025 epoch is model projection, not real-time counts |
| [ohsome API](https://docs.ohsome.org/ohsome-api/development/endpoints.html) | Historical OSM geometry and tags | Public extraction API; planned requests in fetch_historical_osm.js | Request past dates; timestamp is mapping state, not actual opening or lease availability |
| [DOM.RF data platform](https://платформа.дом.рф/) | New development, commercial/nonresidential premises and building commissioning | Access and dataset-specific terms to verify; no download | Filter by document publication timestamp, not just planned completion |
| [Sber geoanalytics](https://sberanalytics.ru/products/geoanalitica) / [MTS GeoEffect](https://geoeffect.ru/business) | Aggregated population, mobility, spending, traffic | Commercial, no account or data obtained | Ask for archived dates and aggregate grid export, not individual records |

Direct archive links as published in the catalog (not downloaded):
- https://data.apicrafter.ru/packages/datamos-sto/build/datamos-7710881420-STO-2021-10-23-7-45/get
- https://data.apicrafter.ru/packages/datamos-objectsrealestate/build/datamos-7710168515-objectsrealestate-2021-10-23-7-42/get

## What has actually been obtained

- adg_portfolio_2014.csv: 39 independently listed 2014 addresses, current catalog listing flags and five address variants. Original address from Moscow 24 (2014-12-15), current list from ADG. Current fields must not enter earlier forecasts.
- adg_announcements.csv: nine announced projects. Use for announcement-aware forecasts only; segregate from blind validation.
- adg_osm_name_candidates.csv: 35 name-match candidates for 27 distinct centers, NOT verified geocodes. Some names are ambiguous or wrong (e.g. Praha). Match addresses independently before using coordinates. Do not copy DDX pins to establish the owner layer.
- adg_ddx_address_audit.csv: 16 manually linked DDX locations for audit only: 10/28 training, 4/17 time-test and two district holdout. This is not a feature layer and must never establish the universe of eligible buildings.

Primary evidence: https://www.adggroup.ru/news-ddx-fitness (2024-02-06), https://www.adggroup.ru/ddx-fitness-mesto-vstrechi (2021), https://cre.ru/players/company/adg-group (2021-10-19 dated article index), https://www.adggroup.ru/news/razresheniya-na-stroitelstvo-polucheny-dlya-20-iz-39-obektov (2018), https://www.m24.ru/articles/15122014/62327 (2014).

The portfolio acquisition and package agreements imply clustered firm decisions. Group by owner/deal in sensitivity validation, not just spatial tile. Record public date, asset opening date, vacancy date, and announcement date separately; do not treat absent data as proof of ineligibility.

## Proposed ingestion schema

asset_id, address, lon, lat, coordinate_source, coordinate_verified, owner_id, owner_valid_from, owner_valid_to, building_type, building_open_date, gba_m2, gla_m2, contiguous_space_m2, ceiling_height_m, rent_rub_m2_year, listing_published, listing_withdrawn, snapshot_date, source_url, known_at, confidence.

Model a time-specific feasible set, then rank by expected incremental network demand minus observed/scenario costs. When vacancy is unknown, model availability probability rather than pretend to know it. No use of post-opening tenant identity or DDX labels in candidate features. Keep the 2025+ reused test descriptive; future unannounced openings are necessary for independent evaluation.

Network limitation: container DNS failed for external sites; web browsing verified source pages but did not retrieve full archives or paid data. Only the four small curated CSVs and acquisition plan are actually present. No claim of measured model improvement from the new layer yet.
