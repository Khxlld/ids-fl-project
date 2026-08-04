"""Build the Phase 1 UAVIDS-2025 raw-dataset audit notebook."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "notebooks" / "04_uavids_dataset_audit.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    markdown(
        r"""
# Phase 1 — UAVIDS-2025 raw-dataset audit

This notebook audits the published UAVIDS-2025 CSV without cleaning, deleting,
partitioning, or training. Its purpose is to document the schema, test data-quality
and leakage risks, and propose—but not create—five source-based logical clients.

Interpretation labels used below:

- **Documented fact**: stated in the dataset record or associated paper.
- **Evidence-based inference**: supported by values in the released CSV.
- **Unresolved**: not established by the available documentation or CSV.

The raw CSV is read only. Generated audit tables are written to `results_audit/`.
"""
    ),
    markdown(
        r"""
## 1. Evidence base and scope

Primary sources:

- [Official Zenodo dataset record (DOI 10.5281/zenodo.15336998)](https://doi.org/10.5281/zenodo.15336998)
- [Associated IEEE CNS 2025 paper (DOI 10.1109/CNS66487.2025.11194990)](https://doi.org/10.1109/CNS66487.2025.11194990)

The paper documents an NS-3.24 UAV-network simulation using IEEE 802.11ac,
AODV, BOID mobility, UDP traffic, and the Nakagami channel model. It states that
each simulation instance lasted 600 seconds, packet captures were deployed at all
UAV nodes, flows were based on the network 5-tuple, and scenario labels were
assigned from synchronized attack schedules.

The released record contains only the CSV; it does **not** provide a topology,
scenario/run identifier, capture-point identifier, or IP-to-physical-UAV map.
Consequently, `SrcAddr` is documented as the source IP of a communication, but an
individual address cannot be claimed to be a distinct physical UAV. This matters
especially for Sybil traffic, where the paper says an adversarial UAV forges
multiple identities or addresses.
"""
    ),
    markdown(
        r"""
## 2. Imports, paths, and integrity checks

The official Zenodo MD5 is checked before analysis. Path detection supports
execution from either this experiment root or its `notebooks` directory.
"""
    ),
    code(
        r"""
import hashlib
import ipaddress
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import Markdown, display

pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 220)
pd.set_option("display.width", 180)

HERE = Path.cwd().resolve()
if (HERE / "notebooks" / "04_uavids_dataset_audit.ipynb").exists():
    PROJECT_ROOT = HERE
elif HERE.name == "notebooks" and (HERE / "04_uavids_dataset_audit.ipynb").exists():
    PROJECT_ROOT = HERE.parent
else:
    raise RuntimeError(f"Run from the uavids_federated root or notebooks directory, not {HERE}")

EXPERIMENTS_ROOT = PROJECT_ROOT.parent
DATA_PATH = Path(os.environ.get(
    "UAVIDS_DATASET",
    EXPERIMENTS_ROOT / "UAVIDS-2025" / "UAVIDS-2025.csv",
)).resolve()
RESULTS_DIR = PROJECT_ROOT / "results_audit"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OFFICIAL_MD5 = "ec84ed5390d5de42b07e8a011709ff82"
assert DATA_PATH.is_file(), f"Dataset not found: {DATA_PATH}"
actual_md5 = hashlib.md5(DATA_PATH.read_bytes()).hexdigest()
assert actual_md5 == OFFICIAL_MD5, (actual_md5, OFFICIAL_MD5)

print(f"Dataset: {DATA_PATH}")
print(f"Output directory: {RESULTS_DIR}")
print(f"MD5 verified against Zenodo: {actual_md5}")
"""
    ),
    markdown(
        r"""
## 3. Load the raw CSV and confirm its structure

No values are transformed. The shape, column order, and inferred pandas data
types below describe the released file exactly as loaded.
"""
    ),
    code(
        r"""
raw = pd.read_csv(DATA_PATH, low_memory=False)

EXPECTED_COLUMNS = [
    "FlowID", "FlowDuration/s", "SrcAddr", "SrcPort", "DstAddr", "DstPort",
    "Protocol", "TxPackets", "RxPackets", "LostPackets", "TxBytes", "RxBytes",
    "TxPacketRate/s", "RxPacketRate/s", "TxByteRate/s", "RxByteRate/s",
    "MeanDelay/s", "MeanJitter/s", "Throughput/Kbps", "MeanPacketSize",
    "PacketDropRate", "AverageHopCount", "label",
]
assert raw.columns.tolist() == EXPECTED_COLUMNS
assert raw.shape == (122_171, 23)

shape_table = pd.DataFrame({
    "measure": ["rows", "columns", "documented flow features", "label columns"],
    "value": [len(raw), raw.shape[1], 22, 1],
})
dtype_table = raw.dtypes.rename("pandas_dtype").to_frame()

display(shape_table)
display(dtype_table)
"""
    ),
    markdown(
        r"""
## 4. Column dictionary

The 22 feature meanings are paraphrased from Table III of the associated paper.
The `label` meaning follows the paper's data-collection and labeling section.
The CSV column order differs from the table order, but the names match.
"""
    ),
    code(
        r"""
column_dictionary = pd.DataFrame([
    ("FlowID", "Connection", "Unique identifier assigned to each network flow.", "Documented fact", "IEEE paper Table III"),
    ("FlowDuration/s", "Connection", "Total duration of the flow in seconds.", "Documented fact", "IEEE paper Table III"),
    ("SrcAddr", "Connection", "Source IP address of the communication.", "Documented fact", "IEEE paper Table III"),
    ("SrcPort", "Connection", "Source port used by the connection.", "Documented fact", "IEEE paper Table III"),
    ("DstAddr", "Connection", "Destination IP address of the communication.", "Documented fact", "IEEE paper Table III"),
    ("DstPort", "Connection", "Destination port used by the connection.", "Documented fact", "IEEE paper Table III"),
    ("Protocol", "Connection", "Communication protocol used by the flow.", "Documented fact", "IEEE paper Table III"),
    ("TxPackets", "Traffic volume", "Number of packets transmitted in the flow.", "Documented fact", "IEEE paper Table III"),
    ("RxPackets", "Traffic volume", "Number of packets received in the flow.", "Documented fact", "IEEE paper Table III"),
    ("LostPackets", "Traffic volume", "Number of packets lost during transmission.", "Documented fact", "IEEE paper Table III"),
    ("TxBytes", "Traffic volume", "Number of bytes transmitted in the flow.", "Documented fact", "IEEE paper Table III"),
    ("RxBytes", "Traffic volume", "Number of bytes received in the flow.", "Documented fact", "IEEE paper Table III"),
    ("TxPacketRate/s", "Traffic volume", "Packet transmission rate per second.", "Documented fact", "IEEE paper Table III"),
    ("RxPacketRate/s", "Traffic volume", "Packet reception rate per second.", "Documented fact", "IEEE paper Table III"),
    ("TxByteRate/s", "Traffic volume", "Byte transmission rate per second.", "Documented fact", "IEEE paper Table III"),
    ("RxByteRate/s", "Traffic volume", "Byte reception rate per second.", "Documented fact", "IEEE paper Table III"),
    ("MeanDelay/s", "Performance", "Average end-to-end delay in seconds.", "Documented fact", "IEEE paper Table III"),
    ("MeanJitter/s", "Performance", "Average variation in packet delay.", "Documented fact", "IEEE paper Table III"),
    ("Throughput/Kbps", "Performance", "Effective transfer rate in kilobits per second.", "Documented fact", "IEEE paper Table III"),
    ("MeanPacketSize", "Traffic volume", "Average packet size in the flow.", "Documented fact", "IEEE paper Table III"),
    ("PacketDropRate", "Performance", "Ratio of dropped packets to transmitted packets.", "Documented fact", "IEEE paper Table III"),
    ("AverageHopCount", "Performance", "Average number of network hops traversed by packets.", "Documented fact", "IEEE paper Table III"),
    ("label", "Ground truth", "Scenario annotation: normal traffic or one of four documented attacks.", "Documented fact", "IEEE paper labeling section and CSV values"),
], columns=["column", "category", "meaning", "evidence_status", "source"])

assert column_dictionary["column"].tolist() == EXPECTED_COLUMNS
display(column_dictionary)
"""
    ),
    markdown(
        r"""
## 5. Ground-truth label and overall class distribution

`label` is the ground-truth scenario label. Counts are calculated directly from
the CSV and checked against the totals reported by the paper.
"""
    ),
    code(
        r"""
LABEL_COLUMN = "label"
CLASS_ORDER = [
    "Normal Traffic", "Blackhole Attack", "Flooding Attack",
    "Sybil Attack", "Wormhole Attack",
]
DOCUMENTED_COUNTS = {
    "Normal Traffic": 26_172,
    "Blackhole Attack": 26_110,
    "Flooding Attack": 19_726,
    "Sybil Attack": 24_077,
    "Wormhole Attack": 26_086,
}

class_distribution = raw[LABEL_COLUMN].value_counts().reindex(CLASS_ORDER).rename("count").to_frame()
class_distribution["percentage"] = 100 * class_distribution["count"] / len(raw)
assert class_distribution["count"].to_dict() == DOCUMENTED_COUNTS
assert set(raw[LABEL_COLUMN].dropna().unique()) == set(CLASS_ORDER)

display(class_distribution.style.format({"percentage": "{:.3f}%"}))
"""
    ),
    markdown(
        r"""
## 6. Missing, infinite, unique, and basic validity checks

The audit checks nulls, blank strings, infinities, negative numeric values,
IPv4 syntax, port ranges, label membership, and documented ratio bounds. Values
are reported but not repaired or removed.
"""
    ),
    code(
        r"""
numeric_columns = raw.select_dtypes(include=[np.number]).columns.tolist()
text_columns = raw.select_dtypes(exclude=[np.number]).columns.tolist()

quality_rows = []
for column in raw.columns:
    series = raw[column]
    counts = series.value_counts(dropna=False)
    is_numeric = column in numeric_columns
    quality_rows.append({
        "column": column,
        "dtype": str(series.dtype),
        "missing_count": int(series.isna().sum()),
        "infinite_count": int(np.isinf(series.to_numpy()).sum()) if is_numeric else 0,
        "blank_string_count": int(series.astype("string").str.strip().eq("").sum()) if not is_numeric else 0,
        "unique_count_including_missing": int(series.nunique(dropna=False)),
        "top_value": str(counts.index[0]),
        "top_count": int(counts.iloc[0]),
        "top_percentage": float(100 * counts.iloc[0] / len(raw)),
        "negative_count": int(series.lt(0).sum()) if is_numeric else 0,
    })
column_quality = pd.DataFrame(quality_rows)

def valid_ipv4(value):
    try:
        return ipaddress.ip_address(value).version == 4
    except ValueError:
        return False

all_finite = np.isfinite(raw[numeric_columns].to_numpy()).all(axis=1)
invalid_summary = pd.DataFrame([
    ("Missing values in any column", int(raw.isna().sum().sum()), "invalid if nonzero"),
    ("Blank strings in text columns", int(sum(raw[c].astype("string").str.strip().eq("").sum() for c in text_columns)), "invalid if nonzero"),
    ("Rows with non-finite numeric values", int((~all_finite).sum()), "invalid if nonzero"),
    ("Negative numeric cells", int((raw[numeric_columns] < 0).sum().sum()), "invalid for the documented measures"),
    ("Non-positive flow durations", int(raw["FlowDuration/s"].le(0).sum()), "invalid if nonzero"),
    ("Invalid source IPv4 values", int((~raw["SrcAddr"].map(valid_ipv4)).sum()), "invalid if nonzero"),
    ("Invalid destination IPv4 values", int((~raw["DstAddr"].map(valid_ipv4)).sum()), "invalid if nonzero"),
    ("Rows with ports outside 0..65535", int((~raw["SrcPort"].between(0, 65535) | ~raw["DstPort"].between(0, 65535)).sum()), "invalid if nonzero"),
    ("Rows with undocumented labels", int((~raw[LABEL_COLUMN].isin(CLASS_ORDER)).sum()), "invalid if nonzero"),
    ("Rows with PacketDropRate outside 0..1", int((~raw["PacketDropRate"].between(0, 1)).sum()), "suspicious: documented as a ratio"),
    ("Rows where LostPackets exceeds TxPackets", int(raw["LostPackets"].gt(raw["TxPackets"]).sum()), "suspicious under the documented meanings"),
], columns=["check", "count", "interpretation"])

display(column_quality)
display(invalid_summary)
"""
    ),
    code(
        r"""
# Evidence for the 81 suspicious ratio records; these rows remain untouched.
suspicious_drop_rows = raw.loc[
    ~raw["PacketDropRate"].between(0, 1),
    ["FlowID", "SrcAddr", "DstAddr", "TxPackets", "RxPackets", "LostPackets", "PacketDropRate", "label"],
]
suspicious_drop_by_class = suspicious_drop_rows["label"].value_counts().reindex(CLASS_ORDER, fill_value=0).rename("count").to_frame()

drop_formula = np.divide(
    raw["LostPackets"].to_numpy(dtype=float),
    raw["TxPackets"].to_numpy(dtype=float),
    out=np.zeros(len(raw), dtype=float),
    where=raw["TxPackets"].to_numpy() != 0,
)
drop_formula_matches = int(np.isclose(raw["PacketDropRate"], drop_formula, rtol=1e-4, atol=1e-7).sum())

print(f"PacketDropRate agrees with LostPackets / TxPackets (within tolerance) in {drop_formula_matches:,} / {len(raw):,} rows.")
display(suspicious_drop_by_class)
display(suspicious_drop_rows.head(20))
"""
    ),
    markdown(
        r"""
### Documentation discrepancy: normalization

The paper's feature-extraction section says numerical features were normalized.
The official hash-matched CSV is plainly not normalized to a shared [0, 1] or
standard-like scale: packet/byte measures retain large raw-scale values. The
available sources do not clarify whether normalization was applied only inside
the paper's modeling pipeline or whether a normalized release was intended.
Phase 2 must therefore treat this CSV as unscaled input and fit any preprocessing
on training data only.
"""
    ),
    code(
        r"""
numeric_ranges = raw[numeric_columns].agg(["min", "max"]).T
display(numeric_ranges)
"""
    ),
    markdown(
        r"""
## 7. Constant and near-constant columns

A column is called near-constant here when one value occupies at least 99% of
rows (excluding columns that are fully constant). The threshold is explicit so
it can be changed later; it is an audit definition, not a cleaning decision.
"""
    ),
    code(
        r"""
constant_near = column_quality[[
    "column", "unique_count_including_missing", "top_value", "top_count", "top_percentage"
]].copy()
constant_near["is_constant"] = constant_near["unique_count_including_missing"].eq(1)
constant_near["is_near_constant_at_99pct"] = (
    ~constant_near["is_constant"] & constant_near["top_percentage"].ge(99.0)
)
constant_near = constant_near.sort_values(["top_percentage", "column"], ascending=[False, True])

display(constant_near)
print("Constant columns:", constant_near.loc[constant_near["is_constant"], "column"].tolist())
print("Near-constant (>=99%, excluding constant):", constant_near.loc[constant_near["is_near_constant_at_99pct"], "column"].tolist())
"""
    ),
    markdown(
        r"""
## 8. Source-address population and source-by-class tables

`SrcAddr` is the only explicit source identifier in the CSV. The tables below
show every source, its sample count, class counts, and within-source percentages.
They answer whether candidate clients contain normal and attack observations;
they do not establish that each address is a physical UAV.
"""
    ),
    code(
        r"""
source_by_class = pd.crosstab(raw["SrcAddr"], raw[LABEL_COLUMN]).reindex(columns=CLASS_ORDER, fill_value=0)
source_by_class["Total"] = source_by_class.sum(axis=1)
source_by_class["Normal count"] = source_by_class["Normal Traffic"]
source_by_class["Attack count"] = source_by_class["Total"] - source_by_class["Normal Traffic"]
source_by_class["Classes present"] = (source_by_class[CLASS_ORDER] > 0).sum(axis=1)
source_by_class["Attack types present"] = (source_by_class[CLASS_ORDER[1:]] > 0).sum(axis=1)
source_by_class["Profile"] = np.select(
    [
        source_by_class["Normal count"].eq(0),
        source_by_class["Attack count"].eq(0),
    ],
    ["attack-only", "normal-only"],
    default="normal+attack",
)

source_percentages = source_by_class[CLASS_ORDER].div(source_by_class["Total"], axis=0).mul(100)
source_percentages["Normal %"] = 100 * source_by_class["Normal count"] / source_by_class["Total"]
source_percentages["Attack %"] = 100 * source_by_class["Attack count"] / source_by_class["Total"]

source_frequency = source_by_class[["Total", "Profile"]].sort_values("Total", ascending=False)
source_profile_summary = pd.DataFrame({
    "measure": [
        "distinct sources", "normal+attack sources", "attack-only sources", "normal-only sources",
        "sources with all five classes", "sources with all four attacks", "sources below 10 rows",
        "sources below 100 rows", "sources at least 1,000 rows",
    ],
    "value": [
        len(source_by_class),
        int(source_by_class["Profile"].eq("normal+attack").sum()),
        int(source_by_class["Profile"].eq("attack-only").sum()),
        int(source_by_class["Profile"].eq("normal-only").sum()),
        int(source_by_class["Classes present"].eq(5).sum()),
        int(source_by_class["Attack types present"].eq(4).sum()),
        int(source_by_class["Total"].lt(10).sum()),
        int(source_by_class["Total"].lt(100).sum()),
        int(source_by_class["Total"].ge(1000).sum()),
    ],
})

display(source_profile_summary)
display(source_frequency)
display(source_by_class.sort_values("Total", ascending=False))
display(source_percentages.loc[source_by_class.sort_values("Total", ascending=False).index].style.format("{:.2f}%"))
"""
    ),
    markdown(
        r"""
### Source/entity interpretation

- **Documented fact:** `SrcAddr` is the source IP of the communication; captures
  were deployed at all simulated UAV nodes.
- **Documented fact:** Sybil attacks forge multiple identities or addresses;
  blackhole and Sybil attackers are described as adversarial/compromised UAVs,
  while wormholes involve two colluding UAVs.
- **Evidence-based inference:** 95 source values have both normal and attack
  observations, while 81 are attack-only. Some attack-only values may be forged
  Sybil identities or attack-specific entities rather than stable UAVs.
- **Unresolved:** the paper and released record do not map IP values to UAVs,
  attackers, a ground-control station, routers, capture interfaces, or simulation
  instances. Source addresses are therefore defensible only as **logical source
  clients** unless further simulator metadata is obtained.
"""
    ),
    markdown(
        r"""
## 9. Duplicate and conflicting-label audit

Literal duplicate checks can be misleading because `FlowID` is unique. Four
scopes are therefore reported explicitly:

1. all 23 columns, including `label`;
2. all 22 published features, excluding only `label` (therefore including `FlowID`);
3. the 21 published features excluding `label` and `FlowID`;
4. 16 provisional numeric behavioral inputs, excluding label, row ID, addresses,
   ports, and the constant protocol.

For each label-excluded scope, same-label repetitions and conflicting-label
repetitions are counted separately. No record is deleted.
"""
    ),
    code(
        r"""
PUBLISHED_FEATURES = [c for c in raw.columns if c != LABEL_COLUMN]
CANDIDATE_NUMERIC_FEATURES = [
    c for c in raw.columns
    if c not in {LABEL_COLUMN, "FlowID", "SrcAddr", "DstAddr", "SrcPort", "DstPort", "Protocol"}
]

duplicate_scopes = {
    "Full row (all 23 columns, label included)": list(raw.columns),
    "Published feature vector (22 columns; label excluded)": PUBLISHED_FEATURES,
    "Published feature vector without FlowID (21 columns)": [c for c in PUBLISHED_FEATURES if c != "FlowID"],
    "Provisional numeric behavior vector (16 columns)": CANDIDATE_NUMERIC_FEATURES,
}

duplicate_rows = []
for scope, columns in duplicate_scopes.items():
    sizes = raw.groupby(columns, dropna=False, sort=False).size()
    repeated = sizes[sizes > 1]
    item = {
        "scope": scope,
        "columns_included": ", ".join(columns),
        "columns_excluded": ", ".join(c for c in raw.columns if c not in columns) or "None",
        "repeated_groups": int(len(repeated)),
        "rows_in_repeated_groups": int(repeated.sum()),
        "redundant_rows_beyond_first": int((repeated - 1).sum()),
        "max_group_size": int(sizes.max()),
        "same_label_repeated_groups": np.nan,
        "same_label_rows": np.nan,
        "same_label_redundant_rows": np.nan,
        "conflicting_label_groups": np.nan,
        "conflicting_label_rows": np.nan,
    }
    if LABEL_COLUMN not in columns:
        label_groups = raw.groupby(columns, dropna=False, sort=False)[LABEL_COLUMN].agg(
            size="size", label_count="nunique"
        )
        label_groups = label_groups[label_groups["size"] > 1]
        same = label_groups[label_groups["label_count"] == 1]
        conflicts = label_groups[label_groups["label_count"] > 1]
        item.update({
            "same_label_repeated_groups": int(len(same)),
            "same_label_rows": int(same["size"].sum()),
            "same_label_redundant_rows": int((same["size"] - 1).sum()),
            "conflicting_label_groups": int(len(conflicts)),
            "conflicting_label_rows": int(conflicts["size"].sum()),
        })
    duplicate_rows.append(item)

duplicate_summary = pd.DataFrame(duplicate_rows)

cross_source_groups = raw.groupby(CANDIDATE_NUMERIC_FEATURES, dropna=False).agg(
    rows=(LABEL_COLUMN, "size"),
    sources=("SrcAddr", "nunique"),
    labels=(LABEL_COLUMN, "nunique"),
)
cross_source_groups = cross_source_groups[
    cross_source_groups["rows"].gt(1) & cross_source_groups["sources"].gt(1)
]
cross_source_duplicate_summary = pd.DataFrame({
    "measure": ["behavior groups spanning multiple sources", "rows in those groups", "maximum sources in one group", "conflicting-label groups among them"],
    "value": [
        len(cross_source_groups),
        int(cross_source_groups["rows"].sum()),
        int(cross_source_groups["sources"].max()) if len(cross_source_groups) else 0,
        int(cross_source_groups["labels"].gt(1).sum()),
    ],
})

display(duplicate_summary)
display(cross_source_duplicate_summary)
"""
    ),
    markdown(
        r"""
## 10. Identifier, metadata, and leakage-risk assessment

`FlowID`, addresses, ports, and protocol are assessed separately from behavioral
measurements. A value is called label-pure when it appears with only one label;
this is descriptive evidence of shortcut risk, not a causal test.
"""
    ),
    code(
        r"""
def label_purity(column):
    table = pd.crosstab(raw[column], raw[LABEL_COLUMN])
    pure = table.gt(0).sum(axis=1).eq(1)
    return int(pure.sum()), int(table.loc[pure].to_numpy().sum())

identifier_rows = []
assessments = {
    "FlowID": "Unique sequential row/flow identifier; can encode file order and simulation blocks.",
    "SrcAddr": "Source identity and proposed grouping field; 81 values are attack-only. Sybil addresses may be forged.",
    "DstAddr": "Destination identity; can support endpoint or topology memorization and crosses source groups.",
    "SrcPort": "Only two values (9 and 654); absent combinations align strongly with attack scenarios.",
    "DstPort": "Only two values, identical to SrcPort in every row; redundant and scenario-associated.",
    "Protocol": "Constant UDP in all rows; carries no predictive variation.",
}
for column, assessment in assessments.items():
    pure_values, pure_rows = label_purity(column)
    counts = raw[column].value_counts(dropna=False)
    identifier_rows.append({
        "column": column,
        "cardinality": int(raw[column].nunique(dropna=False)),
        "dominant_percentage": float(100 * counts.iloc[0] / len(raw)),
        "label_pure_values": pure_values,
        "rows_in_label_pure_values": pure_rows,
        "rows_in_label_pure_values_pct": float(100 * pure_rows / len(raw)),
        "assessment": assessment,
    })
identifier_risk = pd.DataFrame(identifier_rows)

port_by_class = pd.crosstab(raw["SrcPort"], raw[LABEL_COLUMN]).reindex(columns=CLASS_ORDER, fill_value=0)
port_relationships = pd.DataFrame({
    "check": ["SrcPort equals DstPort", "Protocol is UDP", "Port 9 has MeanPacketSize 76"],
    "rows": [
        int(raw["SrcPort"].eq(raw["DstPort"]).sum()),
        int(raw["Protocol"].eq("UDP").sum()),
        int((raw["SrcPort"].eq(9) & raw["MeanPacketSize"].eq(76)).sum()),
    ],
    "out_of": len(raw),
})

display(identifier_risk.style.format({
    "dominant_percentage": "{:.3f}%",
    "rows_in_label_pure_values_pct": "{:.3f}%",
}))
display(port_by_class)
display(port_relationships)
"""
    ),
    markdown(
        r"""
## 11. Proposed feature roles for Phase 2 review

These are conservative proposals, not a final training schema. Ports remain
unresolved because they are documented connection features but are unusually
low-cardinality, mutually redundant, and scenario-associated in this simulation.
Behavioral fields remain candidate inputs; their derived relationships and
outliers should be handled only after a Phase 2 decision.
"""
    ),
    code(
        r"""
role_map = {
    "label": ("Label", "Ground-truth scenario class."),
    "SrcAddr": ("Client/grouping field only", "Strongest available logical-source key; remove before modeling."),
    "FlowID": ("Exclude as identifier/leakage risk", "Unique sequential flow ID and possible file/run-order shortcut."),
    "DstAddr": ("Exclude as identifier/leakage risk", "Destination identity/topology shortcut; not a behavioral measure."),
    "Protocol": ("Exclude as identifier/leakage risk", "Constant UDP; no usable variation."),
    "SrcPort": ("Unresolved pending documentation", "Two scenario-associated values; duplicates DstPort on every row."),
    "DstPort": ("Unresolved pending documentation", "Two scenario-associated values; duplicates SrcPort on every row."),
}

feature_roles = column_dictionary[["column", "meaning", "evidence_status"]].copy()
feature_roles[["proposed_role", "reason"]] = feature_roles["column"].apply(
    lambda c: pd.Series(role_map.get(c, (
        "Candidate model input",
        "Documented traffic/performance measurement; retain for later leakage and sensitivity review.",
    )))
)
assert set(feature_roles["proposed_role"]) <= {
    "Label", "Client/grouping field only", "Candidate model input",
    "Exclude as identifier/leakage risk", "Unresolved pending documentation",
}

display(feature_roles)
"""
    ),
    markdown(
        r"""
## 12. Provisional five source-based logical clients

Selection criteria are intentionally binary-training oriented:

- every candidate has both normal and attack rows;
- the federation collectively contains all four attack families;
- candidates have enough observations for a local experiment;
- class prevalence and attack composition vary substantially, creating meaningful
  non-IID behavior;
- attack-only/Sybil-only addresses are not selected as the initial five.

These are **not verified physical UAV identities** and no mission roles are
assigned. The table is a proposal only; no partition files are created.
"""
    ),
    code(
        r"""
CLIENT_SOURCES = [
    ("uav-client-1", "192.168.0.26"),
    ("uav-client-2", "192.168.0.25"),
    ("uav-client-3", "192.168.0.100"),
    ("uav-client-4", "192.168.0.5"),
    ("uav-client-5", "192.168.0.32"),
]

selection_reasons = {
    "192.168.0.26": "Near-balanced binary mix and all four attacks; broad reference client.",
    "192.168.0.25": "Attack-heavy and all four attacks; complements client 1 with different prevalence.",
    "192.168.0.100": "Smaller attack-heavy source with substantial Sybil representation; preserves a rare attack locally.",
    "192.168.0.5": "Large, strongly attack-heavy and flooding-dominated source; supplies deliberate non-IID stress.",
    "192.168.0.32": "Near-balanced source dominated by blackhole/wormhole and no Sybil; contrasts client 4.",
}
concerns = {
    "192.168.0.26": "Physical-UAV identity unverified; Sybil rows may involve forged identities.",
    "192.168.0.25": "75% attack; physical-UAV identity unverified.",
    "192.168.0.100": "Only 596 rows and 145 normal; physical-UAV identity unverified.",
    "192.168.0.5": "94% attack, flooding dominates, and no Sybil rows.",
    "192.168.0.32": "No Sybil rows; physical-UAV identity unverified.",
}

candidate_rows = []
for client_id, source in CLIENT_SOURCES:
    counts = source_by_class.loc[source]
    attacks_present = [attack for attack in CLASS_ORDER[1:] if counts[attack] > 0]
    candidate_rows.append({
        "client_id": client_id,
        "original_source_value": source,
        "entity_evidence": "Documented source IP; normal+attack observations support a logical client, but no physical-UAV map is published.",
        "samples": int(counts["Total"]),
        "normal_count": int(counts["Normal count"]),
        "attack_count": int(counts["Attack count"]),
        "blackhole_count": int(counts["Blackhole Attack"]),
        "flooding_count": int(counts["Flooding Attack"]),
        "sybil_count": int(counts["Sybil Attack"]),
        "wormhole_count": int(counts["Wormhole Attack"]),
        "attacks_present": ", ".join(attacks_present),
        "normal_percentage": float(100 * counts["Normal count"] / counts["Total"]),
        "attack_percentage": float(100 * counts["Attack count"] / counts["Total"]),
        "imbalance_or_reliability_concern": concerns[source],
        "selection_reason": selection_reasons[source],
    })
client_candidates = pd.DataFrame(candidate_rows)

selected_sources = [source for _, source in CLIENT_SOURCES]
selected_raw = raw[raw["SrcAddr"].isin(selected_sources)]
remaining_raw = raw[~raw["SrcAddr"].isin(selected_sources)]
remaining_counts = remaining_raw[LABEL_COLUMN].value_counts().reindex(CLASS_ORDER, fill_value=0)
remaining_source_table = pd.crosstab(remaining_raw["SrcAddr"], remaining_raw[LABEL_COLUMN]).reindex(columns=CLASS_ORDER, fill_value=0)
remaining_both = int((
    remaining_source_table["Normal Traffic"].gt(0)
    & remaining_source_table[CLASS_ORDER[1:]].sum(axis=1).gt(0)
).sum())

display(client_candidates.style.format({
    "normal_percentage": "{:.2f}%",
    "attack_percentage": "{:.2f}%",
}))
"""
    ),
    markdown(
        r"""
## 13. Remaining groups and evaluation feasibility

This section assesses capacity only; it does not design a split. Source-disjoint
validation/test partitions are mechanically possible, but source values are not
proven independent physical UAVs or independent simulation runs. The CSV also
lacks an explicit scenario/run field, so a defensible scenario-disjoint split
cannot be constructed from this release alone.
"""
    ),
    code(
        r"""
remaining_assessment = pd.DataFrame({
    "measure": [
        "rows in proposed five clients",
        "rows remaining untouched",
        "source values remaining untouched",
        "remaining sources with both normal and attack",
        "remaining attack-only sources",
        *[f"remaining {label} rows" for label in CLASS_ORDER],
    ],
    "value": [
        len(selected_raw),
        len(remaining_raw),
        remaining_raw["SrcAddr"].nunique(),
        remaining_both,
        int(remaining_raw["SrcAddr"].nunique() - remaining_both),
        *[int(remaining_counts[label]) for label in CLASS_ORDER],
    ],
})
display(remaining_assessment)

display(Markdown(
    "**Interpretation.** All five classes remain in large numbers after the five candidate "
    "sources are set aside. However, 7 exact provisional behavioral signatures span multiple "
    "source values (21 rows total), so a later source-disjoint design must also audit behavior-"
    "signature overlap. Destination addresses and undocumented simulation membership can still "
    "link nominally separate source groups."
))
"""
    ),
    markdown(
        r"""
## 14. Save Phase 1 audit tables and summary

Every reported count in the summary below is interpolated from variables computed
in this notebook. The output records the decision gate rather than silently
choosing a Phase 2 split.
"""
    ),
    code(
        r"""
column_dictionary.to_csv(RESULTS_DIR / "column_dictionary.csv", index=False)
column_quality.to_csv(RESULTS_DIR / "column_quality.csv", index=False)
constant_near.to_csv(RESULTS_DIR / "constant_near_constant.csv", index=False)
class_distribution.to_csv(RESULTS_DIR / "class_distribution.csv")
source_frequency.to_csv(RESULTS_DIR / "source_frequency.csv")
source_by_class.to_csv(RESULTS_DIR / "source_by_class_counts.csv")
source_percentages.to_csv(RESULTS_DIR / "source_by_class_percentages.csv")
duplicate_summary.to_csv(RESULTS_DIR / "duplicate_summary.csv", index=False)
identifier_risk.to_csv(RESULTS_DIR / "identifier_leakage_risk.csv", index=False)
feature_roles.to_csv(RESULTS_DIR / "proposed_feature_roles.csv", index=False)
client_candidates.to_csv(RESULTS_DIR / "proposed_client_candidates.csv", index=False)
suspicious_drop_rows.to_csv(RESULTS_DIR / "suspicious_packet_drop_rows.csv", index=False)

literal_features = duplicate_summary.iloc[1]
without_flowid = duplicate_summary.iloc[2]
behavior_scope = duplicate_summary.iloc[3]

audit_summary = {
    "phase": 1,
    "dataset_path": str(DATA_PATH),
    "official_md5": actual_md5,
    "shape": {"rows": int(len(raw)), "columns": int(raw.shape[1]), "features": 22, "label_columns": 1},
    "label_column": LABEL_COLUMN,
    "class_counts": {k: int(v) for k, v in class_distribution["count"].items()},
    "quality": {
        "missing_cells": int(raw.isna().sum().sum()),
        "infinite_numeric_cells": int(np.isinf(raw[numeric_columns].to_numpy()).sum()),
        "negative_numeric_cells": int((raw[numeric_columns] < 0).sum().sum()),
        "constant_columns": constant_near.loc[constant_near["is_constant"], "column"].tolist(),
        "near_constant_columns_at_99pct": constant_near.loc[constant_near["is_near_constant_at_99pct"], "column"].tolist(),
        "packet_drop_rate_outside_0_1_rows": int((~raw["PacketDropRate"].between(0, 1)).sum()),
        "normalization_documentation_note": (
            "The paper says numerical features were normalized, but the official CSV retains "
            "raw-scale values; whether normalization occurred only during modeling is unresolved."
        ),
    },
    "sources": {
        "distinct": int(raw["SrcAddr"].nunique()),
        "normal_and_attack": int(source_by_class["Profile"].eq("normal+attack").sum()),
        "attack_only": int(source_by_class["Profile"].eq("attack-only").sum()),
        "normal_only": int(source_by_class["Profile"].eq("normal-only").sum()),
        "physical_uav_mapping_verified": False,
    },
    "duplicates": {
        "full_row_exact_duplicates": int(duplicate_summary.iloc[0]["redundant_rows_beyond_first"]),
        "published_feature_redundant_rows_including_flowid": int(literal_features["redundant_rows_beyond_first"]),
        "without_flowid_repeated_groups": int(without_flowid["repeated_groups"]),
        "without_flowid_rows_in_repeated_groups": int(without_flowid["rows_in_repeated_groups"]),
        "without_flowid_redundant_rows": int(without_flowid["redundant_rows_beyond_first"]),
        "without_flowid_conflicting_label_groups": int(without_flowid["conflicting_label_groups"]),
        "behavior_repeated_groups": int(behavior_scope["repeated_groups"]),
        "behavior_rows_in_repeated_groups": int(behavior_scope["rows_in_repeated_groups"]),
        "behavior_redundant_rows": int(behavior_scope["redundant_rows_beyond_first"]),
        "behavior_conflicting_label_groups": int(behavior_scope["conflicting_label_groups"]),
    },
    "proposed_clients": client_candidates.to_dict(orient="records"),
    "remaining": {
        "rows": int(len(remaining_raw)),
        "sources": int(remaining_raw["SrcAddr"].nunique()),
        "sources_with_normal_and_attack": remaining_both,
        "class_counts": {k: int(v) for k, v in remaining_counts.items()},
    },
    "phase_2_decision": (
        "Decide whether SrcAddr values may be used explicitly as source-based logical clients "
        "without claiming physical-UAV identity. If stronger entity independence is required, "
        "obtain simulator topology/IP and scenario-run metadata before partitioning."
    ),
}
(RESULTS_DIR / "audit_summary.json").write_text(
    json.dumps(audit_summary, indent=2), encoding="utf-8"
)

candidate_lines = "\n".join(
    f"- `{row.client_id}` = `{row.original_source_value}`: {row.samples:,} rows, "
    f"{row.normal_count:,} normal / {row.attack_count:,} attack "
    f"({row.normal_percentage:.2f}% / {row.attack_percentage:.2f}%); "
    f"attacks: {row.attacks_present}."
    for row in client_candidates.itertuples()
)

summary_md = f'''# Phase 1 — UAVIDS-2025 audit summary

Generated by `04_uavids_dataset_audit.ipynb`; no raw records were changed or removed.

## Most important findings

- The verified Zenodo CSV contains **{len(raw):,} flows, 22 published features, and one label column**.
- `label` has five classes: {', '.join(f'{name} ({count:,})' for name, count in class_distribution['count'].items())}.
- `SrcAddr` is the source IP and the strongest available client key: **{raw['SrcAddr'].nunique()} values**; **{int(source_by_class['Profile'].eq('normal+attack').sum())}** contain both binary classes and **{int(source_by_class['Profile'].eq('attack-only').sum())}** are attack-only.
- The sources are not documented as separate physical UAV identities. Sybil traffic may contain forged source addresses. They must be called **source-based logical clients** unless mapping metadata is obtained.
- Missing cells, infinite numeric cells, negative numeric cells: **0 / 0 / 0**. `Protocol` is constant (`UDP`); no non-constant column reaches the 99% near-constant threshold.
- **{int((~raw['PacketDropRate'].between(0, 1)).sum())}** rows have `PacketDropRate > 1` and `LostPackets > TxPackets`; these are suspicious under the published definitions and remain untouched.
- The paper says numerical features were normalized, but the official hash-matched CSV retains raw-scale values (for example, `TxBytes` reaches **{int(numeric_ranges.loc['TxBytes', 'max']):,}**). Whether normalization was only part of the paper's modeling pipeline is unresolved.
- Exact full-row duplicates: **0**. Literal 22-feature repeats excluding only `label`: **0**, because `FlowID` is unique.
- Excluding `FlowID`: **{int(without_flowid['repeated_groups']):,}** repeated groups, **{int(without_flowid['rows_in_repeated_groups']):,}** involved rows, **{int(without_flowid['redundant_rows_beyond_first']):,}** redundant rows, and **0 conflicting-label groups**.
- Using the provisional 16 behavioral inputs: **{int(behavior_scope['repeated_groups']):,}** repeated groups, **{int(behavior_scope['rows_in_repeated_groups']):,}** involved rows, **{int(behavior_scope['redundant_rows_beyond_first']):,}** redundant rows, and **0 conflicting-label groups**.

## Proposed logical clients (not verified physical UAVs)

{candidate_lines}

The five candidates total **{len(selected_raw):,}** rows. They deliberately range from near-balanced to strongly attack-heavy and collectively include all four attacks.

## Suspected leakage or shortcut fields

- Exclude `FlowID` (unique sequential identifier), `SrcAddr` (grouping identity), and `DstAddr` (endpoint/topology identity) from model inputs.
- Exclude constant `Protocol`.
- Resolve `SrcPort` and `DstPort` before training: they are identical in every row, have only values 9/654, and are strongly scenario-associated. `MeanPacketSize` mirrors part of this setting and merits sensitivity analysis.
- The CSV has no explicit timestamp, scenario, simulation-run, capture-point, or node-role field; absence prevents direct scenario-disjoint splitting rather than proving scenario leakage is absent.

## Remaining capacity and unresolved risks

After setting aside the five candidates, **{len(remaining_raw):,} rows across {remaining_raw['SrcAddr'].nunique()} source values** remain, including all five classes; {remaining_both} remaining sources contain both normal and attack rows. This is enough for source-disjoint experiments mechanically, but physical and scenario independence are unproven. Exact behavioral signatures also span multiple sources in **{len(cross_source_groups)} groups ({int(cross_source_groups['rows'].sum())} rows)**.

## Decision required before Phase 2

Decide whether the project will explicitly accept `SrcAddr` values as **source-based logical clients**, with no claim that each is a physical UAV. If physical-UAV or scenario independence is required, obtain simulator topology/IP-role and scenario/run metadata before creating partitions. Phase 2 must also choose how to keep repeated behavioral signatures disjoint and whether to exclude or sensitivity-test the two port fields.
'''

(RESULTS_DIR / "PHASE1_AUDIT_SUMMARY.md").write_text(summary_md, encoding="utf-8")
display(Markdown(summary_md))
print(f"Saved {len(list(RESULTS_DIR.iterdir()))} audit files to {RESULTS_DIR}")
"""
    ),
    markdown(
        r"""
## Phase boundary

Phase 1 ends here. No record has been cleaned or deleted, no final partitions
have been created, and no centralized or federated model has been trained.
"""
    ),
]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
)
NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, NOTEBOOK_PATH)
print(f"Wrote {NOTEBOOK_PATH}")
