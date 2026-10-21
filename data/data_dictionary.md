# SaveOnCare Intelligence Lab — Data Dictionary

## Overview & Synthetic Data Notice
**CRITICAL NOTICE:** Every record in this dataset (`data/synthetic_billing_data.csv`) is synthetically generated for analytical demonstration, benchmark coverage measurement, data-quality auditing, and research purposes by VoltEdge.
- None of the records originate from actual patient hospital records.
- Hospital identifiers (`HOSP-BLR-01`, etc.) are fictionalized simulation entities.
- Reference prices are synthetic benchmarks modeled on median tender quotes, insurer schedules, and fair health price estimates. They **do not** represent statutory NPPA (National Pharmaceutical Pricing Authority) ceiling prices or government-mandated price caps.
- Rows labeled `injected_discrepancy = True` are controlled synthetic test cases created to evaluate detection precision and recall; they do not imply fraud, malpractice, or wrongdoing.

---

## Schema & Field Definitions

| Column Name | Data Type | Nullable | Description & Analytical Role | Allowable Values / Examples |
|---|---|---|---|---|
| `line_item_id` | string | No | Unique identifier for the individual bill line item. | e.g. `BILL-2024-0001-L01` |
| `bill_id` | string | No | Synthetic hospital bill identifier. Used for bill-level grouping. | e.g. `BILL-2024-0001` (500 distinct bills) |
| `hospital_id` | string | No | Synthetic hospital identifier. | `HOSP-BLR-01` to `HOSP-KOL-08` |
| `hospital_name` | string | No | Synthetic hospital name representing regional facility tiers. | Fictionalized institutional names |
| `hospital_tier` | string | No | Geographic hospital operational tier. | `Tier 1 Metro`, `Tier 2 Urban` |
| `bill_date` | string (ISO-8601) | No | Date of billing event (2024-01-01 to 2024-12-31). | YYYY-MM-DD |
| `category` | string | No | Clinical domain category of the billed item. | `medicines`, `consumables`, `diagnostics`, `procedures`, `services` |
| `product_or_service_id` | string | No | Synthetic master item or procedure code. | e.g. `MED-PCM-500`, `PRC-APP-LAP` |
| `item_description` | string | No | Clinical narrative description of the product or service billed. | e.g. `Paracetamol 500mg IV Infusion` |
| `quantity` | float / integer | Yes (DQ) | Number of units billed. Values <= 0 indicate data quality defects. | >= 1 (anomalies injected: 0, -1) |
| `billed_unit` | string | No | Unit of measure reported on the hospital bill. | `vial`, `piece`, `pair`, `pack`, `scan`, `day`, etc. |
| `billed_unit_price` | float (INR) | No | Unit price charged on the bill in Indian Rupees (₹). | > 0 |
| `billed_line_amount` | float (INR) | No | Total billed amount for the line item. Expected to equal `quantity * billed_unit_price`. | Currency in INR (₹) |
| `reference_unit` | string | Yes | Standardized reference benchmark unit of measure. | Matching unit or null |
| `reference_unit_price` | float (INR) | Yes | Synthetic benchmark unit price. NaN if no benchmark exists. | > 0 (or null; 4 test non-positive injected) |
| `reference_source` | string | Yes | Provenance tier of benchmark price. | `Synthetic State Tender Benchmark`, `Synthetic Market Median`, `Synthetic Insurer Agreed Schedule`, `Synthetic Fair Health Reference`, `None` |
| `reference_date` | string (ISO-8601) | Yes | Effective timestamp of reference benchmark quote. | YYYY-MM-DD or null |
| `match_status` | string | No | Matching pipeline status between bill line and reference catalog. | `matched`, `unmatched`, `unit_mismatch`, `missing_reference` |
| `suitable_for_comparison` | boolean | No | Governs whether line item qualifies for algorithmic variance calculation. `False` if units mismatch or benchmark absent. | `True`, `False` |
| `injected_discrepancy` | boolean | No | Synthetic ground-truth label: item was deliberately modified with an abnormal markup (>50% to >200%) for algorithm validation. | `True`, `False` |
| `synthetic_disclaimer` | string | No | Mandatory regulatory and ethical compliance banner string. | `SYNTHETIC DEMO RECORD - NOT REAL CLINICAL DATA` |
