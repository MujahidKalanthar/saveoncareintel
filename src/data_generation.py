"""
Synthetic Data Generator for SaveOnCare Intelligence Lab.

Generates reproducible synthetic hospital billing data for demonstration,
benchmarking coverage analysis, discrepancy detection, and data quality testing.

IMPORTANT DISCLAIMER:
All records, prices, benchmarks, and hospital identifiers in this dataset are
synthetically generated for demonstration and research purposes only. They do not
represent actual hospitals, real patient bills, NPPA ceiling prices, or verified
statutory rates. Injected test discrepancies are deliberate anomalies created for
algorithm evaluation and must not be interpreted as evidence of actual fraud.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

# Clinical catalogs by category
CATALOG = {
    "medicines": [
        {
            "id": "MED-PCM-500",
            "desc": "Paracetamol 500mg IV Infusion",
            "unit": "vial",
            "ref_price": 42.0,
            "ref_source": "Synthetic State Tender Benchmark",
            "std_markup": 1.15,
        },
        {
            "id": "MED-MRP-1000",
            "desc": "Meropenem 1g Injection",
            "unit": "vial",
            "ref_price": 950.0,
            "ref_source": "Synthetic Market Median",
            "std_markup": 1.25,
        },
        {
            "id": "MED-PAN-40",
            "desc": "Pantoprazole 40mg IV Injection",
            "unit": "vial",
            "ref_price": 48.0,
            "ref_source": "Synthetic State Tender Benchmark",
            "std_markup": 1.20,
        },
        {
            "id": "MED-OND-4",
            "desc": "Ondansetron 4mg IV Injection",
            "unit": "ampoule",
            "ref_price": 18.0,
            "ref_source": "Synthetic State Tender Benchmark",
            "std_markup": 1.18,
        },
        {
            "id": "MED-CTX-1000",
            "desc": "Ceftriaxone 1g Injection",
            "unit": "vial",
            "ref_price": 65.0,
            "ref_source": "Synthetic Market Median",
            "std_markup": 1.22,
        },
        {
            "id": "MED-ENX-40",
            "desc": "Enoxaparin 40mg Prefilled Syringe",
            "unit": "syringe",
            "ref_price": 420.0,
            "ref_source": "Synthetic Insurer Agreed Schedule",
            "std_markup": 1.20,
        },
        {
            "id": "MED-NS-500",
            "desc": "Normal Saline 0.9% 500ml IV Fluid",
            "unit": "bottle",
            "ref_price": 38.0,
            "ref_source": "Synthetic State Tender Benchmark",
            "std_markup": 1.30,
        },
        {
            "id": "MED-AUG-625",
            "desc": "Amoxicillin + Clavulanate 625mg",
            "unit": "strip",
            "ref_price": 135.0,
            "ref_source": "Synthetic Market Median",
            "std_markup": 1.15,
        },
        {
            "id": "MED-INS-GLAR",
            "desc": "Insulin Glargine 100 IU/ml",
            "unit": "cartridge",
            "ref_price": 680.0,
            "ref_source": "Synthetic Market Median",
            "std_markup": 1.12,
        },
        {
            "id": "MED-ATC-40",
            "desc": "Atorvastatin 40mg Tablets",
            "unit": "strip",
            "ref_price": 110.0,
            "ref_source": "Synthetic Fair Health Reference",
            "std_markup": 1.15,
        },
        {
            "id": "MED-EXP-NEW",
            "desc": "Novel Monoclonal Antibody Infusion",
            "unit": "vial",
            "ref_price": None,
            "ref_source": None,
            "std_markup": 1.40,
        },
    ],
    "consumables": [
        {
            "id": "CON-CAN-20",
            "desc": "IV Cannula 20G with Port",
            "unit": "piece",
            "ref_price": 55.0,
            "ref_source": "Synthetic State Tender Benchmark",
            "std_markup": 1.35,
        },
        {
            "id": "CON-SYR-10",
            "desc": "Disposable Syringe 10ml with Needle",
            "unit": "piece",
            "ref_price": 12.0,
            "ref_source": "Synthetic State Tender Benchmark",
            "std_markup": 1.50,
        },
        {
            "id": "CON-GLV-75",
            "desc": "Surgical Latex Gloves Size 7.5 (Sterile)",
            "unit": "pair",
            "ref_price": 34.0,
            "ref_source": "Synthetic Market Median",
            "std_markup": 1.40,
        },
        {
            "id": "CON-IVS-VENT",
            "desc": "IV Infusion Set Vented with Micron Filter",
            "unit": "set",
            "ref_price": 85.0,
            "ref_source": "Synthetic Fair Health Reference",
            "std_markup": 1.35,
        },
        {
            "id": "CON-FOL-16",
            "desc": "Foley 2-Way Catheter 16Fr (Silicone)",
            "unit": "piece",
            "ref_price": 180.0,
            "ref_source": "Synthetic Insurer Agreed Schedule",
            "std_markup": 1.30,
        },
        {
            "id": "CON-URB-2000",
            "desc": "Urine Drainage Bag 2000ml with Anti-Reflux",
            "unit": "bag",
            "ref_price": 75.0,
            "ref_source": "Synthetic State Tender Benchmark",
            "std_markup": 1.45,
        },
        {
            "id": "CON-ECG-ELC",
            "desc": "Disposable ECG Electrodes",
            "unit": "pack",
            "ref_price": 120.0,
            "ref_source": "Synthetic Market Median",
            "std_markup": 1.25,
        },
        {
            "id": "CON-PPE-KIT",
            "desc": "Protective Isolation Kit Standard",
            "unit": "kit",
            "ref_price": 350.0,
            "ref_source": "Synthetic Fair Health Reference",
            "std_markup": 1.60,
        },
        {
            "id": "CON-GAU-ROLL",
            "desc": "Sterile Abdominal Gauze Sponge Pack",
            "unit": "pack",
            "ref_price": 95.0,
            "ref_source": "Synthetic State Tender Benchmark",
            "std_markup": 1.40,
        },
        {
            "id": "CON-SPE-DRP",
            "desc": "Custom Orthopedic Surgical Drape Kit",
            "unit": "kit",
            "ref_price": None,
            "ref_source": None,
            "std_markup": 1.80,
        },
    ],
    "diagnostics": [
        {
            "id": "DIA-CBC",
            "desc": "Complete Blood Count (CBC) with Automated Differential",
            "unit": "test",
            "ref_price": 320.0,
            "ref_source": "Synthetic Fair Health Reference",
            "std_markup": 1.25,
        },
        {
            "id": "DIA-KFT",
            "desc": "Renal Function Test (Urea, Creatinine, Electrolytes)",
            "unit": "panel",
            "ref_price": 650.0,
            "ref_source": "Synthetic Insurer Agreed Schedule",
            "std_markup": 1.30,
        },
        {
            "id": "DIA-LFT",
            "desc": "Liver Function Test Comprehensive",
            "unit": "panel",
            "ref_price": 750.0,
            "ref_source": "Synthetic Fair Health Reference",
            "std_markup": 1.28,
        },
        {
            "id": "DIA-CXR",
            "desc": "Chest X-Ray PA View Digital",
            "unit": "scan",
            "ref_price": 450.0,
            "ref_source": "Synthetic State Tender Benchmark",
            "std_markup": 1.35,
        },
        {
            "id": "DIA-ECG-12",
            "desc": "12-Lead Electrocardiogram with Interpretation",
            "unit": "test",
            "ref_price": 250.0,
            "ref_source": "Synthetic Insurer Agreed Schedule",
            "std_markup": 1.20,
        },
        {
            "id": "DIA-CT-CHEST",
            "desc": "High Resolution Computed Tomography (HRCT) Chest",
            "unit": "scan",
            "ref_price": 3800.0,
            "ref_source": "Synthetic Fair Health Reference",
            "std_markup": 1.30,
        },
        {
            "id": "DIA-MRI-BRAIN",
            "desc": "Magnetic Resonance Imaging (MRI) Brain Contrast",
            "unit": "scan",
            "ref_price": 6200.0,
            "ref_source": "Synthetic Fair Health Reference",
            "std_markup": 1.25,
        },
        {
            "id": "DIA-USG-ABD",
            "desc": "Ultrasonography Whole Abdomen & Pelvis",
            "unit": "scan",
            "ref_price": 1400.0,
            "ref_source": "Synthetic Insurer Agreed Schedule",
            "std_markup": 1.25,
        },
        {
            "id": "DIA-CRP-HS",
            "desc": "High Sensitivity C-Reactive Protein (hs-CRP)",
            "unit": "test",
            "ref_price": 480.0,
            "ref_source": "Synthetic Market Median",
            "std_markup": 1.22,
        },
        {
            "id": "DIA-GEN-PANEL",
            "desc": "Advanced Pharmacogenomics Multigene Assay",
            "unit": "test",
            "ref_price": None,
            "ref_source": None,
            "std_markup": 1.50,
        },
    ],
    "procedures": [
        {
            "id": "PRC-APP-LAP",
            "desc": "Laparoscopic Appendectomy Procedure Fee",
            "unit": "procedure",
            "ref_price": 28000.0,
            "ref_source": "Synthetic Insurer Agreed Schedule",
            "std_markup": 1.25,
        },
        {
            "id": "PRC-HERN-UNI",
            "desc": "Open Inguinal Hernioplasty with Mesh",
            "unit": "procedure",
            "ref_price": 24000.0,
            "ref_source": "Synthetic Fair Health Reference",
            "std_markup": 1.20,
        },
        {
            "id": "PRC-NORM-DEL",
            "desc": "Normal Vaginal Delivery Management",
            "unit": "procedure",
            "ref_price": 18500.0,
            "ref_source": "Synthetic Insurer Agreed Schedule",
            "std_markup": 1.20,
        },
        {
            "id": "PRC-CS-SEC",
            "desc": "Lower Segment Caesarean Section (LSCS)",
            "unit": "procedure",
            "ref_price": 32000.0,
            "ref_source": "Synthetic Insurer Agreed Schedule",
            "std_markup": 1.25,
        },
        {
            "id": "PRC-COR-ANGIO",
            "desc": "Coronary Angiography Diagnostic",
            "unit": "procedure",
            "ref_price": 14500.0,
            "ref_source": "Synthetic Fair Health Reference",
            "std_markup": 1.30,
        },
        {
            "id": "PRC-HEMO-DIAL",
            "desc": "Hemodialysis Session Maintenance",
            "unit": "session",
            "ref_price": 2100.0,
            "ref_source": "Synthetic State Tender Benchmark",
            "std_markup": 1.15,
        },
        {
            "id": "PRC-ENDO-UPP",
            "desc": "Upper Gastrointestinal Diagnostic Endoscopy",
            "unit": "procedure",
            "ref_price": 4200.0,
            "ref_source": "Synthetic Fair Health Reference",
            "std_markup": 1.25,
        },
        {
            "id": "PRC-EXP-ROBOT",
            "desc": "Experimental Robotic Soft Tissue Resection",
            "unit": "procedure",
            "ref_price": None,
            "ref_source": None,
            "std_markup": 1.65,
        },
    ],
    "services": [
        {
            "id": "SRV-ICU-BED",
            "desc": "Intensive Care Unit (ICU) Bed Charge / 24h",
            "unit": "day",
            "ref_price": 8500.0,
            "ref_source": "Synthetic Fair Health Reference",
            "std_markup": 1.30,
        },
        {
            "id": "SRV-GW-BED",
            "desc": "General Ward Room & Board / 24h",
            "unit": "day",
            "ref_price": 1800.0,
            "ref_source": "Synthetic Insurer Agreed Schedule",
            "std_markup": 1.20,
        },
        {
            "id": "SRV-PVT-ROOM",
            "desc": "Single Deluxe Private Room / 24h",
            "unit": "day",
            "ref_price": 4500.0,
            "ref_source": "Synthetic Fair Health Reference",
            "std_markup": 1.35,
        },
        {
            "id": "SRV-NUR-ICU",
            "desc": "Critical Care Dedicated Nursing Service / Day",
            "unit": "day",
            "ref_price": 2200.0,
            "ref_source": "Synthetic Insurer Agreed Schedule",
            "std_markup": 1.25,
        },
        {
            "id": "SRV-CNS-SPEC",
            "desc": "Senior Specialist Inpatient Daily Visit",
            "unit": "visit",
            "ref_price": 1200.0,
            "ref_source": "Synthetic Insurer Agreed Schedule",
            "std_markup": 1.25,
        },
        {
            "id": "SRV-OT-MAJ",
            "desc": "Major Operating Theatre Running Charges (First 2h)",
            "unit": "session",
            "ref_price": 11000.0,
            "ref_source": "Synthetic Fair Health Reference",
            "std_markup": 1.30,
        },
        {
            "id": "SRV-BIO-WASTE",
            "desc": "Bio-Medical Waste Management Handling Surcharge",
            "unit": "day",
            "ref_price": 250.0,
            "ref_source": "Synthetic State Tender Benchmark",
            "std_markup": 1.80,
        },
        {
            "id": "SRV-VIP-CONCIERGE",
            "desc": "Executive Health Suite Concierge Amenity",
            "unit": "day",
            "ref_price": None,
            "ref_source": None,
            "std_markup": 2.20,
        },
    ],
}

HOSPITALS = [
    {"id": "HOSP-BLR-01", "name": "Apex City Multispecialty (Bengaluru)", "tier": "Tier 1 Metro"},
    {"id": "HOSP-BLR-02", "name": "Metro Crest Hospital (Bengaluru)", "tier": "Tier 1 Metro"},
    {"id": "HOSP-DEL-03", "name": "Capital Health Pavilion (New Delhi)", "tier": "Tier 1 Metro"},
    {"id": "HOSP-DEL-04", "name": "NorthStar Medical Centre (New Delhi)", "tier": "Tier 1 Metro"},
    {"id": "HOSP-MUM-05", "name": "Western Super Specialty (Mumbai)", "tier": "Tier 1 Metro"},
    {"id": "HOSP-PUN-06", "name": "Sahyadri Heritage Hospital (Pune)", "tier": "Tier 2 Urban"},
    {"id": "HOSP-HYD-07", "name": "Deccan Care Institute (Hyderabad)", "tier": "Tier 1 Metro"},
    {"id": "HOSP-KOL-08", "name": "Eastern Gateway Health (Kolkata)", "tier": "Tier 2 Urban"},
]


def generate_synthetic_dataset(seed: int = 42, num_bills: int = 500) -> pd.DataFrame:
    """
    Generates a reproducible synthetic billing dataset containing line items
    across ~500 hospital bills, with deliberate data anomalies and test discrepancy labels.
    """
    rng = np.random.default_rng(seed)

    rows: list[dict[str, Any]] = []
    start_date = datetime.date(2024, 1, 1)

    categories = list(CATALOG.keys())

    for bill_idx in range(1, num_bills + 1):
        bill_id = f"BILL-2024-{bill_idx:04d}"
        hospital = rng.choice(HOSPITALS)
        # Random bill date across 2024
        days_offset = rng.integers(0, 365)
        bill_date = start_date + datetime.timedelta(days=int(days_offset))

        # Each bill has between 3 and 10 line items
        num_items = rng.integers(3, 11)

        # Distribute categories realistically: medicines & consumables most frequent
        cat_choices = rng.choice(
            categories,
            size=num_items,
            p=[0.35, 0.25, 0.20, 0.10, 0.10],
        )

        for item_idx, cat in enumerate(cat_choices, 1):
            line_id = f"{bill_id}-L{item_idx:02d}"
            item_pool = CATALOG[cat]
            item_template = rng.choice(item_pool)

            desc = item_template["desc"]
            prod_id = item_template["id"]
            ref_unit = item_template["unit"]
            ref_price = item_template["ref_price"]
            ref_source = item_template["ref_source"]
            std_markup = item_template["std_markup"]

            # Quantities vary by category
            if cat == "medicines":
                qty = int(rng.choice([1, 2, 3, 5, 10], p=[0.25, 0.35, 0.20, 0.15, 0.05]))
            elif cat == "consumables":
                qty = int(rng.choice([1, 2, 4, 6, 10, 20], p=[0.20, 0.30, 0.20, 0.15, 0.10, 0.05]))
            elif cat == "diagnostics":
                qty = 1
            elif cat == "procedures":
                qty = 1
            else:  # services
                qty = int(rng.choice([1, 2, 3, 4, 5, 7], p=[0.30, 0.25, 0.20, 0.10, 0.10, 0.05]))

            billed_unit = ref_unit
            suitable_for_comparison = True
            match_status = "matched" if ref_price is not None else "missing_reference"
            injected_discrepancy = False

            # Synthetic variations:
            # 1. Missing reference (~10% already in catalog)
            if ref_price is None:
                suitable_for_comparison = False
                ref_unit_price = np.nan
                ref_date = None
                billed_unit_price = round(float(rng.uniform(500, 15000)), 2)
            else:
                ref_unit_price = float(ref_price)
                ref_date = (bill_date - datetime.timedelta(days=int(rng.integers(15, 120)))).isoformat()

                # Scenario dice:
                dice = rng.random()

                if dice < 0.13:
                    # Intentionally injected significant test discrepancy (50% to 220% above reference)
                    injected_discrepancy = True
                    markup = float(rng.uniform(1.50, 3.20))
                    billed_unit_price = round(ref_unit_price * markup, 2)
                elif dice < 0.17:
                    # Unit mismatch injection: e.g. billed as 'pack of 10' or 'box' vs reference 'piece'
                    billed_unit = f"pack_of_10_{ref_unit}s"
                    match_status = "unit_mismatch"
                    suitable_for_comparison = False
                    # Billed price represents a pack of 10
                    billed_unit_price = round(ref_unit_price * 9.5, 2)
                elif dice < 0.21:
                    # Product catalog unmatched scenario
                    match_status = "unmatched"
                    suitable_for_comparison = False
                    ref_unit_price = np.nan
                    ref_source = None
                    ref_date = None
                    billed_unit_price = round(ref_unit_price_fallback := ref_price * float(rng.uniform(1.1, 1.4)), 2)
                elif dice < 0.25:
                    # Minor legitimate institutional variation (within -5% to +20%)
                    variation = float(rng.uniform(0.95, 1.20))
                    billed_unit_price = round(ref_unit_price * variation, 2)
                else:
                    # Normal standard contracted / market pricing
                    noise = float(rng.normal(0, 0.06))
                    factor = max(0.90, std_markup + noise)
                    billed_unit_price = round(ref_unit_price * factor, 2)

            billed_line_amount = round(qty * billed_unit_price, 2)

            rows.append(
                {
                    "line_item_id": line_id,
                    "bill_id": bill_id,
                    "hospital_id": hospital["id"],
                    "hospital_name": hospital["name"],
                    "hospital_tier": hospital["tier"],
                    "bill_date": bill_date.isoformat(),
                    "category": cat,
                    "product_or_service_id": prod_id,
                    "item_description": desc,
                    "quantity": qty,
                    "billed_unit": billed_unit,
                    "billed_unit_price": billed_unit_price,
                    "billed_line_amount": billed_line_amount,
                    "reference_unit": ref_unit,
                    "reference_unit_price": ref_unit_price,
                    "reference_source": ref_source,
                    "reference_date": ref_date,
                    "match_status": match_status,
                    "suitable_for_comparison": suitable_for_comparison,
                    "injected_discrepancy": injected_discrepancy,
                    "synthetic_disclaimer": "SYNTHETIC DEMO RECORD - NOT REAL CLINICAL DATA",
                }
            )

    df = pd.DataFrame(rows)

    # Injected deliberate edge cases for Data Quality and Validation testing:
    # A. 10 duplicate rows
    dup_indices = rng.choice(len(df), size=10, replace=False)
    duplicates = df.iloc[dup_indices].copy()
    # append duplicates
    df = pd.concat([df, duplicates], ignore_index=True)

    # B. 5 rows with invalid quantities (0 or negative)
    invalid_qty_idx = rng.choice(len(df), size=5, replace=False)
    for i, idx in enumerate(invalid_qty_idx):
        df.at[idx, "quantity"] = 0 if i % 2 == 0 else -1
        # Recalculate line amount to keep consistent or test math
        df.at[idx, "billed_line_amount"] = round(df.at[idx, "quantity"] * df.at[idx, "billed_unit_price"], 2)

    # C. 4 rows with non-positive reference prices
    non_pos_ref_idx = rng.choice(
        df[df["match_status"] == "matched"].index,
        size=4,
        replace=False,
    )
    for i, idx in enumerate(non_pos_ref_idx):
        df.at[idx, "reference_unit_price"] = 0.0 if i % 2 == 0 else -50.0

    # D. 5 rows with arithmetic calculation mismatches (e.g. manual typing error on bill)
    calc_err_idx = rng.choice(len(df), size=5, replace=False)
    for idx in calc_err_idx:
        df.at[idx, "billed_line_amount"] = round(df.at[idx, "billed_line_amount"] + 350.0, 2)

    return df


def save_synthetic_dataset(output_path: str = "data/synthetic_billing_data.csv", seed: int = 42) -> pd.DataFrame:
    """Generates and saves the synthetic dataset to CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_dataset(seed=seed)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    df = save_synthetic_dataset()
    print(f"Generated synthetic dataset with {len(df)} rows across {df['bill_id'].nunique()} bills.")
