# SaveOnCare Intelligence Lab — Healthcare Billing Analytics & Business Intelligence

A healthcare billing transparency, benchmark coverage auditing, discrepancy detection, and pilot economics simulation platform built for **SaveOnCare by VoltEdge**.

---

## 1. Quick Start & Execution

### Prerequisites
- Python 3.10+
- Dependencies installed via `pip`:
  ```bash
  pip install -r requirements.txt
  ```

### Launch the Streamlit Application
```bash
streamlit run app.py
```
Or with custom port (e.g. port 3000):
```bash
streamlit run app.py --server.port 3000 --server.address 0.0.0.0 --server.headless true
```

### Run Unit Tests
```bash
pytest -v
```

---

## 2. Project Architecture

```
├── app.py                     # Main multi-tab Streamlit intelligence lab dashboard
├── data/
│   ├── synthetic_billing_data.csv # Synthetic dataset (~500 bills, ~3,285 line items)
│   └── data_dictionary.md     # Full schema documentation & synthetic data disclaimer
├── src/
│   ├── __init__.py
│   ├── data_generation.py     # Deterministic dataset generator with clinical catalogs
│   ├── validation.py          # Data quality audit, schema, duplicate, & math verification
│   ├── processing.py          # Pipeline for cleaning, comparability, and variance calculations
│   ├── metrics.py             # Executive KPIs, category summaries, & test validation metrics
│   ├── business_model.py      # Pilot unit economics, break-even, & sensitivity modeling
│   └── visualisations.py      # Interactive Plotly charts & BI dashboards
├── tests/
│   ├── __init__.py
│   ├── test_metrics.py        # Unit tests for difference, variance, coverage, & economics
│   └── test_validation.py     # Unit tests for duplicates, invalid quantities, & units
├── requirements.txt           # Python dependency specifications
└── README.md                  # Comprehensive project documentation
```

---

## 3. Dataset Design & Ground Truth Injections

The dataset simulates 500 hospital inpatient/outpatient admissions with ~3,285 line items across 5 clinical categories:
- **Medicines:** e.g., Paracetamol IV, Meropenem, Pantoprazole, Enoxaparin, Insulin Glargine.
- **Consumables:** e.g., IV Cannulas, Syringes, Sterile Surgical Gloves, Foley Catheters, PPE.
- **Diagnostics:** e.g., Complete Blood Count (CBC), Renal Function, Chest X-Ray, HRCT Chest, MRI Brain.
- **Procedures:** e.g., Laparoscopic Appendectomy, Hernioplasty, Vaginal Delivery, Caesarean Section.
- **Services:** e.g., ICU Bed Charges, Deluxe Private Room, Specialist Doctor Visits, OT Charges.

### Benchmark Provenance Tiers
All benchmarks represent synthetic demonstration market medians:
- `Synthetic State Tender Benchmark` (wholesale procurement baselines)
- `Synthetic Market Median` (retail hospital pricing medians)
- `Synthetic Insurer Agreed Schedule` (negotiated institutional fee schedules)
- `Synthetic Fair Health Reference` (community-adjusted reference standards)

### Injected Data Anomalies for Algorithmic & Quality Testing
1. **Intentionally Injected Test Discrepancies (~13%):** Marked with `injected_discrepancy = True` with artificial markups of 50% to 220% above reference benchmark.
2. **Missing Reference Benchmarks (~10%):** Clinical items lacking an established market benchmark (`match_status = 'missing_reference'`).
3. **Packaging Unit Mismatches (~4%):** Items billed as `pack_of_10` or `box` compared against single-unit benchmarks (`match_status = 'unit_mismatch'`).
4. **Unmatched Products (~6%):** Novel or experimental therapies not found in master catalogs (`match_status = 'unmatched'`).
5. **Duplicate Rows:** 10 injected duplicate pairs to evaluate deduplication logic.
6. **Invalid Quantities:** 5 rows with quantities $\le 0$.
7. **Arithmetic Calculation Errors:** 5 rows where billed line amount $\neq (\text{quantity} \times \text{unit price})$.
8. **Non-Positive Reference Prices:** 4 rows with benchmark price $\le 0$.

---

## 4. Analytical Formulas & Metrics

### 1. Difference and Variance Calculation
For any item qualifying as **comparable** (`is_comparable = True`):
$$\text{Reference Line Amount} = \text{Quantity} \times \text{Reference Unit Price}$$
$$\text{Variance Amount (INR)} = \text{Billed Line Amount} - \text{Reference Line Amount}$$
$$\text{Percentage Variance (\%)} = \left(\frac{\text{Billed Line Amount} - \text{Reference Line Amount}}{\text{Reference Line Amount}}\right) \times 100$$

### 2. Benchmark Coverage
- **Item Coverage (%):**
  $$\text{Item Coverage} = \frac{\text{Number of Comparable Items}}{\text{Total Line Items}} \times 100$$
- **Value Coverage (%):**
  $$\text{Value Coverage} = \frac{\text{Total Billed Amount of Comparable Items}}{\text{Total Billed Amount across All Items}} \times 100$$

### 3. Potential Discrepancy Rate & Value
At user-defined threshold $\tau$ (default $+20\%$):
$$\text{Discrepancy Rate} = \frac{\text{Count of Comparable Items with Variance} > \tau}{\text{Total Comparable Items}} \times 100$$
$$\text{Potential Discrepancy Value} = \sum_{\text{flagged items}} (\text{Billed Line Amount} - \text{Reference Line Amount})$$

### 4. Variance Weighting
- **Item-Weighted Average:** Simple arithmetic mean of line-item percentage variances.
- **Amount-Weighted Average:** $\frac{\sum \text{Comparable Billed} - \sum \text{Comparable Ref}}{\sum \text{Comparable Ref}} \times 100$.

### 5. Synthetic Test Validation
$$\text{Precision} = \frac{TP}{TP + FP} \times 100, \quad \text{Recall} = \frac{TP}{TP + FN} \times 100, \quad F_1 = \frac{2 \cdot P \cdot R}{P + R}$$

---

## 5. Pilot Unit Economics Formulation

- **Assumed Revenue:** $(\text{Bills Processed} \times \text{Contract Price Per Bill}) + \text{Setup Fee}$
- **Variable Direct Costs:** $\text{Bills Processed} \times \text{Variable Cost Per Bill}$
- **Fixed Operating Costs:** $\text{Monthly Support} \times \text{Pilot Months}$
- **Total Modelled Costs:** $\text{Variable Costs} + \text{Fixed Operating Costs}$
- **Gross Contribution:** $\text{Assumed Revenue} - \text{Variable Direct Costs}$
- **Net Operating Surplus:** $\text{Assumed Revenue} - \text{Total Modelled Costs}$
- **Break-Even Volume (Bills):**
  $$\text{Break-Even} = \frac{\text{Fixed Operating Costs} - \text{Setup Fee}}{\text{Contract Price Per Bill} - \text{Variable Cost Per Bill}}$$

---

## 6. Regulatory & Clinical Limitations Before Real-World Deployment

1. **Synthetic Data Boundaries:** Demonstrates pipeline mathematics; cannot be cited as evidence of hospital overcharging.
2. **Clinical Complexity:** Real hospital charges depend on patient acuity, comorbidities, and surgical complications not captured in tabular unit baselines.
3. **Statutory Alignment:** Benchmarks must be officially cross-referenced with NPPA Schedule I drug ceilings, state CGHS guidelines, and bilateral TPA network tariffs before judicial assertion.
