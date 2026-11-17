# AgriVision AI

An agricultural analytics dashboard that builds on the project's existing crop/environmental CSV, hyperspectral TIFFs, pest image folders, and unsupervised ML utilities.

## Run locally

From the project directory:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Dataset paths resolve relative to the project directory by default. Optional environment variables can point to datasets stored elsewhere:

- `AGRIVISION_DATA_ROOT`: base for the default project data layout.
- `AGRIVISION_CROP_CSV`: full crop CSV path.
- `AGRIVISION_HYPERSPECTRAL_DIR`: hyperspectral root with split folders.
- `AGRIVISION_PEST_DIR`: pest classification root with `train`, `val`, and `test`.
- `AGRIVISION_PEST_CLASSES`: pest class-name file.

## Dashboard navigation

**Overview**
- Dashboard Overview

**Data Management**
- Dataset Explorer — independent previews and exports for each dataset.
- Crop Dataset — actual CSV records, field descriptions/types, missing values, distributions, and correlation.
- Hyperspectral Dataset — TIFF browser, metadata, pseudo-RGB, pixel spectra, and per-band statistics.
- Pest Image Dataset — split/class browser, image gallery, image dimensions, and class counts.

**Analysis & AI**
- Crop & Soil Analysis — observed feature patterns, NPK and soil comparisons, correlations, and optional K-Means.
- Hyperspectral Analysis — PCA, K-Means, DBSCAN, or Isolation Forest on TIFF-derived band statistics.
- Pest Image Analysis — image-feature similarity, unsupervised groupings, and representative images.
- Unsupervised ML Lab — run PCA, K-Means, DBSCAN, or Isolation Forest on selected real project features.

**Reports**
- Analytics & Visualizations — filterable dataset charts and saved ML-result plots.
- Reports & Downloads — dataset and class summaries, spectral reports, and session ML results.

The TIFF pseudo-RGB uses band indices for display only. The bundled TIFFs do not contain verified wavelength mappings, so the app does not calculate NDVI. Clusters and outliers represent feature similarity and are not confirmed crop, pest-species, disease, or infestation predictions. PDF export is not included; CSV and interactive HTML reports are available without adding a PDF dependency.
