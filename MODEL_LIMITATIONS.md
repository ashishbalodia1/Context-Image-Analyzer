# CITY GUARDIAN — SCIENTIFIC BOUNDARIES & MODEL LIMITATIONS

In adherence to strict scientific and engineering integrity, this document details the operational boundary conditions, false positive failure modes, optical vs chemical sensor distinctions, and environmental degradation factors for the **City Guardian** system.

---

## 1. Visual Severity vs Physical Sensor Measurements

> [!WARNING]
> **Fundamental Principle**: YOLOv8 is a 2D optical object detection model operating on RGB imagery. It detects **visible features** (e.g. smoke opacity, surface discoloration, debris piles). It **cannot and does not measure chemical concentrations**.

- **Air Quality**: Optical plume detection cannot measure exact parts per million (ppm) of $\text{PM}_{2.5}$, $\text{PM}_{10}$, $\text{SO}_2$, $\text{NO}_x$, or $\text{CO}$. For statutory air quality compliance, visual reports must be correlated with continuous ambient air quality monitoring stations (CAAQMS) via OpenWeatherMap or IoT telemetry.
- **Water Quality**: Surface oil sheen and turbidity can indicate contamination, but optical sensors cannot determine biological oxygen demand (BOD), chemical oxygen demand (COD), pH levels, or heavy metal parts-per-billion without physical sensor telemetry.
- **Naming Rule**: All automated severity ratings produced by this computer vision pipeline are designated as **"Visual Severity"** and not "Environmental Toxicity".

---

## 2. Adverse Lighting & Nocturnal Degradation

- **Extreme Low-Light Conditions (< 5 Lux)**: At night, mobile cameras increase ISO sensitivity, causing heavy CMOS chroma noise and severe loss of high-frequency edges.
  - *Impact*: Dark solid waste piles can blend with unlit pavements; open drains may be completely occluded.
  - *Mitigation*: The `EnvironmentalImageEnhancer` runs adaptive gamma expansion ($\gamma = 1.35\text{--}1.5$), but flash photography or minimum street lighting is mandatory for reliable confidence.

---

## 3. Atmospheric & Weather Occlusion

- **Dense Fog & Smog Inversion**: Early winter mornings frequently experience heavy radiation fog.
  - *Risk*: Atmospheric mist can visually mimic industrial dust clouds or construction plumes.
  - *Mitigation*: The system checks ambient atmospheric humidity via external APIs; if relative humidity exceeds 92%, dust cloud confidence thresholds are elevated to suppress false alarms.
- **Monsoon & Heavy Rain Streaks**: Raindrop streaks on lenses refract light and distort bounding box aspect ratios.

---

## 4. Geometric & Perspective Distortions

- **Extreme Off-Axis Angles (> 60°)**: Photographs captured from high multi-story balconies looking steeply down at road dust or drainage channels cause perspective foreshortening.
- **Distance Degradation (> 20 meters)**: Small waste objects (< 16x16 pixels) lose feature representation after $640 \times 640$ resizing.
- **Best Practice Guideline**: Citizens are prompted in the mobile application to capture photos within **2 to 8 meters** of the incident scene at an angle between 15° and 45°.

---

## 5. False Positive Triggers & Edge Cases

| Visual Anomaly | Potential False Positive | Verification Mechanism |
| :--- | :--- | :--- |
| Wet asphalt after light rain | Stagnant wastewater / sewage leak | Checks road boundaries & water pooling reflection depth |
| Dark shadows from tall structures | Open drainage channel / ditch | CLAHE equalization & shadow edge continuity verification |
| Clean riverbed stones | Construction debris / illegal dumping | Color variance & texture roughness analysis |
| Steam from bakeries / laundry | Industrial toxic smoke emission | Plume dissipation rate & localized commercial zoning flags |

---

## 6. Human-in-the-Loop Verification Requirement

All incident reports generated with **Visual Severity = HIGH or CRITICAL** are flagged with `"verification_required": true`. Municipal sanitation or environmental officers must validate the evidence image and GPS coordinates prior to dispatching punitive statutory penalties under national environmental acts.

