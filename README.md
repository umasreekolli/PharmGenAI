# 💊 PharmGenAI
### Pharmacogenomics-based Clinical Decision Support System

## 🔴 Live Demo
👉 [Click here to open app](https://pharmgenai.streamlit.app)

---

## Problem
Adverse Drug Reactions (ADRs) cause 6.5% of hospital admissions 
worldwide. Genetic variants in CYP450 enzymes directly affect how 
patients metabolize drugs — but this is rarely checked before 
prescribing.

---

## Solution
AI-based ADR risk prediction tool that integrates:
- Patient clinical data (age, comorbidities, polypharmacy)
- Pharmacogenomic markers (CYP2D6, CYP3A4, CYP2C19)
- Drug-drug interaction (DDI) detection
- Pharmacist dose recommendations + alternative drugs
- Explainable AI (Feature importance visualization)

---

## Features
- 🧬 CYP450 genetic marker input (3 enzymes)
- 💊 10 common drugs supported
- ⚠️ Drug-drug interaction alerts
- 🔴 Risk prediction — Low / Medium / High
- 📋 Clinical explanation of WHY risk is high
- 💊 Pharmacist action + monitoring + alternative drug
- 📊 Feature importance chart
- 📁 Session patient log

---

## Tech Stack
- Python
- Scikit-learn (Random Forest — 84% accuracy)
- SHAP (Explainability)
- Streamlit (Web UI)
- Plotly (Charts)
- Pandas, NumPy

---

## How to Run Locally
```bash
git clone https://github.com/umasreekolli/PharmGenAI
cd PharmGenAI
pip install -r requirements.txt
streamlit run app/app.py
```

---

## Project Structure
