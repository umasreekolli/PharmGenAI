import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# --- Page Config ---
st.set_page_config(
    page_title="PharmGenAI",
    page_icon="💊",
    layout="wide"
)
# --- Load Model Files ---
model     = joblib.load('model/adr_model.pkl')
explainer = joblib.load('model/shap_explainer.pkl')
features  = joblib.load('model/feature_names.pkl')

# --- Encoding Maps ---
cyp_map = {
    'Poor': 0, 'Intermediate': 1,
    'Normal': 2, 'Ultrarapid': 3,
    'Fast': 2, 'Rapid': 2
}

drug_list = [
    'Warfarin', 'Paracetamol', 'Ibuprofen', 'Fluconazole',
    'Aspirin', 'Metformin', 'Atorvastatin', 'Omeprazole',
    'Codeine', 'Clopidogrel'
]

drug_codes = {d: i for i, d in enumerate(sorted(drug_list))}

# --- DDI Knowledge Base ---
DDI_PAIRS = {
    ('Aspirin',      'Warfarin'):     'Increased bleeding risk',
    ('Ibuprofen',    'Warfarin'):     'Increased bleeding risk',
    ('Fluconazole',  'Warfarin'):     'Warfarin toxicity via CYP2C19 inhibition',
    ('Codeine',      'Fluconazole'):  'Reduced Codeine efficacy',
    ('Atorvastatin', 'Fluconazole'):  'Statin toxicity via CYP3A4 inhibition',
    ('Clopidogrel',  'Omeprazole'):   'Reduced antiplatelet effect via CYP2C19',
}

# --- Pharmacist Recommendation Database ---
RECOMMENDATIONS = {
    ('Warfarin', 'Poor', 'CYP2D6'): {
        'action':      'Reduce Warfarin dose by 30–50%',
        'monitoring':  'Check INR every 3 days until stable',
        'alternative': 'Consider Apixaban or Rivaroxaban'
    },
    ('Warfarin', 'Poor', 'CYP2C19'): {
        'action':      'Warfarin sensitivity increased — start with low dose',
        'monitoring':  'Monitor INR closely for first 2 weeks',
        'alternative': 'Consider Dabigatran (not CYP dependent)'
    },
    ('Clopidogrel', 'Poor', 'CYP2C19'): {
        'action':      'Clopidogrel ineffective — poor metabolizers cannot activate it',
        'monitoring':  'Monitor platelet aggregation function',
        'alternative': 'Switch to Ticagrelor or Prasugrel'
    },
    ('Codeine', 'Poor', 'CYP2D6'): {
        'action':      'Codeine will not work — cannot convert to active morphine',
        'monitoring':  'Assess pain relief every 2 hours',
        'alternative': 'Use Tramadol or direct Morphine instead'
    },
    ('Codeine', 'Ultrarapid', 'CYP2D6'): {
        'action':      'DANGER — toxic morphine levels, avoid Codeine completely',
        'monitoring':  'Watch for respiratory depression immediately',
        'alternative': 'Use non-opioid analgesic (Paracetamol + Ibuprofen)'
    },
    ('Atorvastatin', 'Poor', 'CYP3A4'): {
        'action':      'High statin toxicity risk — reduce dose significantly',
        'monitoring':  'Check CK levels monthly, watch for muscle pain',
        'alternative': 'Switch to Rosuvastatin (not CYP3A4 dependent)'
    },
    ('Fluconazole', 'Poor', 'CYP3A4'): {
        'action':      'Fluconazole accumulation risk — reduce dose or frequency',
        'monitoring':  'Monitor liver function tests (LFT)',
        'alternative': 'Consider Micafungin (not CYP dependent)'
    },
    ('Omeprazole', 'Poor', 'CYP2C19'): {
        'action':      'Higher Omeprazole levels — may cause over-suppression',
        'monitoring':  'Monitor for headache, GI symptoms',
        'alternative': 'Reduce dose to 10mg or use Pantoprazole'
    },
    ('Ibuprofen', 'Poor', 'CYP3A4'): {
        'action':      'Ibuprofen clearance reduced — GI and renal risk increases',
        'monitoring':  'Monitor kidney function (creatinine)',
        'alternative': 'Use Paracetamol as safer alternative'
    },
    ('Metformin', 'Poor', 'CYP2C19'): {
        'action':      'Monitor for lactic acidosis risk in poor metabolizers',
        'monitoring':  'Check renal function every 3 months',
        'alternative': 'Consider dose reduction if renal function declining'
    },
}

# --- Risk Labels ---
RISK_LABEL = {0: '🟢 LOW',    1: '🟡 MEDIUM',    2: '🔴 HIGH'}
RISK_COLOR = {0: 'green',     1: 'orange',        2: 'red'}
RISK_MSG   = {
    0: 'Low probability of adverse drug reaction.',
    1: 'Moderate risk. Monitor patient closely.',
    2: 'High risk. Review drug choice and dose immediately.'
}

# --- Header ---
st.title("💊 PharmGenAI")
st.caption("Pharmacogenomics-based Clinical Decision Support System")
st.warning("⚠️ For research and educational use only. Not for clinical decisions.")
st.divider()

# --- Input Form ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("🧑 Patient Details")
    drug       = st.selectbox("Primary Drug", sorted(drug_list))
    age        = st.slider("Age", 18, 80, 45)
    gender     = st.radio("Gender", ["Male", "Female"], horizontal=True)
    comorbid   = st.slider("Number of Comorbidities", 0, 3, 0,
                           help="Diabetes, hypertension, kidney disease etc.")
    num_drugs  = st.slider("Number of Concurrent Drugs", 1, 5, 1,
                           help="Total drugs patient is currently taking")

with col2:
    st.subheader("🧬 Genetic Markers (CYP450)")
    cyp2d6  = st.selectbox("CYP2D6 Status",
                           ['Poor', 'Intermediate', 'Normal', 'Ultrarapid'],
                           help="Affects Codeine, Warfarin, Tamoxifen")
    cyp3a4  = st.selectbox("CYP3A4 Status",
                           ['Poor', 'Normal', 'Fast'],
                           help="Affects Atorvastatin, Fluconazole")
    cyp2c19 = st.selectbox("CYP2C19 Status",
                           ['Poor', 'Normal', 'Rapid'],
                           help="Affects Clopidogrel, Omeprazole, Warfarin")

    st.subheader("💊 Drug Interaction Check")
    second_drug = st.selectbox(
        "Second Drug (optional)",
        ['None'] + [d for d in sorted(drug_list) if d != drug]
    )
    if second_drug != 'None':
        pair = tuple(sorted([str(drug), str(second_drug)]))
        if pair in DDI_PAIRS:
            st.error(f"⚠️ DDI Alert: {DDI_PAIRS[pair]}")
        else:
            st.success("✅ No known major interaction found")

st.divider()

# --- Predict Button ---
if st.button("🔍 Predict ADR Risk", use_container_width=True):

    # DDI risk score
    ddi_risk = 0
    if second_drug != 'None':
        pair = tuple(sorted([str(drug), str(second_drug)]))
        ddi_risk = 1 if pair in DDI_PAIRS else 0
    # Encode input
    row = {
        'Drug':          drug_codes[drug],
        'Age':           age,
        'Gender':        0 if gender == 'Male' else 1,
        'CYP2D6':        cyp_map[cyp2d6],
        'CYP3A4':        cyp_map[cyp3a4],
        'CYP2C19':       cyp_map[cyp2c19],
        'Comorbidities': comorbid,
        'NumDrugs':      num_drugs,
        'DDI_Risk':      ddi_risk
    }

    X_input = pd.DataFrame([row])
    for f in features:
        if f not in X_input.columns:
            X_input[f] = 0
    X_input = X_input[features]

    # Predict
    pred  = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0]

    # --- Result ---
    st.subheader("📊 Prediction Result")
    color = RISK_COLOR[pred]
    st.markdown(f"## :{color}[{RISK_LABEL[pred]}]")
    st.markdown(f"**{RISK_MSG[pred]}**")

    c1, c2, c3 = st.columns(3)
    c1.metric("🟢 Low Risk",    f"{proba[0]*100:.1f}%")
    c2.metric("🟡 Medium Risk", f"{proba[1]*100:.1f}%")
    c3.metric("🔴 High Risk",   f"{proba[2]*100:.1f}%")

    st.divider()

    # --- Clinical Explanation ---
    st.subheader("📋 Clinical Explanation")
    reasons = []

    if cyp2d6 == 'Poor':
        reasons.append(f"**CYP2D6 Poor Metabolizer** → Slow breakdown of {drug} → Drug accumulates → Toxicity risk ↑")
    if cyp2d6 == 'Ultrarapid':
        reasons.append(f"**CYP2D6 Ultrarapid** → Too-fast metabolism → Toxic metabolite surge possible")
    if cyp3a4 == 'Poor':
        reasons.append("**CYP3A4 Poor Metabolizer** → Impaired liver clearance → Drug stays longer in body")
    if cyp2c19 == 'Poor':
        reasons.append("**CYP2C19 Poor Metabolizer** → Prodrug activation reduced (critical for Clopidogrel)")
    if comorbid >= 2:
        reasons.append(f"**Multiple Comorbidities ({comorbid})** → Organ function compromised → ADR risk ↑")
    if num_drugs >= 3:
        reasons.append(f"**Polypharmacy ({num_drugs} drugs)** → Higher chance of drug interactions")
    if age > 60:
        reasons.append("**Age > 60** → Reduced renal/hepatic clearance → Slower drug elimination")
    if ddi_risk == 1:
        reasons.append(f"**DDI Detected** → {drug} + {second_drug} interaction increases risk")

    if reasons:
        for r in reasons:
            st.markdown(f"- {r}")
    else:
        st.info("No major individual risk flags detected.")

    st.divider()

    # --- Pharmacist Recommendation ---
    st.subheader("💊 Pharmacist Recommendation")

    cyp_status_map = {
        'CYP2D6':  cyp2d6,
        'CYP3A4':  cyp3a4,
        'CYP2C19': cyp2c19
    }

    matched = []
    for (rec_drug, rec_status, rec_gene), rec in RECOMMENDATIONS.items():
        if rec_drug == drug and cyp_status_map[rec_gene] == rec_status:
            matched.append((rec_gene, rec))

    if matched:
        for gene, rec in matched:
            st.markdown(f"**Based on {gene} {cyp_status_map[gene]} status + {drug}:**")
            st.error(f"🔴 **Action:** {rec['action']}")
            st.warning(f"🟡 **Monitor:** {rec['monitoring']}")
            st.info(f"🔵 **Alternative:** {rec['alternative']}")
            st.markdown("---")
    elif pred == 2:
        st.error("🔴 High risk detected. Consult clinical pharmacist before dispensing.")
    elif pred == 1:
        st.warning("🟡 Moderate risk. Monitor patient response after first dose.")
    else:
        st.success("🟢 Standard precautions apply. No specific genetic interaction detected.")

    st.divider()

    # --- SHAP Chart ---
    st.subheader("🔬 SHAP Feature Contribution")
    st.caption("Shows which factors pushed the risk prediction up or down")

    try:
        shap_values = explainer.shap_values(X_input)
        fig, ax = plt.subplots(figsize=(8, 4))
        shap.summary_plot(
            shap_values[pred],
            X_input,
            feature_names=features,
            plot_type="bar",
            show=False
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.image('model/shap_importance.png',
                 caption="Overall feature importance")

    st.divider()

    # --- Session Log ---
    if 'log' not in st.session_state:
        st.session_state.log = []

    st.session_state.log.append({
        'Drug':      drug,
        'Age':       age,
        'Gender':    gender,
        'CYP2D6':    cyp2d6,
        'CYP3A4':    cyp3a4,
        'CYP2C19':   cyp2c19,
        'Num Drugs': num_drugs,
        'Risk':      RISK_LABEL[pred],
        'High %':    f"{proba[2]*100:.1f}%"
    })

# --- Patient Log ---
if 'log' in st.session_state and st.session_state.log:
    st.subheader("📁 Session Patient Log")
    st.caption("All predictions made in this session")
    st.dataframe(
        pd.DataFrame(st.session_state.log),
        use_container_width=True
    )
    if st.button("🗑️ Clear Log"):
        st.session_state.log = []
        st.rerun()
