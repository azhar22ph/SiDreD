import streamlit as st
import google.generativeai as genai  
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import py3Dmol
from stmol import showmol


# --- REST OF YOUR CODE STARTS HERE ---
# ==========================================
# 1. EXPANDED DRUG DATASET
# ==========================================
presets = {
    "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "Terfenadine": "CC(C)(C)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4",
    "Ciprofloxacin": "O=C(O)C1=CN(C2CC2)C2=CC(F)=C(N3CCNCC3)C=C2C1=O",
    "Diclofenac": "O=C(O)Cc1ccccc1Nc2c(Cl)cccc2Cl",
    "Amlodipine": "CCOC(=O)C1=C(COCCN)NC(=C(C1c2ccccc2Cl)C(=O)OC)C",
    "Omeprazole": "Cc1cncc(c1OC)CS(=O)c2nc3ccccc3[nH]2",
    "Losartan": "CCCCc1nc(c(n1Cc2ccc(cc2)c3ccccc3c4nn[nH]n4)CO)Cl",
    "Albuterol": "CC(C)(C)NCC(c1ccc(c(c1)CO)O)O",
    "Sertraline": "CN[C@H]1CC[C@@H](c2c1cccc2)c3ccc(c(c3)Cl)Cl",
    "Furosemide": "c1cc(c(cc1S(=O)(=O)N)Cl)NCc2ccco2",
    "Diazepam": "CN1C(=O)CN=C(C2=C1C=CC(=C2)Cl)C3=CC=CC=C3",
    "Fluoxetine": "CNCCC(c1ccccc1)Oc2ccc(cc2)C(F)(F)F",
    "Lisinopril": "N[C@@H](CCCCN)C(=O)N[C@@H](CCc1ccccc1)C(=O)N2CCC[C@H]2C(=O)O",
    "Clopidogrel": "COC(=O)C(c1ccccc1Cl)N2CCc3c(c2)sc(c3)C",
    "Tamsulosin": "COc1ccccc1OCCN[C@H](C)Cc2ccc(c(c2)S(=O)(=O)N)OC",
    "Levothyroxine": "N[C@@H](Cc1cc(I)c(Oc2cc(I)c(O)c(I)c2)c(I)c1)C(=O)O",
    "Rosuvastatin": "CC(C)c1nc(nc(n1)N(C)S(=O)(=O)C)C=CC(O)CC(O)CC(=O)O",
    "Pantoprazole": "COc1ccnc(c1)CS(=O)c2nc3ccc(cc3[nH]2)OC(F)F",
    "Carbamazepine": "NC(=O)N1c2ccccc2C=Cc3ccccc13"
}

brand_names = {
    "Ibuprofen": "Advil, Motrin",
    "Aspirin": "Bayer, Ecotrin",
    "Paracetamol": "Tylenol, Panadol",
    "Terfenadine": "Seldane (Discontinued)",
    "Ciprofloxacin": "Cipro",
    "Diclofenac": "Voltaren, Cataflam",
    "Amlodipine": "Norvasc",
    "Omeprazole": "Prilosec, Losec",
    "Losartan": "Cozaar",
    "Albuterol": "ProAir, Ventolin",
    "Sertraline": "Zoloft",
    "Furosemide": "Lasix",
    "Diazepam": "Valium",
    "Fluoxetine": "Prozac",
    "Lisinopril": "Prinivil, Zestril",
    "Clopidogrel": "Plavix",
    "Tamsulosin": "Flomax",
    "Levothyroxine": "Synthroid, Levoxyl",
    "Rosuvastatin": "Crestor",
    "Pantoprazole": "Protonix",
    "Carbamazepine": "Tegretol"
}

base_side_effects = {
    "Ibuprofen": ["Gastrointestinal irritation / Ulcers", "Hypertension (prolonged use)", "Renal impairment"],
    "Aspirin": ["Gastrointestinal bleeding", "Tinnitus", "Increased bleeding risk"],
    "Paracetamol": ["Hepatotoxicity (in overdose)", "Nausea", "Rare skin rash"],
    "Terfenadine": ["Cardiotoxicity (QT prolongation)", "Headache", "Mild sedation"],
    "Ciprofloxacin": ["Tendon rupture", "CNS effects", "QT prolongation risk"],
    "Diclofenac": ["Severe GI toxicity", "Hepatotoxicity", "Cardiovascular thrombotic events"],
    "Amlodipine": ["Peripheral edema", "Flushing", "Palpitations"],
    "Omeprazole": ["Headache", "C. diff associated diarrhea", "Bone fractures (long term)"],
    "Losartan": ["Hyperkalemia", "Dizziness", "Renal impairment"],
    "Albuterol": ["Tachycardia", "Tremor", "Hypokalemia"],
    "Sertraline": ["Sexual dysfunction", "Insomnia", "Weight changes"],
    "Furosemide": ["Electrolyte imbalance (Hypokalemia)", "Dehydration", "Ototoxicity"],
    "Diazepam": ["Drowsiness", "Dependence/Tolerance", "Respiratory depression"],
    "Fluoxetine": ["Anxiety/Agitation", "Insomnia", "Sexual dysfunction"],
    "Lisinopril": ["Dry cough", "Hyperkalemia", "Angioedema"],
    "Clopidogrel": ["Bleeding", "Thrombotic thrombocytopenic purpura (TTP)", "Dyspepsia"],
    "Tamsulosin": ["Orthostatic hypotension", "Ejaculatory dysfunction", "Dizziness"],
    "Levothyroxine": ["Hyperthyroidism symptoms (arrhythmia, weight loss)", "Osteoporosis (chronic high dose)"],
    "Rosuvastatin": ["Myopathy / Rhabdomyolysis", "Hepatotoxicity", "New-onset diabetes"],
    "Pantoprazole": ["Atrophic gastritis", "Hypomagnesemia", "Vitamin B12 deficiency"],
    "Carbamazepine": ["Stevens-Johnson syndrome", "Agranulocytosis", "Hyponatremia"]
}

drug_drug_interactions = {
    "Ibuprofen": ["Aspirin: Decreases antiplatelet effect.", "ACE inhibitors: Decreases antihypertensive efficacy."],
    "Aspirin": ["Warfarin: Synergistic bleeding risk.", "Methotrexate: Decreases renal clearance."],
    "Paracetamol": ["Warfarin: Prolongs INR.", "Phenytoin: Increases toxic NAPQI metabolite."],
    "Terfenadine": ["Ketoconazole: CYP3A4 inhibition leads to fatal arrhythmias."],
    "Ciprofloxacin": ["Theophylline: CYP1A2 inhibition causes seizures.", "Antacids: Chelation reduces absorption."],
    "Diclofenac": ["Lithium: Increases lithium toxicity.", "SSRIs: Increased GI bleeding risk."],
    "Amlodipine": ["Simvastatin: Increases statin toxicity.", "CYP3A4 inhibitors: Enhances edema/hypotension."],
    "Omeprazole": ["Clopidogrel: Inhibits CYP2C19, reducing Clopidogrel activation.", "Digoxin: Increases digoxin absorption."],
    "Losartan": ["Potassium-sparing diuretics: Severe hyperkalemia.", "NSAIDs: Reduces antihypertensive effect."],
    "Albuterol": ["Beta-blockers: Antagonizes bronchodilation.", "MAOIs: Hypertensive crisis risk."],
    "Sertraline": ["MAOIs: Serotonin syndrome.", "Pimozide: QT prolongation."],
    "Furosemide": ["Aminoglycosides: Synergistic ototoxicity.", "Lithium: Decreases lithium clearance."],
    "Diazepam": ["Opioids: Profound respiratory depression.", "Fluoxetine: Increases diazepam half-life."],
    "Fluoxetine": ["MAOIs: Fatal serotonin syndrome.", "TCAs: Dramatically increases TCA plasma levels."],
    "Lisinopril": ["ARBs: Increased renal toxicity.", "NSAIDs: Acute renal failure risk."],
    "Clopidogrel": ["Omeprazole: Reduces antiplatelet efficacy.", "NSAIDs: Increases GI bleeding."],
    "Tamsulosin": ["Sildenafil: Severe hypotension.", "Cimetidine: Decreases tamsulosin clearance."],
    "Levothyroxine": ["Calcium/Iron supplements: Blocks absorption.", "Warfarin: Enhances anticoagulant effect."],
    "Rosuvastatin": ["Cyclosporine: Dramatically increases statin levels.", "Gemfibrozil: Myopathy risk."],
    "Pantoprazole": ["Ketoconazole: Reduces absorption of pH-dependent drugs.", "Methotrexate: Elevates methotrexate levels."],
    "Carbamazepine": ["Oral Contraceptives: Induces metabolism, causing contraceptive failure.", "Valproic Acid: Alters drug levels."]
}

drug_food_interactions = {
    "Ibuprofen": ["Alcohol: Increases GI bleeding risk."],
    "Aspirin": ["Alcohol: Severe gastric mucosal damage."],
    "Paracetamol": ["Alcohol: Depletes glutathione, massive hepatotoxicity risk."],
    "Terfenadine": ["Grapefruit Juice: CYP3A4 inhibition, fatal cardiotoxicity."],
    "Ciprofloxacin": ["Dairy (Calcium): Chelation reduces absorption."],
    "Diclofenac": ["Alcohol: Exacerbates hepatic and GI toxicity."],
    "Amlodipine": ["Grapefruit Juice: Mildly increases systemic exposure."],
    "Omeprazole": ["St. John's Wort: Induces metabolism, reducing efficacy."],
    "Losartan": ["Potassium-rich foods (Bananas/Salt substitutes): Risk of hyperkalemia."],
    "Albuterol": ["Caffeine: Additive CNS/cardiac stimulation."],
    "Sertraline": ["Grapefruit Juice: May slightly increase serum concentrations."],
    "Furosemide": ["Licorice: Increases risk of severe hypokalemia."],
    "Diazepam": ["Alcohol: Synergistic CNS depression, potentially fatal."],
    "Fluoxetine": ["Alcohol: Enhances cognitive and motor impairment."],
    "Lisinopril": ["Potassium-rich foods: Can lead to fatal hyperkalemia."],
    "Clopidogrel": ["Grapefruit Juice: May reduce metabolic activation of the prodrug."],
    "Tamsulosin": ["Food: Taking on an empty stomach drastically alters peak concentrations."],
    "Levothyroxine": ["Soy/Espresso: Significantly decreases GI absorption."],
    "Rosuvastatin": ["Red Yeast Rice: Additive myopathy/rhabdomyolysis risk."],
    "Pantoprazole": ["Food: Must be taken 30 mins before meals for maximum efficacy."],
    "Carbamazepine": ["Grapefruit Juice: Increases bioavailability and toxicity."]
}

# ==========================================
# 2. EXPANDED FUNCTIONAL GROUPS
# ==========================================
smarts_patterns = {
    "Hydroxylation (-OH)": '[cH:1]>>[c:1](O)',
    "Methylation (-CH3)": '[cH:1]>>[c:1](C)',
    "Fluorination (-F)": '[cH:1]>>[c:1](F)',
    "Chlorination (-Cl)": '[cH:1]>>[c:1](Cl)',
    "Bromination (-Br)": '[cH:1]>>[c:1](Br)',
    "Iodination (-I)": '[cH:1]>>[c:1](I)',
    "Amination (-NH2)": '[cH:1]>>[c:1](N)',
    "Nitration (-NO2)": '[cH:1]>>[c:1]([N+](=O)[O-])',
    "Methoxylation (-OCH3)": '[cH:1]>>[c:1](OC)',
    "Trifluoromethylation (-CF3)": '[cH:1]>>[c:1](C(F)(F)F)',
    "Carboxylation (-COOH)": '[cH:1]>>[c:1](C(=O)O)',
    "Acetylation (-NHCOCH3)": '[cH:1]>>[c:1](NC(=O)C)',
    "Sulfonation (-SO3H)": '[cH:1]>>[c:1](S(=O)(=O)O)',
    "Cyanation (-CN)": '[cH:1]>>[c:1](C#N)',
    "Ethylation (-CH2CH3)": '[cH:1]>>[c:1](CC)',
    "Thiolation (-SH)": '[cH:1]>>[c:1](S)',
    "Formylation (-CHO)": '[cH:1]>>[c:1](C=O)',
    "Isopropylation (-CH(CH3)2)": '[cH:1]>>[c:1](C(C)C)',
    "Tert-butylation (-C(CH3)3)": '[cH:1]>>[c:1](C(C)(C)C)',
    "Phenoxylation (-OPh)": '[cH:1]>>[c:1](Oc2ccccc2)'
}

mod_shifts = {
    "Hydroxylation (-OH)": ["⬆️ Increased water solubility; rapid renal clearance.", "⬇️ Reduced BBB penetration.", "⚠️ Target for Phase II Glucuronidation."],
    "Methylation (-CH3)": ["⬆️ Mild lipophilicity increase.", "🛡️ Steric shield against metabolic enzymes."],
    "Fluorination (-F)": ["🛡️ Blocks CYP450 metabolism.", "⬆️ Strong increase in lipophilicity/CNS penetration."],
    "Chlorination (-Cl)": ["⬆️ Significant lipophilicity increase.", "⚠️ Electron-withdrawing, alters pKa of nearby groups."],
    "Bromination (-Br)": ["⬆️ Extreme lipophilicity.", "⚠️ Potential for hepatotoxicity and prolonged half-life."],
    "Iodination (-I)": ["⬆️ Massive steric bulk.", "⚠️ Risk of thyroid pathway interference."],
    "Amination (-NH2)": ["⬇️ Increased water solubility.", "⚠️ Risk of forming toxic hydroxylamine reactive metabolites."],
    "Nitration (-NO2)": ["⬆️ Drastically alters electron distribution.", "⚠️ High association with mutagenicity and toxicity."],
    "Methoxylation (-OCH3)": ["⬆️ Increases lipophilicity.", "⚠️ Vulnerable to O-dealkylation by CYP enzymes."],
    "Trifluoromethylation (-CF3)": ["🛡️ Extreme metabolic stability.", "⬆️ Massive increase in lipophilicity and tissue retention."],
    "Carboxylation (-COOH)": ["⬇️ Plummets LogP.", "🛑 Introduces negative charge, effectively blocking BBB crossing."],
    "Acetylation (-NHCOCH3)": ["⬇️ Reduces amine basicity.", "🛡️ Often serves as a prodrug activation site."],
    "Sulfonation (-SO3H)": ["⬇️ Extreme water solubility.", "🛑 Complete exclusion from CNS; ultra-rapid renal excretion."],
    "Cyanation (-CN)": ["⬆️ Strong electron withdrawal.", "⚠️ Modifies target binding without releasing free cyanide in vivo."],
    "Ethylation (-CH2CH3)": ["⬆️ Adds moderate lipophilicity and steric bulk."],
    "Thiolation (-SH)": ["⚠️ Highly reactive, oxidizes easily.", "⚠️ Strong chelation with metalloenzymes."],
    "Formylation (-CHO)": ["⚠️ Highly reactive aldehyde.", "⚠️ Can form toxic Schiff bases with cellular proteins."],
    "Isopropylation (-CH(CH3)2)": ["⬆️ Major steric hindrance.", "⬆️ Prevents enzymatic degradation at site."],
    "Tert-butylation (-C(CH3)3)": ["🛡️ Ultimate steric block.", "⬆️ Very high lipophilicity and resistance to metabolism."],
    "Phenoxylation (-OPh)": ["⬆️ Adds pi-pi stacking interactions for target binding.", "⬆️ Large increase in molecular weight and lipophilicity."]
}

# ==========================================
# 3. STREAMLIT UI & LOGIC
# ==========================================
st.set_page_config(layout="wide", page_title="Predictive SAR & Clinical Profile")
st.title("💊 Predictive SAR & Clinical Profile Dashboard")
st.markdown("Modify structures to observe physicochemical shifts and review comprehensive clinical interaction profiles.")

# Sidebar Selection
st.sidebar.header("1. Select Base Molecule")
input_method = st.sidebar.radio("Choose Input Method:", ["Preset Molecule", "Custom SMILES"])

smiles_input = ""
selected_preset_name = None

if input_method == "Preset Molecule":
    selected_mol = st.sidebar.selectbox("Select a compound:", list(presets.keys()))
    smiles_input = presets[selected_mol]
    selected_preset_name = selected_mol
else:
    smiles_input = st.sidebar.text_input("Enter base SMILES string:", "CC(C)Cc1ccc(cc1)C(C)C(=O)O")

st.sidebar.header("2. Structural Modification")
st.sidebar.markdown("Simulate lead optimization by swapping functional groups.")

# Dynamic dropdown using the keys from the dictionary
modification_options = ["None"] + list(smarts_patterns.keys())
modification = st.sidebar.selectbox("Choose a functional group to add (Targets aromatic rings):", modification_options)

# Reaction Engine
def modify_structure(mol, mod_type):
    if mod_type == "None":
        return mol
    try:
        pattern = smarts_patterns[mod_type]
        rxn = AllChem.ReactionFromSmarts(pattern)
            
        products = rxn.RunReactants((mol,))
        if products:
            new_mol = products[0][0]
            Chem.SanitizeMol(new_mol) 
            return new_mol
    except Exception as e:
        st.sidebar.error("Modification failed for this specific structure (likely missing an available aromatic hydrogen).")
    return mol

# Molecule Processing
if smiles_input:
    base_mol = Chem.MolFromSmiles(smiles_input)
    
    if base_mol:
        final_mol = modify_structure(base_mol, modification)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Physicochemical Properties")
            
            # --- NEW: Display Brand and Generic Names cleanly ---
            if selected_preset_name:
                st.markdown(f"**Base Generic Name:** {selected_preset_name}")
                st.markdown(f"**Common Brands:** {brand_names.get(selected_preset_name, 'N/A')}")
                st.divider() # Creates a neat visual line before the live properties
                
            if modification != "None":
                st.success(f"Showing properties for modified structure: {modification}")
            else:
                st.info("Showing properties for base structure.")
                
            st.metric("Molecular Weight (g/mol)", f"{Descriptors.MolWt(final_mol):.2f}")
            st.metric("LogP (Lipophilicity)", f"{Descriptors.MolLogP(final_mol):.2f}")
            st.metric("TPSA (Polar Surface Area)", f"{Descriptors.TPSA(final_mol):.2f}")
            st.metric("H-Bond Donors", Descriptors.NumHDonors(final_mol))
            st.metric("H-Bond Acceptors", Descriptors.NumHAcceptors(final_mol))
        
        with col2:
            st.subheader("Interactive 3D Structure")
            mol_3d = Chem.AddHs(final_mol) 
            AllChem.EmbedMolecule(mol_3d, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol_3d) 
            
            mblock = Chem.MolToMolBlock(mol_3d)
            viewer = py3Dmol.view(width=600, height=400)
            viewer.addModel(mblock, 'mol')
            viewer.setStyle({'stick': {}})
            viewer.zoomTo()
            showmol(viewer, height=400, width=600)
            
        st.divider()
        st.subheader("🧬 Clinical Profile & Predicted Pharmacodynamics")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Predicted SAR Shifts", "Baseline Side Effects", "Drug-Drug Interactions", "Drug-Food Interactions"])
        
        with tab1:
            if modification == "None":
                st.info("Select a structural modification from the sidebar to view predicted shifts in toxicity and metabolism.")
            else:
                st.warning(f"**Predicted ADMET Shifts due to {modification}:**")
                for shift in mod_shifts[modification]:
                    st.markdown(shift)
                    
        with tab2:
            if selected_preset_name:
                for effect in base_side_effects[selected_preset_name]:
                    st.markdown(f"- {effect}")
            else:
                st.markdown("*Unknown baseline for custom SMILES structures.*")
                
        with tab3:
            if selected_preset_name:
                for ddi in drug_drug_interactions[selected_preset_name]:
                    st.markdown(f"- {ddi}")
            else:
                st.markdown("*Unknown interactions for custom SMILES structures.*")
                
        with tab4:
            if selected_preset_name:
                for dfi in drug_food_interactions[selected_preset_name]:
                    st.markdown(f"- {dfi}")
            else:
                st.markdown("*Unknown interactions for custom SMILES structures.*")
                    
    else:
        st.error("Invalid SMILES string. Please check your input.")