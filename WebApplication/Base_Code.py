import pubchempy as pcp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
from rdkit import Chem
from rdkit.Chem import AllChem,Draw
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
import os

# Suppress warnings
warnings.filterwarnings("ignore")

os.environ["MISTRAL_API_KEY"] = ""

llm = LLM(
    model="mistral-small-latest",
    api_base="https://api.mistral.ai/v1",
    api_key=os.getenv("MISTRAL_API_KEY"),
    temperature=0.2
)
#tool = SerperDevTool()
#from transformers import RobertaTokenizer, TFRobertaForSequenceClassification
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')
#from transformers import TFRobertaForSequenceClassification, RobertaTokenizer
import streamlit as st
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, Lipinski, Draw, QED
import matplotlib.pyplot as plt
import math
from PIL import Image
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

def compare_fingerprints_sy(smiles, radius=6, nBits=150):
    morgan_gen = GetMorganGenerator(radius=radius, fpSize=nBits)
    mol = Chem.MolFromSmiles(smiles)
    fp = morgan_gen.GetFingerprint(mol)
    return fp.ToBitString()

def compare_fingerprints(smiles1, radius=6, nBits=150):
    # Create a MorganGenerator with the specified radius and bit size
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
    
    # Convert SMILES strings to RDKit molecules
    mol1 = Chem.MolFromSmiles(smiles1)
    # Generate Morgan fingerprints using MorganGenerator
    fp1 = morgan_gen.GetFingerprint(mol1)
    
    # Convert fingerprints to bit strings
    bit_str1 = fp1.ToBitString()
    
    # Create a new bit string based on similarity
    new_bit_string = bit_str1 
    
    return new_bit_string
class ClassificationModel(nn.Module):
    def __init__(self, input_dim=150, num_classes=3):
        super(ClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)  # No activation for logits
        return x
       
class CNNRegressionModel(nn.Module):
    def __init__(self, input_dim=150):
        super(CNNRegressionModel, self).__init__()
        # Assumes input is (batch_size, channels, input_dim), here channels=1
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.relu = nn.ReLU()

        # Fully connected layers
        self.fc1 = nn.Linear(64 * (input_dim // 8), 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
def bit_string_to_tensor(bit_string, n_bits=150):
    bit_list = [int(bit) for bit in bit_string]
    bit_tensor = torch.tensor(bit_list, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    return bit_tensor

# Define the function to predict using the trained model
def predict_y(model, smiles1, n_bits=150):
    bit_string = compare_fingerprints(smiles1, nBits=n_bits)
    bit_tensor = bit_string_to_tensor(bit_string, n_bits)
    
    model.eval()
    with torch.no_grad():
        prediction = model(bit_tensor)
    
    return prediction.item()

def predict_cyp(model, smiles1, n_bits):
    # Assuming `compare_fingerprints` is a function that returns a bit string of length n_bits
    bit_string = compare_fingerprints(smiles1, nBits=n_bits)
    bit_tensor = bit_string_to_tensor(bit_string, n_bits)
    
    model.eval()
    with torch.no_grad():
        output = model(bit_tensor)
        predicted_class = torch.argmax(output, dim=1).item()  # Get the index of the highest score
    
   
    
    return predicted_class

def bit_string_to_tensor_sy(bit_string1, bit_string2, n_bits=150):
    bit_list1 = [int(bit) for bit in bit_string1]
    bit_list2 = [int(bit) for bit in bit_string2]
    combined_bits = bit_list1 + bit_list2
    bit_tensor = torch.tensor(combined_bits, dtype=torch.float32).unsqueeze(0)
    return bit_tensor

def predict_sy(model, smiles1, smiles2, n_bits=150):
    bit_string1 = compare_fingerprints_sy(smiles1, nBits=n_bits)
    bit_string2 = compare_fingerprints_sy(smiles2, nBits=n_bits)
    bit_tensor = bit_string_to_tensor_sy(bit_string1, bit_string2, n_bits * 2)
    model_sy.eval()
    with torch.no_grad():
        prediction = model_sy(bit_tensor)
    return prediction.item()

model_sy= CNNRegressionModel(input_dim=300)
model = CNNRegressionModel(input_dim=150)
model_cyp = ClassificationModel(input_dim=150)
def lipinski_rule_of_five(mol):
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    h_donors = Lipinski.NumHDonors(mol)
    h_acceptors = Lipinski.NumHAcceptors(mol)
    violations = sum([mw > 500, logp > 5, h_donors > 5, h_acceptors > 10])
    return violations, violations <= 1

def water_solubility(mol):
    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    mw = Descriptors.MolWt(mol)
    logS = -0.74 * logp - 0.006 * mw + 0.003 * tpsa + 0.63
    return logS

def synthetic_accessibility(mol):
    return 1 - QED.qed(mol)  # Lower means easier synthesis

def bioavailability_score(mol):
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    rot_bonds = Lipinski.NumRotatableBonds(mol)
    return int(tpsa <= 140 and rot_bonds <= 10)

def draw_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        img = Draw.MolToImage(mol, size=(300, 300))
        return img
    return None

def predict_adme(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return {"Error": "Invalid SMILES"}

    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    h_donors = Lipinski.NumHDonors(mol)
    h_acceptors = Lipinski.NumHAcceptors(mol)
    rot_bonds = Lipinski.NumRotatableBonds(mol)

    lipinski_violations, lipinski_pass = lipinski_rule_of_five(mol)
    logS = water_solubility(mol)
    sa_score = synthetic_accessibility(mol)
    bio_score = bioavailability_score(mol)
    
    bbb_permeability = -0.3 < logp < 6 and mw < 450 and h_donors <= 3 and h_acceptors <= 7

    result = [
        ("Molecular Weight", mw, "Should be < 500 for good permeability"),
        ("logP", logp, "Measures hydrophobicity; affects absorption"),
        ("TPSA", tpsa, "Below 140 Å² favors permeability"),
        ("H-bond Donors", h_donors, "Should be ≤ 5 for drug-likeness"),
        ("H-bond Acceptors", h_acceptors, "Should be ≤ 10 for permeability"),
        ("Rotatable Bonds", rot_bonds, "Flexibility affects oral bioavailability"),
        ("Lipinski Violations", lipinski_violations, "≤ 1 violation preferred"),
        ("Lipinski Rule of Five Pass", lipinski_pass, "Indicates drug-likeness"),
        ("Water Solubility (LogS)", logS, "Lower LogS = better solubility"),
        ("Synthetic Accessibility", sa_score, "Lower value = easier synthesis"),
        ("Bioavailability Score", bio_score, "1 indicates good oral bioavailability"),
        ("BBB Permeability (Heuristic)", bbb_permeability, "Predicts CNS drug potential")
    ]
    return result


userchoice=["Name","SMILES"]
userchoice2=["Name","SMILES"]
userchoice3=["Name","SMILES"]
cell_line=["MCF7","7860","A549","DU145","HCT116","K562","OVCAR3","SNB75"]
st.set_page_config(page_title="DrugXplorer",page_icon=":pill")
st.title("DrugXplorer")
tab = st.sidebar.radio(
    "**Navigation**",
    ["🏠 Home", "🔬 ADME Analysis", "⚛️ Binding Affinity", "🧪 Drug Synergy"]
)
protein_groups = {
    "Hormone Receptors": ["PPARD","PPARG", "AR", "ESR1", "NR3C1"],
    "Kinases & Cell Signaling": ["ABL1", "JAK2", "AKT1", "MAPK1", "PLK1"],
    "Metabolism & Drug Processing": ["HMGCR", "PTGS2", "BACE1", "CYP3A4", "DPP4"],
    "Neurotransmitter & CNS Targets": ["ADRB1", "ADORA2A", "DRD2", "ACHE"],
    "Apoptosis & DNA Repair": ["CASP3", "PARP1", "ROCK1"],
    "Cancer Growth & Angiogenesis": ["EGFR", "KDR"]
}
if tab =="🏠 Home":
    st.subheader("Welcome to DrugXplorer, your AI-powered companion for drug discovery")
    st.markdown(
        """
        **DrugXplorer** is an advanced web application designed to streamline the drug discovery process by providing insights into key molecular properties.  
        Whether you're a researcher, chemist, or biotechnology enthusiast, **DrugXplorer** enables you to predict crucial pharmacokinetic properties,  
        assess drug-protein interactions, and analyze potential drug synergies—all in one platform.
        
        ### Features:
        🔬 **ADME Prediction** - Evaluate the Absorption, Distribution, Metabolism, and Excretion (ADME) properties of drug-like molecules.  
        🧬 **Binding Affinity Analysis** - Predict the binding strength between a drug molecule and various target proteins.  
        💊 **Drug Synergy Prediction** - Analyze potential synergistic effects between drug combinations.  
        
        Navigate through the different tabs to perform specific analyses:
        - **ADME Properties**: Input your molecule's name or SMILES representation and obtain detailed ADME predictions.
        - **Binding Affinity**: Select a target protein and provide a drug molecule to predict binding affinity values.
        - **Drug Synergy**: Explore drug pair interactions and their potential for combination therapy.
        
        **Harness the power of AI-driven drug discovery with DrugXplorer and accelerate your research with data-driven insights!** 🚀  
        """
    )

if tab =="🔬 ADME Analysis":
    st.subheader("ADME Analysis")
    st.markdown(
    """
    - ### **🔬Molecular Properties:**
        - **Molecular Weight (MW):** Measures the size of the molecule.  
        - **logP (Hydrophobicity):** Indicates lipid solubility; affects absorption.  
        - **Topological Polar Surface Area (TPSA):** Predicts permeability & solubility.  
        - **H-bond Donors:** Number of hydrogen bond donors in the molecule.  
        - **H-bond Acceptors:** Number of hydrogen bond acceptors in the molecule.  
        - **Rotatable Bonds:** Determines molecule flexibility; impacts oral bioavailability.
        - **Lipinski Violations:** Rules for drug-likeness (≤1 violation is preferred).
        - **Lipinski Rule of Five Pass:** Whether the molecule meets Lipinski’s criteria.    
        - **Water Solubility (LogS):** Predicts solubility; lower LogS = better solubility.  
        - **Synthetic Accessibility Score:** Estimates ease of synthesis (lower is better).
        - **Bioavailability Score:** Probability of good oral bioavailability.
        - **Blood-Brain Barrier (BBB) Permeability:** Predicts CNS drug potential.
    --- 
    - ### **💊Drug-Likeness & CYP Inhibition:**
        - **Lipinski Rule of Five Pass:** Evaluates overall drug-likeness.  
        - **Bioavailability Score:** Assesses potential for oral absorption.  
        - **Blood-Brain Barrier (BBB) Permeability:** Predicts CNS drug capability.
        - **CYP Inhibition:** Predicts whether molecule will inhibit CYP1A2, CYP2C9, CYP2C19, CYP2D6, and CYP34.
    --- 
    - ### **⚡How to Use This App**
        1. Enter drug name or its SMILES representation
        2. Click **Analyze** to get the ADME properties.
    """
    )
    user_ch=st.selectbox("Do you want to enter Name or SMILES representation",options=userchoice)
    if user_ch=="Name":
        drug_name= st.text_input("Enter molecule's name")
        def get_smiles(drug_name):
            try:
                compound = pcp.get_compounds(drug_name, 'name')[0]
                return compound.isomeric_smiles
            except:
                return 0
        smiles_ip=get_smiles(drug_name)
       
    elif user_ch=="SMILES":
        smiles_ip=st.text_input("Enter molecule's SMILES representation in capital letters")

    if st.button("Analyze"):
        if smiles_ip:
            result = predict_adme(smiles_ip)
            st.subheader("Molecular Structure")
            mol_img = draw_molecule(smiles_ip)
            if mol_img:
                st.image(mol_img, caption="Generated from SMILES", use_column_width=False)
            else:
                st.warning("Invalid SMILES provided. Please enter a valid SMILES string.")
            st.subheader("ADME Properties")
            df = pd.DataFrame(result, columns=["Property", "Value","Interpretation"])
            st.table(df)
            lipinski_pass = result[7][1]  # Extracts the boolean value (True/False)
            bioavailability_score = result[10][1]  # Extracts the score (1 or 0)
            bbb_permeability = result[11][1]  # Extracts the boolean value (True/False)

        # Additional Summary
       
            if lipinski_pass:
                st.success("✅ This molecule **passes** Lipinski's Rule of Five (drug-like).")
            else:
                st.warning("⚠️ This molecule **violates** Lipinski's Rule of Five.")

            if bioavailability_score:
                st.success("✅ This molecule **meets** Veber's bioavailability criteria.")
            else:
                st.warning("⚠️ This molecule **may have poor oral bioavailability**.")

            if bbb_permeability:
                st.success("✅ This molecule has **good potential** for Blood-Brain Barrier (BBB) permeability.")
            else:
                st.warning("⚠️ This molecule **may have limited** BBB permeability.")
            model_cyp.load_state_dict(torch.load('CYP1A2_model.pth'))
            model_cyp.eval()
            predicted_1A2 = predict_cyp(model_cyp,smiles_ip,150)
            model_cyp.load_state_dict(torch.load('CYP2C9_model.pth'))
            model_cyp.eval()
            predicted_2C9 = predict_cyp(model_cyp,smiles_ip,150)
            model_cyp.load_state_dict(torch.load('CYP2C19_model.pth'))
            model_cyp.eval()
            predicted_2C19 = predict_cyp(model_cyp,smiles_ip,150)
            model_cyp.load_state_dict(torch.load('CYP2D6_model.pth'))
            model_cyp.eval()
            predicted_2D6 = predict_cyp(model_cyp,smiles_ip,150)
            model_cyp.load_state_dict(torch.load('CYP3A4_model.pth'))
            model_cyp.eval()
            predicted_3A4 = predict_cyp(model_cyp,smiles_ip,150)
            if predicted_1A2 == 1:
                st.warning("⚠️ This molecule is an inhibitor of CYP1A2.")
            else:
                st.success("✅ This molecule is not an inhibitor of CYP1A2.")
            if predicted_2C9 == 1:
                st.warning("⚠️ This molecule is an inhibitor of CYP2C9.")
            else:
                st.success("✅ This molecule is not an inhibitor of CYP2C9.")
            if predicted_2C19 == 1:
                st.warning("⚠️ This molecule is an inhibitor of CYP2C19.")
            else:
                st.success("✅ This molecule is not an inhibitor of CYP2C19.")
            if predicted_2D6 == 1:
                st.warning("⚠️ This molecule is an inhibitor of CYP2D6.")
            else:
                st.success("✅ This molecule is not an inhibitor of CYP2D6.")
            if predicted_3A4 == 1:
                st.warning("⚠️ This molecule is an inhibitor of CYP3A4.")
            else:
                st.success("✅ This molecule is not an inhibitor of CYP3A4.")
            

            



if tab=="⚛️ Binding Affinity":
    st.subheader("Binding Affinity")
    st.markdown("""

- ### **🧬Protein Groups and Their Functions**

    **Nuclear Receptors (Hormone-Responsive)**
    - **PPARD** – Regulates fatty acid metabolism.
    - **PPARG** – Controls glucose metabolism and adipogenesis.
    - **AR** – Androgen receptor, crucial in prostate cancer.
    - **ESR1** – Estrogen receptor, a major breast cancer target.
    - **NR3C1** – Glucocorticoid receptor, involved in inflammation.

    **Kinases & Cell Signaling**
    - **ABL1** – Involved in chronic myeloid leukemia.
    - **JAK2** – Regulates immune signaling.
    - **AKT1** – Controls cell survival and metabolism.
    - **MAPK1** – Key player in cell proliferation.
    - **PLK1** – Regulates mitosis and tumor progression.

    **Metabolism & Drug Processing**
    - **HMGCR** – Cholesterol biosynthesis enzyme.
    - **PTGS2** – Involved in inflammation (COX-2).
    - **BACE1** – Alzheimer’s disease target.
    - **CYP3A4** – Drug metabolism enzyme in the liver.
    - **DPP4** – Target for diabetes treatment.

    **Neurotransmitter & CNS Targets**
    - **ADRB1** – Regulates heart rate.
    - **ADORA2A** – Parkinson’s disease target.
    - **DRD2** – Dopamine receptor, key in schizophrenia.
    - **ACHE** – Alzheimer’s disease target.

    **Apoptosis & DNA Repair**
    - **CASP3** – Key executioner of apoptosis.
    - **PARP1** – DNA repair enzyme targeted in cancer therapy.
    - **ROCK1** – Regulates cell movement and apoptosis.

    **Cancer Growth & Angiogenesis**
    - **EGFR** – Lung and breast cancer target.
    - **KDR** – Controls new blood vessel formation (angiogenesis).

---

- ### **📊How to interpret results**
        
    **Binding Affinity (ΔG)**
    Binding affinity (**ΔG**, in kcal/mol) represents how strongly a drug binds to a target protein.

    | **Binding Affinity (ΔG)** | **Binding Strength** |
    |--------------------------|--------------------|
    | **ΔG ≤ -10 kcal/mol** | Very strong binding |
    | **-10 < ΔG ≤ -8 kcal/mol** | Strong binding |
    | **-8 < ΔG ≤ -6 kcal/mol** | Moderate binding |
    | **-6 < ΔG ≤ -4 kcal/mol** | Weak binding |
    | **ΔG > -4 kcal/mol** | Very weak/no binding |

    **Equilibrium Constant (K)**
    The equilibrium constant (**K** in µM) represents the ratio of bound and unbound states of a drug-protein interaction:

    | **K Value (µM)** | **Biological Interpretation** |
    |------------------|------------------------------------|
    | **< 0.001 µM** | Likely irreversible inhibition or very strong target modulation |
    | **0.001 – 0.1 µM** | Highly potent modulator, strong effect at low concentrations |
    | **0.1 – 1 µM** | Effective modulation, commonly seen in drug candidates |
    | **1 – 10 µM** | Moderate effect, may require optimization for potency |
    | **10 – 100 µM** | Weak modulation, may be non-specific or require high doses |
    | **> 100 µM** | Very weak or no modulation, likely not effective |

---                
- ### **⚡How to Use This App**
    1. Enter drug name or its SMILES representation
    2. Select a protein group from the dropdown menu.
    3. Choose a target protein within the selected group.
    4. Click **Predict** to get the predicted interaction strength.

""")

    user_ch_2=st.selectbox("Do you want to enter Name or SMILES representation of a molecule",options=userchoice2)
    if user_ch_2=="Name":
        drug_name_2= st.text_input("Enter name of the molecule")
        def get_smiles(drug_name_2):
            try:
                compound = pcp.get_compounds(drug_name_2, 'name')[0]
                return compound.isomeric_smiles
            except:
                return 0
        smiles_ip_2=get_smiles(drug_name_2)
       
    elif user_ch_2=="SMILES":
        smiles_ip_2=st.text_input("Enter molecule's SMILES representation in capital")
    
    if smiles_ip_2:
        group = st.selectbox("Select Protein Group", list(protein_groups.keys()))
        if group:
            protein = st.selectbox("Select Target Protein", protein_groups[group])

    if smiles_ip_2:
        if st.button("Predict"):
            model.load_state_dict(torch.load(f"{protein}_MODEL_BEST.pth"))
            model.eval()
            predicted_y = predict_y(model, smiles_ip_2)
            integer_value = round(predicted_y,2)
            del_g=str(integer_value)
            K=integer_value*4184
            K=K/(298*8.314)
            K=np.exp(K)
            K=K * 10**6
            K=round(K,4)
            eqb=str(K)
            L=" µM"
            cal= " kcal/mol"
            del_g=del_g+cal
            eqb=eqb+L
            st.write("Binding affinity of your drug molecule with selected protein is")
            st.write(del_g)
            st.write("Amount of drug in micromolar needed to modulate selected protein is")
            st.write(eqb)
            with st.spinner("Processing... Please wait."):
                inferencer = Agent(
                    role="Drug Discovery Scientist",
                    goal="From a given output of a prediction or experiment, provide valuable insights and inferences and find supporting studies relating to the experiment",
                    backstory = "You give inferences and context from a given molecular property prediction",
                    verbose = True,
                    llm = llm,
                    #tool = [tool]
                )
                give_insight = Task(
                    description="""Write a concise paragraph to infer the following result: A Molecule with SMILES: {smiles_ip_2} when docked with a protein {protein} is predicted to generate a binding affinity of {del_g} and an IC50 value of {eqb}""",
                    expected_output="A well-written, fact-based paragraph from credible resources",
                    agent=inferencer,
                )
                crew = Crew(
                    agents=[inferencer],
                    tasks=[give_insight],
                    verbose=False
                )
                result = crew.kickoff(inputs={"smiles_ip_2":smiles_ip_2 , "protein": protein,"del_g": del_g, "eqb":eqb})
                st.subheader("Inference")
                st.markdown(give_insight.output)

if tab=="🧪 Drug Synergy":
    st.subheader("Drug Synergy Prediction")
    st.markdown("""  

- ### **🧪Cell Lines**  

    **Cell Lines & Cancer Types**  
    - **MCF7** – Breast cancer (ER⁺, Luminal A).  
    - **A549** – Lung adenocarcinoma (NSCLC).  
    - **HCT116** – Colorectal carcinoma.  
    - **DU145** – Prostate cancer (androgen-independent).  
    - **K562** – Chronic myelogenous leukemia (CML).  
    - **OVCAR3** – Ovarian adenocarcinoma.  
    - **SNB75** – Glioblastoma (brain tumor).   
    - **786-O** – Renal cell carcinoma (RCC, kidney cancer).  

---  

- ### **📊 How to Interpret Bliss Synergy Score**  

    The **Bliss Synergy Score** quantifies the interaction between two drugs compared to their expected independent effects.  

    | **Bliss Synergy Score** | **Interpretation** |  
    |-------------------------|--------------------|  
    | **> 1**  | Strong Synergy – Drugs work significantly better together than expected. |  
    | **0.1 to 1**  | Moderate Synergy – Some benefit from combination but not extreme. |  
    | **-0.1 to 0.1**  | Additive – Drugs work as expected without interaction. |  
    | **< -1**  | Antagonism – Drug combination reduces effectiveness. |  

---  

- ### **⚡ How to Use This App**  

    1. Select two drugs from the input list or enter SMILES.  
    2. Choose a cancer cell line from the dropdown menu.  
    3. Click **Predict** to calculate the **Bliss Synergy Score**.    

""")
    user_ch_3=st.selectbox("Do you want to enter the Name or SMILES representation",options=userchoice3)
    if user_ch_3=="Name":
        drug_name_sy1= st.text_input("Enter name of the first molecule")
        drug_name_sy2=st.text_input("Enter name of the second molecule")
        def get_smiles(drug_name):
            try:
                compound = pcp.get_compounds(drug_name, 'name')[0]
                return compound.isomeric_smiles
            except:
                return 0
        smiles_sy_1=get_smiles(drug_name_sy1)
        smiles_sy_2=get_smiles(drug_name_sy2)
       
    elif user_ch_3=="SMILES":
        smiles_sy_1=st.text_input("Enter first molecule's SMILES representation in capital letters")
        smiles_sy_2=st.text_input("Enter second molecule's SMILES representation in capital letters")
        
    if smiles_sy_1 and smiles_sy_2:
        cell=st.selectbox("Choose your desired cell line",options=cell_line)
    if smiles_sy_1 and smiles_sy_2:
        if st.button("Predict Synergy"):
            model_sy.load_state_dict(torch.load(f"{cell}_MODEL.pth"))
            model_sy.eval()
            predicted_sy = predict_sy(model_sy, smiles_sy_1, smiles_sy_2)
            synergy_value = round(predicted_sy,2)
            bliss_score=str(synergy_value)
            st.write("Bliss score of the two molecules with desired cell line is")
            st.write(bliss_score)
            with st.spinner("Processing... Please wait."):
                inferencer = Agent(
                    role="Drug Discovery Scientist",
                    goal="From a given output of a prediction or experiment, provide valuable insights and inferences and find supporting studies relating to the experiment",
                    backstory = "You give inferences and context from a given molecular property prediction",
                    verbose = True,
                    llm = llm,
                    #tool = [tool]
                )
                give_insight = Task(
                    description="""Write a concise paragraph to infer the following result: 
                                The Bliss Drug Synergy score for two drugs of SMILES {d1} and {d2} 
                                on a cell line {cl} is predicted to be {value}
                                A bliss score less than 0 is considered antagonistic, near 0 is considered additive
                                and a score higher than 0 is considered synergistic.""",
                    expected_output="A well-written, fact-based paragraph from credible resources",
                    agent=inferencer,
                )
                crew = Crew(
                    agents=[inferencer],
                    tasks=[give_insight],
                    verbose=False
                )
                result = crew.kickoff(inputs={"d1":smiles_sy_1 , "d2": smiles_sy_2,"cl": cell, "value":bliss_score})
                st.subheader("Inference")
                st.markdown(give_insight.output)



        
    
