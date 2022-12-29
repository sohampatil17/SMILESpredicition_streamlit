import numpy as np
import pandas as pd 
import streamlit as st
import pickle
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Descriptors

def AromaticProportion(m):
	aromatic_atoms = [m.GetAtomWithIdx(i).GetIsAromatic() for i in range(m.GetNumAtoms())]
	aa_count = []
	for i in aromatic_atoms:
		if i == True:
			aa_count.append(1)
	AromaticAtom = sum(aa_count)
	HeavyAtom = Descriptors.HeavyAtomCount(m)
	AR = AromaticAtom/HeavyAtom
	return AR


def generate(smiles, verbose = False):

	moldata = []
	for elem in smiles:
		mol=Chem.MolFromSmiles(elem)
		moldata.append(mol)

	baseData = np.arange(1,1)
	i = 0
	for mol in moldata:
		desc_MolLogP = Descriptors.MolLogP(mol)
		desc_MolWt = Descriptors.MolWt(mol)
		desc_NumRotatableBonds = Descriptors.NumRotatableBonds(mol)
		desc_AromaticProportion = AromaticProportion(mol)

		row = np.array([desc_MolLogP,
						desc_MolWt,
						desc_NumRotatableBonds,
						desc_AromaticProportion])

		if(i==0):
			baseData = row
		else: 
			baseData = np.vstack([baseData, row])
		i = i + 1

	columnNames=["MolLogP","MolWt","NumRotatableBonds","AroamticProperties"]
	descriptors = pd.DataFrame(data=baseData,columns=columnNames)

	return descriptors


st.write("""
# Molecular Solubility Prediction WebApp

This app can predict the solubility (logs)** values of molecules :)

Data obtained from the John S. Delaney. [ESOL:  Estimating Aqueous Solubility Directly from Molecular Structure](https://pubs.acs.org/doi/10.1021/ci034243x). ***J. Chem. Inf. Comput. Sci.*** 2004, 44, 3, 1000-1005.
***
""")

st.sidebar.header('User Input Features')
SMILES_input = "NCCCC\nCCC\nCN"

SMILES = st.sidebar.text_area("SMILES input", SMILES_input)
SMILES = "C\n" + SMILES
SMILES = SMILES.split('\n')

st.header('Input SMILES')
SMILES[1:]

st.header('Computed molecular descriptors')
X = generate(SMILES)
X[1:]

load_model = pickle.load(open('/Users/sohampradhneshpatil/Desktop/Random/Solubility_Model.pkl', 'rb'))

prediction = load_model.predict(X)

st.header('Predicted LogS values')
prediction[1:]




