import sys
from rdkit import Chem
from scripts.inference import WordVocab,Vocab
from scripts.inference import generate_smiles
from rdkit.Chem import Draw





def props_to_image(lam_abs, lam_em, eps, out_png):
    smiles = generate_smiles([lam_abs, lam_em, eps])
    print("✅ Predicted SMILES:", smiles)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"无法解析 SMILES: {smiles}")
    img = Draw.MolToImage(mol, size=(300, 300))
    img.save(out_png)
    print(f"✅ Saved structure to {out_png}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python main.py lam_abs lam_em eps out.png")
        sys.exit(1)
    lam_abs, lam_em, eps = map(float, sys.argv[1:4])
    out_png = sys.argv[4]
    props_to_image(lam_abs, lam_em, eps, out_png)

#python main.py 722 754 109647 output.png