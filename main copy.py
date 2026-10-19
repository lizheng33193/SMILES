from flask import Flask, request, jsonify, send_file
import sys
import os
from scripts.build_vocab import WordVocab,Vocab
from scripts.inference import generate_smiles   
from rdkit import Chem                  
from rdkit.Chem import Draw
import io
import base64

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate_image():
    data = request.json
    lam_abs = float(data.get('lam_abs'))
    lam_em = float(data.get('lam_em'))
    eps = float(data.get('eps'))
    
    try:
        smiles = generate_smiles([lam_abs, lam_em, eps])
        print("✅ Predicted SMILES:", smiles)

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"无法解析 SMILES: {smiles}")

        # 300×300 像素，返回 PIL.Image.Image
        img = Draw.MolToImage(mol, size=(300, 300))
        
        # 将图片转换为base64编码
        img_byte_array = io.BytesIO()
        img.save(img_byte_array, format='PNG')
        img_str = base64.b64encode(img_byte_array.getvalue()).decode('utf-8')
        
        return jsonify({
            'smiles': smiles,
            'image': img_str
        })
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 400

if __name__ == "__main__":
    app.run(debug=True,port=5001)