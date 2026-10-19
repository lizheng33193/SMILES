from flask import Flask, request, render_template, jsonify
from scripts.build_vocab import WordVocab,Vocab
from scripts.inference import generate_smiles
from rdkit import Chem
from rdkit.Chem import Draw
import io, base64

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    image_data = None
    smiles = None

    if request.method == 'POST':
        # 1. 接收并验证整数输入
        try:
            lam_abs = int(request.form['lam_abs'])
            lam_em  = int(request.form['lam_em'])
            eps     = int(request.form['eps'])
        except (KeyError, ValueError):
            # 如果缺失参数或无法转为 int，则返回带错误提示的页面
            return render_template('index.html',
                                   error="请输入合法的整数",
                                   lam_abs=request.form.get('lam_abs',''),
                                   lam_em=request.form.get('lam_em',''),
                                   eps=request.form.get('eps',''))

        # 2. 模型推理，得到 SMILES
        try:
            smiles = generate_smiles([lam_abs, lam_em, eps])
        except Exception as e:
            return render_template('index.html',
                                   error=f"推理出错：{e}",
                                   lam_abs=lam_abs, lam_em=lam_em, eps=eps)

        # 3. RDKit 渲染分子图
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return render_template('index.html',
                                   error=f"无法解析 SMILES：{smiles}",
                                   lam_abs=lam_abs, lam_em=lam_em, eps=eps)
        img = Draw.MolToImage(mol, size=(300, 300))

        # 4. 转 base64
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        image_data = base64.b64encode(buf.getvalue()).decode('utf-8')

    # 渲染表单（GET 或 POST 完成后都渲染同一模板）
    return render_template('index.html',
                           image_data=image_data,
                           smiles=smiles,
                           lam_abs=request.form.get('lam_abs',''),
                           lam_em=request.form.get('lam_em',''),
                           eps=request.form.get('eps',''))

if __name__ == '__main__':
    app.run(debug=True, port=5001)
