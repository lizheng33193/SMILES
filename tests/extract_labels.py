from docx import Document
import pandas as pd

# 读取 Word
doc = Document(r"D:\workspace\pycharm\smiles-transformer\data\model_v2_datasets\data_198\198.docx")
table = doc.tables[0]

# 跳过表头，提取 SMILES 与 λ_abs, λ_em, ε
rows = []
for row in table.rows[1:]:
    cells = [c.text.strip() for c in row.cells]
    # cells = [序号, SMILES, λ_abs, λ_em, ε]
    rows.append(cells[1:])  

df_labels = pd.DataFrame(rows, columns=["smiles","lam_abs","lam_em","eps"])
df_labels.to_csv(r"D:\workspace\pycharm\smiles-transformer\data\model_v2_datasets\data_198\data_label.csv", index=False)
print("✓ 已生成 D:\workspace\pycharm\smiles-transformer\data\model_v2_datasets\data_100\data_one.csv")
