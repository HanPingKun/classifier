import os
from flask import Flask, request, jsonify, send_file, render_template
import pandas as pd
import web_ft_bert_class_1k_tpl_50k_predict50k
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_smile():
    data = request.json
    smile = data.get('smile', '')
    if not smile:
        return jsonify({"error": "No SMILES string provided"}), 400
    result = web_ft_bert_class_1k_tpl_50k_predict50k.process_smiles(smile)
    return jsonify({
        "predicted_class_id": result[0],
        "rxn_str_id": result[1],
        "class_name": result[2]
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file and file.filename.endswith('.tsv'):
        out_dir = os.path.join(os.getcwd(), 'out', 'data', 'save')
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, file.filename)
        file.save(path)
        df = pd.read_csv(path, sep='\t')
        results_df = web_ft_bert_class_1k_tpl_50k_predict50k.process_smiles_batch(df)
        output_dir = os.path.join(os.getcwd(), 'out', 'data', 'output')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'results.csv')
        results_df.to_csv(output_path, index=False)
        return send_file(output_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
