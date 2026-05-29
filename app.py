"""
app.py — Breast Cancer Diagnosis Web Application
Biotech ML Final Project — Deployment Step (Step 6)

Run:
    pip install flask joblib scikit-learn numpy pandas
    python app.py
Then open: http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import os

app = Flask(__name__)

# ── Load saved artifacts ──────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

model = joblib.load(os.path.join(MODEL_DIR, 'best_model.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
selected_features = joblib.load(os.path.join(MODEL_DIR, 'selected_features.pkl'))
le_hospital = joblib.load(os.path.join(MODEL_DIR, 'le_hospital.pkl'))

HOSPITAL_CLASSES = list(le_hospital.classes_)

# ── HTML Template (single-file app) ──────────────────────────────────────────
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>🧬 Breast Cancer Diagnosis Predictor</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Segoe UI', sans-serif;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      min-height: 100vh; padding: 30px 15px;
    }
    .container {
      max-width: 820px; margin: 0 auto;
      background: white; border-radius: 20px;
      box-shadow: 0 20px 60px rgba(0,0,0,0.25); overflow: hidden;
    }
    .header {
      background: linear-gradient(135deg, #1a1a2e, #16213e);
      color: white; padding: 35px 40px; text-align: center;
    }
    .header h1 { font-size: 2rem; margin-bottom: 8px; }
    .header p  { color: #a0aec0; font-size: 0.95rem; }
    .badge {
      display: inline-block; background: #48bb78;
      color: white; padding: 4px 12px; border-radius: 20px;
      font-size: 0.78rem; margin-top: 10px; font-weight: 600;
    }
    .form-body { padding: 35px 40px; }
    .section-title {
      font-size: 0.75rem; font-weight: 700; letter-spacing: 1.5px;
      text-transform: uppercase; color: #718096; margin: 25px 0 14px;
      padding-bottom: 6px; border-bottom: 2px solid #e2e8f0;
    }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .field label {
      display: block; font-size: 0.82rem; font-weight: 600;
      color: #4a5568; margin-bottom: 5px;
    }
    .field input, .field select {
      width: 100%; padding: 10px 13px; border: 1.5px solid #e2e8f0;
      border-radius: 8px; font-size: 0.92rem; transition: border 0.2s;
      background: #f7fafc;
    }
    .field input:focus, .field select:focus {
      outline: none; border-color: #667eea; background: white;
    }
    .hint { font-size: 0.72rem; color: #a0aec0; margin-top: 3px; }
    .btn {
      width: 100%; margin-top: 28px; padding: 15px;
      background: linear-gradient(135deg, #667eea, #764ba2);
      color: white; border: none; border-radius: 10px;
      font-size: 1rem; font-weight: 700; cursor: pointer;
      letter-spacing: 0.5px; transition: opacity 0.2s;
    }
    .btn:hover { opacity: 0.9; }
    .result-box {
      margin-top: 28px; padding: 25px; border-radius: 12px;
      text-align: center; display: none;
    }
    .result-malignant { background: #fff5f5; border: 2px solid #fc8181; }
    .result-benign    { background: #f0fff4; border: 2px solid #68d391; }
    .result-label { font-size: 1.6rem; font-weight: 800; margin-bottom: 6px; }
    .result-malignant .result-label { color: #c53030; }
    .result-benign    .result-label { color: #276749; }
    .prob-bar-wrap { margin: 14px auto; max-width: 360px; }
    .prob-bar-bg { background: #e2e8f0; border-radius: 20px; height: 12px; overflow: hidden; }
    .prob-bar-fill { height: 100%; border-radius: 20px; transition: width 0.6s ease; }
    .prob-label { font-size: 0.85rem; color: #718096; margin-top: 6px; }
    .disclaimer {
      margin-top: 12px; font-size: 0.78rem; color: #a0aec0;
      font-style: italic;
    }
    .footer {
      background: #f7fafc; padding: 18px 40px;
      text-align: center; font-size: 0.8rem; color: #a0aec0;
      border-top: 1px solid #e2e8f0;
    }
    @media (max-width: 560px) { .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>🧬 Breast Cancer Diagnosis</h1>
    <p>ML-powered classification using Random Forest — Biotech Final Project</p>
    <span class="badge">✅ Model AUC-ROC: 0.850</span>
  </div>

  <div class="form-body">
    <p class="section-title">Patient & Morphological Features</p>
    <div class="grid">
      <div class="field">
        <label>Radius Mean (mm)</label>
        <input type="number" id="radius_mean" step="0.01" value="14.5" />
        <div class="hint">Mean of distances from center to perimeter</div>
      </div>
      <div class="field">
        <label>Texture Mean</label>
        <input type="number" id="texture_mean" step="0.01" value="19.0" />
        <div class="hint">Standard deviation of gray-scale values</div>
      </div>
      <div class="field">
        <label>Perimeter Mean (mm)</label>
        <input type="number" id="perimeter_mean" step="0.1" value="92.0" />
      </div>
      <div class="field">
        <label>Area Mean (mm²)</label>
        <input type="number" id="area_mean" step="1" value="655" />
      </div>
      <div class="field">
        <label>Smoothness Mean</label>
        <input type="number" id="smoothness_mean" step="0.001" value="0.096" />
      </div>
      <div class="field">
        <label>Compactness Mean</label>
        <input type="number" id="compactness_mean" step="0.001" value="0.104" />
      </div>
      <div class="field">
        <label>Concavity Mean</label>
        <input type="number" id="concavity_mean" step="0.001" value="0.089" />
      </div>
      <div class="field">
        <label>Fractal Dimension Mean</label>
        <input type="number" id="fractal_dimension_mean" step="0.001" value="0.063" />
      </div>
    </div>

    <p class="section-title">Standard Error Features</p>
    <div class="grid">
      <div class="field">
        <label>Perimeter SE</label>
        <input type="number" id="perimeter_se" step="0.01" value="2.87" />
      </div>
      <div class="field">
        <label>Concavity SE</label>
        <input type="number" id="concavity_se" step="0.001" value="0.032" />
      </div>
    </div>

    <p class="section-title">Patient Information</p>
    <div class="grid">
      <div class="field">
        <label>Patient Age</label>
        <input type="number" id="age" min="18" max="100" value="52" />
      </div>
      <div class="field">
        <label>Hospital</label>
        <select id="hospital">
          {% for h in hospitals %}
          <option value="{{ h }}">{{ h }}</option>
          {% endfor %}
        </select>
      </div>
    </div>

    <button class="btn" onclick="predict()">🔍 Predict Diagnosis</button>

    <div class="result-box" id="result-box">
      <div class="result-label" id="result-label"></div>
      <div class="prob-bar-wrap">
        <div class="prob-bar-bg">
          <div class="prob-bar-fill" id="prob-bar" style="width:0%"></div>
        </div>
        <div class="prob-label" id="prob-label"></div>
      </div>
      <div class="disclaimer">
        ⚠️ This tool is for educational/research purposes only.<br>
        Always consult a qualified medical professional for diagnosis.
      </div>
    </div>
  </div>

  <div class="footer">
    Biotech ML Final Project · Haneen Mohamed Ismail · ID: 221000582<br>
    Random Forest Classifier · Precision: 0.776 · Recall: 0.846 · F1: 0.810
  </div>
</div>

<script>
async function predict() {
  const fields = [
    'radius_mean','texture_mean','perimeter_mean','area_mean',
    'smoothness_mean','compactness_mean','concavity_mean',
    'fractal_dimension_mean','perimeter_se','concavity_se','age'
  ];

  const data = { hospital: document.getElementById('hospital').value };
  for (const f of fields) {
    const val = parseFloat(document.getElementById(f).value);
    if (isNaN(val)) { alert('Please fill all numeric fields.'); return; }
    data[f] = val;
  }

  const btn = document.querySelector('.btn');
  btn.textContent = '⏳ Predicting...';
  btn.disabled = true;

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });
    const result = await res.json();
    showResult(result);
  } catch(e) {
    alert('Prediction failed. Make sure the Flask server is running.');
  } finally {
    btn.textContent = '🔍 Predict Diagnosis';
    btn.disabled = false;
  }
}

function showResult(r) {
  const box   = document.getElementById('result-box');
  const label = document.getElementById('result-label');
  const bar   = document.getElementById('prob-bar');
  const prob  = document.getElementById('prob-label');

  const isMalignant = r.prediction === 1;
  const pct = Math.round(r.probability_malignant * 100);

  box.className = 'result-box ' + (isMalignant ? 'result-malignant' : 'result-benign');
  label.textContent = isMalignant ? '🔴 Malignant' : '🟢 Benign';
  bar.style.width = pct + '%';
  bar.style.background = isMalignant ? '#fc8181' : '#68d391';
  prob.textContent = `Malignancy probability: ${pct}%`;
  box.style.display = 'block';
  box.scrollIntoView({ behavior: 'smooth' });
}
</script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, hospitals=HOSPITAL_CLASSES)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        # Encode hospital
        hospital_name = data.get('hospital', HOSPITAL_CLASSES[0])
        if hospital_name in HOSPITAL_CLASSES:
            hospital_encoded = le_hospital.transform([hospital_name])[0]
        else:
            hospital_encoded = 0

        # Build full feature dict with engineered features
        radius_mean = float(data['radius_mean'])
        area_mean = float(data['area_mean'])
        compactness_mean = float(data['compactness_mean'])
        fractal_dimension_mean = float(data['fractal_dimension_mean'])
        age = float(data['age'])

        compactness_ratio = compactness_mean / (fractal_dimension_mean + 1e-8)
        area_radius_ratio = area_mean / (radius_mean + 1e-8)

        if age <= 40:
            age_group_encoded = 0
        elif age <= 55:
            age_group_encoded = 1
        elif age <= 70:
            age_group_encoded = 2
        else:
            age_group_encoded = 3

        all_features = {
            'radius_mean': radius_mean,
            'texture_mean': float(data['texture_mean']),
            'perimeter_mean': float(data['perimeter_mean']),
            'area_mean': area_mean,
            'smoothness_mean': float(data['smoothness_mean']),
            'compactness_mean': compactness_mean,
            'concavity_mean': float(data['concavity_mean']),
            'fractal_dimension_mean': fractal_dimension_mean,
            'perimeter_se': float(data['perimeter_se']),
            'concavity_se': float(data['concavity_se']),
            'age': age,
            'hospital_encoded': hospital_encoded,
            'compactness_ratio': compactness_ratio,
            'area_radius_ratio': area_radius_ratio,
            'age_group_encoded': age_group_encoded,
        }

        # Extract only selected features in correct order
        feature_vector = np.array([[all_features[f] for f in selected_features]])
        feature_scaled = scaler.transform(feature_vector)

        prediction = int(model.predict(feature_scaled)[0])
        prob = model.predict_proba(feature_scaled)[0]

        return jsonify({
            'prediction': prediction,
            'diagnosis': 'Malignant' if prediction == 1 else 'Benign',
            'probability_malignant': round(float(prob[1]), 4),
            'probability_benign': round(float(prob[0]), 4),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    print("🧬 Breast Cancer Diagnosis App — Starting...")
    print("   Open: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
