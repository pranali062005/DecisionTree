from flask import Flask, render_template_string, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# Automatically detect whether you are using logistic.pkl or DecisionTree.pkl
MODEL_PATH = "logistic.pkl" if os.path.exists("logistic.pkl") else "DecisionTree.pkl"

model = None
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Warning: Model file not found at {MODEL_PATH}")

# Embedded HTML Template with responsive Bootstrap styling
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Placement Prediction Portal</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    
    <style>
        :root {
            --primary-gradient: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            --card-bg: rgba(255, 255, 255, 0.96);
        }

        body {
            background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            padding: 40px 15px;
            color: #2d3748;
        }

        .main-card {
            border: none;
            border-radius: 24px;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            overflow: hidden;
            max-width: 950px;
            width: 100%;
        }

        .header-banner {
            background: var(--primary-gradient);
            color: #ffffff;
            padding: 35px 25px;
            text-align: center;
        }

        .header-banner h2 {
            font-weight: 700;
            letter-spacing: -0.5px;
            margin-bottom: 8px;
        }

        .header-banner p {
            font-size: 0.95rem;
            opacity: 0.85;
            margin: 0;
        }

        .form-section-title {
            font-size: 0.85rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #1e3c72;
            margin-bottom: 15px;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 6px;
        }

        .form-label {
            font-size: 0.82rem;
            font-weight: 600;
            color: #4a5568;
            margin-bottom: 6px;
        }

        .form-control, .form-select {
            border-radius: 12px;
            padding: 10px 14px;
            border: 1px solid #cbd5e0;
            font-size: 0.9rem;
            transition: all 0.25s ease-in-out;
        }

        .form-control:focus, .form-select:focus {
            border-color: #2a5298;
            box-shadow: 0 0 0 3px rgba(42, 82, 152, 0.15);
        }

        .btn-predict {
            background: var(--primary-gradient);
            color: #ffffff;
            font-weight: 600;
            font-size: 1rem;
            padding: 14px;
            border-radius: 12px;
            border: none;
            letter-spacing: 0.5px;
            transition: all 0.3s ease;
            box-shadow: 0 8px 20px rgba(30, 60, 114, 0.3);
        }

        .btn-predict:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 25px rgba(30, 60, 114, 0.4);
            color: #ffffff;
        }

        .result-container {
            border-radius: 16px;
            padding: 20px;
            margin-top: 30px;
            text-align: center;
            font-weight: 600;
            font-size: 1.15rem;
            animation: slideUp 0.4s ease-out forwards;
        }

        .result-success {
            background: #d1fae5;
            color: #065f46;
            border: 1px solid #a7f3d0;
        }

        .result-danger {
            background: #fee2e2;
            color: #991b1b;
            border: 1px solid #fca5a5;
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(15px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>

<div class="main-card">
    <div class="header-banner">
        <h2><i class="fa-solid fa-graduation-cap me-2"></i>Campus Placement Predictor</h2>
        <p>Fill in student academic records to predict placement status</p>
    </div>

    <div class="p-4 p-md-5">
        <form action="/predict" method="POST">
            
            <div class="form-section-title"><i class="fa-solid fa-school me-2"></i>School Education</div>
            <div class="row g-3 mb-4">
                <div class="col-md-4">
                    <label class="form-label">Gender</label>
                    <select class="form-select" name="gender" required>
                        <option value="1">Male</option>
                        <option value="0">Female</option>
                    </select>
                </div>
                <div class="col-md-4">
                    <label class="form-label">10th Percentage (SSC %)</label>
                    <input type="number" step="0.01" min="0" max="100" class="form-control" name="ssc_p" placeholder="e.g. 67.00" required>
                </div>
                <div class="col-md-4">
                    <label class="form-label">10th Board (SSC)</label>
                    <select class="form-select" name="ssc_b" required>
                        <option value="1">Central</option>
                        <option value="0">Others</option>
                    </select>
                </div>
                <div class="col-md-4">
                    <label class="form-label">12th Percentage (HSC %)</label>
                    <input type="number" step="0.01" min="0" max="100" class="form-control" name="hsc_p" placeholder="e.g. 70.00" required>
                </div>
                <div class="col-md-4">
                    <label class="form-label">12th Board (HSC)</label>
                    <select class="form-select" name="hsc_b" required>
                        <option value="1">Central</option>
                        <option value="0">Others</option>
                    </select>
                </div>
                <div class="col-md-4">
                    <label class="form-label">12th Stream (HSC Stream)</label>
                    <select class="form-select" name="hsc_s" required>
                        <option value="2">Commerce</option>
                        <option value="1">Science</option>
                        <option value="0">Arts</option>
                    </select>
                </div>
            </div>

            <div class="form-section-title"><i class="fa-solid fa-user-graduate me-2"></i>Higher Education & Skill Testing</div>
            <div class="row g-3 mb-4">
                <div class="col-md-4">
                    <label class="form-label">Degree Percentage</label>
                    <input type="number" step="0.01" min="0" max="100" class="form-control" name="degree_p" placeholder="e.g. 65.00" required>
                </div>
                <div class="col-md-4">
                    <label class="form-label">Undergrad Degree Type</label>
                    <select class="form-select" name="degree_t" required>
                        <option value="2">Sci & Tech</option>
                        <option value="0">Comm & Mgmt</option>
                        <option value="1">Others</option>
                    </select>
                </div>
                <div class="col-md-4">
                    <label class="form-label">Work Experience</label>
                    <select class="form-select" name="workex" required>
                        <option value="1">Yes</option>
                        <option value="0">No</option>
                    </select>
                </div>
                <div class="col-md-4">
                    <label class="form-label">Employability Test Score %</label>
                    <input type="number" step="0.01" min="0" max="100" class="form-control" name="etest_p" placeholder="e.g. 55.00" required>
                </div>
                <div class="col-md-4">
                    <label class="form-label">MBA Specialization</label>
                    <select class="form-select" name="specialisation" required>
                        <option value="1">Mkt & Fin</option>
                        <option value="0">Mkt & HR</option>
                    </select>
                </div>
                <div class="col-md-4">
                    <label class="form-label">MBA Percentage</label>
                    <input type="number" step="0.01" min="0" max="100" class="form-control" name="mba_p" placeholder="e.g. 58.80" required>
                </div>
            </div>

            <div class="d-grid mt-4">
                <button type="submit" class="btn btn-predict text-uppercase">
                    <i class="fa-solid fa-chart-line me-2"></i>Predict Placement Status
                </button>
            </div>
        </form>

        {% if prediction_text %}
        <div class="result-container {% if 'Placed' in prediction_text and 'Not' not in prediction_text %}result-success{% else %}result-danger{% endif %}">
            {{ prediction_text | safe }}
        </div>
        {% endif %}
    </div>
</div>

</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template_string(
            HTML_TEMPLATE, 
            prediction_text="Error: Model file (.pkl) was not found on the server."
        )
    
    try:
        feature_values = [
            float(request.form["gender"]),
            float(request.form["ssc_p"]),
            float(request.form["ssc_b"]),
            float(request.form["hsc_p"]),
            float(request.form["hsc_b"]),
            float(request.form["hsc_s"]),
            float(request.form["degree_p"]),
            float(request.form["degree_t"]),
            float(request.form["workex"]),
            float(request.form["etest_p"]),
            float(request.form["specialisation"]),
            float(request.form["mba_p"])
        ]
        
        final_features = [np.array(feature_values)]
        prediction = model.predict(final_features)
        
        output = "Placed" if prediction[0] == 1 else "Not Placed"
        
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(final_features)[0][prediction[0]] * 100
            result_msg = f'<i class="fa-solid fa-circle-check me-2"></i>Prediction: Student is likely to be <strong>{output}</strong> (Confidence: {proba:.2f}%)'
        else:
            result_msg = f'<i class="fa-solid fa-circle-check me-2"></i>Prediction: Student is likely to be <strong>{output}</strong>'

    except Exception as e:
        result_msg = f"Error during prediction processing: {str(e)}"

    return render_template_string(HTML_TEMPLATE, prediction_text=result_msg)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)