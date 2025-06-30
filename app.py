from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from fpdf import FPDF
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import re

app = Flask(__name__)
app.secret_key = "startup_secret_key"

# Load trained ML components
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")  # To match input vector

# User store file
USERS_FILE = "users.json"
if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, "w") as f:
        json.dump({}, f)

def load_users():
    with open(USERS_FILE) as f:
        return json.load(f)

def save_user(email, password):
    users = load_users()
    users[email] = password
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

@app.route("/")
def index():
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    users = load_users()
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        if email in users and users[email] == password:
            session["user"] = email
            return redirect(url_for("dashboard"))
        else:
            return render_template("login.html", error="Invalid credentials!")
    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    users = load_users()
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        if email in users:
            return render_template("signup.html", error="Email already registered!")
        save_user(email, password)
        flash("Account created! Please log in.")
        return redirect(url_for("login"))
    return render_template("signup.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html", user=session["user"])

@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return redirect(url_for("login"))

    try:
        # Step 1: Input values
        funding = float(request.form["funding"])
        rounds = int(request.form["rounds"])
        participants = float(request.form["participants"])
        relationships = int(request.form["relationships"])
        age_ffy = int(request.form["age_ffy"])
        age_lfy = int(request.form["age_lfy"])
        age_fmy = int(request.form["age_fmy"])
        age_lmy = int(request.form["age_lmy"])

        funding_log = np.log1p(funding)
        participants_log = np.log1p(participants)

        input_vector = {}

        # Location
        location = request.form.get("location")
        for loc in ["CA", "NY", "MA", "TX", "otherstate"]:
            input_vector[f"is_{loc}"] = 1 if loc == location else 0

        # Domains
        selected_domains = request.form.getlist("domains")
        for domain in ["software", "web", "mobile", "enterprise", "advertising", "gamesvideo",
                       "ecommerce", "biotech", "consulting", "othercategory"]:
            input_vector[f"is_{domain}"] = 1 if domain in selected_domains else 0

        # Sources
        sources = request.form.getlist("sources")
        for src in ["VC", "angel", "roundA", "roundB", "roundC", "roundD"]:
            input_vector[f"has_{src}"] = 1 if src in sources else 0

        # Other features
        input_vector["is_top500"] = 1 if request.form.get("top500") == "on" else 0
        input_vector["funding_total_usd"] = funding_log
        input_vector["funding_rounds"] = rounds
        input_vector["avg_participants"] = participants_log
        input_vector["relationships"] = relationships
        input_vector["milestones"] = 4
        input_vector["age_first_funding_year"] = age_ffy
        input_vector["age_last_funding_year"] = age_lfy
        input_vector["age_first_milestone_year"] = age_fmy
        input_vector["age_last_milestone_year"] = age_lmy

        # Ensure all required features are filled
        for feature in features:
            if feature not in input_vector:
                input_vector[feature] = 0

        input_array = np.array([input_vector[f] for f in features]).reshape(1, -1)
        scaled_input = scaler.transform(input_array)
        success_prob = model.predict_proba(scaled_input)[0][1]

        # Chart save
        if not os.path.exists("static"):
            os.makedirs("static")
        plt.figure(figsize=(5, 3))
        plt.bar(["Failure", "Success"], [1 - success_prob, success_prob], color=["red", "green"])
        plt.ylabel("Probability")
        plt.title("Startup Success Prediction")
        plt.tight_layout()
        plt.savefig("static/prob_chart.png", transparent=True)
        plt.close()

        # Advice
        role = request.form.get("role")
        if success_prob > 0.75:
            advice = "🌟 Strong potential for success. Excellent opportunity!"
        elif success_prob > 0.5:
            advice = "⚠️ Moderate potential. Proceed with strategic planning."
        else:
            advice = "❌ Low success likelihood. Reconsider your approach."

        # Store result in session for download
        session["last_prob"] = round(float(success_prob) * 100, 2)
        session["last_advice"] = advice
        session["last_role"] = role


        return render_template("result.html",
                               prob=round(success_prob * 100, 2),
                               advice=advice,
                               role=role,
                               user=session["user"])
    except Exception as e:
        return f"❌ Error: {str(e)}"

@app.route("/download_pdf")
def download_pdf():
    if "user" not in session:
        return redirect(url_for("login"))

    from fpdf import FPDF
    import re

    prob = session.get("last_prob", 0)
    advice = session.get("last_advice", "No advice available.")
    user = session["user"]
    role = session.get("last_role", "Unknown")

    # Load metrics from file
    metrics_text = "No metrics available."
    if os.path.exists("static/metrics.txt"):
        with open("static/metrics.txt") as f:
            metrics_text = f.read()

    # Clean advice for latin1
    clean_advice = re.sub(r'[^\x00-\xFF]+', '', advice)

    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "Startup Success Prediction Report", ln=True, align='C')

    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, f"User: {user}", ln=True)
    pdf.cell(200, 10, f"Role: {role}", ln=True)
    pdf.cell(200, 10, f"Predicted Success Probability: {prob}%", ln=True)

    pdf.ln(5)
    pdf.multi_cell(0, 10, f"Advice: {clean_advice}")

    # Section: Model Metrics
    pdf.set_font("Arial", 'B', 14)
    pdf.ln(10)
    pdf.cell(200, 10, "Model Evaluation Metrics", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 8, metrics_text)

    # Section: Plots
    def add_image(path, title, w=180):
        if os.path.exists(path):
            pdf.set_font("Arial", 'B', 13)
            pdf.ln(5)
            pdf.cell(200, 10, title, ln=True)
            pdf.image(path, w=w)
            pdf.ln(5)

    add_image("static/prob_chart.png", "Prediction Probability Chart")
    add_image("static/confusion_matrix.png", "Confusion Matrix")
    add_image("static/univariate_funding_total_usd.png", "Univariate: Funding Total")
    add_image("static/univariate_funding_rounds.png", "Univariate: Funding Rounds")
    add_image("static/univariate_avg_participants.png", "Univariate: Avg Participants")
    add_image("static/multivariate_heatmap.png", "Multivariate: Feature Correlation Heatmap")

    file_path = "static/startup_result.pdf"
    pdf.output(file_path)

    return send_file(file_path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
