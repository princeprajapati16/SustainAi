# 🌱 AI-Powered Lifecycle Analyzer for Products

An intelligent tool designed to analyze the environmental impact of consumer products using lifecycle data. This project uses data science and machine learning to predict sustainability ratings, identify high-impact factors, and support green product development.

---

## 📌 Problem Statement

Sustainable product design is becoming essential, yet most companies lack data-driven tools to measure and reduce product lifecycle impact. This project aims to fill that gap by using AI to analyze key lifecycle indicators and predict product sustainability.

---

## 🎯 Project Goals

- Analyze lifecycle metrics like carbon footprint, water usage, energy consumption, etc.
- Predict sustainability ratings using regression modeling
- Identify features that contribute most to environmental impact
- Visualize insights for eco-conscious product design

---

## 🗂️ Dataset Overview

- **Rows:** 1500 products  
- **Columns:** Product name, material, energy use, carbon emissions, disposal method, etc.  
- **Engineered Features:**
  - `Impact_per_Year`
  - `Eco_Index`

Stored in: `data/AI_Lifecycle_Analyzer_Updated.csv`

---

## 🧪 Technologies Used

- **Python** (pandas, seaborn, matplotlib, scikit-learn)
- **Machine Learning:** Random Forest Regressor
- **Visualization:** Matplotlib, Seaborn
- **App Deployment:** Streamlit

---

## 🔍 Key Visualizations

- 📊 Distribution of Carbon Footprint
- 📦 Sustainability Rating by Product Type
- 📈 Average Footprint by Product
- 🔗 Weight vs. Emissions Scatter Plot
- 🧠 Correlation Heatmap
- 🥧 Disposal Method Distribution
- 🟢 Eco Index Ranking

---

## 🤖 Machine Learning Model

- **Model:** Random Forest Regressor
- **Target:** Sustainability Rating
- **R² Score:** ~0.85  
- **RMSE:** ~0.3  
- **Top Features:** Carbon Footprint, Waste Generated, Material, Weight

Model saved at: `models/sustainability_rf_model.pkl`

---

## 🚀 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/AI-Lifecycle-Analyzer.git
   cd AI-Lifecycle-Analyzer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

---

## 📦 Folder Structure

```
AI-Lifecycle-Analyzer/
│
├── data/                        # Input dataset
├── models/                      # Trained model
├── visuals/                     # Plots and images
├── sustainai.py                 # Main ML pipeline script
├── app.py                       # Streamlit app
├── generate_visuals.py          # Visualization generator
├── README.md                    # Project overview
└── requirements.txt             # Python dependencies
```

---
see this give me this format
