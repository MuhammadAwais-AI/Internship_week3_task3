# ⚡ Week 3 Task-1: PCA Dimensionality Reduction + FastAPI Energy Dashboard

## 🎯 Objective of the Task
This week is split into two parts to reflect a real ML engineering workflow:

**Part 1**: Apply PCA (Principal Component Analysis) to reduce input features while keeping model accuracy close to the Week 2 best model.  
**Part 2**: Deploy the best PCA model using FastAPI and build a simple web dashboard that displays key insights and allows real-time energy consumption predictions.

## 📦 Dataset & Model
- **Source**: The preprocessed and engineered dataset from Week 2
- **Model**: The best performing model selected after cross-validation in Week 2.  
  Save the full trained pipeline including the scaler and PCA if applicable using `joblib` so preprocessing is not lost.

## ✅ Part 1 — Dimensionality Reduction with PCA
### Requirements
- [ ] 1. Load the preprocessed dataset from Week 2 with the same features and encoding
- [ ] 2. Use the same train-test split as Week 2. Fit `StandardScaler` and `PCA` on the training set only, then transform both training and testing sets. Do not fit on full dataset to avoid data leakage
- [ ] 3. Apply PCA with `n_components` equal to total number of features to see all variance explained
- [ ] 4. Plot a scree plot (bar chart) showing explained variance ratio of each component
- [ ] 5. Plot cumulative explained variance curve with horizontal line at 95%. Identify how many components are needed to reach that threshold
- [ ] 6. Retrain the best model from Week 2 using only 3 PCA components as input features
- [ ] 7. Retrain again using the number of components that capture 95% of variance
- [ ] 8. Compare `RMSE` and `R-squared` across 3 versions: Original model, 3-component PCA model, 95% variance PCA model
- [ ] 9. Create a loadings heatmap with original feature names on y-axis and first 3 principal components on x-axis
- [ ] 10. Write a **Dimensionality Reduction Report** in markdown: Did accuracy drop significantly? How many features can be safely removed? Would you recommend PCA for a memory-constrained device and why?

## 🚀 Part 2 — FastAPI Dashboard
### Requirements
- [ ] 1. Install FastAPI and Uvicorn. Set up project structure: `main.py`, `templates/`, `static/`
- [ ] 2. Load the saved model pipeline inside FastAPI app using `joblib`
- [ ] 3. Use `Jinja2Templates` to render HTML pages and `StaticFiles` to serve charts
- [ ] 4. Create a home route `/` that renders a welcome page with navigation bar linking to Dashboard and Prediction pages
- [ ] 5. Create a `/dashboard` route displaying at least 3 visualizations from Week 2 EDA. Save plots as PNG in `static/` and display them
- [ ] 6. Create a `/predict` route that accepts a POST request. HTML form must include input field for every feature used by the final model and return predicted energy consumption on the same page
- [ ] 7. Run the app locally using `uvicorn main:app --reload` and confirm all routes work
- [ ] 8. HTML pages do not need heavy styling - clean and readable is enough

## 🛠️ Tools & Libraries
```python
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
fastapi
uvicorn
jinja2
python-multipart
```
## 📁 Project structure
.
├── week3.pca_analysis.ipynb   # Part 1: PCA + Model Retraining + Report
├── main.py                    # FastAPI app
├── templates/
│   ├── index.html             # Home page
│   ├── dashboard.html         # Dashboard with plots
│   └── predict.html           # Prediction form + result
├── static/
│   ├── energy_by_hour.png     # Saved EDA plots
│   ├── energy_by_loadtype.png
│   └── correlation_heatmap.png
├── model_pipeline.pkl         # Saved best model + scaler + PCA
├── requirements.txt
└── README.md
`
## 💻 How to run 
Clone this repository 
install dependencies
   pip install -r requirements.txt
`
## 🧑‍💻Author 
Muhammad Awais khan 
AI Engineer|📍 Peshawar, Pakistan
✉️ muhammadawaisaiengineer1@gmail.com