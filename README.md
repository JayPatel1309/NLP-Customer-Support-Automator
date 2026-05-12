# 🎫 NLP Customer Support Automator

An end-to-end Python-based NLP pipeline designed to automate customer support ticket classification, priority prediction, and response generation. This project handles the entire lifecycle of a machine learning project, from synthetic data generation to business-ready reporting.

---

## 🚀 Key Features

* **Synthetic Data Generation:** Creates realistic support tickets across 6 distinct categories (Billing, Tech Support, Order Management, etc.) with custom augmentations, typos, and varying priorities.
* **Custom Text Preprocessing:** Includes contraction expansion, regex-based cleaning (handling URLs and emails), and a custom stemming algorithm to normalize ticket content.
* **Multi-Model Benchmarking:** Automatically trains and evaluates five different classifiers: Logistic Regression, Naive Bayes, Linear SVM, Random Forest, and Gradient Boosting.
* **Evaluation & Cross-Validation:** Implements 5-fold Stratified Cross-Validation and detailed metrics (Accuracy, Precision, Recall, and F1-Score) to ensure model robustness.
* **Explainable AI:** Extracts and visualizes the "Top Discriminative Terms" per category to show which keywords drive the model's decisions.
* **Business Intelligence:** Generates 8 distinct visualizations and a comprehensive text report detailing automation potential and projected response time improvements.
* **Live Chat Simulation:** Features an interactive CLI interface to test the winning pipeline in real-time with automated response templates.

---

## 🛠️ Technical Stack

* **Core Logic:** Python 3.x
* **Data Science:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn`
* **Visualization:** `matplotlib`, `seaborn`
* **NLP Tools:** `re` (Regex), Custom Stemming, TF-IDF Vectorization

---

## 📊 Pipeline Overview

1. **Data Acquisition:** Generates 120 samples per category or loads an existing CSV dataset.
2. **Preprocessing:** Normalizes text by removing noise, expanding contractions, and stripping stopwords.
3. **Feature Engineering:** Converts text into numerical vectors using TF-IDF with bigram support (`ngram_range=(1, 2)`).
4. **Model Training:** Executes a `Pipeline` for each model to ensure a clean data flow between vectorization and classification.
5. **Visualization:** Saves PNG files to an `Outputs/` directory, including confusion matrices and priority breakdowns.
6. **Reporting:** Generates a full `nlp_report.txt` containing key findings and strategic recommendations for automation.

---

## 📈 Analysis & Insights

The script provides deep dives into support operations, including:

* **Priority Distribution:** Visualizes the mix of Low, Medium, High, and Critical tickets across categories.
* **Response Time Impact:** Models a projected 20% reduction in response time through targeted automation.
* **Automated Recommendations:** Categorizes tickets based on "Automation Scores" (e.g., 95% for Product Info vs. 45% for Complaints).

---

## 🚦 Getting Started

### Prerequisites

Ensure you have the following libraries installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn

```

### Running the Pipeline

Simply execute the main script to run the full training and reporting suite:

```bash
python nlp_pipeline.py

```

### Interactive Mode

Once the report is generated, the script enters **Live Ticket Classification** mode. You can input custom messages to see how the model routes the ticket:

> **Customer:** "I keep getting logged out of my account after the new update."
> **Routing:** Technical Support
> **Auto-Reply:** "Thank you for reaching out. Please try: (1) Clear cache & cookies..."

---

## 📂 Output Structure

* `Outputs/customer_support_data.csv`: The generated/processed dataset.
* `Outputs/*.png`: Eight distinct visualization plots.
* `Outputs/nlp_report.txt`: Final executive summary and model breakdown.
