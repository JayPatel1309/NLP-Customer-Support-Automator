"""
=============================================================
NLP for Customer Support
=============================================================
Covers:
  1. Synthetic data generation
  2. Text preprocessing
  3. Feature engineering (TF-IDF)
  4. Multi-model training & comparison
  5. Evaluation (accuracy, precision, recall, F1, confusion matrix)
  6. Feature importance / top terms per category
  7. Automated-response recommendations
  8. Visualisations + text report saved to disk
=============================================================
"""

import re
import os
import random
import warnings
import textwrap
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

OUTPUT_DIR = "Outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


CATEGORIES = {
    "Billing & Payments": [
        "I was charged twice for the same order, please help",
        "My invoice shows an incorrect amount",
        "When will I receive my refund for the cancelled order?",
        "I can't update my payment method on the account",
        "Why was my credit card declined?",
        "I need a receipt for my recent purchase",
        "Can I get a refund for the subscription I didn't use?",
        "The discount code was not applied to my bill",
        "I am being charged after I cancelled my subscription",
        "My bank shows a charge I don't recognise from your company",
        "Please send me a copy of all invoices from last year",
        "I want to dispute a charge on my account",
        "The payment failed but money was deducted",
        "How do I update my billing address?",
        "I need to change my subscription plan",
    ],
    "Technical Support": [
        "The app keeps crashing when I try to log in",
        "I cannot access my account after the recent update",
        "Error 500 appears whenever I submit the form",
        "My password reset email never arrived",
        "The website is loading very slowly today",
        "I can't download the file I purchased",
        "Two-factor authentication is not working on my phone",
        "The mobile app is freezing on the home screen",
        "I lost access to my account after phone number changed",
        "Browser extension stopped working after update",
        "API integration is returning authentication errors",
        "The desktop application will not install on Windows 11",
        "My data is not syncing across devices",
        "Search feature returns no results",
        "I keep getting logged out automatically",
    ],
    "Order Management": [
        "Where is my order? It has been 10 days",
        "I want to cancel my order before it ships",
        "Can I change the delivery address for my order?",
        "I received the wrong item in my package",
        "My package arrived damaged",
        "The tracking number is not showing any updates",
        "I placed a duplicate order by mistake",
        "My order says delivered but I haven't received it",
        "How long does standard shipping take?",
        "Can I expedite the delivery of my order?",
        "I want to return the product I received",
        "Part of my order is missing from the package",
        "Can I combine two orders into one shipment?",
        "I haven't received a confirmation email for my order",
        "How do I track my international shipment?",
    ],
    "Account Management": [
        "I want to delete my account permanently",
        "How do I change my email address on my profile?",
        "I forgot my username and cannot recover it",
        "My account was hacked and I need help",
        "How do I enable or disable notifications?",
        "I want to update my personal information",
        "How do I add a secondary user to my account?",
        "My account was suspended without any reason",
        "I want to export all my data from your platform",
        "How do I link my social media accounts?",
        "Can I merge two accounts into one?",
        "I want to change my username",
        "My profile picture is not updating",
        "How do I set up parental controls?",
        "I need to verify my identity to unlock my account",
    ],
    "Product & Service Info": [
        "What features are included in the premium plan?",
        "Does your product support multiple languages?",
        "What is the difference between the basic and pro plans?",
        "Is there a free trial available?",
        "Do you offer discounts for annual subscriptions?",
        "What payment methods do you accept?",
        "Is my data encrypted and secure?",
        "Does the software work on Mac and Windows?",
        "How many users can share a single account?",
        "What is your cancellation and refund policy?",
        "Do you offer enterprise pricing for large teams?",
        "Is there a student discount available?",
        "What integrations do you support?",
        "Can I use the service offline?",
        "What is the storage limit on the basic plan?",
    ],
    "Feedback & Complaints": [
        "Your customer service representative was very rude",
        "The new interface is confusing and harder to use",
        "I am very disappointed with the product quality",
        "This is the third time I have had the same issue",
        "I have been waiting two weeks for a response",
        "Your service has gone downhill recently",
        "I want to file a formal complaint about my experience",
        "The product does not match its description at all",
        "I feel my privacy has been violated",
        "Your chatbot is completely useless and unhelpful",
        "I would like to speak with a manager about this",
        "This is unacceptable and I want compensation",
        "I am considering switching to a competitor",
        "Please pass my feedback to the product team",
        "I love the product but the checkout process needs work",
    ],
}

AUGMENTATIONS = [
    "Hello, I need help with the following: {}",
    "Hi support team, {}",
    "Good morning, I am writing because {}",
    "Urgent: {}",
    "I hope you can assist me. {}",
    "To whom it may concern, {}",
    "I've been trying to resolve this myself but {}",
    "I've been a customer for years and {}",
    "This is very frustrating: {}",
    "{}. Please respond as soon as possible.",
    "{}. Looking forward to your reply.",
    "{} Thank you in advance.",
    "{} This needs urgent attention.",
    "I would appreciate your help. {}",
]

TYPO_REPLACEMENTS = {
    "the": ["teh", "th"],
    "my": ["mmy", "m"],
    "your": ["youre", "ur"],
    "account": ["acount", "accnt"],
    "payment": ["paymnt", "payement"],
    "please": ["pleas", "pls"],
    "received": ["recieved", "recived"],
    "cancelled": ["canceled", "cancled"],
}


def inject_typos(text, rate=0.05):
    words = text.split()
    for i, word in enumerate(words):
        if random.random() < rate and word.lower() in TYPO_REPLACEMENTS:
            words[i] = random.choice(TYPO_REPLACEMENTS[word.lower()])
    return " ".join(words)


def generate_dataset(samples_per_category=120):
    records = []
    ticket_id = 1000

    for category, templates in CATEGORIES.items():
        for _ in range(samples_per_category):
            base = random.choice(templates)
            # Optionally wrap in an augmentation template
            if random.random() > 0.3:
                base = random.choice(AUGMENTATIONS).format(base)
            # Optionally inject typos
            if random.random() > 0.7:
                base = inject_typos(base)
            # Add varying punctuation / case
            if random.random() > 0.8:
                base = base.upper()
            elif random.random() > 0.9:
                base = base.lower()

            priority = random.choices(
                ["Low", "Medium", "High", "Critical"],
                weights=[0.3, 0.4, 0.2, 0.1]
            )[0]

            records.append({
                "ticket_id": f"TKT-{ticket_id}",
                "text": base,
                "category": category,
                "priority": priority,
                "response_time_hours": round(random.uniform(0.5, 72), 1),
                "resolved": random.choice([True, False]),
            })
            ticket_id += 1

    df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "he", "him", "his", "she", "her", "hers",
    "it", "its", "they", "them", "their", "what", "which", "who", "whom",
    "this", "that", "these", "those", "am", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall", "can",
    "a", "an", "the", "and", "but", "if", "or", "because", "as", "at",
    "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up",
    "down", "in", "out", "on", "off", "over", "under", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "same", "so", "than", "too", "very", "just", "of", "also",
    "hello", "hi", "dear", "regards", "sincerely", "please", "thank",
    "thanks", "hope", "write", "assist", "help", "team", "support",
    "customer", "service", "response", "forward", "reply",
}

CONTRACTIONS = {
    "can't": "cannot", "won't": "will not", "don't": "do not",
    "doesn't": "does not", "didn't": "did not", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "wouldn't": "would not", "couldn't": "could not", "shouldn't": "should not",
    "i'm": "i am", "i've": "i have", "i'll": "i will", "i'd": "i would",
    "it's": "it is", "he's": "he is", "she's": "she is", "that's": "that is",
    "there's": "there is", "they're": "they are", "they've": "they have",
    "we're": "we are", "we've": "we have", "you're": "you are",
    "you've": "you have", "let's": "let us",
}


def expand_contractions(text):
    for contraction, expansion in CONTRACTIONS.items():
        text = re.sub(r'\b' + re.escape(contraction) + r'\b', expansion, text)
    return text


def simple_stem(word):
    suffixes = ["ing", "tion", "ed", "er", "ly", "ness", "ment", "ful", "less", "ize", "ise", "ous", "al"]
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def preprocess_text(text, stem=True):
    """Full preprocessing pipeline."""
    text = str(text).lower()
    text = expand_contractions(text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'\b\d{10,}\b', ' NUM ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    if stem:
        tokens = [simple_stem(t) for t in tokens]
    return " ".join(tokens)



def build_pipelines():
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=15_000,
        sublinear_tf=True,
        min_df=2,
    )

    models = {
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=15000, sublinear_tf=True, min_df=2)),
            ("clf", LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")),
        ]),
        "Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=15000, min_df=2)),
            ("clf", MultinomialNB(alpha=0.1)),
        ]),
        "Linear SVM": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=15000, sublinear_tf=True, min_df=2)),
            ("clf", LinearSVC(max_iter=2000, C=1.0)),
        ]),
        "Random Forest": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 1), max_features=5000, min_df=2)),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        ]),
        "Gradient Boosting": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 1), max_features=5000, min_df=2)),
            ("clf", GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ]),
    }
    return models



def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    return {
        "Model": model_name,
        "Accuracy": round(acc * 100, 2),
        "Precision": round(prec * 100, 2),
        "Recall": round(rec * 100, 2),
        "F1-Score": round(f1 * 100, 2),
    }, y_pred


def cross_validate_model(model, X, y, cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
    return scores.mean() * 100, scores.std() * 100



def get_top_features_per_class(pipeline, categories, top_n=10):
    """Works with pipelines that have a TF-IDF step + a linear classifier."""
    vectorizer = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    feature_names = np.array(vectorizer.get_feature_names_out())
    top_features = {}

    if hasattr(clf, "coef_"):
        coef = clf.coef_
        for idx, cat in enumerate(categories):
            top_idx = np.argsort(coef[idx])[-top_n:][::-1]
            top_features[cat] = list(feature_names[top_idx])
    elif hasattr(clf, "feature_log_prob_"):
        probs = clf.feature_log_prob_
        for idx, cat in enumerate(categories):
            top_idx = np.argsort(probs[idx])[-top_n:][::-1]
            top_features[cat] = list(feature_names[top_idx])

    return top_features


PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
sns.set_theme(style="whitegrid", font_scale=1.05)


def plot_category_distribution(df, path):
    fig, ax = plt.subplots(figsize=(10, 5))
    counts = df["category"].value_counts()
    bars = ax.barh(counts.index, counts.values, color=PALETTE[:len(counts)])
    ax.set_xlabel("Number of Tickets", fontsize=12)
    ax.set_title("Customer Ticket Distribution by Category", fontsize=14, fontweight="bold")
    for bar, val in zip(bars, counts.values):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                f"{val}", va="center", fontsize=10)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_model_comparison(results_df, path):
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    x = np.arange(len(results_df))
    width = 0.18

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, results_df[metric], width, label=metric, color=PALETTE[i], alpha=0.88)

    ax.axhline(85, color="red", linestyle="--", linewidth=1.2, label="Target (85%)")
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(results_df["Model"], rotation=15, ha="right")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrix(y_test, y_pred, categories, path):
    cm = confusion_matrix(y_test, y_pred, labels=categories)
    short_labels = [c.split(" & ")[0].split(" ")[0] for c in categories]

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=short_labels, yticklabels=short_labels,
        ax=ax, linewidths=0.5
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix — Best Model", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_top_features(top_features, path):
    categories = list(top_features.keys())
    n = len(categories)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3.5))
    axes = axes.flatten()

    for i, (cat, terms) in enumerate(top_features.items()):
        ax = axes[i]
        y_pos = np.arange(len(terms))
        ax.barh(y_pos, range(len(terms), 0, -1),
                color=PALETTE[i % len(PALETTE)], alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(terms, fontsize=9)
        ax.set_title(cat, fontsize=10, fontweight="bold")
        ax.set_xlabel("Importance Rank", fontsize=8)
        ax.invert_yaxis()

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Top Discriminative Terms per Category", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_cross_val(cv_results, path):
    models = list(cv_results.keys())
    means = [cv_results[m]["mean"] for m in models]
    stds = [cv_results[m]["std"] for m in models]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(models, means, yerr=stds, color=PALETTE[:len(models)],
                  capsize=5, alpha=0.85, edgecolor="white")
    ax.axhline(85, color="red", linestyle="--", linewidth=1.2, label="Target 85%")
    ax.set_ylabel("CV Accuracy (%)", fontsize=12)
    ax.set_title("5-Fold Cross-Validation Accuracy (Mean ± Std)", fontsize=13, fontweight="bold")
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0, 105)
    ax.legend()
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{mean:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_priority_breakdown(df, path):
    pivot = df.groupby(["category", "priority"]).size().unstack(fill_value=0)
    pivot = pivot.reindex(columns=["Low", "Medium", "High", "Critical"])
    priority_colors = ["#55A868", "#4C72B0", "#DD8452", "#C44E52"]

    fig, ax = plt.subplots(figsize=(12, 5))
    pivot.plot(kind="bar", ax=ax, color=priority_colors, edgecolor="white", alpha=0.88)
    ax.set_xlabel("Category", fontsize=11)
    ax.set_ylabel("Ticket Count", fontsize=11)
    ax.set_title("Priority Distribution Across Categories", fontsize=13, fontweight="bold")
    ax.set_xticklabels(pivot.index, rotation=20, ha="right")
    ax.legend(title="Priority", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_text_length_dist(df, path):
    df = df.copy()
    df["text_length"] = df["clean_text"].apply(lambda x: len(x.split()))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Overall distribution
    axes[0].hist(df["text_length"], bins=30, color=PALETTE[0], edgecolor="white", alpha=0.85)
    axes[0].set_xlabel("Word Count (after preprocessing)", fontsize=11)
    axes[0].set_ylabel("Frequency", fontsize=11)
    axes[0].set_title("Distribution of Ticket Text Length", fontsize=12, fontweight="bold")

    categories = df["category"].unique()
    for i, cat in enumerate(categories):
        vals = df[df["category"] == cat]["text_length"]
        axes[1].hist(vals, bins=20, alpha=0.6, label=cat.split(" & ")[0], color=PALETTE[i % len(PALETTE)])

    axes[1].set_xlabel("Word Count", fontsize=11)
    axes[1].set_ylabel("Frequency", fontsize=11)
    axes[1].set_title("Text Length by Category", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_response_time(df, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    cat_avg = df.groupby("category")["response_time_hours"].mean().sort_values()
    axes[0].barh(cat_avg.index, cat_avg.values, color=PALETTE[:len(cat_avg)])
    axes[0].axvline(cat_avg.mean(), color="red", linestyle="--", label=f"Mean: {cat_avg.mean():.1f}h")
    axes[0].set_xlabel("Avg Response Time (hours)", fontsize=11)
    axes[0].set_title("Average Response Time by Category", fontsize=12, fontweight="bold")
    axes[0].legend()

    improved = cat_avg * 0.8
    x = np.arange(len(cat_avg))
    width = 0.35
    axes[1].bar(x - width / 2, cat_avg.values, width, label="Current", color=PALETTE[0], alpha=0.8)
    axes[1].bar(x + width / 2, improved.values, width, label="After 20% Reduction", color=PALETTE[2], alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([c.split(" & ")[0].split(" ")[0] for c in cat_avg.index],
                             rotation=20, ha="right", fontsize=9)
    axes[1].set_ylabel("Response Time (hours)", fontsize=11)
    axes[1].set_title("Projected Response Time Improvement", fontsize=12, fontweight="bold")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")



RESPONSE_TEMPLATES = {
    "Billing & Payments": {
        "template": (
            "Thank you for contacting us regarding your billing concern. "
            "We have reviewed your account and our billing team will investigate within 24 hours. "
            "You may also self-serve at: Settings > Billing > Transaction History."
        ),
        "automation_score": 0.82,
        "avg_resolution_minutes": 15,
    },
    "Technical Support": {
        "template": (
            "Thank you for reaching out. Our automated diagnostics show  "
            "Please try: (1) Clear cache & cookies, (2) Use a different browser, (3) Restart the app. "
            "If the issue persists, a technical agent will respond within 4 hours."
        ),
        "automation_score": 0.68,
        "avg_resolution_minutes": 30,
    },
    "Order Management": {
        "template": (
            "Thank you for your inquiry about your order. Your current status is: ORDER_STATUS. "
            "You can track your shipment at TRACKING_URL. "
            "If you need to modify or cancel your order, please reply within 1 hour of placing it."
        ),
        "automation_score": 0.88,
        "avg_resolution_minutes": 10,
    },
    "Account Management": {
        "template": (
            "Thank you for contacting us about your account. "
            "For security verification, please confirm your registered email. "
            "Common self-service options: Profile Settings > Security & Privacy."
        ),
        "automation_score": 0.75,
        "avg_resolution_minutes": 12,
    },
    "Product & Service Info": {
        "template": (
            "Thank you for your interest! Here is what you need to know: FAQ_LINK. "
            "Our plans start at $X/month and include FEATURES. "
            "Would you like to start a free 14-day trial? Reply YES to activate instantly."
        ),
        "automation_score": 0.95,
        "avg_resolution_minutes": 5,
    },
    "Feedback & Complaints": {
        "template": (
            "We sincerely apologise for your experience. Your feedback has been escalated to "
            "our Quality Assurance team and a senior agent will personally follow up within 2 hours. "
            "We value your loyalty and will make this right."
        ),
        "automation_score": 0.45,
        "avg_resolution_minutes": 60,
    },
}


def write_report(results_df, best_model_name, best_metrics, cv_results,
                 top_features, df, path):
    lines = []
    sep = "=" * 70

    def h1(t): lines.append(f"\n{sep}\n{t}\n{sep}")
    def h2(t): lines.append(f"\n{'─' * 60}\n{t}\n{'─' * 60}")
    def p(t=""):  lines.append(t)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    h1(f"NLP CUSTOMER SUPPORT SYSTEM — FULL REPORT")
    p(f"Generated: {now}")
    p(f"Dataset size: {len(df):,} tickets  |  Categories: {df['category'].nunique()}")

    h2("1. DATASET SUMMARY")
    for cat, cnt in df["category"].value_counts().items():
        p(f"  {cat:<35} {cnt:>4} tickets")

    h2("2. MODEL PERFORMANCE COMPARISON")
    col_w = [25, 10, 10, 10, 10]
    header = f"  {'Model':<{col_w[0]}} {'Accuracy':>{col_w[1]}} {'Precision':>{col_w[2]}} {'Recall':>{col_w[3]}} {'F1-Score':>{col_w[4]}}"
    p(header)
    p("  " + "-" * (sum(col_w) + 4))
    for _, row in results_df.iterrows():
        p(f"  {row['Model']:<{col_w[0]}} {row['Accuracy']:>{col_w[1]}.2f}%"
          f" {row['Precision']:>{col_w[2]}.2f}% {row['Recall']:>{col_w[3]}.2f}%"
          f" {row['F1-Score']:>{col_w[4]}.2f}%")

    h2("3. CROSS-VALIDATION RESULTS (5-Fold)")
    for model, cv in cv_results.items():
        p(f"  {model:<30}  Mean: {cv['mean']:.2f}%  ±  {cv['std']:.2f}%")

    h2("4. BEST MODEL DETAILS")
    p(f"  Selected Model : {best_model_name}")
    p(f"  Accuracy       : {best_metrics['Accuracy']:.2f}%  (Target ≥ 85%)")
    p(f"  F1-Score       : {best_metrics['F1-Score']:.2f}%")
    target_met = "✓ TARGET MET" if best_metrics["Accuracy"] >= 85 else "✗ Below Target"
    p(f"  Status         : {target_met}")

    h2("5. TOP DISCRIMINATIVE TERMS PER CATEGORY")
    for cat, terms in top_features.items():
        p(f"  {cat}")
        p(f"    {', '.join(terms)}")

    h2("6. AUTOMATED RESPONSE RECOMMENDATIONS")
    total_tickets = len(df)
    for cat, info in RESPONSE_TEMPLATES.items():
        cat_count = len(df[df["category"] == cat])
        pct = cat_count / total_tickets * 100
        can_automate = round(cat_count * info["automation_score"])
        p(f"\n  Category      : {cat}")
        p(f"  Volume        : {cat_count} tickets ({pct:.1f}% of total)")
        p(f"  Automatable   : ~{can_automate} tickets ({info['automation_score']*100:.0f}%)")
        p(f"  Avg Resolution: {info['avg_resolution_minutes']} minutes (automated)")
        p(f"  Template      :")
        for line in textwrap.wrap(info["template"], 64):
            p(f"    {line}")

    h2("7. KEY FINDINGS & BUSINESS IMPACT")
    total_automatable = sum(
        round(len(df[df["category"] == cat]) * info["automation_score"])
        for cat, info in RESPONSE_TEMPLATES.items()
    )
    pct_auto = total_automatable / total_tickets * 100
    avg_rt = df["response_time_hours"].mean()
    improved_rt = avg_rt * 0.80

    p(f"  • Automatable tickets        : {total_automatable:,} / {total_tickets:,} ({pct_auto:.1f}%)")
    p(f"  • Current avg response time  : {avg_rt:.1f} hours")
    p(f"  • Projected (20% reduction)  : {improved_rt:.1f} hours")
    p(f"  • Best classifier            : {best_model_name} ({best_metrics['Accuracy']:.2f}% acc)")
    p("")
    p("  RECOMMENDATIONS:")
    p("  1. Deploy Logistic Regression or SVM as the primary classifier.")
    p("  2. Automate 'Product & Service Info' responses immediately (95% safe).")
    p("  3. Use hybrid approach for 'Feedback & Complaints' (human-in-the-loop).")
    p("  4. Retrain monthly on new ticket data to maintain performance.")
    p("  5. Add confidence thresholding: route low-confidence predictions to agents.")
    p("  6. Track customer satisfaction (CSAT) post-automation to measure quality.")

    h1("END OF REPORT")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {path}")

def main():
    print("\n" + "=" * 60)
    print("  NLP CUSTOMER SUPPORT PIPELINE")
    print("=" * 60)

    # ── Data
    csv_path = os.path.join(OUTPUT_DIR, "customer_support_data.csv")

    if os.path.exists(csv_path):
        print(f"\n[1 & 2/7] Loading existing dataset from {csv_path}...")
        df = pd.read_csv(csv_path)
        # Prevent pandas from turning completely blank preprocessed strings into NaN
        df["clean_text"] = df["clean_text"].fillna("")
    else:
        print("\n[1/7] Generating dataset...")
        df = generate_dataset(samples_per_category=120)

        print("\n[2/7] Preprocessing text...")
        df["clean_text"] = df["text"].apply(preprocess_text)

        # Save dataset for next time
        df.to_csv(csv_path, index=False)
        print(f"      Dataset saved → {csv_path}")
        # Save dataset for next time
        df.to_csv(csv_path, index=False)
        print(f"      Dataset saved → {csv_path}")

    # ── Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["category"],
        test_size=0.2,
        random_state=42,
        stratify=df["category"]
    )

    categories = sorted(df["category"].unique())
    # ── Train models
    print("\n[3/7] Training models...")
    pipelines = build_pipelines()
    trained = {}
    results = []
    cv_results = {}

    for name, pipeline in pipelines.items():
        print(f"      Training: {name}...", end=" ", flush=True)
        pipeline.fit(X_train, y_train)
        metrics, y_pred = evaluate_model(pipeline, X_test, y_test, name)
        results.append(metrics)
        trained[name] = (pipeline, y_pred)
        # Cross-validation (use subset for speed)
        cv_mean, cv_std = cross_validate_model(pipeline, df["clean_text"], df["category"], cv=5)
        cv_results[name] = {"mean": cv_mean, "std": cv_std}
        print(f"Accuracy: {metrics['Accuracy']:.2f}%")

    results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)

    # ── Best model
    best_model_name = results_df.iloc[0]["Model"]
    best_metrics = results_df.iloc[0].to_dict()
    best_pipeline, best_y_pred = trained[best_model_name]
    print(f"\n      Best Model: {best_model_name} ({best_metrics['Accuracy']:.2f}%)")

    # ── Feature importance
    print("\n[4/7] Extracting top features per category...")
    top_features = {}
    for model_name in ["Logistic Regression", "Naive Bayes", "Linear SVM"]:
        if model_name in trained:
            pipeline, _ = trained[model_name]
            try:
                clf = pipeline.named_steps["clf"]
                if hasattr(clf, "coef_") or hasattr(clf, "feature_log_prob_"):
                    tf = pipeline.named_steps["tfidf"]
                    feature_names = np.array(tf.get_feature_names_out())
                    coef = clf.coef_ if hasattr(clf, "coef_") else clf.feature_log_prob_
                    label_order = clf.classes_ if hasattr(clf, "classes_") else categories
                    for idx, cat in enumerate(label_order):
                        top_idx = np.argsort(coef[idx])[-10:][::-1]
                        top_features[cat] = list(feature_names[top_idx])
                    break
            except Exception:
                continue

    # ── Visualisations
    print("\n[5/7] Generating visualisations...")
    plot_category_distribution(df, os.path.join(OUTPUT_DIR, "01_category_distribution.png"))
    plot_model_comparison(results_df, os.path.join(OUTPUT_DIR, "02_model_comparison.png"))
    plot_confusion_matrix(y_test, best_y_pred, categories,
                          os.path.join(OUTPUT_DIR, "03_confusion_matrix.png"))
    if top_features:
        plot_top_features(top_features, os.path.join(OUTPUT_DIR, "04_top_features.png"))
    plot_cross_val(cv_results, os.path.join(OUTPUT_DIR, "05_cross_validation.png"))
    plot_priority_breakdown(df, os.path.join(OUTPUT_DIR, "06_priority_breakdown.png"))
    plot_text_length_dist(df, os.path.join(OUTPUT_DIR, "07_text_length.png"))
    plot_response_time(df, os.path.join(OUTPUT_DIR, "08_response_time.png"))

    # ── Report
    print("\n[6/7] Writing text report...")
    write_report(results_df, best_model_name, best_metrics,
                 cv_results, top_features, df,
                 os.path.join(OUTPUT_DIR, "nlp_report.txt"))

    # ── Summary
    print("\n[7/7] Summary")
    print(f"\n{'─'*50}")
    print(f"  Best Model       : {best_model_name}")
    print(f"  Accuracy         : {best_metrics['Accuracy']:.2f}%")
    print(f"  F1-Score         : {best_metrics['F1-Score']:.2f}%")
    print(f"  Target ≥ 85%     : {'✓ MET' if best_metrics['Accuracy'] >= 85 else '✗ Not met'}")
    print(f"\n  Output files → {OUTPUT_DIR}/")
    print(f"{'─'*50}\n")

    # ── Classification report
    print("Per-category Classification Report:")
    print(classification_report(y_test, best_y_pred, target_names=categories))

    return df, results_df, trained, best_model_name


def interactive_chat(best_pipeline):
    print("\n" + "=" * 60)
    print("  LIVE TICKET CLASSIFICATION (Type 'quit' to exit)")
    print("=" * 60)

    while True:
        user_input = input("\nYou (Customer): ")
        if user_input.lower() in ['quit', 'exit']:
            print("Exiting chat...")
            break

        if not user_input.strip():
            continue

        # 1. Clean the text using the exact same function from training
        cleaned_text = preprocess_text(user_input)

        # 2. Ask the winning pipeline to predict the category
        prediction = best_pipeline.predict([cleaned_text])[0]

        # 3. Fetch the automated response template
        response = RESPONSE_TEMPLATES.get(prediction, {}).get("template", "An agent will be with you shortly.")

        print(f"\n [Routing to: {prediction}]")
        print(f"Auto-Reply: {response}")


if __name__ == "__main__":
    df, results_df, trained, best_model_name = main()

    # Extract the actual winning pipeline object from the dictionary
    winning_pipeline = trained[best_model_name][0]

    # Launch the chat interface
    interactive_chat(winning_pipeline)
