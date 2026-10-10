import streamlit as st
import re
import pandas as pd
import plotly.express as px
from deep_translator import GoogleTranslator
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
import joblib
import os


from modules.database import (
    create_table,
    add_expense,
    get_expenses,
    total_spending,
    delete_expense,
    monthly_spending,
    update_expense
)

from modules.categorizer import predict_category

from modules.ai_insights import (
    generate_financial_review,
    answer_financial_question
)

from modules.budget import save_budget, get_budget
from modules.ocr import extract_receipt_details
from modules.anomaly_detector import detect_spending_anomalies
from modules.forecasting import forecast_next_month
from modules.recommendations import (
    generate_recommendations
)
from modules.forecast_engine import forecast_spending


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="FinSight AI",
    page_icon="💰",
    layout="wide"
)


# ============================================================
# FinSight AI — Premium FinTech UI Theme
# ============================================================

st.markdown("""
<style>
/* ========================= FinSight AI — Premium FinTech UI ========================= */
.stApp {
    background:
        radial-gradient(circle at 8% 8%, rgba(124,58,237,.20), transparent 30%),
        radial-gradient(circle at 92% 12%, rgba(34,211,238,.13), transparent 28%),
        radial-gradient(circle at 50% 105%, rgba(79,70,229,.12), transparent 34%),
        #080d1a !important;
    color:#f8fafc;
}
.main .block-container { position:relative; z-index:1; padding-top:1.5rem; padding-bottom:4rem; max-width:1220px; }
h1,h2,h3 { color:#f8fafc !important; letter-spacing:-.4px; }
p,label { color:#dbe4f0; }
hr { border:0; border-top:1px solid rgba(148,163,184,.14); margin:1.6rem 0 1.4rem; }

/* Hero */
.finsight-hero { position:relative; text-align:center; padding:42px 28px 34px; margin:0 0 26px; background:linear-gradient(145deg,rgba(20,30,52,.90),rgba(10,16,31,.78)); border:1px solid rgba(139,92,246,.25); border-radius:26px; box-shadow:0 22px 55px rgba(0,0,0,.30), inset 0 1px 0 rgba(255,255,255,.05); overflow:hidden; }
.finsight-hero:before,.finsight-hero:after { content:""; position:absolute; border-radius:50%; filter:blur(70px); pointer-events:none; }
.finsight-hero:before { width:280px;height:280px;top:-170px;left:10%;background:rgba(124,58,237,.25); }
.finsight-hero:after { width:260px;height:260px;right:8%;bottom:-180px;background:rgba(34,211,238,.13); }
.finsight-icon { position:relative; display:inline-flex; align-items:center; justify-content:center; width:68px;height:68px;margin-bottom:10px; background:linear-gradient(135deg,#8b5cf6,#22d3ee); border-radius:19px; font-size:34px; box-shadow:0 12px 32px rgba(124,58,237,.30); }
.finsight-title { position:relative; margin:4px 0 7px; font-size:3rem !important; font-weight:850 !important; letter-spacing:-1.7px; background:linear-gradient(90deg,#fff,#c4b5fd,#67e8f9); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.finsight-subtitle { position:relative; font-size:1.18rem;font-weight:650;color:#cbd5e1;margin-bottom:10px; }
.finsight-description { position:relative; max-width:720px;margin:0 auto;font-size:.98rem;line-height:1.65;color:#94a3b8; }
.finsight-badges { position:relative;display:flex;justify-content:center;flex-wrap:wrap;gap:9px;margin-top:20px; }
.finsight-badges span { padding:7px 13px;border-radius:999px;background:rgba(30,41,59,.70);border:1px solid rgba(148,163,184,.16);color:#cbd5e1;font-size:.8rem;font-weight:650;transition:.2s ease; }
.finsight-badges span:hover { transform:translateY(-2px);border-color:rgba(139,92,246,.5);background:rgba(124,58,237,.15); }

/* Section headings */
.stApp h2 { margin-top:1rem !important; margin-bottom:.55rem !important; }
.stApp h3 { margin-top:.8rem !important; }
.section-kicker { color:#8b5cf6;font-size:.72rem;font-weight:800;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:3px; }
.dashboard-section-title { display:flex;align-items:center;gap:13px;margin:5px 0 20px; }
.dashboard-section-icon { width:44px;height:44px;display:flex;align-items:center;justify-content:center;border-radius:13px;background:linear-gradient(135deg,rgba(124,58,237,.23),rgba(34,211,238,.12));border:1px solid rgba(139,92,246,.25);font-size:21px;box-shadow:0 8px 22px rgba(0,0,0,.18); }
.dashboard-section-heading { color:#f8fafc;font-size:1.38rem;font-weight:800;line-height:1.2; }
.dashboard-section-subtitle { color:#94a3b8;font-size:.86rem;margin-top:3px; }

/* Metric cards */
div[data-testid="metric-container"], [data-testid="stMetric"] { position:relative !important;overflow:hidden !important;background:linear-gradient(145deg,rgba(30,41,59,.94),rgba(15,23,42,.88)) !important;border:1px solid rgba(139,92,246,.20) !important;border-radius:18px !important;padding:19px 19px 15px !important;min-height:108px;box-shadow:0 10px 30px rgba(0,0,0,.22),inset 0 1px 0 rgba(255,255,255,.04) !important;transition:transform .2s ease,border-color .2s ease,box-shadow .2s ease !important; }
div[data-testid="metric-container"]:before,[data-testid="stMetric"]:before { content:"" !important;position:absolute !important;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#8b5cf6,#22d3ee) !important;opacity:.85; }
div[data-testid="metric-container"]:hover,[data-testid="stMetric"]:hover { transform:translateY(-4px) !important;border-color:rgba(139,92,246,.52) !important;box-shadow:0 17px 38px rgba(0,0,0,.30),0 0 24px rgba(139,92,246,.10) !important; }
[data-testid="stMetricLabel"] { color:#94a3b8 !important;font-size:.86rem !important;font-weight:650 !important; }
[data-testid="stMetricValue"] { color:#f8fafc !important;font-size:1.75rem !important;font-weight:800 !important;letter-spacing:-.5px; }

/* Inputs / buttons */
.stTextInput input,.stNumberInput input,.stDateInput input,.stTextArea textarea { background:#111b2e !important;color:#f8fafc !important;border:1px solid #334155 !important;border-radius:11px !important; }
.stTextInput input:focus,.stNumberInput input:focus,.stDateInput input:focus,.stTextArea textarea:focus { border-color:#8b5cf6 !important;box-shadow:0 0 0 2px rgba(139,92,246,.12) !important; }
div[data-baseweb="select"] > div { background:#111b2e !important;border-color:#334155 !important;border-radius:11px !important; }
.stButton > button { background:linear-gradient(135deg,#7c3aed,#2563eb) !important;color:#fff !important;border:1px solid rgba(255,255,255,.08) !important;border-radius:11px !important;font-weight:700 !important;min-height:42px;transition:.2s ease !important;box-shadow:0 7px 18px rgba(37,99,235,.16); }
.stButton > button:hover { transform:translateY(-2px) !important;box-shadow:0 10px 24px rgba(124,58,237,.28) !important; }

/* Charts / tables / expanders */
div[data-testid="stPlotlyChart"] { background:rgba(10,16,31,.56);border:1px solid rgba(148,163,184,.13);border-radius:18px;padding:8px 8px 2px;box-shadow:0 8px 25px rgba(0,0,0,.18);transition:.2s ease; }
div[data-testid="stPlotlyChart"]:hover { transform:translateY(-2px);border-color:rgba(139,92,246,.35); }
div[data-testid="stVerticalBlockBorderWrapper"] { border-color:rgba(148,163,184,.13) !important;border-radius:18px !important;background:rgba(15,23,42,.22); }
[data-testid="stExpander"] { background:rgba(15,23,42,.58);border:1px solid rgba(148,163,184,.15);border-radius:14px; }
[data-testid="stAlert"] { border-radius:14px !important;border:1px solid rgba(148,163,184,.13) !important; }
[data-testid="stDataFrame"] { border-radius:14px;overflow:hidden; }

/* Tabs */
div[data-baseweb="tab-list"] { gap:7px; background:rgba(15,23,42,.42);padding:5px;border-radius:13px; }
button[data-baseweb="tab"] { color:#94a3b8 !important;font-weight:700 !important;border-radius:9px !important; }
button[data-baseweb="tab"][aria-selected="true"] { color:#f8fafc !important;background:linear-gradient(135deg,rgba(124,58,237,.28),rgba(37,99,235,.20)); }

/* ============================================================
   FINsight AI — Responsive Mobile Layout
   UI ONLY. Backend / ML / database logic unchanged.
   ============================================================ */

@media (max-width: 900px) {

    .main .block-container {
        max-width: 100% !important;
        padding: 1rem 0.85rem 3rem !important;
    }

    /* Make every Streamlit column become a full-width mobile row.
       This is the key difference from the previous CSS pass. */
    [data-testid="stHorizontalBlock"] {
        width: 100% !important;
        display: flex !important;
        flex-direction: column !important;
        flex-wrap: nowrap !important;
        gap: 0.8rem !important;
    }

    [data-testid="stHorizontalBlock"] > [data-testid="column"] {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 100% !important;
        flex: 1 1 100% !important;
    }

    /* Keep the hero compact */
    .finsight-hero {
        padding: 28px 16px 24px !important;
        margin-bottom: 18px !important;
        border-radius: 20px !important;
    }

    .finsight-icon {
        width: 56px !important;
        height: 56px !important;
        font-size: 28px !important;
        border-radius: 16px !important;
    }

    .finsight-title {
        font-size: clamp(1.75rem, 8vw, 2.35rem) !important;
        line-height: 1.1 !important;
        letter-spacing: -1px !important;
    }

    .finsight-subtitle {
        font-size: 0.95rem !important;
        line-height: 1.45 !important;
    }

    .finsight-description {
        font-size: 0.84rem !important;
        line-height: 1.55 !important;
    }

    .finsight-badges {
        gap: 6px !important;
        margin-top: 14px !important;
    }

    .finsight-badges span {
        font-size: 0.72rem !important;
        padding: 6px 10px !important;
    }

    /* Section headings */
    .dashboard-section-title {
        gap: 10px !important;
        margin: 4px 0 15px !important;
    }

    .dashboard-section-icon {
        width: 40px !important;
        min-width: 40px !important;
        height: 40px !important;
        font-size: 19px !important;
    }

    .dashboard-section-heading {
        font-size: 1.12rem !important;
    }

    .dashboard-section-subtitle {
        font-size: 0.76rem !important;
    }

    /* Metrics become obvious single cards */
    div[data-testid="metric-container"],
    [data-testid="stMetric"] {
        width: 100% !important;
        min-height: 88px !important;
        padding: 15px !important;
        border-radius: 15px !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.76rem !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.4rem !important;
        line-height: 1.2 !important;
    }

    /* Touch-friendly controls */
    .stButton > button,
    .stDownloadButton > button {
        width: 100% !important;
        min-height: 46px !important;
        border-radius: 12px !important;
    }

    .stTextInput input,
    .stNumberInput input,
    .stDateInput input,
    .stTextArea textarea {
        min-height: 46px !important;
        font-size: 16px !important;
        border-radius: 12px !important;
    }

    div[data-baseweb="select"] > div {
        min-height: 46px !important;
        border-radius: 12px !important;
    }

    /* Charts */
    div[data-testid="stPlotlyChart"] {
        width: 100% !important;
        max-width: 100% !important;
        padding: 3px !important;
        border-radius: 14px !important;
        overflow: hidden !important;
        transform: none !important;
    }

    /* Tables scroll rather than break the viewport */
    [data-testid="stDataFrame"] {
        width: 100% !important;
        max-width: 100% !important;
        overflow-x: auto !important;
        border-radius: 12px !important;
    }

    /* Tabs remain horizontally usable */
    div[data-baseweb="tab-list"] {
        width: 100% !important;
        overflow-x: auto !important;
        white-space: nowrap !important;
        scrollbar-width: none !important;
    }

    div[data-baseweb="tab-list"]::-webkit-scrollbar {
        display: none !important;
    }

    button[data-baseweb="tab"] {
        flex: 0 0 auto !important;
        padding: 9px 12px !important;
        font-size: 0.78rem !important;
    }

    /* Alerts / expanders */
    [data-testid="stAlert"],
    [data-testid="stExpander"] {
        border-radius: 13px !important;
        overflow-wrap: anywhere !important;
    }

    /* Prevent touch hover effects */
    div[data-testid="metric-container"]:hover,
    [data-testid="stMetric"]:hover,
    div[data-testid="stPlotlyChart"]:hover,
    .stButton > button:hover {
        transform: none !important;
    }
}

@media (max-width: 480px) {

    .main .block-container {
        padding-left: 0.65rem !important;
        padding-right: 0.65rem !important;
    }

    .finsight-hero {
        padding: 22px 12px 20px !important;
        border-radius: 17px !important;
    }

    .finsight-title {
        font-size: 1.72rem !important;
    }

    .finsight-subtitle {
        font-size: 0.84rem !important;
    }

    .finsight-description {
        font-size: 0.78rem !important;
    }

    .finsight-badges span {
        font-size: 0.68rem !important;
        padding: 5px 8px !important;
    }

    div[data-testid="metric-container"],
    [data-testid="stMetric"] {
        min-height: 82px !important;
        padding: 13px !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.25rem !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.72rem !important;
    }

    button[data-baseweb="tab"] {
        font-size: 0.72rem !important;
        padding: 8px 10px !important;
    }
}
</style>
""", unsafe_allow_html=True)


create_table()


# ==========================================
# 🌐 LANGUAGE SYSTEM
# ==========================================

LANGUAGE_OPTIONS = {
    "English": "en",
    "தமிழ்": "ta",
    "हिन्दी": "hi",
    "తెలుగు": "te",
    "മലയാളം": "ml",
    "ಕನ್ನಡ": "kn"
}

# Static UI translations.
# English is the fallback for anything not translated here.
UI_TRANSLATIONS = {

    # ========================================================
    # 🇬🇧 ENGLISH
    # ========================================================

    "en": {

        "app_subtitle": "🤖 AI-Powered Personal Finance Tracker",
        "app_description": "Track your spending, automatically categorize transactions, and understand where your money goes.",

        "add_transaction": "➕ Add Transaction",
        "description": "Transaction Description",
        "amount": "Amount (₹)",
        "predict": "🤖 Predict Category",
        "add_expense": "💾 Add Expense",

        "ai_predicted": "🤖 AI Predicted Category",
        "category_selected": "Category selected by FinSight AI",

        "empty_description": "Please enter a transaction description.",
        "invalid_amount": "Please enter an amount greater than ₹0.",
        "expense_added": "✅ Expense added successfully as",

        "ai_review": "🧠 AI Financial Review",
        "ai_review_empty": "Add some expenses first and FinSight AI will analyze your spending.",

        "budget_intelligence": "🎯 Budget Intelligence",
        "monthly_budget": "Set Monthly Budget (₹)",
        "save_budget": "💾 Save Budget",
        "budget_saved": "✅ Monthly budget set to",

        "budget": "💰 Budget",
        "budget_used": "📊 Budget Used",
        "transactions": "🧾 Transactions",
        "average": "📌 Average Transaction",
        "total_spending": "💰 Total Spending",

        "spending_analysis": "📊 Spending Analysis",
        "spending_by_category": "💰 Spending by Category",
        "spending_distribution": "📈 Spending Distribution",
        "where_money_goes": "Where Your Money Goes",
        "spending_over_time": "📈 Spending Over Time",

        "no_expenses_analysis": "Add expenses to see spending analysis.",

        "expense_history": "🧾 Expense History",
        "no_transactions": "No transactions have been added yet.",

        "edit_expense": "✏️ Edit Expense",
        "select_expense": "Select Expense ID",
        "edit_description": "Description",
        "edit_amount": "Amount (₹)",
        "edit_category": "Category",
        "update_expense": "💾 Update Expense",
        "expense_updated": "✅ Expense updated successfully!",

        "delete_transaction": "🗑️ Delete Transaction",
        "enter_id": "Enter Transaction ID",
        "delete": "🗑️ Delete",
        "transaction_deleted": "Transaction deleted.",

        "finance_qa": "💬 Ask FinSight AI",
        "ask_question": "Ask a question about your spending",
        "clear_question": "Clear question",
        "ask_ai": "🤖 Ask FinSight AI",
        "question_first": "Please enter a question first.",

        "translation_unavailable": "Translation is unavailable right now.",

        # Monthly analysis
        "monthly_analysis": "📅 Monthly Spending Analysis",
        "select_month": "Select Month",

        # Budget comparison
        "budget_vs_actual": "🎯 Budget vs Actual Spending",
        "set_budget_comparison": "Set a monthly budget to see your budget comparison.",
        "add_expenses_comparison": "Add expenses to see your budget comparison.",
        "actual_spending": "💸 Actual Spending",
        "remaining": "💰 Remaining",
        "over_budget": "⚠️ Over Budget",
        "budget_used_text": "Budget used",

        "comfortably_within": "🟢 You're comfortably within your budget.",
        "approaching_budget": "🟡 You're approaching your monthly budget.",
        "very_close": "🟠 You're very close to your budget limit.",
        "exceeded": "🔴 You've exceeded your monthly budget.",

        # Language
        "language": "🌐 Language",
        "choose_language": "🌐 Choose Your Language",
        "continue": "🚀 Continue",

        # Welcome
        "welcome_title": "💰 FinSight AI",
        "welcome_subtitle": "🤖 AI-Powered Personal Finance Assistant",
        "welcome_text": "Manage your expenses, understand your spending, and make smarter financial decisions.",
        "select_language_text": "Select the language you want to use",

        # General
        "no_valid_dates": "No valid dates available for the trend chart.",
        "no_expenses_edit": "No expenses available to edit.",
        "no_expenses_budget": "Set a monthly budget to activate Budget Intelligence.",

        # OCR
        "receipt": "📷 Receipt Scanner",
        "upload_receipt": "Upload your receipt or bill",
        "uploaded_receipt": "Uploaded Receipt",
        "scan_receipt": "🔍 Scan Receipt",
        "scanning_receipt": "Scanning receipt...",
        "receipt_scanned": "✅ Receipt scanned successfully!",
        "receipt_details": "📄 Extracted Receipt Details",
        "receipt_description": "Description",
        "receipt_amount": "Amount",
        "save_scanned_expense": "💾 Save Scanned Expense",
        "receipt_amount_not_found": "⚠️ Could not detect the amount from this receipt. Please add the expense manually.",
        "view_ocr_text": "🔎 View extracted receipt text",
        "receipt_scan_failed": "Receipt scanning failed:",

        # Buttons
        "cancel": "Cancel",
        "yes_delete": "Yes, Delete",

        # Footer
        "footer": "FinSight AI • Personal Finance Analytics & ML Categorization",

        # AI recommendations
        "high_concentration": "⚠️ High Spending Concentration",
        "large_transaction": "🚨 Large Transaction Detected",
        "recommendation": "💡 AI Recommendation",

        "personalized_recommendations": "🧠 Personalized Financial Recommendations",

        "recommendations_description": "FinSight AI analyzes your spending, budget, anomalies and forecast to provide personalized recommendations."
    },


    # ========================================================
    # 🇮🇳 TAMIL
    # ========================================================

    "ta": {

        "app_subtitle": "🤖 AI மூலம் இயங்கும் தனிப்பட்ட நிதி கண்காணிப்பான்",
        "app_description": "உங்கள் செலவுகளை கண்காணித்து, பரிவர்த்தனைகளை தானாக வகைப்படுத்தி, பணம் எங்கு செல்கிறது என்பதை புரிந்துகொள்ளுங்கள்.",

        "add_transaction": "➕ பரிவர்த்தனை சேர்க்கவும்",
        "description": "பரிவர்த்தனை விவரம்",
        "amount": "தொகை (₹)",
        "predict": "🤖 வகையை கணிக்கவும்",
        "add_expense": "💾 செலவை சேமிக்கவும்",

        "ai_predicted": "🤖 AI கணித்த வகை",
        "category_selected": "FinSight AI தேர்ந்தெடுத்த வகை",

        "empty_description": "பரிவர்த்தனை விவரத்தை உள்ளிடவும்.",
        "invalid_amount": "₹0-க்கு அதிகமான தொகையை உள்ளிடவும்.",
        "expense_added": "✅ செலவு வெற்றிகரமாக சேர்க்கப்பட்டது. வகை:",

        "ai_review": "🧠 AI நிதி மதிப்பாய்வு",
        "ai_review_empty": "முதலில் சில செலவுகளைச் சேர்க்கவும். FinSight AI உங்கள் செலவுகளை பகுப்பாய்வு செய்யும்.",

        "budget_intelligence": "🎯 பட்ஜெட் மேலாண்மை",
        "monthly_budget": "மாதாந்திர பட்ஜெட் அமைக்கவும் (₹)",
        "save_budget": "💾 பட்ஜெட்டை சேமிக்கவும்",
        "budget_saved": "✅ மாதாந்திர பட்ஜெட் அமைக்கப்பட்டது:",

        "budget": "💰 பட்ஜெட்",
        "budget_used": "📊 பயன்படுத்திய பட்ஜெட்",
        "transactions": "🧾 பரிவர்த்தனைகள்",
        "average": "📌 சராசரி பரிவர்த்தனை",
        "total_spending": "💰 மொத்த செலவு",

        "spending_analysis": "📊 செலவு பகுப்பாய்வு",
        "spending_by_category": "💰 வகை வாரியான செலவு",
        "spending_distribution": "📈 செலவு பகிர்வு",
        "where_money_goes": "உங்கள் பணம் எங்கு செல்கிறது",
        "spending_over_time": "📈 காலப்போக்கில் செலவு",

        "no_expenses_analysis": "செலவு பகுப்பாய்வைக் காண செலவுகளைச் சேர்க்கவும்.",

        "expense_history": "🧾 செலவு வரலாறு",
        "no_transactions": "இன்னும் பரிவர்த்தனைகள் சேர்க்கப்படவில்லை.",

        "edit_expense": "✏️ செலவைத் திருத்தவும்",
        "select_expense": "செலவு ID-ஐ தேர்ந்தெடுக்கவும்",
        "edit_description": "விவரம்",
        "edit_amount": "தொகை (₹)",
        "edit_category": "வகை",
        "update_expense": "💾 செலவை புதுப்பிக்கவும்",
        "expense_updated": "✅ செலவு வெற்றிகரமாக புதுப்பிக்கப்பட்டது!",

        "delete_transaction": "🗑️ பரிவர்த்தனையை நீக்கவும்",
        "enter_id": "பரிவர்த்தனை ID-ஐ உள்ளிடவும்",
        "delete": "🗑️ நீக்கு",
        "transaction_deleted": "பரிவர்த்தனை நீக்கப்பட்டது.",

        "finance_qa": "💬 FinSight AI-யிடம் கேளுங்கள்",
        "ask_question": "உங்கள் செலவு பற்றி கேள்வி கேளுங்கள்",
        "clear_question": "கேள்வியை அழிக்கவும்",
        "ask_ai": "🤖 FinSight AI-யிடம் கேளுங்கள்",
        "question_first": "முதலில் ஒரு கேள்வியை உள்ளிடவும்.",

        "translation_unavailable": "மொழிபெயர்ப்பு தற்போது கிடைக்கவில்லை.",

        "monthly_analysis": "📅 மாதாந்திர செலவு பகுப்பாய்வு",
        "select_month": "மாதத்தை தேர்ந்தெடுக்கவும்",

        "budget_vs_actual": "🎯 பட்ஜெட் மற்றும் உண்மையான செலவு",
        "set_budget_comparison": "பட்ஜெட் ஒப்பீட்டைக் காண மாதாந்திர பட்ஜெட்டை அமைக்கவும்.",
        "add_expenses_comparison": "பட்ஜெட் ஒப்பீட்டைக் காண செலவுகளைச் சேர்க்கவும்.",
        "actual_spending": "💸 உண்மையான செலவு",
        "remaining": "💰 மீதம்",
        "over_budget": "⚠️ பட்ஜெட்டை மீறியது",
        "budget_used_text": "பயன்படுத்திய பட்ஜெட்",

        "comfortably_within": "🟢 உங்கள் செலவு பட்ஜெட்டுக்குள் நன்றாக உள்ளது.",
        "approaching_budget": "🟡 நீங்கள் மாதாந்திர பட்ஜெட்டை நெருங்குகிறீர்கள்.",
        "very_close": "🟠 நீங்கள் பட்ஜெட் வரம்பிற்கு மிகவும் அருகில் உள்ளீர்கள்.",
        "exceeded": "🔴 நீங்கள் மாதாந்திர பட்ஜெட்டை மீறிவிட்டீர்கள்.",

        "language": "🌐 மொழி",
        "choose_language": "🌐 உங்கள் மொழியைத் தேர்ந்தெடுக்கவும்",
        "continue": "🚀 தொடரவும்",

        "welcome_title": "💰 FinSight AI",
        "welcome_subtitle": "🤖 AI மூலம் இயங்கும் தனிப்பட்ட நிதி உதவியாளர்",
        "welcome_text": "உங்கள் செலவுகளை நிர்வகித்து, செலவுகளைப் புரிந்துகொண்டு, சிறந்த நிதி முடிவுகளை எடுக்கவும்.",
        "select_language_text": "நீங்கள் பயன்படுத்த விரும்பும் மொழியைத் தேர்ந்தெடுக்கவும்",

        "no_valid_dates": "போக்கு வரைபடத்திற்கு சரியான தேதிகள் இல்லை.",
        "no_expenses_edit": "திருத்துவதற்கு செலவுகள் இல்லை.",
        "no_expenses_budget": "Budget Intelligence-ஐ செயல்படுத்த மாதாந்திர பட்ஜெட்டை அமைக்கவும்.",

        "receipt": "📷 ரசீது ஸ்கேனர்",
        "upload_receipt": "உங்கள் ரசீது அல்லது பில்லை பதிவேற்றவும்",
        "uploaded_receipt": "பதிவேற்றிய ரசீது",
        "scan_receipt": "🔍 ரசீதை ஸ்கேன் செய்யவும்",
        "scanning_receipt": "ரசீது ஸ்கேன் செய்யப்படுகிறது...",
        "receipt_scanned": "✅ ரசீது வெற்றிகரமாக ஸ்கேன் செய்யப்பட்டது!",
        "receipt_details": "📄 பெறப்பட்ட ரசீது விவரங்கள்",
        "receipt_description": "விவரம்",
        "receipt_amount": "தொகை",
        "save_scanned_expense": "💾 ஸ்கேன் செய்த செலவை சேமிக்கவும்",
        "receipt_amount_not_found": "⚠️ ரசீதிலிருந்து தொகையை கண்டறிய முடியவில்லை. செலவை கைமுறையாக சேர்க்கவும்.",
        "view_ocr_text": "🔎 பெறப்பட்ட ரசீது உரையைப் பார்க்கவும்",
        "receipt_scan_failed": "ரசீது ஸ்கேன் தோல்வியடைந்தது:",

        "cancel": "ரத்து",
        "yes_delete": "ஆம், நீக்கு",

        "footer": "FinSight AI • தனிப்பட்ட நிதி பகுப்பாய்வு மற்றும் ML வகைப்படுத்தல்",

        "high_concentration": "⚠️ அதிக செலவு குவிப்பு",
        "large_transaction": "🚨 பெரிய பரிவர்த்தனை கண்டறியப்பட்டது",
        "recommendation": "💡 AI பரிந்துரை",

        "personalized_recommendations": "🧠 தனிப்பயனாக்கப்பட்ட நிதி பரிந்துரைகள்",

        "recommendations_description": "FinSight AI உங்கள் செலவுகள், பட்ஜெட், அசாதாரண செலவுகள் மற்றும் முன்னறிவிப்பை பகுப்பாய்வு செய்து தனிப்பயனாக்கப்பட்ட பரிந்துரைகளை வழங்குகிறது."
    },
    # ========================================================
    # 🇮🇳 HINDI
    # ========================================================

    "hi": {

        "app_subtitle": "🤖 AI-संचालित व्यक्तिगत वित्त ट्रैकर",
        "app_description": "अपने खर्चों को ट्रैक करें, लेन-देन को स्वचालित रूप से वर्गीकृत करें और समझें कि आपका पैसा कहाँ जा रहा है।",

        "add_transaction": "➕ लेन-देन जोड़ें",
        "description": "लेन-देन का विवरण",
        "amount": "राशि (₹)",
        "predict": "🤖 श्रेणी की भविष्यवाणी करें",
        "add_expense": "💾 खर्च सहेजें",

        "ai_predicted": "🤖 AI द्वारा अनुमानित श्रेणी",
        "category_selected": "FinSight AI द्वारा चुनी गई श्रेणी",

        "empty_description": "कृपया लेन-देन का विवरण दर्ज करें।",
        "invalid_amount": "₹0 से अधिक राशि दर्ज करें।",
        "expense_added": "✅ खर्च सफलतापूर्वक जोड़ा गया। श्रेणी:",

        "ai_review": "🧠 AI वित्तीय समीक्षा",
        "ai_review_empty": "पहले कुछ खर्च जोड़ें। FinSight AI आपके खर्चों का विश्लेषण करेगा।",

        "budget_intelligence": "🎯 बजट प्रबंधन",
        "monthly_budget": "मासिक बजट सेट करें (₹)",
        "save_budget": "💾 बजट सहेजें",
        "budget_saved": "✅ मासिक बजट सेट किया गया:",

        "budget": "💰 बजट",
        "budget_used": "📊 उपयोग किया गया बजट",
        "transactions": "🧾 लेन-देन",
        "average": "📌 औसत लेन-देन",
        "total_spending": "💰 कुल खर्च",

        "spending_analysis": "📊 खर्च विश्लेषण",
        "spending_by_category": "💰 श्रेणी के अनुसार खर्च",
        "spending_distribution": "📈 खर्च वितरण",
        "where_money_goes": "आपका पैसा कहाँ जाता है",
        "spending_over_time": "📈 समय के साथ खर्च",

        "no_expenses_analysis": "खर्च विश्लेषण देखने के लिए खर्च जोड़ें.",

        "expense_history": "🧾 खर्च इतिहास",
        "no_transactions": "अभी तक कोई लेन-देन नहीं जोड़ा गया है।",

        "edit_expense": "✏️ खर्च संपादित करें",
        "select_expense": "खर्च ID चुनें",
        "edit_description": "विवरण",
        "edit_amount": "राशि (₹)",
        "edit_category": "श्रेणी",
        "update_expense": "💾 खर्च अपडेट करें",
        "expense_updated": "✅ खर्च सफलतापूर्वक अपडेट किया गया!",

        "delete_transaction": "🗑️ लेन-देन हटाएँ",
        "enter_id": "लेन-देन ID दर्ज करें",
        "delete": "🗑️ हटाएँ",
        "transaction_deleted": "लेन-देन हटा दिया गया।",

        "finance_qa": "💬 FinSight AI से पूछें",
        "ask_question": "अपने खर्च के बारे में प्रश्न पूछें",
        "clear_question": "प्रश्न साफ करें",
        "ask_ai": "🤖 FinSight AI से पूछें",
        "question_first": "पहले एक प्रश्न दर्ज करें।",

        "translation_unavailable": "अनुवाद अभी उपलब्ध नहीं है।",

        "monthly_analysis": "📅 मासिक खर्च विश्लेषण",
        "select_month": "महीना चुनें",

        "budget_vs_actual": "🎯 बजट बनाम वास्तविक खर्च",
        "set_budget_comparison": "बजट तुलना देखने के लिए मासिक बजट सेट करें।",
        "add_expenses_comparison": "बजट तुलना देखने के लिए खर्च जोड़ें।",
        "actual_spending": "💸 वास्तविक खर्च",
        "remaining": "💰 शेष राशि",
        "over_budget": "⚠️ बजट से अधिक",
        "budget_used_text": "उपयोग किया गया बजट",

        "comfortably_within": "🟢 आपका खर्च बजट के अंदर है।",
        "approaching_budget": "🟡 आप अपने मासिक बजट के करीब पहुँच रहे हैं।",
        "very_close": "🟠 आप बजट सीमा के बहुत करीब हैं।",
        "exceeded": "🔴 आपने मासिक बजट पार कर लिया है।",

        "language": "🌐 भाषा",
        "choose_language": "🌐 अपनी भाषा चुनें",
        "continue": "🚀 जारी रखें",

        "welcome_title": "💰 FinSight AI",
        "welcome_subtitle": "🤖 AI-संचालित व्यक्तिगत वित्त सहायक",
        "welcome_text": "अपने खर्च प्रबंधित करें, खर्चों को समझें और बेहतर वित्तीय निर्णय लें।",
        "select_language_text": "वह भाषा चुनें जिसका आप उपयोग करना चाहते हैं",

        "no_valid_dates": "ट्रेंड चार्ट के लिए मान्य तिथियाँ उपलब्ध नहीं हैं।",
        "no_expenses_edit": "संपादित करने के लिए कोई खर्च नहीं है।",
        "no_expenses_budget": "बजट प्रबंधन सक्रिय करने के लिए मासिक बजट सेट करें।",

        "receipt": "📷 रसीद स्कैनर",
        "upload_receipt": "अपनी रसीद या बिल अपलोड करें",
        "uploaded_receipt": "अपलोड की गई रसीद",
        "scan_receipt": "🔍 रसीद स्कैन करें",
        "scanning_receipt": "रसीद स्कैन की जा रही है...",
        "receipt_scanned": "✅ रसीद सफलतापूर्वक स्कैन हुई!",
        "receipt_details": "📄 निकाली गई रसीद जानकारी",
        "receipt_description": "विवरण",
        "receipt_amount": "राशि",
        "save_scanned_expense": "💾 स्कैन किया गया खर्च सहेजें",
        "receipt_amount_not_found": "⚠️ रसीद से राशि नहीं मिल सकी। कृपया खर्च मैन्युअल रूप से जोड़ें।",
        "view_ocr_text": "🔎 निकाला गया रसीद टेक्स्ट देखें",
        "receipt_scan_failed": "रसीद स्कैन विफल:",

        "cancel": "रद्द करें",
        "yes_delete": "हाँ, हटाएँ",

        "footer": "FinSight AI • व्यक्तिगत वित्त विश्लेषण और ML वर्गीकरण",

        "high_concentration": "⚠️ उच्च खर्च एकाग्रता",
        "large_transaction": "🚨 बड़ा लेन-देन पाया गया",
        "recommendation": "💡 AI सुझाव",

        "personalized_recommendations": "🧠 व्यक्तिगत वित्तीय सुझाव",

        "recommendations_description": "FinSight AI आपके खर्च, बजट, असामान्य लेन-देन और पूर्वानुमान का विश्लेषण करके व्यक्तिगत सुझाव देता है।"
    },


    # ========================================================
    # 🇮🇳 TELUGU
    # ========================================================

    "te": {

        "app_subtitle": "🤖 AI ఆధారిత వ్యక్తిగత ఆర్థిక ట్రాకర్",
        "app_description": "మీ ఖర్చులను ట్రాక్ చేయండి, లావాదేవీలను ఆటోమేటిక్‌గా వర్గీకరించండి మరియు మీ డబ్బు ఎక్కడ ఖర్చవుతుందో తెలుసుకోండి.",

        "add_transaction": "➕ లావాదేవీని జోడించండి",
        "description": "లావాదేవీ వివరణ",
        "amount": "మొత్తం (₹)",
        "predict": "🤖 వర్గాన్ని అంచనా వేయండి",
        "add_expense": "💾 ఖర్చును సేవ్ చేయండి",

        "ai_predicted": "🤖 AI అంచనా వేసిన వర్గం",
        "category_selected": "FinSight AI ఎంచుకున్న వర్గం",

        "empty_description": "దయచేసి లావాదేవీ వివరణను నమోదు చేయండి.",
        "invalid_amount": "₹0 కంటే ఎక్కువ మొత్తాన్ని నమోదు చేయండి.",
        "expense_added": "✅ ఖర్చు విజయవంతంగా జోడించబడింది. వర్గం:",

        "ai_review": "🧠 AI ఆర్థిక సమీక్ష",
        "ai_review_empty": "ముందుగా కొన్ని ఖర్చులను జోడించండి. FinSight AI మీ ఖర్చులను విశ్లేషిస్తుంది.",

        "budget_intelligence": "🎯 బడ్జెట్ నిర్వహణ",
        "monthly_budget": "నెలవారీ బడ్జెట్‌ను సెట్ చేయండి (₹)",
        "save_budget": "💾 బడ్జెట్‌ను సేవ్ చేయండి",
        "budget_saved": "✅ నెలవారీ బడ్జెట్ సెట్ చేయబడింది:",

        "budget": "💰 బడ్జెట్",
        "budget_used": "📊 ఉపయోగించిన బడ్జెట్",
        "transactions": "🧾 లావాదేవీలు",
        "average": "📌 సగటు లావాదేవీ",
        "total_spending": "💰 మొత్తం ఖర్చు",

        "spending_analysis": "📊 ఖర్చు విశ్లేషణ",
        "spending_by_category": "💰 వర్గం వారీగా ఖర్చు",
        "spending_distribution": "📈 ఖర్చు పంపిణీ",
        "where_money_goes": "మీ డబ్బు ఎక్కడ ఖర్చవుతోంది",
        "spending_over_time": "📈 కాలక్రమంలో ఖర్చు",

        "no_expenses_analysis": "ఖర్చు విశ్లేషణ చూడటానికి ఖర్చులను జోడించండి.",

        "expense_history": "🧾 ఖర్చు చరిత్ర",
        "no_transactions": "ఇంకా ఎలాంటి లావాదేవీలు జోడించలేదు.",

        "edit_expense": "✏️ ఖర్చును సవరించండి",
        "select_expense": "ఖర్చు IDని ఎంచుకోండి",
        "edit_description": "వివరణ",
        "edit_amount": "మొత్తం (₹)",
        "edit_category": "వర్గం",
        "update_expense": "💾 ఖర్చును నవీకరించండి",
        "expense_updated": "✅ ఖర్చు విజయవంతంగా నవీకరించబడింది!",

        "delete_transaction": "🗑️ లావాదేవీని తొలగించండి",
        "enter_id": "లావాదేవీ IDని నమోదు చేయండి",
        "delete": "🗑️ తొలగించండి",
        "transaction_deleted": "లావాదేవీ తొలగించబడింది.",

        "finance_qa": "💬 FinSight AIని అడగండి",
        "ask_question": "మీ ఖర్చు గురించి ప్రశ్న అడగండి",
        "clear_question": "ప్రశ్నను క్లియర్ చేయండి",
        "ask_ai": "🤖 FinSight AIని అడగండి",
        "question_first": "ముందుగా ఒక ప్రశ్నను నమోదు చేయండి.",

        "translation_unavailable": "అనువాదం ప్రస్తుతం అందుబాటులో లేదు.",

        "monthly_analysis": "📅 నెలవారీ ఖర్చు విశ్లేషణ",
        "select_month": "నెలను ఎంచుకోండి",

        "budget_vs_actual": "🎯 బడ్జెట్ వర్సెస్ వాస్తవ ఖర్చు",
        "set_budget_comparison": "బడ్జెట్ పోలిక చూడటానికి నెలవారీ బడ్జెట్‌ను సెట్ చేయండి.",
        "add_expenses_comparison": "బడ్జెట్ పోలిక చూడటానికి ఖర్చులను జోడించండి.",
        "actual_spending": "💸 వాస్తవ ఖర్చు",
        "remaining": "💰 మిగిలిన మొత్తం",
        "over_budget": "⚠️ బడ్జెట్ మించిపోయింది",
        "budget_used_text": "ఉపయోగించిన బడ్జెట్",

        "comfortably_within": "🟢 మీ ఖర్చు బడ్జెట్‌లోపలే ఉంది.",
        "approaching_budget": "🟡 మీరు నెలవారీ బడ్జెట్‌కు దగ్గరగా ఉన్నారు.",
        "very_close": "🟠 మీరు బడ్జెట్ పరిమితికి చాలా దగ్గరగా ఉన్నారు.",
        "exceeded": "🔴 మీరు నెలవారీ బడ్జెట్‌ను మించిపోయారు.",

        "language": "🌐 భాష",
        "choose_language": "🌐 మీ భాషను ఎంచుకోండి",
        "continue": "🚀 కొనసాగించండి",

        "welcome_title": "💰 FinSight AI",
        "welcome_subtitle": "🤖 AI ఆధారిత వ్యక్తిగత ఆర్థిక సహాయకుడు",
        "welcome_text": "మీ ఖర్చులను నిర్వహించండి, మీ ఖర్చులను అర్థం చేసుకోండి మరియు మెరుగైన ఆర్థిక నిర్ణయాలు తీసుకోండి.",
        "select_language_text": "మీరు ఉపయోగించాలనుకునే భాషను ఎంచుకోండి",

        "no_valid_dates": "ట్రెండ్ చార్ట్ కోసం చెల్లుబాటు అయ్యే తేదీలు లేవు.",
        "no_expenses_edit": "సవరించడానికి ఖర్చులు లేవు.",
        "no_expenses_budget": "బడ్జెట్ నిర్వహణను ప్రారంభించడానికి నెలవారీ బడ్జెట్‌ను సెట్ చేయండి.",

        "receipt": "📷 రసీదు స్కానర్",
        "upload_receipt": "మీ రసీదు లేదా బిల్లును అప్‌లోడ్ చేయండి",
        "uploaded_receipt": "అప్‌లోడ్ చేసిన రసీదు",
        "scan_receipt": "🔍 రసీదును స్కాన్ చేయండి",
        "scanning_receipt": "రసీదును స్కాన్ చేస్తున్నాము...",
        "receipt_scanned": "✅ రసీదు విజయవంతంగా స్కాన్ చేయబడింది!",
        "receipt_details": "📄 తీసుకున్న రసీదు వివరాలు",
        "receipt_description": "వివరణ",
        "receipt_amount": "మొత్తం",
        "save_scanned_expense": "💾 స్కాన్ చేసిన ఖర్చును సేవ్ చేయండి",
        "receipt_amount_not_found": "⚠️ ఈ రసీదు నుంచి మొత్తాన్ని గుర్తించలేకపోయాము. ఖర్చును మాన్యువల్‌గా జోడించండి.",
        "view_ocr_text": "🔎 తీసుకున్న రసీదు టెక్స్ట్‌ను చూడండి",
        "receipt_scan_failed": "రసీదు స్కాన్ విఫలమైంది:",

        "cancel": "రద్దు చేయండి",
        "yes_delete": "అవును, తొలగించండి",

        "footer": "FinSight AI • వ్యక్తిగత ఆర్థిక విశ్లేషణ మరియు ML వర్గీకరణ",

        "high_concentration": "⚠️ అధిక ఖర్చు కేంద్రీకరణ",
        "large_transaction": "🚨 పెద్ద లావాదేవీ గుర్తించబడింది",
        "recommendation": "💡 AI సిఫార్సు",

        "personalized_recommendations": "🧠 వ్యక్తిగతీకరించిన ఆర్థిక సిఫార్సులు",

        "recommendations_description": "FinSight AI మీ ఖర్చులు, బడ్జెట్, అసాధారణ లావాదేవీలు మరియు అంచనాలను విశ్లేషించి వ్యక్తిగతీకరించిన సిఫార్సులను అందిస్తుంది."
    },


    # ========================================================
    # 🇮🇳 MALAYALAM
    # ========================================================

    "ml": {

        "app_subtitle": "🤖 AI ഉപയോഗിച്ചുള്ള വ്യക്തിഗത ധനകാര്യ ട്രാക്കർ",
        "app_description": "നിങ്ങളുടെ ചെലവുകൾ ട്രാക്ക് ചെയ്യുക, ഇടപാടുകൾ സ്വയമേവ വിഭാഗീകരിക്കുക, നിങ്ങളുടെ പണം എവിടേക്ക് പോകുന്നു എന്ന് മനസ്സിലാക്കുക.",

        "add_transaction": "➕ ഇടപാട് ചേർക്കുക",
        "description": "ഇടപാട് വിവരണം",
        "amount": "തുക (₹)",
        "predict": "🤖 വിഭാഗം പ്രവചിക്കുക",
        "add_expense": "💾 ചെലവ് സംരക്ഷിക്കുക",

        "ai_predicted": "🤖 AI പ്രവചിച്ച വിഭാഗം",
        "category_selected": "FinSight AI തിരഞ്ഞെടുത്ത വിഭാഗം",

        "empty_description": "ദയവായി ഇടപാട് വിവരണം നൽകുക.",
        "invalid_amount": "₹0-ൽ കൂടുതലുള്ള തുക നൽകുക.",
        "expense_added": "✅ ചെലവ് വിജയകരമായി ചേർത്തു. വിഭാഗം:",

        "ai_review": "🧠 AI സാമ്പത്തിക അവലോകനം",
        "ai_review_empty": "ആദ്യം ചില ചെലവുകൾ ചേർക്കുക. FinSight AI നിങ്ങളുടെ ചെലവുകൾ വിശകലനം ചെയ്യും.",

        "budget_intelligence": "🎯 ബജറ്റ് മാനേജ്മെന്റ്",
        "monthly_budget": "പ്രതിമാസ ബജറ്റ് സജ്ജമാക്കുക (₹)",
        "save_budget": "💾 ബജറ്റ് സംരക്ഷിക്കുക",
        "budget_saved": "✅ പ്രതിമാസ ബജറ്റ് സജ്ജമാക്കി:",

        "budget": "💰 ബജറ്റ്",
        "budget_used": "📊 ഉപയോഗിച്ച ബജറ്റ്",
        "transactions": "🧾 ഇടപാടുകൾ",
        "average": "📌 ശരാശരി ഇടപാട്",
        "total_spending": "💰 ആകെ ചെലവ്",

        "spending_analysis": "📊 ചെലവ് വിശകലനം",
        "spending_by_category": "💰 വിഭാഗം അനുസരിച്ചുള്ള ചെലവ്",
        "spending_distribution": "📈 ചെലവ് വിതരണം",
        "where_money_goes": "നിങ്ങളുടെ പണം എവിടേക്ക് പോകുന്നു",
        "spending_over_time": "📈 കാലക്രമത്തിലുള്ള ചെലവ്",

        "no_expenses_analysis": "ചെലവ് വിശകലനം കാണാൻ ചെലവുകൾ ചേർക്കുക.",

        "expense_history": "🧾 ചെലവ് ചരിത്രം",
        "no_transactions": "ഇതുവരെ ഇടപാടുകളൊന്നും ചേർത്തിട്ടില്ല.",

        "edit_expense": "✏️ ചെലവ് തിരുത്തുക",
        "select_expense": "ചെലവ് ID തിരഞ്ഞെടുക്കുക",
        "edit_description": "വിവരണം",
        "edit_amount": "തുക",
        "edit_category": "വിഭാഗം",
        "update_expense": "💾 ചെലവ് പുതുക്കുക",
        "expense_updated": "✅ ചെലവ് വിജയകരമായി പുതുക്കി!",

        "delete_transaction": "🗑️ ഇടപാട് ഇല്ലാതാക്കുക",
        "enter_id": "ഇടപാട് ID നൽകുക",
        "delete": "🗑️ ഇല്ലാതാക്കുക",
        "transaction_deleted": "ഇടപാട് ഇല്ലാതാക്കി.",

        "finance_qa": "💬 FinSight AI-യോട് ചോദിക്കുക",
        "ask_question": "നിങ്ങളുടെ ചെലവിനെക്കുറിച്ച് ചോദിക്കുക",
        "clear_question": "ചോദ്യം മായ്ക്കുക",
        "ask_ai": "🤖 FinSight AI-യോട് ചോദിക്കുക",
        "question_first": "ആദ്യം ഒരു ചോദ്യം നൽകുക.",

        "translation_unavailable": "വിവർത്തനം ഇപ്പോൾ ലഭ്യമല്ല.",

        "monthly_analysis": "📅 പ്രതിമാസ ചെലവ് വിശകലനം",
        "select_month": "മാസം തിരഞ്ഞെടുക്കുക",

        "budget_vs_actual": "🎯 ബജറ്റും യഥാർത്ഥ ചെലവും",
        "set_budget_comparison": "ബജറ്റ് താരതമ്യം കാണാൻ പ്രതിമാസ ബജറ്റ് സജ്ജമാക്കുക.",
        "add_expenses_comparison": "ബജറ്റ് താരതമ്യം കാണാൻ ചെലവുകൾ ചേർക്കുക.",
        "actual_spending": "💸 യഥാർത്ഥ ചെലവ്",
        "remaining": "💰 ശേഷിക്കുന്നത്",
        "over_budget": "⚠️ ബജറ്റ് കവിഞ്ഞു",
        "budget_used_text": "ഉപയോഗിച്ച ബജറ്റ്",

        "comfortably_within": "🟢 നിങ്ങളുടെ ചെലവ് ബജറ്റിനുള്ളിലാണ്.",
        "approaching_budget": "🟡 നിങ്ങൾ പ്രതിമാസ ബജറ്റിനോട് അടുക്കുകയാണ്.",
        "very_close": "🟠 നിങ്ങൾ ബജറ്റ് പരിധിക്ക് വളരെ അടുത്താണ്.",
        "exceeded": "🔴 നിങ്ങൾ പ്രതിമാസ ബജറ്റ് കവിഞ്ഞു.",

        "language": "🌐 ഭാഷ",
        "choose_language": "🌐 നിങ്ങളുടെ ഭാഷ തിരഞ്ഞെടുക്കുക",
        "continue": "🚀 തുടരുക",

        "welcome_title": "💰 FinSight AI",
        "welcome_subtitle": "🤖 AI അടിസ്ഥാനമാക്കിയ വ്യക്തിഗത സാമ്പത്തിക സഹായി",
        "welcome_text": "നിങ്ങളുടെ ചെലവുകൾ നിയന്ത്രിക്കുക, ചെലവുകൾ മനസ്സിലാക്കുക, മികച്ച സാമ്പത്തിക തീരുമാനങ്ങൾ എടുക്കുക.",
        "select_language_text": "നിങ്ങൾ ഉപയോഗിക്കാൻ ആഗ്രഹിക്കുന്ന ഭാഷ തിരഞ്ഞെടുക്കുക",

        "no_valid_dates": "ട്രെൻഡ് ചാർട്ടിനായി സാധുവായ തീയതികൾ ലഭ്യമല്ല.",
        "no_expenses_edit": "തിരുത്താൻ ചെലവുകളൊന്നുമില്ല.",
        "no_expenses_budget": "ബജറ്റ് മാനേജ്മെന്റ് സജീവമാക്കാൻ പ്രതിമാസ ബജറ്റ് സജ്ജമാക്കുക.",

        "receipt": "📷 രസീത് സ്കാനർ",
        "upload_receipt": "നിങ്ങളുടെ രസീത് അല്ലെങ്കിൽ ബിൽ അപ്‌ലോഡ് ചെയ്യുക",
        "uploaded_receipt": "അപ്‌ലോഡ് ചെയ്ത രസീത്",
        "scan_receipt": "🔍 രസീത് സ്കാൻ ചെയ്യുക",
        "scanning_receipt": "രസീത് സ്കാൻ ചെയ്യുന്നു...",
        "receipt_scanned": "✅ രസീത് വിജയകരമായി സ്കാൻ ചെയ്തു!",
        "receipt_details": "📄 ലഭിച്ച രസീത് വിശദാംശങ്ങൾ",
        "receipt_description": "വിവരണം",
        "receipt_amount": "തുക",
        "save_scanned_expense": "💾 സ്കാൻ ചെയ്ത ചെലവ് സംരക്ഷിക്കുക",
        "receipt_amount_not_found": "⚠️ രസീതിൽ നിന്ന് തുക കണ്ടെത്താനായില്ല. ചെലവ് സ്വമേധയാ ചേർക്കുക.",
        "view_ocr_text": "🔎 ലഭിച്ച രസീത് ടെക്സ്റ്റ് കാണുക",
        "receipt_scan_failed": "രസീത് സ്കാൻ പരാജയപ്പെട്ടു:",

        "cancel": "റദ്ദാക്കുക",
        "yes_delete": "അതെ, ഇല്ലാതാക്കുക",

        "footer": "FinSight AI • വ്യക്തിഗത സാമ്പത്തിക വിശകലനവും ML വിഭാഗീകരണവും",

        "high_concentration": "⚠️ ഉയർന്ന ചെലവ് കേന്ദ്രീകരണം",
        "large_transaction": "🚨 വലിയ ഇടപാട് കണ്ടെത്തി",
        "recommendation": "💡 AI ശുപാർശ",

        "personalized_recommendations": "🧠 വ്യക്തിഗത സാമ്പത്തിക ശുപാർശകൾ",

        "recommendations_description": "FinSight AI നിങ്ങളുടെ ചെലവുകൾ, ബജറ്റ്, അസാധാരണ ഇടപാടുകൾ, പ്രവചനം എന്നിവ വിശകലനം ചെയ്ത് വ്യക്തിഗത ശുപാർശകൾ നൽകുന്നു."
    },
# ========================================================
    # 🇮🇳 KANNADA
    # ========================================================

    "kn": {

        "app_subtitle": "🤖 AI ಆಧಾರಿತ ವೈಯಕ್ತಿಕ ಹಣಕಾಸು ಟ್ರ್ಯಾಕರ್",
        "app_description": "ನಿಮ್ಮ ಖರ್ಚುಗಳನ್ನು ಟ್ರ್ಯಾಕ್ ಮಾಡಿ, ವ್ಯವಹಾರಗಳನ್ನು ಸ್ವಯಂಚಾಲಿತವಾಗಿ ವರ್ಗೀಕರಿಸಿ ಮತ್ತು ನಿಮ್ಮ ಹಣ ಎಲ್ಲಿ ಹೋಗುತ್ತಿದೆ ಎಂಬುದನ್ನು ಅರ್ಥಮಾಡಿಕೊಳ್ಳಿ.",

        "add_transaction": "➕ ವ್ಯವಹಾರವನ್ನು ಸೇರಿಸಿ",
        "description": "ವ್ಯವಹಾರದ ವಿವರಣೆ",
        "amount": "ಮೊತ್ತ (₹)",
        "predict": "🤖 ವರ್ಗವನ್ನು ಊಹಿಸಿ",
        "add_expense": "💾 ಖರ್ಚನ್ನು ಉಳಿಸಿ",

        "ai_predicted": "🤖 AI ಊಹಿಸಿದ ವರ್ಗ",
        "category_selected": "FinSight AI ಆಯ್ಕೆ ಮಾಡಿದ ವರ್ಗ",

        "empty_description": "ದಯವಿಟ್ಟು ವ್ಯವಹಾರದ ವಿವರಣೆಯನ್ನು ನಮೂದಿಸಿ.",
        "invalid_amount": "₹0 ಕ್ಕಿಂತ ಹೆಚ್ಚಿನ ಮೊತ್ತವನ್ನು ನಮೂದಿಸಿ.",
        "expense_added": "✅ ಖರ್ಚು ಯಶಸ್ವಿಯಾಗಿ ಸೇರಿಸಲಾಗಿದೆ. ವರ್ಗ:",

        "ai_review": "🧠 AI ಹಣಕಾಸು ವಿಮರ್ಶೆ",
        "ai_review_empty": "ಮೊದಲು ಕೆಲವು ಖರ್ಚುಗಳನ್ನು ಸೇರಿಸಿ. FinSight AI ನಿಮ್ಮ ಖರ್ಚುಗಳನ್ನು ವಿಶ್ಲೇಷಿಸುತ್ತದೆ.",

        "budget_intelligence": "🎯 ಬಜೆಟ್ ನಿರ್ವಹಣೆ",
        "monthly_budget": "ಮಾಸಿಕ ಬಜೆಟ್ ಹೊಂದಿಸಿ (₹)",
        "save_budget": "💾 ಬಜೆಟ್ ಉಳಿಸಿ",
        "budget_saved": "✅ ಮಾಸಿಕ ಬಜೆಟ್ ಹೊಂದಿಸಲಾಗಿದೆ:",

        "budget": "💰 ಬಜೆಟ್",
        "budget_used": "📊 ಬಳಸಿದ ಬಜೆಟ್",
        "transactions": "🧾 ವ್ಯವಹಾರಗಳು",
        "average": "📌 ಸರಾಸರಿ ವ್ಯವಹಾರ",
        "total_spending": "💰 ಒಟ್ಟು ಖರ್ಚು",

        "spending_analysis": "📊 ಖರ್ಚಿನ ವಿಶ್ಲೇಷಣೆ",
        "spending_by_category": "💰 ವರ್ಗದ ಪ್ರಕಾರ ಖರ್ಚು",
        "spending_distribution": "📈 ಖರ್ಚಿನ ವಿತರಣೆ",
        "where_money_goes": "ನಿಮ್ಮ ಹಣ ಎಲ್ಲಿ ಹೋಗುತ್ತದೆ",
        "spending_over_time": "📈 ಕಾಲಾನುಗತ ಖರ್ಚು",

        "no_expenses_analysis": "ಖರ್ಚಿನ ವಿಶ್ಲೇಷಣೆ ನೋಡಲು ಖರ್ಚುಗಳನ್ನು ಸೇರಿಸಿ.",

        "expense_history": "🧾 ಖರ್ಚಿನ ಇತಿಹಾಸ",
        "no_transactions": "ಇನ್ನೂ ಯಾವುದೇ ವ್ಯವಹಾರಗಳನ್ನು ಸೇರಿಸಲಾಗಿಲ್ಲ.",

        "edit_expense": "✏️ ಖರ್ಚನ್ನು ಸಂಪಾದಿಸಿ",
        "select_expense": "ಖರ್ಚಿನ ID ಆಯ್ಕೆಮಾಡಿ",
        "edit_description": "ವಿವರಣೆ",
        "edit_amount": "ಮೊತ್ತ (₹)",
        "edit_category": "ವರ್ಗ",
        "update_expense": "💾 ಖರ್ಚನ್ನು ನವೀಕರಿಸಿ",
        "expense_updated": "✅ ಖರ್ಚು ಯಶಸ್ವಿಯಾಗಿ ನವೀಕರಿಸಲಾಗಿದೆ!",

        "delete_transaction": "🗑️ ವ್ಯವಹಾರವನ್ನು ಅಳಿಸಿ",
        "enter_id": "ವ್ಯವಹಾರ ID ನಮೂದಿಸಿ",
        "delete": "🗑️ ಅಳಿಸಿ",
        "transaction_deleted": "ವ್ಯವಹಾರವನ್ನು ಅಳಿಸಲಾಗಿದೆ.",

        "finance_qa": "💬 FinSight AI ಅನ್ನು ಕೇಳಿ",
        "ask_question": "ನಿಮ್ಮ ಖರ್ಚಿನ ಬಗ್ಗೆ ಪ್ರಶ್ನೆ ಕೇಳಿ",
        "clear_question": "ಪ್ರಶ್ನೆಯನ್ನು ತೆರವುಗೊಳಿಸಿ",
        "ask_ai": "🤖 FinSight AI ಅನ್ನು ಕೇಳಿ",
        "question_first": "ಮೊದಲು ಪ್ರಶ್ನೆಯನ್ನು ನಮೂದಿಸಿ.",

        "translation_unavailable": "ಅನುವಾದವು ಪ್ರಸ್ತುತ ಲಭ್ಯವಿಲ್ಲ.",

        "monthly_analysis": "📅 ಮಾಸಿಕ ಖರ್ಚಿನ ವಿಶ್ಲೇಷಣೆ",
        "select_month": "ತಿಂಗಳನ್ನು ಆಯ್ಕೆಮಾಡಿ",

        "budget_vs_actual": "🎯 ಬಜೆಟ್ ಮತ್ತು ನಿಜವಾದ ಖರ್ಚು",
        "set_budget_comparison": "ಬಜೆಟ್ ಹೋಲಿಕೆ ನೋಡಲು ಮಾಸಿಕ ಬಜೆಟ್ ಹೊಂದಿಸಿ.",
        "add_expenses_comparison": "ಬಜೆಟ್ ಹೋಲಿಕೆ ನೋಡಲು ಖರ್ಚುಗಳನ್ನು ಸೇರಿಸಿ.",
        "actual_spending": "💸 ನಿಜವಾದ ಖರ್ಚು",
        "remaining": "💰 ಉಳಿದದ್ದು",
        "over_budget": "⚠️ ಬಜೆಟ್ ಮೀರಿದೆ",
        "budget_used_text": "ಬಳಸಿದ ಬಜೆಟ್",

        "comfortably_within": "🟢 ನಿಮ್ಮ ಖರ್ಚು ಬಜೆಟ್‌ನೊಳಗಿದೆ.",
        "approaching_budget": "🟡 ನೀವು ಮಾಸಿಕ ಬಜೆಟ್‌ಗೆ ಹತ್ತಿರವಾಗುತ್ತಿದ್ದೀರಿ.",
        "very_close": "🟠 ನೀವು ಬಜೆಟ್ ಮಿತಿಗೆ ತುಂಬಾ ಹತ್ತಿರದಲ್ಲಿದ್ದೀರಿ.",
        "exceeded": "🔴 ನೀವು ಮಾಸಿಕ ಬಜೆಟ್ ಮೀರಿದ್ದೀರಿ.",

        "language": "🌐 ಭಾಷೆ",
        "choose_language": "🌐 ನಿಮ್ಮ ಭಾಷೆಯನ್ನು ಆಯ್ಕೆಮಾಡಿ",
        "continue": "🚀 ಮುಂದುವರಿಸಿ",

        "welcome_title": "💰 FinSight AI",
        "welcome_subtitle": "🤖 AI ಆಧಾರಿತ ವೈಯಕ್ತಿಕ ಹಣಕಾಸು ಸಹಾಯಕ",
        "welcome_text": "ನಿಮ್ಮ ಖರ್ಚುಗಳನ್ನು ನಿರ್ವಹಿಸಿ, ಖರ್ಚುಗಳನ್ನು ಅರ್ಥಮಾಡಿಕೊಳ್ಳಿ ಮತ್ತು ಉತ್ತಮ ಹಣಕಾಸು ನಿರ್ಧಾರಗಳನ್ನು ತೆಗೆದುಕೊಳ್ಳಿ.",
        "select_language_text": "ನೀವು ಬಳಸಲು ಬಯಸುವ ಭಾಷೆಯನ್ನು ಆಯ್ಕೆಮಾಡಿ",

        "no_valid_dates": "ಟ್ರೆಂಡ್ ಚಾರ್ಟ್‌ಗೆ ಮಾನ್ಯ ದಿನಾಂಕಗಳು ಲಭ್ಯವಿಲ್ಲ.",
        "no_expenses_edit": "ಸಂಪಾದಿಸಲು ಯಾವುದೇ ಖರ್ಚುಗಳಿಲ್ಲ.",
        "no_expenses_budget": "ಬಜೆಟ್ ನಿರ್ವಹಣೆಯನ್ನು ಸಕ್ರಿಯಗೊಳಿಸಲು ಮಾಸಿಕ ಬಜೆಟ್ ಹೊಂದಿಸಿ.",

        "receipt": "📷 ರಸೀದಿ ಸ್ಕ್ಯಾನರ್",
        "upload_receipt": "ನಿಮ್ಮ ರಸೀದಿ ಅಥವಾ ಬಿಲ್ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ",
        "uploaded_receipt": "ಅಪ್‌ಲೋಡ್ ಮಾಡಿದ ರಸೀದಿ",
        "scan_receipt": "🔍 ರಸೀದಿಯನ್ನು ಸ್ಕ್ಯಾನ್ ಮಾಡಿ",
        "scanning_receipt": "ರಸೀದಿಯನ್ನು ಸ್ಕ್ಯಾನ್ ಮಾಡಲಾಗುತ್ತಿದೆ...",
        "receipt_scanned": "✅ ರಸೀದಿ ಯಶಸ್ವಿಯಾಗಿ ಸ್ಕ್ಯಾನ್ ಮಾಡಲಾಗಿದೆ!",
        "receipt_details": "📄 ಹೊರತೆಗೆಯಲಾದ ರಸೀದಿ ವಿವರಗಳು",
        "receipt_description": "ವಿವರಣೆ",
        "receipt_amount": "ಮೊತ್ತ",
        "save_scanned_expense": "💾 ಸ್ಕ್ಯಾನ್ ಮಾಡಿದ ಖರ್ಚನ್ನು ಉಳಿಸಿ",
        "receipt_amount_not_found": "⚠️ ರಸೀದಿಯಿಂದ ಮೊತ್ತವನ್ನು ಪತ್ತೆಹಚ್ಚಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಖರ್ಚನ್ನು ಕೈಯಾರೆ ಸೇರಿಸಿ.",
        "view_ocr_text": "🔎 ಹೊರತೆಗೆಯಲಾದ ರಸೀದಿ ಪಠ್ಯವನ್ನು ವೀಕ್ಷಿಸಿ",
        "receipt_scan_failed": "ರಸೀದಿ ಸ್ಕ್ಯಾನ್ ವಿಫಲವಾಗಿದೆ:",

        "cancel": "ರದ್ದುಮಾಡಿ",
        "yes_delete": "ಹೌದು, ಅಳಿಸಿ",

        "footer": "FinSight AI • ವೈಯಕ್ತಿಕ ಹಣಕಾಸು ವಿಶ್ಲೇಷಣೆ ಮತ್ತು ML ವರ್ಗೀಕರಣ",

        "high_concentration": "⚠️ ಹೆಚ್ಚಿನ ಖರ್ಚಿನ ಕೇಂದ್ರೀಕರಣ",
        "large_transaction": "🚨 ದೊಡ್ಡ ವ್ಯವಹಾರ ಪತ್ತೆಯಾಗಿದೆ",
        "recommendation": "💡 AI ಶಿಫಾರಸು",

        "personalized_recommendations": "🧠 ವೈಯಕ್ತಿಕ ಹಣಕಾಸು ಶಿಫಾರಸುಗಳು",

        "recommendations_description": "FinSight AI ನಿಮ್ಮ ಖರ್ಚು, ಬಜೆಟ್, ಅಸಾಮಾನ್ಯ ವ್ಯವಹಾರಗಳು ಮತ್ತು ಮುನ್ಸೂಚನೆಯನ್ನು ವಿಶ್ಲೇಷಿಸಿ ವೈಯಕ್ತಿಕ ಶಿಫಾರಸುಗಳನ್ನು ನೀಡುತ್ತದೆ."
    }
} 
def t(key):
    lang = st.session_state.get(
        "selected_language_code",
        "en"
    )

    english = UI_TRANSLATIONS.get(
        "en",
        {}
    ).get(
        key,
        key
    )

    return UI_TRANSLATIONS.get(
        lang,
        {}
    ).get(
        key,
        english
    )
CATEGORIES = [
    "Food",
    "Travel",
    "Shopping",
    "Rent",
    "Utilities",
    "Education",
    "Entertainment",
    "Health",
    "Savings",
    "Investment",
    "Salary",
    "Freelance",
    "Bonus",
    "Others"
]


@st.cache_data(show_spinner=False)
def translate_dynamic_text(text, target_language):
    """Best-effort fallback for free-form text. Never displays HTTP errors."""
    if text is None or not str(text).strip() or target_language == "en":
        return text
    try:
        result = GoogleTranslator(source="auto", target=target_language).translate(str(text))
        if result and not str(result).lower().startswith("error "):
            return result
    except Exception:
        pass
    return text
# ============================================================
# PERSONALIZED RECOMMENDATION TRANSLATION
# ============================================================

def translate_recommendation(title, message, target_language):
    """
    Translate built-in FinSight AI recommendations while
    preserving dynamic values such as amounts, percentages,
    categories, anomaly counts, and descriptions.
    """

    if target_language == "en":
        return title, message

    # --------------------------------------------------------
    # Extract dynamic values from the original English message
    # --------------------------------------------------------

    def extract(pattern, default=""):
        match = re.search(pattern, message)
        return match.group(1) if match else default

    # --------------------------------------------------------
    # Dynamic values
    # --------------------------------------------------------

    budget_amount = extract(r"₹([\d,]+\.\d{2}) budget")
    exceeded_amount = extract(r"by ₹([\d,]+\.\d{2})")
    percentage = extract(r"used ([\d.]+)%")
    category = extract(r"^(.+?) accounts for")
    category_percentage = extract(r"at ([\d.]+)% of total")
    anomaly_count = extract(r"detected (\d+) unusual")
    transaction_amount = extract(r"Your ₹([\d,]+\.\d{2}) transaction")
    transaction_description = extract(
        r"transaction \((.*?)\) is more"
    )
    forecast_amount = extract(
        r"projected spending of ₹([\d,]+\.\d{2})"
    )
    remaining_budget = extract(
        r"currently have ₹([\d,]+\.\d{2})"
    )

    # --------------------------------------------------------
    # Translation templates
    # --------------------------------------------------------

    translations = {

        "ta": {

            "Budget Exceeded": (
                "பட்ஜெட் மீறப்பட்டது",
                f"நீங்கள் உங்கள் ₹{budget_amount} பட்ஜெட்டை "
                f"₹{exceeded_amount} மீறியுள்ளீர்கள். "
                "அத்தியாவசியமற்ற செலவுகளை குறைக்கவும்."
            ),

            "Budget Almost Used": (
                "பட்ஜெட் கிட்டத்தட்ட பயன்படுத்தப்பட்டது",
                f"உங்கள் மாதாந்திர பட்ஜெட்டில் {percentage}% "
                "பயன்படுத்தப்பட்டுள்ளது. அத்தியாவசியமற்ற "
                "செலவுகளை கட்டுப்படுத்துங்கள்."
            ),

            "Watch Your Budget": (
                "உங்கள் பட்ஜெட்டை கவனியுங்கள்",
                f"உங்கள் மாதாந்திர பட்ஜெட்டில் {percentage}% "
                "ஏற்கனவே பயன்படுத்தப்பட்டுள்ளது. மீதமுள்ள "
                "செலவுகளை கவனமாக கண்காணியுங்கள்."
            ),

            "Budget On Track": (
                "பட்ஜெட் சரியான நிலையில் உள்ளது",
                f"உங்கள் மாதாந்திர பட்ஜெட்டில் {percentage}% "
                "பயன்படுத்தப்பட்டுள்ளது. உங்கள் தற்போதைய செலவு "
                "சரியான நிலையில் உள்ளது."
            ),

            "High Category Concentration": (
                "ஒரு வகையில் அதிக செலவு",
                f"{category} உங்கள் மொத்த செலவில் "
                f"{category_percentage}% கொண்டுள்ளது. "
                "இந்த வகையின் செலவுகளை கவனமாக மதிப்பாய்வு செய்யுங்கள்."
            ),

            "Largest Spending Category": (
                "அதிக செலவு வகை",
                f"{category} உங்கள் மொத்த செலவில் "
                f"{category_percentage}% உடன் அதிகம் செலவாகும் வகையாகும். "
                "இந்த வகையை தொடர்ந்து கண்காணியுங்கள்."
            ),

            "Unusual Spending Detected": (
                "வழக்கத்திற்கு மாறான செலவு கண்டறியப்பட்டது",
                f"FinSight AI {anomaly_count} வழக்கத்திற்கு மாறான "
                "பரிவர்த்தனைகளை கண்டறிந்துள்ளது. அவை திட்டமிட்ட "
                "செலவுகளா என்பதை சரிபார்க்கவும்."
            ),

            "Large Transaction": (
                "பெரிய பரிவர்த்தனை",
                f"உங்கள் ₹{transaction_amount} பரிவர்த்தனை "
                f"({transaction_description}) உங்கள் சராசரி "
                "பரிவர்த்தனையை விட மூன்று மடங்கிற்கும் அதிகமாக உள்ளது. "
                "இந்த செலவை மதிப்பாய்வு செய்யுங்கள்."
            ),

            "Forecast Above Budget": (
                "முன்னறிவிப்பு பட்ஜெட்டை விட அதிகம்",
                f"உங்கள் எதிர்பார்க்கப்படும் செலவு ₹{forecast_amount} "
                f"உங்கள் ₹{budget_amount} பட்ஜெட்டை விட அதிகமாக உள்ளது. "
                "அத்தியாவசியமற்ற செலவுகளை குறைக்கவும்."
            ),

            "Forecast Near Budget": (
                "முன்னறிவிப்பு பட்ஜெட்டிற்கு அருகில் உள்ளது",
                f"உங்கள் எதிர்பார்க்கப்படும் செலவு ₹{forecast_amount} "
                "உங்கள் மாதாந்திர பட்ஜெட்டிற்கு அருகில் உள்ளது. "
                "வரவிருக்கும் செலவுகளை கண்காணியுங்கள்."
            ),

            "Savings Opportunity": (
                "சேமிப்பு வாய்ப்பு",
                f"உங்கள் பட்ஜெட்டில் ₹{remaining_budget} மீதமுள்ளது. "
                "அதில் ஒரு பகுதியை சேமிப்பு அல்லது முதலீட்டிற்கு "
                "ஒதுக்கலாம்."
            ),

            "Keep Monitoring": (
                "தொடர்ந்து கண்காணியுங்கள்",
                "தற்போதைய செலவு முறையில் பெரிய எச்சரிக்கைகள் இல்லை. "
                "உங்கள் செலவுகளை தொடர்ந்து கண்காணியுங்கள்."
            ),
        },

        "hi": {

            "Budget Exceeded": (
                "बजट पार हो गया",
                f"आपने अपना ₹{budget_amount} बजट "
                f"₹{exceeded_amount} पार कर लिया है। "
                "गैर-जरूरी खर्चों को कम करने पर विचार करें।"
            ),

            "Budget Almost Used": (
                "बजट लगभग पूरा हो गया",
                f"आपके मासिक बजट का {percentage}% उपयोग हो चुका है। "
                "गैर-जरूरी खर्चों को सीमित करने पर विचार करें।"
            ),

            "Watch Your Budget": (
                "अपने बजट पर ध्यान दें",
                f"आपके मासिक बजट का {percentage}% पहले ही उपयोग हो चुका है। "
                "बाकी खर्चों पर नजर रखें।"
            ),

            "Budget On Track": (
                "बजट नियंत्रण में है",
                f"आपके मासिक बजट का {percentage}% उपयोग हुआ है। "
                "आपका वर्तमान खर्च बजट के भीतर है।"
            ),

            "High Category Concentration": (
                "एक श्रेणी में अधिक खर्च",
                f"{category} आपके कुल खर्च का {category_percentage}% "
                "है। इस श्रेणी के खर्चों की समीक्षा करें।"
            ),

            "Largest Spending Category": (
                "सबसे अधिक खर्च वाली श्रेणी",
                f"{category} आपके कुल खर्च का {category_percentage}% "
                "है और आपकी सबसे बड़ी खर्च श्रेणी है। "
                "इस श्रेणी की नियमित निगरानी करें।"
            ),

            "Unusual Spending Detected": (
                "असामान्य खर्च पाया गया",
                f"FinSight AI ने {anomaly_count} असामान्य लेनदेन पाए हैं। "
                "जांचें कि ये खर्च जानबूझकर किए गए थे।"
            ),

            "Large Transaction": (
                "बड़ा लेनदेन",
                f"आपका ₹{transaction_amount} लेनदेन "
                f"({transaction_description}) आपके औसत लेनदेन से "
                "तीन गुना से अधिक है। इस खर्च की समीक्षा करें।"
            ),

            "Forecast Above Budget": (
                "पूर्वानुमान बजट से अधिक है",
                f"आपका अनुमानित खर्च ₹{forecast_amount} आपके "
                f"₹{budget_amount} बजट से अधिक है। "
                "गैर-जरूरी खर्चों को कम करने पर विचार करें।"
            ),

            "Forecast Near Budget": (
                "पूर्वानुमान बजट के करीब है",
                f"आपका अनुमानित खर्च ₹{forecast_amount} "
                "आपके मासिक बजट के करीब है। "
                "आने वाले खर्चों पर नजर रखें।"
            ),

            "Savings Opportunity": (
                "बचत का अवसर",
                f"आपके बजट में ₹{remaining_budget} अभी बचा है। "
                "इसका कुछ हिस्सा बचत या निवेश के लिए अलग रखने पर "
                "विचार करें।"
            ),

            "Keep Monitoring": (
                "निगरानी जारी रखें",
                "वर्तमान खर्च पैटर्न में कोई बड़ी चेतावनी नहीं है। "
                "अपने खर्चों की नियमित निगरानी जारी रखें।"
            ),
        },

        "te": {

            "Budget Exceeded": (
                "బడ్జెట్ మించిపోయింది",
                f"మీరు మీ ₹{budget_amount} బడ్జెట్‌ను "
                f"₹{exceeded_amount} మించిపోయారు. "
                "అవసరం లేని ఖర్చులను తగ్గించండి."
            ),

            "Budget Almost Used": (
                "బడ్జెట్ దాదాపు పూర్తైంది",
                f"మీ నెలవారీ బడ్జెట్‌లో {percentage}% ఉపయోగించబడింది. "
                "అవసరం లేని ఖర్చులను తగ్గించండి."
            ),

            "Watch Your Budget": (
                "మీ బడ్జెట్‌ను గమనించండి",
                f"మీ నెలవారీ బడ్జెట్‌లో {percentage}% ఇప్పటికే "
                "ఉపయోగించబడింది. మిగిలిన ఖర్చులను జాగ్రత్తగా గమనించండి."
            ),

            "Budget On Track": (
                "బడ్జెట్ నియంత్రణలో ఉంది",
                f"మీ నెలవారీ బడ్జెట్‌లో {percentage}% ఉపయోగించబడింది. "
                "మీ ప్రస్తుత ఖర్చు బడ్జెట్‌లోనే ఉంది."
            ),

            "High Category Concentration": (
                "ఒక వర్గంలో అధిక ఖర్చు",
                f"{category} మీ మొత్తం ఖర్చులో {category_percentage}% "
                "కలిగి ఉంది. ఈ వర్గం ఖర్చులను సమీక్షించండి."
            ),

            "Largest Spending Category": (
                "అత్యధిక ఖర్చు వర్గం",
                f"{category} మీ మొత్తం ఖర్చులో {category_percentage}% "
                "తో అత్యధిక ఖర్చు వర్గంగా ఉంది. దీన్ని క్రమం తప్పకుండా గమనించండి."
            ),

            "Unusual Spending Detected": (
                "అసాధారణ ఖర్చు గుర్తించబడింది",
                f"FinSight AI {anomaly_count} అసాధారణ లావాదేవీలను "
                "గుర్తించింది. అవి ఉద్దేశపూర్వక ఖర్చులా అని తనిఖీ చేయండి."
            ),

            "Large Transaction": (
                "పెద్ద లావాదేవీ",
                f"మీ ₹{transaction_amount} లావాదేవీ "
                f"({transaction_description}) మీ సగటు లావాదేవీ కంటే "
                "మూడు రెట్లకు పైగా ఉంది. ఈ ఖర్చును సమీక్షించండి."
            ),

            "Forecast Above Budget": (
                "అంచనా బడ్జెట్ కంటే ఎక్కువగా ఉంది",
                f"మీ అంచనా ఖర్చు ₹{forecast_amount} మీ "
                f"₹{budget_amount} బడ్జెట్ కంటే ఎక్కువగా ఉంది. "
                "అవసరం లేని ఖర్చులను తగ్గించండి."
            ),

            "Forecast Near Budget": (
                "అంచనా బడ్జెట్‌కు దగ్గరగా ఉంది",
                f"మీ అంచనా ఖర్చు ₹{forecast_amount} "
                "మీ నెలవారీ బడ్జెట్‌కు దగ్గరగా ఉంది. "
                "రాబోయే ఖర్చులను గమనించండి."
            ),

            "Savings Opportunity": (
                "పొదుపు అవకాశం",
                f"మీ బడ్జెట్‌లో ₹{remaining_budget} ఇంకా మిగిలి ఉంది. "
                "దానిలో కొంత భాగాన్ని పొదుపు లేదా పెట్టుబడికి కేటాయించండి."
            ),

            "Keep Monitoring": (
                "పర్యవేక్షణ కొనసాగించండి",
                "ప్రస్తుత ఖర్చు విధానంలో పెద్ద హెచ్చరికలు లేవు. "
                "మీ ఖర్చులను క్రమం తప్పకుండా గమనించండి."
            ),
        },

        "ml": {

            "Budget Exceeded": (
                "ബജറ്റ് കവിഞ്ഞു",
                f"നിങ്ങളുടെ ₹{budget_amount} ബജറ്റ് "
                f"₹{exceeded_amount} കവിഞ്ഞിരിക്കുന്നു. "
                "അനാവശ്യ ചെലവുകൾ കുറയ്ക്കുക."
            ),

            "Budget Almost Used": (
                "ബജറ്റ് ഏകദേശം ഉപയോഗിച്ചു",
                f"നിങ്ങളുടെ മാസ ബജറ്റിന്റെ {percentage}% ഉപയോഗിച്ചു കഴിഞ്ഞു. "
                "അനാവശ്യ ചെലവുകൾ നിയന്ത്രിക്കുക."
            ),

            "Watch Your Budget": (
                "നിങ്ങളുടെ ബജറ്റ് ശ്രദ്ധിക്കുക",
                f"നിങ്ങളുടെ മാസ ബജറ്റിന്റെ {percentage}% ഇതിനകം ഉപയോഗിച്ചു. "
                "ശേഷിക്കുന്ന ചെലവുകൾ ശ്രദ്ധിക്കുക."
            ),

            "Budget On Track": (
                "ബജറ്റ് നിയന്ത്രണത്തിലാണ്",
                f"നിങ്ങളുടെ മാസ ബജറ്റിന്റെ {percentage}% ഉപയോഗിച്ചു. "
                "നിലവിലെ ചെലവ് ബജറ്റിനുള്ളിലാണ്."
            ),

            "High Category Concentration": (
                "ഒരു വിഭാഗത്തിൽ ഉയർന്ന ചെലവ്",
                f"{category} നിങ്ങളുടെ മൊത്തം ചെലവിന്റെ "
                f"{category_percentage}% ആണ്. ഈ വിഭാഗത്തിലെ ചെലവുകൾ പരിശോധിക്കുക."
            ),

            "Largest Spending Category": (
                "ഏറ്റവും കൂടുതൽ ചെലവുള്ള വിഭാഗം",
                f"{category} നിങ്ങളുടെ മൊത്തം ചെലവിന്റെ "
                f"{category_percentage}% ആണ്. ഈ വിഭാഗം നിരീക്ഷിക്കുക."
            ),

            "Unusual Spending Detected": (
                "അസാധാരണ ചെലവ് കണ്ടെത്തി",
                f"FinSight AI {anomaly_count} അസാധാരണ ഇടപാടുകൾ കണ്ടെത്തി. "
                "അവ ഉദ്ദേശിച്ച ചെലവുകളാണോ എന്ന് പരിശോധിക്കുക."
            ),

            "Large Transaction": (
                "വലിയ ഇടപാട്",
                f"നിങ്ങളുടെ ₹{transaction_amount} ഇടപാട് "
                f"({transaction_description}) ശരാശരി ഇടപാടിനേക്കാൾ "
                "മൂന്ന് മടങ്ങിലധികമാണ്. ഈ ചെലവ് പരിശോധിക്കുക."
            ),

            "Forecast Above Budget": (
                "പ്രവചനം ബജറ്റിനേക്കാൾ കൂടുതലാണ്",
                f"നിങ്ങളുടെ പ്രതീക്ഷിക്കുന്ന ചെലവ് ₹{forecast_amount} "
                f"നിങ്ങളുടെ ₹{budget_amount} ബജറ്റിനേക്കാൾ കൂടുതലാണ്. "
                "അനാവശ്യ ചെലവുകൾ കുറയ്ക്കുക."
            ),

            "Forecast Near Budget": (
                "പ്രവചനം ബജറ്റിനോട് അടുത്താണ്",
                f"നിങ്ങളുടെ പ്രതീക്ഷിക്കുന്ന ചെലവ് ₹{forecast_amount} "
                "മാസ ബജറ്റിനോട് അടുത്താണ്. വരാനിരിക്കുന്ന ചെലവുകൾ നിരീക്ഷിക്കുക."
            ),

            "Savings Opportunity": (
                "സേവിംഗ്സ് അവസരം",
                f"നിങ്ങളുടെ ബജറ്റിൽ ₹{remaining_budget} ബാക്കിയുണ്ട്. "
                "അതിന്റെ ഒരു ഭാഗം സേവിംഗ്സിനോ നിക്ഷേപത്തിനോ മാറ്റിവയ്ക്കാം."
            ),

            "Keep Monitoring": (
                "നിരീക്ഷണം തുടരുക",
                "നിലവിലെ ചെലവ് രീതിയിൽ വലിയ മുന്നറിയിപ്പുകളൊന്നുമില്ല. "
                "നിങ്ങളുടെ ചെലവുകൾ പതിവായി നിരീക്ഷിക്കുക."
            ),
        },

        "kn": {

            "Budget Exceeded": (
                "ಬಜೆಟ್ ಮೀರಿದೆ",
                f"ನೀವು ನಿಮ್ಮ ₹{budget_amount} ಬಜೆಟ್ ಅನ್ನು "
                f"₹{exceeded_amount} ಮೀರಿದ್ದೀರಿ. "
                "ಅಗತ್ಯವಿಲ್ಲದ ಖರ್ಚುಗಳನ್ನು ಕಡಿಮೆ ಮಾಡಿ."
            ),

            "Budget Almost Used": (
                "ಬಜೆಟ್ ಬಹುತೇಕ ಬಳಕೆಯಾಗಿದೆ",
                f"ನಿಮ್ಮ ಮಾಸಿಕ ಬಜೆಟ್‌ನ {percentage}% ಈಗಾಗಲೇ ಬಳಕೆಯಾಗಿದೆ. "
                "ಅಗತ್ಯವಿಲ್ಲದ ಖರ್ಚುಗಳನ್ನು ಕಡಿಮೆ ಮಾಡಿ."
            ),

            "Watch Your Budget": (
                "ನಿಮ್ಮ ಬಜೆಟ್ ಗಮನಿಸಿ",
                f"ನಿಮ್ಮ ಮಾಸಿಕ ಬಜೆಟ್‌ನ {percentage}% ಈಗಾಗಲೇ ಬಳಕೆಯಾಗಿದೆ. "
                "ಉಳಿದ ಖರ್ಚುಗಳನ್ನು ಗಮನಿಸಿ."
            ),

            "Budget On Track": (
                "ಬಜೆಟ್ ನಿಯಂತ್ರಣದಲ್ಲಿದೆ",
                f"ನಿಮ್ಮ ಮಾಸಿಕ ಬಜೆಟ್‌ನ {percentage}% ಬಳಕೆಯಾಗಿದೆ. "
                "ನಿಮ್ಮ ಪ್ರಸ್ತುತ ಖರ್ಚು ಬಜೆಟ್‌ನೊಳಗಿದೆ."
            ),

            "High Category Concentration": (
                "ಒಂದು ವರ್ಗದಲ್ಲಿ ಹೆಚ್ಚಿನ ಖರ್ಚು",
                f"{category} ನಿಮ್ಮ ಒಟ್ಟು ಖರ್ಚಿನ {category_percentage}% "
                "ಹೊಂದಿದೆ. ಈ ವರ್ಗದ ಖರ್ಚುಗಳನ್ನು ಪರಿಶೀಲಿಸಿ."
            ),

            "Largest Spending Category": (
                "ಅತಿ ಹೆಚ್ಚು ಖರ್ಚಿನ ವರ್ಗ",
                f"{category} ನಿಮ್ಮ ಒಟ್ಟು ಖರ್ಚಿನ {category_percentage}% "
                "ಹೊಂದಿದ್ದು, ನಿಮ್ಮ ಅತಿ ಹೆಚ್ಚು ಖರ್ಚಿನ ವರ್ಗವಾಗಿದೆ."
            ),

            "Unusual Spending Detected": (
                "ಅಸಾಮಾನ್ಯ ಖರ್ಚು ಪತ್ತೆಯಾಗಿದೆ",
                f"FinSight AI {anomaly_count} ಅಸಾಮಾನ್ಯ ವಹಿವಾಟುಗಳನ್ನು "
                "ಪತ್ತೆಹಚ್ಚಿದೆ. ಅವು ಉದ್ದೇಶಪೂರ್ವಕ ಖರ್ಚುಗಳೇ ಎಂದು ಪರಿಶೀಲಿಸಿ."
            ),

            "Large Transaction": (
                "ದೊಡ್ಡ ವಹಿವಾಟು",
                f"ನಿಮ್ಮ ₹{transaction_amount} ವಹಿವಾಟು "
                f"({transaction_description}) ನಿಮ್ಮ ಸರಾಸರಿ ವಹಿವಾಟಿಗಿಂತ "
                "ಮೂರು ಪಟ್ಟು ಹೆಚ್ಚು ಇದೆ. ಈ ಖರ್ಚನ್ನು ಪರಿಶೀಲಿಸಿ."
            ),

            "Forecast Above Budget": (
                "ಮುನ್ಸೂಚನೆ ಬಜೆಟ್‌ಗಿಂತ ಹೆಚ್ಚಾಗಿದೆ",
                f"ನಿಮ್ಮ ನಿರೀಕ್ಷಿತ ಖರ್ಚು ₹{forecast_amount} ನಿಮ್ಮ "
                f"₹{budget_amount} ಬಜೆಟ್‌ಗಿಂತ ಹೆಚ್ಚಾಗಿದೆ. "
                "ಅಗತ್ಯವಿಲ್ಲದ ಖರ್ಚುಗಳನ್ನು ಕಡಿಮೆ ಮಾಡಿ."
            ),

            "Forecast Near Budget": (
                "ಮುನ್ಸೂಚನೆ ಬಜೆಟ್‌ಗೆ ಹತ್ತಿರದಲ್ಲಿದೆ",
                f"ನಿಮ್ಮ ನಿರೀಕ್ಷಿತ ಖರ್ಚು ₹{forecast_amount} "
                "ನಿಮ್ಮ ಮಾಸಿಕ ಬಜೆಟ್‌ಗೆ ಹತ್ತಿರದಲ್ಲಿದೆ. "
                "ಮುಂಬರುವ ಖರ್ಚುಗಳನ್ನು ಗಮನಿಸಿ."
            ),

            "Savings Opportunity": (
                "ಉಳಿತಾಯದ ಅವಕಾಶ",
                f"ನಿಮ್ಮ ಬಜೆಟ್‌ನಲ್ಲಿ ₹{remaining_budget} ಇನ್ನೂ ಉಳಿದಿದೆ. "
                "ಅದರ ಒಂದು ಭಾಗವನ್ನು ಉಳಿತಾಯ ಅಥವಾ ಹೂಡಿಕೆಗೆ ಮೀಸಲಿಡಿ."
            ),

            "Keep Monitoring": (
                "ಗಮನಿಸುವುದನ್ನು ಮುಂದುವರಿಸಿ",
                "ಪ್ರಸ್ತುತ ಖರ್ಚಿನ ಮಾದರಿಯಲ್ಲಿ ಯಾವುದೇ ಪ್ರಮುಖ ಎಚ್ಚರಿಕೆಗಳಿಲ್ಲ. "
                "ನಿಮ್ಮ ಖರ್ಚುಗಳನ್ನು ನಿಯಮಿತವಾಗಿ ಗಮನಿಸಿ."
            ),
        }
    }

    language_map = translations.get(target_language)

    if not language_map:
        return title, message

    return language_map.get(
        title,
        (title, message)
    )

    translations = {

        "ta": {
            "Budget Exceeded": (
                "பட்ஜெட் மீறப்பட்டது",
                "நீங்கள் உங்கள் பட்ஜெட்டை மீறியுள்ளீர்கள்."
            ),
            "Budget Almost Used": (
                "பட்ஜெட் கிட்டத்தட்ட பயன்படுத்தப்பட்டது",
                "உங்கள் மாதாந்திர பட்ஜெட்டின் பெரும்பகுதி பயன்படுத்தப்பட்டுள்ளது. "
                "அத்தியாவசியமற்ற செலவுகளை கட்டுப்படுத்துங்கள்."
            ),
            "Watch Your Budget": (
                "உங்கள் பட்ஜெட்டை கவனியுங்கள்",
                "உங்கள் மாதாந்திர பட்ஜெட்டின் பெரும்பகுதி ஏற்கனவே பயன்படுத்தப்பட்டுள்ளது. "
                "மீதமுள்ள செலவுகளை கவனமாக கண்காணியுங்கள்."
            ),
            "Budget On Track": (
                "பட்ஜெட் சரியான நிலையில் உள்ளது",
                "உங்கள் தற்போதைய செலவுகள் உங்கள் பட்ஜெட்டிற்குள் உள்ளன."
            ),
            "High Category Concentration": (
                "ஒரு வகையில் அதிக செலவு",
                "இந்த வகை உங்கள் மொத்த செலவில் பெரிய பங்கை கொண்டுள்ளது. "
                "இந்த செலவுகளை கவனமாக மதிப்பாய்வு செய்யுங்கள்."
            ),
            "Largest Spending Category": (
                "அதிக செலவு வகை",
                "இது உங்கள் அதிக செலவு செய்யும் வகையாகும். "
                "இந்த வகையை தொடர்ந்து கண்காணியுங்கள்."
            ),
            "Unusual Spending Detected": (
                "வழக்கத்திற்கு மாறான செலவு கண்டறியப்பட்டது",
                "வழக்கத்திற்கு மாறான பரிவர்த்தனைகள் கண்டறியப்பட்டுள்ளன. "
                "அவை திட்டமிட்ட செலவுகளா என்பதை சரிபார்க்கவும்."
            ),
            "Large Transaction": (
                "பெரிய பரிவர்த்தனை",
                "இந்த பரிவர்த்தனை உங்கள் சராசரி பரிவர்த்தனையை விட அதிகமாக உள்ளது. "
                "இந்த செலவை மதிப்பாய்வு செய்யுங்கள்."
            ),
            "Forecast Above Budget": (
                "முன்னறிவிப்பு பட்ஜெட்டை விட அதிகம்",
                "உங்கள் எதிர்பார்க்கப்படும் செலவு உங்கள் மாதாந்திர பட்ஜெட்டை விட அதிகமாக உள்ளது. "
                "அத்தியாவசியமற்ற செலவுகளை குறைக்கவும்."
            ),
            "Forecast Near Budget": (
                "முன்னறிவிப்பு பட்ஜெட்டிற்கு அருகில் உள்ளது",
                "உங்கள் எதிர்பார்க்கப்படும் செலவு உங்கள் மாதாந்திர பட்ஜெட்டிற்கு அருகில் உள்ளது. "
                "வரவிருக்கும் செலவுகளை கண்காணியுங்கள்."
            ),
            "Savings Opportunity": (
                "சேமிப்பு வாய்ப்பு",
                "உங்கள் பட்ஜெட்டில் இன்னும் தொகை மீதமுள்ளது. "
                "அதில் ஒரு பகுதியை சேமிப்பு அல்லது முதலீட்டிற்கு ஒதுக்கலாம்."
            ),
            "Keep Monitoring": (
                "தொடர்ந்து கண்காணியுங்கள்",
                "தற்போதைய செலவு முறையில் பெரிய எச்சரிக்கைகள் இல்லை. "
                "உங்கள் செலவுகளை தொடர்ந்து கண்காணியுங்கள்."
            )
        },

        "hi": {
            "Budget Exceeded": (
                "बजट पार हो गया",
                "आपने अपना बजट पार कर लिया है।"
            ),
            "Budget Almost Used": (
                "बजट लगभग पूरा हो गया",
                "आपके मासिक बजट का अधिकांश हिस्सा उपयोग हो चुका है। "
                "गैर-जरूरी खर्चों को सीमित करने पर विचार करें।"
            ),
            "Watch Your Budget": (
                "अपने बजट पर ध्यान दें",
                "आपके मासिक बजट का एक बड़ा हिस्सा पहले ही उपयोग हो चुका है। "
                "बाकी खर्चों पर नजर रखें।"
            ),
            "Budget On Track": (
                "बजट नियंत्रण में है",
                "आपका वर्तमान खर्च आपके बजट के भीतर है।"
            ),
            "High Category Concentration": (
                "एक श्रेणी में अधिक खर्च",
                "यह श्रेणी आपके कुल खर्च का बड़ा हिस्सा है। "
                "इस श्रेणी के खर्चों की समीक्षा करें।"
            ),
            "Largest Spending Category": (
                "सबसे अधिक खर्च वाली श्रेणी",
                "यह आपकी सबसे अधिक खर्च वाली श्रेणी है। "
                "इस श्रेणी की नियमित निगरानी करें।"
            ),
            "Unusual Spending Detected": (
                "असामान्य खर्च पाया गया",
                "कुछ असामान्य लेनदेन पाए गए हैं। "
                "जांचें कि ये खर्च जानबूझकर किए गए थे।"
            ),
            "Large Transaction": (
                "बड़ा लेनदेन",
                "यह लेनदेन आपके औसत लेनदेन से काफी अधिक है। "
                "इस खर्च की समीक्षा करें।"
            ),
            "Forecast Above Budget": (
                "पूर्वानुमान बजट से अधिक है",
                "आपका अनुमानित खर्च आपके मासिक बजट से अधिक है। "
                "गैर-जरूरी खर्चों को कम करने पर विचार करें।"
            ),
            "Forecast Near Budget": (
                "पूर्वानुमान बजट के करीब है",
                "आपका अनुमानित खर्च आपके मासिक बजट के करीब है। "
                "आने वाले खर्चों पर नजर रखें।"
            ),
            "Savings Opportunity": (
                "बचत का अवसर",
                "आपके बजट में अभी कुछ राशि बची हुई है। "
                "उसका कुछ हिस्सा बचत या निवेश के लिए अलग रखने पर विचार करें।"
            ),
            "Keep Monitoring": (
                "निगरानी जारी रखें",
                "वर्तमान खर्च पैटर्न में कोई बड़ी चेतावनी नहीं है। "
                "अपने खर्चों की नियमित निगरानी जारी रखें।"
            )
        },

        "te": {
            "Budget Exceeded": (
                "బడ్జెట్ మించిపోయింది",
                "మీరు మీ బడ్జెట్‌ను మించిపోయారు."
            ),
            "Budget Almost Used": (
                "బడ్జెట్ దాదాపు పూర్తైంది",
                "మీ నెలవారీ బడ్జెట్‌లో ఎక్కువ భాగం ఇప్పటికే ఉపయోగించబడింది. "
                "అవసరం లేని ఖర్చులను తగ్గించండి."
            ),
            "Watch Your Budget": (
                "మీ బడ్జెట్‌ను గమనించండి",
                "మీ నెలవారీ బడ్జెట్‌లో ఎక్కువ భాగం ఇప్పటికే ఉపయోగించబడింది. "
                "మిగిలిన ఖర్చులను జాగ్రత్తగా గమనించండి."
            ),
            "Budget On Track": (
                "బడ్జెట్ నియంత్రణలో ఉంది",
                "మీ ప్రస్తుత ఖర్చులు మీ బడ్జెట్‌లోనే ఉన్నాయి."
            ),
            "High Category Concentration": (
                "ఒక వర్గంలో అధిక ఖర్చు",
                "ఈ వర్గం మీ మొత్తం ఖర్చులో పెద్ద భాగాన్ని కలిగి ఉంది. "
                "ఈ ఖర్చులను సమీక్షించండి."
            ),
            "Largest Spending Category": (
                "అత్యధిక ఖర్చు వర్గం",
                "ఇది మీ అత్యధిక ఖర్చు వర్గం. "
                "ఈ వర్గాన్ని క్రమం తప్పకుండా గమనించండి."
            ),
            "Unusual Spending Detected": (
                "అసాధారణ ఖర్చు గుర్తించబడింది",
                "కొన్ని అసాధారణ లావాదేవీలు గుర్తించబడ్డాయి. "
                "అవి ఉద్దేశపూర్వక ఖర్చులా అని తనిఖీ చేయండి."
            ),
            "Large Transaction": (
                "పెద్ద లావాదేవీ",
                "ఈ లావాదేవీ మీ సగటు లావాదేవీ కంటే చాలా ఎక్కువగా ఉంది. "
                "ఈ ఖర్చును సమీక్షించండి."
            ),
            "Forecast Above Budget": (
                "అంచనా బడ్జెట్ కంటే ఎక్కువగా ఉంది",
                "మీ అంచనా ఖర్చు మీ నెలవారీ బడ్జెట్ కంటే ఎక్కువగా ఉంది. "
                "అవసరం లేని ఖర్చులను తగ్గించండి."
            ),
            "Forecast Near Budget": (
                "అంచనా బడ్జెట్‌కు దగ్గరగా ఉంది",
                "మీ అంచనా ఖర్చు మీ నెలవారీ బడ్జெட்டுக்கு దగ్గరగా உள்ளது. "
                "రాబోయే ఖర్చులను గమనించండి."
            ),
            "Savings Opportunity": (
                "పొదుపు అవకాశం",
                "మీ బడ్జెట్‌లో కొంత మొత్తం ఇంకా మిగిలి ఉంది. "
                "దానిలో కొంత భాగాన్ని పొదుపు లేదా పెట్టుబడికి కేటాయించండి."
            ),
            "Keep Monitoring": (
                "పర్యవేక్షణ కొనసాగించండి",
                "ప్రస్తుత ఖర్చు విధానంలో పెద్ద హెచ్చరికలు లేవు. "
                "మీ ఖర్చులను క్రమం తప్పకుండా గమనించండి."
            )
        },

        "ml": {
            "Budget Exceeded": (
                "ബജറ്റ് കവിഞ്ഞു",
                "നിങ്ങളുടെ ബജറ്റ് കവിഞ്ഞിരിക്കുന്നു."
            ),
            "Budget Almost Used": (
                "ബജറ്റ് ഏകദേശം ഉപയോഗിച്ചു",
                "നിങ്ങളുടെ മാസ ബജറ്റിന്റെ ഭൂരിഭാഗവും ഉപയോഗിച്ചു കഴിഞ്ഞു. "
                "അനാവശ്യ ചെലവുകൾ നിയന്ത്രിക്കുന്നത് പരിഗണിക്കുക."
            ),
            "Watch Your Budget": (
                "നിങ്ങളുടെ ബജറ്റ് ശ്രദ്ധിക്കുക",
                "നിങ്ങളുടെ മാസ ബജറ്റിന്റെ വലിയൊരു ഭാഗം ഇതിനകം ഉപയോഗിച്ചു. "
                "ശേഷിക്കുന്ന ചെലവുകൾ ശ്രദ്ധിക്കുക."
            ),
            "Budget On Track": (
                "ബജറ്റ് നിയന്ത്രണത്തിലാണ്",
                "നിങ്ങളുടെ നിലവിലെ ചെലവ് ബജറ്റിനുള്ളിലാണ്."
            ),
            "High Category Concentration": (
                "ഒരു വിഭാഗത്തിൽ ഉയർന്ന ചെലവ്",
                "ഈ വിഭാഗം നിങ്ങളുടെ മൊത്തം ചെലവിന്റെ വലിയൊരു ഭാഗമാണ്. "
                "ഈ ചെലവുകൾ പരിശോധിക്കുക."
            ),
            "Largest Spending Category": (
                "ഏറ്റവും കൂടുതൽ ചെലവുള്ള വിഭാഗം",
                "ഇതാണ് നിങ്ങളുടെ ഏറ്റവും കൂടുതൽ ചെലവുള്ള വിഭാഗം. "
                "ഈ വിഭാഗം പതിവായി നിരീക്ഷിക്കുക."
            ),
            "Unusual Spending Detected": (
                "അസാധാരണ ചെലവ് കണ്ടെത്തി",
                "അസാധാരണമായ ചില ഇടപാടുകൾ കണ്ടെത്തി. "
                "അവ ഉദ്ദേശിച്ച ചെലവുകളാണോ എന്ന് പരിശോധിക്കുക."
            ),
            "Large Transaction": (
                "വലിയ ഇടപാട്",
                "ഈ ഇടപാട് നിങ്ങളുടെ ശരാശരി ഇടപാടിനേക്കാൾ വളരെ കൂടുതലാണ്. "
                "ഈ ചെലവ് പരിശോധിക്കുക."
            ),
            "Forecast Above Budget": (
                "പ്രവചനം ബജറ്റിനേക്കാൾ കൂടുതലാണ്",
                "നിങ്ങളുടെ പ്രതീക്ഷിക്കുന്ന ചെലവ് മാസ ബജറ്റിനേക്കാൾ കൂടുതലാണ്. "
                "അനാവശ്യ ചെലവുകൾ കുറയ്ക്കുക."
            ),
            "Forecast Near Budget": (
                "പ്രവചനം ബജറ്റിനോട് അടുത്താണ്",
                "നിങ്ങളുടെ പ്രതീക്ഷിക്കുന്ന ചെലവ് മാസ ബജറ്റിനോട് അടുത്താണ്. "
                "വരാനിരിക്കുന്ന ചെലവുകൾ നിരീക്ഷിക്കുക."
            ),
            "Savings Opportunity": (
                "സേവിംഗ്സ് അവസരം",
                "നിങ്ങളുടെ ബജറ്റിൽ തുക ബാക്കിയുണ്ട്. "
                "അതിന്റെ ഒരു ഭാഗം സേവിംഗ്സിനോ നിക്ഷേപത്തിനോ മാറ്റിവയ്ക്കാം."
            ),
            "Keep Monitoring": (
                "നിരീക്ഷണം തുടരുക",
                "നിലവിലെ ചെലവ് രീതിയിൽ വലിയ മുന്നറിയിപ്പുകളൊന്നുമില്ല. "
                "നിങ്ങളുടെ ചെലവുകൾ പതിവായി നിരീക്ഷിക്കുക."
            )
        },

        "kn": {
            "Budget Exceeded": (
                "ಬಜೆಟ್ ಮೀರಿದೆ",
                "ನೀವು ನಿಮ್ಮ ಬಜೆಟ್ ಅನ್ನು ಮೀರಿದ್ದೀರಿ."
            ),
            "Budget Almost Used": (
                "ಬಜೆಟ್ ಬಹುತೇಕ ಬಳಕೆಯಾಗಿದೆ",
                "ನಿಮ್ಮ ಮಾಸಿಕ ಬಜೆಟ್‌ನ ಹೆಚ್ಚಿನ ಭಾಗವನ್ನು ಈಗಾಗಲೇ ಬಳಸಲಾಗಿದೆ. "
                "ಅಗತ್ಯವಿಲ್ಲದ ಖರ್ಚುಗಳನ್ನು ಕಡಿಮೆ ಮಾಡುವುದನ್ನು ಪರಿಗಣಿಸಿ."
            ),
            "Watch Your Budget": (
                "ನಿಮ್ಮ ಬಜೆಟ್ ಗಮನಿಸಿ",
                "ನಿಮ್ಮ ಮಾಸಿಕ ಬಜೆಟ್‌ನ ದೊಡ್ಡ ಭಾಗವನ್ನು ಈಗಾಗಲೇ ಬಳಸಲಾಗಿದೆ. "
                "ಉಳಿದ ಖರ್ಚುಗಳನ್ನು ಗಮನಿಸಿ."
            ),
            "Budget On Track": (
                "ಬಜೆಟ್ ನಿಯಂತ್ರಣದಲ್ಲಿದೆ",
                "ನಿಮ್ಮ ಪ್ರಸ್ತುತ ಖರ್ಚು ನಿಮ್ಮ ಬಜೆಟ್‌ನೊಳಗಿದೆ."
            ),
            "High Category Concentration": (
                "ಒಂದು ವರ್ಗದಲ್ಲಿ ಹೆಚ್ಚಿನ ಖರ್ಚು",
                "ಈ ವರ್ಗವು ನಿಮ್ಮ ಒಟ್ಟು ಖರ್ಚಿನ ದೊಡ್ಡ ಭಾಗವಾಗಿದೆ. "
                "ಈ ಖರ್ಚುಗಳನ್ನು ಪರಿಶೀಲಿಸಿ."
            ),
            "Largest Spending Category": (
                "ಅತಿ ಹೆಚ್ಚು ಖರ್ಚಿನ ವರ್ಗ",
                "ಇದು ನಿಮ್ಮ ಅತಿ ಹೆಚ್ಚು ಖರ್ಚಾಗುವ ವರ್ಗವಾಗಿದೆ. "
                "ಈ ವರ್ಗವನ್ನು ನಿಯಮಿತವಾಗಿ ಗಮನಿಸಿ."
            ),
            "Unusual Spending Detected": (
                "ಅಸಾಮಾನ್ಯ ಖರ್ಚು ಪತ್ತೆಯಾಗಿದೆ",
                "ಕೆಲವು ಅಸಾಮಾನ್ಯ ವಹಿವಾಟುಗಳು ಪತ್ತೆಯಾಗಿವೆ. "
                "ಅವು ಉದ್ದೇಶಪೂರ್ವಕ ಖರ್ಚುಗಳೇ ಎಂದು ಪರಿಶೀಲಿಸಿ."
            ),
            "Large Transaction": (
                "ದೊಡ್ಡ ವಹಿವಾಟು",
                "ಈ ವಹಿವಾಟು ನಿಮ್ಮ ಸರಾಸರಿ ವಹಿವಾಟಿಗಿಂತ ಹೆಚ್ಚು ಇದೆ. "
                "ಈ ಖರ್ಚನ್ನು ಪರಿಶೀಲಿಸಿ."
            ),
            "Forecast Above Budget": (
                "ಮುನ್ಸೂಚನೆ ಬಜೆಟ್‌ಗಿಂತ ಹೆಚ್ಚಾಗಿದೆ",
                "ನಿಮ್ಮ ನಿರೀಕ್ಷಿತ ಖರ್ಚು ನಿಮ್ಮ ಮಾಸಿಕ ಬಜೆಟ್‌ಗಿಂತ ಹೆಚ್ಚಾಗಿದೆ. "
                "ಅಗತ್ಯವಿಲ್ಲದ ಖರ್ಚುಗಳನ್ನು ಕಡಿಮೆ ಮಾಡಿ."
            ),
            "Forecast Near Budget": (
                "ಮುನ್ಸೂಚನೆ ಬಜೆಟ್‌ಗೆ ಹತ್ತಿರದಲ್ಲಿದೆ",
                "ನಿಮ್ಮ ನಿರೀಕ್ಷಿತ ಖರ್ಚು ನಿಮ್ಮ ಮಾಸಿಕ ಬಜೆಟ್‌ಗೆ ಹತ್ತಿರದಲ್ಲಿದೆ. "
                "ಮುಂಬರುವ ಖರ್ಚುಗಳನ್ನು ಗಮನಿಸಿ."
            ),
            "Savings Opportunity": (
                "ಉಳಿತಾಯದ ಅವಕಾಶ",
                "ನಿಮ್ಮ ಬಜೆಟ್‌ನಲ್ಲಿ ಇನ್ನೂ ಹಣ ಉಳಿದಿದೆ. "
                "ಅದರ ಒಂದು ಭಾಗವನ್ನು ಉಳಿತಾಯ ಅಥವಾ ಹೂಡಿಕೆಗೆ ಮೀಸಲಿಡಿ."
            ),
            "Keep Monitoring": (
                "ಗಮನಿಸುವುದನ್ನು ಮುಂದುವರಿಸಿ",
                "ಪ್ರಸ್ತುತ ಖರ್ಚಿನ ಮಾದರಿಯಲ್ಲಿ ಯಾವುದೇ ಪ್ರಮುಖ ಎಚ್ಚರಿಕೆಗಳಿಲ್ಲ. "
                "ನಿಮ್ಮ ಖರ್ಚುಗಳನ್ನು ನಿಯಮಿತವಾಗಿ ಗಮನಿಸಿ."
            )
        }
    }

    # English fallback for unsupported language
    language_map = translations.get(target_language)

    if not language_map:
        return title, message

    translated_title, translated_base = language_map.get(
        title,
        (title, message)
    )

    # Preserve the dynamic values already generated by the
    # recommendation engine.
    if title == "Budget Exceeded":
        translated_message = (
            f"நீங்கள் உங்கள் பட்ஜெட்டை மீறியுள்ளீர்கள்."
            if target_language == "ta"
            else f"आपने अपना बजट पार कर लिया है।"
            if target_language == "hi"
            else f"మీరు మీ బడ్జెట్‌ను మించిపోయారు."
            if target_language == "te"
            else f"നിങ്ങളുടെ ബജറ്റ് കവിഞ്ഞിരിക്കുന്നു."
            if target_language == "ml"
            else f"ನೀವು ನಿಮ್ಮ ಬಜೆಟ್ ಅನ್ನು ಮೀರಿದ್ದೀರಿ."
            if target_language == "kn"
            else message
        )
    else:
        translated_message = translated_base

    # The complete dynamic English message is still retained
    # for unsupported/unknown recommendation formats.
    return translated_title, translated_message

def t(key):
    lang = st.session_state.get(
        "selected_language_code",
        "en"
    )

    # Get English text as fallback
    english = UI_TRANSLATIONS.get("en", {}).get(key, key)

    # Return translated text if available
    return UI_TRANSLATIONS.get(lang, {}).get(
        key,
        english
    )
# ============================================================
# 🔮 SPENDING FORECASTING TRANSLATIONS
# ============================================================

FORECAST_TEXT = {
    "en": {
        "title": "🔮 AI Spending Forecast",
        "description": "A machine-learning model uses your historical monthly spending to estimate your next month's spending.",
        "need_more": "Add at least 4 months of spending history to generate a meaningful forecast.",
        "prediction": "Predicted Spending Next Month",
        "mae": "MAE",
        "rmse": "RMSE",
        "r2": "R² Score",
        "evaluation": "📊 Model Evaluation",
        "trend": "Recent Monthly Spending",
        "insight": "💡 Forecast Insight",
        "over_budget": "Your forecast is above your current monthly budget.",
        "within_budget": "Your forecast is within your current monthly budget.",
        "model": "Model: Automatic Forecast Selection • Statistical Forecast or Production LSTM based on available history",
    },
    "ta": {
        "title": "🔮 AI செலவு முன்னறிவிப்பு",
        "description": "உங்கள் கடந்த மாதங்களின் செலவு தரவைப் பயன்படுத்தி, அடுத்த மாத செலவை இயந்திரக் கற்றல் மாதிரி கணிக்கிறது.",
        "need_more": "நம்பகமான முன்னறிவிப்பை உருவாக்க குறைந்தது 4 மாத செலவு வரலாற்றைச் சேர்க்கவும்.",
        "prediction": "அடுத்த மாத கணிக்கப்பட்ட செலவு", "mae": "MAE", "rmse": "RMSE", "r2": "R² மதிப்பெண்",
        "evaluation": "📊 மாதிரி மதிப்பீடு", "trend": "சமீபத்திய மாத செலவு", "insight": "💡 முன்னறிவிப்பு தகவல்",
        "over_budget": "உங்கள் முன்னறிவிக்கப்பட்ட செலவு தற்போதைய மாதாந்திர பட்ஜெட்டை விட அதிகமாக உள்ளது.",
        "within_budget": "உங்கள் முன்னறிவிக்கப்பட்ட செலவு தற்போதைய மாதாந்திர பட்ஜெட்டுக்குள் உள்ளது.",
        "model": "மாதிரி: தானியங்கி முன்னறிவிப்பு தேர்வு • கிடைக்கும் வரலாற்றுத் தரவைப் பொறுத்து Statistical Forecast அல்லது Production LSTM",
    },
    "hi": {
        "title": "🔮 AI खर्च पूर्वानुमान", "description": "मशीन लर्निंग मॉडल आपके पिछले मासिक खर्च के आधार पर अगले महीने के खर्च का अनुमान लगाता है।",
        "need_more": "विश्वसनीय पूर्वानुमान के लिए कम से कम 4 महीनों का खर्च इतिहास जोड़ें।", "prediction": "अगले महीने का अनुमानित खर्च", "mae": "MAE", "rmse": "RMSE", "r2": "R² स्कोर", "evaluation": "📊 मॉडल मूल्यांकन", "trend": "हाल का मासिक खर्च", "insight": "💡 पूर्वानुमान जानकारी", "over_budget": "आपका अनुमानित खर्च वर्तमान मासिक बजट से अधिक है।", "within_budget": "आपका अनुमानित खर्च वर्तमान मासिक बजट के भीतर है।", "model": "मॉडल: स्वचालित पूर्वानुमान चयन • उपलब्ध इतिहास के आधार पर Statistical Forecast या Production LSTM"
    },
    "te": {
        "title": "🔮 AI ఖర్చు అంచనా", "description": "మెషిన్ లెర్నింగ్ మోడల్ మీ గత నెలల ఖర్చులను ఉపయోగించి వచ్చే నెల ఖర్చును అంచనా వేస్తుంది.", "need_more": "నమ్మకమైన అంచనా కోసం కనీసం 4 నెలల ఖర్చు చరిత్రను జోడించండి.", "prediction": "వచ్చే నెల అంచనా ఖర్చు", "mae": "MAE", "rmse": "RMSE", "r2": "R² స్కోర్", "evaluation": "📊 మోడల్ మూల్యాంకనం", "trend": "ఇటీవలి నెలవారీ ఖర్చు", "insight": "💡 అంచనా సమాచారం", "over_budget": "మీ అంచనా ఖర్చు ప్రస్తుత నెలవారీ బడ్జెట్ కంటే ఎక్కువగా ఉంది.", "within_budget": "మీ అంచనా ఖర్చు ప్రస్తుత నెలవారీ బడ్జెట్‌లో ఉంది.", "model": "మోడల్: ఆటోమేటిక్ ఫోర్‌కాస్ట్ ఎంపిక • అందుబాటులో ఉన్న చరిత్ర ఆధారంగా Statistical Forecast లేదా Production LSTM"
    },
    "ml": {
        "title": "🔮 AI ചെലവ് പ്രവചനം", "description": "മെഷീൻ ലേണിംഗ് മോഡൽ നിങ്ങളുടെ കഴിഞ്ഞ മാസങ്ങളിലെ ചെലവുകൾ ഉപയോഗിച്ച് അടുത്ത മാസത്തെ ചെലവ് കണക്കാക്കുന്നു.", "need_more": "അർത്ഥവത്തായ പ്രവചനത്തിന് കുറഞ്ഞത് 4 മാസത്തെ ചെലവ് ചരിത്രം ചേർക്കുക.", "prediction": "അടുത്ത മാസത്തെ പ്രവചിച്ച ചെലവ്", "mae": "MAE", "rmse": "RMSE", "r2": "R² സ്കോർ", "evaluation": "📊 മോഡൽ വിലയിരുത്തൽ", "trend": "സമീപകാല മാസ ചെലവ്", "insight": "💡 പ്രവചന വിവരം", "over_budget": "നിങ്ങളുടെ പ്രവചിച്ച ചെലവ് നിലവിലെ മാസ ബജറ്റിനേക്കാൾ കൂടുതലാണ്.", "within_budget": "നിങ്ങളുടെ പ്രവചിച്ച ചെലവ് നിലവിലെ മാസ ബജറ്റിനുള്ളിലാണ്.", "model": "മോഡൽ: ഓട്ടോമാറ്റിക് പ്രവചന തിരഞ്ഞെടുപ്പ് • ലഭ്യമായ ചരിത്രത്തെ അടിസ്ഥാനമാക്കി Statistical Forecast അല്ലെങ്കിൽ Production LSTM"
    },
    "kn": {
        "title": "🔮 AI ಖರ್ಚಿನ ಮುನ್ಸೂಚನೆ", "description": "ಮಷಿನ್ ಲರ್ನಿಂಗ್ ಮಾದರಿ ನಿಮ್ಮ ಹಿಂದಿನ ತಿಂಗಳ ಖರ್ಚನ್ನು ಬಳಸಿ ಮುಂದಿನ ತಿಂಗಳ ಖರ್ಚನ್ನು ಅಂದಾಜಿಸುತ್ತದೆ.", "need_more": "ಅರ್ಥಪೂರ್ಣ ಮುನ್ಸೂಚನೆಗಾಗಿ ಕನಿಷ್ಠ 4 ತಿಂಗಳ ಖರ್ಚಿನ ಇತಿಹಾಸವನ್ನು ಸೇರಿಸಿ.", "prediction": "ಮುಂದಿನ ತಿಂಗಳ ಅಂದಾಜು ಖರ್ಚು", "mae": "MAE", "rmse": "RMSE", "r2": "R² ಸ್ಕೋರ್", "evaluation": "📊 ಮಾದರಿ ಮೌಲ್ಯಮಾಪನ", "trend": "ಇತ್ತೀಚಿನ ಮಾಸಿಕ ಖರ್ಚು", "insight": "💡 ಮುನ್ಸೂಚನೆ ಮಾಹಿತಿ", "over_budget": "ನಿಮ್ಮ ಅಂದಾಜು ಖರ್ಚು ಪ್ರಸ್ತುತ ಮಾಸಿಕ ಬಜೆಟ್‌ಗಿಂತ ಹೆಚ್ಚಾಗಿದೆ.", "within_budget": "ನಿಮ್ಮ ಅಂದಾಜು ಖರ್ಚು ಪ್ರಸ್ತುತ ಮಾಸಿಕ ಬಜೆಟ್‌ನೊಳಗಿದೆ.", "model": "ಮಾದರಿ: ಸ್ವಯಂಚಾಲಿತ ಮುನ್ಸೂಚನೆ ಆಯ್ಕೆ • ಲಭ್ಯವಿರುವ ಇತಿಹಾಸದ ಆಧಾರದ ಮೇಲೆ Statistical Forecast ಅಥವಾ Production LSTM"
    }
}

def forecast_t(key):
    lang = st.session_state.get("selected_language_code", "en")
    return FORECAST_TEXT.get(lang, FORECAST_TEXT["en"]).get(key, FORECAST_TEXT["en"].get(key, key))


# ============================================================
# 🚨 SPENDING ANOMALY DETECTION TRANSLATIONS
# ============================================================

ANOMALY_TEXT = {
    "en": {
        "title": "🚨 AI Spending Anomaly Detection",
        "description": "Isolation Forest checks your transaction amounts and highlights spending that looks unusual compared with your normal pattern.",
        "need_more": "Add at least 5 valid transactions to activate anomaly detection.",
        "no_anomalies": "🟢 No unusual spending patterns were detected in your current transactions.",
        "detected": "🚨 Unusual spending detected",
        "transaction": "Transaction",
        "amount": "Amount",
        "category": "Category",
        "reason": "Why it was flagged",
        "reason_text": "This transaction is unusually different from the other amounts in your spending history.",
        "model": "Model: Isolation Forest • Feature: transaction amount",
        "tip": "💡 Tip: An anomaly is not automatically a bad expense. It simply means the amount is unusual and worth reviewing.",
    },
    "ta": {
        "title": "🚨 AI அசாதாரண செலவு கண்டறிதல்",
        "description": "Isolation Forest உங்கள் பரிவர்த்தனை தொகைகளை ஆய்வு செய்து, உங்கள் வழக்கமான செலவு முறையிலிருந்து மாறுபடும் செலவுகளை கண்டறிகிறது.",
        "need_more": "அசாதாரண செலவு கண்டறிதலை செயல்படுத்த குறைந்தது 5 சரியான பரிவர்த்தனைகளைச் சேர்க்கவும்.",
        "no_anomalies": "🟢 உங்கள் தற்போதைய பரிவர்த்தனைகளில் அசாதாரண செலவு கண்டறியப்படவில்லை.",
        "detected": "🚨 அசாதாரண செலவு கண்டறியப்பட்டது",
        "transaction": "பரிவர்த்தனை",
        "amount": "தொகை",
        "category": "வகை",
        "reason": "கண்டறியப்பட்ட காரணம்",
        "reason_text": "இந்த பரிவர்த்தனை உங்கள் மற்ற செலவு தொகைகளுடன் ஒப்பிடும்போது வழக்கத்திற்கு மாறாக உள்ளது.",
        "model": "மாதிரி: Isolation Forest • அம்சம்: பரிவர்த்தனை தொகை",
        "tip": "💡 குறிப்பு: அசாதாரணம் என்பது செலவு தவறானது என்று அர்த்தமல்ல. தொகை வழக்கத்திற்கு மாறாக இருப்பதால் அதை மதிப்பாய்வு செய்ய வேண்டும் என்பதே பொருள்.",
    },
    "hi": {
        "title": "🚨 AI असामान्य खर्च पहचान",
        "description": "Isolation Forest आपके लेन-देन की राशियों का विश्लेषण करता है और सामान्य खर्च पैटर्न से अलग दिखने वाले खर्चों को पहचानता है।",
        "need_more": "असामान्य खर्च पहचान सक्रिय करने के लिए कम से कम 5 मान्य लेन-देन जोड़ें।",
        "no_anomalies": "🟢 आपके वर्तमान लेन-देन में कोई असामान्य खर्च पैटर्न नहीं मिला।",
        "detected": "🚨 असामान्य खर्च मिला",
        "transaction": "लेन-देन",
        "amount": "राशि",
        "category": "श्रेणी",
        "reason": "पहचान का कारण",
        "reason_text": "यह लेन-देन आपके अन्य खर्चों की राशियों की तुलना में असामान्य है।",
        "model": "मॉडल: Isolation Forest • फीचर: लेन-देन राशि",
        "tip": "💡 सुझाव: असामान्य लेन-देन का मतलब यह नहीं है कि खर्च गलत है। इसका मतलब केवल यह है कि राशि असामान्य है और इसकी समीक्षा करना उचित है।",
    },
    "te": {
        "title": "🚨 AI అసాధారణ ఖర్చు గుర్తింపు",
        "description": "Isolation Forest మీ లావాదేవీల మొత్తాలను విశ్లేషించి, సాధారణ ఖర్చు విధానానికి భిన్నంగా ఉన్న ఖర్చులను గుర్తిస్తుంది.",
        "need_more": "అసాధారణ ఖర్చు గుర్తింపును ప్రారంభించడానికి కనీసం 5 చెల్లుబాటు అయ్యే లావాదేవీలను జోడించండి.",
        "no_anomalies": "🟢 మీ ప్రస్తుత లావాదేవీలలో అసాధారణ ఖర్చు నమూనాలు గుర్తించబడలేదు.",
        "detected": "🚨 అసాధారణ ఖర్చు గుర్తించబడింది",
        "transaction": "లావాదేవీ",
        "amount": "మొత్తం",
        "category": "వర్గం",
        "reason": "గుర్తించిన కారణం",
        "reason_text": "ఈ లావాదేవీ మీ ఇతర ఖర్చుల మొత్తాలతో పోలిస్తే అసాధారణంగా ఉంది.",
        "model": "మోడల్: Isolation Forest • ఫీచర్: లావాదేవీ మొత్తం",
        "tip": "💡 సూచన: అసాధారణ లావాదేవీ అంటే అది తప్పు ఖర్చు అని కాదు. మొత్తం సాధారణానికి భిన్నంగా ఉందని, కాబట్టి దాన్ని సమీక్షించాలి అని మాత్రమే అర్థం.",
    },
    "ml": {
        "title": "🚨 AI അസാധാരണ ചെലവ് കണ്ടെത്തൽ",
        "description": "Isolation Forest നിങ്ങളുടെ ഇടപാട് തുകകൾ പരിശോധിച്ച് സാധാരണ ചെലവ് രീതിയിൽ നിന്ന് വ്യത്യസ്തമായ ചെലവുകൾ കണ്ടെത്തുന്നു.",
        "need_more": "അസാധാരണ ചെലവ് കണ്ടെത്തൽ സജീവമാക്കാൻ കുറഞ്ഞത് 5 സാധുവായ ഇടപാടുകൾ ചേർക്കുക.",
        "no_anomalies": "🟢 നിലവിലെ ഇടപാടുകളിൽ അസാധാരണമായ ചെലവ് കണ്ടെത്തിയില്ല.",
        "detected": "🚨 അസാധാരണ ചെലവ് കണ്ടെത്തി",
        "transaction": "ഇടപാട്",
        "amount": "തുക",
        "category": "വിഭാഗം",
        "reason": "കണ്ടെത്തിയ കാരണം",
        "reason_text": "മറ്റ് ചെലവുകളുടെ തുകയുമായി താരതമ്യം ചെയ്യുമ്പോൾ ഈ ഇടപാട് അസാധാരണമാണ്.",
        "model": "മോഡൽ: Isolation Forest • ഫീച്ചർ: ഇടപാട് തുക",
        "tip": "💡 കുറിപ്പ്: അസാധാരണ ഇടപാട് എന്നത് തെറ്റായ ചെലവാണെന്ന് അർത്ഥമില്ല. തുക സാധാരണയിൽ നിന്ന് വ്യത്യസ്തമാണെന്നും അത് പരിശോധിക്കുന്നത് നല്ലതാണെന്നും മാത്രമാണ് അർത്ഥം.",
    },
    "kn": {
        "title": "🚨 AI ಅಸಾಮಾನ್ಯ ಖರ್ಚು ಪತ್ತೆ",
        "description": "Isolation Forest ನಿಮ್ಮ ವ್ಯವಹಾರಗಳ ಮೊತ್ತಗಳನ್ನು ಪರಿಶೀಲಿಸಿ, ಸಾಮಾನ್ಯ ಖರ್ಚಿನ ಮಾದರಿಯಿಂದ ಭಿನ್ನವಾಗಿರುವ ಖರ್ಚುಗಳನ್ನು ಗುರುತಿಸುತ್ತದೆ.",
        "need_more": "ಅಸಾಮಾನ್ಯ ಖರ್ಚು ಪತ್ತೆ ಸಕ್ರಿಯಗೊಳಿಸಲು ಕನಿಷ್ಠ 5 ಮಾನ್ಯ ವ್ಯವಹಾರಗಳನ್ನು ಸೇರಿಸಿ.",
        "no_anomalies": "🟢 ನಿಮ್ಮ ಪ್ರಸ್ತುತ ವ್ಯವಹಾರಗಳಲ್ಲಿ ಅಸಾಮಾನ್ಯ ಖರ್ಚಿನ ಮಾದರಿ ಕಂಡುಬಂದಿಲ್ಲ.",
        "detected": "🚨 ಅಸಾಮಾನ್ಯ ಖರ್ಚು ಪತ್ತೆಯಾಗಿದೆ",
        "transaction": "ವ್ಯವಹಾರ",
        "amount": "ಮೊತ್ತ",
        "category": "ವರ್ಗ",
        "reason": "ಪತ್ತೆಯಾದ ಕಾರಣ",
        "reason_text": "ನಿಮ್ಮ ಇತರ ಖರ್ಚಿನ ಮೊತ್ತಗಳಿಗೆ ಹೋಲಿಸಿದರೆ ಈ ವ್ಯವಹಾರವು ಅಸಾಮಾನ್ಯವಾಗಿದೆ.",
        "model": "ಮಾದರಿ: Isolation Forest • ವೈಶಿಷ್ಟ್ಯ: ವ್ಯವಹಾರದ ಮೊತ್ತ",
        "tip": "💡 ಸಲಹೆ: ಅಸಾಮಾನ್ಯ ವ್ಯವಹಾರ ಎಂದರೆ ಅದು ತಪ್ಪಾದ ಖರ್ಚು ಎಂದು ಅಲ್ಲ. ಮೊತ್ತವು ಸಾಮಾನ್ಯಕ್ಕಿಂತ ಭಿನ್ನವಾಗಿದೆ ಮತ್ತು ಅದನ್ನು ಪರಿಶೀಲಿಸುವುದು ಉತ್ತಮ ಎಂಬುದಷ್ಟೇ ಅರ್ಥ.",
    },
}


def anomaly_t(key):
    lang = st.session_state.get("selected_language_code", "en")
    return ANOMALY_TEXT.get(lang, ANOMALY_TEXT["en"]).get(
        key,
        ANOMALY_TEXT["en"].get(key, key)
    )


# ============================================================
# MULTILINGUAL AI FINANCIAL REVIEW
# ============================================================

def build_financial_review(df, lang):
    """Create the same data-driven financial review directly in the selected language.
    This avoids unreliable online translation for the most important dashboard section.
    """
    if df.empty:
        return t("ai_review_empty")

    work = df.copy()
    work["Amount"] = pd.to_numeric(work["Amount"], errors="coerce")
    work = work.dropna(subset=["Amount"])
    if work.empty:
        return t("ai_review_empty")

    total = float(work["Amount"].sum())
    count = int(len(work))
    average = total / count if count else 0.0
    category_totals = work.groupby("Category")["Amount"].sum().sort_values(ascending=False)
    top_category = str(category_totals.index[0]) if not category_totals.empty else "Others"
    top_amount = float(category_totals.iloc[0]) if not category_totals.empty else 0.0
    top_pct = (top_amount / total * 100) if total else 0.0
    largest = work.loc[work["Amount"].idxmax()]
    largest_desc = str(largest["Description"])
    largest_amount = float(largest["Amount"])

    # Category display names remain canonical because they are stored in the database/ML model.
    if lang == "ta":
        lines = [
            f"நீங்கள் ₹{total:,.2f} மதிப்புள்ள {count} பரிவர்த்தனைகளை பதிவு செய்துள்ளீர்கள்.",
            f"உங்கள் சராசரி பரிவர்த்தனை ₹{average:,.2f}.",
            f"உங்கள் அதிக செலவு செய்யப்பட்ட வகை {top_category}, இதில் ₹{top_amount:,.2f} செலவாகியுள்ளது.",
            f"இது உங்கள் மொத்த செலவில் சுமார் {top_pct:.1f}% ஆகும்.",
            f"உங்கள் மிகப்பெரிய தனிப்பட்ட பரிவர்த்தனை {largest_desc}, இதன் விலை ₹{largest_amount:,.2f}.",
            "",
            t("high_concentration"),
            f"உங்கள் செலவின் பெரிய பகுதி {top_category} வகையில் குவிந்துள்ளது.",
            "உங்கள் சமீபத்திய செலவுகளை மதிப்பாய்வு செய்து, குறைக்கக்கூடிய வாங்குதல்களை அடையாளம் காணுங்கள்.",
            "",
            t("large_transaction"),
            f"₹{largest_amount:,.2f} மதிப்புள்ள {largest_desc} பரிவர்த்தனை உங்கள் சராசரி பரிவர்த்தனையை விட குறிப்பிடத்தக்க அளவு அதிகமாக உள்ளது.",
            "இது அவசியமான அல்லது திட்டமிட்ட கொள்முதலா என்பதை சரிபார்க்கவும்.",
            "",
            t("recommendation"),
            f"உங்கள் அதிக செலவு வகையான {top_category}-ஐ தொடர்ந்து கண்காணிக்கவும், வழக்கத்திற்கு மாறாக பெரிய பரிவர்த்தனைகளை கவனிக்கவும், உங்கள் செலவுகளை மாதாந்திர பட்ஜெட்டுடன் ஒப்பிடவும்.",
            "தொடர்ச்சியான கண்காணிப்பு தேவையற்ற செலவுகளை கண்டறிந்து உங்கள் நிதித் திட்டமிடலை மேம்படுத்த உதவும்."
        ]
    elif lang == "hi":
        lines = [
            f"आपने {count} लेन-देन में कुल ₹{total:,.2f} खर्च दर्ज किया है।",
            f"आपका औसत लेन-देन ₹{average:,.2f} है।",
            f"आपकी सबसे अधिक खर्च वाली श्रेणी {top_category} है, जिसमें ₹{top_amount:,.2f} खर्च हुए हैं।",
            f"यह आपके कुल खर्च का लगभग {top_pct:.1f}% है।",
            f"आपका सबसे बड़ा व्यक्तिगत लेन-देन {largest_desc} था, जिसकी राशि ₹{largest_amount:,.2f} है।",
            "",
            t("high_concentration"),
            f"आपके खर्च का बड़ा हिस्सा {top_category} श्रेणी में केंद्रित है।",
            "अपने हाल के खर्चों की समीक्षा करें और उन खरीदों की पहचान करें जिन्हें कम किया जा सकता है।",
            "",
            t("large_transaction"),
            f"₹{largest_amount:,.2f} का {largest_desc} लेन-देन आपके औसत लेन-देन से काफी अधिक है।",
            "जाँचें कि यह आवश्यक या योजनाबद्ध खरीद थी या नहीं।",
            "",
            t("recommendation"),
            f"अपनी सबसे अधिक खर्च वाली श्रेणी {top_category} की नियमित समीक्षा करें, असामान्य रूप से बड़े लेन-देन पर नज़र रखें और अपने खर्च की तुलना मासिक बजट से करें।",
            "नियमित निगरानी अनावश्यक खर्चों की पहचान करने और आपकी वित्तीय योजना को बेहतर बनाने में मदद कर सकती है।"
        ]
    elif lang == "te":
        lines = [
            f"మీరు {count} లావాదేవీలలో మొత్తం ₹{total:,.2f} ఖర్చు నమోదు చేశారు.",
            f"మీ సగటు లావాదేవీ ₹{average:,.2f}.",
            f"మీ అత్యధిక ఖర్చు వర్గం {top_category}, ఇందులో ₹{top_amount:,.2f} ఖర్చు చేశారు.",
            f"ఇది మీ మొత్తం ఖర్చులో సుమారు {top_pct:.1f}%.",
            f"మీ అతిపెద్ద వ్యక్తిగత లావాదేవీ {largest_desc}, దీని మొత్తం ₹{largest_amount:,.2f}.",
            "",
            t("high_concentration"),
            f"మీ ఖర్చులో పెద్ద భాగం {top_category} వర్గంలో కేంద్రీకృతమై ఉంది.",
            "మీ ఇటీవలి ఖర్చులను సమీక్షించి, తగ్గించగల కొనుగోళ్లను గుర్తించండి.",
            "",
            t("large_transaction"),
            f"₹{largest_amount:,.2f} విలువైన {largest_desc} లావాదేవీ మీ సగటు లావాదేవీ కంటే గణనీయంగా ఎక్కువగా ఉంది.",
            "ఇది అవసరమైన లేదా ముందుగా ప్రణాళిక చేసిన కొనుగోలేనా అని తనిఖీ చేయండి.",
            "",
            t("recommendation"),
            f"మీ అత్యధిక ఖర్చు వర్గమైన {top_category}ను క్రమం తప్పకుండా సమీక్షించండి, అసాధారణంగా పెద్ద లావాదేవీలను గమనించండి మరియు మీ ఖర్చులను నెలవారీ బడ్జెట్‌తో పోల్చండి.",
            "నిరంతర పర్యవేక్షణ అవసరం లేని ఖర్చులను గుర్తించి మీ ఆర్థిక ప్రణాళికను మెరుగుపరచడంలో సహాయపడుతుంది."
        ]
    elif lang == "ml":
        lines = [
            f"നിങ്ങൾ {count} ഇടപാടുകളിലായി ആകെ ₹{total:,.2f} ചെലവ് രേഖപ്പെടുത്തിയിട്ടുണ്ട്.",
            f"നിങ്ങളുടെ ശരാശരി ഇടപാട് ₹{average:,.2f} ആണ്.",
            f"നിങ്ങളുടെ ഏറ്റവും കൂടുതൽ ചെലവുള്ള വിഭാഗം {top_category} ആണ്; ഇതിൽ ₹{top_amount:,.2f} ചെലവായി.",
            f"ഇത് നിങ്ങളുടെ ആകെ ചെലവിന്റെ ഏകദേശം {top_pct:.1f}% ആണ്.",
            f"നിങ്ങളുടെ ഏറ്റവും വലിയ വ്യക്തിഗത ഇടപാട് {largest_desc} ആയിരുന്നു; തുക ₹{largest_amount:,.2f}.",
            "",
            t("high_concentration"),
            f"നിങ്ങളുടെ ചെലവിന്റെ വലിയൊരു ഭാഗം {top_category} വിഭാഗത്തിലാണ് കേന്ദ്രീകരിച്ചിരിക്കുന്നത്.",
            "സമീപകാല ചെലവുകൾ പരിശോധിച്ച് കുറയ്ക്കാവുന്ന വാങ്ങലുകൾ കണ്ടെത്തുക.",
            "",
            t("large_transaction"),
            f"₹{largest_amount:,.2f} വിലയുള്ള {largest_desc} ഇടപാട് നിങ്ങളുടെ ശരാശരി ഇടപാടിനേക്കാൾ വളരെ കൂടുതലാണ്.",
            "ഇത് ആവശ്യമായതോ മുൻകൂട്ടി ആസൂത്രണം ചെയ്തതോ ആയ വാങ്ങലാണോ എന്ന് പരിശോധിക്കുക.",
            "",
            t("recommendation"),
            f"നിങ്ങളുടെ ഏറ്റവും കൂടുതൽ ചെലവുള്ള {top_category} വിഭാഗം പതിവായി പരിശോധിക്കുക, അസാധാരണമായി വലിയ ഇടപാടുകൾ നിരീക്ഷിക്കുക, ചെലവുകൾ മാസ ബജറ്റുമായി താരതമ്യം ചെയ്യുക.",
            "തുടർച്ചയായ നിരീക്ഷണം അനാവശ്യ ചെലവുകൾ കണ്ടെത്താനും സാമ്പത്തിക ആസൂത്രണം മെച്ചപ്പെടുത്താനും സഹായിക്കും."
        ]
    elif lang == "kn":
        lines = [
            f"ನೀವು {count} ವ್ಯವಹಾರಗಳಲ್ಲಿ ಒಟ್ಟು ₹{total:,.2f} ಖರ್ಚನ್ನು ದಾಖಲಿಸಿದ್ದೀರಿ.",
            f"ನಿಮ್ಮ ಸರಾಸರಿ ವ್ಯವಹಾರ ₹{average:,.2f}.",
            f"ನಿಮ್ಮ ಹೆಚ್ಚಿನ ಖರ್ಚಿನ ವರ್ಗ {top_category}; ಇದರಲ್ಲಿ ₹{top_amount:,.2f} ಖರ್ಚಾಗಿದೆ.",
            f"ಇದು ನಿಮ್ಮ ಒಟ್ಟು ಖರ್ಚಿನ ಸುಮಾರು {top_pct:.1f}% ಆಗಿದೆ.",
            f"ನಿಮ್ಮ ಅತಿದೊಡ್ಡ ವೈಯಕ್ತಿಕ ವ್ಯವಹಾರ {largest_desc}, ಮೊತ್ತ ₹{largest_amount:,.2f}.",
            "",
            t("high_concentration"),
            f"ನಿಮ್ಮ ಖರ್ಚಿನ ದೊಡ್ಡ ಭಾಗ {top_category} ವರ್ಗದಲ್ಲಿ ಕೇಂದ್ರೀಕೃತವಾಗಿದೆ.",
            "ನಿಮ್ಮ ಇತ್ತೀಚಿನ ಖರ್ಚುಗಳನ್ನು ಪರಿಶೀಲಿಸಿ ಮತ್ತು ಕಡಿಮೆ ಮಾಡಬಹುದಾದ ಖರೀದಿಗಳನ್ನು ಗುರುತಿಸಿ.",
            "",
            t("large_transaction"),
            f"₹{largest_amount:,.2f} ಮೊತ್ತದ {largest_desc} ವ್ಯವಹಾರವು ನಿಮ್ಮ ಸರಾಸರಿ ವ್ಯವಹಾರಕ್ಕಿಂತ ಗಮನಾರ್ಹವಾಗಿ ಹೆಚ್ಚಾಗಿದೆ.",
            "ಇದು ಅಗತ್ಯವಾದ ಅಥವಾ ಯೋಜಿತ ಖರೀದಿಯೇ ಎಂದು ಪರಿಶೀಲಿಸಿ.",
            "",
            t("recommendation"),
            f"ನಿಮ್ಮ ಹೆಚ್ಚಿನ ಖರ್ಚಿನ {top_category} ವರ್ಗವನ್ನು ನಿಯಮಿತವಾಗಿ ಪರಿಶೀಲಿಸಿ, ಅಸಾಮಾನ್ಯವಾಗಿ ದೊಡ್ಡ ವ್ಯವಹಾರಗಳನ್ನು ಗಮನಿಸಿ ಮತ್ತು ನಿಮ್ಮ ಖರ್ಚನ್ನು ಮಾಸಿಕ ಬಜೆಟ್‌ಗೆ ಹೋಲಿಸಿ.",
            "ನಿರಂತರ ಮೇಲ್ವಿಚಾರಣೆ ಅನಗತ್ಯ ಖರ್ಚುಗಳನ್ನು ಗುರುತಿಸಲು ಮತ್ತು ನಿಮ್ಮ ಹಣಕಾಸು ಯೋಜನೆಯನ್ನು ಸುಧಾರಿಸಲು ಸಹಾಯ ಮಾಡುತ್ತದೆ."
        ]
    else:
        lines = [
            f"You have recorded ₹{total:,.2f} across {count} transactions.",
            f"Your average transaction is ₹{average:,.2f}.",
            f"Your highest spending category is {top_category}, with ₹{top_amount:,.2f}.",
            f"This represents approximately {top_pct:.1f}% of your total spending.",
            f"Your largest individual transaction was {largest_desc}, costing ₹{largest_amount:,.2f}.",
            "",
            t("high_concentration"),
            f"A large portion of your spending is concentrated in {top_category}.",
            "Review your recent expenses and identify purchases that could be reduced.",
            "",
            t("large_transaction"),
            f"Your ₹{largest_amount:,.2f} transaction for {largest_desc} is significantly higher than your average transaction.",
            "Consider checking whether this was a necessary or planned purchase.",
            "",
            t("recommendation"),
            f"Review your highest spending category, {top_category}, regularly, monitor unusually large transactions, and compare your spending against your monthly budget.",
            "Consistent monitoring can help identify unnecessary expenses and improve your financial planning."
        ]
    return "\n\n".join(lines)


def build_financial_answer(df, question, lang):
    """Answer common personal-finance questions directly in the selected language."""
    q = str(question).lower().strip()
    work = df.copy()
    work["Amount"] = pd.to_numeric(work["Amount"], errors="coerce")
    work = work.dropna(subset=["Amount"])

    # Handle unusual-expense questions before the normal finance intents.
    anomaly_keywords = [
        "unusual", "unusually", "anomal", "outlier",
        "suspicious", "abnormal", "unexpected",
        "unusual expense", "unusual spending",
        "அசாதாரண", "வழக்கத்திற்கு மாறான",
        "असामान्य", "अप्रत्याशित",
        "అసాధారణ", "అనూహ్య",
        "അസാധാരണ", "അപ്രതീക്ഷിത",
        "ಅಸಾಮಾನ್ಯ", "ಅನಿರೀಕ್ಷಿತ"
    ]

    if any(word in q for word in anomaly_keywords):
        if len(work) < 5:
            if lang == "ta":
                return "வழக்கத்திற்கு மாறான செலவுகளைக் கண்டறிய குறைந்தது 5 பரிவர்த்தனைகள் தேவை."
            if lang == "hi":
                return "असामान्य खर्चों की जाँच के लिए कम से कम 5 लेन-देन आवश्यक हैं."
            return "I need at least 5 transactions to check for statistically unusual expenses."

        flagged = detect_spending_anomalies(work)
        flagged = flagged[flagged["Is_Anomaly"] == True].sort_values(
            "Amount", ascending=False
        )

        if flagged.empty:
            if lang == "ta":
                return "உங்கள் பதிவுகளில் புள்ளிவிவர அடிப்படையில் அசாதாரணமான செலவுகள் எதுவும் கண்டறியப்படவில்லை."
            if lang == "hi":
                return "आपके रिकॉर्ड में सांख्यिकीय रूप से असामान्य खर्च नहीं मिले."
            return "I didn't detect any statistically unusual expenses in your records."

        lines = []
        for _, row in flagged.iterrows():
            description = str(row.get("Description", "Expense"))
            category = str(row.get("Category", "Unknown"))
            amount = float(row["Amount"])
            date = str(row.get("Date", "Date unavailable"))
            lines.append(f"- {description} ({category}): ₹{amount:,.2f} — {date}")

        details = "\\n".join(lines)

        if lang == "ta":
            return "உங்கள் பதிவுகளில் மதிப்பாய்வு செய்ய வேண்டிய அசாதாரண செலவுகள்:\\n" + details + "\\n\\nஇவை புள்ளிவிவர அடிப்படையிலான எச்சரிக்கைகள் மட்டுமே; மோசடி என்று உறுதிப்படுத்துவதில்லை."
        if lang == "hi":
            return "इन असामान्य खर्चों की समीक्षा करें:\\n" + details + "\\n\\nये केवल सांख्यिकीय संकेत हैं, धोखाधड़ी का प्रमाण नहीं."
        return "These expenses were flagged as statistically unusual and may be worth reviewing:\\n" + details + "\\n\\nThese are statistical flags, not proof of fraud."

    total = float(work["Amount"].sum()) if not work.empty else 0.0
    count = int(len(work))
    average = total / count if count else 0.0
    category_totals = work.groupby("Category")["Amount"].sum().sort_values(ascending=False) if not work.empty else pd.Series(dtype=float)
    top_cat = str(category_totals.index[0]) if not category_totals.empty else "Others"
    top_amt = float(category_totals.iloc[0]) if not category_totals.empty else 0.0
    largest = work.loc[work["Amount"].idxmax()] if not work.empty else None

    intent = "total" if any(k in q for k in ["total", "overall", "மொத்த", "कुल", "మొత్తం", "ആകെ", "ಒಟ್ಟು"]) else None
    if any(k in q for k in ["average", "avg", "சராசரி", "औसत", "సగటు", "ശരാശരി", "ಸರಾಸರಿ"]): intent = "average"
    elif any(k in q for k in ["highest", "most", "top", "அதிக", "सबसे", "అత్యధిక", "ഏറ്റവും", "ಹೆಚ್ಚಿನ"]): intent = "top"
    elif any(k in q for k in ["largest", "biggest", "பெரிய", "बड़ा", "పెద్ద", "വലിയ", "ದೊಡ್ಡ"]): intent = "largest"
    elif any(k in q for k in ["how many", "count", "எத்தனை", "कितने", "ఎన్ని", "എത്ര", "ಎಷ್ಟು"]): intent = "count"

    if lang == "ta":
        if intent == "average": return f"உங்கள் சராசரி பரிவர்த்தனை ₹{average:,.2f}."
        if intent == "top": return f"உங்கள் அதிக செலவு வகை {top_cat}, இதில் ₹{top_amt:,.2f} செலவாகியுள்ளது."
        if intent == "largest" and largest is not None: return f"உங்கள் மிகப்பெரிய பரிவர்த்தனை {largest['Description']}, தொகை ₹{float(largest['Amount']):,.2f}."
        if intent == "count": return f"நீங்கள் {count} பரிவர்த்தனைகளை பதிவு செய்துள்ளீர்கள்."
        return f"உங்கள் மொத்த செலவு ₹{total:,.2f}."
    if lang == "hi":
        if intent == "average": return f"आपका औसत लेन-देन ₹{average:,.2f} है।"
        if intent == "top": return f"आपकी सबसे अधिक खर्च वाली श्रेणी {top_cat} है, जिसमें ₹{top_amt:,.2f} खर्च हुए हैं।"
        if intent == "largest" and largest is not None: return f"आपका सबसे बड़ा लेन-देन {largest['Description']} है, जिसकी राशि ₹{float(largest['Amount']):,.2f} है।"
        if intent == "count": return f"आपने {count} लेन-देन दर्ज किए हैं।"
        return f"आपका कुल खर्च ₹{total:,.2f} है।"
    if lang == "te":
        if intent == "average": return f"మీ సగటు లావాదేవీ ₹{average:,.2f}."
        if intent == "top": return f"మీ అత్యధిక ఖర్చు వర్గం {top_cat}, ఇందులో ₹{top_amt:,.2f} ఖర్చు చేశారు."
        if intent == "largest" and largest is not None: return f"మీ అతిపెద్ద లావాదేవీ {largest['Description']}, మొత్తం ₹{float(largest['Amount']):,.2f}."
        if intent == "count": return f"మీరు {count} లావాదేవీలను నమోదు చేశారు."
        return f"మీ మొత్తం ఖర్చు ₹{total:,.2f}."
    if lang == "ml":
        if intent == "average": return f"നിങ്ങളുടെ ശരാശരി ഇടപാട് ₹{average:,.2f} ആണ്."
        if intent == "top": return f"നിങ്ങളുടെ ഏറ്റവും കൂടുതൽ ചെലവുള്ള വിഭാഗം {top_cat} ആണ്; തുക ₹{top_amt:,.2f}."
        if intent == "largest" and largest is not None: return f"നിങ്ങളുടെ ഏറ്റവും വലിയ ഇടപാട് {largest['Description']} ആണ്; തുക ₹{float(largest['Amount']):,.2f}."
        if intent == "count": return f"നിങ്ങൾ {count} ഇടപാടുകൾ രേഖപ്പെടുത്തിയിട്ടുണ്ട്."
        return f"നിങ്ങളുടെ ആകെ ചെലവ് ₹{total:,.2f} ആണ്."
    if lang == "kn":
        if intent == "average": return f"ನಿಮ್ಮ ಸರಾಸರಿ ವ್ಯವಹಾರ ₹{average:,.2f}."
        if intent == "top": return f"ನಿಮ್ಮ ಹೆಚ್ಚಿನ ಖರ್ಚಿನ ವರ್ಗ {top_cat}; ಇದರಲ್ಲಿ ₹{top_amt:,.2f} ಖರ್ಚಾಗಿದೆ."
        if intent == "largest" and largest is not None: return f"ನಿಮ್ಮ ಅತಿದೊಡ್ಡ ವ್ಯವಹಾರ {largest['Description']}, ಮೊತ್ತ ₹{float(largest['Amount']):,.2f}."
        if intent == "count": return f"ನೀವು {count} ವ್ಯವಹಾರಗಳನ್ನು ದಾಖಲಿಸಿದ್ದೀರಿ."
        return f"ನಿಮ್ಮ ಒಟ್ಟು ಖರ್ಚು ₹{total:,.2f}."
    if intent == "average": return f"Your average transaction is ₹{average:,.2f}."
    if intent == "top": return f"Your highest spending category is {top_cat}, with ₹{top_amt:,.2f}."
    if intent == "largest" and largest is not None: return f"Your largest transaction is {largest['Description']}, costing ₹{float(largest['Amount']):,.2f}."
    if intent == "count": return f"You have recorded {count} transactions."
    return f"Your total spending is ₹{total:,.2f}."

# ==========================================
# FIRST SCREEN — LANGUAGE SELECTION
# ==========================================

if "language_selected" not in st.session_state:
    st.session_state["language_selected"] = False

if "selected_language_code" not in st.session_state:
    st.session_state["selected_language_code"] = "en"


if not st.session_state["language_selected"]:

    st.markdown(
        """
<div class="finsight-hero">
<div class="finsight-icon">💰</div>
<h1 class="finsight-title">FinSight AI</h1>
<div class="finsight-subtitle">🤖 AI-Powered Personal Finance Assistant</div>
<div class="finsight-description">Manage your expenses, understand your spending, and make smarter financial decisions.</div>
<div class="finsight-badges">
<span>📊 Smart Analytics</span>
<span>🤖 AI Insights</span>
<span>🔮 Predictive Forecasting</span>
</div>
</div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    welcome_language = st.selectbox(
        "🌐 Choose Your Language",
        list(LANGUAGE_OPTIONS.keys()),
        key="welcome_language"
    )

    if st.button(
        "🚀 Continue",
        width="stretch",
        key="language_continue"
    ):
        st.session_state["selected_language_code"] = LANGUAGE_OPTIONS[
            welcome_language
        ]
        st.session_state["selected_language_name"] = welcome_language
        st.session_state["language_selected"] = True
        st.rerun()

    st.stop()


language_code = st.session_state["selected_language_code"]
selected_language = st.session_state.get(
    "selected_language_name",
    "English"
)


# ==========================================
# HEADER
# ==========================================

st.markdown(
    f"""
    <div class="finsight-hero">
        <div class="finsight-icon">💰</div>
        <div class="finsight-title">FinSight AI</div>
        <div class="finsight-subtitle">{t("app_subtitle")}</div>
        <div class="finsight-description">{t("app_description")}</div>
        <div class="finsight-badges">
            <span>📊 Data Analytics</span>
            <span>🤖 Machine Learning</span>
            <span>🔮 Forecasting</span>
            <span>💡 Smart Insights</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ==========================================
# ➕ ADD TRANSACTION
# ==========================================

st.markdown("---")
st.header(t("add_transaction"))

col1, col2 = st.columns(2)

with col1:
    description = st.text_input(
        t("description"),
        placeholder="e.g. Uber ride to college",
        key="transaction_description"
    )

with col2:
    amount = st.number_input(
        t("amount"),
        min_value=0.0,
        step=100.0,
        format="%.2f",
        key="transaction_amount"
    )


if st.button(
    t("predict"),
    width="stretch",
    key="predict_category_button"
):

    if not description.strip():
        st.warning(t("empty_description"))

    elif amount <= 0:
        st.warning(t("invalid_amount"))

    else:
        category = predict_category(description)

        st.session_state["predicted_category"] = category

        st.success(
            f"{t('ai_predicted')}: **{category}**"
        )


# ==========================================
# SAVE TRANSACTION
# ==========================================

if "predicted_category" in st.session_state:

    category = st.session_state["predicted_category"]

    st.info(
        f"{t('category_selected')}: **{category}**"
    )

    if st.button(
        t("add_expense"),
        width="stretch",
        key="save_predicted_expense_button"
    ):

        if not description.strip():
            st.warning(t("empty_description"))

        elif amount <= 0:
            st.warning(t("invalid_amount"))

        else:

            add_expense(
                description,
                amount,
                category
            )

            st.success(
                f"{t('expense_added')} **{category}**"
            )

            del st.session_state["predicted_category"]

            st.rerun()


# ==========================================
# 📷 RECEIPT SCANNER
# ==========================================

st.markdown("---")
st.header(t("receipt"))

receipt_file = st.file_uploader(
    t("upload_receipt"),
    type=["png", "jpg", "jpeg"],
    key="receipt_uploader"
)

if receipt_file is not None:

    st.image(
        receipt_file,
        caption=t("uploaded_receipt"),
        width="stretch"
    )

    if st.button(
        t("scan_receipt"),
        key="scan_receipt_button",
        width="stretch"
    ):

        try:
            with st.spinner(t("scanning_receipt")):

                receipt_description, receipt_amount, receipt_text = (
                    extract_receipt_details(receipt_file)
                )

            # Categorize the full OCR text so supermarket/receipt terms
            # such as grocery, supermarket and provision store are available
            # to the categorizer, while retaining the extracted description.
            predicted_category = predict_category(receipt_text)

            st.success(t("receipt_scanned"))

            st.subheader(t("receipt_details"))

            st.write(
                f"**{t('receipt_description')}:** "
                f"{receipt_description}"
            )

            st.write(
                f"**{t('receipt_amount')}:** "
                f"₹{receipt_amount:,.2f}"
            )

            st.write(
                f"**{t('ai_predicted')}:** "
                f"🤖 {predicted_category}"
            )

            if receipt_amount > 0:

                if st.button(
                    t("save_scanned_expense"),
                    key="save_scanned_expense_button",
                    width="stretch"
                ):

                    add_expense(
                        receipt_description,
                        receipt_amount,
                        predicted_category
                    )

                    st.success(t("expense_added"))
                    st.rerun()

            else:
                st.warning(t("receipt_amount_not_found"))

            with st.expander(t("view_ocr_text")):
                st.text(receipt_text)

        except Exception as e:
            st.error(f"{t('receipt_scan_failed')} {e}")


# ==========================================
# LOAD DATA
# ==========================================

expenses = get_expenses()

if expenses:

    df = pd.DataFrame(
        expenses,
        columns=[
            "ID",
            "Description",
            "Amount",
            "Category",
            "Date"
        ]
    )

    df["Amount"] = pd.to_numeric(
        df["Amount"],
        errors="coerce"
    )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

else:

    df = pd.DataFrame(
        columns=[
            "ID",
            "Description",
            "Amount",
            "Category",
            "Date"
        ]
    )

# ==========================================
# 🧾 EXPENSE HISTORY
# ==========================================
st.markdown("---")
st.header(t("expense_history"))

if df.empty:

    st.info(t("no_transactions"))

else:

    st.caption(t("delete_transaction"))

    # Initialize delete target
    if "delete_target_id" not in st.session_state:
        st.session_state["delete_target_id"] = None

    for _, expense in df.iterrows():

        expense_id = int(expense["ID"])

        row = st.columns([1, 3, 2, 2, 2, 1])

        row[0].write(expense_id)
        row[1].write(str(expense["Description"]))
        row[2].write(f"₹{float(expense['Amount']):,.2f}")
        row[3].write(str(expense["Category"]))
        row[4].write(str(expense["Date"]))

        # -----------------------------------------
        # DELETE BUTTON
        # -----------------------------------------

        with row[5]:

            if st.button(
                "🗑️",
                key=f"delete_expense_{expense_id}",
                help=t("delete_transaction")
            ):

                st.session_state["delete_target_id"] = expense_id
                st.rerun()

        # -----------------------------------------
        # DELETE CONFIRMATION
        # -----------------------------------------

        if st.session_state.get("delete_target_id") == expense_id:

            st.warning(
                f"{t('delete_transaction')}: "
                f"**{expense['Description']} — "
                f"₹{float(expense['Amount']):,.2f}**"
            )

            cancel_col, confirm_col = st.columns(2)

            # -------------------------------------
            # CANCEL
            # -------------------------------------

            with cancel_col:

                if st.button(
                    t("cancel"),
                    key=f"cancel_delete_{expense_id}"
                ):

                    st.session_state["delete_target_id"] = None
                    st.rerun()

            # -------------------------------------
            # CONFIRM DELETE
            # -------------------------------------

            with confirm_col:

                if st.button(
                    t("yes_delete"),
                    key=f"yes_delete_{expense_id}"
                ):

                    delete_expense(expense_id)

                    st.session_state["delete_target_id"] = None

                    st.success(
                        t("transaction_deleted")
                    )

                    st.rerun()

# ==========================================
# ✏️ EDIT EXPENSE
# ==========================================

st.markdown("---")
st.header(t("edit_expense"))

if df.empty:

    st.info(t("no_expenses_edit"))

else:

    expense_ids = df["ID"].tolist()

    selected_id = st.selectbox(
        t("select_expense"),
        expense_ids,
        key="edit_expense_id"
    )

    selected_expense = df[
        df["ID"] == selected_id
    ].iloc[0]

    edit_description = st.text_input(
        t("edit_description"),
        value=str(selected_expense["Description"]),
        key="edit_description"
    )

    edit_amount = st.number_input(
        t("edit_amount"),
        min_value=0.01,
        value=float(selected_expense["Amount"]),
        step=1.0,
        key="edit_amount"
    )

    current_category = str(
        selected_expense["Category"]
    )

    category_index = (
        CATEGORIES.index(current_category)
        if current_category in CATEGORIES
        else 0
    )

    edit_category = st.selectbox(
        t("edit_category"),
        CATEGORIES,
        index=category_index,
        key="edit_category"
    )

    if st.button(
        t("update_expense"),
        width="stretch",
        key="update_expense_button"
    ):

        update_expense(
            selected_id,
            edit_description,
            edit_amount,
            edit_category
        )

        st.success(t("expense_updated"))
        st.rerun()

# ==========================================
# 🧠 AI FINANCIAL REVIEW
# ==========================================

st.markdown("---")
st.header(t("ai_review"))

if df.empty:

    st.info(t("ai_review_empty"))

else:

    financial_review = build_financial_review(df, language_code)
    st.markdown(financial_review)


# ==========================================
# 🎯 BUDGET INTELLIGENCE
# ==========================================

st.markdown("---")
st.header(t("budget_intelligence"))

current_budget = get_budget()

budget_col1, budget_col2 = st.columns(2)

with budget_col1:

    new_budget = st.number_input(
        t("monthly_budget"),
        min_value=0.0,
        value=float(current_budget),
        step=500.0,
        key="monthly_budget_input"
    )

    if st.button(
        t("save_budget"),
        width="stretch",
        key="save_budget_button"
    ):

        if new_budget <= 0:

            st.warning(t("invalid_amount"))

        else:

            save_budget(new_budget)

            st.success(
                f"{t('budget_saved')} ₹{new_budget:,.2f}"
            )

            st.rerun()


with budget_col2:

    if current_budget > 0:

        today = datetime.now()

        spending = monthly_spending(
            today.year,
            today.month
        )

        remaining = current_budget - spending

        percentage_used = (
            spending / current_budget
        ) * 100

        st.metric(
            t("budget"),
            f"₹{current_budget:,.2f}"
        )

        st.metric(
            t("budget_used"),
            f"{percentage_used:.1f}%"
        )

        if remaining < 0:

            st.error(
                f"🚨 You exceeded your budget by "
                f"₹{abs(remaining):,.2f}"
            )

        elif percentage_used >= 80:

            st.warning(
                f"⚠️ You have only ₹{remaining:,.2f} remaining."
            )

        else:

            st.success(
                f"✅ You have ₹{remaining:,.2f} remaining."
            )

    else:

        st.info(t("no_expenses_budget"))


# ==========================================
# 📊 DASHBOARD METRICS
# ==========================================

st.markdown("---")

st.markdown(
    f"""
    <div class="dashboard-section-title">
        <div class="dashboard-section-icon">📊</div>
        <div>
            <div class="section-kicker">FinSight Overview</div>
            <div class="dashboard-section-heading">{("Financial Dashboard" if language_code == "en" else {"ta":"நிதி டாஷ்போர்டு","hi":"वित्तीय डैशबोर्ड","te":"ఆర్థిక డ్యాష్‌బోర్డ్","ml":"സാമ്പത്തിക ഡാഷ്ബോർഡ്","kn":"ಹಣಕಾಸು ಡ್ಯಾಶ್‌ಬೋರ್ಡ್"}.get(language_code, "Financial Dashboard"))}</div>
            <div class="dashboard-section-subtitle">A quick overview of your spending activity</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

m1, m2, m3 = st.columns(3)

with m1:
    st.metric(
        t("total_spending"),
        f"₹{total_spending():,.2f}"
    )

with m2:
    st.metric(
        t("transactions"),
        len(df)
    )

with m3:

    if not df.empty:
        average = df["Amount"].mean()
    else:
        average = 0

    st.metric(
        t("average"),
        f"₹{average:,.2f}"
    )


# ==========================================
# 📊 SPENDING ANALYSIS
# ==========================================

st.markdown("---")
st.header(t("spending_analysis"))

if df.empty:

    st.info(t("no_expenses_analysis"))

else:

    category_summary = (
        df.groupby("Category")["Amount"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:

        st.subheader(t("spending_by_category"))

        fig_category = px.bar(
            category_summary,
            x="Category",
            y="Amount",
            title=t("spending_by_category")
        )

        st.plotly_chart(
            fig_category,
            width="stretch",
            key="category_bar_chart"
        )

    with chart_col2:

        st.subheader(t("spending_distribution"))

        fig_pie = px.pie(
            category_summary,
            names="Category",
            values="Amount",
            title=t("where_money_goes")
        )

        st.plotly_chart(
            fig_pie,
            width="stretch",
            key="category_pie_chart"
        )

    # Spending trend appears only once.
    date_data = (
        df.dropna(subset=["Date"])
        .groupby("Date")["Amount"]
        .sum()
        .reset_index()
    )

    st.subheader(t("spending_over_time"))

    if not date_data.empty:

        fig_trend = px.line(
            date_data,
            x="Date",
            y="Amount",
            title=t("spending_over_time"),
            markers=True
        )

        st.plotly_chart(
            fig_trend,
            width="stretch",
            key="spending_trend_chart"
        )

    else:

        st.info(t("no_valid_dates"))
# ==========================================
# 💬 FINANCE Q&A
# ==========================================

st.markdown("---")
st.header(t("finance_qa"))


def clear_finance_question():
    # Do NOT assign to finance_question here because
    # Streamlit does not allow changing a widget's value
    # after the widget has been created in the same run.
    st.session_state["clear_question_requested"] = True


if "clear_question_requested" not in st.session_state:
    st.session_state["clear_question_requested"] = False

if st.session_state["clear_question_requested"]:
    st.session_state["clear_question_requested"] = False
    st.session_state["finance_question"] = ""


# Question input
qa_col1, qa_col2 = st.columns([6, 1])

with qa_col1:

    question = st.text_input(
        t("ask_question"),
        key="finance_question"
    )

with qa_col2:

    st.write("")
    st.write("")

    st.button(
        "✖",
        key="clear_finance_question_button",
        help=t("clear_question"),
        on_click=clear_finance_question
    )


# Ask button
if st.button(
    t("ask_ai"),
    key="ask_financial_question_button"
):

    if not question.strip():

        st.warning(t("question_first"))

    else:

        st.session_state["finance_answer"] = build_financial_answer(
            df,
            question,
            language_code
        )


# Display answer exactly once
if "finance_answer" in st.session_state:

    st.success("🤖 FinSight AI")
    st.markdown(st.session_state["finance_answer"])


# ==========================================
# 🚨 AI SPENDING ANOMALY DETECTION
# ==========================================

st.markdown("---")
st.header(anomaly_t("title"))
st.write(anomaly_t("description"))

if df.empty:
    st.info(anomaly_t("need_more"))
else:
    anomaly_df = detect_spending_anomalies(df)
    valid_anomaly_count = int(
        anomaly_df["Amount"].notna().sum()
    )

    if valid_anomaly_count < 5:
        st.info(anomaly_t("need_more"))
    else:
        anomalies = anomaly_df[
            anomaly_df["Is_Anomaly"]
        ].copy()

        if anomalies.empty:
            st.success(anomaly_t("no_anomalies"))
        else:
            st.warning(
                f"{anomaly_t('detected')}: **{len(anomalies)}**"
            )

            display_anomalies = anomalies.copy()
            display_anomalies[anomaly_t("transaction")] = (
                display_anomalies["Description"].astype(str)
            )
            display_anomalies[anomaly_t("amount")] = display_anomalies[
                "Amount"
            ].apply(lambda x: f"₹{x:,.2f}")
            display_anomalies[anomaly_t("category")] = (
                display_anomalies["Category"].astype(str)
            )
            display_anomalies[anomaly_t("reason")] = anomaly_t("reason_text")

            st.dataframe(
                display_anomalies[
                    [
                        anomaly_t("transaction"),
                        anomaly_t("amount"),
                        anomaly_t("category"),
                        anomaly_t("reason"),
                    ]
                ],
                width="stretch",
                hide_index=True,
            )

            # Highlight the largest anomaly for quick action.
            largest_anomaly = anomalies.loc[
                anomalies["Amount"].idxmax()
            ]
            st.error(
                f"🚨 **₹{largest_anomaly['Amount']:,.2f}** — "
                f"{largest_anomaly['Description']}"
            )

            st.caption(anomaly_t("model"))
            st.info(anomaly_t("tip"))
# ============================================================
# 🔮 AI SPENDING FORECAST
# ============================================================

st.markdown("---")

st.header(forecast_t("title"))
st.write(forecast_t("description"))

if df.empty:

    st.info(forecast_t("need_more"))

else:

    # ========================================================
    # AUTOMATIC FORECAST ENGINE
    # ========================================================

    forecast_result = forecast_spending(df)

    # ========================================================
    # FORECAST NOT AVAILABLE
    # ========================================================

    if not forecast_result.get("success"):

        st.info(
            forecast_result.get(
                "message",
                forecast_t("need_more")
            )
        )

    # ========================================================
    # FORECAST AVAILABLE
    # ========================================================

    else:

        prediction = forecast_result.get(
            "prediction",
            0
        )

        model_used = forecast_result.get(
            "model",
            "Unknown"
        )

        months_available = forecast_result.get(
            "months_available",
            0
        )

        # ====================================================
        # MODEL INFORMATION
        # ====================================================

        st.subheader("🤖 Forecast Model")

        model_col1, model_col2 = st.columns(2)

        with model_col1:

            st.metric(
                "Model Used",
                model_used
            )

        with model_col2:

            st.metric(
                "Months Available",
                months_available
            )

        # ====================================================
        # PREDICTION
        # ====================================================

        st.metric(
            forecast_t("prediction"),
            f"₹{prediction:,.2f}"
        )

        # ====================================================
        # LSTM RESULT
        # ====================================================

        if model_used == "LSTM":

            st.success(
                "🧠 LSTM forecasting model was used "
                "automatically."
            )

            months_used = forecast_result.get(
                "months_used"
            )

            next_month = forecast_result.get(
                "next_month"
            )

            if months_used is not None:

                st.caption(
                    f"Historical sequence used: "
                    f"{months_used} months"
                )

            if next_month:

                st.caption(
                    f"Forecast period: {next_month}"
                )

        # ====================================================
        # STATISTICAL FORECAST
        # ====================================================

        elif model_used == "Statistical Forecast":

            if forecast_result.get(
                "fallback",
                False
            ):

                st.warning(
                    "⚠️ LSTM was unavailable, "
                    "so FinSight AI automatically "
                    "used the statistical forecasting model."
                )

            else:

                st.info(
                    "📊 Statistical forecasting model "
                    "was selected based on available history."
                )

        # ====================================================
        # MODEL PERFORMANCE
        # ====================================================

        if model_used == "Statistical Forecast":

            mae = forecast_result.get("mae")
            rmse = forecast_result.get("rmse")
            r2 = forecast_result.get("r2")

            if (
                mae is not None
                or rmse is not None
                or r2 is not None
            ):

                st.subheader(
                    forecast_t("evaluation")
                )

                eval_col1, eval_col2, eval_col3 = (
                    st.columns(3)
                )

                with eval_col1:

                    if mae is not None and not pd.isna(mae):

                        st.metric(
                            forecast_t("mae"),
                            f"₹{mae:,.2f}"
                        )

                    else:

                        st.metric(
                            forecast_t("mae"),
                            "N/A"
                        )

                with eval_col2:

                    if rmse is not None and not pd.isna(rmse):

                        st.metric(
                            forecast_t("rmse"),
                            f"₹{rmse:,.2f}"
                        )

                    else:

                        st.metric(
                            forecast_t("rmse"),
                            "N/A"
                        )

                with eval_col3:

                    if r2 is not None and not pd.isna(r2):

                        st.metric(
                            forecast_t("r2"),
                            f"{r2:.2f}"
                        )

                    else:

                        st.metric(
                            forecast_t("r2"),
                            "N/A"
                        )

        # ====================================================
        # HISTORICAL TREND
        # ====================================================

        history = forecast_result.get(
            "history"
        )

        if history is not None and not history.empty:

            st.subheader(
                forecast_t("evaluation")
            )

            chart_df = history.copy()

            if "Date" in chart_df.columns:

                chart_df["Month"] = (
                    pd.to_datetime(
                        chart_df["Date"],
                        errors="coerce"
                    ).dt.strftime("%Y-%m")
                )

            elif "Month" not in chart_df.columns:

                chart_df["Month"] = range(
                    1,
                    len(chart_df) + 1
                )

            fig_forecast = px.line(
                chart_df,
                x="Month",
                y="Amount",
                markers=True,
                title=forecast_t("trend")
            )

            st.plotly_chart(
                fig_forecast,
                width="stretch",
                key="forecast_history_chart"
            )

        # ====================================================
        # BUDGET COMPARISON
        # ====================================================

        budget_for_forecast = get_budget()

        st.subheader(
            forecast_t("insight")
        )

        if budget_for_forecast > 0:

            if prediction > budget_for_forecast:

                st.warning(
                    forecast_t("over_budget")
                )

            else:

                st.success(
                    forecast_t("within_budget")
                )

        # ====================================================
        # MODEL NOTE
        # ====================================================

        st.caption(
            forecast_t("model")
        )
# ==========================================
# 🧪 ML MODEL EVALUATION DASHBOARD
# ==========================================

EVAL_TEXT = {

    # ======================================================
    # ENGLISH
    # ======================================================

    "en": {
        "title": "🧪 ML Model Performance",
        "desc": "Evaluate the machine-learning models used by FinSight AI.",

        "cat": "🤖 Expense Categorizer",
        "anom": "🚨 Anomaly Detector",
        "forecast": "🔮 Spending Forecast",

        "cat_title": "Expense Categorization Model",
        "cat_desc": "Performance of the trained text-classification model on your stored transactions.",

        "not_enough": "Add more transactions with categories to evaluate this model.",

        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1 Score",

        "cm": "Confusion Matrix",
        "actual": "Actual",
        "predicted": "Predicted",

        "cat_unavailable": "The saved categorization model could not be evaluated:",
        "evaluation_error": "The model evaluation could not be completed:",

        "anom_title": "Isolation Forest Performance",
        "analyzed": "Transactions Analyzed",
        "normal": "Normal Transactions",
        "anomalies": "Anomalies Detected",
        "rate": "Anomaly Rate",

        "anom_note": "Isolation Forest identifies transactions that are unusual compared with the overall spending pattern.",

        "forecast_title": "Spending Forecast Performance",

        "lstm_demo_title": "🧠 LSTM Demo Evaluation",
        "lstm_demo_dataset": "Dataset: Synthetic / Demo Data",
        "lstm_demo_model": "Model: LSTM Demo",
        "lstm_demo_status": "Status: Experimental — Not Used for Production Predictions",
        "lstm_demo_note": "This LSTM was trained on synthetic/demo data to validate the forecasting pipeline. It is kept separate from the production forecasting model.",
        "lstm_demo_no_chart": "Actual vs Predicted values are not displayed because the original demo evaluation dataset is not stored in this application.",

        "forecast_note": "Forecast quality is measured using historical holdout data. Lower MAE/RMSE is generally better; R² closer to 1 is generally better.",

        "insufficient": "Not enough historical monthly data for evaluation.",

        "mae": "MAE",
        "rmse": "RMSE",
        "r2": "R²",

        "status": "Status",
        "count": "Count",
        "normal_status": "Normal",
        "anomaly_status": "Anomaly",
    },


    # ======================================================
    # TAMIL
    # ======================================================

    "ta": {
        "title": "🧪 ML மாதிரி செயல்திறன்",
        "desc": "FinSight AI பயன்படுத்தும் இயந்திரக் கற்றல் மாதிரிகளை மதிப்பிடுங்கள்.",

        "cat": "🤖 செலவு வகைப்படுத்தி",
        "anom": "🚨 அசாதாரண செலவு கண்டறிதல்",
        "forecast": "🔮 செலவு முன்னறிவிப்பு",

        "cat_title": "செலவு வகைப்படுத்தல் மாதிரி",
        "cat_desc": "சேமிக்கப்பட்ட பரிவர்த்தனைகளில் பயிற்சி பெற்ற உரை வகைப்படுத்தல் மாதிரியின் செயல்திறன்.",

        "not_enough": "இந்த மாதிரியை மதிப்பிட மேலும் வகைப்படுத்தப்பட்ட பரிவர்த்தனைகளைச் சேர்க்கவும்.",

        "accuracy": "துல்லியம்",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1 மதிப்பெண்",

        "cm": "Confusion Matrix",
        "actual": "உண்மை",
        "predicted": "கணிக்கப்பட்டது",

        "cat_unavailable": "சேமிக்கப்பட்ட வகைப்படுத்தல் மாதிரியை மதிப்பிட முடியவில்லை:",
        "evaluation_error": "மாதிரி மதிப்பீட்டை முடிக்க முடியவில்லை:",

        "anom_title": "Isolation Forest செயல்திறன்",
        "analyzed": "பகுப்பாய்வு செய்யப்பட்ட பரிவர்த்தனைகள்",
        "normal": "சாதாரண பரிவர்த்தனைகள்",
        "anomalies": "கண்டறியப்பட்ட அசாதாரணங்கள்",
        "rate": "அசாதாரண விகிதம்",

        "anom_note": "Isolation Forest உங்கள் மொத்த செலவு முறையுடன் ஒப்பிடும்போது அசாதாரணமான பரிவர்த்தனைகளை கண்டறிகிறது.",

        "forecast_title": "செலவு முன்னறிவிப்பு செயல்திறன்",

        "lstm_demo_title": "🧠 LSTM டெமோ மதிப்பீடு",
        "lstm_demo_dataset": "தரவுத்தொகுப்பு: செயற்கை / டெமோ தரவு",
        "lstm_demo_model": "மாதிரி: LSTM டெமோ",
        "lstm_demo_status": "நிலை: பரிசோதனை — உற்பத்தி முன்னறிவிப்புகளுக்கு பயன்படுத்தப்படவில்லை",
        "lstm_demo_note": "முன்னறிவிப்பு செயல்முறையை சரிபார்க்க இந்த LSTM செயற்கை / டெமோ தரவில் பயிற்சி அளிக்கப்பட்டது. இது உற்பத்தி முன்னறிவிப்பு மாதிரியிலிருந்து தனியாக வைக்கப்பட்டுள்ளது.",
        "lstm_demo_no_chart": "அசல் டெமோ மதிப்பீட்டு தரவுத்தொகுப்பு இந்த பயன்பாட்டில் சேமிக்கப்படாததால், Actual மற்றும் Predicted மதிப்புகள் காட்டப்படவில்லை.",

        "forecast_note": "முன்னறிவிப்பு செயல்திறன் வரலாற்றுத் தரவின் சோதனைப் பகுதியால் மதிப்பிடப்படுகிறது. குறைந்த MAE/RMSE பொதுவாக சிறந்தது; R² 1க்கு அருகில் இருப்பது பொதுவாக சிறந்தது.",

        "insufficient": "மாதாந்திர முன்னறிவிப்பு மதிப்பீட்டிற்கு போதுமான வரலாற்றுத் தரவு இல்லை.",

        "mae": "MAE",
        "rmse": "RMSE",
        "r2": "R²",

        "status": "நிலை",
        "count": "எண்ணிக்கை",
        "normal_status": "சாதாரணம்",
        "anomaly_status": "அசாதாரணம்",
    },


    # ======================================================
    # HINDI
    # ======================================================

    "hi": {
        "title": "🧪 ML मॉडल प्रदर्शन",
        "desc": "FinSight AI में उपयोग किए गए मशीन-लर्निंग मॉडलों का मूल्यांकन करें।",

        "cat": "🤖 खर्च वर्गीकरण",
        "anom": "🚨 असामान्य खर्च पहचान",
        "forecast": "🔮 खर्च पूर्वानुमान",

        "cat_title": "खर्च वर्गीकरण मॉडल",
        "cat_desc": "सहेजे गए लेन-देन पर प्रशिक्षित टेक्स्ट वर्गीकरण मॉडल का प्रदर्शन।",

        "not_enough": "इस मॉडल का मूल्यांकन करने के लिए अधिक वर्गीकृत लेन-देन जोड़ें।",

        "accuracy": "सटीकता",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1 स्कोर",

        "cm": "Confusion Matrix",
        "actual": "वास्तविक",
        "predicted": "अनुमानित",

        "cat_unavailable": "सहेजे गए वर्गीकरण मॉडल का मूल्यांकन नहीं हो सका:",
        "evaluation_error": "मॉडल मूल्यांकन पूरा नहीं किया जा सका:",

        "anom_title": "Isolation Forest प्रदर्शन",
        "analyzed": "विश्लेषित लेन-देन",
        "normal": "सामान्य लेन-देन",
        "anomalies": "पता लगाए गए असामान्य लेन-देन",
        "rate": "असामान्यता दर",

        "anom_note": "Isolation Forest आपके सामान्य खर्च पैटर्न से अलग लेन-देन पहचानता है।",

        "forecast_title": "खर्च पूर्वानुमान प्रदर्शन",

        "lstm_demo_title": "🧠 LSTM डेमो मूल्यांकन",
        "lstm_demo_dataset": "डेटासेट: सिंथेटिक / डेमो डेटा",
        "lstm_demo_model": "मॉडल: LSTM डेमो",
        "lstm_demo_status": "स्थिति: प्रयोगात्मक — उत्पादन पूर्वानुमानों के लिए उपयोग नहीं किया जाता",
        "lstm_demo_note": "पूर्वानुमान प्रक्रिया को सत्यापित करने के लिए इस LSTM को सिंथेटिक / डेमो डेटा पर प्रशिक्षित किया गया था। इसे उत्पादन पूर्वानुमान मॉडल से अलग रखा गया है।",
        "lstm_demo_no_chart": "मूल डेमो मूल्यांकन डेटासेट इस एप्लिकेशन में संग्रहीत नहीं है, इसलिए Actual और Predicted मान प्रदर्शित नहीं किए गए हैं।",

        "forecast_note": "पूर्वानुमान गुणवत्ता ऐतिहासिक holdout डेटा से मापी जाती है। कम MAE/RMSE सामान्यतः बेहतर है; R² का 1 के करीब होना सामान्यतः बेहतर है।",

        "insufficient": "मासिक पूर्वानुमान मूल्यांकन के लिए पर्याप्त ऐतिहासिक डेटा नहीं है।",

        "mae": "MAE",
        "rmse": "RMSE",
        "r2": "R²",

        "status": "स्थिति",
        "count": "गिनती",
        "normal_status": "सामान्य",
        "anomaly_status": "असामान्य",
    },


    # ======================================================
    # TELUGU
    # ======================================================

    "te": {
        "title": "🧪 ML మోడల్ పనితీరు",
        "desc": "FinSight AI ఉపయోగించే మెషిన్ లెర్నింగ్ మోడళ్లను మూల్యాంకనం చేయండి.",

        "cat": "🤖 ఖర్చు వర్గీకరణ",
        "anom": "🚨 అసాధారణ ఖర్చు గుర్తింపు",
        "forecast": "🔮 ఖర్చు అంచనా",

        "cat_title": "ఖర్చు వర్గీకరణ మోడల్",
        "cat_desc": "నిల్వ చేసిన లావాదేవీలపై శిక్షణ పొందిన టెక్స్ట్ వర్గీకరణ మోడల్ పనితీరు.",

        "not_enough": "ఈ మోడల్‌ను మూల్యాంకనం చేయడానికి మరిన్ని వర్గీకరించిన లావాదేవీలను జోడించండి.",

        "accuracy": "ఖచ్చితత్వం",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1 స్కోర్",

        "cm": "Confusion Matrix",
        "actual": "నిజమైనది",
        "predicted": "అంచనా",

        "cat_unavailable": "నిల్వ చేసిన వర్గీకరణ మోడల్‌ను మూల్యాంకనం చేయలేకపోయాము:",
        "evaluation_error": "మోడల్ మూల్యాంకనాన్ని పూర్తి చేయలేకపోయాము:",

        "anom_title": "Isolation Forest పనితీరు",
        "analyzed": "విశ్లేషించిన లావాదేవీలు",
        "normal": "సాధారణ లావాదేవీలు",
        "anomalies": "గుర్తించిన అసాధారణాలు",
        "rate": "అసాధారణ శాతం",

        "anom_note": "Isolation Forest సాధారణ ఖర్చు నమూనాతో పోలిస్తే అసాధారణమైన లావాదేవీలను గుర్తిస్తుంది.",

        "forecast_title": "ఖర్చు అంచనా పనితీరు",
        "lstm_demo_title": "🧠 LSTM డెమో మూల్యాంకనం",
        "lstm_demo_dataset": "డేటాసెట్: సింథటిక్ / డెమో డేటా",
        "lstm_demo_model": "మోడల్: LSTM డెమో",
        "lstm_demo_status": "స్థితి: ప్రయోగాత్మకం — ఉత్పత్తి అంచనాల కోసం ఉపయోగించబడదు",
        "lstm_demo_note": "ఫోర్‌కాస్టింగ్ ప్రక్రియను ధృవీకరించడానికి ఈ LSTM సింథటిక్ / డెమో డేటాపై శిక్షణ పొందింది. ఇది ఉత్పత్తి ఫోర్‌కాస్టింగ్ మోడల్ నుండి వేరుగా ఉంచబడింది.",
        "lstm_demo_no_chart": "అసలు డెమో మూల్యాంకన డేటాసెట్ ఈ అప్లికేషన్‌లో నిల్వ చేయబడనందున, Actual మరియు Predicted విలువలు చూపబడలేదు.",
        "forecast_note": "అంచనా నాణ్యత చారిత్రక holdout డేటాతో కొలుస్తారు. తక్కువ MAE/RMSE సాధారణంగా మెరుగైనది; R² 1కి దగ్గరగా ఉండటం సాధారణంగా మెరుగైనది.",

        "insufficient": "నెలవారీ అంచనా మూల్యాంకనానికి తగిన చారిత్రక డేటా లేదు.",

        "mae": "MAE",
        "rmse": "RMSE",
        "r2": "R²",

        "status": "స్థితి",
        "count": "సంఖ్య",
        "normal_status": "సాధారణం",
        "anomaly_status": "అసాధారణం",
    },


    # ======================================================
    # MALAYALAM
    # ======================================================

    "ml": {
        "title": "🧪 ML മോഡൽ പ്രകടനം",
        "desc": "FinSight AI ഉപയോഗിക്കുന്ന മെഷീൻ ലേണിംഗ് മോഡലുകൾ വിലയിരുത്തുക.",

        "cat": "🤖 ചെലവ് വിഭാഗീകരണം",
        "anom": "🚨 അസാധാരണ ചെലവ് കണ്ടെത്തൽ",
        "forecast": "🔮 ചെലവ് പ്രവചനം",

        "cat_title": "ചെലവ് വിഭാഗീകരണ മോഡൽ",
        "cat_desc": "സംരക്ഷിച്ച ഇടപാടുകളിൽ പരിശീലിപ്പിച്ച ടെക്സ്റ്റ് ക്ലാസിഫിക്കേഷൻ മോഡലിന്റെ പ്രകടനം.",

        "not_enough": "ഈ മോഡൽ വിലയിരുത്താൻ കൂടുതൽ വിഭാഗീകരിച്ച ഇടപാടുകൾ ചേർക്കുക.",

        "accuracy": "കൃത്യത",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1 സ്കോർ",

        "cm": "Confusion Matrix",
        "actual": "യഥാർത്ഥം",
        "predicted": "പ്രവചിച്ചത്",

        "cat_unavailable": "സംരക്ഷിച്ച വിഭാഗീകരണ മോഡൽ വിലയിരുത്താൻ കഴിഞ്ഞില്ല:",
        "evaluation_error": "മോഡൽ വിലയിരുത്തൽ പൂർത്തിയാക്കാൻ കഴിഞ്ഞില്ല:",

        "anom_title": "Isolation Forest പ്രകടനം",
        "analyzed": "വിശകലനം ചെയ്ത ഇടപാടുകൾ",
        "normal": "സാധാരണ ഇടപാടുകൾ",
        "anomalies": "കണ്ടെത്തിയ അസാധാരണങ്ങൾ",
        "rate": "അസാധാരണ നിരക്ക്",

        "anom_note": "Isolation Forest നിങ്ങളുടെ സാധാരണ ചെലവ് മാതൃകയിൽ നിന്ന് വ്യത്യസ്തമായ ഇടപാടുകൾ കണ്ടെത്തുന്നു.",

        "forecast_title": "ചെലവ് പ്രവചന പ്രകടനം",
        "lstm_demo_title": "🧠 LSTM ഡെമോ മൂല്യനിർണ്ണയം",
        "lstm_demo_dataset": "ഡാറ്റാസെറ്റ്: സിന്തറ്റിക് / ഡെമോ ഡാറ്റ",
        "lstm_demo_model": "മോഡൽ: LSTM ഡെമോ",
        "lstm_demo_status": "നില: പരീക്ഷണാടിസ്ഥാനത്തിൽ — പ്രൊഡക്ഷൻ പ്രവചനങ്ങൾക്ക് ഉപയോഗിക്കുന്നില്ല",
        "lstm_demo_note": "ഫോർക്കാസ്റ്റിംഗ് പ്രക്രിയ പരിശോധിക്കുന്നതിനായി ഈ LSTM സിന്തറ്റിക് / ഡെമോ ഡാറ്റ ഉപയോഗിച്ച് പരിശീലിപ്പിച്ചതാണ്. ഇത് പ്രൊഡക്ഷൻ ഫോർക്കാസ്റ്റിംഗ് മോഡലിൽ നിന്ന് വേർതിരിച്ചാണ് സൂക്ഷിച്ചിരിക്കുന്നത്.",
        "lstm_demo_no_chart": "യഥാർത്ഥ ഡെമോ മൂല്യനിർണ്ണയ ഡാറ്റാസെറ്റ് ഈ ആപ്ലിക്കേഷനിൽ സംഭരിച്ചിട്ടില്ലാത്തതിനാൽ Actual, Predicted മൂല്യങ്ങൾ പ്രദർശിപ്പിക്കുന്നില്ല.",
        "forecast_note": "ചരിത്രപരമായ holdout ഡാറ്റ ഉപയോഗിച്ചാണ് പ്രവചന നിലവാരം അളക്കുന്നത്. കുറഞ്ഞ MAE/RMSE സാധാരണയായി മികച്ചതാണ്; R² 1-നോട് അടുത്തത് സാധാരണയായി മികച്ചതാണ്.",

        "insufficient": "മാസാന്ത്യ പ്രവചന വിലയിരുത്തലിന് മതിയായ ചരിത്ര ഡാറ്റയില്ല.",

        "mae": "MAE",
        "rmse": "RMSE",
        "r2": "R²",

        "status": "നില",
        "count": "എണ്ണം",
        "normal_status": "സാധാരണം",
        "anomaly_status": "അസാധാരണം",
    },


    # ======================================================
    # KANNADA
    # ======================================================

    "kn": {
        "title": "🧪 ML ಮಾದರಿ ಕಾರ್ಯಕ್ಷಮತೆ",
        "desc": "FinSight AI ಬಳಸುವ ಯಂತ್ರ ಕಲಿಕೆ ಮಾದರಿಗಳನ್ನು ಮೌಲ್ಯಮಾಪನ ಮಾಡಿ.",

        "cat": "🤖 ಖರ್ಚಿನ ವರ್ಗೀಕರಣ",
        "anom": "🚨 ಅಸಾಮಾನ್ಯ ಖರ್ಚು ಪತ್ತೆ",
        "forecast": "🔮 ಖರ್ಚಿನ ಮುನ್ಸೂಚನೆ",

        "cat_title": "ಖರ್ಚಿನ ವರ್ಗೀಕರಣ ಮಾದರಿ",
        "cat_desc": "ಉಳಿಸಿದ ವ್ಯವಹಾರಗಳ ಮೇಲೆ ತರಬೇತಿ ಪಡೆದ ಪಠ್ಯ ವರ್ಗೀಕರಣ ಮಾದರಿಯ ಕಾರ್ಯಕ್ಷಮತೆ.",

        "not_enough": "ಈ ಮಾದರಿಯನ್ನು ಮೌಲ್ಯಮಾಪನ ಮಾಡಲು ಇನ್ನಷ್ಟು ವರ್ಗೀಕರಿಸಿದ ವ್ಯವಹಾರಗಳನ್ನು ಸೇರಿಸಿ.",

        "accuracy": "ನಿಖರತೆ",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1 ಸ್ಕೋರ್",

        "cm": "Confusion Matrix",
        "actual": "ನಿಜವಾದ",
        "predicted": "ಅಂದಾಜು",

        "cat_unavailable": "ಉಳಿಸಿದ ವರ್ಗೀಕರಣ ಮಾದರಿಯನ್ನು ಮೌಲ್ಯಮಾಪನ ಮಾಡಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ:",
        "evaluation_error": "ಮಾದರಿ ಮೌಲ್ಯಮಾಪನವನ್ನು ಪೂರ್ಣಗೊಳಿಸಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ:",

        "anom_title": "Isolation Forest ಕಾರ್ಯಕ್ಷಮತೆ",
        "analyzed": "ವಿಶ್ಲೇಷಿಸಿದ ವ್ಯವಹಾರಗಳು",
        "normal": "ಸಾಮಾನ್ಯ ವ್ಯವಹಾರಗಳು",
        "anomalies": "ಪತ್ತೆಯಾದ ಅಸಾಮಾನ್ಯಗಳು",
        "rate": "ಅಸಾಮಾನ್ಯ ದರ",

        "anom_note": "Isolation Forest ಸಾಮಾನ್ಯ ಖರ್ಚಿನ ಮಾದರಿಗಿಂತ ವಿಭಿನ್ನವಾದ ವ್ಯವಹಾರಗಳನ್ನು ಗುರುತಿಸುತ್ತದೆ.",

        "forecast_title": "ಖರ್ಚಿನ ಮುನ್ಸೂಚನೆ ಕಾರ್ಯಕ್ಷಮತೆ",
        "lstm_demo_title": "🧠 LSTM ಡೆಮೋ ಮೌಲ್ಯಮಾಪನ",
        "lstm_demo_dataset": "ಡೇಟಾಸೆಟ್: ಸಿಂಥೆಟಿಕ್ / ಡೆಮೋ ಡೇಟಾ",
        "lstm_demo_model": "ಮಾದರಿ: LSTM ಡೆಮೋ",
        "lstm_demo_status": "ಸ್ಥಿತಿ: ಪ್ರಾಯೋಗಿಕ — ಉತ್ಪಾದನಾ ಮುನ್ಸೂಚನೆಗಳಿಗೆ ಬಳಸಲಾಗುವುದಿಲ್ಲ",
        "lstm_demo_note": "ಮುನ್ಸೂಚನೆ ಪ್ರಕ್ರಿಯೆಯನ್ನು ಪರಿಶೀಲಿಸಲು ಈ LSTM ಅನ್ನು ಸಿಂಥೆಟಿಕ್ / ಡೆಮೋ ಡೇಟಾದ ಮೇಲೆ ತರಬೇತಿ ನೀಡಲಾಗಿದೆ. ಇದನ್ನು ಉತ್ಪಾದನಾ ಮುನ್ಸೂಚನೆ ಮಾದರಿಯಿಂದ ಪ್ರತ್ಯೇಕವಾಗಿ ಇರಿಸಲಾಗಿದೆ.",
        "lstm_demo_no_chart": "ಮೂಲ ಡೆಮೋ ಮೌಲ್ಯಮಾಪನ ಡೇಟಾಸೆಟ್ ಈ ಅಪ್ಲಿಕೇಶನ್‌ನಲ್ಲಿ ಸಂಗ್ರಹಿಸಲಾಗಿಲ್ಲವಾದ್ದರಿಂದ Actual ಮತ್ತು Predicted ಮೌಲ್ಯಗಳನ್ನು ಪ್ರದರ್ಶಿಸಲಾಗುವುದಿಲ್ಲ.",
        "forecast_note": "ಐತಿಹಾಸಿಕ holdout ಡೇಟಾದಿಂದ ಮುನ್ಸೂಚನೆಯ ಗುಣಮಟ್ಟವನ್ನು ಅಳೆಯಲಾಗುತ್ತದೆ. ಕಡಿಮೆ MAE/RMSE ಸಾಮಾನ್ಯವಾಗಿ ಉತ್ತಮ; R² 1ಕ್ಕೆ ಹತ್ತಿರವಾಗಿರುವುದು ಸಾಮಾನ್ಯವಾಗಿ ಉತ್ತಮ.",

        "insufficient": "ಮಾಸಿಕ ಮುನ್ಸೂಚನೆ ಮೌಲ್ಯಮಾಪನಕ್ಕೆ ಸಾಕಷ್ಟು ಐತಿಹಾಸಿಕ ಡೇಟಾ ಇಲ್ಲ.",

        "mae": "MAE",
        "rmse": "RMSE",
        "r2": "R²",

        "status": "ಸ್ಥಿತಿ",
        "count": "ಎಣಿಕೆ",
        "normal_status": "ಸಾಮಾನ್ಯ",
        "anomaly_status": "ಅಸಾಮಾನ್ಯ",
    }
}


# ============================================================
# EVALUATION TRANSLATION FUNCTION
# ============================================================

def eval_t(key):

    lang = st.session_state.get(
        "selected_language_code",
        "en"
    )

    language_text = EVAL_TEXT.get(
        lang,
        EVAL_TEXT["en"]
    )

    return language_text.get(
        key,
        EVAL_TEXT["en"].get(key, key)
    )


# ============================================================
# ML MODEL EVALUATION DASHBOARD
# ============================================================

st.markdown("---")

st.header(
    eval_t("title")
)

st.write(
    eval_t("desc")
)


# ============================================================
# TABS
# ============================================================

tab_cat, tab_anom, tab_forecast = st.tabs([
    eval_t("cat"),
    eval_t("anom"),
    eval_t("forecast")
])


# ============================================================
# CATEGORIZER EVALUATION
# ============================================================

with tab_cat:

    st.subheader(
        eval_t("cat_title")
    )

    st.write(
        eval_t("cat_desc")
    )

    try:

        model_path = "models/expense_model.pkl"

        if os.path.exists(model_path):

            eval_model = joblib.load(
                model_path
            )

        else:

            eval_model = None


        # ----------------------------------------------
        # Prepare evaluation data
        # ----------------------------------------------

        eval_data = df[
            ["Description", "Category"]
        ].dropna().copy()

        eval_data["Description"] = (
            eval_data["Description"]
            .astype(str)
        )

        eval_data["Category"] = (
            eval_data["Category"]
            .astype(str)
        )


        # ----------------------------------------------
        # Remove categories with only one sample
        # ----------------------------------------------

        counts = (
            eval_data["Category"]
            .value_counts()
        )

        eval_data = eval_data[
            eval_data["Category"].isin(
                counts[counts >= 2].index
            )
        ]


        # ----------------------------------------------
        # Check if evaluation is possible
        # ----------------------------------------------

        if (
            eval_model is None
            or len(eval_data) < 10
            or eval_data["Category"].nunique() < 2
        ):

            st.info(
                eval_t("not_enough")
            )

        else:

            # ------------------------------------------
            # Predictions
            # ------------------------------------------

            y_true = eval_data["Category"]

            y_pred = eval_model.predict(
                eval_data["Description"].tolist()
            )


            # ------------------------------------------
            # Metrics
            # ------------------------------------------

            acc = accuracy_score(
                y_true,
                y_pred
            )

            prec = precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            )

            rec = recall_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            )

            f1 = f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            )


            # ------------------------------------------
            # Display metrics
            # ------------------------------------------

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                eval_t("accuracy"),
                f"{acc * 100:.2f}%"
            )

            c2.metric(
                eval_t("precision"),
                f"{prec * 100:.2f}%"
            )

            c3.metric(
                eval_t("recall"),
                f"{rec * 100:.2f}%"
            )

            c4.metric(
                eval_t("f1"),
                f"{f1 * 100:.2f}%"
            )


            # ------------------------------------------
            # Confusion Matrix
            # ------------------------------------------

            labels = sorted(
                set(y_true) | set(y_pred)
            )

            cm = confusion_matrix(
                y_true,
                y_pred,
                labels=labels
            )

            st.subheader(
                eval_t("cm")
            )

            fig_cm = px.imshow(
                cm,
                x=labels,
                y=labels,
                text_auto=True,
                aspect="auto",
                title=eval_t("cm")
            )

            fig_cm.update_xaxes(
                title=eval_t("predicted")
            )

            fig_cm.update_yaxes(
                title=eval_t("actual")
            )

            st.plotly_chart(
                fig_cm,
                width="stretch",
                key="ml_eval_confusion_matrix"
            )


    except Exception as e:

        st.warning(
            f"{eval_t('cat_unavailable')} {e}"
        )


# ============================================================
# ANOMALY EVALUATION
# ============================================================

with tab_anom:

    st.subheader(
        eval_t("anom_title")
    )

    try:

        if df.empty:

            st.info(
                eval_t("not_enough")
            )

        else:

            evaluated_anomalies = (
                detect_spending_anomalies(df)
            )

            valid = (
                evaluated_anomalies
                .dropna(subset=["Amount"])
            )


            if len(valid) < 5:

                st.info(
                    eval_t("not_enough")
                )

            else:

                # ------------------------------------------
                # Calculate anomaly statistics
                # ------------------------------------------

                anomaly_count = int(
                    valid["Is_Anomaly"].sum()
                )

                normal_count = int(
                    len(valid) - anomaly_count
                )

                rate = (
                    anomaly_count
                    / len(valid)
                    * 100
                )


                # ------------------------------------------
                # Metrics
                # ------------------------------------------

                a1, a2, a3 = st.columns(3)

                a1.metric(
                    eval_t("analyzed"),
                    len(valid)
                )

                a2.metric(
                    eval_t("normal"),
                    normal_count
                )

                a3.metric(
                    eval_t("anomalies"),
                    anomaly_count
                )


                st.write(
                    f"**{eval_t('rate')}: "
                    f"{rate:.1f}%**"
                )


                # ------------------------------------------
                # Anomaly chart
                # ------------------------------------------

                chart = pd.DataFrame({

                    eval_t("status"): [

                        eval_t(
                            "normal_status"
                        ),

                        eval_t(
                            "anomaly_status"
                        )

                    ],

                    eval_t("count"): [

                        normal_count,
                        anomaly_count

                    ]
                })


                fig = px.pie(

                    chart,

                    names=eval_t(
                        "status"
                    ),

                    values=eval_t(
                        "count"
                    ),

                    title=eval_t(
                        "anom_title"
                    )
                )


                st.plotly_chart(

                    fig,

                    width="stretch",

                    key="ml_eval_anomaly_chart"
                )


                st.info(
                    eval_t("anom_note")
                )


    except Exception as e:

        st.warning(
            f"{eval_t('evaluation_error')} {e}"
        )
# ============================================================
# FORECAST EVALUATION
# ============================================================

with tab_forecast:

    st.subheader(
        eval_t("forecast_title")
    )

    try:

        result = forecast_next_month(df)


        if not result.get("success"):

            st.info(
                eval_t("insufficient")
            )

        else:

            # ------------------------------------------
            # Forecast metrics
            # ------------------------------------------

            m1, m2, m3 = st.columns(3)


            m1.metric(
                eval_t("mae"),
                f"₹{result['mae']:,.2f}"
            )


            m2.metric(
                eval_t("rmse"),
                f"₹{result['rmse']:,.2f}"
            )


            r2_value = result.get(
                "r2"
            )


            m3.metric(

                eval_t("r2"),

                "N/A"
                if pd.isna(r2_value)
                else f"{r2_value:.3f}"
            )


            st.info(
                eval_t("forecast_note")
            )


    except Exception as e:

        st.warning(
            f"{eval_t('evaluation_error')} {e}"
        )


    # ========================================================
    # LSTM DEMO EVALUATION
    # ========================================================

    st.markdown("---")

    st.subheader(
        eval_t("lstm_demo_title")
    )

    st.write(
        eval_t("lstm_demo_dataset")
    )

    st.write(
        eval_t("lstm_demo_model")
    )

    st.warning(
        eval_t("lstm_demo_status")
    )

    st.info(
        eval_t("lstm_demo_note")
    )

    st.caption(
        eval_t("lstm_demo_no_chart")
    )

# ============================================================
# 🧠 PERSONALIZED FINANCIAL RECOMMENDATIONS
# ============================================================

st.markdown("---")

st.header(
    "🧠" + t("personalized_recommendations")
)

st.write(
    t("recommendations_description")
)


# ------------------------------------------------------------
# GET ANOMALY COUNT
# ------------------------------------------------------------

anomaly_count = 0

try:

    from modules.anomaly_detector import detect_spending_anomalies

    if not df.empty:

        anomaly_results = detect_spending_anomalies(df)

        anomaly_count = int(
            anomaly_results["Is_Anomaly"].sum()
        )

except Exception as e:

    anomaly_count = 0


# ------------------------------------------------------------
# GET FORECAST
# ------------------------------------------------------------

forecast_value = None

try:

    forecast_result = forecast_next_month(df)

    if forecast_result.get("success"):

        forecast_value = forecast_result.get(
            "prediction"
        )

except Exception:

    forecast_value = None


# ------------------------------------------------------------
# GET BUDGET
# ------------------------------------------------------------

current_budget = get_budget()


# ------------------------------------------------------------
# GENERATE RECOMMENDATIONS
# ------------------------------------------------------------

recommendations = generate_recommendations(
    df=df,
    budget=current_budget,
    forecast=forecast_value,
    anomaly_count=anomaly_count
)


# ------------------------------------------------------------
# DISPLAY RECOMMENDATIONS
# ------------------------------------------------------------

if not recommendations:

    st.info(
        t("no_recommendations")
    )

else:
  for recommendation in recommendations:

    rec_type = recommendation.get("type", "info")

    original_title = recommendation.get("title", "")
    original_message = recommendation.get("message", "")

    title, message = translate_recommendation(
        original_title,
        original_message,
        language_code
    )

    if rec_type == "danger":
        st.error(
            f"🚨 **{title}**\n\n{message}"
        )

    elif rec_type == "warning":
        st.warning(
            f"⚠️ **{title}**\n\n{message}"
        )

    elif rec_type == "success":
        st.success(
            f"✅ **{title}**\n\n{message}"
        )

    else:
        st.info(
            f"ℹ️ **{title}**\n\n{message}"
        )
# ==========================================
# 📅 MONTHLY SPENDING ANALYSIS
# ==========================================

st.markdown("---")
st.header(t("monthly_analysis"))

if df.empty:

    st.info(t("no_expenses_analysis"))

else:

    monthly_df_source = df.copy()

    monthly_df_source["Month"] = (
        monthly_df_source["Date"]
        .dt.strftime("%Y-%m")
    )

    available_months = sorted(
        monthly_df_source["Month"]
        .dropna()
        .unique(),
        reverse=True
    )

    if available_months:

        selected_month = st.selectbox(
            t("select_month"),
            available_months,
            key="monthly_analysis_month"
        )

        monthly_df = monthly_df_source[
            monthly_df_source["Month"] == selected_month
        ]

        monthly_total = monthly_df["Amount"].sum()

        st.metric(
            t("total_spending"),
            f"₹{monthly_total:,.2f}"
        )

        monthly_category = (
            monthly_df
            .groupby("Category")["Amount"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )

        fig_monthly = px.bar(
            monthly_category,
            x="Category",
            y="Amount",
            title=f"Spending by Category — {selected_month}"
        )

        st.plotly_chart(
            fig_monthly,
            width="stretch",
            key="monthly_category_chart"
        )


# ==========================================
# 🎯 BUDGET VS ACTUAL
# ==========================================

st.markdown("---")
st.header(t("budget_vs_actual"))

budget = get_budget()

if budget <= 0:

    st.info(t("set_budget_comparison"))

elif df.empty:

    st.info(t("add_expenses_comparison"))

else:

    today = datetime.now()

    current_month_spending = monthly_spending(
        today.year,
        today.month
    )

    remaining = budget - current_month_spending

    usage_percentage = (
        current_month_spending / budget
    ) * 100

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            t("monthly_budget"),
            f"₹{budget:,.2f}"
        )

    with col2:
        st.metric(
            t("actual_spending"),
            f"₹{current_month_spending:,.2f}"
        )

    with col3:

        if remaining >= 0:

            st.metric(
                t("remaining"),
                f"₹{remaining:,.2f}"
            )

        else:

            st.metric(
                t("over_budget"),
                f"₹{abs(remaining):,.2f}"
            )

    progress = min(
        current_month_spending / budget,
        1.0
    )

    st.progress(progress)

    st.write(
        f"{t('budget_used_text')}: "
        f"**{usage_percentage:.1f}%**"
    )

    if usage_percentage < 70:

        st.success(t("comfortably_within"))

    elif usage_percentage < 90:

        st.warning(t("approaching_budget"))

    elif usage_percentage <= 100:

        st.warning(t("very_close"))

    else:

        st.error(t("exceeded"))


# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.caption(t("footer"))