import streamlit as st
from gradio_client import Client
import re

# ---- Config ----
st.set_page_config(page_title="Entrepreneurial Readiness Chatbot", page_icon="🤖", layout="centered")

@st.cache_resource
def load_client():
    return Client("https://mjpsm-Entrepreneurial-readiness.hf.space")

client = load_client()

# Input feature questions (removed runway_months & savings_to_expense_ratio)
feature_questions = [
    ("savings", "How much liquid savings do you currently have (in $)?", float),
    ("monthly_income", "What is your monthly income (in $)?", float),
    ("monthly_bills", "What are your fixed monthly expenses (in $)?", float),
    ("monthly_entertainment_spend", "What is your discretionary monthly spending (in $)?", float),
    ("sales_skills_1to10", "Rate your sales skills on a scale from 1 to 10.", int),
    ("age", "What is your age (in years)?", int),
    ("dependents_count", "How many dependents do you have?", int),
    ("assets", "What is the approximate value of your assets (in $)?", float),
    ("risk_tolerance_1to10", "Rate your risk tolerance from 1 to 10.", int),
    ("confidence_1to10", "Rate your confidence from 1 to 10.", int),
    ("idea_difficulty_1to10", "On a scale from 1 to 10, how difficult is your business idea?", int),
    ("prior_businesses_started_", "How many prior businesses have you started?", int),
    ("prior_exits", "How many prior exits have you had?", int),
    ("time_available_hours_per_week", "How many hours per week can you dedicate to your venture?", int),
]

# ---- Session State ----
if "messages" not in st.session_state:
    st.session_state.messages = []

if "entrepreneur_mode" not in st.session_state:
    st.session_state.entrepreneur_mode = False

if "feature_index" not in st.session_state:
    st.session_state.feature_index = 0

if "user_answers" not in st.session_state:
    st.session_state.user_answers = {}

# ---- Reset Chat ----
if st.button("🔄 Reset Chat"):
    st.session_state.messages = []
    st.session_state.entrepreneur_mode = False
    st.session_state.feature_index = 0
    st.session_state.user_answers = {}
    st.rerun()

# ---- Chat UI ----
st.title("🚀 Entrepreneurial Readiness Chatbot")
st.caption("Chat casually, or tell me you want to be an entrepreneur to begin an assessment.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---- Chat Input ----
if user_input := st.chat_input("Say something..."):
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Detect entrepreneurship intent
    if not st.session_state.entrepreneur_mode and re.search(r"(entrepreneur|start a business|my own company)", user_input, re.IGNORECASE):
        st.session_state.entrepreneur_mode = True
        st.session_state.feature_index = 0
        st.session_state.user_answers = {}
        reply = "Awesome! Let's assess your entrepreneurial readiness. I'll ask you a few questions."
        st.session_state.messages.append({"role": "assistant", "content": reply})
        first_q = feature_questions[0][1]
        st.session_state.messages.append({"role": "assistant", "content": first_q})

    elif st.session_state.entrepreneur_mode:
        # Current feature
        key, question, cast_type = feature_questions[st.session_state.feature_index]
        try:
            st.session_state.user_answers[key] = cast_type(user_input)
        except:
            st.session_state.messages.append(
                {"role": "assistant", "content": f"❌ Please enter a valid {cast_type.__name__} value."}
            )
            st.stop()

        # Next question or prediction
        st.session_state.feature_index += 1
        if st.session_state.feature_index < len(feature_questions):
            next_q = feature_questions[st.session_state.feature_index][1]
            st.session_state.messages.append({"role": "assistant", "content": next_q})
        else:
            # Auto-calculate derived features
            savings = st.session_state.user_answers.get("savings", 0)
            monthly_bills = st.session_state.user_answers.get("monthly_bills", 0)
            monthly_entertainment = st.session_state.user_answers.get("monthly_entertainment_spend", 0)
            expenses = monthly_bills + monthly_entertainment

            if expenses > 0:
                runway_months = min(savings / expenses, 60)
                savings_to_expense_ratio = min(savings / expenses, 12)
            else:
                runway_months = 60
                savings_to_expense_ratio = 12

            st.session_state.user_answers["runway_months"] = runway_months
            st.session_state.user_answers["savings_to_expense_ratio"] = savings_to_expense_ratio

            # Run prediction
            st.session_state.messages.append({"role": "assistant", "content": "✅ Thanks for answering! Analyzing your readiness..."})

            payload = st.session_state.user_answers
            result = client.predict(payload, api_name="/predict")

            # Handle dict-based API response
            if isinstance(result, dict) and "prediction" in result:
                final_text = f"🎯 Your entrepreneurial readiness level is: **{result['prediction']}**"
            else:
                final_text = f"⚠️ Unexpected API response: {result}"

            st.session_state.messages.append({"role": "assistant", "content": final_text})
            st.session_state.entrepreneur_mode = False  # Reset back to chat mode

    else:
        # Regular conversation
        reply = "I'm here to chat! Tell me your goals or say you want to be an entrepreneur."
        st.session_state.messages.append({"role": "assistant", "content": reply})

    st.rerun()
