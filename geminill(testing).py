import google.generativeai as genai
import os

# --- Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")  # Replace or use env variable
try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")
except Exception as e:
    raise Exception(f"Failed to configure Gemini API: {e}")

# --- Gemini Agent Functions ---
def agent_financial_advice(user_data):
    """Provide personalized financial advice based on user data."""
    prompt = (
        "You are a personal financial advisor. Based on the user's financial data, "
        "provide concise, actionable advice on managing finances better. Focus on budgeting, debt management, and savings. "
        "Use bullet points for clarity."
        f"\nUser Data: {user_data}"
    )
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating financial advice: {e}"

def agent_credit_score_explanation(user_data, prediction_label):
    """Explain why the user's credit score is predicted to change."""
    prompt = (
        "You are a credit score expert. Explain why the user's credit score is predicted to "
        f"{prediction_label or 'unknown'}. Consider factors like credit utilization, repayment history, and debts. "
        "Provide a clear, concise explanation in bullet points based on the user's financial data."
        f"\nUser Data: {user_data}"
    )
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating credit score explanation: {e}"

def agent_financial_health_qna(user_question, user_data):
    """Answer financial health questions using user data context."""
    prompt = (
        "You are a financial health assistant. Answer the user's question about financial health, "
        "using the provided data for context. Be accurate, concise, and user-friendly. Use bullet points where appropriate."
        f"\nUser Data: {user_data}\nUser Question: {user_question}"
    )
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error answering question: {e}"

def agent_credit_score_improvement(user_data):
    """Provide actionable steps to improve the user's credit score."""
    prompt = (
        "You are a credit score improvement advisor. Based on the user's financial data, "
        "provide 3-5 actionable steps to improve their credit score. Be specific, concise, and prioritize impactful actions. "
        "Use bullet points for clarity."
        f"\nUser Data: {user_data}"
    )
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating improvement plan: {e}"

def agent_explain_financial_terms(term):
    """Explain a financial term in simple language."""
    prompt = (
        "You are a finance expert. Explain the financial term '{term}' in simple, clear language "
        "for someone with no finance background. Avoid jargon and provide a practical example."
        f"\nTerm: {term}"
    )
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error explaining term: {e}"

def agent_router(text, user_data, prediction_label=None):
    """Route user query to the appropriate Gemini agent."""
    if not text or not user_data:
        return "Please provide a question and financial data."
    text_lower = text.lower().strip()
    if any(keyword in text_lower for keyword in ["financial advice", "manage money", "budget", "save money"]):
        return agent_financial_advice(user_data)
    elif "credit score" in text_lower and any(keyword in text_lower for keyword in ["explain", "why", "reason"]):
        return agent_credit_score_explanation(user_data, prediction_label)
    elif any(keyword in text_lower for keyword in ["improve credit", "better credit", "increase credit score"]):
        return agent_credit_score_improvement(user_data)
    elif any(keyword in text_lower for keyword in ["what is", "define", "explain term"]):
        term = text_lower.replace("what is", "").replace("define", "").replace("explain term", "").strip()
        return agent_explain_financial_terms(term)
    else:
        return agent_financial_health_qna(text, user_data)