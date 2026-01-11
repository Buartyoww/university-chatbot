import streamlit as st
import pandas as pd
import joblib
import os
import re
import google.generativeai as genai

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="University AI Assistant", page_icon="🎓", layout="centered")

# --- 🔑 SETUP API KEY ---
try:
    # Tries to get the key from Streamlit Cloud Secrets
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    # Fallback for local testing (REPLACE THIS WITH YOUR KEY IF TESTING LOCALLY)
    # BUT REMOVE IT BEFORE UPLOADING TO GITHUB
    GOOGLE_API_KEY = "AIzaSyAv6AB0eToxMx4puRAeCcIN8yJxMghMB4Q"

genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# --- LOAD SYSTEM ---
@st.cache_resource
def load_system():
    MODEL_FILE = 'university_model.pkl'
    DATA_FILE = 'dataset1.csv'
    
    if not os.path.exists(MODEL_FILE) or not os.path.exists(DATA_FILE):
        return None, None
    
    model = joblib.load(MODEL_FILE)
    df = pd.read_csv(DATA_FILE)
    return model, df

ai_model, df = load_system()

# --- CHAT LOGIC ---
def get_bot_response(message):
    if ai_model is None:
        return "❌ Error: System files missing. Please upload university_model.pkl"

    # Clean message
    clean_message = re.sub(r'[^\w\s]', '', message.lower())
    message_words = clean_message.split()
    found_person = None

    # ======================================================
    # 🔍 PRIORITY 1: CHECK UNIVERSITY DATA (Strict Search)
    # ======================================================
    ignored_words = ["mr", "ms", "mrs", "dr", "prof", "engr", "chef", "officer", "staff", "sir", "maam"]

    for index, row in df.iterrows():
        db_name = str(row['Name']).lower()
        clean_name = db_name.replace(".", "")
        name_parts = clean_name.split()
        
        for part in name_parts:
            if part not in ignored_words:
                if part in message_words:
                    found_person = row
                    break 
        if found_person is not None:
            break

    if found_person is not None:
        return (f"### 👤 Personnel Found\n"
                f"---\n"
                f"**📛 Name:** &nbsp; {found_person['Name']}\n\n"
                f"**💼 Role:** &nbsp;&nbsp;&nbsp; {found_person['Role']}\n\n"
                f"**🏛️ Dept:** &nbsp;&nbsp;&nbsp; {found_person['Department']}\n\n"
                f"**📍 Location:** {found_person['Location']}")

    # ======================================================
    # 🔍 PRIORITY 2: UNIVERSITY INTENT (AI Classifier)
    # ======================================================
    try:
        all_probs = ai_model.predict_proba([message])[0]
        confidence = max(all_probs)
        
        # If confidence is > 35%, it's likely a school question
        if confidence >= 0.35:
            prediction = ai_model.predict([message])[0]
            parts = prediction.split('_')
            dept = parts[0]
            category = parts[1]

            dept_data = df[df['Department'] == dept]

            if not dept_data.empty:
                if category == "OrgChart":
                    dean = dept_data[dept_data['Role'] == 'Dean']['Name'].values[0] if len(dept_data[dept_data['Role'] == 'Dean']) > 0 else "N/A"
                    chair = dept_data[dept_data['Role'] == 'Chairperson']['Name'].values[0] if len(dept_data[dept_data['Role'] == 'Chairperson']) > 0 else "N/A"
                    faculty = dept_data[dept_data['Role'] == 'Faculty']['Name'].values
                    
                    faculty_tree = ""
                    if len(faculty) > 0:
                        for i, f in enumerate(faculty):
                            if i == len(faculty) - 1:
                                faculty_tree += f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── {f}\n\n"
                            else:
                                faculty_tree += f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── {f}\n\n"
                    else:
                        faculty_tree = "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── (No faculty listed)"

                    return (f"### 📊 {dept} Organizational Structure\n"
                            f"---\n"
                            f"**🏢 Dean:** {dean}\n\n"
                            f"│\n\n"
                            f"└── **👤 Chairperson:** {chair}\n\n"
                            f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│\n\n"
                            f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── **👥 Faculty Members:**\n\n"
                            f"{faculty_tree}")

                elif category == "Location":
                    loc = dept_data[dept_data['Role'] == 'Dean']['Location'].values[0]
                    return f"📍 **Location Found:**\n\nThe {dept} department is located at **{loc}**."
                
                else: 
                    results = dept_data[dept_data['Role'] == category]['Name'].values
                    if len(results) > 0:
                        names = ", ".join(results)
                        return f"Here is the information:\n\n**{category} of {dept}:** {names}"

    except Exception:
        pass # If Local AI fails, go to Gemini

    # ======================================================
    # 🤖 PRIORITY 3: FALLBACK TO GEMINI
    # ======================================================
    try:
        prompt = f"User question: {message}\nAnswer nicely and concisely. If it's a greeting, be friendly."
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception:
        return "I am not sure about that, and I couldn't reach Google Gemini."

# --- MAIN UI ---
st.title("✨ University Hybrid Bot")
st.caption("Ask me about the School... or anything else!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        response = get_bot_response(prompt)

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
