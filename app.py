import streamlit as st
import pandas as pd
import joblib
import os
import re

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="University AI Assistant",
    page_icon="🎓",
    layout="centered"
)

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
        return "❌ Error: System files missing. Please run train_model.py first."

    # Clean message for strict search
    clean_message = re.sub(r'[^\w\s]', '', message.lower())
    message_words = clean_message.split()

    found_person = None

    # ======================================================
    # 🔍 STEP 1: STRICT NAME SEARCH
    # ======================================================
    ignored_words = ["mr", "ms", "mrs", "dr", "prof", "engr", "chef", "officer", "staff", "sir", "maam"]

    for index, row in df.iterrows():
        db_name = str(row['Name']).lower()
        clean_name = db_name.replace(".", "")
        name_parts = clean_name.split()
        
        for part in name_parts:
            if part not in ignored_words:
                # Match WHOLE WORDS only (Fixes the "Art" vs "Chart" bug)
                if part in message_words:
                    found_person = row
                    break 
        if found_person is not None:
            break

    # ======================================================
    # 🗣️ STEP 2: GENERATE RESPONSE
    # ======================================================
    
    # 🎨 STYLE 1: PERSONNEL CARD (If a person is found)
    if found_person is not None:
        return (f"### 👤 Personnel Found\n"
                f"---\n"  # Horizontal Line
                f"**📛 Name:** &nbsp; {found_person['Name']}\n\n"
                f"**💼 Role:** &nbsp;&nbsp;&nbsp; {found_person['Role']}\n\n"
                f"**🏛️ Dept:** &nbsp;&nbsp;&nbsp; {found_person['Department']}\n\n"
                f"**📍 Location:** {found_person['Location']}")

    # 🎨 STYLE 2: ORG CHART TREE (If AI predicts OrgChart)
    try:
        all_probs = ai_model.predict_proba([message])[0]
        confidence = max(all_probs)
        
        if confidence < 0.35:
            return "I am not sure about that. 😅 I can only answer questions about CFMS, CTE, CBM, CAS, and CHM."
        
        prediction = ai_model.predict([message])[0]
        parts = prediction.split('_')
        dept = parts[0]
        category = parts[1]

        dept_data = df[df['Department'] == dept]

        if dept_data.empty:
            return f"I couldn't find any data for {dept}."

        if category == "OrgChart":
            dean = dept_data[dept_data['Role'] == 'Dean']['Name'].values[0] if len(dept_data[dept_data['Role'] == 'Dean']) > 0 else "N/A"
            chair = dept_data[dept_data['Role'] == 'Chairperson']['Name'].values[0] if len(dept_data[dept_data['Role'] == 'Chairperson']) > 0 else "N/A"
            faculty = dept_data[dept_data['Role'] == 'Faculty']['Name'].values
            
            # --- BUILD THE TREE VISUAL ---
            faculty_tree = ""
            if len(faculty) > 0:
                for i, f in enumerate(faculty):
                    # Use &nbsp; (Non-Breaking Space) to force indentation in Streamlit
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
                return f"Here is the information you requested:\n\n**{category} of {dept}:** {names}"
            else:
                return f"I checked {dept}, but I couldn't find a name for {category}."

    except Exception as e:
        return f"Error processing request: {e}"

# --- MAIN UI ---
st.title("✨ University AI Assistant")
st.caption("Ask me about CFMS, CTE, CBM, CAS, or CHM (Dean, Location, Faculty)")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle Input
if prompt := st.chat_input("Where is Dr. Mendez?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = get_bot_response(prompt)

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})