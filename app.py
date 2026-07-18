import streamlit as st
import requests
import uuid

# --- Page Configuration ---
st.set_page_config(page_title="Engineering Assistant", page_icon="🤖", layout="centered")

# --- Session Initialization ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar Navigation Pane ---
st.sidebar.title("⚙️ Control Panel")
st.sidebar.markdown("---")

# Navigation Options
page = st.sidebar.radio(
    "Select a View:",
    ["💬 Chat Assistant", "📂 Update Knowledge Base"]
)

st.sidebar.markdown("---")

# --- Page 1: Chat Assistant View ---
if page == "💬 Chat Assistant":
    st.title("🤖 Engineering RAG Assistant")
    st.caption("Your technical support bot powered by FastAPI and ChromaDB.")

    # Render Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input Box
    user_input = st.chat_input("Ask a technical question about the infrastructure...")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("assistant"):
            with st.spinner("Searching documentation and thinking..."):
                try:
                    response = requests.post(
                        "http://127.0.0.1:8000/chat",
                        json={
                            "session_id": st.session_state.session_id,
                            "question": user_input
                        }
                    )
                    response.raise_for_status()
                    bot_answer = response.json().get("answer", "No answer found.")
                    st.markdown(bot_answer)
                    st.session_state.messages.append({"role": "assistant", "content": bot_answer})
                    
                except requests.exceptions.ConnectionError:
                    st.error("Error: Could not connect to the backend. Is your FastAPI server running on port 8000?")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

# --- Page 2: Update Knowledge Base View ---
elif page == "📂 Update Knowledge Base":
    st.title("📂 Update System Knowledge Base")
    st.markdown("Add new technical materials, structured FAQs, or past team chat logs to the vector database.")

    # Form components for ingestion inputs
    with st.form("ingestion_form"):
        file_path_input = st.text_input(
            "Enter Local File Path:",
            placeholder="e.g., data/docs/payment_gateway.md"
        )
        
        data_type = st.selectbox(
            "Select Data Category:",
            ["Documentation", "FAQs", "Chat History"]
        )
        
        submit_button = st.form_submit_button(label="🚀 Upload and Ingest")

    if submit_button:
        if not file_path_input.strip():
            st.warning("Please enter a valid file path before submitting.")
        else:
            # Map the dropdown selection to the corresponding backend URL endpoint
            endpoint_map = {
                "Documentation": "http://127.0.0.1:8000/ingest/documentation",
                "FAQs": "http://127.0.0.1:8000/ingest/faq",
                "Chat History": "http://127.0.0.1:8000/ingest/chats"
            }
            
            target_url = endpoint_map[data_type]
            
            with st.spinner(f"Ingesting into {data_type}..."):
                try:
                    response = requests.post(
                        target_url,
                        json={"file_path": file_path_input.strip()}
                    )
                    response.raise_for_status()
                    res_data = response.json()
                    
                    # Display success metrics
                    st.success(f"Backend Status: {res_data.get('status').upper()}")
                    st.info(res_data.get("message"))
                    
                    # If chat history was processed, show the curated LLM output summary block
                    if "extracted_data" in res_data and res_data["extracted_data"]:
                        st.subheader("🤖 Curated QA Content Extracted by LLM:")
                        st.code(res_data["extracted_data"], language="markdown")
                        
                except requests.exceptions.ConnectionError:
                    st.error("Error: Could not connect to the backend. Is your FastAPI server running on port 8000?")
                except requests.exceptions.HTTPError as http_err:
                    st.error(f"Backend Error: {response.json().get('detail', str(http_err))}")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")