import os
import json
import tempfile
import streamlit as st
import requests
import uuid

st.set_page_config(page_title="Engineering Assistant", page_icon="🤖", layout="centered")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

st.sidebar.title("⚙️ Control Panel")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select a View:",
    ["💬 Chat Assistant", "📂 Update Knowledge Base"]
)

st.sidebar.markdown("---")

# --- Page 1: Chat Assistant ---
if page == "💬 Chat Assistant":
    st.title("🤖 Engineering RAG Assistant")
    st.caption("Your technical support bot powered by FastAPI, LangGraph, and ChromaDB.")

    # Render existing chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("trace"):
                with st.expander("🔍 Graph Execution Trace", expanded=False):
                    for step in message["trace"]:
                        st.markdown(f"✅ `{step}`")

    user_input = st.chat_input("Ask a technical question about the infrastructure...")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            try:
                bot_answer = ""
                current_trace = []
                answer_placeholder = st.empty()

                # Live streaming trace panel using st.status
                with st.status("🔍 Analyzing your question...", expanded=True) as status:
                    with requests.post(
                        "http://127.0.0.1:8000/chat/stream",
                        json={
                            "session_id": st.session_state.session_id,
                            "question": user_input,
                        },
                        stream=True,
                        timeout=120,
                    ) as response:
                        response.raise_for_status()
                        for line in response.iter_lines():
                            if line and line.startswith(b"data: "):
                                data = json.loads(line[6:].decode())
                                if data.get("trace_step"):
                                    st.write(f"✅ {data['trace_step']}")
                                    current_trace.append(data["trace_step"])
                                if data.get("done"):
                                    bot_answer = data["answer"]
                                    break

                    status.update(label="✅ Done", state="complete", expanded=False)

                answer_placeholder.markdown(bot_answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": bot_answer,
                    "trace": current_trace,
                })

            except requests.exceptions.ConnectionError:
                st.error("Error: Could not connect to the backend. Is your FastAPI server running on port 8000?")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# --- Page 2: Update Knowledge Base ---
elif page == "📂 Update Knowledge Base":
    st.title("📂 Update System Knowledge Base")
    st.markdown("Add new technical materials, structured FAQs, or past team chat logs to the vector database.")

    allowed_md_types = {"Documentation", "FAQs"}

    with st.form("ingestion_form"):
        uploaded_file = st.file_uploader(
            "Upload a file:",
            type=None,
            accept_multiple_files=False,
            help="Drag and drop a file here, or click to browse. Documentation and FAQ files must be .md."
        )
        data_type = st.selectbox(
            "Select Data Category:",
            ["Documentation", "FAQs", "Chat History"]
        )
        submit_button = st.form_submit_button(label="🚀 Upload and Ingest")

    if submit_button:
        if not uploaded_file:
            st.warning("Please upload a file before submitting.")
        else:
            endpoint_map = {
                "Documentation": "http://127.0.0.1:8000/ingest/documentation",
                "FAQs": "http://127.0.0.1:8000/ingest/faq",
                "Chat History": "http://127.0.0.1:8000/ingest/chats"
            }
            target_url = endpoint_map[data_type]

            file_name = uploaded_file.name or "uploaded_file"
            if data_type in allowed_md_types and not file_name.lower().endswith(".md"):
                st.error("Documentation and FAQ uploads must be Markdown files ending in .md.")
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                try:
                    with st.spinner(f"Ingesting uploaded {file_name} into {data_type}..."):
                        response = requests.post(
                            target_url,
                            json={
                                "file_path": tmp_file_path,
                                "original_filename": file_name,
                            }
                        )
                        response.raise_for_status()
                        res_data = response.json()

                        st.success(f"Backend Status: {res_data.get('status').upper()}")
                        st.info(res_data.get("message"))

                        if "extracted_data" in res_data and res_data["extracted_data"]:
                            st.subheader("🤖 Curated QA Content Extracted by LLM:")
                            st.code(res_data["extracted_data"], language="markdown")
                except requests.exceptions.ConnectionError:
                    st.error("Error: Could not connect to the backend. Is your FastAPI server running on port 8000?")
                except requests.exceptions.HTTPError as http_err:
                    st.error(f"Backend Error: {response.json().get('detail', str(http_err))}")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                finally:
                    try:
                        os.unlink(tmp_file_path)
                    except Exception:
                        pass
