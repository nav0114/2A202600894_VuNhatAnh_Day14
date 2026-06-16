import os
import sys
from pathlib import Path
import streamlit as st

# Setup paths to import from src
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

# Ensure stdout encodes to utf-8 (prevents encoding issues on Windows consoles)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.task10_generation import generate_with_citation_and_memory, generate_with_citation_and_memory_stream

# Page Configuration
st.set_page_config(
    page_title="DRUG LAW // Hệ thống Tra cứu & Tư vấn Pháp luật Ma tuý",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom Neo-Brutalism CSS styling matching mockup_ui_ver_2.html
custom_css = """
<style>
    /* Base Page Styling overrides */
    .stApp {
        background-color: #f9fafb !important;
        color: #121314 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    }

    /* Remove streamlit header decoration */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
        border-bottom: 2px solid #121314 !important;
    }
    
    /* Header layout styling */
    .custom-header {
        background-color: #ffffff;
        border: 2px solid #121314;
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 16px 24px;
        margin-bottom: 24px;
    }
    
    .brand-title {
        font-size: 18px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #121314;
        margin: 0;
    }
    
    .system-status {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        background-color: #10b981; /* green dot */
        display: inline-block;
    }
    
    /* Panel Containers */
    .panel-container {
        background-color: #ffffff;
        border: 2px solid #121314;
        padding: 24px;
        height: 80vh;
        display: flex;
        flex-direction: column;
        overflow-y: auto;
    }
    
    .panel-title {
        font-size: 12px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #121314;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .panel-title::after {
        content: '';
        flex-grow: 1;
        height: 2px;
        background-color: #121314;
    }
    
    /* Chat bubbles matching mockup */
    .chat-bubble-container {
        display: flex;
        flex-direction: column;
        gap: 16px;
        overflow-y: auto;
        flex-grow: 1;
        margin-bottom: 20px;
        padding-right: 8px;
    }
    
    .custom-chat-message {
        display: flex;
        gap: 16px;
        padding: 16px;
        border: 2px solid #121314;
        background-color: #ffffff;
    }
    
    .custom-chat-message.user {
        background-color: #ffffff;
    }
    
    .custom-chat-message.assistant {
        background-color: #f0fdf4; /* status-low-bg light green */
        border-left: 8px solid #121314;
    }
    
    .message-avatar {
        width: 32px;
        height: 32px;
        border: 1.5px solid #121314;
        background-color: #f3f4f6;
        color: #121314;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 11px;
        font-family: ui-monospace, SFMono-Regular, monospace;
    }
    
    .custom-chat-message.assistant .message-avatar {
        background-color: #121314;
        color: #ffffff;
    }
    
    .message-meta {
        font-family: ui-monospace, SFMono-Regular, monospace;
        font-size: 10px;
        text-transform: uppercase;
        color: #4a4d4f;
        margin-bottom: 6px;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    
    .message-body {
        font-size: 14px;
        line-height: 1.6;
        color: #121314;
    }
    
    /* Telemetry classes */
    .telemetry-text {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
        font-weight: 500;
    }
    
    /* Input Styling overrides */
    div[data-testid="stChatInput"] {
        border: 2px solid #121314 !important;
        border-radius: 0px !important;
        background-color: #ffffff !important;
    }
    
    div[data-testid="stChatInput"] textarea {
        color: #121314 !important;
        font-family: -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-size: 14px !important;
    }
    
    /* Metrics panel styling */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 12px;
        margin-bottom: 24px;
    }
    
    .metric-card {
        border: 2px solid #121314;
        background-color: #ffffff;
        padding: 12px;
        text-align: left;
    }
    
    .metric-label {
        font-size: 9px;
        font-weight: 700;
        text-transform: uppercase;
        color: #4a4d4f;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    
    .metric-value {
        font-size: 18px;
        font-weight: 800;
        color: #121314;
    }
    
    /* Custom Table Styling */
    .custom-table {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 20px;
    }
    
    .custom-table th {
        background-color: #f3f4f6;
        border: 1px solid #121314;
        padding: 8px 12px;
        font-size: 10px;
        font-weight: 800;
        text-transform: uppercase;
        color: #121314;
        text-align: left;
    }
    
    .custom-table td {
        border: 1px solid #121314;
        padding: 8px 12px;
        font-size: 12px;
        color: #121314;
    }
    
    .custom-table tr:hover {
        background-color: #fafbfc;
    }
    
    /* Badges */
    .status-badge {
        display: inline-block;
        padding: 2px 6px;
        font-size: 9px;
        font-weight: 800;
        text-transform: uppercase;
        border: 1px solid #121314;
    }
    
    .status-badge.active {
        background-color: #e6fcf5;
        color: #0ca678;
        border-color: #0ca678;
    }
    
    .status-badge.secondary {
        background-color: #f1f3f5;
        color: #4a4d4f;
        border-color: #4a4d4f;
    }
    
    .log-container {
        background-color: #fafbfc;
        border: 1.5px dashed #121314;
        padding: 12px;
        font-family: ui-monospace, SFMono-Regular, monospace;
        font-size: 11px;
        color: #4a4d4f;
        white-space: pre-wrap;
        word-break: break-all;
    }
    
    /* Custom table style for streamlit markdown tables (Neo-Brutalism) */
    div[data-testid="stMarkdownContainer"] table {
        width: 100% !important;
        border-collapse: collapse !important;
        border: 2px solid #121314 !important;
        margin-bottom: 20px !important;
    }
    div[data-testid="stMarkdownContainer"] th {
        background-color: #f3f4f6 !important;
        border: 2px solid #121314 !important;
        padding: 8px 12px !important;
        font-size: 11px !important;
        font-weight: 800 !important;
        text-transform: uppercase !important;
        color: #121314 !important;
        text-align: left !important;
    }
    div[data-testid="stMarkdownContainer"] td {
        border: 1.5px solid #121314 !important;
        padding: 8px 12px !important;
        font-size: 13px !important;
        color: #121314 !important;
    }
    div[data-testid="stMarkdownContainer"] tr:hover {
        background-color: #fafbfc !important;
    }
    div[data-testid="stMarkdownContainer"] blockquote {
        border-left: 6px solid #121314 !important;
        background-color: #f0fdf4 !important; /* light green matching assistant */
        padding: 12px 16px !important;
        margin: 16px 0 !important;
        color: #121314 !important;
        font-style: italic !important;
        font-size: 14px !important;
        border: 2px solid #121314 !important;
        border-left: 8px solid #121314 !important;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_query_metrics" not in st.session_state:
    st.session_state.last_query_metrics = {
        "standalone_query": "-",
        "chunks_retrieved": 0,
        "source": "-",
        "fallback_used": "NO",
        "latency": "-"
    }

# Mockup documents list matching our system's source documents
documents_list = [
    {"name": "luat-phong-chong-ma-tuy-2021.pdf", "type": "Legal", "status": "ACTIVE"},
    {"name": "bo-luat-hinh-su-2015.pdf", "type": "Legal", "status": "ACTIVE"},
    {"name": "nghi-dinh-chi-tiet-huong-dan-2021.pdf", "type": "Legal", "status": "ACTIVE"},
    {"name": "thong-tu-lien-tich-2015.pdf", "type": "Legal", "status": "ACTIVE"},
]

# Custom Header HTML
header_html = """
<div class="custom-header">
    <div class="brand-title">DRUG LAW // Hệ thống Tra cứu & Tư vấn Pháp luật Ma tuý</div>
    <div class="system-status">
        <span class="status-dot"></span>
        <span>AGENT CONSOLE: ACTIVE</span>
    </div>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

# Main Grid Layout
col_chat, col_stats = st.columns([0.60, 0.40])

with col_chat:
    st.markdown('<div class="panel-title">CHAT CONSOLE // Kênh tư vấn trực tuyến</div>', unsafe_allow_html=True)
    
    # Render chat messages inside a single panel scroll wrapper
    chat_container_html = '<div class="panel-container" style="height: 68vh; margin-bottom: 12px;">'
    chat_container_html += '<div class="chat-bubble-container">'
    if not st.session_state.chat_history:
        chat_container_html += (
            '<div class="custom-chat-message assistant">'
            '<div class="message-avatar">AI</div>'
            '<div class="message-content">'
            '<div class="message-meta">ASSISTANT // CORE</div>'
            '<div class="message-body">'
            'Xin chào! Tôi là Trợ lý Pháp luật Ma tuý. Bạn có thể hỏi tôi bất kỳ câu hỏi nào liên quan đến các văn bản pháp luật phòng chống ma túy tại Việt Nam.'
            '</div>'
            '</div>'
            '</div>'
        )
    else:
        for idx, msg in enumerate(st.session_state.chat_history):
            role_class = "user" if msg["role"] == "user" else "assistant"
            avatar_txt = "U" if msg["role"] == "user" else "AI"
            role_name = "USER // ID_092" if msg["role"] == "user" else "ASSISTANT // RAG_AGENT"
            
            # Escape HTML characters to prevent breaking markdown/rendering
            content_escaped = msg["content"].replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
            
            # Format citations to look like neat badges
            import re
            content_formatted = re.sub(
                r'(\[[^\]]+\])',
                r'<span class="citation-link">\1</span>',
                content_escaped
            )
            
            chat_container_html += (
                f'<div class="custom-chat-message {role_class}">'
                f'<div class="message-avatar">{avatar_txt}</div>'
                f'<div class="message-content">'
                f'<div class="message-meta">{role_name}</div>'
                f'<div class="message-body">{content_formatted}</div>'
                f'</div>'
                f'</div>'
            )
    chat_container_html += '</div></div>'
    st.markdown(chat_container_html, unsafe_allow_html=True)

    # Placeholder for streaming (only used when pending_generation is True)
    streaming_placeholder = st.empty()

    # Chat input box at the bottom
    user_query = st.chat_input("Nhập câu hỏi pháp luật của bạn ở đây...")
    
    # Check if there is a pending generation from a previous submit rerun
    if st.session_state.get("pending_generation"):
        st.session_state.pending_generation = False
        
        # Append empty assistant message to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": ""})
        
        import time
        import re
        
        start_time = time.time()
        
        # Show a loading spinner during initial retrieval (before LLM starts streaming)
        with st.spinner("Đang truy xuất tài liệu pháp lý và kết nối AI..."):
            try:
                # Initialize streaming generator
                stream_generator = generate_with_citation_and_memory_stream(
                    query=st.session_state.chat_history[-2]["content"], # The user's query we just rendered
                    history=st.session_state.chat_history[:-2] # Chat history before this turn
                )
                
                # Retrieve metadata first (instantly)
                metadata_dict = next(stream_generator)
                sources = metadata_dict["sources"]
                retrieval_source = metadata_dict["retrieval_source"]
                standalone_query = metadata_dict["standalone_query"]
                
                elapsed = time.time() - start_time
                fallback_val = "YES" if retrieval_source == "pageindex" else "NO"
                st.session_state.last_query_metrics = {
                    "standalone_query": standalone_query,
                    "chunks_retrieved": len(sources),
                    "source": retrieval_source,
                    "fallback_used": fallback_val,
                    "latency": f"{elapsed:.2f}s (retrieval)"
                }
            except Exception as e:
                error_msg = f"Lỗi hệ thống: {str(e)}"
                st.session_state.chat_history[-1]["content"] = error_msg
                st.rerun()
        
        # Now stream content
        full_text = ""
        try:
            for event in stream_generator:
                if event["type"] == "content":
                    full_text += event["delta"]
                    st.session_state.chat_history[-1]["content"] = full_text
                    
                    total_elapsed = time.time() - start_time
                    st.session_state.last_query_metrics["latency"] = f"{total_elapsed:.2f}s"
                    
                    content_escaped = full_text.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
                    content_formatted = re.sub(
                        r'(\[[^\]]+\])',
                        r'<span class="citation-link">\1</span>',
                        content_escaped
                    )
                    
                    streamed_bubble_html = (
                        f'<div class="custom-chat-message assistant">'
                        f'<div class="message-avatar">AI</div>'
                        f'<div class="message-content">'
                        f'<div class="message-meta">ASSISTANT // RAG_AGENT (Đang phản hồi...)</div>'
                        f'<div class="message-body">{content_formatted}</div>'
                        f'</div>'
                        f'</div>'
                    )
                    streaming_placeholder.markdown(streamed_bubble_html, unsafe_allow_html=True)
                    
            # Update final latency
            total_elapsed = time.time() - start_time
            st.session_state.last_query_metrics["latency"] = f"{total_elapsed:.2f}s"
            
        except Exception as e:
            error_msg = f"Lỗi hệ thống: {str(e)}"
            st.session_state.chat_history[-1]["content"] = error_msg
            
        st.rerun()

    if user_query:
        # Append User Message to session state and trigger rerun to draw it
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.session_state.pending_generation = True
        st.rerun()

with col_stats:
    tab_monitor, tab_eval = st.tabs(["🖥️ SYSTEM MONITOR", "📊 RAG EVALUATION"])
    
    with tab_monitor:
        st.markdown('<div class="panel-title">SYSTEM MONITOR // Giám sát hệ thống</div>', unsafe_allow_html=True)
        
        # Wrap in metrics panel
        stats_html = '<div class="panel-container" style="height: 70vh;">'
        
        # General Metrics Grid
        stats_html += (
            '<div class="metric-grid">'
            '<div class="metric-card">'
            '<div class="metric-label">LLM ENGINE</div>'
            '<div class="metric-value telemetry-text">Gemini 3.1 FL</div>'
            '</div>'
            '<div class="metric-card">'
            '<div class="metric-label">EMBEDDING MODEL</div>'
            '<div class="metric-value telemetry-text">MiniLM-L6</div>'
            '</div>'
            '<div class="metric-card">'
            '<div class="metric-label">RETRIEVAL METHOD</div>'
            '<div class="metric-value telemetry-text">Hybrid + Fallback</div>'
            '</div>'
            '<div class="metric-card">'
            '<div class="metric-label">SYSTEM LATENCY</div>'
            f'<div class="metric-value telemetry-text">{st.session_state.last_query_metrics["latency"]}</div>'
            '</div>'
            '</div>'
        )
        
        # Document Directory Table
        stats_html += '<div style="font-size: 11px; font-weight: 800; text-transform: uppercase; margin-bottom: 8px;">DOCUMENT REPOSITORY (Cơ sở dữ liệu pháp luật)</div>'
        
        table_rows = ""
        for doc in documents_list:
            table_rows += (
                '<tr>'
                f'<td>{doc["name"]}</td>'
                f'<td class="telemetry-text">{doc["type"]}</td>'
                f'<td><span class="status-badge active">{doc["status"]}</span></td>'
                '</tr>'
            )
            
        stats_html += (
            '<table class="custom-table">'
            '<thead>'
            '<tr>'
            '<th>Document Name</th>'
            '<th>Type</th>'
            '<th>Status</th>'
            '</tr>'
            '</thead>'
            '<tbody>'
            f'{table_rows}'
            '</tbody>'
            '</table>'
        )
        
        # Last Query Retrieval Log
        stats_html += '<div style="font-size: 11px; font-weight: 800; text-transform: uppercase; margin-bottom: 8px;">RETRIEVAL METRICS (Nhật ký truy xuất hiện tại)</div>'
        
        metrics = st.session_state.last_query_metrics
        log_content = (
            f"Standalone Query:  {metrics['standalone_query']}<br>"
            f"Chunks Retrieved:  {metrics['chunks_retrieved']}<br>"
            f"Primary Source:    {metrics['source'].upper()}<br>"
            f"Fallback PageIndex: {metrics['fallback_used']}"
        )
        
        stats_html += f'<div class="log-container">{log_content}</div>'
        stats_html += '</div>'
        
        st.markdown(stats_html, unsafe_allow_html=True)
        
        # Clear conversation history button styled neatly
        st.write("") # Spacer
        if st.button("CLEAR CONVERSATION HISTORY", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.last_query_metrics = {
                "standalone_query": "-",
                "chunks_retrieved": 0,
                "source": "-",
                "fallback_used": "NO",
                "latency": "-"
            }
            st.rerun()
            
    with tab_eval:
        st.markdown('<div class="panel-title">EVALUATION REPORT // Đánh giá chất lượng RAG</div>', unsafe_allow_html=True)
        
        # Read the evaluation report from results.md
        results_file = PROJECT_DIR / "group_project" / "evaluation" / "results.md"
        if results_file.exists():
            try:
                with open(results_file, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Render content inside the neo-brutalism panel-container
                st.markdown(f'<div class="panel-container" style="height: 75vh; overflow-y: auto; padding: 20px;">\n\n{content}\n\n</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Lỗi khi đọc file kết quả đánh giá: {e}")
        else:
            st.markdown('<div class="panel-container" style="height: 75vh; display: flex; align-items: center; justify-content: center; text-align: center;"><div style="font-family: ui-monospace, monospace; font-size: 13px; color: #4a4d4f;">⏳ Đang thực hiện chạy đánh giá RAG hoặc file kết quả chưa được tạo. Hãy quay lại sau khi tiến trình nền hoàn tất.</div></div>', unsafe_allow_html=True)
