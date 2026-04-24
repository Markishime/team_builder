import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
import json
import re
import io
import requests
import hashlib
import time

# Configure Gemini API
client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
GEMINI_MODEL = "gemini-2.5-flash"
generation_config = types.GenerateContentConfig(
    temperature=0.7,
    top_p=0.9,
    top_k=40,
    max_output_tokens=2048
)

# Custom CSS for enhanced UI with fixed scrollbar
st.markdown("""
    <style>
    /* ── Scrollbar ── */
    body {
        overflow-y: auto;
        scrollbar-width: thin;
        scrollbar-color: #4a90e2 #e0e6ed;
    }
    body::-webkit-scrollbar { width: 12px; }
    body::-webkit-scrollbar-track { background: #e0e6ed; border-radius: 10px; }
    body::-webkit-scrollbar-thumb {
        background: #4a90e2;
        border-radius: 10px;
        border: 3px solid #e0e6ed;
    }
    body::-webkit-scrollbar-thumb:hover { background: #357abd; }

    /* ── Button ── */
    .stButton>button {
        background: linear-gradient(90deg, #4a90e2, #357abd) !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        border: none !important;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #357abd, #4a90e2) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2) !important;
    }

    /* ── Headings – target Streamlit's markdown containers directly ── */
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
    h1, h2, h3 {
        color: #1a2535 !important;
        font-family: 'Helvetica Neue', sans-serif !important;
    }
    [data-testid="stMarkdownContainer"] h1,
    .stMarkdown h1, h1 {
        font-size: 2.8em !important;
        text-align: center !important;
        margin-bottom: 0.5em !important;
    }

    /* ── Labels ── */
    label, .stTextArea label, .stNumberInput label,
    [data-testid="stWidgetLabel"] {
        font-weight: 600 !important;
        color: #2c3e50 !important;
        font-size: 1.05em !important;
    }
    .stTextArea textarea {
        border-radius: 10px !important;
        border: 2px solid #c8d6e5 !important;
        transition: border-color 0.3s ease !important;
    }
    .stTextArea textarea:focus { border-color: #4a90e2 !important; }

    /* ── Card ── */
    .card {
        background-color: #ffffff !important;
        color: #2c3e50 !important;
        border-radius: 12px !important;
        padding: 20px !important;
        margin-bottom: 20px !important;
        border: 1px solid #c8d6e5 !important;
        transition: all 0.3s ease !important;
    }
    .card strong, .card p, .card * { color: #2c3e50 !important; }
    .card:hover {
        box-shadow: 0 4px 15px rgba(0,0,0,0.12) !important;
        transform: translateY(-3px) !important;
    }

    /* ── Header / subheader ── */
    .header-section { text-align: center !important; margin-bottom: 2em !important; }
    .subheader {
        color: #3d5a78 !important;
        font-size: 1.2em !important;
        line-height: 1.6 !important;
    }

    /* ── Highlight box – force light style, no OS dark mode bleed ── */
    .highlight {
        background-color: #deeafd !important;
        color: #1a2535 !important;
        padding: 12px 16px !important;
        border-radius: 8px !important;
        margin: 10px 0 !important;
        border-left: 4px solid #4a90e2 !important;
    }
    .highlight p, .highlight li, .highlight ul,
    .highlight * { color: #1a2535 !important; }

    /* ── Dark mode: ONLY via Streamlit's data-theme, NOT OS media query ── */
    [data-theme="dark"] [data-testid="stMarkdownContainer"] h1,
    [data-theme="dark"] [data-testid="stMarkdownContainer"] h2,
    [data-theme="dark"] [data-testid="stMarkdownContainer"] h3,
    [data-theme="dark"] h1, [data-theme="dark"] h2, [data-theme="dark"] h3 {
        color: #e8eaf0 !important;
    }
    [data-theme="dark"] label,
    [data-theme="dark"] [data-testid="stWidgetLabel"] { color: #a8bcd4 !important; }
    [data-theme="dark"] .card {
        background-color: #1e2d3d !important;
        color: #e0e8f0 !important;
        border-color: #3a5068 !important;
    }
    [data-theme="dark"] .card strong,
    [data-theme="dark"] .card p,
    [data-theme="dark"] .card * { color: #e0e8f0 !important; }
    [data-theme="dark"] .highlight {
        background-color: #1a2d4a !important;
        color: #cfd8e8 !important;
    }
    [data-theme="dark"] .highlight * { color: #cfd8e8 !important; }
    [data-theme="dark"] .subheader { color: #8aaac8 !important; }
    </style>
""", unsafe_allow_html=True)

# Utility functions (unchanged)
def extract_and_parse_json(text):
    try:
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if not json_match:
            return None, False
        return json.loads(json_match.group()), True
    except json.JSONDecodeError:
        return None, False

def validate_salary_json(json_data):
    if not isinstance(json_data, dict) or "salary_comparison" not in json_data:
        return None, False
    salaries = json_data["salary_comparison"]
    if not all(key in salaries for key in ["philippines", "united_states"]):
        return None, False
    try:
        ph_salary = float(salaries["philippines"])
        us_salary = float(salaries["united_states"])
        if not (0 < ph_salary < 10000 and 0 < us_salary < 10000 and ph_salary < us_salary):
            return None, False
        return json_data, True
    except (ValueError, TypeError):
        return None, False

def validate_input(text):
    if not text or len(text.strip()) < 10:
        return False
    generic_phrases = ["general", "n/a", "not sure", "don't know"]
    irrelevant_keywords = ["unrelated", "out of context", "irrelevant"]
    relevant_keywords = ["job", "hire", "business", "project", "team", "company", "role", "position", "expertise"]
    text_lower = text.lower()
    if any(phrase in text_lower for phrase in generic_phrases + irrelevant_keywords):
        return False
    return any(keyword in text_lower for keyword in relevant_keywords)

def analyze_url_content(url):
    try:
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            return None
        text = response.text
        about_us_start = text.lower().find('about us') or text.lower().find('our company')
        if about_us_start != -1:
            about_us_end = text.lower().find('</div>', about_us_start)
            return text[about_us_start:about_us_end] if about_us_end != -1 else text[about_us_start:about_us_start+500]
        return text[:500]
    except:
        return None

def cache_result(key, func, *args, timeout=3600):
    if key not in st.session_state or (st.session_state.get(f"{key}_time", 0) + timeout) < time.time():
        st.session_state[key] = func(*args)
        st.session_state[f"{key}_time"] = time.time()
    return st.session_state[key]

def generate_content(prompt):
    max_retries = 5
    delay = 5
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=generation_config
            ).text
        except genai_errors.ClientError as e:
            if e.status == "RESOURCE_EXHAUSTED" and attempt < max_retries - 1:
                time.sleep(delay * (2 ** attempt))
            else:
                raise

def render_html_table(df):
    """Render a DataFrame as an HTML table without using pyarrow."""
    headers = "".join(f"<th style='padding:8px 12px;text-align:left;border-bottom:2px solid #c8d6e5;'>{col}</th>" for col in df.columns)
    rows = ""
    for i, row in df.iterrows():
        bg = "#f8fafc" if i % 2 == 0 else "#ffffff"
        cells = "".join(f"<td style='padding:8px 12px;border-bottom:1px solid #e0e6ed;'>{val}</td>" for val in row)
        rows += f"<tr style='background:{bg};'>{cells}</tr>"
    html = f"""
    <div style='overflow-x:auto;'>
    <table style='width:100%;border-collapse:collapse;font-size:0.95em;'>
        <thead><tr style='background:#4a90e2;color:white;'>{headers}</tr></thead>
        <tbody>{rows}</tbody>
    </table>
    </div>"""
    st.markdown(html, unsafe_allow_html=True)

# Main app logic
def main():
    with st.container():
        st.markdown("""
            <div class='header-section'>
                <h1>Team Builder</h1>
                <p class='subheader'>Craft your ideal team with personalized job role recommendations and salary insights.</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<h2>Describe Your Company's Needs</h2>", unsafe_allow_html=True)
        st.markdown("""
            <div class='highlight'>
                <p>Share your challenges, goals, or expertise needs. Examples include:</p>
                <ul>
                    <li>Project bottlenecks requiring specific skills</li>
                    <li>New initiatives needing specialized talent</li>
                    <li>Roles to strengthen your team</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        company_needs_description = st.text_area("Enter Description or Paste Company URL/About Us Page:", height=150, placeholder="E.g., 'We need a team to develop a new mobile app and improve our cloud infrastructure.'")

    # Session state initialization
    for key in ['main_response', 'job_list', 'relevant_job_list', 'irrelevant_job_list', 'job_list_salary', 'additional_info', 'show_job_list']:
        if key not in st.session_state:
            st.session_state[key] = [] if 'list' in key else "" if key != 'show_job_list' else False

    if st.button("Analyze Needs", key="analyze_button"):
        is_url_provided = company_needs_description.startswith("http")
        if is_url_provided:
            about_us_content = analyze_url_content(company_needs_description)
            if about_us_content:
                company_needs_description = about_us_content
            else:
                st.session_state['job_list'] = []
                st.session_state.show_job_list = False

        if not validate_input(company_needs_description):
            st.error("Please provide specific details related to jobs or business needs.", icon="🚫")
            st.session_state['job_list'] = []
            st.session_state.show_job_list = False
        else:
            with st.spinner("Analyzing your needs..."):
                prompt = f"""
                Analyze the company needs description and return a JSON object with job roles split into two categories.
                Description: {company_needs_description}
                Format:
                {{
                    "relevant_roles": [
                        {{"role": "title", "description": "20-40 word explanation"}}
                    ],
                    "optional_roles": [
                        {{"role": "title", "description": "20-40 word explanation"}}
                    ]
                }}
                Guidelines:
                - Return only JSON, no markdown
                - relevant_roles: roles directly needed based on the description
                - optional_roles: roles that could help but are not explicitly required
                - Descriptions must be concise (20-40 words)
                """
                cache_key = hashlib.md5(prompt.encode()).hexdigest()
                response_text = cache_result(cache_key, generate_content, prompt)
                st.session_state.main_response = response_text

                parsed_json, success = extract_and_parse_json(response_text)
                if success and ("relevant_roles" in parsed_json or "optional_roles" in parsed_json):
                    relevant = parsed_json.get("relevant_roles", [])
                    optional = parsed_json.get("optional_roles", [])
                    all_roles = relevant + optional
                    st.session_state['job_list'] = [j["role"] for j in all_roles]
                    st.session_state.relevant_job_list = [j["role"] for j in relevant]
                    st.session_state.irrelevant_job_list = [j["role"] for j in optional]
                    st.session_state.show_job_list = True
                else:
                    st.error("Could not identify job roles. Please refine your description.", icon="⚠️")
                    st.session_state['job_list'] = []
                    st.session_state.show_job_list = False

    if st.session_state.main_response:
        with st.expander("View Detailed Analysis", expanded=False):
            parsed_json, _ = extract_and_parse_json(st.session_state.main_response)
            if parsed_json and ("relevant_roles" in parsed_json or "optional_roles" in parsed_json):
                relevant = parsed_json.get("relevant_roles", [])
                optional = parsed_json.get("optional_roles", [])
                if relevant:
                    st.markdown("<h3>Relevant Roles</h3>", unsafe_allow_html=True)
                    for job in relevant:
                        st.markdown(f"<div class='card'><strong>{job['role']}</strong><p>{job['description']}</p></div>", unsafe_allow_html=True)
                if optional:
                    st.markdown("<h3>Optional Roles</h3>", unsafe_allow_html=True)
                    for job in optional:
                        st.markdown(f"<div class='card'><strong>{job['role']}</strong><p>{job['description']}</p></div>", unsafe_allow_html=True)
            else:
                st.write(st.session_state.main_response)

        st.markdown("<h3>Add More Details</h3>", unsafe_allow_html=True)
        additional_info = st.text_area("Provide Additional Context (Optional):", height=100, placeholder="E.g., 'We need expertise in AI-driven analytics.'")
        st.session_state.additional_info = additional_info

        if st.button("Submit Additional Info", key="submit_additional"):
            if not validate_input(additional_info):
                st.error("Please provide specific, relevant additional details.", icon="🚫")
            else:
                with st.spinner("Processing additional info..."):
                    full_description = f"{company_needs_description}\n\nAdditional Info: {additional_info}"
                    prompt = f"""
                    Analyze the company needs description and return a JSON object with job roles split into two categories.
                    Description: {full_description}
                    Format:
                    {{
                        "relevant_roles": [
                            {{"role": "title", "description": "20-40 word explanation"}}
                        ],
                        "optional_roles": [
                            {{"role": "title", "description": "20-40 word explanation"}}
                        ]
                    }}
                    Guidelines:
                    - Return only JSON, no markdown
                    - relevant_roles: roles directly needed based on the description
                    - optional_roles: roles that could help but are not explicitly required
                    - Descriptions must be concise (20-40 words)
                    """
                    cache_key = hashlib.md5(prompt.encode()).hexdigest()
                    response_text = cache_result(cache_key, generate_content, prompt)
                    st.session_state.main_response = response_text

                    parsed_json, success = extract_and_parse_json(response_text)
                    if success and ("relevant_roles" in parsed_json or "optional_roles" in parsed_json):
                        relevant = parsed_json.get("relevant_roles", [])
                        optional = parsed_json.get("optional_roles", [])
                        all_roles = relevant + optional
                        st.session_state['job_list'] = [j["role"] for j in all_roles]
                        st.session_state.relevant_job_list = [j["role"] for j in relevant]
                        st.session_state.irrelevant_job_list = [j["role"] for j in optional]
                        st.session_state.show_job_list = True
                    else:
                        st.error("Could not identify job roles. Please refine your description.", icon="⚠️")
                        st.session_state['job_list'] = []
                        st.session_state.show_job_list = False

    if st.session_state.show_job_list and st.session_state['job_list']:
        st.markdown("<h2>Suggested Job Roles</h2>", unsafe_allow_html=True)
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            st.markdown("<h3>Relevant Roles</h3>", unsafe_allow_html=True)
            relevant_roles_str = ', '.join(st.session_state.relevant_job_list)
            relevant_roles_input = st.text_area("Edit Relevant Roles (comma-separated):", value=relevant_roles_str, height=100, placeholder="E.g., Software Engineer, Data Analyst")
            st.session_state.relevant_job_list = [role.strip() for role in relevant_roles_input.split(',') if role.strip()]
        with col2:
            st.markdown("<h3>Irrelevant Roles</h3>", unsafe_allow_html=True)
            irrelevant_roles_str = ', '.join(st.session_state.irrelevant_job_list)
            irrelevant_roles_input = st.text_area("Edit Irrelevant Roles (comma-separated):", value=irrelevant_roles_str, height=100, placeholder="E.g., Graphic Designer, HR Manager")
            st.session_state.irrelevant_job_list = [role.strip() for role in irrelevant_roles_input.split(',') if role.strip()]

        if st.button("Proceed with Roles", key="proceed_button"):
            st.session_state.show_job_list = False

    if st.session_state.relevant_job_list:
        job_list_salary = []
        for job in st.session_state["relevant_job_list"]:
            prompt = f"""
            Provide a JSON object with realistic monthly median salaries in USD for the job role '{job}' in the Philippines and the United States as of April 2025. Ensure salaries are whole numbers below 10,000 USD, with US salaries higher than Philippines. Format:
            {{
                "salary_comparison": {{
                    "philippines": <number>,
                    "united_states": <number>
                }}
            }}
            """
            cache_key = hashlib.md5(prompt.encode()).hexdigest()
            response_text = cache_result(cache_key, generate_content, prompt)
            salary_json, valid = validate_salary_json(extract_and_parse_json(response_text)[0] or {})
            if valid:
                job_salary = {
                    "job_role": job,
                    "currency": "USD",
                    "salary_comparison": salary_json["salary_comparison"]
                }
                job_list_salary.append(job_salary)
            else:
                st.warning(f"Could not fetch valid salary data for {job}. Skipping.", icon="⚠️")

        st.session_state["job_list_salary"] = job_list_salary
        if job_list_salary:
            df = pd.DataFrame(job_list_salary)
            df = pd.concat([df.drop(['salary_comparison'], axis=1), pd.json_normalize(df['salary_comparison'])], axis=1)
            st.markdown("<h2>Salary Comparison</h2>", unsafe_allow_html=True)
            display_df = df.copy()
            for col in ["philippines", "united_states"]:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"${x:,.0f}")
            render_html_table(display_df)

    if st.session_state['job_list_salary']:
        st.markdown("<h2>Team Size Configuration</h2>", unsafe_allow_html=True)
        cols = st.columns(2, gap="medium")
        for i, job in enumerate(st.session_state['job_list_salary']):
            with cols[i % 2]:
                st.markdown(f"<div class='card'><strong>{job['job_role']}</strong></div>", unsafe_allow_html=True)
                job["no_employees"] = st.number_input(f"Number of {job['job_role']}s:", min_value=0, key=f"num_{job['job_role']}", format="%d")

        if st.button("Calculate Total Cost", key="calculate_cost"):
            st.markdown("<h2>Cost Breakdown</h2>", unsafe_allow_html=True)
            for job in st.session_state['job_list_salary']:
                job["ph_cost"] = job["no_employees"] * job["salary_comparison"]["philippines"]
                job["us_cost"] = job["no_employees"] * job["salary_comparison"]["united_states"]
                job["savings"] = job["us_cost"] - job["ph_cost"]
                job["currency_symbol"] = "$"

            df = pd.DataFrame(st.session_state['job_list_salary'])
            df = pd.concat([df.drop(['salary_comparison'], axis=1), pd.json_normalize(df['salary_comparison'])], axis=1)
            display_df = df.copy()
            for col in ["philippines", "united_states", "ph_cost", "us_cost", "savings"]:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"${x:,.0f}")
            render_html_table(display_df)

            buffer = io.BytesIO()
            df.to_csv(buffer, index=False)
            st.download_button(
                label="Download Full Report",
                data=buffer,
                file_name='team_builder_report.csv',
                mime='text/csv',
                key="download_full"
            )

            philippines_total = df["ph_cost"].sum()
            us_total = df["us_cost"].sum()
            savings = df["savings"].sum()

            st.markdown(f"<div class='card'><strong>Philippines Total Cost:</strong> ${philippines_total:,.2f}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='card'><strong>United States Total Cost:</strong> ${us_total:,.2f}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='card'><strong>Total Savings:</strong> ${savings:,.2f}</div>", unsafe_allow_html=True)

            st.markdown("<h2>Refined Job Role Insights</h2>", unsafe_allow_html=True)
            refined_df = df[["job_role", "ph_cost", "us_cost", "savings"]]
            display_refined = refined_df.copy()
            for col in ["ph_cost", "us_cost", "savings"]:
                if col in display_refined.columns:
                    display_refined[col] = display_refined[col].apply(lambda x: f"${x:,.0f}")
            render_html_table(display_refined)

            buffer = io.BytesIO()
            refined_df.to_csv(buffer, index=False)
            st.download_button(
                label="Download Refined Report",
                data=buffer,
                file_name='refined_team_builder_report.csv',
                mime='text/csv',
                key="download_refined"
            )
            st.success("By hiring in the Philippines, you can save significantly on labor costs while maintaining high-quality talent.", icon="✅")

if __name__ == "__main__":
    main()