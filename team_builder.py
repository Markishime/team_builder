import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
import re
import io
import requests
import hashlib
import time

# Configure Gemini API
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
generation_config = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "max_output_tokens": 2048
}
model = genai.GenerativeModel('gemini-1.5-flash', generation_config=generation_config)

# Custom CSS for dark mode UI with fixed scrollbar
st.markdown("""
    <style>
    body {
        background-color: #1e293b;
        color: #e2e8f0;
        overflow-y: auto;
        scrollbar-width: thin;
        scrollbar-color: #60a5fa #334155;
    }
    body::-webkit-scrollbar {
        width: 12px;
    }
    body::-webkit-scrollbar-track {
        background: #334155;
        border-radius: 10px;
    }
    body::-webkit-scrollbar-thumb {
        background: #60a5fa;
        border-radius: 10px;
        border: 3px solid #334155;
    }
    body::-webkit-scrollbar-thumb:hover {
        background: #3b82f6;
    }
    .main {
        background-color: #2d3748;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        max-width: 1200px;
        margin: 20px auto;
    }
    .stButton>button {
        background: linear-gradient(90deg, #60a5fa, #3b82f6);
        color: #ffffff;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        border: none;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #3b82f6, #60a5fa);
        transform: translateY(-2px);
        box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    }
    h1 {
        color: #f1f5f9;
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 2.8em;
        text-align: center;
        margin-bottom: 0.5em;
    }
    h2, h3 {
        color: #d1d5db;
        font-family: 'Helvetica Neue', sans-serif;
        margin-top: 1.5em;
    }
    .stTextArea>label, .stNumberInput>label {
        font-weight: 600;
        color: #d1d5db;
        font-size: 1.1em;
    }
    .stTextArea textarea {
        background-color: #374151;
        color: #e2e8f0;
        border-radius: 10px;
        border: 2px solid #4b5563;
        transition: border-color 0.3s ease;
    }
    .stTextArea textarea:focus {
        border-color: #60a5fa;
    }
    .stExpander {
        background-color: #374151;
        border-radius: 12px;
        border: 1px solid #4b5563;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        padding: 15px;
    }
    .stDataFrame {
        background-color: #374151;
        border-radius: 12px;
        padding: 15px;
        border: 1px solid #4b5563;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    .card {
        background-color: #4b5563;
        color: #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid #6b7280;
        transition: all 0.3s ease;
    }
    .card:hover {
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        transform: translateY(-3px);
    }
    .header-section {
        text-align: center;
        margin-bottom: 2em;
    }
    .subheader {
        color: #94a3b8;
        font-size: 1.2em;
        line-height: 1.6;
    }
    .highlight {
        background-color: #3b82f6;
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .highlight p, .highlight ul, .highlight li {
        color: #e2e8f0;
    }
    .stSpinner > div > div {
        border-color: #60a5fa transparent #60a5fa transparent !important;
    }
    .stAlert {
        background-color: #4b5563;
        color: #e2e8f0;
    }
    .stAlert div {
        color: #e2e8f0 !important;
    }
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
                Analyze the company needs description and return a JSON object with relevant job roles and brief explanations.
                Description: {company_needs_description}
                Format:
                {{
                    "job_roles": [
                        {{"role": "title", "description": "20-40 word explanation"}},
                        ...
                    ]
                }}
                Guidelines:
                - Return only JSON
                - Ensure roles are specific and relevant
                - Descriptions must be concise
                """
                cache_key = hashlib.md5(prompt.encode()).hexdigest()
                response_text = cache_result(cache_key, lambda p: model.generate_content(p).text, prompt)
                st.session_state.main_response = response_text

                parsed_json, success = extract_and_parse_json(response_text)
                if success and "job_roles" in parsed_json:
                    job_list = parsed_json["job_roles"]
                    st.session_state['job_list'] = [j["role"] for j in job_list]
                    st.session_state.relevant_job_list = [j["role"] for j in job_list if j["role"].lower() not in company_needs_description.lower()]
                    st.session_state.irrelevant_job_list = [j["role"] for j in job_list if j["role"].lower() in company_needs_description.lower()]
                    st.session_state.show_job_list = True
                else:
                    st.error("Could not identify job roles. Please refine your description.", icon="⚠️")
                    st.session_state['job_list'] = []
                    st.session_state.show_job_list = False

    if st.session_state.main_response:
        with st.expander("View Detailed Analysis", expanded=False):
            parsed_json, _ = extract_and_parse_json(st.session_state.main_response)
            if parsed_json and "job_roles" in parsed_json:
                for job in parsed_json["job_roles"]:
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
                    Analyze the company needs description and return a JSON object with 3-5 relevant job roles and brief explanations.
                    Description: {full_description}
                    Format:
                    {{
                        "job_roles": [
                            {{"role": "title", "description": "20-40 word explanation"}},
                            ...
                        ]
                    }}
                    Guidelines:
                    - Return only JSON
                    - Ensure roles are specific and relevant
                    - Descriptions must be concise
                    """
                    cache_key = hashlib.md5(prompt.encode()).hexdigest()
                    response_text = cache_result(cache_key, lambda p: model.generate_content(p).text, prompt)
                    st.session_state.main_response = response_text

                    parsed_json, success = extract_and_parse_json(response_text)
                    if success and "job_roles" in parsed_json:
                        job_list = parsed_json["job_roles"]
                        st.session_state['job_list'] = [j["role"] for j in job_list]
                        st.session_state.relevant_job_list = [j["role"] for j in job_list if j["role"].lower() in full_description.lower()]
                        st.session_state.irrelevant_job_list = [j["role"] for j in job_list if j["role"].lower() not in full_description.lower()]
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
            response_text = cache_result(cache_key, lambda p: model.generate_content(p).text, prompt)
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
            st.dataframe(df.style.format({"philippines": "${:,.0f}", "united_states": "${:,.0f}"}), use_container_width=True)

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
            st.dataframe(df.style.format({
                "philippines": "${:,.0f}",
                "united_states": "${:,.0f}",
                "ph_cost": "${:,.0f}",
                "us_cost": "${:,.0f}",
                "savings": "${:,.0f}"
            }), use_container_width=True)

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
            st.dataframe(refined_df.style.format({
                "ph_cost": "${:,.0f}",
                "us_cost": "${:,.0f}",
                "savings": "${:,.0f}"
            }), use_container_width=True)

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