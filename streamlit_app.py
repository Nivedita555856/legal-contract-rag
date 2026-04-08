import requests
import streamlit as st

st.set_page_config(page_title='Legal Contract Intelligence', layout='wide')
st.title('⚖️ Legal Contract Intelligence System (Corrective RAG)')

api_url = st.sidebar.text_input('FastAPI URL', value='http://localhost:8000')

st.header('1) Upload Contract')
uploaded = st.file_uploader('Upload PDF or TXT contract', type=['pdf', 'txt'])

if uploaded and st.button('Ingest Document'):
    files = {'file': (uploaded.name, uploaded.getvalue(), uploaded.type or 'application/octet-stream')}
    with st.spinner('Ingesting...'):
        resp = requests.post(f'{api_url}/api/ingest', files=files, timeout=120)
    if resp.ok:
        data = resp.json()
        st.success(f"Document ingested: {data['document_id']} ({data['clauses_extracted']} clauses)")
        st.session_state['document_id'] = data['document_id']
    else:
        st.error(resp.text)

st.header('2) Ask Questions with Corrective Loop')
doc_id = st.text_input('Document ID (optional)', value=st.session_state.get('document_id', ''))
question = st.text_area('Question', placeholder='What termination risks exist in this contract?')

if st.button('Get Answer'):
    payload = {'question': question, 'document_id': doc_id or None}
    with st.spinner('Retrieving and validating...'):
        resp = requests.post(f'{api_url}/api/ask', json=payload, timeout=120)
    if resp.ok:
        data = resp.json()
        st.subheader('Answer')
        st.write(data['answer'])
        c1, c2 = st.columns(2)
        c1.metric('Grounded', 'Yes' if data['grounded'] else 'No')
        c2.metric('Confidence', f"{data['confidence']:.2f}")

        st.subheader('Corrective Steps')
        st.write(' → '.join(data['corrective_steps']))

        st.subheader('Evidence Clauses')
        for ev in data['evidence']:
            with st.expander(f"{ev['clause_id']} | score={ev['score']:.3f}"):
                st.caption(f"Section: {ev.get('section')}")
                st.write(ev['text'])
    else:
        st.error(resp.text)
