# MDM2–p53 RAG Drug Discovery Chatbot

A scientific RAG chatbot for exploring computational evidence from an MDM2–p53 inhibitor discovery project.

## Overview

This application integrates candidate-level computational results from an AI-driven drug discovery workflow, including:

- Predicted pIC50
- ADMET properties
- QED and physicochemical properties
- Molecular docking
- Molecular dynamics stability metrics
- MM/GBSA binding energy
- Final candidate priority ranking

The chatbot uses semantic retrieval over candidate evidence and generates grounded answers with an OpenAI model.

## Architecture

MDM2 project data  
→ candidate summary  
→ sentence-transformer embeddings  
→ semantic retrieval  
→ OpenAI grounded response  
→ Streamlit chatbot

## Project Structure

```text
mdm2-rag-chatbot/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── outputs/
│   └── rag/
│       └── mdm2_gen3_candidate_summary.csv
└── src/
    ├── __init__.py
    └── rag/
        ├── __init__.py
        ├── build_candidate_summary.py
        └── rag_pipeline.py

Example Questions
- Which Gen3 candidate has the highest predicted potency?
- Which candidate has the best docking score?
- Which candidate has the most favorable MM/GBSA energy?
- Which candidate appears most stable during MD?
- Why is Gen3_Candidate_1 ranked first?
- What are the main trade-offs among the three candidates?