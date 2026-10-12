# MDM2–p53 RAG Drug Discovery Chatbot

**Live App:**  
https://mdm2-rag-chatbot-3yl9ntlurwsaseefvjms2k.streamlit.app/

A scientific Retrieval-Augmented Generation (RAG) chatbot for exploring computational evidence from an AI-driven MDM2–p53 inhibitor discovery project.

## Overview

This application integrates candidate-level computational results generated throughout the MDM2–p53 drug discovery workflow, including:

- Predicted pIC50
- ADMET properties
- QED and physicochemical properties
- Molecular docking scores
- Molecular dynamics stability metrics
- MM/GBSA binding energies
- Final candidate priority rankings

The chatbot uses semantic retrieval to identify relevant candidate evidence and an OpenAI model to generate grounded, context-aware responses based on the retrieved project data.

## Architecture

MDM2 project data  
→ Candidate-level evidence summary  
→ Sentence-transformer embeddings  
→ Semantic retrieval  
→ Grounded OpenAI response  
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
```

## Example Questions

- Which Gen3 candidate has the highest predicted potency?
- Which candidate has the best docking score?
- Which candidate has the most favorable MM/GBSA binding energy?
- Which candidate appears most stable during molecular dynamics simulation
