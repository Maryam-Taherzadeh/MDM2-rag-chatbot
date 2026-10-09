from pathlib import Path
import sys

import streamlit as st


# ============================================================
# 1. Project setup
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 2. Import RAG pipeline
# ============================================================

from src.rag.rag_pipeline import rag_answer, candidate_df


# ============================================================
# 3. Page configuration
# ============================================================

st.set_page_config(
    page_title="MDM2-p53 Drug Discovery Assistant",
    page_icon="🧬",
    layout="wide"
)


# ============================================================
# 4. Header
# ============================================================

st.title("🧬 MDM2–p53 Drug Discovery Assistant")

st.write(
    """
    Ask scientific questions about the final Gen3 MDM2 inhibitor
    candidates using predicted potency, ADMET, molecular docking,
    molecular dynamics, and MM/GBSA evidence.
    """
)

st.info(
    "This assistant answers using computational evidence from the "
    "MDM2–p53 project. Results are computational predictions and "
    "simulations, not experimental measurements."
)


# ============================================================
# 5. Candidate overview
# ============================================================

with st.expander("📊 View Final Gen3 Candidate Summary"):

    sorted_df = candidate_df.sort_values(
        "final_priority_rank"
    )

    display_df = sorted_df[
        [
            "candidate_name",
            "final_priority_rank",
            "predicted_pIC50",
            "QED",
            "vina_score_kcal_mol",
            "MMGBSA_kcal_mol",
            "Ligand_RMSD_mean_A",
            "Ligand_RMSD_final_A",
            "ADMET_composite"
        ]
    ].copy()

    display_df.columns = [
        "Candidate",
        "Priority Rank",
        "Predicted pIC50",
        "QED",
        "Vina",
        "MM/GBSA",
        "Ligand RMSD Mean",
        "Ligand RMSD Final",
        "ADMET Composite"
    ]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 6. Suggested questions
# ============================================================

st.markdown("### Example questions")

st.markdown(
    """
    - Which Gen3 candidate has the highest predicted potency?
    - Which candidate has the best docking score?
    - Which candidate has the most favorable MM/GBSA energy?
    - Which candidate appears most stable during MD?
    - Compare Gen3_Candidate_1 and Gen3_Candidate_3.
    - Why is Gen3_Candidate_1 ranked first?
    - Which candidate has the strongest ADMET profile?
    - What are the main trade-offs among the three candidates?
    """
)


# ============================================================
# 7. Chat history
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# 8. Chat input
# ============================================================

question = st.chat_input(
    "Ask a question about the MDM2 candidates..."
)


# ============================================================
# 9. Process question
# ============================================================

if question:

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)


    # Assistant response
    with st.chat_message("assistant"):

        with st.spinner(
            "Analyzing computational evidence..."
        ):

            try:

                answer, sources = rag_answer(
                    question,
                    top_k=3
                )

                st.markdown(answer)


                # Save assistant answer
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )


                # ====================================================
                # Retrieved evidence
                # ====================================================

                with st.expander(
                    "🔎 View retrieved candidate evidence"
                ):

                    evidence_columns = [
                        "candidate_name",
                        "final_priority_rank",
                        "predicted_pIC50",
                        "QED",
                        "vina_score_kcal_mol",
                        "MMGBSA_kcal_mol",
                        "Ligand_RMSD_mean_A",
                        "Ligand_RMSD_final_A",
                        "ADMET_composite",
                        "similarity_score"
                    ]

                    evidence_df = sources[
                        evidence_columns
                    ].copy()

                    evidence_df.columns = [
                        "Candidate",
                        "Priority Rank",
                        "Predicted pIC50",
                        "QED",
                        "Vina",
                        "MM/GBSA",
                        "Ligand RMSD Mean",
                        "Ligand RMSD Final",
                        "ADMET Composite",
                        "Retrieval Similarity"
                    ]

                    st.dataframe(
                        evidence_df,
                        use_container_width=True,
                        hide_index=True
                    )


            except Exception as error:

                st.error(
                    f"Error: {error}"
                )


# ============================================================
# 10. Footer
# ============================================================

st.divider()

st.caption(
    "MDM2–p53 AI Drug Discovery Project | "
    "RAG-based Scientific Assistant"
)