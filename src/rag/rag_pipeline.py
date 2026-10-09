from pathlib import Path

import os
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CANDIDATE_FILE = (
    PROJECT_ROOT
    / "outputs/rag/mdm2_gen3_candidate_summary.csv"
)


# ============================================================
# 2. Load candidate summary
# ============================================================

candidate_df = pd.read_csv(
    CANDIDATE_FILE
)

print(
    "Loaded candidate summary:",
    candidate_df.shape
)

print(
    candidate_df["candidate_name"].tolist()
)


# ============================================================
# 3. Prepare scientific RAG documents
# ============================================================

def safe_text(value):
    if pd.isna(value):
        return "Not available"

    return str(value)


candidate_df["document"] = candidate_df.apply(
    lambda row: f"""
Candidate: {safe_text(row["candidate_name"])}
Generation: {safe_text(row["generation"])}
Target: {safe_text(row["target"])}

Predicted pIC50: {safe_text(row["predicted_pIC50"])}
Applicability-domain max similarity: {safe_text(row["pIC50_AD_max_similarity"])}
Applicability-domain soft score: {safe_text(row["pIC50_AD_soft_score"])}

QED: {safe_text(row["QED"])}
Similarity to Nutlin-3a: {safe_text(row["similarity_to_Nutlin3a"])}

Molecular weight: {safe_text(row["MW"])}
cLogP: {safe_text(row["cLogP"])}
TPSA: {safe_text(row["TPSA"])}
HBD: {safe_text(row["HBD"])}
HBA: {safe_text(row["HBA"])}
Rotatable bonds: {safe_text(row["RotB"])}
Lipinski violations: {safe_text(row["Lipinski_violations"])}

Predicted solubility: {safe_text(row["Solubility_AqSolDB"])}
Bioavailability: {safe_text(row["Bioavailability_Ma"])}
hERG: {safe_text(row["hERG"])}
DILI: {safe_text(row["DILI"])}
P-gp: {safe_text(row["Pgp_Broccatelli"])}
CYP3A4: {safe_text(row["CYP3A4_Veith"])}
ADMET composite: {safe_text(row["ADMET_composite"])}
Worst liability: {safe_text(row["worst_liability"])}

Vina docking score: {safe_text(row["vina_score_kcal_mol"])} kcal/mol
Hydrogen-bond interactions: {safe_text(row["hydrogen_bond"])}
Hydrophobic interactions: {safe_text(row["hydrophobic"])}

MMGBSA binding energy: {safe_text(row["MMGBSA_kcal_mol"])} kcal/mol
MMGBSA result status: {safe_text(row["Result_status"])}
Backbone RMSD mean: {safe_text(row["Backbone_RMSD_mean_A"])} A
Ligand RMSD mean: {safe_text(row["Ligand_RMSD_mean_A"])} A
Ligand RMSD final: {safe_text(row["Ligand_RMSD_final_A"])} A
MMGBSA relative to Nutlin-3a: {safe_text(row["MMGBSA_vs_Nutlin"])}
Mean final-500-ps contacts: {safe_text(row["Contacts_mean_final500ps"])}
Median final-500-ps contacts: {safe_text(row["Contacts_median_final500ps"])}

Generation 3 reward: {safe_text(row["generation3_reward"])}
Generation 3 rank: {safe_text(row["generation3_rank"])}
Final priority rank: {safe_text(row["final_priority_rank"])}

Reference compound: {safe_text(row["reference_compound"])}
""".strip(),
    axis=1
)


# ============================================================
# 4. Load embedding model
# ============================================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ============================================================
# 5. Create embeddings
# ============================================================

documents = candidate_df[
    "document"
].tolist()

document_embeddings = embedding_model.encode(
    documents,
    normalize_embeddings=True
)

print(
    "Embedding shape:",
    document_embeddings.shape
)


# ============================================================
# 6. Semantic search
# ============================================================

def semantic_search(
    question,
    top_k=3
):

    top_k = min(
        top_k,
        len(candidate_df)
    )

    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )

    scores = cosine_similarity(
        query_embedding,
        document_embeddings
    )[0]

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = candidate_df.iloc[
        top_indices
    ].copy()

    results["similarity_score"] = scores[
        top_indices
    ]

    return results


# ============================================================
# 7. Build retrieved context
# ============================================================

def build_context(
    question,
    top_k=3
):

    results = semantic_search(
        question,
        top_k=top_k
    )

    context = "\n\n---\n\n".join(
        results["document"].tolist()
    )

    return context, results


# ============================================================
# 8. OpenAI client
# ============================================================

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(
        "OPENAI_API_KEY is not available in the environment."
    )

client = OpenAI()


# ============================================================
# 9. Grounded RAG answer
# ============================================================

def rag_answer(
    question,
    top_k=3
):

    context, sources = build_context(
        question,
        top_k=top_k
    )

    prompt = f"""
You are a scientific assistant for an MDM2-p53
small-molecule drug-discovery project.

Answer the user's question using only the retrieved
candidate evidence provided below.

Rules:
1. Do not invent measurements.
2. Do not invent interactions.
3. Distinguish predicted properties from
   physics-based results.
4. Mention candidate names explicitly.
5. When comparing candidates, explain which metrics
   support the comparison.
6. More negative Vina or MMGBSA values indicate
   more favorable computed binding scores.
7. If the available evidence is insufficient,
   say so clearly.

Retrieved evidence:

{context}

Question:
{question}

Answer:
"""

    response = client.responses.create(
        model="gpt-5.5",
        input=prompt
    )

    return response.output_text, sources


# ============================================================
# 10. Local test
# ============================================================

if __name__ == "__main__":

    question = (
        "Which Gen3 candidate has the strongest "
        "overall computational evidence?"
    )

    answer, sources = rag_answer(
        question,
        top_k=3
    )

    print("\nQUESTION:")
    print(question)

    print("\nANSWER:")
    print(answer)

    print("\nRETRIEVED SOURCES:")

    print(
        sources[
            [
                "candidate_name",
                "predicted_pIC50",
                "QED",
                "vina_score_kcal_mol",
                "MMGBSA_kcal_mol",
                "similarity_score"
            ]
        ].to_string(index=False)
    )