from pathlib import Path
import pandas as pd


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "rag"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Input files
# ============================================================

GEN3_FINAL = (
    PROJECT_ROOT
    / "outputs/reinvent/mdm2_generation3_100step/sampling/"
    / "generation3_final_3_candidates.csv"
)

GEN3_VS_NUTLIN = (
    PROJECT_ROOT
    / "outputs/reinvent/mdm2_generation3_100step/sampling/"
    / "generation3_final_candidates_vs_nutlin3a.csv"
)

STRUCTURE_SUMMARY = (
    PROJECT_ROOT
    / "docking/generation3/results/"
    / "generation3_final_structure_based_summary.csv"
)

MMGBSA_SUMMARY = (
    PROJECT_ROOT
    / "outputs/mmgbsa/generation3_final_candidates/results/"
    / "gen3_final_integrated_comparison.csv"
)

ADMET_SUMMARY = (
    PROJECT_ROOT
    / "selected_candidates_gen1_gen2_gen3_admet.csv"
)


# ============================================================
# 3. Load real project tables
# ============================================================

gen3 = pd.read_csv(GEN3_FINAL)
gen3_vs_nutlin = pd.read_csv(GEN3_VS_NUTLIN)
structure = pd.read_csv(STRUCTURE_SUMMARY)
mmgbsa = pd.read_csv(MMGBSA_SUMMARY)
admet = pd.read_csv(ADMET_SUMMARY)


print("Loaded:")
print("Gen3:", gen3.shape)
print("Gen3 vs Nutlin:", gen3_vs_nutlin.shape)
print("Structure:", structure.shape)
print("MMGBSA:", mmgbsa.shape)
print("ADMET:", admet.shape)


# ============================================================
# 4. Start from the 3 final Gen3 candidates
# ============================================================

candidate_summary = gen3[
    [
        "candidate_name",
        "canonical_smiles",
        "predicted_pIC50",
        "pIC50_AD_max_similarity",
        "pIC50_AD_soft_score",
        "QED",
        "similarity_to_Nutlin3a",
        "Solubility_AqSolDB",
        "Bioavailability_Ma",
        "hERG",
        "DILI",
        "Pgp_Broccatelli",
        "CYP3A4_Veith",
        "generation3_reward",
        "generation3_rank",
        "MW",
        "cLogP",
        "TPSA",
        "HBD",
        "HBA",
        "RotB",
        "Lipinski_violations",
        "ADMET_composite",
        "worst_liability",
        "cluster_id",
        "cluster_size",
        "final_priority_rank"
    ]
].copy()


# ============================================================
# 5. Add Nutlin comparison properties
# ============================================================

nutlin_compare = gen3_vs_nutlin[
    [
        "candidate_name",
        "MW",
        "cLogP",
        "TPSA",
        "HBD",
        "HBA",
        "RotB",
        "QED",
        "similarity_to_Nutlin3a"
    ]
].copy()

# Most of these already exist in gen3.
# Keep this table mainly for validation/reference.

print("\nNutlin comparison candidates:")
print(gen3_vs_nutlin["candidate_name"].tolist())


# ============================================================
# 6. Prepare structure-based results
# ============================================================

structure_clean = structure[
    [
        "candidate_name",
        "vina_score_kcal_mol",
        "hydrogen_bond",
        "hydrophobic"
    ]
].copy()


candidate_summary = candidate_summary.merge(
    structure_clean,
    on="candidate_name",
    how="left",
    validate="one_to_one"
)


# ============================================================
# 7. Prepare MM/GBSA results
# ============================================================

mmgbsa_clean = mmgbsa[
    [
        "system",
        "Docking_kcal_mol",
        "MMGBSA_kcal_mol",
        "Result_status",
        "Backbone_RMSD_mean_A",
        "Ligand_RMSD_mean_A",
        "Ligand_RMSD_final_A",
        "MMGBSA_vs_Nutlin",
        "Contacts_mean_final500ps",
        "Contacts_median_final500ps"
    ]
].copy()

# Rename MMGBSA key so it matches candidate_name
mmgbsa_clean = mmgbsa_clean.rename(
    columns={
        "system": "candidate_name"
    }
)

candidate_summary = candidate_summary.merge(
    mmgbsa_clean,
    on="candidate_name",
    how="left",
    validate="one_to_one"
)


# ============================================================
# 8. Prepare Gen3 ADMET table
# ============================================================

gen3_admet = admet[
    admet["Generation"]
    .astype(str)
    .str.lower()
    .eq("gen3")
].copy()


# Convert:
# Candidate_1 -> Gen3_Candidate_1

gen3_admet["candidate_name"] = (
    "Gen3_"
    + gen3_admet["Candidate"].astype(str)
)


gen3_admet = gen3_admet[
    [
        "candidate_name",
        "Bioavailability_Ma",
        "hERG",
        "DILI",
        "Pgp_Broccatelli",
        "CYP3A4_Veith",
        "QED"
    ]
].copy()


# Rename duplicate ADMET columns so we can compare them
gen3_admet = gen3_admet.rename(
    columns={
        "Bioavailability_Ma": "admet_table_Bioavailability_Ma",
        "hERG": "admet_table_hERG",
        "DILI": "admet_table_DILI",
        "Pgp_Broccatelli": "admet_table_Pgp_Broccatelli",
        "CYP3A4_Veith": "admet_table_CYP3A4_Veith",
        "QED": "admet_table_QED"
    }
)


candidate_summary = candidate_summary.merge(
    gen3_admet,
    on="candidate_name",
    how="left",
    validate="one_to_one"
)


# ============================================================
# 9. Add generation / target metadata
# ============================================================

candidate_summary["generation"] = "Gen3"
candidate_summary["target"] = "MDM2"
candidate_summary["reference_compound"] = "Nutlin-3a"


# ============================================================
# 10. Reorder important columns
# ============================================================

important_columns = [
    "candidate_name",
    "generation",
    "target",
    "canonical_smiles",

    "predicted_pIC50",
    "pIC50_AD_max_similarity",
    "pIC50_AD_soft_score",

    "QED",
    "similarity_to_Nutlin3a",

    "MW",
    "cLogP",
    "TPSA",
    "HBD",
    "HBA",
    "RotB",
    "Lipinski_violations",

    "Solubility_AqSolDB",
    "Bioavailability_Ma",
    "hERG",
    "DILI",
    "Pgp_Broccatelli",
    "CYP3A4_Veith",
    "ADMET_composite",
    "worst_liability",

    "vina_score_kcal_mol",
    "hydrogen_bond",
    "hydrophobic",

    "MMGBSA_kcal_mol",
    "Result_status",
    "Backbone_RMSD_mean_A",
    "Ligand_RMSD_mean_A",
    "Ligand_RMSD_final_A",
    "MMGBSA_vs_Nutlin",
    "Contacts_mean_final500ps",
    "Contacts_median_final500ps",

    "generation3_reward",
    "generation3_rank",
    "final_priority_rank",

    "cluster_id",
    "cluster_size",
    "reference_compound"
]


candidate_summary = candidate_summary[
    important_columns
]


# ============================================================
# 11. Validate result
# ============================================================

print("\n" + "=" * 80)
print("FINAL MDM2 CANDIDATE SUMMARY")
print("=" * 80)

print("\nShape:")
print(candidate_summary.shape)

print("\nCandidates:")
print(candidate_summary["candidate_name"].tolist())

print("\nMissing values by column:")
print(
    candidate_summary
    .isna()
    .sum()
    .sort_values(ascending=False)
)

print("\nDuplicate candidates:")
print(
    candidate_summary[
        "candidate_name"
    ].duplicated().sum()
)

print("\nPreview:")
print(
    candidate_summary[
        [
            "candidate_name",
            "predicted_pIC50",
            "QED",
            "vina_score_kcal_mol",
            "MMGBSA_kcal_mol",
            "Bioavailability_Ma",
            "hERG",
            "DILI"
        ]
    ].to_string(index=False)
)


# ============================================================
# 12. Save integrated RAG-ready table
# ============================================================

OUTPUT_FILE = (
    OUTPUT_DIR
    / "mdm2_gen3_candidate_summary.csv"
)

candidate_summary.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    "\nSaved candidate summary to:",
    OUTPUT_FILE
)