"""The script produces accuracy metrics for classifAI results. It is assumed that classifAI results and original sample are stored in a directory data/evaluation_data."""

import json

import pandas as pd

from classifai.evaluation import LabelAccuracy
from classifai.utils import flatten_json_to_wide_df

metadata_all = {
    "evaluation_type": "classifai_accuracy_all",
    "coverage": "all_samples",
    "classification_type": "sic",
    "classifai_method": "baseline_embedding_search_full_knowledge_base",
    "knowledge_base": "sic_knowledge_base",
}

# Load output from classifAI API and validated sample
with open("data/evaluation_data/response_1734536101812.json", "rb") as f:
    classifai_results_json = json.load(f)

classifai_results = flatten_json_to_wide_df(classifai_results_json)

labelled_set = pd.read_csv(
    "data/evaluation_data/coding_df_with_validated.csv", index_col=0
)

merged_results = pd.merge(
    labelled_set,
    classifai_results,
    left_on="id",
    right_on="input_id",
    how="inner",
)
merged_results = merged_results.drop_duplicates(subset="id")
merged_results["score_1"] = 1 - merged_results["distance_1"]
merged_results["score_2"] = 1 - merged_results["distance_2"]
merged_results["label_1"] = merged_results["label_1"].astype(str)
merged_results["label_2"] = merged_results["label_2"].astype(str)

print(f"Total number of samples: {merged_results.shape[0]}")

coded_df = merged_results[merged_results["source_type"] != "Uncoded"]
print(f"Number of coded samples: {coded_df.shape[0]}")

# Get 2-digit version
coded_df["sic_validated_1_2d"] = coded_df["sic_validated_1"].str[:2]
coded_df["sic_validated_2_2d"] = coded_df["sic_validated_2"].str[:2]
coded_df["sic_validated_3_2d"] = coded_df["sic_validated_3"].str[:2]
coded_df["label_1_2d"] = coded_df["label_1"].str[:2]
coded_df["sic_5d_2d"] = coded_df["sic_5d"].str[:2]


# Calculate overall classifAI accuracy at 5-digit level
analyzer = LabelAccuracy(
    coded_df,
    id_col="id",
    desc_col="desc_processed",
    model_label_cols=["label_1"],
    model_score_cols=["score_1"],
    clerical_label_cols=[
        "sic_validated_1",
        "sic_validated_2",
        "sic_validated_3",
    ],
)

print(f"Overall accuracy: {analyzer.get_accuracy():.1f}%")

# Plot accuracy, coverage and similarity
analyzer.plot_threshold_curves()

stats = analyzer.get_summary_stats()
LabelAccuracy.save_output(metadata_all, stats)

# Calculate overall classifAI accuracy at 2-digit level
analyzer_2d = LabelAccuracy(
    coded_df,
    id_col="id",
    desc_col="desc_processed",
    model_label_cols=["label_1_2d"],
    model_score_cols=["score_1"],
    clerical_label_cols=[
        "sic_validated_1_2d",
        "sic_validated_2_2d",
        "sic_validated_3_2d",
    ],
)

print(f"Overall accuracy at 2-digit level: {analyzer_2d.get_accuracy():.1f}%")

stats = analyzer_2d.get_summary_stats()
# LabelAccuracy.save_output(metadata_all, stats)


# Calculate classifAI accuracy on autocoded samples
metadata_autocoded_classifai = {
    "evaluation_type": "classifai_accuracy_autocoded",
    "coverage": "autocoded_samples",
    "classification_type": "sic",
    "classifai_method": "baseline_embedding_search_full_knowledge_base",
    "knowledge_base": "sic_knowledge_base",
}

autocoded = coded_df[coded_df["source_type"] == "G-Code"]
print(f"Number of autocoded samples: {autocoded.shape[0]}")


analyzer_autocoded = LabelAccuracy(
    autocoded,
    id_col="id",
    desc_col="desc_processed",
    model_label_cols=["label_1"],
    model_score_cols=["score_1"],
    clerical_label_cols=[
        "sic_validated_1",
        "sic_validated_2",
        "sic_validated_3",
    ],
)

stats = analyzer_autocoded.get_summary_stats()
LabelAccuracy.save_output(metadata_autocoded_classifai, stats)

analyzer_autocoded_2d = LabelAccuracy(
    autocoded,
    id_col="id",
    desc_col="desc_processed",
    model_label_cols=["label_1_2d"],
    model_score_cols=["score_1"],
    clerical_label_cols=[
        "sic_validated_1_2d",
        "sic_validated_2_2d",
        "sic_validated_3_2d",
    ],
)

print(
    f"Overall accuracy autocoded classifAI: {analyzer_autocoded_2d.get_accuracy():.1f}%"
)


# Get G-Code accuracy on autocoded samples, ignore score_1 param, just needed for instantiation
metadata_autocoded_gcoded = {
    "evaluation_type": "g_code_accuracy_autocoded",
    "coverage": "autocoded_samples",
    "classification_type": "sic",
    "classifai_method": "baseline_embedding_search_full_knowledge_base",
    "knowledge_base": "sic_knowledge_base",
}

analyzer_autocoded_gcode = LabelAccuracy(
    autocoded,
    id_col="id",
    desc_col="desc_processed",
    model_label_cols=["sic_5d"],
    model_score_cols=["score_1"],
    clerical_label_cols=[
        "sic_validated_1",
        "sic_validated_2",
        "sic_validated_3",
    ],
)

stats = analyzer_autocoded_gcode.get_summary_stats()
LabelAccuracy.save_output(metadata_autocoded_gcoded, stats)

analyzer_autocoded_gcode_2d = LabelAccuracy(
    autocoded,
    id_col="id",
    desc_col="desc_processed",
    model_label_cols=["sic_5d_2d"],
    model_score_cols=["score_1"],
    clerical_label_cols=[
        "sic_validated_1_2d",
        "sic_validated_2_2d",
        "sic_validated_3_2d",
    ],
)

print(
    f"Overall accuracy autocoded G-Code: {analyzer_autocoded_gcode_2d.get_accuracy():.1f}%"
)
