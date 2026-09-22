from pathlib import Path
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import os
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =========================
# 1. Load model
# =========================

model_name = "delpot/steganograph-ia-detector"

print("Loading SteganographIA model...")

processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)
model.eval()

print("Model loaded.")
print("Class mapping:", model.config.id2label)


# =========================
# 2. Dataset location
# =========================

PROJECT_DIR = Path(__file__).resolve().parent
dataset_path = PROJECT_DIR / "data" / "MyDataset"

if not dataset_path.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {dataset_path}"
    )

folders = {
    "Fake": "ai_generated",
    "Real": "real"
}

# =========================
# 3. Process images
# =========================

results = []

for folder_name, ground_truth in folders.items():

    folder_path = dataset_path / folder_name

    if not folder_path.exists():
        raise FileNotFoundError(
            f"Required folder not found: {folder_path}"
        )

    print(f"\nProcessing {folder_name} images...")

    for image_path in sorted(folder_path.iterdir()):

        # Skip directories and unsupported files
        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in (
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".avif",
        ):
            continue

        filename = image_path.name

        try:
            image = Image.open(image_path).convert("RGB")

            inputs = processor(image, return_tensors="pt")

            with torch.no_grad():
                logits = model(**inputs).logits

            probabilities = torch.softmax(logits, dim=-1)

            predicted_id = logits.argmax(-1).item()
            predicted_label = model.config.id2label[predicted_id]

            confidence = probabilities[0][predicted_id].item()

            correct = int(predicted_label == ground_truth)

            results.append({
                "Image": filename,
                "Ground_Truth": ground_truth,
                "Real_Probability": probabilities[0][0].item(),
                "AI_Probability": probabilities[0][1].item(),
                "Prediction": predicted_label,
                "Confidence": confidence,
                "Correct": correct
            })

            print(
                f"{filename} -> "
                f"Ground Truth: {ground_truth} | "
                f"Prediction: {predicted_label} | "
                f"Confidence: {confidence:.4f}"
            )

        except Exception as e:
            print(f"ERROR processing {filename}: {e}")


# =========================
# 4. Create results table
# =========================

df = pd.DataFrame(results)
print("\nDataset verification")
print("--------------------")
print(f"Images processed: {len(df)}")
print("\nGround-truth distribution:")
print(df["Ground_Truth"].value_counts())

if len(df) != 60:
    print("\nWARNING: Expected 60 images.")

if (
    df["Ground_Truth"].value_counts().get("ai_generated", 0) != 30
    or
    df["Ground_Truth"].value_counts().get("real", 0) != 30
):
    print("WARNING: Expected 30 AI and 30 real images.")

results_dir = PROJECT_DIR / "results"
results_dir.mkdir(exist_ok=True)

output_file = results_dir / "steganographia_results.csv"

df.to_csv(output_file, index=False)

print("\nResults saved to:")
print(output_file)


# =========================
# 5. Calculate metrics
# =========================

y_true = df["Ground_Truth"]
y_pred = df["Prediction"]

accuracy = accuracy_score(y_true, y_pred)

balanced_accuracy = balanced_accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    pos_label="ai_generated"
)

recall = recall_score(
    y_true,
    y_pred,
    pos_label="ai_generated"
)

f1 = f1_score(
    y_true,
    y_pred,
    pos_label="ai_generated"
)


# =========================
# 6. Confusion matrix
# =========================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=["ai_generated", "real"]
)

tp = cm[0, 0]
fn = cm[0, 1]
fp = cm[1, 0]
tn = cm[1, 1]

specificity = tn / (tn + fp)


# =========================
# 7. Print results
# =========================

print("\n")
print("=" * 50)
print("FINAL RESULTS")
print("=" * 50)

print(f"Images tested       : {len(df)}")
print(f"Correct predictions : {df['Correct'].sum()}")
print(f"Incorrect predictions: {len(df) - df['Correct'].sum()}")

print(f"\nAccuracy            : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(
    f"Balanced Accuracy   : "
    f"{balanced_accuracy:.4f} "
    f"({balanced_accuracy * 100:.2f}%)"
)

print(
    f"Precision (AI)      : "
    f"{precision:.4f} "
    f"({precision * 100:.2f}%)"
)

print(
    f"Recall (AI)         : "
    f"{recall:.4f} "
    f"({recall * 100:.2f}%)"
)

print(
    f"Specificity         : "
    f"{specificity:.4f} "
    f"({specificity * 100:.2f}%)"
)

print(
    f"F1 Score (AI)      : "
    f"{f1:.4f} "
    f"({f1 * 100:.2f}%)"
)


print("\nConfusion Matrix")
print("----------------")
print("              Predicted")
print("              AI    Real")
print(f"Actual AI     {tp:3d}   {fn:3d}")
print(f"Actual Real   {fp:3d}   {tn:3d}")


print("\nClassification Report")
print("---------------------")

print(
    classification_report(
        y_true,
        y_pred,
        labels=["ai_generated", "real"],
        target_names=["AI", "Real"]
    )
)
