from transformers import AutoImageProcessor, AutoModelForImageClassification

model_name = "delpot/steganograph-ia-detector"

print("Loading model...")

processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

model.eval()

print("\nModel loaded successfully!")
print("Class mapping:")
print(model.config.id2label)
