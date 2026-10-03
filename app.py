# ============================================================
# PATIENT CONVERSATION RESPONSE GENERATION
# Dataset: fastino/fast-decisions
# Subset: agent_handoff
# Model: T5-small
# ============================================================


# ============================================================
# 1. INSTALL REQUIRED LIBRARIES
# ============================================================

!pip install -q -U transformers datasets accelerate sentencepiece


# ============================================================
# 2. IMPORT LIBRARIES
# ============================================================

import re
import torch
import pandas as pd

from datasets import load_dataset, Dataset

from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer
)


# ============================================================
# 3. CHECK DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("DEVICE INFORMATION")
print("=" * 70)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("GPU not available.")
    print("The notebook will use CPU.")


# ============================================================
# 4. LOAD DATASET
# ============================================================

print("\n")
print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

dataset = load_dataset(
    "fastino/fast-decisions",
    "agent_handoff",
    split="train"
)

print(dataset)
print("\nColumns:")
print(dataset.column_names)


# ============================================================
# 5. CONVERT TO PANDAS
# ============================================================

df = dataset.to_pandas()

print("\nDataset shape:")
print(df.shape)


# ============================================================
# 6. VIEW FIRST DATASET EXAMPLE
# ============================================================

print("\n")
print("=" * 70)
print("FIRST DATASET EXAMPLE")
print("=" * 70)

print(df.iloc[0])


# ============================================================
# 7. EXTRACT LABEL SAFELY
# ============================================================

def get_label(output):

    try:

        classifications = output.get(
            "classifications",
            []
        )

        if len(classifications) > 0:

            true_label = classifications[0].get(
                "true_label",
                []
            )

            if isinstance(true_label, list):

                if len(true_label) > 0:
                    return str(true_label[0]).lower()

            return str(true_label).lower()

    except:
        pass

    return "no"


# ============================================================
# 8. CREATE BASIC DATAFRAME
# ============================================================

df["conversation"] = df["input"]

df["label"] = df["output"].apply(
    get_label
)

df = df[
    [
        "conversation",
        "label"
    ]
].copy()

print("\nExtracted data:")
print(df.head())


# ============================================================
# 9. HEALTHCARE KEYWORDS
# ============================================================

health_keywords = [
    "clinic",
    "patient",
    "medical",
    "doctor",
    "prescription",
    "symptom",
    "appointment",
    "health",
    "medicine",
    "refill",
    "urgent",
    "clinician",
    "hospital",
    "healthcare",
    "treatment",
    "pain",
    "diagnosis",
    "pharmacy",
    "medication",
    "sick"
]

health_pattern = "|".join(
    health_keywords
)


# ============================================================
# 10. SELECT HEALTHCARE CONVERSATIONS
# ============================================================

health_df = df[
    df["conversation"]
    .astype(str)
    .str.lower()
    .str.contains(
        health_pattern,
        regex=True,
        na=False
    )
].copy()

print("\nHealthcare conversations found:")
print(len(health_df))


# ============================================================
# 11. FALLBACK
# ============================================================

if len(health_df) < 10:

    print("\nNot enough healthcare conversations found.")
    print("Using the complete agent_handoff dataset.")

    health_df = df.copy()


# ============================================================
# 12. TEXT CLEANING FUNCTION
# ============================================================

def clean_text(text):

    text = str(text)

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Remove mentions
    text = re.sub(
        r"@\w+",
        " ",
        text
    )

    # Remove hashtags
    text = re.sub(
        r"#\w+",
        " ",
        text
    )

    # Normalize line breaks
    text = re.sub(
        r"[\r\n\t]+",
        " ",
        text
    )

    # Remove special characters
    text = re.sub(
        r"[^a-zA-Z0-9\s.,!?'-]",
        " ",
        text
    )

    # Remove unnecessary spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    # Remove leading/trailing spaces
    text = text.strip()

    return text


# ============================================================
# 13. CLEAN CONVERSATIONS
# ============================================================

health_df["cleaned_message"] = (
    health_df["conversation"]
    .apply(clean_text)
)

health_df = health_df[
    health_df["cleaned_message"]
    .str.len() > 5
].reset_index(drop=True)


print("\nSamples after cleaning:")
print(len(health_df))


# ============================================================
# 14. SHOW BEFORE AND AFTER CLEANING
# ============================================================

print("\n")
print("=" * 70)
print("BEFORE AND AFTER CLEANING")
print("=" * 70)

for i in range(
    min(5, len(health_df))
):

    print("\n" + "-" * 70)

    print("BEFORE:")
    print(
        health_df.loc[
            i,
            "conversation"
        ]
    )

    print("\nAFTER:")
    print(
        health_df.loc[
            i,
            "cleaned_message"
        ]
    )


# ============================================================
# 15. CREATE RESPONSE TARGETS
# ============================================================

def create_response(label):

    label = str(label).lower()

    if label == "yes":

        return (
            "I understand that this situation may be concerning. "
            "Please contact a qualified healthcare professional "
            "or appropriate clinic staff for further assistance."
        )

    else:

        return (
            "Thank you for contacting us. "
            "I understand your request. "
            "We can provide general information and help "
            "with routine questions. Please contact the clinic "
            "if you need further assistance."
        )


health_df["expected_response"] = (
    health_df["label"]
    .apply(create_response)
)


# ============================================================
# 16. CREATE T5 INPUT FORMAT
# ============================================================

health_df["input_text"] = (
    "Generate a polite and empathetic response "
    "to the patient: "
    + health_df["cleaned_message"]
)

health_df["target_text"] = (
    health_df["expected_response"]
)


# ============================================================
# 17. SHOW INPUT AND OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("INPUT / OUTPUT EXAMPLES")
print("=" * 70)

for i in range(
    min(5, len(health_df))
):

    print("\n" + "-" * 70)

    print("INPUT:")
    print(
        health_df.loc[
            i,
            "input_text"
        ]
    )

    print("\nOUTPUT:")
    print(
        health_df.loc[
            i,
            "target_text"
        ]
    )


# ============================================================
# 18. KEEP DATASET SMALL
# ============================================================

max_samples = 100

if len(health_df) > max_samples:

    health_df = health_df.sample(
        n=max_samples,
        random_state=42
    ).reset_index(drop=True)


print("\nFinal samples used:")
print(len(health_df))


# ============================================================
# 19. CREATE HUGGING FACE DATASET
# ============================================================

model_df = health_df[
    [
        "input_text",
        "target_text"
    ]
].copy()

hf_dataset = Dataset.from_pandas(
    model_df,
    preserve_index=False
)

print("\nHugging Face dataset:")
print(hf_dataset)


# ============================================================
# 20. TRAIN / TEST SPLIT
# ============================================================

split_dataset = hf_dataset.train_test_split(
    test_size=0.2,
    seed=42
)

train_dataset = split_dataset["train"]

test_dataset = split_dataset["test"]

print("\nTraining samples:")
print(len(train_dataset))

print("\nTesting samples:")
print(len(test_dataset))


# ============================================================
# 21. LOAD T5-SMALL TOKENIZER
# ============================================================

print("\n")
print("=" * 70)
print("LOADING T5-SMALL")
print("=" * 70)

model_name = "t5-small"

tokenizer = T5Tokenizer.from_pretrained(
    model_name
)


# ============================================================
# 22. LOAD T5-SMALL MODEL
# ============================================================

model = T5ForConditionalGeneration.from_pretrained(
    model_name
)

model = model.to(device)

print(
    "Model device:",
    next(model.parameters()).device
)


# ============================================================
# 23. TOKENIZATION
# ============================================================

def tokenize_function(examples):

    model_inputs = tokenizer(
        examples["input_text"],
        max_length=256,
        truncation=True
    )

    labels = tokenizer(
        text_target=examples["target_text"],
        max_length=128,
        truncation=True
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


# ============================================================
# 24. TOKENIZE TRAINING DATA
# ============================================================

tokenized_train = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=[
        "input_text",
        "target_text"
    ]
)


# ============================================================
# 25. TOKENIZE TEST DATA
# ============================================================

tokenized_test = test_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=[
        "input_text",
        "target_text"
    ]
)

print("\nTokenization completed.")


# ============================================================
# 26. DATA COLLATOR
# ============================================================

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)


# ============================================================
# 27. TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(

    output_dir="./patient_response_t5",

    eval_strategy="epoch",

    save_strategy="epoch",

    logging_strategy="steps",

    logging_steps=5,

    learning_rate=5e-5,

    per_device_train_batch_size=4,

    per_device_eval_batch_size=4,

    num_train_epochs=3,

    weight_decay=0.01,

    save_total_limit=2,

    report_to="none",

    fp16=torch.cuda.is_available(),

    load_best_model_at_end=False
)


# ============================================================
# 28. CREATE TRAINER
# ============================================================
# IMPORTANT:
# New Transformers versions use:
# processing_class=tokenizer
#
# NOT:
# tokenizer=tokenizer
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=tokenized_train,

    eval_dataset=tokenized_test,

    processing_class=tokenizer,

    data_collator=data_collator
)

print("\nTrainer created successfully.")


# ============================================================
# 29. TRAIN MODEL
# ============================================================

print("\n")
print("=" * 70)
print("STARTING T5-SMALL TRAINING")
print("=" * 70)

training_result = trainer.train()

print("\nTraining completed.")

print("\nTraining result:")
print(training_result)


# ============================================================
# 30. EVALUATE MODEL
# ============================================================

print("\n")
print("=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

evaluation = trainer.evaluate()

print(evaluation)


# ============================================================
# 31. SAVE TRAINED MODEL
# ============================================================

trainer.save_model(
    "./patient_response_t5"
)

tokenizer.save_pretrained(
    "./patient_response_t5"
)

print(
    "\nTrained model saved successfully."
)


# ============================================================
# 32. INFERENCE FUNCTION
# ============================================================

def generate_response(
    text,
    model_to_use
):

    model_to_use.eval()

    cleaned = clean_text(text)

    prompt = (
        "Generate a polite and empathetic "
        "response to the patient: "
        + cleaned
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    # Get model's actual device
    model_device = next(
        model_to_use.parameters()
    ).device

    # Move input tensors to same device
    inputs = {
        key: value.to(model_device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        output = model_to_use.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )

    response = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    return response


# ============================================================
# 33. TEST TRAINED MODEL
# ============================================================

print("\n")
print("=" * 70)
print("TEST TRAINED MODEL")
print("=" * 70)

test_message = health_df.iloc[0]["conversation"]

print("\nPatient message:")
print(test_message)

print("\nCleaned message:")
print(
    clean_text(test_message)
)

print("\nGenerated response:")
print(
    generate_response(
        test_message,
        model
    )
)


# ============================================================
# 34. LOAD ORIGINAL T5-SMALL
# ============================================================

print("\n")
print("=" * 70)
print("LOADING ORIGINAL T5-SMALL")
print("=" * 70)

base_model = T5ForConditionalGeneration.from_pretrained(
    model_name
)

base_model = base_model.to(device)

print(
    "Base model device:",
    next(base_model.parameters()).device
)


# ============================================================
# 35. SELECT FIVE REAL DATASET EXAMPLES
# ============================================================

five_examples = health_df.head(
    min(5, len(health_df))
).copy()


# ============================================================
# 36. BEFORE / AFTER COMPARISON
# ============================================================

results = []

print("\n")
print("=" * 70)
print("BEFORE VS AFTER FINE-TUNING")
print("=" * 70)

for i, row in five_examples.iterrows():

    patient_message = row[
        "conversation"
    ]

    cleaned_message = row[
        "cleaned_message"
    ]

    expected_response = row[
        "expected_response"
    ]

    before_response = generate_response(
        patient_message,
        base_model
    )

    after_response = generate_response(
        patient_message,
        model
    )

    print("\n")
    print("=" * 70)

    print("\nPATIENT MESSAGE:")
    print(patient_message)

    print("\nCLEANED PATIENT MESSAGE:")
    print(cleaned_message)

    print("\nEXPECTED RESPONSE:")
    print(expected_response)

    print("\nMODEL RESPONSE BEFORE FINE-TUNING:")
    print(before_response)

    print("\nMODEL RESPONSE AFTER FINE-TUNING:")
    print(after_response)

    results.append({

        "Patient Message":
            patient_message,

        "Cleaned Patient Message":
            cleaned_message,

        "Expected Response":
            expected_response,

        "Model Before Fine-Tuning":
            before_response,

        "Model After Fine-Tuning":
            after_response
    })


# ============================================================
# 37. COMPARISON TABLE
# ============================================================

comparison_df = pd.DataFrame(
    results
)

pd.set_option(
    "display.max_colwidth",
    500
)

print("\n")
print("=" * 70)
print("FINAL COMPARISON TABLE")
print("=" * 70)

display(
    comparison_df
)


# ============================================================
# 38. SAVE FINAL MODEL
# ============================================================

trainer.save_model(
    "./patient_response_t5_final"
)

tokenizer.save_pretrained(
    "./patient_response_t5_final"
)

print(
    "\nFinal model saved successfully."
)


# ============================================================
# 39. ZIP MODEL
# ============================================================

!zip -r patient_response_t5_final.zip patient_response_t5_final


# ============================================================
# 40. FINAL STATUS
# ============================================================

print("\n")
print("=" * 70)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 70)

print("""
Pipeline:

Fast Decisions Dataset
        ↓
Healthcare Conversation Selection
        ↓
Text Cleaning
        ↓
Before / After Cleaning
        ↓
Input / Output Formatting
        ↓
Train / Test Split
        ↓
T5-small Tokenization
        ↓
T5-small Fine-tuning
        ↓
Training Logs
        ↓
Evaluation
        ↓
Model Saving
        ↓
Inference
        ↓
Five Real Dataset Examples
        ↓
Before / After Comparison

Model:
T5-small

Dataset:
fastino/fast-decisions
Subset:
agent_handoff

The generated responses are for educational
machine-learning demonstration only.

They are NOT medical diagnosis, treatment,
or professional medical advice.
""")
