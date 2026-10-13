# Dataset splits

The source dataset is `Data/processed/standardized_1000.jsonl`. Run the split
script from the project root in the VS Code PowerShell terminal:

```powershell
.\.venv\Scripts\python.exe .\prepare_dataset_splits.py
```

This creates `train.jsonl` (80%), `validation.jsonl` (10%), and `test.jsonl`
(10%) under `Data/processed/splits/`. The default random seed is `42`; use
`--seed 123` to choose another reproducible seed. The source dataset is only
read, never modified.

The script groups records that share a normalized dialect or Standard Kannada
sentence, then assigns each group as a unit. Its checks verify IDs, cross-split
sentence leakage, and that dialect plus changed/unchanged proportions stay
within two percentage points of the source distribution. Dataset files under
`Data/`, generated outputs, and environment files are excluded from Git.

## Live conversion demo

To demonstrate live D3/D5-to-Standard-Kannada conversion with Gemini, set
`GEMINI_API_KEY` in the Git-ignored project `.env` file and run:

```powershell
.\.venv\Scripts\python.exe .\demo_kannada_conversion.py
```

The demo selects five reproducible examples per dialect from the held-out test
split, sends only each dialect sentence to the model, and prints the model
output and request latency. It also reports average client-observed latency.
The overall run time includes the five-second spacing between API requests.
This is a live Gemini demonstration, not a measurement of a fine-tuned
IndicBART model.
