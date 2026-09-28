# Tanglish Autocomplete — DistilGPT2

A two-stage pipeline: **train on Kaggle** (free GPU), **serve locally** (CPU or GPU).

```
autocomplete-genAI/
├── notebook/
│   └── train_tanglish_distilgpt2.ipynb   ← Kaggle training notebook
├── models/
│   └── tanglish-distilgpt2/              ← extract Kaggle zip here
├── static/
│   └── index.html                        ← ghost-text autocomplete UI
├── inference.py                          ← generate_completion() API
├── app.py                                ← FastAPI backend
├── requirements.txt                      ← local serving only
└── README.md
```

---

## Step-by-step workflow

### 1 · Prepare your dataset on Kaggle

Create a **Kaggle Dataset** containing two CSV files:

| File | Columns |
|------|---------|
| `train.csv` | `text`, `language` |
| `val.csv` | `text`, `language` |

Example rows:
```csv
text,language
"Naan gym ku pogaren bro, enna solre?",tanglish
"Avan romba late ah varuvaan",tanglish
```

The `language` column is optional — only `text` is used during training.

---

### 2 · Run the notebook on Kaggle

1. Go to [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**.
2. Upload (or import from GitHub) `notebook/train_tanglish_distilgpt2.ipynb`.
3. Attach your dataset: *Add Data → Your Datasets → select the one you created*.
4. Enable GPU: *Session Options → Accelerator → GPU T4 x2*.
5. **Open the first code cell** and set:
   ```python
   DATASET_NAME = "your-dataset-folder-name"   # must match what Kaggle shows under /kaggle/input/
   ```
6. Click **Run All**.

Training runs ~10–20 min on a T4. When done, the notebook saves the model and prints:
```
Done. ZIP size: ~320 MB
Download it from: Kaggle Output panel → tanglish-distilgpt2.zip
```

---

### 3 · Download and extract the model

From the Kaggle Output panel, download `tanglish-distilgpt2.zip`.

Extract it so the files land in `models/tanglish-distilgpt2/`:

```bash
# macOS / Linux
unzip tanglish-distilgpt2.zip -d models/

# Windows (PowerShell)
Expand-Archive tanglish-distilgpt2.zip -DestinationPath models\
```

The folder should look like:
```
models/tanglish-distilgpt2/
├── config.json
├── generation_config.json
├── model.safetensors      (or pytorch_model.bin)
├── tokenizer.json
├── tokenizer_config.json
├── vocab.json
└── merges.txt
```

---

### 4 · Install local dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

### 5 · Run the server

```bash
uvicorn app:app --reload
```

Open **http://localhost:8000** in your browser.

You'll see an inline autocomplete input: type a Tanglish phrase, wait ~400 ms, and the model's suggestion appears as **greyed-out ghost text**. Press **Tab** to accept it.

---

## API reference

### `POST /complete`

```json
// Request
{ "text": "Naan gym ku pogaren", "max_new_tokens": 15 }

// Response
{ "completion": "bro, enna panre?" }
```

### `GET /health`

```json
{ "status": "ok", "device": "cpu" }
```

---

## CLI inference (no server)

```bash
python inference.py
```

Or import directly:

```python
from inference import generate_completion

result = generate_completion("Naan gym ku pogaren", max_new_tokens=20)
print(result)  # → "bro, enna panre?"
```

---

## Generation parameters (inference.py defaults)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `max_new_tokens` | `15` | Keep short for autocomplete feel |
| `temperature` | `0.8` | Balanced creativity |
| `top_p` | `0.9` | Nucleus sampling |
| `top_k` | `50` | Hard cap on token candidates |
| `repetition_penalty` | `1.2` | Reduces repetitive output |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Model directory not found` | Verify zip was extracted into `models/tanglish-distilgpt2/` |
| OOM on Kaggle | Reduce `BATCH_SIZE` to `4` in the config cell |
| Empty completions | Increase `max_new_tokens`; lower `temperature` to `0.6` |
| Slow first request | Expected — model loads lazily on the first `/complete` call |
| `DATASET_NAME` error | Check the exact folder name under `/kaggle/input/` in the notebook sidebar |

---

## License

MIT
