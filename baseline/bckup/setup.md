# Baseline Syllogism Classifier Setup

## Prerequisites

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start vLLM server:**
   ```bash
   # Replace with your actual model path
   python -m vllm.entrypoints.openai.api_server \
     --model /path/to/your/llama/model \
     --port 8192 \
     --host localhost
   ```

## Usage

Run the baseline classifier:

```bash
python baseline.py \
  --model llama \
  --prompt prompt1.prompt \
  train_data.json \
  results.json
```

## Files

- `baseline.py` - Main classifier script
- `prompt.prompt` - Simple prompt template
- `prompt1.prompt` - Detailed formal logic prompt template
- `train_data.json` - Training/test data with syllogisms
- `test_baseline.py` - Test script to verify setup
- `requirements.txt` - Python dependencies

## Testing

Run the test script to verify everything works:

```bash
python test_baseline.py
```

This will create a small test dataset and run the classifier on it.