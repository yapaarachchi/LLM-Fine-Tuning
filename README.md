## 🤖 Efficient Fine-Tuning for JSON Extraction (Phi-3 Mini with Unsloth) ##

This repository contains a streamlined Jupyter notebook (Fine Tuning.ipynb) demonstrating how to fine-tune the Phi-3-mini-4k-instruct model for a specialized task: extracting unstructured data into a structured JSON format.

We utilize the highly efficient Unsloth library alongside QLoRA/PEFT techniques to achieve rapid training and reduced memory consumption, making state-of-the-art LLM fine-tuning accessible on consumer-grade GPUs (like the T4 used in the original notebook).

## ✨ Key Features ##
* Task-Specific LLM Fine-Tuning: Focuses on training a model for reliable and accurate JSON data extraction from arbitrary text inputs (e.g., HTML snippets).

* Optimized Training: Uses Unsloth to implement QLoRA (Quantized LoRA) with bitsandbytes (4-bit quantization), providing up to 2x faster training and 70% less memory usage compared to standard methods.

* Model: Fine-tunes the Phi-3-mini-4k-instruct model (a powerful and efficient Microsoft Small Language Model).

* SFT (Supervised Fine-Tuning): Leverages the trl (Transformer Reinforcement Learning) library's SFTTrainer for efficient training loop management.

* Quantized Deployment: Includes steps to export the fine-tuned model into the GGUF format (q4_k_m quantization), optimized for CPU inference and tools like llama.cpp.

## ⚙️ Prerequisites and Setup ##

* Environment: A Python environment (Jupyter, Google Colab, or local machine) with a CUDA-enabled GPU is highly recommended for performance.

* Dataset: Requires a training dataset named json_extraction_dataset_500.json (or similar) in the root directory. The data must be in a format that maps an input (text to be processed) to an output (the desired JSON structure).

## 🛠️ Installation ##

```
# Uninstall existing conflicting packages (if any)
!pip uninstall -y unsloth peft

# Install the necessary, optimized packages
!pip install unsloth trl peft accelerate bitsandbytes
```

## 🚀 Usage and Workflow ##

1. The fine-tuning process is structured sequentially in the Fine Tuning.ipynb notebook:

2. Data Loading: The notebook loads the training data from json_extraction_dataset_500.json.

3. Model Loading: The base model (unsloth/Phi-3-mini-4k-instruct-bnb-4bit) is loaded using FastLanguageModel.from_pretrained with 4-bit quantization.

4. Data Preparation: A custom format_prompt function is used to convert the raw JSON dataset into the specific chat/instruction format required for instruction-tuning the Phi-3 model.

5. LoRA Setup: LoRA adapters are initialized on the model with specific rank (r=64) and scaling factor (lora_alpha=128).

6. Training: The SFTTrainer handles the Supervised Fine-Tuning with optimized training arguments (e.g., adamw_8bit optimizer, gradient_accumulation_steps=4, num_train_epochs=3).

7. Inference: A test prompt is run using the fine-tuned model to demonstrate its JSON extraction capabilities.

8. Export: The final, trained adapter weights are merged and saved to the GGUF format, enabling deployment on local machines.

## 📄 Example Input and Output Structure ##

The training format follows an instruction-following pattern:

### Input Prompt (Example from Notebook): ###

```
### Input: Extract the product information:
<div class='product'><h2>iPad Air</h2><span class='price'>$1344</span><span class='category'>audio</span><span class='brand'>Dell</span></div>
### Output: <The model is trained to output the desired JSON here>
```

### Expected Output (Inferred JSON Structure): ###

```
{
  "name": "iPad Air",
  "price": "$1344",
  "category": "audio",
  "brand": "Dell"
}
```

