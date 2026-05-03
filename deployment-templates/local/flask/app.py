"""
Flask Deployment Template for NTF Models
Simple API serving for fine-tuned transformer models
"""

from flask import Flask, request, jsonify
import torch
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global model variables
model = None
tokenizer = None
model_name = None
device = None


def load_model(model_path=None, model_name_param=None):
    """Load model and tokenizer at startup."""
    global model, tokenizer, model_name, device
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        if model_path:
            from ntf.models import ModelRegistry
            
            registry = ModelRegistry(registry_path=model_path)
            versions = registry.list_versions()
            if not versions:
                raise ValueError("No model versions found in registry")
            latest_version = versions[-1]
            model, tokenizer = registry.load_model_and_tokenizer(version=latest_version)
            model_name = f"local:{latest_version}"
            
        elif model_name_param:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name_param,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name_param)
            model_name = model_name_param
        else:
            raise ValueError("Either model_path or model_name must be provided")
        
        model.eval()
        model.to(device)
        logger.info(f"Model loaded successfully: {model_name}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


# Load model on startup
if __name__ == "__main__":
    model_path = os.getenv("NTF_MODEL_PATH")
    model_name_param = os.getenv("NTF_MODEL_NAME")
    if model_path or model_name_param:
        load_model(model_path, model_name_param)


@app.route("/health", methods=["GET"])
def health_check():
    """Check API health and model status."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": model_name,
        "device": str(device) if device else "not_initialized",
        "timestamp": datetime.utcnow().isoformat()
    })


@app.route("/generate", methods=["POST"])
def generate():
    """Generate text based on input prompt."""
    
    if model is None or tokenizer is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    try:
        data = request.get_json()
        prompt = data.get("prompt", "")
        max_tokens = data.get("max_tokens", 256)
        temperature = data.get("temperature", 0.7)
        
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
        
        start_time = datetime.now()
        
        # Tokenize input
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048 - max_tokens
        ).to(device)
        
        input_length = inputs["input_ids"].shape[1]
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode output
        generated_ids = outputs[0][input_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        end_time = datetime.now()
        generation_time = (end_time - start_time).total_seconds() * 1000
        
        return jsonify({
            "generated_text": generated_text,
            "prompt": prompt,
            "full_text": prompt + generated_text,
            "tokens_used": len(generated_ids),
            "generation_time_ms": generation_time,
            "model_version": model_name or "unknown",
            "timestamp": end_time.isoformat()
        })
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/model/info", methods=["GET"])
def model_info():
    """Get information about the loaded model."""
    
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    info = {
        "model_name": model_name,
        "device": str(device),
        "num_parameters": sum(p.numel() for p in model.parameters()),
    }
    
    return jsonify(info)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
