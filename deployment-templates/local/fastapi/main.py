"""
FastAPI Deployment Template for NTF Models
Production-ready API serving for fine-tuned transformer models
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import torch
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NTF Model Serving API",
    description="API for serving fine-tuned transformer models using NTF",
    version="1.0.0"
)


class GenerationRequest(BaseModel):
    """Request model for text generation."""
    
    prompt: str = Field(..., min_length=1, max_length=4096, description="Input prompt for generation")
    max_tokens: int = Field(default=256, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    top_k: int = Field(default=50, ge=0, le=500, description="Top-k sampling parameter")
    repetition_penalty: float = Field(default=1.0, ge=0.0, le=2.0, description="Repetition penalty")
    do_sample: bool = Field(default=True, description="Whether to use sampling")
    stop_sequences: Optional[List[str]] = Field(default=None, description="Stop sequences")


class GenerationResponse(BaseModel):
    """Response model for text generation."""
    
    generated_text: str
    prompt: str
    full_text: str
    tokens_used: int
    generation_time_ms: float
    model_version: str
    timestamp: str


class ClassificationRequest(BaseModel):
    """Request model for classification tasks."""
    
    text: str = Field(..., min_length=1, max_length=4096, description="Text to classify")


class ClassificationResponse(BaseModel):
    """Response model for classification tasks."""
    
    label: str
    score: float
    all_scores: List[Dict[str, float]]
    model_version: str
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    model_loaded: bool
    model_name: Optional[str]
    device: str
    timestamp: str


# Global model variables
model = None
tokenizer = None
model_name = None
device = None


def load_model(model_path: Optional[str] = None, model_name_param: Optional[str] = None):
    """Load model and tokenizer at startup."""
    global model, tokenizer, model_name, device
    
    try:
        from ntf.models import ModelRegistry
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        if model_path:
            # Load from local registry
            registry = ModelRegistry(registry_path=model_path)
            # Get latest version or specify
            versions = registry.list_versions()
            if not versions:
                raise ValueError("No model versions found in registry")
            latest_version = versions[-1]  # Assuming sorted by date
            model, tokenizer = registry.load_model_and_tokenizer(version=latest_version)
            model_name = f"local:{latest_version}"
            
        elif model_name_param:
            # Load from HuggingFace
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name_param,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                device_map="auto" if device.type == "cuda" else None
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


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    model_path = os.getenv("NTF_MODEL_PATH")
    model_name_param = os.getenv("NTF_MODEL_NAME")
    
    if model_path or model_name_param:
        load_model(model_path, model_name_param)
    else:
        logger.warning("No model configured. Set NTF_MODEL_PATH or NTF_MODEL_NAME environment variable.")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_name=model_name,
        device=str(device) if device else "not_initialized",
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    """Generate text based on input prompt."""
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start_time = datetime.now()
        
        # Tokenize input
        inputs = tokenizer(
            request.prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048 - request.max_tokens
        ).to(device)
        
        input_length = inputs["input_ids"].shape[1]
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k if request.top_k > 0 else None,
                repetition_penalty=request.repetition_penalty,
                do_sample=request.do_sample,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode output
        generated_ids = outputs[0][input_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Handle stop sequences
        if request.stop_sequences:
            for stop_seq in request.stop_sequences:
                if stop_seq in generated_text:
                    generated_text = generated_text.split(stop_seq)[0]
        
        end_time = datetime.now()
        generation_time = (end_time - start_time).total_seconds() * 1000
        
        return GenerationResponse(
            generated_text=generated_text,
            prompt=request.prompt,
            full_text=request.prompt + generated_text,
            tokens_used=len(generated_ids),
            generation_time_ms=generation_time,
            model_version=model_name or "unknown",
            timestamp=end_time.isoformat()
        )
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify", response_model=ClassificationResponse)
async def classify(request: ClassificationRequest):
    """Classify input text."""
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Tokenize
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            if hasattr(model, 'config') and hasattr(model.config, 'id2label'):
                # Classification model with id2label mapping
                probs = torch.softmax(logits, dim=-1)
                scores, indices = torch.topk(probs[0], k=min(5, probs.size(-1)))
                
                all_scores = [
                    {"label": model.config.id2label[idx.item()], "score": score.item()}
                    for idx, score in zip(indices, scores)
                ]
                
                label = all_scores[0]["label"]
                score = all_scores[0]["score"]
            else:
                # Causal LM - return perplexity-based score
                loss = outputs.loss if hasattr(outputs, 'loss') else None
                label = "causal_lm"
                score = torch.exp(-loss).item() if loss is not None else 0.0
                all_scores = [{"label": label, "score": score}]
        
        return ClassificationResponse(
            label=label,
            score=score,
            all_scores=all_scores,
            model_version=model_name or "unknown",
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
async def model_info():
    """Get information about the loaded model."""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    info = {
        "model_name": model_name,
        "device": str(device),
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "is_training": model.training,
    }
    
    if hasattr(model, 'config'):
        info["config"] = {
            "vocab_size": model.config.vocab_size,
            "hidden_size": model.config.hidden_size,
            "num_layers": model.config.num_hidden_layers,
            "num_attention_heads": model.config.num_attention_heads,
        }
    
    return info


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
