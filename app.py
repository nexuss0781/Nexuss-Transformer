"""
Gradio Inference UI for Nexuss Transformer - Blank Slate Model
Deploy on Hugging Face Spaces for interactive demo
"""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Global variables for model caching
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_path: str = "nexuss0781/Nexuss-AIv1"):
    """Load the trained model and tokenizer"""
    global model, tokenizer
    
    try:
        # Try to load from local checkpoint first
        if os.path.exists("checkpoints/nexuss-aiv1"):
            model_path = "checkpoints/nexuss-aiv1"
        
        print(f"Loading model from: {model_path}")
        print(f"Using device: {device}")
        
        # Load tokenizer (EthioBBPE)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        
        if device != "cuda":
            model = model.to(device)
        
        model.eval()
        print("✓ Model loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


def generate_text(
    prompt: str,
    max_length: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    do_sample: bool = True
) -> str:
    """Generate text from the model"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        return "❌ Model not loaded. Please wait for initialization or check the model path."
    
    try:
        # Tokenize input
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=max_length,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Remove the prompt from the output for cleaner display
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):]
        
        return generated_text.strip()
        
    except Exception as e:
        return f"❌ Generation error: {str(e)}"


def create_demo():
    """Create Gradio interface"""
    
    with gr.Blocks(
        title="Nexuss AI - Ethiopian Orthodox Religious Texts",
        theme=gr.themes.Soft(),
        css="""
        .container { max-width: 900px; margin: auto; }
        .header { text-align: center; margin-bottom: 20px; }
        """
    ) as demo:
        
        gr.Markdown("""
        # 🇪🇹 Nexuss AI - Ethiopian Orthodox Religious Knowledge
        
        **Base Knowledge:** Synaxarium (ስንክሳር) + Amharic Bible (መጽሐፍ ቅዱስ)
        
        This AI has been trained on Ethiopian Orthodox religious texts and can:
        - Answer questions about saints and their stories
        - Recite biblical verses in Amharic
        - Provide religious context and teachings
        
        *Note: This is a blank slate model trained only on religious texts.*
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                prompt_input = gr.Textbox(
                    label="Your Question / Prompt (አማርኛ / English)",
                    placeholder="e.g., ስለ አቡነ ተከለ ሃይማኖት ንገረኝ (Tell me about Abune Tekle Haymanot)",
                    lines=3,
                    max_lines=5
                )
                
                with gr.Accordion("Advanced Settings", open=False):
                    max_length_slider = gr.Slider(
                        minimum=32, maximum=512, value=256, step=32,
                        label="Max Generation Length"
                    )
                    temperature_slider = gr.Slider(
                        minimum=0.1, maximum=1.5, value=0.7, step=0.1,
                        label="Temperature (Creativity)"
                    )
                    top_p_slider = gr.Slider(
                        minimum=0.5, maximum=1.0, value=0.9, step=0.05,
                        label="Top-p Sampling"
                    )
                    repetition_penalty_slider = gr.Slider(
                        minimum=1.0, maximum=2.0, value=1.2, step=0.1,
                        label="Repetition Penalty"
                    )
                    do_sample_checkbox = gr.Checkbox(
                        value=True, label="Enable Sampling (more creative)"
                    )
                
                generate_btn = gr.Button("🚀 Generate Response", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                output_text = gr.Textbox(
                    label="AI Response",
                    lines=8,
                    max_lines=15,
                    show_copy_button=True
                )
        
        # Example prompts
        examples = [
            ["ስለ አቡነ ተከለ ሃይማኖት ንገረኝ"],
            ["የዛሬው ቅዱስ ማነው?"],
            ["መጽሐፍ ቅዱስ ምን ይላል ስለ ፍቅር?"],
            ["Who is Saint George in Ethiopian Orthodoxy?"],
            ["Explain the story of the Nine Saints"],
        ]
        
        gr.Examples(
            examples=examples,
            inputs=prompt_input,
            label="Example Prompts"
        )
        
        # Connect button to generation function
        generate_btn.click(
            fn=generate_text,
            inputs=[
                prompt_input,
                max_length_slider,
                temperature_slider,
                top_p_slider,
                repetition_penalty_slider,
                do_sample_checkbox
            ],
            outputs=output_text
        )
        
        # Footer
        gr.Markdown("""
        ---
        **Powered by Nexuss Transformer Framework (NTF)**  
        Model: Blank Slate trained on Synaxarium + Canon Biblical Corpus  
        Tokenizer: EthioBBPE (16K vocabulary)
        """)
    
    return demo


if __name__ == "__main__":
    print("🚀 Starting Nexuss AI Gradio Demo...")
    print(f"Device: {device}")
    
    # Try to load model (optional for Spaces - can load on first request)
    # load_model()
    
    # Create and launch demo
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
