# 🚀 Deploying Nexuss Transformer to Hugging Face Spaces

Complete guide for deploying your Ethiopian Orthodox Religious AI model on Hugging Face Spaces.

## 📋 Prerequisites

1. **Hugging Face Account**: Sign up at https://huggingface.co
2. **Git LFS**: Install from https://git-lfs.com
3. **Hugging Face CLI**: `pip install huggingface_hub`

## 🔧 Step 1: Login to Hugging Face

```bash
huggingface-cli login
# Enter your HF token (get it from https://huggingface.co/settings/tokens)
```

## 🏗️ Step 2: Create a New Space

1. Go to https://huggingface.co/spaces
2. Click **"Create new Space"**
3. Configure:
   - **Space name**: `nexuss-ai-ethiopian` (or your preferred name)
   - **License**: MIT
   - **SDK**: Docker
   - **Visibility**: Public (or Private)
4. Click **"Create Space"**

## 📦 Step 3: Push Code to Your Space

### Option A: Push via Git (Recommended)

```bash
# Clone your new space
cd /workspace
git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
cd YOUR_SPACE_NAME

# Copy files from Nexuss-Transformer
cp /workspace/Dockerfile .
cp /workspace/app.py .
cp /workspace/requirements.txt .
cp -r /workspace/Nexuss-Transformer/* .  # Optional: include full framework

# Add README for Spaces (already created as README.md)
git add .
git commit -m "Initial commit: Nexuss AI deployment"
git push
```

### Option B: Direct Push from Current Repo

```bash
# Add HF Spaces as remote
cd /workspace
git remote add spaces https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME

# Push only deployment files
git subtree split --prefix=Dockerfile,app.py,requirements.txt -b spaces-deploy
git push spaces spaces-deploy:main --force
```

## ⚙️ Step 4: Configure Space Settings

In your Space settings (Settings tab):

1. **Hardware**: Select GPU if available (T4, A10G, etc.)
2. **Docker Image**: Auto-detected from Dockerfile
3. **Secrets** (if needed):
   - `HF_TOKEN`: Your Hugging Face token
   - `MODEL_PATH`: Path to your trained model

## 🏃 Step 5: Run Training in Spaces (Optional)

If you want to train directly in the Space:

### Create `train.sh`:

```bash
#!/bin/bash
set -e

echo "🚀 Starting Blank Slate Training..."

python train_blank_slate.py \
  --model_size small \
  --num_epochs 3 \
  --batch_size 8 \
  --learning_rate 0.0005 \
  --warmup_steps 500 \
  --max_seq_length 512 \
  --mixed_precision fp16 \
  --gradient_checkpointing \
  --output_dir checkpoints/nexuss-aiv1 \
  --run_name hf_spaces_training

echo "✅ Training complete! Model saved to checkpoints/"
```

Make it executable and update Dockerfile:

```bash
chmod +x train.sh
```

Update `Dockerfile` CMD:
```dockerfile
CMD ["./train.sh"]
```

## 🎨 Step 6: Test the Gradio Interface

Once deployed, your Space will show:

- **Interactive UI**: Ask questions in Amharic or English
- **Example Prompts**: Pre-loaded with religious questions
- **Advanced Settings**: Temperature, top-p, max length controls

Visit: `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`

## 📊 Step 7: Upload Trained Model (After Training)

After training completes locally or in another environment:

```bash
# Login if not already
huggingface-cli login

# Upload model checkpoint
huggingface-cli upload nexuss0781/Nexuss-AIv1 ./checkpoints/nexuss-aiv1 .

# Or use git
git lfs install
git clone https://huggingface.co/nexuss0781/Nexuss-AIv1
cp -r checkpoints/nexuss-aiv1/* Nexuss-AIv1/
cd Nexuss-AIv1
git add .
git commit -m "Upload trained blank slate model"
git push
```

## 🔍 Monitoring & Logs

- **Build Logs**: Click "Logs" tab in your Space
- **App Logs**: Real-time stdout/stderr in the interface
- **Metrics**: GPU usage, memory, requests in Settings → Metrics

## 🛠️ Troubleshooting

### Space stuck on "Building"
- Check Docker build logs
- Ensure all files are present (Dockerfile, app.py, requirements.txt)
- Try rebuilding: Settings → Factory reboot

### Out of Memory Error
- Reduce batch size in training
- Use smaller model: `--model_size small`
- Request better GPU in Space settings

### Model Not Loading
- Verify model path in `app.py`
- Check if model is uploaded to HF Hub
- Add error handling in `load_model()`

### Slow Inference
- Enable GPU in Space settings
- Use mixed precision (FP16)
- Reduce max_new_tokens

## 📈 Advanced: CI/CD Pipeline

Create `.github/workflows/deploy-space.yml`:

```yaml
name: Deploy to HF Spaces

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Push to HF Spaces
        run: |
          git config --global user.email "github-actions[bot]@users.noreply.github.com"
          git config --global user.name "github-actions[bot]"
          git remote add spaces https://${{ secrets.HF_TOKEN }}@huggingface.co/spaces/${{ secrets.HF_USERNAME }}/nexuss-ai-ethiopian
          git push spaces main --force
```

## 💡 Best Practices

1. **Start Small**: Use `small` model for testing, upgrade later
2. **Cache Models**: Use `HF_HOME` environment variable
3. **Use Secrets**: Never hardcode tokens in code
4. **Monitor Costs**: GPU Spaces consume credits
5. **Version Control**: Tag releases with model versions

## 📚 Additional Resources

- [Hugging Face Spaces Docs](https://huggingface.co/docs/hub/spaces)
- [Gradio Documentation](https://gradio.app/docs/)
- [Nexuss Transformer Guide](./TRAINING_GUIDE.md)
- [EthioBBPE Tokenizer](https://huggingface.co/Nexuss0781/Ethio-BBPE)

## 🎉 Success!

Your Ethiopian Orthodox Religious AI is now live on Hugging Face Spaces! Share it with the community and gather feedback.

---

**Need Help?** Open an issue at https://github.com/nexuss0781/Nexuss-Transformer
