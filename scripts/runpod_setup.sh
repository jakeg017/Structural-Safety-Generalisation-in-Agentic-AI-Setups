#!/bin/bash
# runpod_setup.sh
# Full setup for a fresh RunPod pod (PyTorch 2.4.0 template, RTX 4090).
#
# =============================================================================
# FIRST TIME ON A FRESH POD (this file won't exist yet):
# Copy and paste the following block into the pod terminal to create and run
# this script in one step:
#
#   cat > /workspace/setup.sh << 'EOF'
#   #!/bin/bash
#   set -e
#   apt-get update -qq && apt-get install -y -qq zstd git curl
#   curl -fsSL https://ollama.com/install.sh | sh
#   OLLAMA_HOST=0.0.0.0:11434 ollama serve &
#   until curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do sleep 1; done
#   ollama pull qwen3:8b
#   echo "Done. Ollama serving at 0.0.0.0:11434"
#   EOF
#   chmod +x /workspace/setup.sh
#   bash /workspace/setup.sh
#
# BEFORE RUNNING: make sure port 11434 is exposed as an HTTP port in the
# RunPod dashboard when creating the pod.
#
# SUBSEQUENT RESTARTS (script already exists on pod):
#   bash /workspace/setup.sh
# =============================================================================

set -e

echo ""
echo "=============================="
echo " RunPod Setup: Agentic Safety Eval"
echo "=============================="
echo ""

# --- System dependencies ---
echo "==> Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq zstd git curl

# --- Install Ollama ---
echo "==> Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

# --- Start Ollama bound to all interfaces ---
echo "==> Starting Ollama on 0.0.0.0:11434..."
OLLAMA_HOST=0.0.0.0:11434 ollama serve &

# Wait for Ollama to be ready
echo "==> Waiting for Ollama to start..."
until curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
    sleep 1
done

# --- Pull model ---
echo "==> Pulling qwen3:8b (fast if already cached)..."
ollama pull qwen3:8b

# --- Done ---
echo ""
echo "=============================="
echo " Setup complete!"
echo ""
echo " Ollama:  http://0.0.0.0:11434"
echo " Model:   qwen3:8b"
echo " Run experiments from your local machine:"
echo "   MODEL_API_KEY=EMPTY python -m harness.run_experiment \\"
echo "     --model qwen3:8b \\"
echo "     --base-url https://<pod-id>-11434.proxy.runpod.net/v1 \\"
echo "     --goals 10 --runs 20"
echo "=============================="
echo ""
