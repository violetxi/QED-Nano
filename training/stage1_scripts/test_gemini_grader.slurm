#!/bin/bash
#SBATCH --job-name=gemini_grader_smoke
#SBATCH --account=ingrai
#SBATCH --partition=hai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# envs
source /hai/scratch/ziyxiang/miniconda3/etc/profile.d/conda.sh
conda activate qed

##### Fix AMD/NVIDIA env conflict (*** ONLY ON HAI CLUSTER ***) #####
unset ROCR_VISIBLE_DEVICES
########################################################################

# project dir
cd /hai/scratch/ziyxiang/QED-Nano/training

: "${GEMINI_API_KEY:?Set GEMINI_API_KEY in ~/.bashrc}"

echo "=== Environment ==="
hostname
nvidia-smi -L | head -4
python --version
python -c "import pipelinerl; print('pipelinerl at', pipelinerl.__file__)"
python -c "from google import genai; import google.genai as g; print('google-genai', g.__version__)"

echo
echo "=== Provider resolution & config checks ==="
python -c "
from pipelinerl.domains.math.verifier_api import resolve_provider, _strip_google_prefix, _build_gemini_config
assert resolve_provider('google/gemini-3.1-flash-lite-preview', None) == 'gemini'
assert resolve_provider('openai/gpt-oss-20b', None) == 'openai'
assert resolve_provider('anything', 'gemini') == 'gemini'
assert _strip_google_prefix('google/gemini-3.1-flash-lite-preview') == 'gemini-3.1-flash-lite-preview'
cfg = _build_gemini_config({'temperature': 1.0, 'max_output_tokens': 16384, 'thinking_level': 'medium'})
assert cfg.temperature == 1.0 and cfg.max_output_tokens == 16384
assert cfg.thinking_config.include_thoughts is True
print('STATIC CHECKS OK')
"

echo
echo "=== Live Gemini grader call (3 iterations) ==="
python -m pipelinerl.domains.math.verifier_api \
  --model google/gemini-3.1-flash-lite-preview \
  --sampling-kwargs '{"temperature": 1.0, "max_output_tokens": 16384, "thinking_level": "medium"}' \
  --prompt-name v1 \
  --iterations 3
