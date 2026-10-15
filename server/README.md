# Reward services

HPSv3 and VideoAlign run as separate Ray Serve HTTP services. Use a **dedicated GPU node for each service**, with a separate environment from FlowAWR training. The launchers start their own Ray Head and one model replica per GPU reported by that cluster.

> **Trusted internal use only.** These endpoints retain the existing pickle request/response protocol. Deserializing untrusted pickle can execute arbitrary code on the server **or the training client**. There is no authentication or TLS in these services. HTTP defaults to `127.0.0.1`; do not expose either HTTP or the Ray control plane to the public internet. A private bind address alone is not an access-control mechanism: restrict network access to trusted hosts and users. Obtain checkpoints only from trusted sources as well.

## 1. Install an independent environment

Run from the FlowAWR repository root on each reward node:

```bash
conda create -n reward_server python=3.10.16 -y
conda activate reward_server

python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r server/requirements-reward-server.txt
```

VideoAlign requires FlashAttention 2. HPSv3 has an SDPA fallback, but installing FlashAttention is required for the two-service setup:

```bash
python -m pip install packaging ninja wheel
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
```

This installation may compile CUDA extensions. Use a compatible NVIDIA driver, CUDA toolkit/compiler, and GPU. If installing a prebuilt FlashAttention wheel instead, match **Python 3.10, PyTorch 2.6, CUDA, architecture, and PyTorch's C++ ABI**; a wheel from another machine is not necessarily compatible. GPU memory requirements depend on the model and request size and are not established by the CPU-only release checks.

The launchers use `python` and `ray` from the environment you activated. They require Linux, Bash, `realpath`, and GNU `timeout`. They set the bundled HPSv3 and service source paths before starting Ray. Do not install a different `hpsv3` package as a replacement for this source subset.

**Do not run `pip install -e .` in this environment.** The root package pins training dependencies, including a different Transformers version. Training continues to use its own environment and the HTTP clients in [`flow_grpo/rewards.py`](../flow_grpo/rewards.py).

## 2. Prepare local models

Download the models separately and complete any upstream access/license requirements. No weights, model caches, or training data are included here. The service launchers require existing local paths; they do not download missing models.

### HPSv3

Prepare:

- [`MizzenAI/HPSv3`](https://huggingface.co/MizzenAI/HPSv3): `HPSv3.safetensors`.
- A complete local Qwen2-VL-7B-Instruct base model directory, including the model, tokenizer, and processor files.

```bash
export HPSV3_CHECKPOINT_PATH="/path/to/HPSv3.safetensors"
export HPSV3_BASE_MODEL_PATH="/path/to/Qwen2-VL-7B-Instruct"
```

The default configuration is [`HPSv3_7B.yaml`](../flow_grpo/HPSv3/hpsv3/config/HPSv3_7B.yaml). `HPSV3_BASE_MODEL_PATH` overrides only its base-model path in memory. Its reward head, special tokens, precision, and image-processing settings remain unchanged. `HPSV3_CONFIG_PATH` optionally selects another local YAML; changing its scoring settings can change rewards. Training dataset and DeepSpeed paths in the bundled YAML are not used by the inference loader.

### VideoAlign / VideoReward

Prepare:

- [`KwaiVGI/VideoReward`](https://huggingface.co/KwaiVGI/VideoReward): `model_config.json` and a complete `checkpoint-<step>` directory.
- A complete local [`Qwen/Qwen2-VL-2B-Instruct`](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) base model directory, including its tokenizer and processor.

```text
VideoReward/
├── model_config.json
└── checkpoint-11352/
    └── model.pth
```

The step above illustrates the directory format. The loader selects the **highest numbered checkpoint**. A split checkpoint may instead contain both `adapter_model.safetensors` and `non_lora_state_dict.pth`. Do not mix incomplete or unrelated checkpoints in this directory.

```bash
export VIDEOALIGN_CHECKPOINT_DIR="/path/to/VideoReward"
export VIDEOALIGN_BASE_MODEL_PATH="/path/to/Qwen2-VL-2B-Instruct"
```

`VIDEOALIGN_BASE_MODEL_PATH` overrides the base-model path stored inside `model_config.json` without editing that file. Retain the checkpoint's data, prompt, LoRA, and normalization settings. The service uses BF16 and `use_norm=True`; it returns VQ, MQ, TA, and Overall. The FlowAWR `videoalign_score` client selects **MQ**, not Overall.

**Video paths must be shared.** VideoAlign requests contain filesystem paths, not uploaded video bytes. The training process and every scorer must be able to read the same video at the same path. In contrast, HPSv3 requests contain JPEG bytes, so its server does not need access to the training node's video files.

## 3. Start Ray and the service

Use a fresh dedicated node without an existing Ray runtime. Each script checks its configuration and ports, starts a Ray Head, waits for `ray status` with a bounded timeout, and connects the service to that explicit address. It does not attach to or stop an existing cluster. The dashboard is disabled; Ray's other control/worker ports still require network isolation.

### HPSv3 node

After activating `reward_server` and setting the two HPSv3 model paths:

```bash
export CUDA_VISIBLE_DEVICES=0
export REWARD_INGRESS_REPLICAS=1
bash server/server_hpsv3.sh
```

The default Ray address is `127.0.0.1:6379`, and the default HTTP endpoint is `http://127.0.0.1:8080`.

### VideoAlign node

On a different GPU node, after activating `reward_server` and setting both VideoAlign model paths:

```bash
export CUDA_VISIBLE_DEVICES=0
export REWARD_INGRESS_REPLICAS=1
bash server/server_videoalign.sh
```

The default Ray address is `127.0.0.1:6380`, and the default HTTP endpoint is `http://127.0.0.1:8092`.

The examples explicitly use one ingress replica for a small deployment. Omitting that setting preserves the original defaults: **8 for HPSv3 and 96 for VideoAlign**. Each ingress needs 1 logical CPU; each GPU scorer also uses Ray Serve's default of 1 CPU. With `G` GPU replicas, provide at least `G + REWARD_INGRESS_REPLICAS` logical CPUs, plus capacity for the driver, HTTP proxy, Ray control plane, and operating system. The service rejects a cluster below the replica CPU minimum. Do not advertise fictitious CPU resources to bypass this check.

One full model is loaded on each GPU. Select GPUs with `CUDA_VISIBLE_DEVICES` **before starting the launcher**. Replica counts use the cluster's GPU total at service startup; later-added workers are not automatically included. Multi-node deployment and simultaneous services on one node are not validated by this release.

### Remote training clients

For clients on another trusted node, explicitly bind HTTP to the reward node's private interface **before** starting the launcher:

```bash
export REWARD_HTTP_HOST="<REWARD_NODE_PRIVATE_IP>"
```

Replace the placeholder with an actual interface address. Allow only the intended trusted clients through your network controls. `RAY_NODE_IP` is separate from HTTP binding and can remain at its single-node default; training clients do not need Ray access.

On the **training node**, configure the matching endpoints:

```bash
# Image HPSv3 and video HPSv3 use the same server protocol.
export REWARD_HPSV3_URL="http://<HPSV3_NODE_PRIVATE_IP>:8080"
export REWARD_VIDEO_HPSV3_URL="http://<HPSV3_NODE_PRIVATE_IP>:8080"
export REWARD_VIDEOALIGN_SCORE_URL="http://<VIDEOALIGN_NODE_PRIVATE_IP>:8092"
```

Use the root endpoint, not `/predict`. Video HPSv3 decodes frames on the client and averages the highest-scoring 30% after scoring. It may send all frames of a video in one request, so test the expected video sizes before training.

### Launcher settings

| Variable | HPSv3 default | VideoAlign default | Meaning |
| --- | --- | --- | --- |
| `RAY_NODE_IP` | `127.0.0.1` | `127.0.0.1` | Local Ray node address |
| `RAY_PORT` | `6379` | `6380` | Ray Head port |
| `REWARD_HTTP_HOST` | `127.0.0.1` | `127.0.0.1` | HTTP bind interface |
| `REWARD_HTTP_PORT` | `8080` | `8092` | HTTP port |
| `REWARD_INGRESS_REPLICAS` | `8` | `96` | Positive integer, 1 CPU per ingress |
| `RAY_START_TIMEOUT` | `60` | `60` | Ray readiness deadline in seconds; a final status probe may add up to 2 seconds |
| `CUDA_VISIBLE_DEVICES` | inherited | inherited | GPUs visible when starting the dedicated runtime |

The script sets `RAY_ADDRESS` from `RAY_NODE_IP:RAY_PORT`; it does not honor an inherited cluster address. Model environment variables must be set before starting Ray so its workers inherit them. Do not invoke the Python entrypoints merely to inspect them: they connect to Ray at import time.

Keep the launcher running. Press **Ctrl+C** to stop it; its exit handler signals only the service process and Ray Head process that it started, and waits for them. It never runs a global `ray stop --force`. With Ray's `--block` mode, the Head owns its child-process cleanup. Real-runtime shutdown behavior remains to be validated on the deployment machine. Service output appears in the terminal; Ray reports its session log directory during startup.

## 4. Verify with one trusted request

These examples perform real GPU inference when **you run them after deployment**. They are not installation-only checks. Do not send requests to an untrusted endpoint, and do not unpickle a response from an untrusted server.

HPSv3, from the client environment with Pillow and requests:

```bash
export REWARD_TEST_IMAGE="/path/to/test.jpg"
python - <<'PY'
import io
import os
import pickle
import requests
from PIL import Image

with Image.open(os.environ["REWARD_TEST_IMAGE"]) as image:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG")
response = requests.post(
    os.environ["REWARD_HPSV3_URL"],
    data=pickle.dumps({"images": [buf.getvalue()], "prompts": ["Describe this image."]}),
    timeout=120,
)
response.raise_for_status()
result = pickle.loads(response.content)
assert len(result["outputs"]) == 1
print(result)
PY
```

VideoAlign, using a real video readable at the same path by the server:

```bash
export REWARD_TEST_VIDEO="/shared/path/to/test.mp4"
python - <<'PY'
import os
import pickle
import requests

response = requests.post(
    os.environ["REWARD_VIDEOALIGN_SCORE_URL"],
    data=pickle.dumps({"path": [os.environ["REWARD_TEST_VIDEO"]], "prompts": ["Describe this video."]}),
    timeout=600,
)
response.raise_for_status()
result = pickle.loads(response.content)
assert len(result["videoalign_score_all"]) == 1
print(result["videoalign_score_all"][0])
PY
```

A successful `ray status` only confirms cluster connectivity. A GET request is not a model health check: these endpoints expect a pickle POST. Confirm successful model initialization and the expected output fields before launching training.

## Troubleshooting and verification scope

- **Port conflict:** use a dedicated node and unused ports; do not stop another user's Ray processes. Two different Head ports alone do not isolate all Ray ports on one machine.
- **Pending replicas / insufficient CPUs:** lower `REWARD_INGRESS_REPLICAS` explicitly and check `ray status --address=127.0.0.1:6379` (or `:6380`). Leave control-plane CPU capacity in addition to the replica minimum.
- **Model load failure:** check both the reward checkpoint and the base model/processor directory, not just one of them. Missing base files and checkpoint incompatibility are not repaired by changing paths.
- **Video read failure:** verify shared path visibility and codec support. VideoAlign uses its bundled video reader; `FORCE_QWENVL_VIDEO_READER=decord` or `torchvision` can select an installed backend before startup. Validate the resulting scoring pipeline on your data.
- **HTTP 500 / repeated client retries:** inspect service logs before retrying training. Existing training clients may retry 500 responses many times. The one-request checks above call `raise_for_status()` before parsing a response.
- **CUDA/FlashAttention failure:** check GPU/driver compatibility, installed PyTorch, compiler/ABI, and available memory. The release does not establish a per-model memory or throughput guarantee.

The release preparation checks syntax, deployment configuration, and mocked request/response behavior without starting Ray or loading models. A clean dependency installation, real GPU scoring, video decoding, multi-node scheduling, and throughput have **not** been validated by those checks.

## Third-party sources and licenses

The bundled source subsets preserve local inference adaptations rather than claiming equivalence to an unspecified upstream version:

- [HPSv3](https://github.com/MizzenAI/HPSv3), HPSv3 Team: [MIT license](../flow_grpo/HPSv3/LICENSE).
- [VideoAlign](https://github.com/KwaiVGI/VideoAlign), Kling Team, Kuaishou Technology: [MIT license](../flow_grpo/videoalign/LICENSE).
- Qwen/Hugging Face copyright and Apache-2.0 headers in derived files are retained; the repository includes the [Apache-2.0 license text](../LICENSE). Video preprocessing files retain their `qwen-vl-utils` source attribution.

Some modules named `train`, `trainer`, or `dataset` are imported to construct the inference model and are therefore included. This is not a standalone distribution of the upstream training projects; their datasets, launchers, and evaluation assets are excluded. Exact upstream revisions and model redistribution terms have not been independently audited. Obtain weights from their upstream publishers under their applicable terms.
