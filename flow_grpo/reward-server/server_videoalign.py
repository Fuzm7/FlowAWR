import logging
import os
import pickle
import random
import time
import traceback

import ray
from ray import serve
from starlette.requests import Request
from starlette.responses import Response

from reward_server.videoalign import load_videoalign

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ray.serve")

ray_address = os.environ.get("RAY_ADDRESS")
if not ray_address or ray_address == "auto":
    raise ValueError("Set RAY_ADDRESS to the explicit address of your dedicated reward cluster.")
ingress_replicas = int(os.environ.get("REWARD_INGRESS_REPLICAS", "96"))
if ingress_replicas < 1:
    raise ValueError("REWARD_INGRESS_REPLICAS must be a positive integer.")
ray.init(address=ray_address, ignore_reinit_error=True)
resources = ray.cluster_resources()
total_gpus = int(resources.get("GPU", 0))
if total_gpus < 1:
    raise RuntimeError("VideoAlign requires at least one GPU in the Ray cluster.")
if resources.get("CPU", 0) < total_gpus + ingress_replicas:
    raise RuntimeError("Insufficient Ray CPU resources; reduce REWARD_INGRESS_REPLICAS.")


@serve.deployment(num_replicas=total_gpus, ray_actor_options={"num_gpus": 1}, max_ongoing_requests=24)
class ScorerWorker:
    def __init__(self):
        port = random.randint(20000, 60000)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        self.inference_fn = load_videoalign()

    def calculate_score(self, video_paths, prompts):
        return self.inference_fn(video_paths, prompts)


@serve.deployment(num_replicas=ingress_replicas, ray_actor_options={"num_cpus": 1})
class VideoScorerService:
    def __init__(self, scorer_handle):
        self.scorer_handle = scorer_handle

    async def __call__(self, request: Request):
        try:
            raw_data = await request.body()
            # Pickle is retained for trusted internal clients only; never expose publicly.
            data = pickle.loads(raw_data)
            video_paths = data.get('path', [])
            prompts = data.get('prompts', [])
            result_scores = await self.scorer_handle.calculate_score.remote(video_paths, prompts)
            response_data = {"videoalign_score_all": result_scores}
            return Response(pickle.dumps(response_data), media_type="application/octet-stream")
        except Exception:
            tb = traceback.format_exc()
            logger.error(f"Service Dispatch Error: {tb}")
            return Response(tb.encode("utf-8"), status_code=500)


if __name__ == "__main__":
    serve.start(http_options={
        "host": os.environ.get("REWARD_HTTP_HOST", "127.0.0.1"),
        "port": int(os.environ.get("REWARD_HTTP_PORT", "8092")),
    })
    worker = ScorerWorker.options(num_replicas=total_gpus).bind()
    ingress = VideoScorerService.bind(worker)
    serve.run(ingress)
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        pass
