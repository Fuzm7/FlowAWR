import logging
import os
import pickle
import time
import traceback
from io import BytesIO

from PIL import Image
import ray
from ray import serve
from starlette.requests import Request
from starlette.responses import Response

from reward_server.hpsv3 import load_hpsv3score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ray.serve")

ray_address = os.environ.get("RAY_ADDRESS")
if not ray_address or ray_address == "auto":
    raise ValueError("Set RAY_ADDRESS to the explicit address of your dedicated reward cluster.")
ingress_replicas = int(os.environ.get("REWARD_INGRESS_REPLICAS", "8"))
if ingress_replicas < 1:
    raise ValueError("REWARD_INGRESS_REPLICAS must be a positive integer.")
ray.init(address=ray_address, ignore_reinit_error=True)
resources = ray.cluster_resources()
total_gpus = int(resources.get("GPU", 0))
if total_gpus < 1:
    raise RuntimeError("HPSv3 requires at least one GPU in the Ray cluster.")
if resources.get("CPU", 0) < total_gpus + ingress_replicas:
    raise RuntimeError("Insufficient Ray CPU resources; reduce REWARD_INGRESS_REPLICAS.")


@serve.deployment(num_replicas=total_gpus, ray_actor_options={"num_gpus": 1}, max_ongoing_requests=1)
class ScorerWorker:
    def __init__(self):
        logger.info("Initializing HPSv3 Scorer on GPU...")
        try:
            self.hps_scorer = load_hpsv3score()
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise e

    def calculate_score(self, images, prompts):
        images = [Image.open(BytesIO(d), formats=["jpeg"]) for d in images]
        h_scores = self.hps_scorer(images, prompts)
        return h_scores


@serve.deployment(num_replicas=ingress_replicas, ray_actor_options={"num_cpus": 1})
class VideoScorerService:
    def __init__(self, scorer_handle):
        self.scorer_handle = scorer_handle

    async def __call__(self, request: Request):
        try:
            raw_data = await request.body()
            # Pickle is retained for trusted internal clients only; never expose publicly.
            data = pickle.loads(raw_data)
            images = data.get('images', [])
            prompts = data.get('prompts', [])
            if len(images) != len(prompts):
                return Response("Length mismatch", status_code=400)
            result_scores = await self.scorer_handle.calculate_score.remote(images, prompts)
            response_data = {"outputs": result_scores}
            return Response(pickle.dumps(response_data), media_type="application/octet-stream")
        except Exception:
            tb = traceback.format_exc()
            logger.error(f"Service Dispatch Error: {tb}")
            return Response(tb.encode("utf-8"), status_code=500)


if __name__ == "__main__":
    serve.start(http_options={
        "host": os.environ.get("REWARD_HTTP_HOST", "127.0.0.1"),
        "port": int(os.environ.get("REWARD_HTTP_PORT", "8080")),
    })
    worker = ScorerWorker.options(num_replicas=total_gpus).bind()
    ingress = VideoScorerService.bind(worker)
    serve.run(ingress)
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        pass
