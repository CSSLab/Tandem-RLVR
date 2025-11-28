import subprocess
import os

num_gpus_output = subprocess.check_output(['nvidia-smi', '-L']).decode().strip()
num_gpus = len(num_gpus_output.split('\n'))
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(num_gpus))

import torch
import ray
from transformers import AutoModelForCausalLM


@ray.remote(num_gpus=0)
class FrozenModelActor:
    def __init__(self, model_path, device_id=1):
        print(f"[FrozenModelActor] CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
        print(f"[FrozenModelActor] torch.cuda.device_count() = {torch.cuda.device_count()}")

        self.device = torch.device(f"cuda:{device_id}")
        print(f"[FrozenModelActor] Loading model on {self.device}")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": self.device},
            trust_remote_code=True,
        )
        self.model.eval()

        print(f"[FrozenModelActor] Model loaded on {next(self.model.parameters()).device}")

        self.cache = {}

    @torch.no_grad()
    def forward(self, batch_id, input_ids, attention_mask):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        past_key_values = self.cache.get(batch_id, None)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

        self.cache[batch_id] = outputs.past_key_values

        return {'logits': outputs.logits.cpu()}

    def clear_cache(self, batch_id):
        if batch_id in self.cache:
            del self.cache[batch_id]

    def get_device_info(self):
        import os
        return {
            'device': str(self.device),
            'model_device': str(next(self.model.parameters()).device),
            'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', 'not set'),
            'device_count': torch.cuda.device_count(),
        }
