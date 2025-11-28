import torch
from verl import DataProto
from verl.workers.fsdp_workers import ActorRolloutRefWorker


class TandemActorRolloutWorker(ActorRolloutRefWorker):

    def __init__(self, config):
        super().__init__(config)
        self.tandem_rollout = None
        self.frozen_model = None

    def _init_tandem_rollout(self):
        if self.tandem_rollout is not None:
            return

        from tandem_rollout import TandemRollout

        tandem_config = self.config.rollout.get("tandem", {})
        prob_a = tandem_config.get("prob_a", 0.5)

        if self._is_rollout:
            self.tandem_rollout = TandemRollout(
                model_a=self.rollout.model,
                model_b=self.frozen_model,
                tokenizer=self.tokenizer,
                config=tandem_config
            )
            self.tandem_rollout.prob_a = prob_a

    def generate_sequences(self, prompts: DataProto):
        tandem_enabled = self.config.rollout.get("tandem", {}).get("enabled", False)

        if not tandem_enabled:
            return super().generate_sequences(prompts)

        self._init_tandem_rollout()

        timing_generate = {}
        if self._is_actor:
            loop = get_event_loop()
            loop.run_until_complete(self.rollout_mode())

        with Timer(name="generate", text="{name}: {milliseconds:.1f} ms", logger=None) as t:
            outputs = self.tandem_rollout.generate_sequences(prompts)

        timing_generate["generate"] = t.last

        outputs.meta_info.setdefault("timing", {})
        outputs.meta_info["timing"].update(timing_generate)

        return outputs
