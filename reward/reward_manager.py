import os
import importlib


class RewardManager:
    def __init__(self, reward_name: str):
        self.reward_name = reward_name

        self.base_dir = os.path.join(
            os.path.dirname(__file__),
            reward_name
        )

        self.context = None
        self.context_loaded = False

        self.cache = {}
        self.current_stage = None
        self.rw = None


    def load_context(self):
        """
        Load context.py if it exists.
        Future: If not, generate it.
        """
        if self.context_loaded:
            return

        context_path = os.path.join(self.base_dir, "context.py")

        if not os.path.exists(context_path):
            print("[RewardManager] context.py not found.")
            return
        # TODO: Future implementation to replace the above
        # if not os.path.exists(context_path):
        #     generate_context(self.reward_name)
        #     importlib.invalidate_caches()

        #     if not os.path.exists(context_path):
        #         return

        module_path = f"reward.{self.reward_name}.context"
        self.context = importlib.import_module(module_path)

        self.context_loaded = True


    def update_stage(self, stage: int):
        """
        Load reward_{stage}.py dynamically if it exists.
        Future: If not, generate it.
        """
        if stage == self.current_stage:
            return

        reward_file_path = os.path.join(
            self.base_dir,
            f"reward_{stage}.py"
        )

        if not os.path.exists(reward_file_path):
            self.rw = None
            self.current_stage = None
            return
        # TODO: Future implementation to replace the above
        # if not os.path.exists(reward_file_path):
        #     generate_reward(stage)
        #     importlib.invalidate_caches()

        #     if not os.path.exists(reward_file_path):
        #         self.rw = None
        #         self.current_stage = None
        #         return

        if stage in self.cache:
            self.rw = self.cache[stage]
            self.current_stage = stage
            return

        module_path = f"reward.{self.reward_name}.reward_{stage}"
        module = importlib.import_module(module_path)

        self.cache[stage] = module
        self.rw = module
        self.current_stage = stage
