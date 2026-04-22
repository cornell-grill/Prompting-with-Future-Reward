import copy
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

        # Snapshot of the very first context in the episode. Used by reward
        # functions / success checks that need to compare current state to t=0
        # (e.g. "charger has lifted N cm from its starting position").
        self.initial_context = None

    def set_initial_context(self, context):
        """Store a deep-copied snapshot of the starting context.

        Call exactly once per episode, immediately after the first
        ``mesh_world.get_context()`` call. Subsequent calls overwrite the
        snapshot, which is usually a bug; callers should avoid it.
        """
        self.initial_context = copy.deepcopy(context)

    def attach_initial_state(self, context):
        """Expose the starting object states on a live context dict.

        After this call, reward code can read
        ``context["initial_state"]["<obj>"]["position"]`` to get t=0 values.
        No-op if ``set_initial_context`` has not been called yet, so early
        callers (before the snapshot exists) do not crash.

        Mutates and returns ``context`` for convenience.
        """
        if self.initial_context is not None:
            context["initial_state"] = self.initial_context["objects"]
        return context


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
