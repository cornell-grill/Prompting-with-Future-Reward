from reward.reward_helpers import *

KEEP_GRIPPER_CLOSED = False

subgoals = ['Grasp the charger plug', 'Pull the charger out of the power strip']


def determine_stage(context):
    gripper = context.get("gripper", {})
    is_grasping = gripper.get("is_grasping")[0]

    if not is_grasping:
        return 1
    return 2


def determine_success(context):
    """Task is successful once the charger has been lifted from its starting position.

    Requires ``context["initial_state"]`` to be populated by
    :meth:`RewardManager.attach_initial_state`. The main/CEM loops call that
    on every fresh context.

    If the snapshot is missing (e.g. a caller forgot to attach it), we
    fail-safe by returning ``False`` and warning once per process, so the
    episode simply runs to its step cap instead of crashing.
    """
    initial_state = context.get("initial_state")
    if initial_state is None:
        if not getattr(determine_success, "_warned_missing_initial", False):
            print(
                "[charger_human.determine_success] context['initial_state'] is None; "
                "RewardManager.attach_initial_state was not called on this context. "
                "Returning False (episode cannot be marked successful)."
            )
            determine_success._warned_missing_initial = True
        return False

    charger = context.get("objects").get("charger")
    charger_pos = charger.get("position")[0]
    initial_pos = initial_state.get("charger").get("position")[0]

    z_displacement = charger_pos[2] - initial_pos[2]

    return z_displacement > 0.015