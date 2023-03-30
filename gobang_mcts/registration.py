from envpool.registration import register

register(
    task_id="GobangSelfPlay",
    import_path="envpool.gobang_envpool",
    spec_cls="GobangEnvSpec",
    dm_cls="GobangDMEnvPool",
    gym_cls="GobangGymEnvPool",
    gymnasium_cls="GobangGymnasiumEnvPool",
    board_size=15,
    win_length=5,
)
