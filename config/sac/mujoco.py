### SAC Mujoco Config ###

env = {
    # "name": it should be defined in the command. ex) python main.py --config config.AGENT.mujoco --env.name half_cheetah
    "render": False,
    "action_type": "continuous",
}

agent = {
    "name": "sac",
    "actor": "continuous_policy",
    "critic": "continuous_q_network",
    #"use_dynamic_alpha": True,
    "use_dynamic_alpha": False,
    "gamma": 0.99,
    "tau": 5e-3,
    #"buffer_size": 50000,
    "buffer_size": 100000,
    "batch_size": 256,
    "start_train_step": 25000,
    #"static_log_alpha": -2.0,
    "static_log_alpha": -0.2,
    "lr_decay": True,
}

optim = {
    "actor": "adam",
    "critic": "adam",
    "alpha": "adam",
    #"actor_lr": 5e-4,
    "actor_lr": 1e-3,
    "critic_lr": 1e-3,
    "alpha_lr": 3e-4,
}

train = {
    "training": True,
    #"load_path": None,
    "load_path": "./logs/hopper/sac/20221028115747506941",
    "run_step": 1000000,
    #"run_step": 100000,
    "print_period": 10000,
    "save_period": 50000,
    "eval_iteration": 10,
    "record": False,
    "record_period": 500000,
    # distributed setting
    "update_period": 128,
    "num_workers": 16,
}
