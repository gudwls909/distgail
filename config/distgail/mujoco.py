### DISTGAIL Mujoco Config ###

env = {
    # "name": it should be defined in the command. ex) python main.py --config config.AGENT.mujoco --env.name half_cheetah
    "render": False,
    "action_type": "continuous",
}

agent = {
    "name": "distgail",
    "actor": "continuous_policy",
    "critic": "continuous_q_network",
    "use_dynamic_alpha": True,
    "gamma": 0.99,
    "tau": 5e-3,
    "buffer_size": 100000,
    "batch_size": 512,
    "start_train_step": 100,
    "static_log_alpha": -0.2,
    "lr_decay": True,
    ############################
    "discrim": "continuous_discrim_network"
}

optim = {
    "actor": "adam",
    "critic": "adam",
    "alpha": "adam",
    #########
    "discrim": "adam",
    "discrim_lr": 6e-4,
    ######### #e
    "actor_lr": 1e-3,
    "critic_lr": 1e-3,
    "alpha_lr": 3e-4,
}

train = {
    "training": True,
    "load_path": None,
    #"load_path": "./logs/cartpole/distgail/20221020141001057514",
    "run_step": 500000,
    "print_period": 1000,
    "save_period": 50000,
    "eval_iteration": 10,
    "record": False,
    "record_period": 500000,
    # distributed setting
    "update_period": 128,
    "num_workers": 16,
}
