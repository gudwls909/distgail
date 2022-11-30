### DISTGAIL CartPole Config ###

env = {
    "name": "cartpole",
    "action_type": "continuous",
    "render": False,
}

agent = {
    "name": "distgail",
    "actor": "continuous_policy",
    "critic": "continuous_q_network",
    "use_dynamic_alpha": True,
    "gamma": 0.99,
    "tau": 5e-3,
    "buffer_size": 50000,
    "batch_size": 64,
    "start_train_step": 5000,
    "static_log_alpha": -2.0,
    "target_update_period": 500,
    ############################
    "discrim": "continuous_discrim_network"
}

optim = {
    "actor": "adam",
    "critic": "adam",
    "alpha": "adam",
    #########
    "discrim": "adam",
    "discrim_lr": 5e-4,
    ######### #e
    "actor_lr": 1.5e-4,
    "critic_lr": 3e-4,
    "alpha_lr": 1e-5,
}

train = {
    "training": True,
    #"load_path": None,
    "load_path": "./logs/cartpole/distgail/20221020141001057514",
    "run_step": 100000,
    "print_period": 1000,
    "save_period": 10000,
    "eval_iteration": 10,
    # distributed setting
    "update_period": 32,
    "num_workers": 8,
}
