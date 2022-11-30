### sac cartpole config ###

agent = {
	'name': 'sac',
	'actor': 'continuous_policy',
	'critic': 'continuous_q_network',
	'use_dynamic_alpha': True,
	'gamma': 0.99,
	'tau': 0.005,
	'buffer_size': 50000,
	'batch_size': 64,
	'start_train_step': 5000,
	'static_log_alpha': -2.0,
	'target_update_period': 500,
	'lr_decay': True,
}

optim = {
	'actor': 'adam',
	'critic': 'adam',
	'alpha': 'adam',
	'actor_lr': 0.00015,
	'critic_lr': 0.0003,
	'alpha_lr': 1e-05,
}

env = {
	'name': 'cartpole',
	'action_type': 'continuous',
	'render': False,
}

train = {
	'training': True,
	'load_path': None,
	'run_step': 100000,
	'print_period': 1000,
	'save_period': 10000,
	'eval_iteration': 10,
	'update_period': 32,
	'num_workers': 8,
}
