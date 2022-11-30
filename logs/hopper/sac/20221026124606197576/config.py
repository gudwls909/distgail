### sac hopper config ###

agent = {
	'name': 'sac',
	'actor': 'continuous_policy',
	'critic': 'continuous_q_network',
	'use_dynamic_alpha': True,
	'gamma': 0.99,
	'tau': 0.005,
	'buffer_size': 50000,
	'batch_size': 256,
	'start_train_step': 25000,
	'static_log_alpha': -2.0,
	'lr_decay': True,
}

optim = {
	'actor': 'adam',
	'critic': 'adam',
	'alpha': 'adam',
	'actor_lr': 0.0005,
	'critic_lr': 0.001,
	'alpha_lr': 0.0003,
}

env = {
	'render': False,
	'name': 'hopper',
}

train = {
	'training': True,
	'load_path': None,
	'run_step': 1000000,
	'print_period': 10000,
	'save_period': 100000,
	'eval_iteration': 10,
	'record': False,
	'record_period': 500000,
	'update_period': 128,
	'num_workers': 16,
}
