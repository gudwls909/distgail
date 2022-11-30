### sac hopper config ###

agent = {
	'name': 'sac',
	'actor': 'continuous_policy',
	'critic': 'continuous_q_network',
	'use_dynamic_alpha': False,
	'gamma': 0.99,
	'tau': 0.005,
	'buffer_size': 100000,
	'batch_size': 256,
	'start_train_step': 25000,
	'static_log_alpha': -0.2,
	'lr_decay': True,
}

optim = {
	'actor': 'adam',
	'critic': 'adam',
	'alpha': 'adam',
	'actor_lr': 0.001,
	'critic_lr': 0.001,
	'alpha_lr': 0.0003,
}

env = {
	'render': False,
	'name': 'hopper',
}

train = {
	'training': True,
	'load_path': './logs/hopper/sac/20221026175045155077',
	'run_step': 500000,
	'print_period': 10000,
	'save_period': 50000,
	'eval_iteration': 10,
	'record': False,
	'record_period': 500000,
	'update_period': 128,
	'num_workers': 16,
}
