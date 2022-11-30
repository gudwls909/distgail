### distgail hopper config ###

agent = {
	'name': 'distgail',
	'actor': 'continuous_policy',
	'critic': 'continuous_q_network',
	'use_dynamic_alpha': True,
	'gamma': 0.99,
	'tau': 0.005,
	'buffer_size': 100000,
	'batch_size': 512,
	'start_train_step': 100,
	'static_log_alpha': -0.2,
	'lr_decay': True,
	'discrim': 'continuous_discrim_network',
}

optim = {
	'actor': 'adam',
	'critic': 'adam',
	'alpha': 'adam',
	'discrim': 'adam',
	'discrim_lr': 0.0003,
	'actor_lr': 0.0003,
	'critic_lr': 0.0003,
	'alpha_lr': 0.0003,
}

env = {
	'render': False,
	'action_type': 'continuous',
	'name': 'hopper',
}

train = {
	'training': True,
	'load_path': None,
	'run_step': 500000,
	'print_period': 1000,
	'save_period': 50000,
	'eval_iteration': 10,
	'record': False,
	'record_period': 500000,
	'update_period': 128,
	'num_workers': 16,
}
