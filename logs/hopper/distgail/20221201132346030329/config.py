### distgail hopper config ###

agent = {
	'name': 'distgail',
	'network': 'continuous_policy_value',
	'gamma': 0.99,
	'buffer_size': 100000,
	'batch_size': 512,
	'n_step': 2048,
	'n_epoch': 10,
	'_lambda': 0.95,
	'epsilon_clip': 0.1,
	'vf_coef': 1.0,
	'ent_coef': 0.01,
	'clip_grad_norm': 1.0,
	'lr_decay': True,
	'discrim': 'continuous_discrim_network',
}

optim = {
	'name': 'adam',
	'discrim': 'adam',
	'discrim_lr': 0.0003,
	'lr': 0.0003,
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
	'record_period': 100000,
	'distributed_batch_size': 2048,
	'update_period': 2048,
	'num_workers': 32,
}
