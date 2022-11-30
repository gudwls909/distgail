### distgail cartpole config ###

agent = {
	'name': 'distgail',
	'actor': 'discrete_policy',
	'critic': 'discrete_q_network',
	'use_dynamic_alpha': True,
	'gamma': 0.99,
	'tau': 0.005,
	'buffer_size': 50000,
	'batch_size': 64,
	'start_train_step': 5000,
	'static_log_alpha': -2.0,
	'target_update_period': 500,
	'discrim': 'discrim_network',
}

optim = {
	'actor': 'adam',
	'critic': 'adam',
	'alpha': 'adam',
	'discrim': 'adam',
	'discrim_lr': 0.0005,
	'actor_lr': 0.00015,
	'critic_lr': 0.0003,
	'alpha_lr': 1e-05,
}

env = {
	'name': 'cartpole',
	'action_type': 'discrete',
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
