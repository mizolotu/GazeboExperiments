def mujoco():
    return dict(
        nsteps=2048,
        nminibatches=32,
        lam=0.95,
        gamma=0.99,
        noptepochs=10,
        log_interval=1,
        ent_coef=0.0,
        lr=lambda f: 3e-4 * f,
        cliprange=0.2,
        value_network='copy'
    )

def atari():
    return dict(
        nsteps=128, nminibatches=4,
        lam=0.95, gamma=0.99, noptepochs=4, log_interval=1,
        ent_coef=.01,
        lr=lambda f : f * 2.5e-4,
        cliprange=0.1,
    )
def retro():
   return atari()

def mara_mlp():
    return dict(
        num_layers = 2,
        num_hidden = 64,
        layer_norm = False,
        nsteps = 125,
        nminibatches = 4, #batchsize = nevn * nsteps // nminibatches
        lam = 0.95,
        gamma = 0.99,
        noptepochs = 4,
        log_interval = 10,
        ent_coef = 0.0,
        lr = 1e-3,
        cliprange = 0.2,
        vf_coef = 0.5,
        max_grad_norm = 0.5,
        seed = 0,
        value_network = 'shared',
        network = 'mlp',
        total_timesteps = 4 * 125 * 1000,
        save_interval = 10,
        #env_name = 'MARARandomTarget-v0',
        env_name='MARAOrient-v0',
        transfer_path = 'checkpoints/best',
    )

def mara_lstm():
    return dict(
        nlstm = 64,
        layer_norm = False,
        nsteps = 1024,
        nminibatches = 1, #batchsize = nevn * nsteps // nminibatches
        lam = 0.95,
        gamma = 0.99,
        noptepochs = 10,
        log_interval = 1,
        ent_coef = 0.0,
        lr = lambda f: 1e-3 * f,
        cliprange = 0.2,
        vf_coef = 0.5,
        max_grad_norm = 0.5,
        seed = 0,
        value_network = 'shared',
        network = 'lstm',
        total_timesteps = 8640000 * 5,
        save_interval = 10,
        env_name = 'MARARandomTarget-v0',
        transfer_path = 'checkpoints/best',
    )

def mara_cnn():
    return dict(
        num_stack = 4,
        num_hidden = 32,
        num_dense = 64,
        rf=4,
        stride=1,
        layer_norm = False,
        nsteps = 1024,
        nminibatches = 1, #batchsize = nevn * nsteps // nminibatches
        lam = 0.95,
        gamma = 0.99,
        noptepochs = 10,
        log_interval = 1,
        ent_coef = 0.0,
        lr = lambda f: 1e-3 * f,
        cliprange = 0.2,
        vf_coef = 0.5,
        max_grad_norm = 0.5,
        seed = 0,
        value_network = 'shared',
        network = 'cnn1d',
        total_timesteps = 8640000 * 5,
        save_interval = 10,
        env_name = 'MARARandomTarget-v0',
        transfer_path = None,
    )
