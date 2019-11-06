import os, sys, time, gym, gym_gazebo2, multiprocessing
import tensorflow as tf

from datetime import datetime
from importlib import import_module
from baselines import bench, logger
from baselines.ppo2 import ppo2
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

ncpu = multiprocessing.cpu_count()
config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=ncpu, inter_op_parallelism_threads=ncpu, log_device_placement=False)
config.gpu_options.allow_growth = True
tf.Session(config=config).__enter__()

def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        alg_module = import_module('.'.join(['rl_algs', alg, submodule]))
    return alg_module

def get_learn_function(alg, submodule=None):
    return get_alg_module(alg, submodule).learn

def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs

def make_env():
    env = gym.make(alg_kwargs['env_name'])
    env.set_episode_size(alg_kwargs['nsteps'])
    env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir()), allow_early_resets=True)
    return env

def log_params(logger, alg_kwargs):
    with open(logger.get_dir() + "/parameters.txt", 'w') as out:
        out.write(
            'num_layers = ' + str(alg_kwargs['num_layers']) + '\n'
            + 'num_hidden = ' + str(alg_kwargs['num_hidden']) + '\n'
            + 'layer_norm = ' + str(alg_kwargs['layer_norm']) + '\n'
            + 'nsteps = ' + str(alg_kwargs['nsteps']) + '\n'
            + 'nminibatches = ' + str(alg_kwargs['nminibatches']) + '\n'
            + 'lam = ' + str(alg_kwargs['lam']) + '\n'
            + 'gamma = ' + str(alg_kwargs['gamma']) + '\n'
            + 'noptepochs = ' + str(alg_kwargs['noptepochs']) + '\n'
            + 'log_interval = ' + str(alg_kwargs['log_interval']) + '\n'
            + 'ent_coef = ' + str(alg_kwargs['ent_coef']) + '\n'
            + 'cliprange = ' + str(alg_kwargs['cliprange']) + '\n'
            + 'vf_coef = ' + str(alg_kwargs['vf_coef']) + '\n'
            + 'max_grad_norm = ' + str(alg_kwargs['max_grad_norm']) + '\n'
            + 'seed = ' + str(alg_kwargs['seed']) + '\n'
            + 'value_network = ' + alg_kwargs['value_network'] + '\n'
            + 'network = ' + alg_kwargs['network'] + '\n'
            + 'total_timesteps = ' + str(alg_kwargs['total_timesteps']) + '\n'
            + 'save_interval = ' + str(alg_kwargs['save_interval']) + '\n'
            + 'env_name = ' + alg_kwargs['env_name'] + '\n'
            + 'transfer_path = ' + str(alg_kwargs['transfer_path'])
        )

if __name__ == '__main__':

    policy = sys.argv[1]
    env_type = 'mara_{0}'.format(policy)
    alg_kwargs = get_learn_function_defaults('ppo2', env_type)
    timedate = datetime.now().strftime('%Y-%m-%d_%Hh%Mmin')
    logdir = 'logs/' + alg_kwargs['env_name'] + '/ppo2_mlp/' + timedate
    format_strs = os.getenv('MARA_LOG_FORMAT', 'stdout,log,csv,tensorboard').split(',')
    logger.configure(os.path. abspath(logdir), format_strs)
    log_params(logger, alg_kwargs)
    env = DummyVecEnv([make_env])

    learn = get_learn_function('ppo2')
    transfer_path = alg_kwargs['transfer_path']
    alg_kwargs.pop('env_name')
    alg_kwargs.pop('trained_path')
    alg_kwargs.pop('transfer_path')
    print(alg_kwargs)
    print(learn)
    if transfer_path is not None:
        _ = learn(env=env,load_path=transfer_path, **alg_kwargs)
    else:
        _ = learn(env=env, **alg_kwargs)

    env.dummy().gg2().close()
    os.kill(os.getpid(), 9)