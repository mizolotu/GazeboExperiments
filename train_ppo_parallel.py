import os, sys, gym, gym_gazebo2, multiprocessing, shutil
import tensorflow as tf

from importlib import import_module
from baselines import bench, logger
from baselines.common.mara_wrappers import FrameStack
from baselines.common.cmd_util import make_vec_env

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

def make_cnn_env():
    env = gym.make(alg_kwargs['env_name'])
    env.set_episode_size(alg_kwargs['nsteps'])
    env = FrameStack(env, alg_kwargs['num_stack'])
    env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir()), allow_early_resets=True)
    return env

def log_params(logger, alg_kwargs):
    with open(logger.get_dir() + "/parameters.txt", 'w') as out:
        for key in alg_kwargs.keys():
            out.write('{0} = {1}\n'.format(key, alg_kwargs[key]))

def clean_logs(log_dir):
    for the_file in os.listdir(log_dir):
        file_path = os.path.join(log_dir, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)

if __name__ == '__main__':
    policy = sys.argv[1].split('_')[0]
    hidden = sys.argv[1].split('_')[1]
    env_type = 'mara_{0}'.format(policy)
    alg_kwargs = get_learn_function_defaults('ppo2', env_type)
    # timedate = datetime.now().strftime('%Y-%m-%d_%Hh%Mmin')
    logdir = 'logs/{0}/ppo2_{1}_{2}/'.format(alg_kwargs['env_name'], policy, hidden)
    tb_dir = logdir + '/tb'
    if os.path.isdir(logdir): clean_logs(tb_dir)
    format_strs = os.getenv('MARA_LOG_FORMAT', 'stdout,log,csv,tensorboard').split(',')
    logger.configure(os.path. abspath(logdir), format_strs)
    log_params(logger, alg_kwargs)

    env = make_vec_env(alg_kwargs['env_name'], env_type, 4, 0, reward_scale=1.0)

    learn = get_learn_function('ppo2')
    if alg_kwargs['transfer_path'] is not None and os.path.isfile(logdir  + alg_kwargs['transfer_path']):
        transfer_path = logdir + alg_kwargs['transfer_path']
    else:
        transfer_path = None
    if hidden != '':
        if 'num_hidden' in alg_kwargs.keys():
            alg_kwargs['num_hidden'] = int(hidden)
        elif 'nlstm' in alg_kwargs.keys():
            alg_kwargs['nlstm'] = int(hidden)
    alg_kwargs.pop('env_name')
    alg_kwargs.pop('transfer_path')
    learner = learn(env=env,load_path=transfer_path, **alg_kwargs)
    env.dummy().gg2().close()
    os.kill(os.getpid(), 9)
