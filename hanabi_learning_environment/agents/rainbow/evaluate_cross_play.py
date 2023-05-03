from absl import app
from absl import flags

from third_party.dopamine import logger

import run_experiment

from run_experiment import tf

FLAGS = flags.FLAGS

flags.DEFINE_multi_string(
    'gin_files', [],
    'List of paths to gin configuration files (e.g.'
    '"configs/hanabi_rainbow.gin").')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files '
    '(e.g. "DQNAgent.epsilon_train=0.1").')

flags.DEFINE_string('base_dir', None,
                    'Base directory to host all required sub-directories.')

flags.DEFINE_string('checkpoint_dir_agent_1', '',
                    'Directory where checkpoint files for agent 1 are stored.')
flags.DEFINE_string('checkpoint_dir_agent_2', '',
                    'Directory where checkpoint files for agent 2 are stored.')

flags.DEFINE_integer('num_games', 100,
                     'Number of games to play for evaluation.')


def evaluate_cross_play():
    """Evaluates the cross-play of two Rainbow agents."""
    if FLAGS.base_dir is None:
        raise ValueError('--base_dir is None: please provide a path for '
                         'logs and checkpoints.')
    run_experiment.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
    experiment_logger = logger.Logger('{}/logs'.format(FLAGS.base_dir))

    environment = run_experiment.create_environment()
    obs_stacker = run_experiment.create_obs_stacker(environment)

    
    print("loading agents.")
    try:
        with tf.compat.v1.Graph().as_default():
            agent_1 = run_experiment.load_agent_from_checkpoint(FLAGS.checkpoint_dir_agent_1, 'Rainbow', environment, obs_stacker)
            print("agent 1 load success")
    except:
        print("agent 1 load failure")

    try:
        with tf.compat.v1.Graph().as_default():
            agent_2 = run_experiment.load_agent_from_checkpoint(FLAGS.checkpoint_dir_agent_2, 'Rainbow', environment, obs_stacker)
            print("agent 2 load success")
    except:
        print("agent 2 load failure")

    # Set up the cross-play loop
    num_games = FLAGS.num_games
    scores = []

    for game in range(num_games):
        step_number, total_reward = run_experiment.run_one_cross_play_episode(agent_1, agent_2, environment, obs_stacker)
        scores.append(total_reward)

    # Calculate performance metrics
    average_score = sum(scores) / num_games
    print('Average score:', average_score)

def main(unused_argv):
    """This main function acts as a wrapper around the cross-play evaluation."""
    evaluate_cross_play()

if __name__ == '__main__':
    app.run(main)
