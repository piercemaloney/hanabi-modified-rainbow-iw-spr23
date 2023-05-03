# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""RL environment for Hanabi, using an API similar to OpenAI Gym."""

from __future__ import absolute_import
from __future__ import division
import random, copy, functools, operator

from hanabi_learning_environment import pyhanabi
from hanabi_learning_environment.pyhanabi import color_char_to_idx

MOVE_TYPES = [_.name for _ in pyhanabi.HanabiMoveType]

#-------------------------------------------------------------------------------
# Environment API
#-------------------------------------------------------------------------------


class Environment(object):
  """Abstract Environment interface.

  All concrete implementations of an environment should derive from this
  interface and implement the method stubs.
  """

  def reset(self, config):
    """Reset the environment with a new config.

    Signals environment handlers to reset and restart the environment using
    a config dict.

    Args:
      config: dict, specifying the parameters of the environment to be
        generated.

    Returns:
      observation: A dict containing the full observation state.
    """
    raise NotImplementedError("Not implemented in Abstract Base class")

  def step(self, action):
    """Take one step in the game.

    Args:
      action: dict, mapping to an action taken by an agent.

    Returns:
      observation: dict, Containing full observation state.
      reward: float, Reward obtained from taking the action.
      done: bool, Whether the game is done.
      info: dict, Optional debugging information.

    Raises:
      AssertionError: When an illegal action is provided.
    """
    raise NotImplementedError("Not implemented in Abstract Base class")


class ColorMapper:
  def __init__(self, game_colors):
    self.game_colors = game_colors
    self.agent_colors = random.sample(game_colors, len(game_colors))
    self.mapping = dict(zip(game_colors, self.agent_colors))
    self.inverse_mapping = dict(zip(self.agent_colors, self.game_colors))

  def map_color(self, color):
    return self.mapping[color]

  def unmap_color(self, color):
    return self.inverse_mapping[color]


class HanabiEnv(Environment):
  """RL interface to a Hanabi environment.

  ```python

  environment = rl_env.make()
  config = { 'players': 5 }
  observation = environment.reset(config)
  while not done:
      # Agent takes action
      action =  ...
      # Environment take a step
      observation, reward, done, info = environment.step(action)
  ```
  """

  def __init__(self, config, color_shuffle=False):
    r"""Creates an environment with the given game configuration.

    Args:
      config: dict, With parameters for the game. Config takes the following
        keys and values.
          - colors: int, Number of colors \in [2,5].
          - ranks: int, Number of ranks \in [2,5].
          - players: int, Number of players \in [2,5].
          - hand_size: int, Hand size \in [4,5].
          - max_information_tokens: int, Number of information tokens (>=0).
          - max_life_tokens: int, Number of life tokens (>=1).
          - observation_type: int.
            0: Minimal observation.
            1: First-order common knowledge observation.
          - seed: int, Random seed.
          - random_start_player: bool, Random start player.
      color_shuffle: bool, Whether or not to randomly permute observed colors for OP algorithm
    """
    assert isinstance(config, dict), "Expected config to be of type dict."
    self.game = pyhanabi.HanabiGame(config)

    self.observation_encoder = pyhanabi.ObservationEncoder(
        self.game, pyhanabi.ObservationEncoderType.CANONICAL)
    
    self.players = self.game.num_players()

    # color mapping for each agent
    self.color_shuffle = color_shuffle
    if color_shuffle:
      game_colors = list(pyhanabi.COLOR_CHAR)
      self.color_mappers = [ColorMapper(game_colors) for _ in range(self.players)]

  def reset(self):
    r"""Resets the environment for a new game.

    Returns:
      observation: dict, containing the full observation about the game at the
        current step. *WARNING* This observation contains all the hands of the
        players and should not be passed to the agents.
        An example observation:
        {'current_player': 0,
         'player_observations': [{'current_player': 0,
                                  'current_player_offset': 0,
                                  'deck_size': 40,
                                  'discard_pile': [],
                                  'fireworks': {'B': 0,
                                                'G': 0,
                                                'R': 0,
                                                'W': 0,
                                                'Y': 0},
                                  'information_tokens': 8,
                                  'legal_moves': [{'action_type': 'PLAY',
                                                   'card_index': 0},
                                                  {'action_type': 'PLAY',
                                                   'card_index': 1},
                                                  {'action_type': 'PLAY',
                                                   'card_index': 2},
                                                  {'action_type': 'PLAY',
                                                   'card_index': 3},
                                                  {'action_type': 'PLAY',
                                                   'card_index': 4},
                                                  {'action_type':
                                                  'REVEAL_COLOR',
                                                   'color': 'R',
                                                   'target_offset': 1},
                                                  {'action_type':
                                                  'REVEAL_COLOR',
                                                   'color': 'G',
                                                   'target_offset': 1},
                                                  {'action_type':
                                                  'REVEAL_COLOR',
                                                   'color': 'B',
                                                   'target_offset': 1},
                                                  {'action_type': 'REVEAL_RANK',
                                                   'rank': 0,
                                                   'target_offset': 1},
                                                  {'action_type': 'REVEAL_RANK',
                                                   'rank': 1,
                                                   'target_offset': 1},
                                                  {'action_type': 'REVEAL_RANK',
                                                   'rank': 2,
                                                   'target_offset': 1}],
                                  'life_tokens': 3,
                                  'observed_hands': [[{'color': None, 'rank':
                                  -1},
                                                      {'color': None, 'rank':
                                                      -1},
                                                      {'color': None, 'rank':
                                                      -1},
                                                      {'color': None, 'rank':
                                                      -1},
                                                      {'color': None, 'rank':
                                                      -1}],
                                                     [{'color': 'G', 'rank': 2},
                                                      {'color': 'R', 'rank': 0},
                                                      {'color': 'R', 'rank': 1},
                                                      {'color': 'B', 'rank': 0},
                                                      {'color': 'R', 'rank':
                                                      1}]],
                                  'num_players': 2,
                                  'vectorized': [ 0, 0, 1, ... ]},
                                 {'current_player': 0,
                                  'current_player_offset': 1,
                                  'deck_size': 40,
                                  'discard_pile': [],
                                  'fireworks': {'B': 0,
                                                'G': 0,
                                                'R': 0,
                                                'W': 0,
                                                'Y': 0},
                                  'information_tokens': 8,
                                  'legal_moves': [],
                                  'life_tokens': 3,
                                  'observed_hands': [[{'color': None, 'rank':
                                  -1},
                                                      {'color': None, 'rank':
                                                      -1},
                                                      {'color': None, 'rank':
                                                      -1},
                                                      {'color': None, 'rank':
                                                      -1},
                                                      {'color': None, 'rank':
                                                      -1}],
                                                     [{'color': 'W', 'rank': 2},
                                                      {'color': 'Y', 'rank': 4},
                                                      {'color': 'Y', 'rank': 2},
                                                      {'color': 'G', 'rank': 0},
                                                      {'color': 'W', 'rank':
                                                      1}]],
                                  'num_players': 2,
                                  'vectorized': [ 0, 0, 1, ... ]}]}
    """
    self.state = self.game.new_initial_state()

    while self.state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
      self.state.deal_random_card()

    obs = self._make_observation_all_players(self.color_shuffle)
    obs["current_player"] = self.state.cur_player()
    return obs

  def vectorized_observation_shape(self):
    """Returns the shape of the vectorized observation.

    Returns:
      A list of integer dimensions describing the observation shape.
    """
    return self.observation_encoder.shape()

  def num_moves(self):
    """Returns the total number of moves in this game (legal or not).

    Returns:
      Integer, number of moves.
    """
    return self.game.max_moves()

  def step(self, action):
    """Take one step in the game.

    Args:
      action: dict, mapping to a legal action taken by an agent. The following
        actions are supported:
          - { 'action_type': 'PLAY', 'card_index': int }
          - { 'action_type': 'DISCARD', 'card_index': int }
          - {
              'action_type': 'REVEAL_COLOR',
              'color': str,
              'target_offset': int >=0
            }
          - {
              'action_type': 'REVEAL_RANK',
              'rank': str,
              'target_offset': int >=0
            }
        Alternatively, action may be an int in range [0, num_moves()).

    Returns:
      observation: dict, containing the full observation about the game at the
        current step. *WARNING* This observation contains all the hands of the
        players and should not be passed to the agents.
        An example observation:
        {'current_player': 0,
         'player_observations': [{'current_player': 0,
                            'current_player_offset': 0,
                            'deck_size': 40,
                            'discard_pile': [],
                            'fireworks': {'B': 0,
                                      'G': 0,
                                      'R': 0,
                                      'W': 0,
                                      'Y': 0},
                            'information_tokens': 8,
                            'legal_moves': [{'action_type': 'PLAY',
                                         'card_index': 0},
                                        {'action_type': 'PLAY',
                                         'card_index': 1},
                                        {'action_type': 'PLAY',
                                         'card_index': 2},
                                        {'action_type': 'PLAY',
                                         'card_index': 3},
                                        {'action_type': 'PLAY',
                                         'card_index': 4},
                                        {'action_type': 'REVEAL_COLOR',
                                         'color': 'R',
                                         'target_offset': 1},
                                        {'action_type': 'REVEAL_COLOR',
                                         'color': 'G',
                                         'target_offset': 1},
                                        {'action_type': 'REVEAL_COLOR',
                                         'color': 'B',
                                         'target_offset': 1},
                                        {'action_type': 'REVEAL_RANK',
                                         'rank': 0,
                                         'target_offset': 1},
                                        {'action_type': 'REVEAL_RANK',
                                         'rank': 1,
                                         'target_offset': 1},
                                        {'action_type': 'REVEAL_RANK',
                                         'rank': 2,
                                         'target_offset': 1}],
                            'life_tokens': 3,
                            'observed_hands': [[{'color': None, 'rank': -1},
                                            {'color': None, 'rank': -1},
                                            {'color': None, 'rank': -1},
                                            {'color': None, 'rank': -1},
                                            {'color': None, 'rank': -1}],
                                           [{'color': 'G', 'rank': 2},
                                            {'color': 'R', 'rank': 0},
                                            {'color': 'R', 'rank': 1},
                                            {'color': 'B', 'rank': 0},
                                            {'color': 'R', 'rank': 1}]],
                            'num_players': 2,
                            'vectorized': [ 0, 0, 1, ... ]},
                           {'current_player': 0,
                            'current_player_offset': 1,
                            'deck_size': 40,
                            'discard_pile': [],
                            'fireworks': {'B': 0,
                                      'G': 0,
                                      'R': 0,
                                      'W': 0,
                                      'Y': 0},
                            'information_tokens': 8,
                            'legal_moves': [],
                            'life_tokens': 3,
                            'observed_hands': [[{'color': None, 'rank': -1},
                                            {'color': None, 'rank': -1},
                                            {'color': None, 'rank': -1},
                                            {'color': None, 'rank': -1},
                                            {'color': None, 'rank': -1}],
                                           [{'color': 'W', 'rank': 2},
                                            {'color': 'Y', 'rank': 4},
                                            {'color': 'Y', 'rank': 2},
                                            {'color': 'G', 'rank': 0},
                                            {'color': 'W', 'rank': 1}]],
                            'num_players': 2,
                            'vectorized': [ 0, 0, 1, ... ]}]}
      reward: float, Reward obtained from taking the action.
      done: bool, Whether the game is done.
      info: dict, Optional debugging information.

    Raises:
      AssertionError: When an illegal action is provided.
    """
    if isinstance(action, dict):
      # Convert dict action HanabiMove
      action = self._build_move(action)
    elif isinstance(action, int):
      # Convert int action into a Hanabi move.
      action = self.game.get_move(action)
    else:
      raise ValueError("Expected action as dict or int, got: {}".format(
          action))

    last_score = self.state.score()
    # Apply the action to the state.
    self.state.apply_move(action)

    while self.state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
      self.state.deal_random_card()

    observation = self._make_observation_all_players(self.color_shuffle)
    done = self.state.is_terminal()
    # Reward is score differential. May be large and negative at game end.
    reward = self.state.score() - last_score
    info = {}

    return (observation, reward, done, info)

  def _make_observation_all_players(self, color_shuffle = False):
    """Make observation for all players.

    Returns:
      dict, containing observations for all players.
    """
    obs = {}
    player_observations = [self._extract_dict_from_backend(
        player_id, self.state.observation(player_id), color_shuffle)
        for player_id in range(self.players)]  # pylint: disable=bad-continuation
    obs["player_observations"] = player_observations
    obs["current_player"] = self.state.cur_player()
    return obs
  
  def _apply_color_mapping_to_observation(self, player_id, observation_dict):
    mapped_observation = copy.deepcopy(observation_dict)
    color_mapper = self.color_mappers[player_id]

    # Map fireworks
    mapped_fireworks = {
        color_mapper.map_color(color): count
        for color, count in mapped_observation["fireworks"].items()
    }
    mapped_observation["fireworks"] = mapped_fireworks

    # Map observed_hands
    for hand in mapped_observation["observed_hands"]:
        for card in hand:
            if card["color"] is not None:
                card["color"] = color_mapper.map_color(card["color"])

    # Map discard_pile
    for card in mapped_observation["discard_pile"]:
        card["color"] = color_mapper.map_color(card["color"])

    # Map card_knowledge
    for player_hints in mapped_observation["card_knowledge"]:
      for hint in player_hints:
          if hint["color"] is not None:
              hint["color"] = color_mapper.map_color(hint["color"])

          mapped_color_plausible = {
              color_mapper.map_color(c): plausible
              for c, plausible in zip(pyhanabi.COLOR_CHAR, hint["color_plausible"])
          }
          hint["color_plausible"] = [
              mapped_color_plausible[pyhanabi.color_idx_to_char(c_idx)]
              for c_idx in range(len(mapped_color_plausible))
          ]

    # Map last_move
    if mapped_observation["last_move"] is not None:
      last_move = mapped_observation["last_move"]
      if last_move["action_type"] in ['REVEAL_COLOR', 'PLAY', 'DISCARD']:
        if isinstance(last_move["color"], str):
          last_move["color"] = color_mapper.map_color(last_move["color"])

    # Map legal_moves
    mapped_observation["legal_moves_as_int"] = []
    for move in mapped_observation["legal_moves"]:
        if move["action_type"] == "REVEAL_COLOR":
            move["color"] = color_mapper.map_color(move["color"])
        
        # Calculate uid of the move and append it to mapped_observation["legal_moves_as_int"]
        move_uid = self.game.get_move_uid_from_dict(move)
        mapped_observation["legal_moves_as_int"].append(move_uid)

    return mapped_observation


  def _extract_dict_from_backend(self, player_id, observation, color_shuffle=False):
    """Extract a dict of features from an observation from the backend.

    Args:
      player_id: Int, player from whose perspective we generate the observation.
      observation: A `pyhanabi.HanabiObservation` object.

    Returns:
      obs_dict: dict, mapping from HanabiObservation to a dict.
    """

    obs_dict = {}
    obs_dict["current_player"] = self.state.cur_player()
    obs_dict["current_player_offset"] = observation.cur_player_offset()
    obs_dict["life_tokens"] = observation.life_tokens()
    obs_dict["information_tokens"] = observation.information_tokens()
    obs_dict["num_players"] = observation.num_players()
    obs_dict["deck_size"] = observation.deck_size()

    obs_dict["fireworks"] = {}
    fireworks = self.state.fireworks()
    for color, firework in zip(pyhanabi.COLOR_CHAR, fireworks):
      obs_dict["fireworks"][color] = firework

    obs_dict["legal_moves"] = []
    obs_dict["legal_moves_as_int"] = []
    for move in observation.legal_moves():
      obs_dict["legal_moves"].append(move.to_dict())
      obs_dict["legal_moves_as_int"].append(self.game.get_move_uid(move))

    obs_dict["observed_hands"] = []
    for player_hand in observation.observed_hands():
      cards = [card.to_dict() for card in player_hand]
      obs_dict["observed_hands"].append(cards)

    obs_dict["discard_pile"] = [
        card.to_dict() for card in observation.discard_pile()
    ]

    # # Return hints received. ORIGINAL CODE: LOST INFORMATION
    # obs_dict["card_knowledge"] = []
    # for player_hints in observation.card_knowledge():
    #   player_hints_as_dicts = []
    #   for hint in player_hints:
    #     hint_d = {}
    #     if hint.color() is not None:
    #       hint_d["color"] = pyhanabi.color_idx_to_char(hint.color())
    #     else:
    #       hint_d["color"] = None
    #     hint_d["rank"] = hint.rank()
    #     player_hints_as_dicts.append(hint_d)
    #   obs_dict["card_knowledge"].append(player_hints_as_dicts)

    obs_dict["card_knowledge"] = []
    for player_hints in observation.card_knowledge():
        player_hints_as_dicts = [hint.to_dict() for hint in player_hints]
        obs_dict["card_knowledge"].append(player_hints_as_dicts)

    
    def get_last_non_deal_move(past_moves):
      for item in past_moves:
          move_type = item.move().type()
          if move_type != pyhanabi.HanabiMoveType.DEAL:
              last_move_dict = item.move().to_dict()
              last_move_dict["player"] = item.player()
              last_move_dict["scored"] = item.scored()
              last_move_dict["information_token"] = item.information_token()
              last_move_dict["color"] = item.color()
              last_move_dict["rank"] = item.rank()
              return last_move_dict
      return None

    last_move = get_last_non_deal_move(observation.last_moves())
    obs_dict["last_move"] = last_move
    

    if color_shuffle:
      # overwrite the observation vector and legal moves
      mapped_observation = self._apply_color_mapping_to_observation(player_id, obs_dict)
      py_encoder = PyCanonicalObservationEncoder(self.game, self.vectorized_observation_shape())
      obs_dict["vectorized"] = py_encoder.encode_with_color_shuffle(mapped_observation)

      # it seems that we do not need this...
      # obs_dict["legal_moves_as_int"] = mapped_observation["legal_moves_as_int"] 
    else:
      # ipdb.set_trace()
      obs_dict["vectorized"] = self.observation_encoder.encode(observation)
    
    obs_dict["pyhanabi"] = observation

    return obs_dict

  def _build_move(self, action):
    """Build a move from an action dict.

    Args:
      action: dict, mapping to a legal action taken by an agent. The following
        actions are supported:
          - { 'action_type': 'PLAY', 'card_index': int }
          - { 'action_type': 'DISCARD', 'card_index': int }
          - {
              'action_type': 'REVEAL_COLOR',
              'color': str,
              'target_offset': int >=0
            }
          - {
              'action_type': 'REVEAL_RANK',
              'rank': str,
              'target_offset': int >=0
            }

    Returns:
      move: A `HanabiMove` object constructed from action.

    Raises:
      ValueError: Unknown action type.
    """
    assert isinstance(action, dict), "Expected dict, got: {}".format(action)
    assert "action_type" in action, ("Action should contain `action_type`. "
                                     "action: {}").format(action)
    action_type = action["action_type"]
    assert (action_type in MOVE_TYPES), (
        "action_type: {} should be one of: {}".format(action_type, MOVE_TYPES))

    if action_type == "PLAY":
      card_index = action["card_index"]
      move = pyhanabi.HanabiMove.get_play_move(card_index=card_index)
    elif action_type == "DISCARD":
      card_index = action["card_index"]
      move = pyhanabi.HanabiMove.get_discard_move(card_index=card_index)
    elif action_type == "REVEAL_RANK":
      target_offset = action["target_offset"]
      rank = action["rank"]
      move = pyhanabi.HanabiMove.get_reveal_rank_move(
          target_offset=target_offset, rank=rank)
    elif action_type == "REVEAL_COLOR":
      target_offset = action["target_offset"]
      assert isinstance(action["color"], str)
      color = color_char_to_idx(action["color"])
      move = pyhanabi.HanabiMove.get_reveal_color_move(
          target_offset=target_offset, color=color)
    else:
      raise ValueError("Unknown action_type: {}".format(action_type))

    legal_moves = self.state.legal_moves()
    assert (str(move) in map(
        str,
        legal_moves)), "Illegal action: {}. Move should be one of : {}".format(
            move, legal_moves)

    return move


def make(environment_name="Hanabi-Full", num_players=2, color_shuffle=False, pyhanabi_path=None):
  """Make an environment.

  Args:
    environment_name: str, Name of the environment to instantiate.
    num_players: int, Number of players in this game.
    pyhanabi_path: str, absolute path to header files for c code linkage.

  Returns:
    env: An `Environment` object.

  Raises:
    ValueError: Unknown environment name.
  """

  if pyhanabi_path is not None:
    prefixes=(pyhanabi_path,)
    assert pyhanabi.try_cdef(prefixes=prefixes), "cdef failed to load"
    assert pyhanabi.try_load(prefixes=prefixes), "library failed to load"

  if (environment_name == "Hanabi-Full" or
      environment_name == "Hanabi-Full-CardKnowledge"):
    return HanabiEnv(
        config={
            "colors":
                5,
            "ranks":
                5,
            "players":
                num_players,
            "max_information_tokens":
                8,
            "max_life_tokens":
                3,
            "observation_type":
                pyhanabi.AgentObservationType.CARD_KNOWLEDGE.value
        }, color_shuffle=color_shuffle)
  elif environment_name == "Hanabi-Full-Minimal":
    return HanabiEnv(
        config={
            "colors": 5,
            "ranks": 5,
            "players": num_players,
            "max_information_tokens": 8,
            "max_life_tokens": 3,
            "observation_type": pyhanabi.AgentObservationType.MINIMAL.value
        })
  elif environment_name == "Hanabi-Small":
    return HanabiEnv(
        config={
            "colors":
                2,
            "ranks":
                5,
            "players":
                num_players,
            "hand_size":
                2,
            "max_information_tokens":
                3,
            "max_life_tokens":
                1,
            "observation_type":
                pyhanabi.AgentObservationType.CARD_KNOWLEDGE.value
        })
  elif environment_name == "Hanabi-Very-Small":
    return HanabiEnv(
        config={
            "colors":
                1,
            "ranks":
                5,
            "players":
                num_players,
            "hand_size":
                2,
            "max_information_tokens":
                3,
            "max_life_tokens":
                1,
            "observation_type":
                pyhanabi.AgentObservationType.CARD_KNOWLEDGE.value
        })
  else:
    raise ValueError("Unknown environment {}".format(environment_name))


#-------------------------------------------------------------------------------
# Hanabi Agent API
#-------------------------------------------------------------------------------


class Agent(object):
  """Agent interface.

  All concrete implementations of an Agent should derive from this interface
  and implement the method stubs.


  ```python

  class MyAgent(Agent):
    ...

  agents = [MyAgent(config) for _ in range(players)]
  while not done:
    ...
    for agent_id, agent in enumerate(agents):
      action = agent.act(observation)
      if obs.current_player == agent_id:
        assert action is not None
      else
        assert action is None
    ...
  ```
  """

  def __init__(self, config, *args, **kwargs):
    r"""Initialize the agent.

    Args:
      config: dict, With parameters for the game. Config takes the following
        keys and values.
          - colors: int, Number of colors \in [2,5].
          - ranks: int, Number of ranks \in [2,5].
          - players: int, Number of players \in [2,5].
          - hand_size: int, Hand size \in [4,5].
          - max_information_tokens: int, Number of information tokens (>=0)
          - max_life_tokens: int, Number of life tokens (>=0)
          - seed: int, Random seed.
          - random_start_player: bool, Random start player.
      *args: Optional arguments
      **kwargs: Optional keyword arguments.

    Raises:
      AgentError: Custom exceptions.
    """
    raise NotImplementedError("Not implemeneted in abstract base class.")

  def reset(self, config):
    r"""Reset the agent with a new config.

    Signals agent to reset and restart using a config dict.

    Args:
      config: dict, With parameters for the game. Config takes the following
        keys and values.
          - colors: int, Number of colors \in [2,5].
          - ranks: int, Number of ranks \in [2,5].
          - players: int, Number of players \in [2,5].
          - hand_size: int, Hand size \in [4,5].
          - max_information_tokens: int, Number of information tokens (>=0)
          - max_life_tokens: int, Number of life tokens (>=0)
          - seed: int, Random seed.
          - random_start_player: bool, Random start player.
    """
    raise NotImplementedError("Not implemeneted in abstract base class.")

  def act(self, observation):
    """Act based on an observation.

    Args:
      observation: dict, containing observation from the view of this agent.
        An example:
        {'current_player': 0,
         'current_player_offset': 1,
         'deck_size': 40,
         'discard_pile': [],
         'fireworks': {'B': 0,
                   'G': 0,
                   'R': 0,
                   'W': 0,
                   'Y': 0},
         'information_tokens': 8,
         'legal_moves': [],
         'life_tokens': 3,
         'observed_hands': [[{'color': None, 'rank': -1},
                         {'color': None, 'rank': -1},
                         {'color': None, 'rank': -1},
                         {'color': None, 'rank': -1},
                         {'color': None, 'rank': -1}],
                        [{'color': 'W', 'rank': 2},
                         {'color': 'Y', 'rank': 4},
                         {'color': 'Y', 'rank': 2},
                         {'color': 'G', 'rank': 0},
                         {'color': 'W', 'rank': 1}]],
         'num_players': 2}]}

    Returns:
      action: dict, mapping to a legal action taken by this agent. The following
        actions are supported:
          - { 'action_type': 'PLAY', 'card_index': int }
          - { 'action_type': 'DISCARD', 'card_index': int }
          - {
              'action_type': 'REVEAL_COLOR',
              'color': str,
              'target_offset': int >=0
            }
          - {
              'action_type': 'REVEAL_RANK',
              'rank': str,
              'target_offset': int >=0
            }
    """
    raise NotImplementedError("Not implemented in Abstract Base class")


class PyCanonicalObservationEncoder:
    def __init__(self, parent_game, shape):
        self.parent_game = parent_game
        self.shape = shape # assuming ex: [1041]
        self.max_deck_size = 50 # TODO: not configurable
        self.num_colors = self.parent_game.num_colors()
        self.num_ranks = self.parent_game.num_ranks()
        self.num_players = self.parent_game.num_players()
        self.hand_size = self.parent_game.hand_size()
        self.max_information_tokens = self.parent_game.max_information_tokens()
        self.max_life_tokens = self.parent_game.max_life_tokens()


    def encode_with_color_shuffle(self, mapped_observation):
        encoding = [0] * self.flat_length()

        offset = 0
        offset += self.encode_hands(mapped_observation, offset, encoding)
        offset += self.encode_board(mapped_observation, offset, encoding)
        offset += self.encode_discards(mapped_observation, offset, encoding)
        offset += self.encode_last_action(mapped_observation, offset, encoding)
        # if self.parent_game.observation_type != 'Minimal':
        offset += self.encode_card_knowledge(mapped_observation, offset, encoding)

        assert offset == len(encoding)
        return encoding

    def flat_length(self):
        return self.shape[0]

    @staticmethod
    def card_index(color, rank, num_ranks):
        return color * num_ranks + rank

    def hands_section_length(self):
        return (self.num_players - 1) * self.hand_size * self.bits_per_card() + self.num_players

    def bits_per_card(self):
        return self.num_colors * self.num_ranks

    def encode_hands(self, mapped_observation, start_offset, encoding):
        bits_per_card = self.bits_per_card()

        offset = start_offset
        hands = mapped_observation["observed_hands"]
        for player in range(1, self.num_players):
            cards = hands[player]
            num_cards = 0

            for card in cards:
                encoding[offset + self.card_index(pyhanabi.color_char_to_idx(card['color']), card['rank'], self.num_ranks)] = 1
                num_cards += 1
                offset += bits_per_card

            if num_cards < self.hand_size:
                offset += (self.hand_size - num_cards) * bits_per_card

        for player in range(self.num_players):
            if len(hands[player]) < self.hand_size:
                encoding[offset + player] = 1
        offset += self.num_players

        assert offset - start_offset == self.hands_section_length()
        return offset - start_offset

    def board_section_length(self):
        return self.max_deck_size - self.num_players * self.hand_size + self.num_colors * self.num_ranks + self.max_information_tokens + self.max_life_tokens

    def encode_board(self, mapped_observation, start_offset, encoding):
        offset = start_offset
        deck_size = mapped_observation['deck_size']
        for i in range(deck_size):
            encoding[offset + i] = 1
        offset += (self.max_deck_size - self.hand_size * self.num_players)

        fireworks = mapped_observation['fireworks']
        for c in range(self.num_colors):
            if fireworks[pyhanabi.color_idx_to_char(c)] > 0:
                encoding[offset + fireworks[pyhanabi.color_idx_to_char(c)] - 1] = 1
            offset += self.num_ranks

        information_tokens = mapped_observation['information_tokens']
        for i in range(information_tokens):
            encoding[offset + i] = 1
        offset += self.max_information_tokens

        life_tokens = mapped_observation['life_tokens']
        for i in range(life_tokens):
            encoding[offset + i] = 1
        offset += self.max_life_tokens

        assert offset - start_offset == self.board_section_length()
        return offset - start_offset
    
    def discard_section_length(self):
      return self.max_deck_size

    def encode_discards(self, mapped_observation, start_offset, encoding):
      offset = start_offset
      discard_counts = [0] * (self.num_colors * self.num_ranks)
      for card in mapped_observation['discard_pile']:
          discard_counts[color_char_to_idx(card['color']) * self.num_ranks + card['rank']] += 1

      for c in range(self.num_colors):
          for r in range(self.num_ranks):
              num_discarded = discard_counts[c * self.num_ranks + r]
              for i in range(num_discarded):
                  encoding[offset + i] = 1
              offset += self.parent_game.num_cards(c, r)

      assert offset - start_offset == self.discard_section_length()
      return offset - start_offset
    
    def last_action_section_length(self):
      return (self.num_players + 4 + self.num_players + self.num_colors +
              self.num_ranks + self.hand_size + self.hand_size + self.bits_per_card() + 2)


    def encode_last_action(self, mapped_observation, start_offset, encoding):
      offset = start_offset
      last_move = mapped_observation["last_move"]
      if last_move is None:
          offset += self.last_action_section_length()
      else:
          last_move_type = last_move["action_type"]

          # player_id
          encoding[offset + last_move["player"]] = 1
          offset += self.num_players

          # move type
          if last_move_type == "PLAY":
              encoding[offset] = 1
          elif last_move_type == "DISCARD":
              encoding[offset + 1] = 1
          elif last_move_type == "REVEAL_COLOR":
              encoding[offset + 2] = 1
          elif last_move_type == "REVEAL_RANK":
              encoding[offset + 3] = 1
          else:
              raise ValueError("Invalid move type")
          offset += 4

          # target player (if hint action)
          if last_move_type in [pyhanabi.HanabiMoveType.REVEAL_COLOR, pyhanabi.HanabiMoveType.REVEAL_RANK]:
              observer_relative_target = (last_move["player"] + last_move["target_offset"]) % self.num_players
              encoding[offset + observer_relative_target] = 1
          offset += self.num_players

          # color (if hint action)
          if last_move_type == pyhanabi.HanabiMoveType.REVEAL_COLOR:
              encoding[offset + last_move["color"]] = 1
          offset += self.num_colors

          # rank (if hint action)
          if last_move_type == pyhanabi.HanabiMoveType.REVEAL_RANK:
              encoding[offset + last_move["rank"]] = 1
          offset += self.num_ranks

          # outcome (if hinted action)
          if last_move_type in [pyhanabi.HanabiMoveType.REVEAL_COLOR, pyhanabi.HanabiMoveType.REVEAL_RANK]:
              for i, mask in enumerate(last_move["card_indices"]):
                  encoding[offset + i] = mask
          offset += self.hand_size

          # position (if play or discard action)
          if last_move_type in [pyhanabi.HanabiMoveType.PLAY, pyhanabi.HanabiMoveType.DISCARD]:
              encoding[offset + last_move["card_index"]] = 1
          offset += self.hand_size

          # card (if play or discard action)
          if last_move_type in [pyhanabi.HanabiMoveType.PLAY, pyhanabi.HanabiMoveType.DISCARD]:
              encoding[offset + self.card_index(last_move["color"], last_move["rank"], self.num_ranks)] = 1
          offset += self.bits_per_card()

          # was successful and/or added information token (if play action)
          if last_move_type == pyhanabi.HanabiMoveType.PLAY:
              if last_move["scored"]:
                  encoding[offset] = 1
              if last_move["information_token"]:
                  encoding[offset + 1] = 1
          offset += 2

      assert offset - start_offset == self.last_action_section_length()
      return offset - start_offset
    
    def card_knowledge_section_length(self):
      return self.num_players * self.hand_size * (self.bits_per_card() + self.num_colors + self.num_ranks)

    def encode_card_knowledge(self, mapped_observation, start_offset, encoding):
        offset = start_offset
        hands = mapped_observation["card_knowledge"]
        assert len(hands) == self.num_players

        for player in range(self.num_players):
            knowledge = hands[player]
            num_cards = 0

            for card_knowledge in knowledge:
                # Add bits for plausible cards
                for color in range(self.num_colors):
                    if card_knowledge["color_plausible"][color]:
                        for rank in range(self.num_ranks):
                            if card_knowledge["rank_plausible"][rank]:
                                encoding[offset + self.card_index(color, rank, self.num_ranks)] = 1
                offset += self.bits_per_card()

                # Add bits for explicitly revealed colors and ranks
                if card_knowledge["color_hinted"]:
                    encoding[offset + pyhanabi.color_char_to_idx(card_knowledge["color"])] = 1
                offset += self.num_colors
                if card_knowledge["rank_hinted"]:
                    encoding[offset + card_knowledge["rank"]] = 1
                offset += self.num_ranks

                num_cards += 1

            # Adjust the offset to skip bits for the missing cards
            if num_cards < self.hand_size:
                offset += (self.hand_size - num_cards) * (self.bits_per_card() + self.num_colors + self.num_ranks)

        assert offset - start_offset == self.card_knowledge_section_length()
        return offset - start_offset
