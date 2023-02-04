import random

from gym.spaces import Discrete, Box
from gym import Env
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np


class TicTacToeMultiEnv(MultiAgentEnv):
    """Multi-agent environment for tic tac toe."""

    EMPTY_SYMBOL = 0
    X_SYMBOL = 1
    O_SYMBOL = 2
    SYMBOL_MAP = {
        X_SYMBOL: 'X',
        O_SYMBOL: 'O',
        EMPTY_SYMBOL: '.',
    }
    PLAYER_X_ID = 'player_' + SYMBOL_MAP[X_SYMBOL]
    PLAYER_O_ID = 'player_' + SYMBOL_MAP[O_SYMBOL]
    PLAYERS = [PLAYER_X_ID, PLAYER_O_ID]

    NUMBER_FIELDS = 9
    BOARD_WIDTH = 3

    # Extend space with 1 so a player can also perform no move at all.
    ACTION_SPACE_SIZE = NUMBER_FIELDS + 1
    WAIT_MOVE = ACTION_SPACE_SIZE - 1

    # Extend space with 2 so we can encode:
    # - Whether the user is X or O.
    # - Whether the next turn is X or O.
    OBSERVATION_SPACE_SIZE = NUMBER_FIELDS + 2
    USER_SYMBOL_INDEX = NUMBER_FIELDS + 1
    USER_TURN_INDEX = NUMBER_FIELDS

    ACTION_SPACE = Discrete(ACTION_SPACE_SIZE)
    OBSERVATION_SPACE = Box(0, 2, [OBSERVATION_SPACE_SIZE])

    LOSE_REWARD = -10
    WIN_REWARD = 10
    DRAW_REWARD = -1

    BAD_MOVE_REWARD = -1
    GOOD_MOVE_REWARD = 0

    def __init__(self, config):
        self.action_space = self.ACTION_SPACE
        self.observation_space = self.OBSERVATION_SPACE
        self.turn = random.choice([self.X_SYMBOL, self.O_SYMBOL])
        self.reset()
        self.history = []
        self.verbose = False

    def reset(self):
        self._init_board()
        self.history = []
        self._save_board()
        return self._obs()

    def set_verbose(self):
        self.verbose = True

    def step(self, action: dict):
        rew, invalid = self._action_rewards(action)

        if invalid:
            return self._obs(), rew, self._done(False), {}

        self._perform_actions(action)
        done = self._is_done()

        if done:
            for player_id in rew:
                rew[player_id] = self._evaluate_board(player_id)

        self._change_turn()
        self._save_board()
        self._print_if_verbose(done, rew)
        return self._obs(), rew, self._done(done), {}

    def _is_done(self):
        has_winner = self._get_winner() is not None
        board_full = self._board_is_full()
        return has_winner or board_full

    def _obs(self):
        return {
            self.X_SYMBOL: np.array(self.board + [self.turn, self.X_SYMBOL]),
            self.O_SYMBOL: np.array(self.board + [self.turn, self.O_SYMBOL])
        }

    def _done(self, done):
        return {self.X_SYMBOL: done, self.O_SYMBOL: done, '__all__': done}

    def _action_rewards(self, action):
        rew = {}
        invalid = False
        for player_id, player_action in action.items():
            if not self._valid_action(player_action, player_id):
                rew[player_id] = self.BAD_MOVE_REWARD
                invalid = True
            else:
                rew[player_id] = self.GOOD_MOVE_REWARD

        return rew, invalid

    def _valid_action(self, action, player_id):
        self._validate_action_space(action)

        # Waiting while it's the players turn.
        if self.turn == player_id and action == self.WAIT_MOVE:
            return False

        # Playing while it's not the players turn.
        if self.turn != player_id and action != self.WAIT_MOVE:
            return False

        # Trying to place on a filled field.
        if self._field_is_filled(action):
            return False

        return True

    def _perform_actions(self, actions):
        for player_id, action in actions.items():
            if action != self.WAIT_MOVE:
                self.board[action] = player_id

    def _print_if_verbose(self, done, rew):
        if self.verbose:
            self._print_board(self.board)
            print(f"\n\n-----|{done}|--|{rew}|-----")

    def _validate_action_space(self, action):
        if action > self.ACTION_SPACE_SIZE:
            raise ValueError(
                f"The action integer must be =< {self.ACTION_SPACE_SIZE}, got: {action}"
            )

    def _save_board(self):
        self.history.append(self.board.copy())

    def _get_winner(self):
        horizontal_groups = [self.board[0:3], self.board[3:6], self.board[6:9]]
        vertical_groups = [[self.board[0], self.board[3], self.board[6]],
                           [self.board[1], self.board[4], self.board[7]],
                           [self.board[2], self.board[5], self.board[8]]]
        diagonal_groups = [[self.board[0], self.board[4], self.board[8]],
                           [self.board[2], self.board[4], self.board[6]]]
        for group in (horizontal_groups + vertical_groups + diagonal_groups):
            if group.count(self.X_SYMBOL) == self.BOARD_WIDTH:
                return self.X_SYMBOL
            if group.count(self.O_SYMBOL) == self.BOARD_WIDTH:
                return self.O_SYMBOL

    def _evaluate_board(self, player_id):
        winner = self._get_winner()

        if winner == None:
            return self.DRAW_REWARD
        if player_id == winner:
            return self.WIN_REWARD
        return self.LOSE_REWARD

    def _print_board(self, board):
        for i, field in enumerate(board):
            if i % 3 == 0:
                print('')
            print(self.SYMBOL_MAP[field], end='')

    def _empty_fields(self):
        return [i for i in range(self.NUMBER_FIELDS) if self.board[i] is
                self.EMPTY_SYMBOL]

    def _field_is_filled(self, field_index):
        if field_index == self.WAIT_MOVE:
            return False
        return self.board[field_index] != self.EMPTY_SYMBOL

    def _board_is_full(self):
        return len(self._empty_fields()) == 0

    def _init_board(self):
        self.board = [self.EMPTY_SYMBOL for _ in range(self.NUMBER_FIELDS)]

    def _print_history(self):
        for i, board in enumerate(self.history):
            print(f"\n\n---ROUND-{i}---")
            self._print_board(board)
        print("\nSCORE: " + str(self._evaluate_board()))

    def _change_turn(self):
        if self.turn == self.X_SYMBOL:
            self.turn = self.O_SYMBOL
        else:
            self.turn = self.X_SYMBOL


import ray.rllib.agents
from ray.rllib.agents.registry import get_agent_class

ray.init()

trainer = 'PPO'
trained_policy = trainer + '_policy'

def policy_mapping_fn(agent_id):
    mapping = {TicTacToeMultiEnv.O_SYMBOL: trained_policy,
               TicTacToeMultiEnv.X_SYMBOL: "heuristic"}
    return mapping[agent_id]

config = {
    "env": TicTacToeMultiEnv,
    "multiagent": {
        "policies_to_train": [trained_policy],
        "policies": {
            trained_policy: (None, TicTacToeMultiEnv.OBSERVATION_SPACE,
                           TicTacToeMultiEnv.ACTION_SPACE, {}),
        },
        "policy_mapping_fn": policy_mapping_fn
    },
}

cls = get_agent_class(trainer) if isinstance(trainer, str) else trainer
trainer_obj = cls(config=config)
env = trainer_obj.workers.local_worker().env

while True:
    results = trainer_obj.train()
    results.pop('config')
    if results['episodes_total'] > 10000:
        break