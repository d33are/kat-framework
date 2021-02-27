#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##################################################
#  _   __      _   _               _             #
# | | / /     | | | |             (_)            #
# | |/ /  __ _| |_| |__   ___ _ __ _ _ __   ___  #
# |    \ / _` | __| '_ \ / _ \ '__| | '_ \ / _ \ #
# | |\  \ (_| | |_| | | |  __/ |  | | | | |  __/ #
# \_| \_/\__,_|\__|_| |_|\___|_|  |_|_| |_|\___| #
#                                                #
# General Video Game AI                          #
# Copyright (C) 2020-2021 d33are                 #
##################################################

from kat_api import ConfigurationProperty
from kat_framework.monitor.properties import KatMetrics
from vizdoom.vizdoom import Button, GameVariable, AutomapMode, Mode, ScreenFormat, ScreenResolution
import datetime

DATE_FORMAT = "%Y%m%d-%H%M%S"


class KatConfigurationProperty(ConfigurationProperty):
    """
    Global Application properties.
    """
    # generated run tag per process: run_%Y%m%d-%H%M%S
    RUN_TAG = ("run_tag", str, "run_{}".format(datetime.datetime.now().strftime(DATE_FORMAT)))
    # game implementation class (canonical name)
    GAME_CLASS = ("game_class", str, "games.default.DefaultGame")
    # driver class implementation (canonical name)
    DRIVER_CLASS = ("driver_class", str, "core.default.DefaultDriver")
    # agent class implementation (canonical name)
    AGENT_CLASS = ("agent_class", str, "agents.default.DefaultAgent")
    # network class implementation (canonical name)
    NETWORK_CLASS = ("network_class", str, "networks.tensorflow.models.random.RandomPreprocessNetwork")
    # memory class implementation (canonical name)
    MEMORY_CLASS = ("memory_class", str, "core.memory.ArrayMemory")
    # metric tracer implementation (canonical name)
    METRICS_TRACER_CLASS = ("metrics_tracer_class", str, "networks.tensorflow.metrics.TensorboardTracer")
    # model serializer implementation (canonical name)
    MODEL_SERIALIZER_CLASS = ("model_serializer_class", str, "core.model.KatModelSerializer")
    # storage driver implementation (canonical name)
    MODEL_STORAGE_DRIVER_CLASS = ("model_storage_driver_class", str, "core.default.DefaultStorageDriver")
    # maximum episode number per run
    MAX_EPISODES = ("max_episodes", int, 1)
    # maximum steps per episode
    MAX_STEPS = ("max_steps", int, 100)
    # process' working directory
    WORK_DIRECTORY = ("work_directory", str, "work_directory")
    # cluster information as list of tuples if `distributed_learning_enabled` is true
    CLUSTER_INFO = ("cluster_info", tuple, None)
    # train batch size per train step
    TRAIN_BATCH_SIZE = ("train_batch_size", int, -1)


class ViZDoomConfigurationProperty(ConfigurationProperty):
    """
    VizDoom game properties, for the game wrapper implementation.

    Detailed description can be found at:
    https://github.com/mwydmuch/ViZDoom/blob/master/doc/ConfigFile.md
    """
    DOOM_SCENARIO_PATH = ("doom_scenario_path", str, "scenarios/basic.wad")
    AVAILABLE_BUTTONS = ("available_buttons", Button, [Button.MOVE_LEFT, Button.MOVE_RIGHT, Button.ATTACK])
    AVAILABLE_GAME_VARIABLES = ("available_game_variables", GameVariable, [GameVariable.AMMO2])
    GAME_ARGS = ("game_args", str, None)
    AUTOMAP_BUFFER_ENABLED = ("automap_buffer_enabled", bool, True)
    AUTOMAP_MODE = ("automap_mode", AutomapMode, AutomapMode.NORMAL)
    AUTOMAP_RENDER_TEXTURES = ("automap_render_textures", bool, True)
    AUTOMAP_ROTATE = ("automap_rotate", bool, True)
    BUTTON_MAX_VALUE = ("button_max_value", float, None)
    CONSOLE_ENABLED = ("console_enabled", bool, True)
    DEATH_PENALTY = ("death_penalty", float, None)
    DEPTH_BUFFER_ENABLED = ("depth_buffer_enabled", bool, True)
    DOOM_CONFIG_PATH = ("doom_config_path", str, None)
    DOOM_GAME_PATH = ("doom_game_path", str, None)
    DOOM_MAP = ("doom_map", str, "map01")
    DOOM_SKILL = ("doom_skill", int, 5)
    EPISODE_START_TIME = ("episode_start_time", int, 10)
    EPISODE_TIMEOUT = ("episode_timeout", int, 200)
    LABELS_BUFFER_ENABLED = ("labels_buffer_enabled", bool, True)
    LIVING_REWARD = ("living_reward", float, -1.0)
    MODE = ("mode", Mode, Mode.PLAYER)
    OBJECTS_INFO_ENABLED = ("objects_info_enabled", bool, True)
    RENDER_ALL_FRAMES = ("render_all_frames", bool, True)
    RENDER_CORPSES = ("render_corpses", bool, True)
    RENDER_CROSSHAIR = ("render_crosshair", bool, True)
    RENDER_DECALS = ("render_decals", bool, True)
    RENDER_EFFECTS_SPRITES = ("render_effects_sprites", bool, True)
    RENDER_HUD = ("render_hud", bool, True)
    RENDER_MESSAGES = ("render_messages", bool, True)
    RENDER_MINIMAL_HUD = ("render_minimal_hud", bool, True)
    RENDER_PARTICLES = ("render_particles", bool, True)
    RENDER_SCREEN_FLASHES = ("render_screen_flashes", bool, True)
    RENDER_WEAPON = ("render_weapon", bool, True)
    SCREEN_FORMAT = ("screen_format", ScreenFormat, ScreenFormat.RGB24)
    SCREEN_RESOLUTION = ("screen_resolution", ScreenResolution, ScreenResolution.RES_640X480)
    SECTORS_INFO_ENABLED = ("sectors_info_enabled", bool, True)
    SEED = ("seed", int, None)
    SOUND_ENABLED = ("sound_enabled", bool, True)
    TICKRATE = ("ticrate", int, 60)
    VIZDOOM_PATH = ("vizdoom_path", str, None)
    WINDOW_VISIBLE = ("window_visible", bool, True)


class OpenAIConfigurationProperty(ConfigurationProperty):
    # gym environment name
    ENV_NAME = ("env_name", str, None)
    # screen render is enabled or not
    RENDER_ENABLED = ("render_enabled", bool, True)


class DriverConfigurationProperty(ConfigurationProperty):
    """
    Driver config properties.
    """
    # sleep time between steps (zero or less means no sleep)
    SLEEP_TIME = ("sleep_time", float, 0)
    # "frameskip" parameter, if it is for example "2", it means
    # the agent will interact with the environment in every 2th steps
    ACTION_FREQUENCY = ("action_frequency", int, 2)
    # network training is enabled or not
    TRAINING_ENABLED = ("training_enabled", bool, True)


class ModelSerializerProperty(ConfigurationProperty):
    """
    ModelSerializer config properties.
    """
    # model persistence is enabled or not
    MODEL_PERSISTENCE_ENABLED = ("model_persistence_enabled", bool, False)
    # model checkpoints are enabled or not
    MODEL_CHECKPOINTS_ENABLED = ("model_checkpoints_enabled", bool, False)


class AgentConfigurationProperty(ConfigurationProperty):
    """
    Agent configuration properties.
    """
    # initial value of the agent's exploration rate
    INITIAL_EXPLORATION_RATE = ("initial_exploration_rate", float, 1.0)
    # final value of the agent's exploration rate
    FINAL_EXPLORATION_RATE = ("final_exploration_rate", float, 0.01)
    # network input height (observations will be resized based on this parameter)
    SCREEN_HEIGHT = ("screen_height", int, 84)
    # network input weight (observations will be resized based on this parameter)
    SCREEN_WEIGHT = ("screen_weight", int, 84)
    # network input channels (observations will be reshaped based on this parameter)
    SCREEN_CHANNELS = ("screen_channels", int, 1)
    # maximum deepness of the replay memory
    MEMORY_MAX_SIZE = ("memory_max_size", int, 200)
    # distributed learning is enabled or not (if true, `CLUSTER_INFO` must be present)
    DISTRIBUTED_LEARNING_ENABLED = ("distributed_learning_enabled", bool, False)
    # name of the primary input observation in the observation vector
    INPUT_OBSERVATION_NAME = ("input_observation_name", str, "screen_buffer")
    # only used by eval/target network based architectures, weight synchronization rate (every x steps)
    NETWORK_SYNCHRONIZATION_FREQUENCY = ("network_synchronization_frequency", int, 100)
    # if `SCREEN_CHANNELS` == 1 then it must be true, converting the screen buffer
    # into a monochrome buffer, the original buffer's channel dim must be 3
    CONVERT_TO_MONOCHROME = ("convert_to_monochrome", bool, True)
    # frame stacking is enabled or not
    FRAME_STACKING_ENABLED = ("frame_stacking_enabled", bool, True)
    # if frame stacking is enabled, then the number of stacked frames
    NUMBER_OF_STACKED_FRAMES = ("number_of_stacked_frames", int, 4)
    # the number of observing episodes without training
    MAX_OBSERVE_EPISODES = ("max_observe_episodes", int, 10)
    # enumerated vs one hot encoded action space switch
    ONE_HOT_ENCODED_ACTION_SPACE = ("one_hot_encoded_action_space", bool, False)
    # percentage of the constant exploration phase
    CONSTANT_EXPLORATION_PERCENTAGE = ("constant_exploration_percentage", float, 0.1)
    # percentage of the decaying exploration phase
    DECAYING_EXPLORATION_PERCENTAGE = ("decaying_exploration_percentage", float, 0.6)


class NetworkConfigurationProperty(ConfigurationProperty):
    """
    Network configuration properties.
    """
    # convolution parameters (filters, kernel_size, strides) as a list of tuples
    CONVOLUTION_PARAMETERS = ("convolution_parameters", tuple, None)
    # fully connected layer parameters (num_units, dropout_params, weight_decay) as a list of tuples
    FULLY_CONNECTED_PARAMETERS = ("fully_connected_parameters", tuple, None)
    # network's optimizer learning rate
    OPTIMIZER_LEARNING_RATE = ("optimizer_learning_rate", float, 0.001)
    # reward discount factor (0.0 - 1.0)
    REWARD_DISCOUNT_FACTOR = ("reward_discount_factor", float, 0.99)
    # model's path to restore on process start (Optional)
    RESTORE_MODEL_FROM = ("restore_model_from", str, None)
    # checkpoint's path to restore on process start (Optional)
    RESTORE_CHECKPOINT_FROM = ("restore_checkpoint_from", str, None)
    # checkpoint creation episode frequency (episode % value == 0)
    CHECKPOINT_FREQUENCY = ("checkpoint_frequency", int, 100)


class TensorBoardConfigurationProperty(ConfigurationProperty):
    """
    Tensorboard configuration properties.
    """
    # metrics are enabled or not
    ENABLED_NETWORK_METRICS = ("enabled_network_metrics", KatMetrics, [KatMetrics.TENSORFLOW_TRAIN_LOSS_MEAN])
    # gpu profiler is enabled or not
    GPU_PROFILER_ENABLED = ("gpu_profiler_enabled", bool, False)

