import argparse
import logging
import os
import random
import time
import json
import sys
from datetime import datetime
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import requests
import numpy as np
from urllib.parse import urlparse
from envengine import Profile, TrainingEnv
from envengine.sdk.log import LogManager
from envengine.sdk.writer import WriteConfig, init_writer, get_writer, write_immediately
from user_agents import AttackMissileAgent, DeployAgent
from evaluation import RunSummary

try:
    from scenarios.cases import RewardTracker, load_reward_policy
except ImportError:
    RewardTracker = load_reward_policy = None

from policies.red import Position, RED_POLICY_CHOICES, RedBaselineCommander, initial_targets_from_observation
from policies.red.learning import build_learning_motion_policy


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Simulation Training Environment')

    parser.add_argument('--scenario',
                        type=str,
                        default='./scenarios/platform.json',
                        help='Path or URL to scenario configuration file (default: ./scenarios/platform.json)')

    parser.add_argument('--total-rounds',
                        type=int,
                        default=100,
                        help='Total simulation rounds (default: 1)')

    parser.add_argument('--max-steps',
                        type=int,
                        default=1000,
                        help='Maximum steps per round (default: 1000)')

    parser.add_argument('--render-mode',
                        type=str,
                        default="human",
                        choices=['human', 'none'],
                        help='Render mode (default: human)')

    parser.add_argument('--output-dir',
                        type=str,
                        default='results',
                        help='Output directory base path (default: results)')

    parser.add_argument('--batch-size',
                        type=int,
                        default=100,
                        help='Batch size for writer (default: 100)')

    parser.add_argument('--disable-log-color',
                        action='store_false',
                        dest='enable_log_color',
                        default=True,
                        help='Disable log color')

    parser.add_argument('--verbose',
                        action='store_true',
                        default=False,
                        help='Enable verbose output')

    parser.add_argument('--enable-config',
                        action='store_true',
                        default=False,
                        help='Enable config writing (default: True)')

    parser.add_argument('--enable-state',
                        action='store_true',
                        default=False,
                        help='Enable state writing (default: True)')

    parser.add_argument('--enable-event',
                        action='store_true',
                        default=False,
                        help='Enable event writing (default: True)')

    parser.add_argument('--enable-ai-action',
                        action='store_true',
                        default=False,
                        help='Enable AI action writing (default: True)')

    return parser.parse_args()


def read_profile(url: str) -> Profile:
    """
    通过本地文件url读取或者通过远程url读取
    :param url:
    :return:
    """
    # 判断是否为有效的URL（包含协议）
    parsed_url = urlparse(url)

    if parsed_url.scheme in ['http', 'https']:  # 远程文件路径
        response = requests.get(url)
        if response.status_code == 200:
            profile_data = response.json()['data']
            return Profile.from_dict(profile_data)
        else:
            raise Exception(f"无法从远程URL获取数据: {response.status_code}")
    elif os.path.exists(url):  # 本地文件路径
        with open(url, 'r', encoding='utf-8') as file:
            profile_data = json.load(file)
            return Profile.from_dict(profile_data)
    else:
        raise ValueError(f"无效的URL或文件路径: {url}")


def main():
    # 解析命令行参数
    args = parse_args()
    simulation_seed = os.getenv("SIMULATION_SEED")
    if simulation_seed is not None:
        random.seed(int(simulation_seed))
        np.random.seed(int(simulation_seed))

    # 打印配置信息
    logging.info("=" * 60)
    logging.info("Simulation Configuration:")
    logging.info(f"  Scenario file: {args.scenario}")
    logging.info(f"  Total rounds: {args.total_rounds}")
    logging.info(f"  Max steps per round: {args.max_steps}")
    logging.info(f"  Render mode: {args.render_mode}")
    logging.info(f"  Output directory: {args.output_dir}")
    logging.info(f"  Batch size: {args.batch_size}")
    logging.info(f"  Enable gog color: {args.enable_log_color}")
    logging.info(f"  Verbose: {args.verbose}")
    logging.info(f"  Enable state: {args.enable_state}")
    logging.info(f"  Enable event: {args.enable_event}")
    logging.info(f"  Simulation seed: {simulation_seed}")
    logging.info(f"  Enable AI action: {args.enable_ai_action}")

    logging.info("=" * 60)

    # 全局写文件配置
    write_config = WriteConfig(
        output_dir=f"{args.output_dir}/" + datetime.now().strftime("%Y%m%d%H%M%S"),
        verbose=args.verbose,
        batch_size=args.batch_size,
        enable_config=args.enable_config,
        enable_state=args.enable_state,
        enable_event=args.enable_event,
        enable_ai_action=args.enable_ai_action
    )

    # 初始化日志管理器
    LogManager(color_enabled=args.enable_log_color)

    # 初始化写入器
    init_writer(write_config)

    # 加载配置
    profile: Profile = read_profile(args.scenario)
    # print(profile)
    # 实例化训练环境
    training_env = TrainingEnv(profile, render_mode=args.render_mode)
    # 注册智能体, 为每个实体创建一个智能体
    simulators = training_env.engine.simulator_factory.get_all_simulators()

    # 初始化需要给红方 AI 的信息
    init_observation_ship = training_env._get_init_ship_observation()
    red_policy = os.getenv("RED_POLICY", "r0_random")
    if red_policy not in RED_POLICY_CHOICES:
        raise ValueError(f"Unsupported red policy '{red_policy}'")
    map_area = profile.imagineProfile.mapArea
    commander = RedBaselineCommander(
        initial_targets_from_observation(init_observation_ship),
        policy_name=red_policy,
        seed=int(os.getenv("RED_POLICY_SEED", "1")),
        search_polygon=(
            Position(map_area.lonMin, map_area.latMin),
            Position(map_area.lonMax, map_area.latMin),
            Position(map_area.lonMax, map_area.latMax),
            Position(map_area.lonMin, map_area.latMax),
        ),
    )
    red_motion_policy = os.getenv("RED_MOTION_POLICY", "reactive_evasion")
    hierarchical_learning = red_policy == "r9_hierarchical_learning"
    if hierarchical_learning and red_motion_policy not in {
        "straight", "random_masked", "ppo", "mappo", "local_avoid_s3"
    }:
        raise ValueError(
            "r9_hierarchical_learning requires RED_MOTION_POLICY to be "
            "straight, random_masked, ppo, mappo, or local_avoid_s3"
        )
    learning_training = os.getenv("RED_LEARNING_TRAIN", "0") == "1"
    learning_model = os.getenv("RED_LEARNING_MODEL")
    learning_policy = None
    if red_motion_policy in {"random_masked", "ppo", "mappo", "local_avoid_s3"}:
        learning_policy = build_learning_motion_policy(
            red_motion_policy,
            seed=int(os.getenv("RED_POLICY_SEED", "1")),
            model_path=learning_model,
            max_steps=args.max_steps,
            training=learning_training,
            observation_dim=90 if hierarchical_learning else 85,
        )
        if learning_training or bool(
            getattr(learning_policy, "requires_environment_hooks", False)
        ):
            training_env.set_shared_learning_policy(learning_policy)
    for i, simulator in enumerate(simulators):
        entity_id = simulator.entity_ext.entity.id
        agent_id = i + 1
        # 按类型注册红方飞行器智能体
        if simulator.entity_ext.entity.entityType == 21000 or simulator.entity_ext.entity.entityType == 21001 or simulator.entity_ext.entity.entityType == 21002:
            agent = AttackMissileAgent(
                agent_id,
                entity_id,
                init_observation_ship,
                commander=commander,
                motion_policy=red_motion_policy,
                learning_policy=learning_policy,
                learning_max_steps=args.max_steps,
                hierarchical_learning=hierarchical_learning,
            )
            training_env.agent_manager.register_agent(agent)
            commander.register_platform(entity_id)
    logging.info(f"[测试] 已注册 {training_env.agent_manager.get_agent_count()} 个智能体")

    # 注册部署智能体
    deploy_agent = DeployAgent(-1, -1,{}, profile.imagineProfile.redArea.coordinates, profile.imagineProfile.redArea.coordinatesHM)
    training_env.agent_manager.register_agent(deploy_agent)

    logging.info("[测试] 运行环境初始化完成")

    reward_policy = load_reward_policy(args.scenario) if load_reward_policy is not None else None
    for i in range(args.total_rounds):
        logging.info(f"[测试] 运行第 {i + 1} 轮")
        initial_observation = training_env.reset()
        run_summary = RunSummary(
            scenario=reward_policy.scenario_id if reward_policy is not None else str(args.scenario),
            policies={
                "red": red_policy,
                "red_motion": red_motion_policy,
                "blue": os.getenv("BLUE_POLICY", "engine_default"),
            },
            reward_tracker=RewardTracker(reward_policy) if reward_policy is not None else None,
        )
        run_summary.start(initial_observation)
        final_observation = initial_observation
        termination_reason = "time_limit"
        # 红方模型部署
        training_env.red_model_deploy()
        # time.sleep(1000)
        start_time = time.perf_counter()
        # 运行仿真, 训练环境会自动调用智能体的get_action方法
        for step in range(1, args.max_steps + 1):
            obs, reward, done, info = training_env.step()
            final_observation = obs
            run_summary.update(step, obs)
            if step % 200 == 0:
                # logging.info(f"[测试] Step {step}: obs={obs}, reward={reward}, done={done}, info={info}")
                logging.info(f"[测试] Step {step}")
                # pass
            if done:
                logging.info("[测试] 仿真结束")
                termination_reason = "environment_done"
                break
        end_time = time.perf_counter()

        logging.info(f"[测试] 第 {i + 1} 轮结束，本轮仿真总用时: {end_time - start_time:.6f} 秒")
        # 写剩余缓冲区数据
        write_immediately()
        red_launched = sum(
            getattr(agent, "launch_step", -1) >= 0
            for agent in training_env.agent_manager.get_all_agents()
        )
        summary = run_summary.build(
            final_observation,
            termination_reason=termination_reason,
            red_launched=red_launched,
        )
        summary_filename = "summary.json" if args.total_rounds == 1 else f"summary_round_{i + 1}.json"
        summary_path = run_summary.write(write_config.output_dir, summary, summary_filename)
        print("FINAL_SUMMARY " + json.dumps(summary, ensure_ascii=False))
        logging.info(f"[测试] 单局汇总已写入: {summary_path}")
        if learning_training and learning_policy is not None and learning_model:
            learning_policy.save(learning_model)
            logging.info(
                f"[测试] 第 {i + 1} 轮红方学习模型已保存: {learning_model}"
            )

    training_env.close()
    # 关闭写入器
    get_writer().close()


if __name__ == '__main__':
    main()
