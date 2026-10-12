import argparse
import hashlib
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

    parser.add_argument('--deployment-plan', type=Path, default=None,
                        help='Validated 3 x 8 x 4 grid count plan JSON')

    parser.add_argument('--plan-spec', type=Path, default=None,
                        help='Canonical PlanSpec token-list JSON; enables deterministic plan replay')

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
                        help='Enable config writing (default: False)')

    parser.add_argument('--enable-state',
                        action='store_true',
                        default=False,
                        help='Enable state writing (default: False)')

    parser.add_argument('--enable-event',
                        action='store_true',
                        default=False,
                        help='Enable event writing (default: False)')

    parser.add_argument('--enable-ai-action',
                        action='store_true',
                        default=False,
                        help='Enable AI action writing (default: False)')

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
    unified_mode = red_policy in ("r10_unified", "r12_topformer")
    commander = None
    if not unified_mode:
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
    if hierarchical_learning and red_motion_policy not in {"random_masked", "ppo", "mappo"}:
        raise ValueError(
            "r9_hierarchical_learning requires RED_MOTION_POLICY to be "
            "random_masked, ppo, or mappo"
        )
    learning_training = os.getenv("RED_LEARNING_TRAIN", "0") == "1"
    learning_model = os.getenv("RED_LEARNING_MODEL")
    learning_policy = None
    unified_policy = None
    plan_spec = None
    if args.plan_spec is not None:
        if args.deployment_plan is not None:
            raise ValueError("--plan-spec and --deployment-plan are mutually exclusive")
        if red_policy != "r12_topformer":
            raise ValueError("--plan-spec requires RED_POLICY=r12_topformer")
        if learning_training:
            raise ValueError("--plan-spec is evaluation-only and cannot train")
        from policies.red.learning.plan_skill import load_plan_spec
        plan_spec = load_plan_spec(args.plan_spec)
    # 底层规避技能（unified/topformer 线可选）。local_avoid_s3 默认使用
    # competition-platform-env 中已验证的 V13 gate-aligned S3 checkpoint，
    # mission_v12 门控仅在紧急威胁时交给 PPO，其他时刻直达/回正。
    evasion_skill = None
    if unified_mode and red_motion_policy in ("ppo_evasion", "local_avoid_s3"):
        if red_motion_policy == "local_avoid_s3":
            from policies.red.skills.evasion_ppo import LocalAvoidRuntimePolicy
            skill_model = os.getenv("EVASION_MODEL") or str(
                WORKSPACE_ROOT.parent / "competition-platform-env" / "models"
                / "local_avoid_ppo_v13_gate_aligned_s3_best.pt")
            evasion_skill = LocalAvoidRuntimePolicy(max_steps=args.max_steps)
        else:
            from policies.red.skills.evasion_ppo import (
                EvasionPPOV4Config,
                EvasionPPOV4Policy,
            )
            skill_model = os.getenv("EVASION_MODEL") or str(
                Path(__file__).resolve().parent.parent / "models"
                / "ppo_evasion_v4_final20_e01_best.pt")
            evasion_skill = EvasionPPOV4Policy(
                EvasionPPOV4Config(device="cpu", max_steps=args.max_steps))
        if Path(skill_model).is_file():
            evasion_skill.load(skill_model)
            logging.info(
                "[测试] 已加载底层规避技能: %s | gate=%s",
                skill_model,
                getattr(evasion_skill, "gate_mode", "n/a"),
            )
        else:
            message = f"规避技能模型缺失: {skill_model}"
            if learning_training or red_motion_policy == "local_avoid_s3":
                raise FileNotFoundError(message)
            logging.warning(f"[测试] {message}，退回自动制导")
            evasion_skill = None
        if evasion_skill is not None:
            evasion_skill.set_training(False)
    if plan_spec is not None:
        from policies.red.learning.plan_skill import DeterministicPlanPolicy
        from user_agents.unified_missile_agent import UnifiedAttackMissileAgent

        unified_policy = DeterministicPlanPolicy(
            plan_spec,
            initial_targets=initial_targets_from_observation(init_observation_ship),
            entity_types={
                simulator.entity_ext.entity.id:
                simulator.entity_ext.entity.entityType
                for simulator in simulators
                if simulator.entity_ext.entity.entityType in (21000, 21001, 21002)
            },
            max_steps=args.max_steps,
            search_bounds=(
                float(map_area.lonMin), float(map_area.lonMax),
                float(map_area.latMin), float(map_area.latMax),
            ),
            search_rows=int(os.getenv("SEARCH_ROWS", "4")),
            search_cols=int(os.getenv("SEARCH_COLS", "8")),
        )
        training_env.set_shared_learning_policy(unified_policy)
        logging.info(
            "[计划执行] 已加载 PlanSpec v%s: %s | 无训练参数",
            plan_spec.schema_version,
            plan_spec.plan_id,
        )
    elif red_policy == "r12_topformer":
        # 顶层 Transformer 策略：固定 V13 机动底层，集合编码 + 指针分配
        # + 卫星/部署稀疏决策头 + SMDP 团队标量回报的独立训练线。
        from policies.red.learning.unified import TopFormerConfig, TopFormerMAPPO
        from user_agents.unified_missile_agent import UnifiedAttackMissileAgent

        unified_policy = TopFormerMAPPO(
            TopFormerConfig(
                seed=int(os.getenv("RED_POLICY_SEED", "1")),
                max_steps=args.max_steps,
                # v3 默认四字段联合；goal 仅作为显式消融模式。
                upper_phase=os.getenv("UPPER_PHASE", "joint"),
                sat_trainable=os.getenv("SAT_LEARN", "1") == "1",
                deploy_trainable=os.getenv("DEPLOY_LEARN", "1") == "1",
                search_rows=int(os.getenv("SEARCH_ROWS", "4")),
                search_cols=int(os.getenv("SEARCH_COLS", "8")),
                update_every_rounds=int(os.getenv("UPDATE_EVERY_ROUNDS", "4")),
                min_update_records=int(os.getenv("MIN_UPDATE_RECORDS", "800")),
            ),
            initial_targets=initial_targets_from_observation(init_observation_ship),
            total_platforms=sum(
                1
                for simulator in simulators
                if simulator.entity_ext.entity.entityType in (21000, 21001, 21002)
            ),
            search_bounds=(
                float(map_area.lonMin), float(map_area.lonMax),
                float(map_area.latMin), float(map_area.latMax),
            ),
        )
        if learning_model and Path(learning_model).is_file():
            unified_policy.load(learning_model)
            unified_policy._loaded_model_path = str(Path(learning_model).resolve())
            unified_policy._loaded_model_sha256 = hashlib.sha256(
                Path(learning_model).read_bytes()
            ).hexdigest()
            if os.getenv("UPPER_PHASE"):
                unified_policy.config.upper_phase = os.environ["UPPER_PHASE"]
        unified_policy.set_training(learning_training)
        training_env.set_shared_learning_policy(unified_policy)
    elif unified_mode:
        from policies.red.learning.unified import UnifiedConfig, UnifiedMAPPO
        from user_agents.unified_missile_agent import UnifiedAttackMissileAgent

        unified_policy = UnifiedMAPPO(
            UnifiedConfig(
                seed=int(os.getenv("RED_POLICY_SEED", "1")),
                max_steps=args.max_steps,
                # 训练阶段入口（T1 三态控制）：motion|macro|search|deploy|joint
                train_phase=os.getenv("TRAIN_PHASE", "macro"),
            ),
            initial_targets=initial_targets_from_observation(init_observation_ship),
            total_platforms=sum(
                1
                for simulator in simulators
                if simulator.entity_ext.entity.entityType in (21000, 21001, 21002)
            ),
        )
        if learning_model and Path(learning_model).is_file():
            unified_policy.load(learning_model)
            unified_policy._loaded_model_path = str(Path(learning_model).resolve())
            unified_policy._loaded_model_sha256 = hashlib.sha256(
                Path(learning_model).read_bytes()
            ).hexdigest()
            if os.getenv("TRAIN_PHASE"):
                unified_policy.set_train_phase(os.environ["TRAIN_PHASE"])
        unified_policy.set_training(learning_training)
        training_env.set_shared_learning_policy(unified_policy)
    elif red_motion_policy in {"random_masked", "ppo", "mappo"}:
        learning_policy = build_learning_motion_policy(
            red_motion_policy,
            seed=int(os.getenv("RED_POLICY_SEED", "1")),
            model_path=learning_model,
            max_steps=args.max_steps,
            training=learning_training,
            observation_dim=90 if hierarchical_learning else 85,
        )
        if learning_training:
            training_env.set_shared_learning_policy(learning_policy)
    for i, simulator in enumerate(simulators):
        entity_id = simulator.entity_ext.entity.id
        agent_id = i + 1
        # 按类型注册红方飞行器智能体
        if simulator.entity_ext.entity.entityType == 21000 or simulator.entity_ext.entity.entityType == 21001 or simulator.entity_ext.entity.entityType == 21002:
            if unified_mode:
                agent = UnifiedAttackMissileAgent(
                    agent_id,
                    entity_id,
                    init_observation_ship,
                    unified_policy=unified_policy,
                    max_steps=args.max_steps,
                    agent_index=i,
                    evasion_skill=evasion_skill,
                )
                agent.set_missile_type(simulator.entity_ext.entity.entityType)
                training_env.agent_manager.register_agent(agent)
                unified_policy.register_entity_agent(entity_id, agent_id)
                # TopFormer 需要智能体对象构建场景 token（unified 策略忽略）
                if hasattr(unified_policy, "register_agent_object"):
                    unified_policy.register_agent_object(agent)
                continue
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
    if plan_spec is not None:
        from grid_maps.agent import GridDeployAgent
        deployment_entities = {
            item.entity.id: {"type": item.entity.entityType,
                             "position": {"alt": item.entity.lla.z}}
            for item in profile.imagineProfile.entityList
            if item.entity.entityType in (21000, 21001, 21002)
        }
        plan = unified_policy.build_deployment_plan(
            coordinates=profile.imagineProfile.redArea.coordinates,
            coordinates_hm=profile.imagineProfile.redArea.coordinatesHM,
        )
        deploy_agent = GridDeployAgent(
            -1, -1, {"entities": deployment_entities},
            profile.imagineProfile.redArea.coordinates,
            profile.imagineProfile.redArea.coordinatesHM,
            plan,
        )
        unified_policy.register_deployment_actions(deploy_agent.actions)
        logging.info("[计划执行] 部署模板已确定性编译到 8x4 合法网格")
    elif args.deployment_plan is not None:
        from grid_maps.agent import GridDeployAgent
        plan = json.loads(args.deployment_plan.read_text(encoding="utf-8"))
        deployment_entities = {
            item.entity.id: {"type": item.entity.entityType,
                             "position": {"alt": item.entity.lla.z}}
            for item in profile.imagineProfile.entityList
            if item.entity.entityType in (21000, 21001, 21002)
        }
        deploy_agent = GridDeployAgent(
            -1, -1, {"entities": deployment_entities},
            profile.imagineProfile.redArea.coordinates,
            profile.imagineProfile.redArea.coordinatesHM, plan,
        )
    else:
        # 新图（final20 等）带 deployment_grid.json：默认使用官方均衡教师部署
        # （3×8×4 计数矩阵 → GridDeployAgent 编译为部署命令，仿真开始前完成）
        grid_path = Path(args.scenario).parent / "deployment_grid.json"
        if grid_path.is_file():
            from grid_maps.agent import GridDeployAgent
            from grid_maps.deployment import balanced_plan
            deployment_entities = {
                item.entity.id: {"type": item.entity.entityType,
                                 "position": {"alt": item.entity.lla.z}}
                for item in profile.imagineProfile.entityList
                if item.entity.entityType in (21000, 21001, 21002)
            }
            expected = {t: len([e for e in deployment_entities.values()
                                if e["type"] == t])
                        for t in (21000, 21001, 21002)}
            plan = None
            plan_factory = None
            if unified_mode and unified_policy is not None \
                    and unified_policy.config.deploy_trainable:
                # 每个 episode 在 policy.reset_episode() 之后重新采样部署；旧实现
                # 仅在循环外采样一次，并在首轮 reset 时把 log-prob 记录清掉。
                def plan_factory():
                    try:
                        return unified_policy.build_deploy_plan(
                            coordinates=profile.imagineProfile.redArea.coordinates,
                            coordinates_hm=profile.imagineProfile.redArea.coordinatesHM,
                            expected_counts=expected,
                            init_observation=init_observation_ship,
                        )
                    except Exception as error:
                        logging.warning(
                            f"[测试] 学习部署失败，回退均衡教师: {error}"
                        )
                        return balanced_plan(expected)
            if plan is None:
                plan = balanced_plan(expected)
            deploy_agent = GridDeployAgent(
                -1, -1, {"entities": deployment_entities},
                profile.imagineProfile.redArea.coordinates,
                profile.imagineProfile.redArea.coordinatesHM,
                plan,
                plan_factory=plan_factory,
            )
            logging.info(
                "[测试] 部署方案: %s",
                "TopFormer 策略部署（每回合 reset 重新决策）"
                if plan_factory is not None
                else "官方均衡网格部署",
            )
        else:
            deploy_agent = DeployAgent(
                -1, -1, {}, profile.imagineProfile.redArea.coordinates,
                profile.imagineProfile.redArea.coordinatesHM,
                unified_policy=unified_policy if unified_mode else None,
            )
    training_env.agent_manager.register_agent(deploy_agent)

    logging.info("[测试] 运行环境初始化完成")

    reward_policy = load_reward_policy(args.scenario) if load_reward_policy is not None else None
    best_score = float(
        getattr(unified_policy, "best_score", float("-inf"))
        if unified_policy is not None else float("-inf")
    )
    save_train_best = os.getenv("SAVE_TRAIN_BEST", "1") == "1"
    for i in range(args.total_rounds):
        logging.info(f"[测试] 运行第 {i + 1} 轮")
        initial_observation = training_env.reset()
        if evasion_skill is not None:
            # 规避技能编码器逐回合重置（速度差分/航迹年龄缓存不跨回合）
            evasion_skill.reset_episode()
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
        # 部署阶段开始：统一策略需先构建部署期全局态（部署决策的 Critic 基准）
        if unified_mode:
            unified_policy.begin_deploy_phase(initial_observation)
        # 红方模型部署
        training_env.red_model_deploy()
        observe_deployment = getattr(unified_policy, "observe_deployment", None)
        if callable(observe_deployment):
            observe_deployment(training_env.engine.simulator_factory)
        start_time = time.perf_counter()
        # 运行仿真, 训练环境会自动调用智能体的get_action方法
        for step in range(1, args.max_steps + 1):
            obs, reward, done, info = training_env.step()
            final_observation = obs
            run_summary.update(step, obs)
            if step % 200 == 0:
                logging.info(f"[测试] Step {step}")
            if done:
                logging.info("[测试] 仿真结束")
                termination_reason = "environment_done"
                break
            if (
                os.getenv("PLAN_EARLY_STOP", "0") == "1"
                and unified_policy is not None
                and run_summary.reward_tracker is not None
            ):
                score_frozen = getattr(
                    unified_policy, "plan_score_frozen", None
                )
                if callable(score_frozen) and score_frozen(
                    step=step,
                    destroyed_objective_ids=(
                        run_summary.reward_tracker.destruction_steps
                    ),
                    objective_ids=(
                        run_summary.reward_tracker.policy.objective_ids
                    ),
                ):
                    logging.info(
                        "[测试] 确定性计划已无可改变官方分的后续动作，提前结束"
                    )
                    termination_reason = "plan_score_frozen"
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
        # 统一/顶层策略回合执行指标：发射时间分布/分配直方图/重复集火/
        # 9500发现率/重瞄准延迟/卫星次数（在下一轮 reset_episode 清零前读取）
        if unified_mode and unified_policy is not None:
            ships_total = sum(
                1
                for entity in initial_observation.get("entities", {}).values()
                if isinstance(entity, dict) and int(entity.get("type", -1)) == 9500
            )
            summary["unified_metrics"] = unified_policy.episode_metrics(
                ships_total=ships_total
            )
        # Freeze the P0 lifecycle ledger before finish_episode() can update
        # policy parameters.  This keeps each summary tied to the checkpoint
        # version that actually generated its actions.
        summary["episode_telemetry"] = training_env.build_episode_telemetry(
            summary.get("score") or {}
        )
        reconcile_plan = getattr(
            unified_policy, "reconcile_episode_telemetry", None
        )
        if callable(reconcile_plan):
            reconcile_plan(summary["episode_telemetry"])
        execution_consistency = getattr(
            unified_policy, "execution_consistency", None
        )
        if callable(execution_consistency):
            summary["plan_execution"] = execution_consistency()
        # Ad-hoc/native smoke scenarios may not have a RewardPolicy, in which
        # case RunSummary deliberately stores score=None.  They must still
        # reach finish_episode() so the learning lifecycle can close/update.
        score_summary = summary.get("score") or {}
        round_score = float(score_summary.get("score", 0.0))
        # best 必须在 finish_episode() 可能执行 PPO 更新之前保存；这样文件中的
        # 参数才是实际产生 round_score 的参数，而不是尚未评估的更新后参数。
        if (learning_training and unified_policy is not None and learning_model
                and save_train_best):
            if round_score > best_score:
                best_score = round_score
                unified_policy.best_score = best_score
                best_path = learning_model.replace(".pt", "_best.pt")
                save_evaluated = getattr(unified_policy, "save_evaluated", None)
                if save_evaluated is not None:
                    save_evaluated(best_path)
                else:
                    unified_policy.save(best_path)
                logging.info(
                    f"[测试] 新最优模型(得分 {round_score:.2f})已保存: {best_path}"
                )
        # 终局部署结算：部署转移以整局官方比分做 episode-level return
        if unified_mode and unified_policy is not None:
            unified_policy.finish_episode(round_score)
        summary_filename = "summary.json" if args.total_rounds == 1 else f"summary_round_{i + 1}.json"
        summary_path = run_summary.write(write_config.output_dir, summary, summary_filename)
        print("FINAL_SUMMARY " + json.dumps(summary, ensure_ascii=False))
        logging.info(f"[测试] 单局汇总已写入: {summary_path}")
        if learning_training and learning_policy is not None and learning_model:
            Path(learning_model).parent.mkdir(parents=True, exist_ok=True)
            learning_policy.save(learning_model)
            logging.info(f"[测试] 第 {i + 1} 轮红方学习模型已保存: {learning_model}")
        if learning_training and unified_policy is not None and learning_model:
            # 检查点保护：last 每轮覆盖 / best 按官方比分 / periodic 留档
            unified_policy.save(learning_model)
            logging.info(
                f"[测试] 第 {i + 1} 轮统一策略模型已保存: {learning_model}"
            )
            if (i + 1) % 10 == 0:
                periodic_path = learning_model.replace(
                    ".pt", f"_round_{i + 1}.pt"
                )
                unified_policy.save(periodic_path)
                logging.info(f"[测试] 周期检查点已保存: {periodic_path}")

    training_env.close()
    # 关闭写入器
    get_writer().close()


if __name__ == '__main__':
    main()
