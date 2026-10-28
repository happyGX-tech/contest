"""Convert the current red action protocol without interpreting hidden state."""

import numpy as np

from envengine.action_validation import IllegalActionError
from envengine.agent_manager.actions.aircraft_action import SetDesiredAccZ, ChangeTargetAction
from envengine.agent_manager.actions.aircraft_action.use_satellite import UseSatelliteAction
from envengine.common import Vector3d


class CommandConverter:
    @staticmethod
    def common_converter(actions) -> list[dict]:
        """Accept maneuver (0), search (2), satellite (3), and target ID (4).

        Deployment starts every red unit. An explicit launch action is invalid.
        The engine resolves target IDs against each executor's legal knowledge.
        """
        converted = []
        for raw_action in actions:
            try:
                rows = np.asarray(raw_action, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise IllegalActionError("动作必须是数值数组") from exc
            if rows.shape in ((0,), (0, 4)):
                continue
            if rows.ndim != 2 or rows.shape[1] != 4 or not np.isfinite(rows).all():
                raise IllegalActionError("动作必须是有限数值 N x 4 数组")
            for action_type, executor_id, value, other in rows:
                if not float(executor_id).is_integer() or executor_id < 0:
                    raise IllegalActionError("执行实体编号必须是非负整数")
                executor_id = int(executor_id)
                if action_type == 0:
                    command = SetDesiredAccZ(executor_id=executor_id, acc_z=float(value))
                elif action_type == 2:
                    command = ChangeTargetAction(
                        executor_id=executor_id,
                        intent="search",
                        target=Vector3d(float(value), float(other), 0.0),
                    )
                elif action_type == 3:
                    command = UseSatelliteAction(executor_id=executor_id)
                elif action_type == 4:
                    if not float(value).is_integer() or value < 0:
                        raise IllegalActionError("攻击目标编号必须是非负整数")
                    command = ChangeTargetAction(
                        executor_id=executor_id, intent="track", target_id=int(value)
                    )
                else:
                    raise IllegalActionError(f"不支持动作类型 {action_type}；有效类型为 0、2、3、4")
                converted.append(command.to_dict())
        return converted
