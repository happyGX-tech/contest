# 三档地图与网格部署

20张地图分为简单6张、中等6张、复杂8张。比较均匀8张，部分不均匀12张，比例4∶6。左侧20%为共享4列×8行部署区，三类红方单位可以混合部署。默认方案同时平衡各类型和全部单位的格位数量，各格最多相差1个单位。

|配置|简单|中等|复杂|
|---|---:|---:|---:|
|地图尺寸（公里）|75×60|87.5×70|100×80|
|毁灭者／突击者／清道夫|17／24／85|27／37／128|48／66／214|
|红方总数|126|192|328|
|母巢／哨卫巢／暗巢|3／5／4|4／10／8|5／20／18|
|自爆虫库存|97|224|435|
|自爆虫／红方|0.770|1.167|1.326|
|全部实体数|242|450|828|

母巢、哨卫巢、暗巢耐久为24／18／1.2，计分权重5∶2∶1。蓝方巢体固定部署，出巢自爆虫按模型运动。红方为空中发射，想定高度3000米，模型发射初速300。地图横纵坐标统一使用公里；复杂档地图尺寸为100×80公里。

## 地图结构

哨卫巢通过疏密变化、前后错位和纵深接续来组织视野与拦截窗口。母巢作为少量高价值纵深目标，暗巢形成搜索支线。通道由覆盖与库存的相对压力形成，不设置禁止穿越的地形障碍。同档使用相同资源预算，通过点位和局部库存分布形成差异。

按4条等高横带分别统计公开巢体和自爆虫库存。变异系数CV为总体标准差除以均值。比较均匀要求库存CV≤0.20，公开目标CV≤max(0.20,给定整数总量的最小可达CV)；任一超过即归为部分不均匀。

每张图包含scenario.json、case_info.json、layout.design.json、deployment_grid.json及deployment_plan.example.json。scenario.json与case_info.json须放在同一目录。map_info.png为组织方图，map_public.png隐藏暗巢。淡蓝圆表示逐塔20千米规则范围；引擎实际ECEF X/Y判据与地表正圆有差异，另见map_engine_check.png。

## 运行与生成

在仓库根目录执行，使用CPython 3.11 x64及仓库依赖。地图渲染还需Pillow、NumPy和Windows微软雅黑字体。

```powershell
python run.py run --scenario "final20/easy/E01" --red-policy r7_static_search --blue-policy b0_fixed_ratio_random --red-motion-policy reactive_evasion --blue-interceptor-ratio 2 --seed 42 --run-id E01 -- --total-rounds 1 --max-steps 3000 --render-mode none --deployment-plan "scenarios/cases/final20/easy/E01/deployment_plan.example.json"
python -m tools.current_map_design.build_maps --output rebuilt_cases
python -m tools.current_map_design.render_maps --cases scenarios/cases/final20
```

运行时通过--deployment-plan读取网格数量矩阵。配套输入数据用于确定性生成相同地图与均匀部署方案。
