"""Validate a 3 x 8 x 4 count plan and convert it to the existing Nx5 API."""
import math

RED_TYPES = (21000, 21001, 21002)


def validate_plan(plan, expected_counts):
    if not isinstance(plan, dict) or set(plan) != {'version', 'counts'} or type(plan['version']) is not int or plan['version'] != 1:
        raise ValueError('部署方案必须仅包含 version=1 和 counts。')
    counts = plan['counts']
    if not isinstance(counts, dict) or set(counts) != {str(t) for t in RED_TYPES}:
        raise ValueError('counts 必须提供 21000、21001、21002 三类。')
    for t in RED_TYPES:
        matrix = counts[str(t)]
        if not isinstance(matrix, list) or len(matrix) != 8:
            raise ValueError(f'{t} 必须为 8 行。')
        if any(not isinstance(row, list) or len(row) != 4 for row in matrix):
            raise ValueError(f'{t} 每行必须为 4 列。')
        if any(type(n) is not int or n < 0 for row in matrix for n in row):
            raise ValueError('每格数量必须是非负整数。')
        if sum(map(sum, matrix)) != expected_counts[t]:
            raise ValueError(f'{t} 数量必须等于 {expected_counts[t]}。')
    return counts


def balanced_plan(expected_counts):
    """Spread each type and their combined total evenly over the shared cells."""
    order = [(row, (row + offset) % 4) for offset in range(4) for row in range(8)]
    totals = {cell: 0 for cell in order}
    counts = {}
    for t in RED_TYPES:
        q, rem = divmod(expected_counts[t], 32)
        matrix = [[q] * 4 for _ in range(8)]
        for cell in totals:
            totals[cell] += q
        # Stable ties rotate columns while covering every row.
        for row, col in sorted(order, key=totals.get)[:rem]:
            matrix[row][col] += 1
            totals[row, col] += 1
        counts[str(t)] = matrix
    return {'version': 1, 'counts': counts}


def compile_actions(plan, entities, coordinates, coordinates_hm):
    """Entities follow TrainingEnv red observation: type + position.alt."""
    if coordinates != coordinates_hm or len(coordinates) != 1 or len(coordinates[0]) != 4:
        raise ValueError('三类单位必须共享同一个四角矩形部署区。')
    points = coordinates[0]
    if any(len(p) != 2 or any(type(v) not in (int,float) or not math.isfinite(v) for v in p) for p in points):
        raise ValueError('部署区坐标必须是有限数值。')
    west, east = min(p[0] for p in points), max(p[0] for p in points)
    south, north = min(p[1] for p in points), max(p[1] for p in points)
    if west >= east or south >= north or {tuple(p) for p in points} != {(west,south),(west,north),(east,south),(east,north)}:
        raise ValueError('部署区必须是非退化的轴对齐矩形。')
    by_type = {t: sorted(int(i) for i,e in entities.items() if e['type']==t) for t in RED_TYPES}
    if any(type(i) is not int for i in entities) or len(set(i for ids in by_type.values() for i in ids)) != sum(map(len,by_type.values())):
        raise ValueError('观测实体编号必须为唯一整数。')
    expected = {t:len(by_type[t]) for t in RED_TYPES}
    counts = validate_plan(plan, expected)
    offsets = {t:0 for t in RED_TYPES}; result=[]
    cw, ch = (east-west)/4, (north-south)/8
    for row in range(8):
        for col in range(4):
            ids=[]
            for t in RED_TYPES:
                n=counts[str(t)][row][col]; start=offsets[t]
                ids.extend(by_type[t][start:start+n]); offsets[t]+=n
            if not ids:
                continue
            columns=math.ceil(math.sqrt(len(ids))); rows=math.ceil(len(ids)/columns)
            for index, entity_id in enumerate(ids):
                # All types share a single subgrid, so mixed cells never duplicate points.
                u=.1+.8*((index%columns)+.5)/columns
                v=.1+.8*((index//columns)+.5)/rows
                alt=entities[entity_id]['position']['alt']
                if type(alt) not in (int,float) or not math.isfinite(alt):
                    raise ValueError('想定高度必须为有限数值。')
                result.append([0,entity_id,west+(col+u)*cw,north-(row+v)*ch,alt])
    return result
