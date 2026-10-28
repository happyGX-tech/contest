"""Generate the current three-tier map design and uniform deployment."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib,json,math,statistics
from pathlib import Path
from grid_maps.coordinates import set_position
from grid_maps.deployment import RED_TYPES,compile_actions,balanced_plan

BASE=Path(__file__).resolve().parent
BUDGETS=json.loads((BASE/"selected_budgets.json").read_text(encoding="utf-8"))
STOCK={k:v["stock"] for k,v in BUDGETS.items()}
read=lambda p:json.loads(p.read_text(encoding="utf-8"))
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def cv(v):return statistics.pstdev(v)/statistics.mean(v)

def build(source, output):
    scenario=read(source/"scenario.json");info=read(source/"case_info.json");layout=read(source/"layout.design.json")
    tier=info["difficulty"];mid=info["scenario_id"]
    catalog=BASE/"inputs/mother_catalog"/tier/mid
    old_scenario=read(catalog/"scenario.json");old_info=read(catalog/"case_info.json");old_layout=read(catalog/"layout.design.json")
    entities=scenario["imagineProfile"]["entityList"];by_id={w["entity"]["id"]:w for w in entities}
    existing=set(by_id);parents=sorted([r for r in info["roots"] if r["interceptors"]>0],key=lambda r:r["id"])
    scaled={r["id"]:r["interceptors"]*STOCK[tier]/info["interceptor_total"] for r in parents}
    counts={oid:int(v) for oid,v in scaled.items()}
    for oid in sorted(counts,key=lambda i:(-(scaled[i]-counts[i]),i))[:STOCK[tier]-sum(counts.values())]:counts[oid]+=1
    next_id=max(max(by_id),max(w["entity"]["id"] for w in old_scenario["imagineProfile"]["entityList"]))+1
    for root in parents:
        parent=by_id[root["id"]]["entity"]
        children=[by_id[i] for i in parent["childrenId"] if by_id[i]["entity"]["entityType"]==24000]
        for _ in range(counts[root["id"]]-len(children)):
            child=deepcopy(children[0]);e=child["entity"];e["id"]=next_id;e["nameChn"]=f"自爆虫_{next_id}";e["nameEn"]=f"Interceptor_{next_id}"
            parent["childrenId"].append(next_id);entities.append(child);by_id[next_id]=child;next_id+=1
        root["interceptors"]=counts[root["id"]]
    for r in layout["roots"]:
        if r["source_id"] in counts:r["swarm_stock"]=counts[r["source_id"]]
    public_total=len(info["public_objective_ids"])+BUDGETS[tier]["extra_mother"]
    q,rem=divmod(public_total,4);limit=max(.2,cv([q+(i<rem) for i in range(4)]))
    stock_bands=[sum(r["swarm_stock"] for r in layout["roots"] if int(r["y_norm"]*4)==b) for b in range(4)]
    if BUDGETS[tier]["extra_mother"]:
        candidates=[]
        for r in old_layout["roots"]:
            if r["kind"]!="mother" or r["source_id"] in existing:continue
            bands=[sum(x["kind"]!="hidden" and int(x["y_norm"]*4)==b for x in [*layout["roots"],r]) for b in range(4)]
            uniform=cv(bands)<=limit+1e-10 and cv(stock_bands)<=.2+1e-10
            if uniform!=(info["distribution"]=="比较均匀"):continue
            distance=min(math.hypot(r["x"]-w["x"],r["y"]-w["y"]) for w in layout["roots"] if w["kind"]=="watch")
            candidates.append((distance,-r["x_norm"],r["source_id"],r))
        if not candidates:raise ValueError(f"No mother location preserves distribution for {mid}")
        _,_,oid,new_layout=min(candidates,key=lambda x:x[:3])
        mother=deepcopy(next(w for w in old_scenario["imagineProfile"]["entityList"] if w["entity"]["id"]==oid))
        mother["entity"]["survivePoints"]=24.0
        assert mother["entity"]["childrenId"]==[]
        entities.append(mother);by_id[oid]=mother
        root=deepcopy(next(r for r in old_info["roots"] if r["id"]==oid));root["hp"]=24.0
        info["roots"].append(root);layout["roots"].append(deepcopy(new_layout))
        for k in ("objective_ids","public_objective_ids"):info[k].append(oid)
        info["root_counts"]["target"]+=1;info["blue_root_total"]+=1
    info["blue_root_health_total"]=sum(r["hp"] for r in info["roots"])
    info["interceptor_total"]=STOCK[tier];info["interceptor_counts"]={str(k):v for k,v in counts.items()};info["entity_total"]=len(entities)
    info["blue_health_to_red_nominal_damage_ratio"]=info["blue_root_health_total"]/info["red_nominal_damage_total"]
    red=dict(zip(RED_TYPES,info["red_missile_counts"].values()));plan=balanced_plan(red)
    obs={oid:{"type":w["entity"]["entityType"],"position":{"alt":w["entity"]["lla"]["z"]}} for oid,w in by_id.items() if w["entity"]["entityType"] in RED_TYPES}
    area=scenario["imagineProfile"]["redArea"]
    for _,pid,lon,lat,alt in compile_actions(plan,obs,area["coordinates"],area["coordinatesHM"]):set_position(by_id[pid]["entity"],{"x":lon,"y":lat,"z":alt})
    info["band_counts"]=[{"public":sum(r["kind"]!="hidden" and int(r["y_norm"]*4)==b for r in layout["roots"]),"swarm":stock_bands[b]} for b in range(4)]
    info["band_cv"]={k:cv([r[k] for r in info["band_counts"]]) for k in ("public","swarm")}
    info["distribution_rule"]["public_cv_limit"]=limit;layout["distribution_rule"]=deepcopy(info["distribution_rule"])
    info["design_intent"]=f'{info["title"]}：以哨卫巢视野组织通道，母巢提供纵深目标，暗巢形成搜索支线；默认红方在32格均匀部署。'
    info["validation_status"]="Current design; see representative_results.json for sampled native results."
    info["deployment_rule"]="For each type and all types combined, cell counts differ by at most one."
    for k in ("revision","quantity_revision","health_revision"):info.pop(k,None)
    layout["topology_idea"]=info["design_intent"];layout["revision"]="current_map_design"
    budget=layout["budget"];budget.update(blue=[info["root_counts"][k] for k in ("target","site","ship")],blue_total=info["blue_root_total"],swarm=STOCK[tier],total_entities=len(entities),total_hp=info["blue_root_health_total"],hp_ratio=info["blue_health_to_red_nominal_damage_ratio"],swarm_ratio=STOCK[tier]/sum(red.values()),public_target_budget_ratio=len(info["public_objective_ids"])/(red[21000]+red[21001]/4))
    layout["requested_bands"]={"public_roots":[r["public"] for r in info["band_counts"]],"swarm_stock":stock_bands}
    layout["deployment"]["red_assignment"]="uniform_integer_counts"
    scenario["environmentProfile"]["projectName"]=mid+"_current_design"
    dest=output/tier/mid;dest.mkdir(parents=True,exist_ok=True)
    write(dest/"scenario.json",scenario);info["scenario_sha256"]=hashlib.sha256((dest/"scenario.json").read_bytes()).hexdigest()
    write(dest/"case_info.json",info);write(dest/"layout.design.json",layout);write(dest/"deployment_plan.example.json",plan)
    grid=read(source/"deployment_grid.json");grid["deployment_rule"]=info["deployment_rule"];write(dest/"deployment_grid.json",grid)
    actual=Counter(w["entity"]["entityType"] for w in entities)
    assert actual[9400]<min(actual[9600],actual[9500])
    for w in entities:
        e=w["entity"]
        for child in e["childrenId"]:assert by_id[child]["entity"]["parentId"]==e["id"]
        if e["parentId"]!=-1:assert e["id"] in by_id[e["parentId"]]["entity"]["childrenId"]
    return {"map":mid,"tier":tier,"mothers":actual[9400],"stock":actual[24000],"red":sum(red.values()),"entities":len(entities),"distribution":info["distribution"],"scenario_sha256":info["scenario_sha256"],"uniform_deployment_verified":True}

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=BASE/'cases')
    args=parser.parse_args()
    rows=[build(p.parent,args.output) for p in sorted((BASE/"inputs/starting_cases").glob("*/*/scenario.json"))]
    assert Counter(r["distribution"] for r in rows)=={"比较均匀":8,"部分不均匀":12}
    write(args.output/"manifest.json",{"design":"current_three_tier_maps","maps":rows})
    write(BASE/"design_validation.json",rows)
    print(json.dumps(rows,ensure_ascii=False,indent=2))
