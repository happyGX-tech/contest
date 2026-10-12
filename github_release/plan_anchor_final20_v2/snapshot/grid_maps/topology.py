"""Fixed geometric motifs. No policy-dependent or sensor-evasion optimization."""
import math
from collections import Counter

MOTHER_BANDS = {
 'E01':[1,1,2,1], 'E02':[1,1,0,3], 'E03':[1,0,2,2],
 'E04':[0,2,2,1], 'E05':[0,1,2,2], 'E06':[1,2,1,1],
 'M01':[2,2,1,2], 'M02':[2,1,2,2], 'M03':[1,1,3,2],
 'M04':[1,1,2,3], 'M05':[1,3,2,1], 'M06':[3,1,2,1],
}
IDEAS = {
 'E01':'扇面骨架：上下向外展开，中部内收；各横带资源基本均衡。',
 'E02':'错位双层：两条纵向点列交替布置，局部密集区在南部。',
 'E03':'楔形骨架：前层节点向中部收束，后层保持较宽展开。',
 'E04':'三簇结构：北部小簇、中部主簇、南部小簇，簇间保留可辨间隔。',
 'E05':'偏置新月：沿连续弧形摆点，南部数量逐渐增加。',
 'E06':'三列栅栏：三条纵向骨架错位分布，上下预算保持均衡。',
 'M01':'双层分流：前后两个层次，上下采用不同横向位置，资源均衡。',
 'M02':'菱形外环：外层两侧形成菱形轮廓，内部节点避免全部挤在中轴。',
 'M03':'沙漏结构：上下展开、中部收窄；南侧较多节点形成局部差异。',
 'M04':'北疏南密：北部少量离散点，南部形成矩形主簇。',
 'M05':'弧门结构：前后两条相互错开的弧线，第二横带形成局部密集。',
 'M06':'阶梯结构：四横带依次向右错位，每带内保留小幅纵深差。',
}


def x_position(map_id, kind, y, band, i, n):
    shift={'watch':0,'hidden':.12,'mother':.30}[kind]
    spread=(i-(n-1)/2)*.038
    if map_id=='E01': x=.32+.20*abs(2*y-1)+shift+spread
    elif map_id=='E02': x=(.34 if (band+i)%2==0 else .49)+shift+spread
    elif map_id=='E03': x=.30+.24*(1-abs(2*y-1))+shift+spread
    elif map_id=='E04': x=[.38,.64,.64,.83][band]+{'watch':-.055,'hidden':-.005,'mother':.055}[kind]+spread
    elif map_id=='E05': x=.31+.23*math.sin(math.pi*y)+shift+spread
    elif map_id=='E06': x=[.32,.47,.62][(band+i)%3]+shift+spread*.5
    elif map_id=='M01': x=(.34 if (band+i)%2==0 else .49)+shift+spread
    elif map_id=='M02':
        wing=.22*(1-abs(2*y-1))
        x=.54+(-1 if (band+i)%2==0 else 1)*wing+{'watch':-.035,'hidden':.045,'mother':.17}[kind]+spread*.5
    elif map_id=='M03': x=.30+.26*(1-abs(2*y-1))+shift+spread
    elif map_id=='M04': x=[.40,.48,.57,.66][band]+{'watch':-.08,'hidden':.05,'mother':.22}[kind]+spread
    elif map_id=='M05': x=.34+.13*math.sin(math.pi*(y+.15))+shift+spread
    elif map_id=='M06': x=.30+.09*band+shift+spread
    else: raise ValueError(map_id)
    return max(.265,min(.95,x))


def redesign(layout):
    if layout['map_id'].startswith('H'):
        layout['topology_idea']='保留原复杂图的归一化点位、巢体类型和逐巢库存；本轮仅完善说明与覆盖渲染。'
        return
    map_id=layout['map_id']; roots=layout['roots']; req=layout['requested_bands']
    selected={k:sorted([r for r in roots if r['kind']==k],key=lambda r:r['source_id']) for k in ['mother','watch','hidden']}
    offset=Counter(); placed=[]
    for band in range(4):
        amounts={'mother':MOTHER_BANDS[map_id][band],
                 'watch':req['public_roots'][band]-MOTHER_BANDS[map_id][band],
                 'hidden':len(selected['hidden'])//4}
        for kind in ['watch','mother','hidden']:
            n=amounts[kind]
            for i in range(n):
                r=selected[kind][offset[kind]]; offset[kind]+=1
                local=.15+.70*(i+.5)/n
                local+= {'watch':-.045,'mother':.025,'hidden':.065}[kind]
                if map_id=='E04':
                    local=([.66,.76,.24,.32][band]+(i-(n-1)/2)*.10+{'watch':-.04,'mother':.04,'hidden':0}[kind])
                y=(band+local)/4
                r['x_norm']=x_position(map_id,kind,y,band,i,n); r['y_norm']=y
                r['x']=r['x_norm']*layout['size'][0];r['y']=y*layout['size'][1]
                r['swarm_stock']=0; placed.append(r)
        carriers=sorted([r for r in placed if int(r['y_norm']*4)==band and r['kind']!='mother'],key=lambda r:(r['kind'],r['source_id']))
        q,rem=divmod(req['swarm_stock'][band],len(carriers))
        for i,r in enumerate(carriers):r['swarm_stock']=q+(i<rem)
    layout['roots']=placed
    layout['topology_idea']=IDEAS[map_id]
    layout['revision']='按主题骨架重建；分区预算与档位总量不变'
