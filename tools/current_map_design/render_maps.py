"""Render game-rule tower circles and a separately labelled engine diagnostic."""
import argparse
from functools import lru_cache
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

BASE=Path(__file__).resolve().parent
INK='#203444'; MUTED='#63798b'; COLORS={9400:'#d88525',9600:'#2877bc',9500:'#007f78'}
TIER_COLORS={'easy':'#168779','medium':'#367bb6','hard':'#8460a8'}

# The complex tier is 100 km wide; all maps use the same pixels-per-kilometre scale.
REFERENCE_WIDTH_KM=100.0


@lru_cache(maxsize=None)
def font(size,bold=False):
    return ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc' if bold else 'C:/Windows/Fonts/msyh.ttc',size)


def text(draw,xy,value,size=24,color=INK,bold=False,anchor=None):
    draw.text(xy,str(value),font=font(size,bold),fill=color,anchor=anchor)


def paragraph(draw,xy,value,width,size=22,color=MUTED):
    x,y=xy; line=''
    for ch in value+'\n':
        if ch=='\n' or (line and draw.textlength(line+ch,font=font(size))>width):
            text(draw,(x,y),line,size,color);y+=size+10;line='' if ch=='\n' else ch
        else:line+=ch
    return y


def symbol(draw,x,y,kind,r=8):
    color=COLORS[kind]
    if kind==9400:draw.ellipse((x-r,y-r,x+r,y+r),fill=color,outline='white',width=2)
    elif kind==9600:draw.rectangle((x-r,y-r,x+r,y+r),fill=color,outline='white',width=2)
    else:
        points=[(x,y-r-2),(x+r+2,y),(x,y+r+2),(x-r-2,y)]
        draw.polygon(points,fill=color);draw.line(points+[points[0]],fill='white',width=2)


def coverage_grid(scenario, pixel_width, pixel_height, altitude):
    """Use pixel-center positions and the exact current ECEF XY predicate."""
    profile=scenario['imagineProfile']; area=profile['mapArea']
    lon=np.deg2rad(area['lonMin']+(np.arange(pixel_width)+.5)/pixel_width*(area['lonMax']-area['lonMin']))[None,:]
    lat=np.deg2rad(area['latMax']-(np.arange(pixel_height)+.5)/pixel_height*(area['latMax']-area['latMin']))[:,None]
    f=1/298.257223563; e2=f*(2-f); n=6378137/np.sqrt(1-e2*np.sin(lat)**2)
    xx=(n+altitude)*np.cos(lat)*np.cos(lon); yy=(n+altitude)*np.cos(lat)*np.sin(lon)
    coverage=np.zeros((pixel_height,pixel_width),dtype=np.uint8)
    for wrapper in profile['entityList']:
        entity=wrapper['entity']
        if entity['entityType']!=44000:continue
        p=entity['posEcf']
        coverage+=((xx-p['x'])**2+(yy-p['y'])**2<=20000**2).astype(np.uint8)
    return coverage


def map_geometry(info,size):
    width_km,height_km=info['logical_map_size']
    scale=size[0]/REFERENCE_WIDTH_KM
    return round(width_km*scale),round(height_km*scale),scale


def tower_circles(scenario,info,size):
    """Public rule illustration; this is NOT the engine's ECEF XY predicate."""
    area=scenario['imagineProfile']['mapArea'];bw,bh,scale=map_geometry(info,size)
    towers=sorted((w['entity'] for w in scenario['imagineProfile']['entityList']
                   if w['entity']['entityType']==44000),key=lambda e:e['id'])
    return [{'id':e['id'],'parent_id':e['parentId'],'label':f'T{i+1:02}',
             'x':(e['lla']['x']-area['lonMin'])/(area['lonMax']-area['lonMin'])*bw,
             'y':(area['latMax']-e['lla']['y'])/(area['latMax']-area['latMin'])*bh,
             'radius':info['detection']['range_m']/1000*scale} for i,e in enumerate(towers)]


def nominal_coverage_grid(scenario,info,size):
    bw,bh,_=map_geometry(info,size)
    xx=np.arange(bw)[None,:]+.5;yy=np.arange(bh)[:,None]+.5
    count=np.zeros((bh,bw),dtype=np.uint8)
    for c in tower_circles(scenario,info,size):
        count+=((xx-c['x'])**2+(yy-c['y'])**2<=c['radius']**2).astype(np.uint8)
    return count


def board(scenario,info,size=(1100,880),public=False,labels=True,view='rules'):
    w,h=info['logical_map_size'];bw,bh,scale=map_geometry(info,size)
    area=scenario['imagineProfile']['mapArea']
    im=Image.new('RGBA',(bw,bh),'#f9fcff'); d=ImageDraw.Draw(im)
    d.rectangle((0,0,bw-1,bh-1),fill='#f9fcff')
    d.rectangle((0,0,.2*bw,bh-1),fill='#f9dcd8')
    d.rectangle((.2*bw,0,.24*bw,bh-1),fill='#f9edc9')
    cov=(nominal_coverage_grid(scenario,info,size) if view=='rules' else
         coverage_grid(scenario,bw,bh,info['detection']['slice_altitude_m']))
    rgba=np.zeros((bh,bw,4),dtype=np.uint8)
    rgba[:,:,:3]=(65,129,207)
    rgba[:,:,3]=np.minimum(cov.astype(np.uint16)*13,52).astype(np.uint8)
    im.alpha_composite(Image.fromarray(rgba),dest=(0,0))
    if view=='rules':
        layer=Image.new('RGBA',(bw,bh));ld=ImageDraw.Draw(layer)
        for c in tower_circles(scenario,info,size):
            x,y,r=c['x'],c['y'],c['radius']
            ld.ellipse((x-r,y-r,x+r,y+r),outline=(64,127,177,115),width=1)
        im.alpha_composite(layer)
    else:
        mask=Image.fromarray((cov>0).astype(np.uint8)*255)
        edge=np.asarray(mask.filter(ImageFilter.MaxFilter(3)))-np.asarray(mask.filter(ImageFilter.MinFilter(3)))
        outline=np.zeros((bh,bw,4),dtype=np.uint8);outline[:,:,:3]=(47,105,174);outline[:,:,3]=np.where(edge>0,150,0)
        im.alpha_composite(Image.fromarray(outline))
    d=ImageDraw.Draw(im)
    # The shared deployment area is shown as one region; placement is coordinate-based.
    d.rectangle((0,0,.2*bw,bh-1),outline='#b77d77',width=2 if labels else 1)
    # Coordinate grid follows the labelled kilometre axes.
    for x_km in np.arange(w/5,w,w/5):
        x=x_km/w*bw
        if x>.24*bw:d.line((x,0,x,bh),fill='#e2e7ec',width=1)
    for y_km in np.arange(h/4,h,h/4):
        y=bh-y_km/h*bh
        d.line((.24*bw,y,bw,y),fill='#e2e7ec',width=1)
    area=scenario['imagineProfile']['mapArea']; placed=[];radius=8 if labels else 4
    for wrapper in scenario['imagineProfile']['entityList']:
        e=wrapper['entity'];t=e['entityType']
        if t not in COLORS or (public and t==9500):continue
        ax=(e['lla']['x']-area['lonMin'])/(area['lonMax']-area['lonMin'])*bw
        ay=(area['latMax']-e['lla']['y'])/(area['latMax']-area['latMin'])*bh
        gx,gy=ax,ay
        offsets=[(0,0)]+[(math.cos(i*math.pi/4)*radius*3*k,math.sin(i*math.pi/4)*radius*3*k) for k in [1,2,3] for i in range(8)]
        for dx,dy in ([(0,0)] if t==9600 else offsets):
            x,y=ax+dx,ay+dy
            if .24*bw+radius<=x<=bw-radius and radius<=y<=bh-radius and all(math.hypot(x-u,y-v)>2*radius+3 for u,v in placed):
                gx,gy=x,y;break
        if (gx,gy)!=(ax,ay):
            d.line((ax,ay,gx,gy),fill=COLORS[t]);d.ellipse((ax-2,ay-2,ax+2,ay+2),fill=COLORS[t])
        symbol(d,gx,gy,t,radius);placed.append((gx,gy))
    label_boxes=[]
    for c in tower_circles(scenario,info,size):
        x,y=c['x'],c['y']
        d.line((x,y-radius-7,x,y+radius),fill='#164e79',width=2)
        d.line((x-4,y-radius-5,x+4,y-radius-5),fill='#164e79',width=2)
        if labels:
            width=d.textlength(c['label'],font=font(16))
            candidates=[(x+dx,y+dy) for gap in (14,30,48,68,90)
                        for dx,dy in ((gap,-25),(gap,6),(-width-gap,-25),(-width-gap,6),(-width/2,-gap-25),(-width/2,gap))]
            for tx,ty in candidates:
                box=d.textbbox((tx,ty),c['label'],font=font(16))
                padded=(box[0]-3,box[1]-3,box[2]+3,box[3]+3)
                if not (2<=padded[0] and 2<=padded[1] and padded[2]<bw-2 and padded[3]<bh-2):continue
                obstacles=label_boxes+[(px-11,py-18,px+11,py+11) for px,py in placed]
                if any(padded[0]<r and padded[2]>l and padded[1]<b and padded[3]>t for l,t,r,b in obstacles):continue
                break
            else:
                raise ValueError(f'No clear tower label position: {info["scenario_id"]} {c["label"]}')
            label_boxes.append(padded)
            if math.hypot((box[0]+box[2])/2-x,(box[1]+box[3])/2-y)>45:
                d.line((x,y,(box[0]+box[2])/2,(box[1]+box[3])/2),fill='#8daec9',width=1)
            d.rectangle((box[0]-2,box[1]-1,box[2]+2,box[3]+1),fill='#ffffff')
            text(d,(tx,ty),c['label'],16,'#164e79')
    d.rectangle((0,0,bw-1,bh-1),outline='#506a7a',width=2)
    return im.convert('RGB'),cov


def render_one(directory,public=False,view='rules'):
    scenario=json.loads((directory/'scenario.json').read_text(encoding='utf-8'))
    info=json.loads((directory/'case_info.json').read_text(encoding='utf-8'))
    w,h=info['logical_map_size']; color=TIER_COLORS[info['difficulty']]
    bw,bh,_=map_geometry(info,(1100,880))
    map_top=190
    page_width=max(1100,bw+260);page_height=map_top+bh+55
    im=Image.new('RGB',(page_width,page_height),'#f4f7fa');d=ImageDraw.Draw(im)
    d.rectangle((0,0,page_width,12),fill=color)
    # Keep the public heading neutral. Internal topology names are implementation
    # metadata and are deliberately omitted because they are not part of the
    # difficulty definition shown to participants.
    text(d,(65,34),f'{info["scenario_id"]}  {info["difficulty_name"]}档地图',44,bold=True)
    text(d,(page_width-65,50),'引擎判定核验图' if view=='engine' else ('公开规则示意图' if public else '组织方规则示意图 · 含暗巢'),22,color,anchor='ra')
    text(d,(65,103),f'{info["difficulty_name"]} / {info["distribution"]} / {w:g} × {h:g} 公里 / 左侧共享部署区',24,MUTED)
    board_image,cov=board(scenario,info,public=public,view=view)
    im.paste(board_image,(85,map_top));d=ImageDraw.Draw(im)
    axis_left,axis_top,axis_bottom=85,map_top,map_top+bh
    d.line((axis_left,axis_bottom,axis_left+bw+18,axis_bottom),fill=INK,width=2)
    d.polygon(((axis_left+bw+18,axis_bottom),(axis_left+bw+7,axis_bottom-5),(axis_left+bw+7,axis_bottom+5)),fill=INK)
    d.line((axis_left,axis_bottom,axis_left,axis_top-18),fill=INK,width=2)
    d.polygon(((axis_left,axis_top-18),(axis_left-5,axis_top-7),(axis_left+5,axis_top-7)),fill=INK)
    for i in range(6):
        x=axis_left+bw*i/5
        d.line((x,axis_bottom,x,axis_bottom+6),fill=INK,width=2)
        text(d,(x,axis_bottom+9),f'{w*i/5:g}',16,MUTED,anchor='mt')
    for i in range(5):
        y=axis_bottom-bh*i/4
        d.line((axis_left-6,y,axis_left,y),fill=INK,width=2)
        text(d,(axis_left-10,y),f'{h*i/4:g}',16,MUTED,anchor='rm')
    text(d,(axis_left+bw+27,axis_bottom),'X 坐标（公里）',18,INK,anchor='lm')
    text(d,(axis_left,axis_top-27),'Y 坐标（公里）',18,INK,anchor='mb')
    text(d,(85+bw*.62,map_top-23),'视野塔逐塔视野圈 · 淡蓝色' if view=='rules' else '当前引擎位置判定切片',22,'#286da8',anchor='mb')
    return im,{'map':info['scenario_id'],'coverage_fraction':float(np.mean(cov>0)),
               'view':view,'meaning':'projected circular rule illustration' if view=='rules' else 'actual ECEF XY predicate slice',
               'slice_altitude_m':None if view=='rules' else info['detection']['slice_altitude_m'],'raster_size':list(cov.shape[::-1])}


def overview(directories):
    im=Image.new('RGB',(1800,2450),'#f4f7fa');d=ImageDraw.Draw(im)
    text(d,(45,28),'人族远征 虫巢攻防 · 20 图与视野塔范围',38,bold=True)
    text(d,(45,89),'蓝方血量 ×1.5 · 均匀 8 图 : 部分不均匀 12 图 = 4 : 6 · 组织方规则示意',23,MUTED)
    for i,path in enumerate(directories):
        scenario=json.loads((path/'scenario.json').read_text(encoding='utf-8'));info=json.loads((path/'case_info.json').read_text(encoding='utf-8'))
        x=40+(i%4)*430;y=148+(i//4)*430
        d.rounded_rectangle((x,y,x+412,y+412),radius=12,fill='white')
        text(d,(x+15,y+12),f'{info["scenario_id"]} {info["difficulty_name"]}档地图',22,bold=True)
        b,_=board(scenario,info,size=(380,304),labels=False);im.paste(b,(x+15,y+57))
        text(d,(x+15,y+376),f'{info["distribution"]} | 自爆虫 {info["interceptor_total"]}',19,MUTED)
    text(d,(45,2285),'淡蓝色圆：逐座哨卫巢视野塔的 20 千米视野范围，重叠处略加深。',23,'#286da8')
    text(d,(45,2340),'橙圆 母巢 / 蓝方块＋塔标 哨卫巢与视野塔 / 绿菱形 暗巢（公开图隐藏）',21,MUTED)
    text(d,(45,2390),'圆形为赛题规则示意；当前引擎 ECEF X/Y 判定差异另附核验图，不作为圆形已实现的证明。',21,MUTED)
    return im


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cases',type=Path,default=BASE/'cases')
    args=parser.parse_args();dirs=[]
    for tier in ['easy','medium','hard']:dirs.extend(sorted(p.parent for p in (args.cases/tier).glob('*/scenario.json')))
    results=[]
    for path in dirs:
        image,report=render_one(path);image.save(path/'map_info.png');results.append(report)
        image,_=render_one(path,public=True);image.save(path/'map_public.png')
        image,report=render_one(path,view='engine');image.save(path/'map_engine_check.png')
        results.append(report)
    overview(dirs).save(BASE/'20图拓扑与探测总览.png')
    (BASE/'coverage_report.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Rendered {len(dirs)} scenarios and public views')


if __name__=='__main__':main()
