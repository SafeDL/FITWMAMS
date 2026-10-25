"""Render patent engineering figures from existing arrays; never execute a model."""
from pathlib import Path
import csv, gzip, hashlib, json, os
from collections import Counter, defaultdict

os.environ.setdefault('MPLCONFIGDIR','/tmp/fitwmams_patent_figure_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.interpolate import PchipInterpolator

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parents[1]
DATA=OUT/'data'
BR=ROOT/'results/hierarchical_world_model/bridge'
CACHE=ROOT/'results/highd_shared_training_data/highd_sequence_cache/sequence_cache'
SEARCH=ROOT/'results/risk_stratified_repair/development/diffusion_planner_functional/diffusion_planner/20260926/guided/round_1'
for p in [OUT,DATA,OUT/'_support']:p.mkdir(parents=True,exist_ok=True)

# Register the Chinese font explicitly for consistent PNG rendering.
font_manager.fontManager.addfont('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')

plt.rcParams.update({
    'font.family':['Liberation Serif','Droid Sans Fallback'],'font.size':10,'axes.labelsize':11,
    'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,
    'mathtext.fontset':'stix','axes.unicode_minus':False,
    'axes.spines.top':False,'axes.spines.right':False,
    'axes.linewidth':.75,'lines.linewidth':1.5,'grid.linewidth':.45,
    'savefig.dpi':300,'savefig.pad_inches':.05,
})
BLUE='#176A91'; ORANGE='#B65323'; GRAY='#666666'; BLACK='#202020'
DT=.04; T=np.arange(150)*DT; TA=T[:-1]
FIGURES=[]; SOURCES={}; QA=[]

def source(p):
    p=Path(p);st=p.stat();rel=str(p.relative_to(ROOT))
    if rel not in SOURCES:
        d={'path':rel,'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}
        if st.st_size<64*1024**2:
            h=hashlib.sha256()
            with p.open('rb') as f:
                for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
            d['sha256']=h.hexdigest()
        SOURCES[rel]=d
    return p

def loadjson(p):return json.loads(source(p).read_text())
def dumpcsv(stem,fields,rows):
    with (DATA/(stem+'.csv')).open('w',newline='') as f:
        w=csv.writer(f);w.writerow(fields);w.writerows(rows)

def style(ax,xlabel='时间（s）',ylabel=None):
    ax.set_xlabel(xlabel)
    if ylabel:ax.set_ylabel(ylabel)
    ax.tick_params(direction='out',length=3,width=.7)
    ax.grid(axis='both',color='#d2d2d2',alpha=.65,zorder=0)
    ax.set_axisbelow(True)

def legend(ax,**kw):
    return ax.legend(frameon=False,handlelength=2.1,**kw)

def save(fig,stem,title,caption,requirements,**meta):
    # No figure title, explanatory paragraph, subtitle or result claims in the artwork.
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for ax in fig.axes:
        for tick in ax.get_xticklabels()+ax.get_yticklabels():
            assert '□' not in tick.get_text()
    fig.savefig(OUT/f'{stem}.png',bbox_inches='tight',facecolor='white')
    FIGURES.append({'stem':stem,'title':title,'caption':caption,'requirements':requirements,**meta})
    plt.close(fig)

def polygons(s,dims):
    corners=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
    local=dims[:,None,:]*corners[None]/2
    angle=np.arctan2(s[:,3],s[:,2]);c=np.cos(angle)[:,None];v=np.sin(angle)[:,None]
    return np.stack([local[...,0]*c-local[...,1]*v,local[...,0]*v+local[...,1]*c],-1)+s[:,None,:2]

def lane_edges(maps,valid):
    present=valid.sum(-1)>=2
    centers=(maps[...,1]*valid).sum(-1)/valid.sum(-1).clip(1)
    widths=(maps[...,4]*valid).sum(-1)/valid.sum(-1).clip(1)
    return np.unique(np.round(np.r_[centers[present]-widths[present]/2,centers[present]+widths[present]/2],4))

def road(ax,states,dims,valid,maps,mapvalid,frame,focus,baseline=None,span=94):
    """Snapshot in a translated coordinate frame with unchanged metric scales."""
    origin=states[frame,0,0]
    edges=lane_edges(maps,mapvalid)
    ax.set_facecolor('#fafafa')
    for k,y in enumerate(edges):
        ax.axhline(y,color='#888888',lw=.65,ls='-' if k in (0,len(edges)-1) else '--',zorder=0)
    recent=np.arange(max(0,frame-25),frame+1)
    for i in np.flatnonzero(valid):
        color=BLACK if i==0 else BLUE if i==focus else '#b8b8b8'
        s=states[frame,i:i+1].copy();s[:,0]-=origin
        poly=polygons(s,dims[i:i+1])[0]
        if np.max(poly[:,0]) < -span/2 or np.min(poly[:,0])>span/2:continue
        ax.add_patch(Polygon(poly,closed=True,facecolor=color,edgecolor=BLACK,lw=.65,zorder=4))
        if i in (0,focus):
            ax.plot(states[recent,i,0]-origin,states[recent,i,1],color=color,lw=1.1,zorder=2)
    if baseline is not None:
        s=baseline[frame,focus:focus+1].copy();s[:,0]-=origin
        ax.add_patch(Polygon(polygons(s,dims[focus:focus+1])[0],closed=True,fill=False,edgecolor=ORANGE,lw=1.2,ls='--',zorder=5))
        ax.plot(baseline[recent,focus,0]-origin,baseline[recent,focus,1],ls='--',color=ORANGE,lw=1.2,zorder=2)
    ax.set_aspect('equal',adjustable='box')
    ax.set_xlim(-span/2,span/2)
    ys=states[frame,np.flatnonzero(valid),1]
    ax.set_ylim(ys.min()-2.8,ys.max()+2.8)
    ax.set_xlabel('相对主车纵向位置（m）');ax.set_ylabel('横向位置（m）')
    # Physical time is a necessary measurement label, not an explanatory caption.
    ax.set_title(rf'$t={T[frame]:.2f}\,\mathrm{{s}}$',loc='left',fontsize=9,pad=3)
    ax.tick_params(labelsize=9)

tracefile=source(BR/'intervention_traces.npz')
traces=np.load(tracefile,allow_pickle=False)
protocol=loadjson(BR/'intervention_cases.json')
case_specs={c['case']:c for c in protocol['cases']}
interventions=loadjson(BR/'interventions.json')
records=interventions['records']

def case(n,arm='takeover',seed=20261015):
    prefix=f'{n}_{arm}_{seed}/'
    x={k[len(prefix):]:traces[k] for k in traces.files if k.startswith(prefix)}
    assert x and x['states'].shape==(149,7,6)
    x['full_states']=np.concatenate([x['initial_states'][None],x['states']])
    assert np.isfinite(x['full_states']).all()
    return x

def event(n,actor,seed=20261015):
    r=next(r for r in records if r['case']==n and r['arm']=='takeover' and r['seed']==seed)
    ev=[e for e in r['events'] if e['actor']==actor]
    return r,(ev[0] if ev else None)

cache={k:np.load(source(CACHE/(k+'.npy')),mmap_mode='r') for k in [
    'agent_states','agent_valid','sequence_id','map_polylines','map_polyline_valid','vehicle_dimensions']}
manifest=loadjson(CACHE/'manifest.json')
assert manifest['history_frames']==25 and cache['agent_states'].shape[1]==174
ids=np.asarray(cache['sequence_id']).astype(str)

def natural(sid):
    row=int(np.flatnonzero(ids==sid)[0]);s=np.array(cache['agent_states'][row,24:174])
    return row,s,np.array(cache['agent_valid'][row,24]),np.array(cache['vehicle_dimensions'][row]),np.array(cache['map_polylines'][row]),np.array(cache['map_polyline_valid'][row])

def E01():
    sourcepath=source(ROOT/'results/evt/natural_segments.csv')
    rows=list(csv.DictReader(sourcepath.open()))
    eligible=[r for r in rows if int(r['num_strict_cutins'])>0 and int(r['num_slots_present_at_anchor'])>=3 and not int(r['untracked_risk_exceeds_tracked_peak'])]
    spec=sorted(eligible,key=lambda r:(-float(r['event_risk']),r['segment_id']))[0]
    sid=spec['segment_id'];row,s,valid,dims,maps,mv=natural(sid)
    slots=['same_front','same_rear','left_front','left_rear','right_front','right_rear']
    focus=slots.index(spec['peak_slot_name'])+1
    with np.load(source(ROOT/'results/evt/natural_risk_traces.npz')) as z:
        risk=z['risk_trace'][int(spec['risk_trace_row'])].copy()
    peak=int(spec['peak_instant_offset']);frames=sorted(set([0,peak,75,149]))
    fig=plt.figure(figsize=(10.7,5.4),layout='constrained');gs=fig.add_gridspec(3,2,height_ratios=[1,1,1.5])
    for k,f in enumerate(frames):road(fig.add_subplot(gs[k//2,k%2]),s,dims,valid,maps,mv,f,focus,span=82)
    ax=fig.add_subplot(gs[2,0]);ax.plot(T,risk,color=BLUE);style(ax,ylabel='事件风险')
    ax.axvline(T[peak],color=GRAY,ls=':',lw=1,label='瞬时风险峰值时刻');legend(ax,loc='upper left')
    gap=np.abs(s[:,focus,0]-s[:,0,0])-(dims[0,0]+dims[focus,0])/2
    ax=fig.add_subplot(gs[2,1]);ax.plot(T,gap,color=BLACK);style(ax,ylabel='纵向净距（m）')
    fig.legend(handles=[Patch(facecolor=BLACK,edgecolor=BLACK,label='主车'),Patch(facecolor=BLUE,edgecolor=BLACK,label='关键背景车'),Patch(facecolor='#b8b8b8',edgecolor=BLACK,label='其他背景车')],loc='outside upper center',ncol=3,frameon=False)
    dumpcsv('E01_natural_event',['time_s','prefix_event_risk','longitudinal_clearance_m'],zip(T,risk,gap))
    np.savez_compressed(DATA/'E01_states.npz',states=s,valid=valid,dimensions=dims,maps=maps,map_valid=mv,frames=frames)
    save(fig,'E01_natural_event','自然驾驶事件与关键风险时刻',
         '从存在严格切入、至少三个有效背景位置且未跟踪车辆未超过已跟踪峰值风险的片段中，按事件风险降序选取首例；道路视图按各时刻主车纵向位置平移，保持实际空间比例。risk_trace是截至当前时刻的峰值加暴露项，不是瞬时风险；虚线时刻取原始事件表的瞬时风险峰值。', ['E01'],event_id=sid,row=row,focus_slot=focus,frames=frames,event_risk=float(spec['event_risk']),instant_peak_frame=peak)

def E02():
    c=case(0,'tracker');sid=case_specs[0]['sequence_id'];row,s,valid,dims,maps,mv=natural(sid)
    assert np.allclose(s[0],c['initial_states'],rtol=0,atol=1e-10)
    j=case_specs[0]['actor'];actor=j+1
    with np.load(source(ROOT/'results/highd_shared_training_data/scenario_conditions/dataset.npz')) as z:
        index=int(np.flatnonzero(z['segment_id'].astype(str)==sid)[0]);constraint=z['trajectory_constraint'][index,j].reshape(3,2)
    knots=np.vstack([s[0,actor,:2],s[0,actor,:2]+constraint]);kt=np.array([0,2,4,5.96])
    baseline=PchipInterpolator(kt,knots,axis=0)(T)
    plan=np.vstack([s[0,actor,:2],c['plan'][:,j]])
    actual=c['full_states'][:,actor,:2]
    assert np.allclose(plan[[0,50,100,149]],knots,atol=3e-4)
    fig,axs=plt.subplots(2,2,figsize=(10.7,6.1),layout='constrained')
    series=[(s[:,actor,:2],'自然轨迹',BLACK,'-'),(baseline,'插值基准',GRAY,':'),(plan,'生成参考',ORANGE,'--'),(actual,'跟踪执行',BLUE,'-.')]
    for dim in range(2):
        ax=axs[0,dim];origin=s[0,actor,dim] if dim==0 else 0
        for values,label,color,ls in series:ax.plot(T,values[:,dim]-origin,color=color,ls=ls,label=label)
        ax.scatter(kt,knots[:,dim]-origin,s=24,marker='o',facecolors='white',edgecolors=BLACK,zorder=5,label='条件结点')
        style(ax,ylabel='纵向位移（m）' if dim==0 else '横向位置（m）')
        ax=axs[1,dim]
        for values,label,color,ls in [series[0],series[2],series[3]]:ax.plot(T,values[:,dim]-baseline[:,dim],color=color,ls=ls,label=label)
        ax.axhline(0,color='#bbbbbb',lw=.6);style(ax,ylabel='纵向参考偏差（m）' if dim==0 else '横向参考偏差（m）')
    h,l=axs[0,0].get_legend_handles_labels();fig.legend(h,l,loc='outside upper center',ncol=5,frameon=False)
    dumpcsv('E02_reference_execution',['time_s','natural_x_m','natural_y_m','pchip_x_m','pchip_y_m','plan_x_m','plan_y_m','actual_x_m','actual_y_m'],np.column_stack([T,s[:,actor,:2],baseline,plan,actual]))
    save(fig,'E02_reference_execution','稀疏结点、参考生成与实际执行',
         '案例0，背景位置1。参考来自既有真实结点条件下的确定性扩散计划；下排为各轨迹相对插值基准的位置差，不是预测误差。', ['E02'],case=0,event_id=sid,actor=j,seed=20261015,knots_s=kt.tolist())

def E03():
    with gzip.open(source(BR/'natural_records.json.gz'),'rt') as f:rs=json.load(f)
    grouped=defaultdict(list)
    for r in rs:
        if r['ADE_m'] is not None and r['actor_frames']>0:grouped[(r['row'],r['arm'])].append(float(r['ADE_m']))
    paired=sorted(set(row for row,arm in grouped if (row,'tracker') in grouped and (row,'takeover') in grouped))
    ade={arm:np.array([np.mean(grouped[(row,arm)]) for row in paired]) for arm in ['tracker','takeover']}
    fig,axs=plt.subplots(1,2,figsize=(10.7,3.9),layout='constrained')
    c=case(0);row,s,valid,_,_,_=natural(case_specs[0]['sequence_id']);err=np.linalg.norm(c['full_states'][:,valid,:2]-s[:,valid,:2],axis=-1)
    # Exclude the ego from the per-time background error.
    active=np.flatnonzero(valid);keep=active!=0;meanerr=err[:,keep].mean(-1)
    axs[0].plot(T,meanerr,color=BLUE,label='背景车平均位移误差');style(axs[0],ylabel='位移误差（m）');legend(axs[0])
    for arm,label,color,ls in [('tracker','参考跟踪',ORANGE,'--'),('takeover','响应执行',BLUE,'-')]:
        values=np.sort(ade[arm]);axs[1].step(values,np.arange(1,len(values)+1)/len(values),where='post',color=color,ls=ls,label=label)
    axs[1].set_xscale('log');style(axs[1],xlabel='场景平均位移误差（m）',ylabel='经验累计比例');axs[1].set_ylim(0,1.015);legend(axs[1],loc='lower right')
    dumpcsv('E03_natural_replay',['row','tracker_scene_ade_m','takeover_seed_mean_scene_ade_m'],[(row,ade['tracker'][i],ade['takeover'][i]) for i,row in enumerate(paired)])
    dumpcsv('E03_example_error',['time_s','background_mean_position_error_m'],zip(T,meanerr))
    save(fig,'E03_natural_replay','自然回放的轨迹保持',
         '左图为案例0的背景车逐时平均位移误差，主车不参与；右图为有背景车场景的场景平均误差经验分布，响应臂先在同场景内平均随机实现。自然报告包含原全量运行和后续回归复用，不属于重新盲测。', ['E03'],scene_count=len(paired),example_case=0)

def E04():
    fig,axs=plt.subplots(2,2,figsize=(10.7,5.6),layout='constrained')
    for col,n in enumerate([0,50]):
        a=case(n);b=case(n,'tracker');j=case_specs[n]['actor'];slot=j+1
        assert not a['takeover_owned'][:,j].any()
        for x,label,color,ls in [(b,'参考跟踪',ORANGE,'--'),(a,'响应执行',BLUE,'-')]:
            speed=np.linalg.norm(x['full_states'][:,slot,2:4],axis=-1)
            axs[0,col].plot(T,speed,color=color,ls=ls,label=label)
            axs[1,col].plot(TA,x['actions'][:,j,0],color=color,ls=ls,label=label)
        style(axs[0,col],ylabel='背景车速度（m/s）');style(axs[1,col],ylabel='纵向加速度（m/s²）')
        legend(axs[0,col],loc='best')
        dumpcsv(f'E04_case_{n}',['time_s','tracker_speed_mps','takeover_speed_mps','tracker_acceleration_mps2','takeover_acceleration_mps2','owned'],np.column_stack([TA,np.linalg.norm(b['full_states'][:-1,slot,2:4],axis=-1),np.linalg.norm(a['full_states'][:-1,slot,2:4],axis=-1),b['actions'][:,j,0],a['actions'][:,j,0],a['takeover_owned'][:,j]]))
    save(fig,'E04_reference_retention','无干预与自由行驶下的参考保持',
         '左列为无干预案例0，右列为自由行驶案例50；每列分别比较同一刺激下的参考跟踪与响应执行，所选关注车辆均未接管。两列不是相同初态的配对样本。', ['E04'],cases=[0,50],seed=20261015)

def intervention(n,j,stem,title,req,lateral=False):
    a=case(n);b=case(n,'tracker');slot=j+1;r,ev=event(n,j)
    assert ev is not None
    hand=ev['frame']*DT;onset=case_specs[n]['onset_s']
    frames=[max(0,ev['frame']-10),min(149,ev['frame']+18)]
    fig=plt.figure(figsize=(11.8,5.0),layout='constrained');gs=fig.add_gridspec(2,6,height_ratios=[.6,1.4])
    for col,f in enumerate(frames):
        road(fig.add_subplot(gs[0,3*col:3*col+3]),a['full_states'],a['dimensions'],a['valid'],a['maps'],a['map_valid'],f,slot,baseline=b['full_states'])
    axes=[fig.add_subplot(gs[1,k*2:k*2+2]) for k in range(3)]
    values=[]
    for x,label,color,ls in [(b,'参考跟踪',ORANGE,'--'),(a,'响应执行',BLUE,'-')]:
        s=x['full_states'];speed=np.linalg.norm(s[:,slot,2:4],axis=-1)
        gap=np.abs(s[:,slot,0]-s[:,0,0])-(x['dimensions'][slot,0]+x['dimensions'][0,0])/2
        middle=s[:,slot,1] if lateral else gap
        axes[0].plot(T,speed,color=color,ls=ls,label=label)
        axes[1].plot(T,middle,color=color,ls=ls,label=label)
        axes[2].plot(TA,x['actions'][:,j,0],color=color,ls=ls,label=label)
        values.append((speed,middle,x['actions'][:,j,0]))
    collision_time=(int(r['first_collision_frame'])+1)*DT if r['first_collision_frame']>=0 else None
    labels=['背景车速度（m/s）','横向位置（m）' if lateral else '纵向净距（m）','纵向加速度（m/s²）']
    for ax,ylabel in zip(axes,labels):
        style(ax,ylabel=ylabel);ax.axvline(onset,color=GRAY,ls=':',lw=.9);ax.axvline(hand,color=BLACK,ls='-.',lw=.9);ax.set_xlim(0,5.96)
        if collision_time is not None:ax.axvline(collision_time,color='#9b2834',ls='--',lw=.9)
    legend(axes[0],loc='best')
    handles=[Patch(facecolor=BLACK,edgecolor=BLACK,label='主车'),Patch(facecolor=BLUE,edgecolor=BLACK,label='关注背景车'),Line2D([],[],color=ORANGE,ls='--',label='参考跟踪'),Line2D([],[],color=GRAY,ls=':',label='干预开始'),Line2D([],[],color=BLACK,ls='-.',label='接管')]
    if collision_time is not None:handles.append(Line2D([],[],color='#9b2834',ls='--',label='首次碰撞'))
    fig.legend(handles=handles,loc='outside upper center',ncol=len(handles),frameon=False)
    dumpcsv(stem,['time_s','tracker_speed_mps','takeover_speed_mps','tracker_y_m' if lateral else 'tracker_longitudinal_clearance_m','takeover_y_m' if lateral else 'takeover_longitudinal_clearance_m','tracker_ax_mps2','takeover_ax_mps2','support','interaction_score'],np.column_stack([TA,values[0][0][:-1],values[1][0][:-1],values[0][1][:-1],values[1][1][:-1],values[0][2],values[1][2],a['takeover_score'][:,j],a['takeover_interaction_score'][:,j]]))
    save(fig,stem,title,
         '道路视图为响应臂实际状态，虚线包围框和轨迹为同一刺激下参考跟踪臂的关注车辆；下方为真实执行曲线。主车受脚本控制。首次碰撞时刻按碰撞后状态索引对应的物理时刻标记。', req,case=n,event_id=case_specs[n]['sequence_id'],actor=j,seed=20261015,handover_s=hand,onset_s=onset,any_collision=r['any_collision'],ego_collision=r['ego_collision'],first_collision_s=collision_time,snapshot_frames=frames)

def E08():
    n=28;j=case_specs[n]['actor'];slot=j+1;a=case(n);_,ev=event(n,j)
    assert ev['plan_maneuver_started']
    fig=plt.figure(figsize=(11.8,5.0),layout='constrained');gs=fig.add_gridspec(2,6,height_ratios=[.6,1.4])
    for col,f in enumerate([ev['frame']-8,ev['frame']+16]):road(fig.add_subplot(gs[0,3*col:3*col+3]),a['full_states'],a['dimensions'],a['valid'],a['maps'],a['map_valid'],f,slot,span=88)
    axes=[fig.add_subplot(gs[1,2*k:2*k+2]) for k in range(3)];s=a['full_states']
    axes[0].plot(T,s[:,slot,1],color=BLUE,label='实际横向位置')
    axes[0].plot(T,np.r_[a['initial_states'][slot,1],a['plan'][:,j,1]],color=ORANGE,ls='--',label='生成参考')
    axes[1].plot(T,s[:,slot,3],color=BLUE,label='实际横向速度')
    axes[2].plot(TA,a['actions'][:,j,1],color=BLUE,label='实际横摆角速度')
    r,_=event(n,j);collision_time=(r['first_collision_frame']+1)*DT if r['first_collision_frame']>=0 else None
    for ax,ylabel in zip(axes,['横向位置（m）','横向速度（m/s）','横摆角速度（rad/s）']):
        if collision_time is not None:ax.axvline(collision_time,color='#9b2834',ls='--',label='首次碰撞',lw=.9)
        style(ax,ylabel=ylabel);ax.axvline(ev['frame']*DT,color=BLACK,ls='-.',label='接管');legend(ax,loc='best')
    fig.legend(handles=[Patch(facecolor=BLACK,edgecolor=BLACK,label='主车'),Patch(facecolor=BLUE,edgecolor=BLACK,label='关注背景车'),Patch(facecolor='#b8b8b8',edgecolor=BLACK,label='其他背景车')],loc='outside upper center',ncol=3,frameon=False)
    dumpcsv('E08_maneuver_handover',['time_s','actual_y_m','reference_y_m','actual_vy_mps','yaw_rate_rps','owned'],np.column_stack([TA,s[:-1,slot,1],np.r_[a['initial_states'][slot,1],a['plan'][:-1,j,1]],s[:-1,slot,3],a['actions'][:,j,1],a['takeover_owned'][:,j]]))
    save(fig,'E08_maneuver_handover','已开始换道的接管与机动承接',
         '案例28，关注背景位置1；接管记录在2.60秒记载换道已开始，进度0.854。曲线显示实际位置、速度及控制动作，保留该回合后续碰撞时刻；未补画未保存的连续进度或重置换道对照。', ['E08'],case=n,seed=20261015,event_id=case_specs[n]['sequence_id'],actor=j,handover_s=ev['frame']*DT,logged_progress=ev['plan_progress'],first_collision_s=collision_time)

def E09():
    a=case(1);j=1;_,ev=event(1,j);owned=a['takeover_owned'][:,j];changed=a['takeover_motion_changed'][:,j]
    firsto=int(np.flatnonzero(owned)[0]);firstc=int(np.flatnonzero(changed)[0]);assert (firsto,firstc)==(96,97)
    fig,axs=plt.subplots(3,1,figsize=(8.2,6.7),sharex=True,layout='constrained')
    axs[0].step(TA,owned.astype(int),where='post',color=BLUE,label='接管标记')
    axs[0].step(TA,changed.astype(int),where='post',color=ORANGE,ls='--',label='运动改变标记')
    axs[0].set_ylim(-.08,1.12);axs[0].set_yticks([0,1]);legend(axs[0],loc='lower right')
    delta=a['actions'][:,j]-a['base_actions'][:,j]
    axs[1].plot(TA,delta[:,0],color=BLUE,label='实际动作－基准动作');legend(axs[1],loc='best')
    axs[2].plot(TA,delta[:,1],color=BLUE)
    for ax,ylabel in zip(axs,['状态','纵向加速度差（m/s²）','横摆角速度差（rad/s）']):
        style(ax,ylabel=ylabel);ax.axvline(firsto*DT,color=GRAY,ls=':',lw=.9);ax.axvline(firstc*DT,color=GRAY,ls='--',lw=.9)
    axs[-1].set_xlim((firsto-9)*DT,(firstc+16)*DT)
    dumpcsv('E09_handover_timing',['controller_call_time_s','owned','realised_motion_changed','ax_difference_mps2','yaw_rate_difference_rps'],np.column_stack([TA,owned,changed,delta]))
    save(fig,'E09_handover_timing','接管状态与实际运动改变的逐步记录',
         '案例1，关注背景位置1；横轴为控制器调用时标。运动改变标记检查上一物理步已执行动作，两个标记的首次记录索引差不解释为物理响应延迟。', ['E09'],case=1,actor=j,seed=20261015,first_owned_index=firsto,first_motion_changed_index=firstc)

def E11():
    p=source(SEARCH/'search.jsonl');rs=[json.loads(l) for l in p.open() if l.strip()]
    unique={}
    for r in rs:unique.setdefault(r['instance_hash'],r)
    rs=list(unique.values());counts=Counter(r['interaction_type'] for r in rs)
    types=[t for t,_ in counts.most_common()]
    names={'front_tracking':'前方·跟踪','rear_tracking':'后方·跟踪','adjacent_tracking':'邻侧·跟踪','front_owned':'前方·接管','rear_owned':'后方·接管','adjacent_owned':'邻侧·接管','offroad_right':'右侧驶出道路','offroad_left':'左侧驶出道路','free_road':'自由道路'}
    byroot=Counter(r['root_sample_id'] for r in rs);roots=[r for r,_ in sorted(byroot.items(),key=lambda x:(-x[1],x[0]))[:12]]
    matrix=np.zeros((len(types),len(roots)),int)
    for r in rs:
        if r['root_sample_id'] in roots:matrix[types.index(r['interaction_type']),roots.index(r['root_sample_id'])]+=1
    fig,axs=plt.subplots(1,2,figsize=(11.8,4.5),layout='constrained',gridspec_kw={'width_ratios':[1,1.45]})
    y=np.arange(len(types));axs[0].barh(y,[counts[t] for t in types],color='#c4d7df',edgecolor=BLACK,lw=.6)
    axs[0].set_yticks(y,[names[t] for t in types]);axs[0].invert_yaxis();style(axs[0],xlabel='去重实例数',ylabel='交互类别');axs[0].xaxis.set_major_locator(MaxNLocator(integer=True,nbins=5))
    im=axs[1].imshow(matrix,aspect='auto',cmap='Greys',vmin=0)
    axs[1].set_xticks(np.arange(len(roots)),[f'来源{k+1:02d}' for k in range(len(roots))],rotation=55,ha='right')
    axs[1].set_yticks(y,[names[t] for t in types]);axs[1].set_xlabel('独立抽样来源');axs[1].set_ylabel('交互类别')
    for i in range(len(types)):
        for j in range(len(roots)):
            if matrix[i,j]:axs[1].text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=8,color='white' if matrix[i,j]>matrix.max()*.52 else BLACK)
    fig.colorbar(im,ax=axs[1],label='去重实例数',shrink=.8,pad=.025)
    dumpcsv('E11_interaction_counts',['interaction_type','label_zh','unique_instances'],[(t,names[t],counts[t]) for t in types])
    dumpcsv('E11_source_matrix',['interaction_type']+[f'root_{j+1:02d}' for j in range(len(roots))],[(t,*matrix[i]) for i,t in enumerate(types)])
    (DATA/'E11_source_ids.json').write_text(json.dumps({f'来源{i+1:02d}':r for i,r in enumerate(roots)},ensure_ascii=False,indent=2))
    save(fig,'E11_interaction_sources','实际交互类别与抽样来源',
         '使用已完成开发搜索的5000条执行记录，按实例哈希去重；左图为全部去重实例类别，右图按实例数选取前12个来源，同行来源跨类别统一编号。类别中的“接管”是控制权状态。此图为观察计数，不是联合名额分配或完整失效域识别结果。', ['E11'],raw_records=5000,unique_instances=len(rs),source_ranking='instance_count_desc_then_root_id',root_count=len(byroot))

def E14():
    d=loadjson(BR/'latency.json');fig,axs=plt.subplots(1,2,figsize=(10.7,4.0),layout='constrained')
    rows=[];labels=['P50','P95','P99','最大值'];markers=['o','s','^'];styles=['-','--','-.'];colors=[BLACK,BLUE,ORANGE]
    for key,marker,ls,color in zip(['0','3','6'],markers,styles,colors):
        rec=d['cpu'][key]
        for col,field in enumerate(['bridge_ms_quantiles','world_ms_quantiles']):
            a=np.asarray(rec[field]);axs[col].plot(np.arange(4),a,marker=marker,ls=ls,color=color,label=f'{key}辆背景车',ms=5)
            for q,val in zip(labels,a):rows.append([int(key),field,q,float(val),rec['steps']])
    for ax,ylabel in zip(axs,['桥接更新耗时（ms）','世界更新耗时（ms）']):
        ax.set_xticks(np.arange(4),labels);style(ax,xlabel='统计量',ylabel=ylabel);legend(ax,loc='upper left')
    axs[1].axhline(40,color=GRAY,ls=':',lw=1.1,label='物理步长');legend(axs[1],loc='upper left');axs[1].set_ylim(0,43)
    dumpcsv('E14_runtime',['background_vehicles','timing_component','statistic','time_ms','physical_steps'],rows)
    save(fig,'E14_runtime','不同背景车辆数量下的执行耗时',
         '分别绘制桥接与完整世界更新分位统计。各配置2831步，共8493步；桥接计时包含于世界更新，不相加。不包含ADS计算，40毫秒线为物理步长。', ['E14'],physical_steps=sum(x['steps'] for x in d['cpu'].values()),ADS_compute_included=d['ADS_compute_included'],physical_step_ms=d['physical_step_ms'])

if __name__=='__main__':
    E01();E02();E03();E04()
    intervention(1,1,'E05_braking_response','主车制动后的跟驰响应',['E05'])
    intervention(3,1,'E06_leader_release','前车离开原车道后的背景响应',['E06'])
    intervention(3,5,'E07_lateral_response','目标车道车辆的交互响应',['E07'],lateral=True)
    E08();E09();E11();E14()
    skipped=[
        {'requirement':'E10','reason':'已有一般生成搜索父子记录，但固定自然事件及非选中输入一致性证据尚未完成，不绘制固定事件风险配对图。'},
        {'requirement':'E12','reason':'所核对搜索档案保留输入及执行摘要，缺少用于该图的原始完整逐帧闭环轨迹；按用户要求跳过，不启动重放。'},
        {'requirement':'E13','reason':'缺少原运行与重放的成对轨迹证据；跳过，不启动重放。'},
        {'requirement':'E15','reason':'跨类别联合名额分配尚无完整实际分配与后续调用日志。'},
        {'requirement':'E16','reason':'尚无本案完整联合流程的同预算比较和消融结果。'},
    ]
    provenance={'date':'2026-10-24','format_updated':'2026-10-25','figures':FIGURES,'skipped':skipped,'sources':list(SOURCES.values()),'model_executions_started':False,'training_started':False,'rendering':'matplotlib from existing numeric outputs','font':'Droid Sans Fallback (Chinese) / Liberation Serif (Latin and digits)','formats':['png_300dpi'],'script':str(Path(__file__).relative_to(ROOT))}
    (OUT/'manifest.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
    md=['# 专利工程实例附图\n','绘制日期：2026年10月24日；交付格式更新：2026年10月25日。图片由已有数值结果整理生成，未新增模型采样、仿真、训练或重放。图内保留中文坐标、必要图例及物理时刻，不嵌入解释段落；用途与来源在本文件说明。\n','每图提供300 dpi PNG。\n','## 已绘图片\n']
    for f in FIGURES:
        stem=f['stem'];md.extend([f'### {stem}：{f["title"]}\n',f'[PNG]({stem}.png)\n',f['caption']+'\n','数据身份：`'+json.dumps({k:v for k,v in f.items() if k not in ['stem','title','caption','requirements']},ensure_ascii=False)+'`。\n'])
    md.append('## 跳过项目\n')
    for s in skipped:md.append(f'- **{s["requirement"]}**：{s["reason"]}\n')
    md.extend(['\n## 来源与复现\n','完整来源、数据身份及取材范围见[manifest.json](manifest.json)，成图数据见[data/](data/)，绘图程序见[scripts/draw_engineering_patent.py](scripts/draw_engineering_patent.py)。程序只读取既有数组与报告，不调用模型或仿真器。\n','对应根目录[图片需求](../../../专利工程实例附图需求.md)。未将一般生成搜索的组件结果改标为本案联合流程的已完成效果。\n'])
    (OUT/'README.md').write_text('\n'.join(md))
    print(json.dumps({'figures':len(FIGURES),'skipped':[s['requirement'] for s in skipped],'output':str(OUT)},ensure_ascii=False))
