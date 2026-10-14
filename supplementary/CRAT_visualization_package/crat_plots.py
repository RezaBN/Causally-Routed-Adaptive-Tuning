"""Publication plots: data labels only; descriptions are exported by the caller."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgb, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

PAIRS=['EN-FR','EN-FA','EN-ZH']
MODELS=['Llama 2','Mistral']
METRICS=['BLEU','chrF','COMET']
SETTINGS=[(m,p) for m in MODELS for p in PAIRS]
METHODS=['Full SFT','Targeted SFT','LoRA','rsLoRA','DoRA','CRAT']
VARIANTS=['no_kl','no_causal','no_morph','no_sparse','no_integration','static_role']
VARIANT_LABELS=['No KL loss','No causal-prior loss','No morphology loss','No sparsity loss','No integration','Static roles']
ROLES=['source','indicator','position','integration']
PARAMS=['lambda_kl','lambda_causal','lambda_morph','lambda_sparse','kl_temperature','integration_rank','heads_per_direction','top_k']
PARAM_LABELS=['KL coefficient','Causal coefficient','Morphology coefficient','Sparsity coefficient','KL temperature','Integration rank','Heads per direction','Evaluation top k']
MARKERS={'Full SFT':'s','Targeted SFT':'v','LoRA':'o','rsLoRA':'D','DoRA':'P','CRAT':'*'}


def axes(e,ident,rows,cols,size,**kwargs):
    return plt.subplots(rows,cols,figsize=e.style.size(ident,size),squeeze=False,**kwargs)


def colormap(e,diverging=False):
    c=e.style.colors
    colors=[c['negative'],c['neutral'],c['positive']] if diverging else [c['sequential_low'],c['sequential_high']]
    return LinearSegmentedColormap.from_list('crat_diverging' if diverging else 'crat_sequential',colors)


def heatmap(e,fig,ax,values,row_labels,col_labels,labels=None,signed=False,colorbar=True):
    values=np.asarray(values,dtype=float)
    finite=values[np.isfinite(values)]
    if not len(finite):raise ValueError('Cannot plot a heatmap with no reported values.')
    if signed:
        limit=max(np.max(np.abs(finite)),1e-12)
        norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit)
    else:
        norm=plt.Normalize(vmin=min(0,float(finite.min())),vmax=max(float(finite.max())*1.03,1e-12))
    cmap=colormap(e,signed)
    im=ax.imshow(np.ma.masked_invalid(values),aspect='auto',cmap=cmap,norm=norm)
    ax.set_yticks(range(len(values)),row_labels)
    ax.set_xticks(range(values.shape[1]),col_labels)
    ax.tick_params(length=0,pad=3)
    for i in range(len(values)):
        for j in range(values.shape[1]):
            value=values[i,j]
            label=labels[i][j] if labels is not None else (f'{value:.3f}' if np.isfinite(value) else 'NR')
            rgb=cmap(norm(value))[:3] if np.isfinite(value) else (1,1,1)
            color='white' if np.dot(rgb,[.299,.587,.114])<.48 else e.style.colors['text']
            ax.text(j,i,label,ha='center',va='center',fontsize=e.style.annotation,color=color)
    for spine in ax.spines.values():spine.set_visible(False)
    if colorbar:
        cb=fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.055,pad=.12,aspect=25)
        cb.ax.tick_params(labelsize=e.style.config['tick_font_size'],length=2)
        cb.locator=MaxNLocator(nbins=3);cb.update_ticks()
    return im


def sd_or_none(value):
    return float(value) if pd.notna(value) and value>=0 else None


def difference_text(value,metric):
    if 0<abs(value)<(.0001 if metric=='COMET' else .001):return f'{value:+.7f}' if metric=='COMET' else f'{value:+.6f}'
    return f'{value:+.5f}' if metric=='COMET' else f'{value:+.3f}'


def margins(ident,data,e):
    g=data['comparison']['G']
    fig,axs=axes(e,ident,1,3,(7,3.45))
    for j,(ax,metric) in enumerate(zip(axs.flat,METRICS)):
        rows=[g[(g.model==model)&(g.pair==pair)&(g.metric==metric)].iloc[0] for model,pair in SETTINGS]
        vals=np.array([r.CRAT_minus_best for r in rows])[:,None]
        labels=[[difference_text(r.CRAT_minus_best,metric)+'\n('+r.best_baseline+')'] for r in rows]
        heatmap(e,fig,ax,vals,[m+'  '+p for m,p in SETTINGS] if j==0 else ['']*6,[''],labels,True)
        ax.set_title(metric,fontweight='bold',pad=4)
        ax.axhline(2.5,color=e.style.colors['reference'],lw=.7)
    e.save_figure(ident,fig)


def comet(ident,data,e):
    d=data['comparison']['D'];c=e.style.colors
    fig,axs=axes(e,ident,2,3,(7,4.25))
    for n,(ax,(model,pair)) in enumerate(zip(axs.flat,SETTINGS)):
        g=d[(d.model==model)&(d.pair==pair)].set_index('method')
        for i,method in enumerate(METHODS):
            r=g.loc[method]
            ax.errorbar(r.COMET,i,xerr=sd_or_none(r.COMET_sd),fmt=MARKERS[method],
                color=c['methods'][method],markersize=e.style.config['marker_size']*(1.7 if method=='CRAT' else 1),
                capsize=1.6,elinewidth=.65,zorder=4)
        ax.set_yticks(range(6),METHODS if n%3==0 else ['']*6)
        ax.set_ylim(5.55,-.55);ax.grid(axis='x',zorder=0)
        ax.set_title(model+'  '+pair,pad=4)
        ax.set_xlabel('COMET')
        ax.ticklabel_format(axis='x',style='plain',useOffset=False)
        ax.xaxis.set_major_locator(MaxNLocator(3))
    e.save_figure(ident,fig)


def quality_parameters(ident,data,e):
    d=data['comparison']['D'];c=e.style.colors
    fig,axs=axes(e,ident,2,3,(7,4.3))
    for ax,(model,pair) in zip(axs.flat,SETTINGS):
        g=d[(d.model==model)&(d.pair==pair)].set_index('method')
        for method in METHODS:
            r=g.loc[method]
            ax.errorbar(r.trainable_params/1e6,r.BLEU,yerr=sd_or_none(r.BLEU_sd),fmt=MARKERS[method],
                color=c['methods'][method],markersize=e.style.config['marker_size']*(1.8 if method=='CRAT' else 1),
                capsize=1.5,elinewidth=.65,zorder=6 if method=='CRAT' else 3)
        ax.set_xscale('log');ax.set_xlim(28,11000)
        ax.set_xticks([40,200,1000,7000],['40','200','1,000','7,000']);ax.minorticks_off()
        ax.set_xlabel('Trainable parameters (M)');ax.set_ylabel('BLEU')
        ax.set_title(model+'  '+pair+(' †' if (model,pair)==('Llama 2','EN-FR') else ''),pad=4)
        ax.grid(lw=.45)
    handles=[Line2D([],[],marker=MARKERS[m],linestyle='None',color=c['methods'][m],label=m,
                        markersize=4.5 if m=='CRAT' else 3) for m in METHODS]
    fig.legend(handles=handles,ncol=6,frameon=False,loc='upper center',bbox_to_anchor=(.5,.995),columnspacing=1.0,handletextpad=.4)
    e.save_figure(ident,fig,legend_space=.07)


def resource_ratios(ident,data,e):
    c=data['comparison']['C'];colors=e.style.colors
    keys=[('trainable_params_CRAT_ratio','Trainable parameters'),('peak_vram_GB_CRAT_ratio','Peak memory'),
          ('train_seconds_CRAT_ratio','Training time'),('latency_mean_seconds_CRAT_ratio','Mean generation latency')]
    bases=METHODS[:-1]
    fig,axs=axes(e,ident,2,2,(7,4.8))
    for ax,(key,title) in zip(axs.flat,keys):
        z=np.array([[c[(c.model==m)&(c.pair==p)&(c.baseline==b)].iloc[0][key] for b in bases] for m,p in SETTINGS])
        rgba=np.zeros(z.shape+(4,))
        # Same light tint for each side of one: color does not imply effect magnitude.
        for mask,color in [(z<1,colors['positive']),(z>=1,colors['negative'])]:
            rgba[mask]=(*(.20*np.array(to_rgb(color))+.80),1)
        ax.imshow(rgba,aspect='auto')
        for i in range(6):
            for j in range(5):
                value=z[i,j]
                label=f'{value:.3f}' if value<.1 or abs(value-1)<.015 else f'{value:.2f}'
                ax.text(j,i,label,ha='center',va='center',fontsize=e.style.annotation)
        ax.set_title(title,pad=5)
        ax.set_xticks(range(5),['Full\nSFT','Targeted\nSFT','LoRA','rsLoRA','DoRA'])
        ax.set_yticks(range(6),[('L2' if m=='Llama 2' else 'M')+' '+p+(' †' if (m,p)==('Llama 2','EN-FR') else '') for m,p in SETTINGS])
        ax.tick_params(length=0,pad=3)
        ax.axhline(2.5,color=colors['reference'],lw=.6)
        for spine in ax.spines.values():spine.set_visible(False)
    e.save_figure(ident,fig)


def ablation_quality(ident,data,e):
    d=data['ablation']['E']
    fig,axs=axes(e,ident,1,3,(7,3.5))
    for j,(ax,metric) in enumerate(zip(axs.flat,METRICS)):
        z=np.array([[d.loc[(p,v),metric+'_drop'] for p in PAIRS] for v in VARIANTS])
        labels=[[f'{v:.5f}' if metric=='COMET' else f'{v:.3f}' for v in row] for row in z]
        heatmap(e,fig,ax,z,VARIANT_LABELS if j==0 else ['']*6,PAIRS,labels,signed=bool((z<0).any()))
        ax.set_title(metric,pad=5,fontweight='bold')
    e.save_figure(ident,fig)


def ablation_resources(ident,data,e,integrated=False):
    d=data['ablation']['C']
    columns=['vram_reduction_pct_vs_CRAT','train_time_reduction_pct_vs_CRAT']
    titles=['Peak memory reduction (%)','Training time reduction (%)']
    if integrated:
        fig,axs=axes(e,ident,1,2,(7,3.2))
        for j,(ax,col,title) in enumerate(zip(axs.flat,columns,titles)):
            z=np.array([[d.loc[(p,v),col] for p in PAIRS] for v in VARIANTS])
            heatmap(e,fig,ax,z,VARIANT_LABELS if j==0 else ['']*6,PAIRS,
                    [[f'{v:+.2f}' for v in row] for row in z],signed=True)
            ax.set_title(title,pad=5)
        e.save_figure(ident,fig);return
    fig,axs=axes(e,ident,1,2,(7,4.0),sharey=True)
    for ax,col,title in zip(axs.flat,columns,titles):
        all_values=[]
        for j,p in enumerate(PAIRS):
            values=np.array([d.loc[(p,v),col] for v in VARIANTS]);all_values.extend(values)
            ys=np.arange(len(VARIANTS))+(j-1)*.24
            ax.barh(ys,values,height=.21,color=e.style.colors['pairs'][p],label=p,zorder=3)
            for y,v in zip(ys,values):ax.text(max(v,0)+.4,y,f'{v:.1f}',va='center',fontsize=e.style.annotation)
        ax.axvline(0,color=e.style.colors['reference'],lw=.6);ax.grid(axis='x',zorder=0)
        ax.set_yticks(range(6),VARIANT_LABELS);ax.set_xlabel(title)
        ax.set_xlim(min(-2,min(all_values)*1.3),max(all_values)*1.16+1)
        ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0,pad=3)
    axs[0,0].invert_yaxis()
    handles,labels=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=3,frameon=False,loc='upper center',bbox_to_anchor=(.59,.995))
    e.save_figure(ident,fig,legend_space=.075)


def causal_routing(ident,data,e):
    d=data['ablation']['R'];c=e.style.colors
    fig,axs=axes(e,ident,2,3,(7,3.8),sharey=True)
    for j,pair in enumerate(PAIRS):
        for i,phase in enumerate(['train','eval']):
            ax=axs[i,j]
            for k,(variant,color,label) in enumerate([('CRAT_full',c['positive'],'Full CRAT'),('no_causal',c['negative'],'No causal prior')]):
                values=[d.loc[(pair,variant),phase+'_gate_mean_'+role] for role in ROLES]
                ax.bar(np.arange(4)+(k-.5)*.34,values,width=.32,color=color,label=label,zorder=3)
            ax.set_xticks(range(4),[x.capitalize() for x in ROLES],rotation=28,ha='right')
            ax.set_ylim(0,1.03);ax.grid(axis='y',zorder=0);ax.tick_params(length=0)
            if i==0:ax.set_title(pair,pad=4)
            if j==0:ax.set_ylabel(('Training' if i==0 else 'Evaluation')+' mean soft gate')
            if i==1:ax.axhline(.5,color=c['reference'],ls='--',lw=.65,zorder=2)
    h,l=axs[0,0].get_legend_handles_labels();fig.legend(h,l,ncol=2,frameon=False,loc='upper center',bbox_to_anchor=(.55,.995))
    e.save_figure(ident,fig,legend_space=.075)


def sensitivity_ranges(ident,data,e):
    d=data['sensitivity']['common'].set_index(['pair','parameter'])
    fig,axs=axes(e,ident,1,2,(7,3.9))
    for j,(ax,metric) in enumerate(zip(axs.flat,['BLEU','chrF'])):
        z=np.array([[d.loc[(pair,p),metric+'_range_on_common_grid'] for pair in PAIRS] for p in PARAMS])
        heatmap(e,fig,ax,z,PARAM_LABELS if j==0 else ['']*8,PAIRS)
        ax.set_title(metric+' range',pad=5)
    e.save_figure(ident,fig)


def head_budget(ident,data,e):
    d=data['sensitivity']['D'];c=e.style.colors
    palette=[c['pairs'][p] for p in PAIRS]
    fig,axs=axes(e,ident,2,3,(7,3.7))
    for j,pair in enumerate(PAIRS):
        g=d[(d.pair==pair)&(d.sensitivity_parameter=='heads_per_direction')].sort_values('sensitivity_value')
        for i,metric in enumerate(['BLEU','chrF']):
            ax=axs[i,j]
            ax.plot(g.trainable_params_mean/1e6,g[metric+'_mean_mean'],color=c['reference'],lw=.6,zorder=1)
            for n,(_,r) in enumerate(g.iterrows()):
                x=r.trainable_params_mean/1e6;y=r[metric+'_mean_mean']
                ax.errorbar(x,y,yerr=sd_or_none(r.get(metric+'_mean_std',np.nan)),fmt='o',
                            color=palette[n%len(palette)],capsize=1.5,zorder=3)
                ax.annotate(f'{r.sensitivity_value:g}',(x,y),xytext=(0,5),textcoords='offset points',
                            ha='center',fontsize=e.style.annotation)
            ax.margins(x=.15,y=.27);ax.grid(lw=.45);ax.set_ylabel(metric)
            if i==0:ax.set_title(pair,pad=4)
            if i==1:ax.set_xlabel('Trainable parameters (M)')
            ax.xaxis.set_major_locator(MaxNLocator(3));ax.yaxis.set_major_locator(MaxNLocator(4))
    e.save_figure(ident,fig)


def topk(ident,data,e):
    d=data['sensitivity']['D']
    fig,axs=axes(e,ident,3,3,(7,4.6))
    for j,pair in enumerate(PAIRS):
        g=d[(d.pair==pair)&(d.sensitivity_parameter=='top_k')].sort_values('sensitivity_value')
        for i,metric in enumerate(METRICS):
            ax=axs[i,j]
            ax.plot(g.sensitivity_value,g[metric+'_mean_mean'],'o-',color=e.style.colors['positive'])
            for _,r in g.iterrows():
                sd=sd_or_none(r.get(metric+'_mean_std',np.nan))
                if sd is not None:ax.errorbar(r.sensitivity_value,r[metric+'_mean_mean'],yerr=sd,fmt='none',
                                            color=e.style.colors['positive'],capsize=1.5,elinewidth=.65)
            ax.axvline(2,color=e.style.colors['reference'],ls='--',lw=.6,zorder=0)
            ax.set_xticks(sorted(g.sensitivity_value.unique()))
            ax.set_ylabel(metric);ax.grid(axis='y',lw=.45);ax.margins(y=.20)
            ax.ticklabel_format(axis='y',style='plain',useOffset=False)
            ax.yaxis.set_major_locator(MaxNLocator(4))
            if i==0:ax.set_title(pair,pad=4)
            if i==2:ax.set_xlabel(r'Evaluation top $k$')
    e.save_figure(ident,fig)


def sparse_gates(ident,data,e):
    d=data['sensitivity']['D']
    fig,axs=axes(e,ident,1,3,(7,2.65),sharey=True)
    for j,pair in enumerate(PAIRS):
        g=d[(d.pair==pair)&(d.sensitivity_parameter=='lambda_sparse')].sort_values('sensitivity_value');ax=axs[0,j]
        for role in ROLES:
            ax.plot(g.sensitivity_value,g['eval_gate_mean_'+role+'_mean'],'o-',
                    color=e.style.colors['roles'][role],label=role.capitalize())
        ax.set_xscale('log');ax.set_xticks(g.sensitivity_value,[f'{v:g}' for v in g.sensitivity_value]);ax.minorticks_off()
        if len(g)>3:ax.tick_params(axis='x',labelrotation=25)
        ax.axhline(.5,color=e.style.colors['reference'],ls='--',lw=.6)
        ax.set_ylim(.15,1.0);ax.grid(axis='y',lw=.45)
        ax.set_title(pair,pad=4);ax.set_xlabel('Sparsity coefficient (log scale)')
        if j==0:ax.set_ylabel('Evaluation mean soft gate')
    h,l=axs[0,0].get_legend_handles_labels()
    fig.legend(h,l,ncol=4,frameon=False,loc='upper center',bbox_to_anchor=(.56,.995))
    e.save_figure(ident,fig,legend_space=.115)


def without_kl(ident,data,e):
    costs=data['ablation']['C'];q=data['ablation']['Q'];base=data['comparison']['D'];c=e.style.colors
    fig,axs=axes(e,ident,1,4,(7,2.45),gridspec_kw={'width_ratios':[1.55,1,1,1]})
    ax=axs[0,0];ys=np.arange(3)
    for col,offset,color in [('vram_reduction_pct_vs_CRAT',-.16,c['memory']),('train_time_reduction_pct_vs_CRAT',.16,c['training_time'])]:
        values=np.array([costs.loc[(p,'no_kl'),col] for p in PAIRS])
        ax.barh(ys+offset,values,height=.25,color=color,zorder=3)
        for y,value in zip(ys+offset,values):ax.text(value+.65,y,f'{value:.1f}',va='center',fontsize=e.style.annotation)
    all_values=np.array([[costs.loc[(p,'no_kl'),col] for p in PAIRS] for col in ['vram_reduction_pct_vs_CRAT','train_time_reduction_pct_vs_CRAT']])
    ax.set_xlim(min(0,all_values.min()*1.1),all_values.max()*1.24)
    ax.set_yticks(ys,PAIRS);ax.set_xlabel('Reduction (%)');ax.set_title('(a) Resources',loc='left',pad=5)
    ax.xaxis.set_major_locator(MaxNLocator(3));ax.grid(axis='x',zorder=0);ax.spines['left'].set_visible(False)
    contrasts=[]
    for j,metric in enumerate(METRICS,1):
        ax=axs[0,j];all_deltas=[]
        for method,offset,color,marker in [('Targeted SFT',-.14,c['targeted'],'o'),('LoRA',.14,c['lora'],'D')]:
            g=base[(base.model=='Mistral')&(base.method==method)].set_index('pair')
            values=np.array([q.loc[(p,'no_kl'),metric]-g.loc[p,metric] for p in PAIRS])
            for p,v in zip(PAIRS,values):contrasts.append({'pair':p,'metric':metric,'baseline':method,'no_kl_minus_baseline':v})
            scaled=values*(1000 if metric=='COMET' else 1);all_deltas.extend(scaled)
            ax.scatter(scaled,ys+offset,color=color,marker=marker,s=15,edgecolor='white',linewidth=.3,zorder=4)
        lo=min(0,min(all_deltas));hi=max(0,max(all_deltas));span=max(hi-lo,1e-9)
        ax.set_xlim(lo-.13*span,hi+.15*span)
        ax.axvline(0,color=c['reference'],ls='--',lw=.6,zorder=1)
        for y in ys:ax.axhline(y,color=c['grid'],lw=.35,zorder=0)
        ax.set_yticks(ys,[]);ax.spines['left'].set_visible(False)
        ax.set_title(f'({chr(97+j)}) {metric}',loc='left',pad=5)
        ax.set_xlabel('Difference (×0.001)' if metric=='COMET' else 'Score difference')
        ax.xaxis.set_major_locator(MaxNLocator(3));ax.tick_params(axis='y',length=0)
    for ax in axs.flat:ax.set_ylim(2.5,-.5)
    handles=[Patch(color=c['memory'],label='Peak memory'),Patch(color=c['training_time'],label='Training time'),
             Line2D([],[],marker='o',color=c['targeted'],linestyle='None',label='Targeted SFT',markersize=3),
             Line2D([],[],marker='D',color=c['lora'],linestyle='None',label='LoRA',markersize=3)]
    fig.legend(handles=handles,ncol=4,frameon=False,loc='upper center',bbox_to_anchor=(.55,.995),columnspacing=1.4)
    path=e.paths('integrated','data_tables')/'no_KL_contextual_quality_differences.csv'
    pd.DataFrame(contrasts).to_csv(path,index=False)
    e.save_figure(ident,fig,legend_space=.14)


FUNCTIONS={
    'comparison/Figure_1_CRAT_margins':margins,
    'comparison/Figure_2_COMET_mean_SD':comet,
    'comparison/Figure_3_quality_parameters':quality_parameters,
    'comparison/Figure_4_resource_ratios':resource_ratios,
    'ablation/Figure_1_quality_decreases':ablation_quality,
    'ablation/Figure_2_resource_savings':ablation_resources,
    'ablation/Figure_3_causal_routing':causal_routing,
    'sensitivity/Figure_1_matched_grid_sensitivity':sensitivity_ranges,
    'sensitivity/Figure_2_head_budget_tradeoffs':head_budget,
    'sensitivity/Figure_3_topk_quality':topk,
    'sensitivity/Figure_4_sparsity_and_gates':sparse_gates,
    'integrated/Figure_1_baseline_margins':margins,
    'integrated/Figure_2_ablation_resource_changes':lambda i,d,e:ablation_resources(i,d,e,True),
    'integrated/Figure_3_topk_quality':topk,
    'integrated/Figure_4_CRAT_without_KL':without_kl,
}


def render_figure(ident,data,exporter):
    FUNCTIONS[ident](ident,data,exporter)
