#!/usr/bin/env python3
"""Reproduce CRAT figures and tables from the supplied CSV exports.

Run `python crat_visualize.py --help` for all styling and selection options.
Captions and interpretive notes are always exported as separate UTF-8 text.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import fnmatch
import hashlib
import html
import importlib.metadata
from io import BytesIO
import json
from pathlib import Path
import re
import sys
import textwrap

try:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import is_color_like
    from matplotlib.patches import Rectangle
    from matplotlib.text import Text
except ImportError as exc:
    raise SystemExit(f'Missing dependency: {exc}. Run: python -m pip install -r requirements.txt')

HERE = Path(__file__).resolve().parent
SECTIONS = ['comparison', 'ablation', 'sensitivity', 'integrated']
METHODS = ['Base', 'Full SFT', 'Targeted SFT', 'LoRA', 'rsLoRA', 'DoRA', 'CRAT']
PAIRS = ['EN-FR', 'EN-FA', 'EN-ZH']
ROLES = ['source', 'indicator', 'position', 'integration']

DEFAULT_CONFIG = {
    'font_family': 'Times New Roman',
    'font_size': 7.0,
    'tick_font_size': 6.0,
    'legend_font_size': 6.5,
    'annotation_font_size': 6.5,
    'panel_font_size': 8.0,
    'table_font_size': 7.0,
    'figure_width': 7.0,
    'figure_height': None,
    'figure_sizes': {},
    'table_width': 7.0,
    'table_sizes': {},
    'dpi': 600,
    'formats': ['pdf', 'svg', 'png'],
    'palette': 'colorblind',
    'colors': {},
    'font_files': [],
    'strict_font': False,
    'svg_text': False,
    'table_rows_per_page': 40,
    'line_width': 0.8,
    'marker_size': 3.5,
}

PALETTES = {
    'colorblind': {
        'methods': dict(zip(METHODS, ['#999999','#8C564B','#666666','#0072B2','#CC79A7','#D69F00','#009E73'])),
        'pairs': dict(zip(PAIRS, ['#0072B2','#CC79A7','#D55E00'])),
        'roles': dict(zip(ROLES, ['#0072B2','#CC79A7','#D55E00','#009E73'])),
        'negative': '#A95436', 'neutral': '#FAFAF7', 'positive': '#217B68',
        'sequential_low': '#F2F6F8', 'sequential_high': '#174765',
        'memory': '#367C64', 'training_time': '#8471A8',
        'targeted': '#226888', 'lora': '#A55337',
        'grid': '#E2E6E9', 'text': '#171717', 'reference': '#7A858C',
        'table_header': '#E7EDF2', 'table_band': '#F0F1F2',
        'table_highlight': '#EDF5F1', 'table_border': '#D9D9D9',
    },
    'grayscale': {
        'methods': dict(zip(METHODS, ['#AAAAAA','#222222','#777777','#444444','#999999','#666666','#000000'])),
        'pairs': dict(zip(PAIRS, ['#222222','#777777','#BBBBBB'])),
        'roles': dict(zip(ROLES, ['#111111','#555555','#999999','#CCCCCC'])),
        'negative':'#222222','neutral':'#FFFFFF','positive':'#909090',
        'sequential_low':'#F7F7F7','sequential_high':'#222222',
        'memory':'#333333','training_time':'#999999','targeted':'#222222','lora':'#999999',
        'grid':'#E4E4E4','text':'#111111','reference':'#777777',
        'table_header':'#E8E8E8','table_band':'#F1F1F1',
        'table_highlight':'#ECECEC','table_border':'#D9D9D9',
    },
}


def deep_update(base, patch):
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def clean_text(value):
    return str(value).replace('**','').replace('*k*','k').replace('top-*k*','top-k')


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', type=Path, default=HERE/'data', help='Directory containing the original 27 CSV files.')
    p.add_argument('--output-dir', type=Path, default=HERE/'output')
    p.add_argument('--sections', nargs='+', choices=['all']+SECTIONS, default=['all'])
    p.add_argument('--only', nargs='+', default=[], metavar='ID_OR_GLOB', help='Select output IDs, e.g. integrated/Figure_4*; quote wildcards.')
    p.add_argument('--kind', choices=['all','figures','tables'], default='all')
    p.add_argument('--list', action='store_true', help='List all output IDs without reading data or rendering.')
    p.add_argument('--config', type=Path, help='JSON style settings. Command-line options override the JSON.')
    p.add_argument('--write-config', type=Path, help='Write the editable default JSON configuration and exit.')
    p.add_argument('--font-family', help='Default: Times New Roman.')
    p.add_argument('--font-file', action='append', type=Path, default=[], help='Register a local TTF/OTF file; repeat for other faces.')
    p.add_argument('--strict-font', action='store_true', default=None, help='Fail instead of substituting a missing font.')
    for flag in ['font-size','tick-font-size','legend-font-size','annotation-font-size','panel-font-size','table-font-size']:
        p.add_argument('--'+flag,type=float, help='Size in points.')
    p.add_argument('--figure-width',type=float,help='Width in inches; default 7.0. Aspect ratio is preserved unless height is supplied.')
    p.add_argument('--figure-height',type=float,help='Override height of all figures, in inches.')
    p.add_argument('--figsize',nargs=2,type=float,metavar=('WIDTH','HEIGHT'),help='Convenience override of both figure dimensions, in inches.')
    p.add_argument('--table-width',type=float,help='Width of visual table exports, in inches.')
    p.add_argument('--table-rows-per-page',type=int)
    p.add_argument('--dpi',type=int,help='Raster resolution; default 600 dpi.')
    p.add_argument('--formats',nargs='+',choices=['png','pdf','svg','tiff'],help='Default: pdf svg png.')
    p.add_argument('--palette',choices=list(PALETTES))
    p.add_argument('--colors',nargs='+',metavar='COLOR',help='Custom categorical color cycle. Quote hexadecimal colors.')
    p.add_argument('--crat-color',help='Override the CRAT method color.')
    p.add_argument('--svg-text',action='store_true',default=None,help='Keep editable SVG text; default outlines preserve font appearance.')
    return p.parse_args(argv)


class Style:
    def __init__(self,args):
        self.config=json.loads(json.dumps(DEFAULT_CONFIG))
        if args.config:
            overrides=json.loads(args.config.read_text(encoding='utf-8'))
            unknown=set(overrides)-set(DEFAULT_CONFIG)
            if unknown:raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
            deep_update(self.config,overrides)
        # Changing the base font size changes related defaults proportionally.
        if args.font_size is not None:
            base=args.font_size
            for key,offset in [('font_size',0),('tick_font_size',-1),('legend_font_size',-.5),
                               ('annotation_font_size',-.5),('panel_font_size',1),('table_font_size',0)]:
                self.config[key]=max(1,base+offset)
        for key in DEFAULT_CONFIG:
            v=getattr(args,key,None)
            if v is not None and key not in ['colors','font_files']:
                self.config[key]=v
        if args.figsize:
            self.config['figure_width'],self.config['figure_height']=args.figsize
        for key in ['font_size','tick_font_size','legend_font_size','annotation_font_size','panel_font_size',
                    'table_font_size','figure_width','table_width','dpi','line_width','marker_size']:
            if not isinstance(self.config[key],(int,float)) or self.config[key]<=0:
                raise ValueError(f'{key} must be positive.')
        if self.config['figure_height'] is not None and self.config['figure_height']<=0:
            raise ValueError('figure_height must be positive.')
        if self.config['table_rows_per_page']<3:raise ValueError('table_rows_per_page must be at least 3.')
        if self.config['dpi']<72:raise ValueError('dpi must be at least 72; 600 is recommended for publication.')
        if not self.config['formats'] or any(x not in ['pdf','svg','png','tiff'] for x in self.config['formats']):
            raise ValueError('Invalid formats configuration.')
        for key in ['figure_sizes','table_sizes']:
            for ident,wh in self.config[key].items():
                if len(wh)!=2 or any(not isinstance(x,(int,float)) or x<=0 for x in wh):
                    raise ValueError(f'{key}[{ident!r}] must be [positive width, positive height].')
        self.colors=json.loads(json.dumps(PALETTES[self.config['palette']]))
        deep_update(self.colors,self.config['colors'])
        if args.colors:
            if len(args.colors)<3:raise ValueError('--colors needs at least three colors.')
            for key,labels in [('methods',METHODS),('pairs',PAIRS),('roles',ROLES)]:
                self.colors[key]={label:args.colors[i%len(args.colors)] for i,label in enumerate(labels)}
            self.colors['positive']=args.colors[-1]
            self.colors['negative']=args.colors[0]
            self.colors['sequential_high']=args.colors[-1]
            self.colors['memory'],self.colors['training_time']=args.colors[:2]
            self.colors['targeted'],self.colors['lora']=args.colors[:2]
        if args.crat_color:self.colors['methods']['CRAT']=args.crat_color
        def validate_colors(values):
            for key,value in values.items():
                if isinstance(value,dict):validate_colors(value)
                elif not is_color_like(value):raise ValueError(f'Invalid color {key}: {value!r}')
        validate_colors(self.colors)
        files=[Path(x) for x in self.config['font_files']]+args.font_file
        for path in files:
            if not path.is_file():raise ValueError(f'Font file not found: {path}')
            font_manager.fontManager.addfont(str(path.resolve()))
        requested=self.config['font_family']
        if requested.lower().strip() in ['time new romance','times new romance']:
            requested='Times New Roman'
        try:
            font_path=font_manager.findfont(requested,fallback_to_default=False)
            actual=font_manager.FontProperties(fname=font_path).get_name()
        except ValueError:
            if self.config['strict_font']:
                raise ValueError(f'Font {requested!r} is unavailable. Install it or pass --font-file /path/to/font.ttf.')
            actual='DejaVu Serif'
            font_path=font_manager.findfont(actual,fallback_to_default=False)
            print(f'Font notice: {requested!r} is unavailable; using {actual!r}. '
                  'Use --font-file and --strict-font to require your chosen font.',file=sys.stderr)
        self.actual_font=actual
        self.font_path=str(font_path)
        self.config['resolved_font_family']=actual
        self.config['requested_font_family']=requested
        plt.rcParams.update({
            'font.family':actual,'font.size':self.config['font_size'],
            'axes.labelsize':self.config['font_size'],'axes.titlesize':self.config['panel_font_size'],
            'xtick.labelsize':self.config['tick_font_size'],'ytick.labelsize':self.config['tick_font_size'],
            'legend.fontsize':self.config['legend_font_size'],
            'axes.spines.top':False,'axes.spines.right':False,
            'axes.linewidth':.55,'grid.linewidth':.45,'lines.linewidth':self.config['line_width'],
            'lines.markersize':self.config['marker_size'],'text.color':self.colors['text'],
            'axes.labelcolor':self.colors['text'],'axes.edgecolor':self.colors['reference'],
            'grid.color':self.colors['grid'],'axes.unicode_minus':True,
            'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none' if self.config['svg_text'] else 'path',
            'mathtext.fontset':'stix','savefig.facecolor':'white','figure.facecolor':'white',
        })

    def size(self,ident,base):
        if ident in self.config['figure_sizes']:return self.config['figure_sizes'][ident]
        width=self.config['figure_width']
        return (width,self.config['figure_height'] or base[1]*width/base[0])

    @property
    def annotation(self):return self.config['annotation_font_size']


@dataclass
class TableSpec:
    section:str
    stem:str
    caption:str=''
    notes:list=field(default_factory=list)
    headers:list=field(default_factory=list)
    rows:list=field(default_factory=list)
    widths:list=field(default_factory=list)
    bands:set=field(default_factory=set)
    highlights:set=field(default_factory=set)
    bold:set=field(default_factory=set)

    @property
    def ident(self):return self.section+'/'+self.stem


class TableCollector:
    def __init__(self,section):self.section=section;self.specs=[];self.current=None
    def begin(self,stem):
        self.current=TableSpec(self.section,stem)
        self.specs.append(self.current)
    def caption(self,text):self.current.caption=clean_text(text)
    def note(self,text):self.current.notes.append(clean_text(text))
    def table(self,headers,rows,widths,bands=None,highlights=None,bold=None):
        t=self.current
        t.headers=[clean_text(x) for x in headers]
        t.rows=[[clean_text(x) for x in row]+['']*(len(headers)-len(row)) for row in rows]
        t.widths=widths;t.bands=set(bands or []);t.highlights=set(highlights or []);t.bold=set(bold or [])
        return t


def latex_escape(value):
    mapping={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#',
             '_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}',
             '±':r'$\pm$','†':r'$\dagger$','³':r'$^{3}$','⁰':r'$^{0}$'}
    lines=[''.join(mapping.get(c,c) for c in line) for line in str(value).split('\n')]
    return lines[0] if len(lines)==1 else r'\shortstack{' + r'\\'.join(lines) + '}'


class Exporter:
    def __init__(self,out,style,captions):
        self.out=out;self.style=style;self.captions=captions;self.records=[];self.layout_notices=[]

    def paths(self,section,kind):
        path=self.out/section/kind;path.mkdir(parents=True,exist_ok=True);return path

    def description(self,ident,text):
        section,stem=ident.split('/',1)
        p=self.paths(section,'descriptions')/(stem+'.txt')
        p.write_text(clean_text(text).strip()+'\n',encoding='utf-8')
        return str(p.relative_to(self.out))

    def save_figure(self,ident,fig,legend_space=0,layout=True):
        if fig._suptitle is not None:raise ValueError(f'{ident}: overall titles are forbidden; use the text caption.')
        if fig.texts:raise ValueError(f'{ident}: figure-level descriptions are forbidden; use the text caption.')
        if layout:fig.tight_layout(rect=(0,0,1,1-legend_space),pad=.7,h_pad=.8,w_pad=.9)
        section,stem=ident.split('/',1)
        files=self._save_visual(fig,self.paths(section,'figures'),stem)
        desc=self.description(ident,self.captions[ident])
        self.records.append({'id':ident,'kind':'figure','files':files,'description':desc,
                             'size_inches':list(fig.get_size_inches())})
        plt.close(fig)
        print('Rendered',ident,flush=True)

    def _save_visual(self,fig,directory,stem):
        files=[]
        fig.canvas.draw()
        renderer=fig.canvas.get_renderer()
        fw,fh=fig.get_size_inches()*fig.dpi
        hidden_tick_text=set()
        # Locators may create ticks outside the view interval. Matplotlib does
        # not draw those labels even though their Text.visible flag is True.
        for ax in fig.axes:
            for axis,limits in [(ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())]:
                lower,upper=sorted(limits)
                for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                    if not ax.axison or not axis.get_visible() or not lower<=tick.get_loc()<=upper:
                        hidden_tick_text.update([id(tick.label1),id(tick.label2)])
        # Detect clipped visible text without changing the requested physical size.
        for artist in fig.findobj(match=Text):
            if not artist.get_visible() or not artist.get_text() or id(artist) in hidden_tick_text:continue
            box=artist.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > fw+1 or box.y1 > fh+1:
                self.layout_notices.append(f'{stem}: text near/outside canvas: {artist.get_text()!r}; increase figure size.')
        for fmt in dict.fromkeys(self.style.config['formats']):
            p=directory/(stem+'.'+fmt)
            options={'dpi':self.style.config['dpi'],'facecolor':'white','bbox_inches':None}
            if fmt=='tiff':options['pil_kwargs']={'compression':'tiff_lzw'}
            # Render before opening the destination so a failed backend cannot
            # leave an apparently successful zero-byte publication file.
            buffer=BytesIO()
            fig.savefig(buffer,format=fmt,**options)
            payload=buffer.getvalue()
            if not payload:raise ValueError(f'Empty {fmt} export for {stem}.')
            temporary=p.with_suffix(p.suffix+'.tmp')
            temporary.write_bytes(payload)
            temporary.replace(p)
            if p.stat().st_size!=len(payload):raise ValueError(f'Incomplete export: {p}')
            files.append(str(p.relative_to(self.out)))
        return files

    def save_table(self,t):
        folder=self.paths(t.section,'tables')
        csv=folder/(t.stem+'.csv')
        pd.DataFrame(t.rows,columns=[x.replace('\n',' ') for x in t.headers]).to_csv(csv,index=False)
        font=self.style.config['table_font_size']
        tex=[r'% Requires \usepackage{booktabs}; caption is in the separate .txt file.',
             r'\begingroup',rf'\fontsize{{{font:g}}}{{{font*1.2:g}}}\selectfont',
             r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{l'+'c'*(len(t.headers)-1)+'}',r'\toprule',
             ' & '.join(r'\textbf{'+latex_escape(x)+'}' for x in t.headers)+r' \\',r'\midrule']
        for i,row in enumerate(t.rows):
            if i in t.bands:
                tex += [r'\multicolumn{'+str(len(row))+r'}{l}{\textbf{'+latex_escape(row[0])+r'}} \\']
            else:
                values=[(r'\textbf{'+latex_escape(v)+'}') if (i,j) in t.bold else latex_escape(v) for j,v in enumerate(row)]
                tex += [' & '.join(values)+r' \\']
        tex += [r'\bottomrule',r'\end{tabular}',r'\endgroup']
        texpath=folder/(t.stem+'.tex');texpath.write_text('\n'.join(tex)+'\n',encoding='utf-8')
        htmlpath=folder/(t.stem+'.html')
        h=['<!doctype html><meta charset="utf-8"><style>',
           f'table{{border-collapse:collapse;font-family:"{html.escape(self.style.actual_font)}",serif;font-size:{font:g}pt;}}',
           'th,td{padding:5px 7px;border:1px solid #d9d9d9;text-align:center;}td:first-child{text-align:left;}',
           f'th{{background:{self.style.colors["table_header"]};}}',
           f'.band{{background:{self.style.colors["table_band"]};text-align:left;font-weight:bold;}}',
           f'.ref{{background:{self.style.colors["table_highlight"]};}}','</style><table><thead><tr>']
        h += ['<th>'+html.escape(x).replace('\n','<br>')+'</th>' for x in t.headers]
        h += ['</tr></thead><tbody>']
        for i,row in enumerate(t.rows):
            if i in t.bands:h += [f'<tr><td class="band" colspan="{len(row)}">{html.escape(row[0])}</td></tr>'];continue
            h += ['<tr'+(' class="ref"' if i in t.highlights else '')+'>']
            for j,value in enumerate(row):
                value=html.escape(value).replace('\n','<br>')
                if (i,j) in t.bold:value='<b>'+value+'</b>'
                h += ['<td>'+value+'</td>']
            h += ['</tr>']
        h += ['</tbody></table>'];htmlpath.write_text('\n'.join(h),encoding='utf-8')
        visual=[]
        chunks=self._table_chunks(t)
        for page,indices in enumerate(chunks,1):
            fig=self._table_figure(t,indices)
            stem=t.stem+(f'_page_{page}' if len(chunks)>1 else '')
            visual+=self._save_visual(fig,folder,stem);plt.close(fig)
        desc=self.description(t.ident,t.caption+'\n\n'+'\n\n'.join(t.notes))
        files=[str(x.relative_to(self.out)) for x in [csv,texpath,htmlpath]]+visual
        self.records.append({'id':t.ident,'kind':'table','files':files,'description':desc,
                             'body_rows':len(t.rows),'visual_pages':len(chunks)})
        print('Rendered',t.ident,flush=True)

    def _table_chunks(self,t):
        limit=self.style.config['table_rows_per_page'];chunks=[];current=[];last_band=None
        for i in range(len(t.rows)):
            if len(current)>=limit or (i in t.bands and len(current)>=limit-1):
                chunks.append(current);current=[]
                if i not in t.bands and last_band is not None:current=[last_band]
            if i in t.bands:last_band=i
            current.append(i)
        if current:chunks.append(current)
        return chunks or [[]]

    def _table_figure(self,t,indices):
        c=self.style.config;size=c['table_sizes'].get(t.ident)
        width=size[0] if size else c['table_width'];font=c['table_font_size']
        fractions=np.array(t.widths,dtype=float);fractions/=fractions.sum()
        col_inches=fractions*(width-.18)
        # Wrap only labels; numeric strings keep their mean ± SD together.
        def wrap(value,j):
            lines=[];characters=max(5,int((col_inches[j]*72-9)/(font*.48)))
            for line in value.split('\n'):
                if len(line)>characters and (' ' in line) and ('±' not in line):
                    lines += textwrap.wrap(line,width=characters,break_long_words=False,break_on_hyphens=False)
                else:lines.append(line)
            return '\n'.join(lines)
        headers=[wrap(x,j) for j,x in enumerate(t.headers)]
        rows=[];heights=[]
        header_h=max(x.count('\n')+1 for x in headers)*font*1.30/72+.09
        for i in indices:
            row=[wrap(x,j) for j,x in enumerate(t.rows[i])]
            if i in t.bands:row=[t.rows[i][0]]+['']*(len(row)-1)
            rows.append(row)
            heights.append(max(x.count('\n')+1 for x in row)*font*1.30/72+.07)
        natural=header_h+sum(heights)+.16
        height=size[1] if size else natural
        if height<natural-.03:
            raise ValueError(f'{t.ident}: specified height {height:g} in is too small for {font:g} pt text; use at least {natural:.2f} in.')
        fig=plt.figure(figsize=(width,height));ax=fig.add_axes([.09/width,.08/height,1-.18/width,1-.16/height]);ax.axis('off')
        body_h=height-.16
        table=ax.table(cellText=rows,colLabels=headers,colWidths=fractions,cellLoc='center',bbox=[0,0,1,1])
        table.auto_set_font_size(False);table.set_fontsize(font)
        for (ri,cj),cell in table.get_celld().items():
            cell.set_linewidth(.35);cell.set_edgecolor(self.style.colors['table_border']);cell.PAD=.06
            cell.set_height((header_h if ri==0 else heights[ri-1])/body_h)
            cell.get_text().set_color(self.style.colors['text'])
            if ri==0:
                cell.set_facecolor(self.style.colors['table_header']);cell.get_text().set_weight('bold')
            else:
                original=indices[ri-1]
                if original in t.bands:
                    cell.set_facecolor(self.style.colors['table_band'])
                    cell.get_text().set_text('')
                    cell.visible_edges='closed'
                elif original in t.highlights:cell.set_facecolor(self.style.colors['table_highlight'])
                else:cell.set_facecolor('white')
                if (original,cj) in t.bold:cell.get_text().set_weight('bold')
            if cj==0:cell.get_text().set_ha('left')
        # Group headings span the complete table, so long labels never get
        # clipped to the narrow first data column. Resolve row positions before
        # drawing the shared band above the individual table cells.
        fig.canvas.draw()
        for ri,original in enumerate(indices,1):
            if original not in t.bands:continue
            cell=table[ri,0]
            x,y=cell.get_xy();h=cell.get_height()
            ax.add_patch(Rectangle((x,y),1,h,transform=ax.transAxes,
                facecolor=self.style.colors['table_band'],
                edgecolor=self.style.colors['table_border'],linewidth=.35,zorder=3))
            ax.text(x+.008,y+h/2,t.rows[original][0],transform=ax.transAxes,
                ha='left',va='center',fontsize=font,fontweight='bold',
                color=self.style.colors['text'],zorder=4)
        return fig


def selected(ident,args,sections):
    return ident.split('/')[0] in sections and (not args.only or any(fnmatch.fnmatchcase(ident,x) for x in args.only))


def check_sources(path,needed):
    expected=[]
    if 'comparison' in needed:
        expected += [f'{pair}_{model}_OPUS100_Results_main-comparison.csv' for pair in PAIRS for model in ['LlaMA2','Mistral']]
    if 'ablation' in needed:
        expected += [f'{pair.lower()}_ablation_posthoc_opus100_{kind}.csv' for pair in PAIRS for kind in ['efficiency','routing']]
    if 'sensitivity' in needed:
        expected += [f'{pair.lower()}_sensitivity_posthoc_opus100_{kind}.csv' for pair in PAIRS
                     for kind in ['summary','efficiency','routing_summary','parameter_robustness','comet']]
    missing=[x for x in expected if not (path/x).is_file()]
    if missing:raise ValueError('Missing CSV inputs in '+str(path)+':\n  '+'\n  '.join(missing))
    if 'comparison' in needed:
        extras=set(x.name for x in path.glob('*main-comparison.csv'))-set(expected)
        if extras:raise ValueError('Unexpected comparison CSV files: '+', '.join(sorted(extras))+'. Keep only the six intended comparison files in this directory.')
    return [path/x for x in expected]


def main(argv=None):
    args=parse_args(argv)
    if args.write_config:
        args.write_config.parent.mkdir(parents=True,exist_ok=True)
        args.write_config.write_text(json.dumps(DEFAULT_CONFIG,indent=2)+'\n',encoding='utf-8')
        print('Wrote',args.write_config);return 0
    captions=json.loads((HERE/'captions.json').read_text(encoding='utf-8'))
    table_catalog=json.loads((HERE/'table_catalog.json').read_text(encoding='utf-8'))
    sections=SECTIONS if 'all' in args.sections else args.sections
    figures=[ident for ident in captions if selected(ident,args,sections) and args.kind!='tables']
    tables=[section+'/'+stem for section,stems in table_catalog.items() for stem in stems
            if selected(section+'/'+stem,args,sections) and args.kind!='figures']
    if args.list:
        print('\n'.join(figures+tables));print(f'\n{len(figures)} figures; {len(tables)} tables.');return 0
    if not figures and not tables:raise ValueError('No output IDs match the selection. Use --list to see valid IDs.')
    style=Style(args)
    out=args.output_dir.resolve();src=args.data_dir.resolve()
    if out==src or src.is_relative_to(out):raise ValueError('Output directory must not be the data directory or its ancestor.')
    needed={ident.split('/')[0] for ident in figures+tables}
    if 'integrated' in needed:needed.update(['comparison','ablation','sensitivity']);needed.remove('integrated')
    if 'ablation' in needed:needed.add('comparison')
    source_files=check_sources(src,needed)
    out.mkdir(parents=True,exist_ok=True)
    from crat_data import comparison_data,ablation_data,sensitivity_data
    import crat_tables
    from crat_plots import render_figure
    data={}
    if 'comparison' in needed:data['comparison']=comparison_data(src,out/'comparison')
    if 'ablation' in needed:data['ablation']=ablation_data(src,out/'ablation',data['comparison'])
    if 'sensitivity' in needed:data['sensitivity']=sensitivity_data(src,out/'sensitivity')
    e=Exporter(out,style,captions)
    for ident in figures:render_figure(ident,data,e)
    for section in sections:
        if not any(x.startswith(section+'/') for x in tables):continue
        collector=TableCollector(section)
        getattr(crat_tables,section+'_tables')(data,collector)
        for spec in collector.specs:
            if spec.ident in tables:e.save_table(spec)
    combined=[]
    for record in e.records:
        combined += [record['id'],'='*len(record['id']),(out/record['description']).read_text(encoding='utf-8').strip(),'']
    (out/'all_descriptions.txt').write_text('\n'.join(combined)+'\n',encoding='utf-8')
    data_notes=[]
    if 'comparison' in data:
        d=data['comparison'];data_notes += [f'Comparison additive-delta discrepancies: {len(d["A"])}. Plots use primary means.',
            'Main quality values are reported means over six seeds; SD is not a paired confidence interval.']
    if 'ablation' in data:
        d=data['ablation'];data_notes += [f'Ablation quality-delta discrepancies: {int(d["A"].mismatch.sum())}.',
            'Ablation backbone and seed metadata are not independently identified; displayed full references require reconciliation.']
    if 'sensitivity' in data:
        d=data['sensitivity'];data_notes += [f'Sensitivity settings: {len(d["D"])}.',
            'Missing COMET and variability values are unreported, not zero. Directional-average COMET is used, not pooled COMET.']
    data_notes += ['Resource percentages use the stated within-study reference. No significance tests or missing SDs are invented.',
        'Figures and visual tables contain data labels, panel labels, axes and legends only. Captions and interpretive notes are separate.',
        'The bundled descriptive text was prepared for these experiments. Recheck experimental metadata and text if substituting a new study.']
    (out/'data_notes.txt').write_text('\n'.join(data_notes)+'\n',encoding='utf-8')
    notices=list(dict.fromkeys(e.layout_notices))
    (out/'layout_notices.txt').write_text('\n'.join(notices)+'\n',encoding='utf-8')
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version.split()[0],
        'versions':{x:importlib.metadata.version(x) for x in ['numpy','pandas','matplotlib']},
        'style':style.config,'resolved_colors':style.colors,'source_files':[
            {'name':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in source_files],
        'outputs':e.records,'layout_notices':notices}
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(f'Completed {len(figures)} figures and {len(tables)} tables in {out}')
    if notices:print(f'Layout notice: inspect layout_notices.txt ({len(notices)} entries); consider larger figure sizes.',file=sys.stderr)
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except (ValueError,KeyError,AssertionError,FileNotFoundError) as exc:
        print('ERROR:',str(exc) or 'CSV validation failed; check unique rows, measurement columns and source consistency.',file=sys.stderr)
        raise SystemExit(2)
