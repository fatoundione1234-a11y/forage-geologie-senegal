import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyArrowPatch
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import plotly.graph_objects as go
import plotly.express as px
from scipy.interpolate import griddata
import datetime
import math

np.random.seed(42)

st.set_page_config(page_title="Dashboard Géologie Minière — Sénégal", layout="wide", page_icon="⛏️")
st.markdown("""<style>
.main-header{background:linear-gradient(90deg,#1A237E,#0D47A1);color:white;padding:15px 20px;border-radius:10px;margin-bottom:20px;}
</style>""", unsafe_allow_html=True)
st.markdown("""<div class='main-header'>
<h2>⛏️ Dashboard Géologie Minière — Projet Sénégal</h2>
<p>Sections · Cartes · 3D · Logues · SGI · Estimation · Cartographie · Graphiques structuraux · Monitoring · Rapport</p>
</div>""", unsafe_allow_html=True)

# ── CONSTANTES ────────────────────────────────────────────────────────────────
BASE_E, BASE_N  = 350000.0, 1480000.0
NOM_PROSPECT    = "Prospect Dakar-Gold"
NOM_PERMIS      = "Permis PR-2024-SN-001"

LITHOS       = ['Latérite','Saprolite','Saprock','Bédrock/Schiste','Quartzite aurifère','Granite frais']
LITHO_COLORS = {'Latérite':'#8B4513','Saprolite':'#DAA520','Saprock':'#CD853F',
                'Bédrock/Schiste':'#696969','Quartzite aurifère':'#FFD700','Granite frais':'#708090'}
ALTERATIONS  = ['Silicification','Argilisation','Séricitisation','Carbonatation','Chloritisation','Épidotisation']
ALTER_COLORS = {'Silicification':'#FF6B35','Argilisation':'#A8D8EA','Séricitisation':'#AA96DA',
                'Carbonatation':'#FCBAD3','Chloritisation':'#B8F0B8','Épidotisation':'#FFE66D'}
MINERALISATIONS = ['Aurifère disséminée','Aurifère filonienne','Sulfures disséminés','Magnétite','Pyrite massive','Stérile']
MINER_COLORS    = {'Aurifère disséminée':'#FFD700','Aurifère filonienne':'#FFA500',
                   'Sulfures disséminés':'#808080','Magnétite':'#2F4F4F','Pyrite massive':'#B8860B','Stérile':'#F5F5F5'}
STRUCTURES   = ['Faille normale','Faille inverse','Cisaillement','Veine de quartz','Zone altérée']
STRUCT_COLORS= {'Faille normale':'#FF0000','Faille inverse':'#0000FF',
                'Cisaillement':'#FF6600','Veine de quartz':'#FFFF00','Zone altérée':'#00CC00'}

# ── GÉNÉRATION DONNÉES ────────────────────────────────────────────────────────
forages=[]
for i in range(15):
    ftype=np.random.choice(['RC','Aircore','Diamond'],p=[0.4,0.3,0.3])
    prof=np.random.choice([80,100,120,150,200]) if ftype=='Diamond' else np.random.choice([30,40,50,60])
    forages.append({
        'trou':f'SG{i+1:03d}','type':ftype,
        'easting':round(BASE_E+np.random.uniform(-400,400),1),
        'northing':round(BASE_N+np.random.uniform(-400,400),1),
        'elevation':round(np.random.uniform(80,120),1),
        'profondeur':prof,
        'azimut':round(np.random.uniform(0,360),1),
        'inclinaison':round(np.random.uniform(-85,-60),1),
        'statut':np.random.choice(['Complété','En cours','Planifié'],p=[0.6,0.2,0.2]),
        'Au_max_ppb':round(np.random.lognormal(2.5,1.5),1),
        'Au_moy_ppb':round(np.random.lognormal(1.5,1),1),
        'longueur_mineralisee':round(np.random.uniform(2,30),1),
        'equipe':np.random.choice(['Équipe A','Équipe B','Équipe C']),
    })
df_forages=pd.DataFrame(forages)

intervals=[]
for _,f in df_forages.iterrows():
    depth=0
    while depth<f['profondeur']:
        thick=np.random.uniform(2,15)
        litho_idx=min(int(depth/f['profondeur']*len(LITHOS)),len(LITHOS)-1)
        litho=LITHOS[litho_idx] if np.random.random()>0.3 else np.random.choice(LITHOS)
        alter=np.random.choice(ALTERATIONS)
        miner=np.random.choice(MINERALISATIONS,p=[0.15,0.10,0.15,0.10,0.15,0.35])
        # Teneurs réalistes selon lithologie
        if litho == 'Quartzite aurifère':
            au = round(np.random.lognormal(5.5, 1.2), 2)  # ~300-2000 ppb
        elif litho == 'Bédrock/Schiste':
            au = round(np.random.lognormal(4.0, 1.5), 2)  # ~50-500 ppb
        elif litho == 'Saprock':
            au = round(np.random.lognormal(3.0, 1.2), 2)  # ~20-200 ppb
        else:
            au = round(np.random.lognormal(1.5, 1.0), 2)  # ~5-30 ppb
        # Bonus si minéralisation aurifère
        if miner in ['Aurifère disséminée', 'Aurifère filonienne']:
            au = round(au * np.random.uniform(1.5, 4.0), 2)
        intervals.append({
            'trou':f['trou'],'type':f['type'],
            'de':round(depth,1),'a':round(min(depth+thick,f['profondeur']),1),
            'lithologie':litho,'alteration':alter,'mineralisation':miner,
            'Au_ppb':au,'Cu_ppm':round(np.random.uniform(1,80),1),
            'As_ppm':round(np.random.uniform(1,50),1),
            'Ag_ppm':round(np.random.uniform(0.1,10),2),
            'mineralisé':au>=100,
        })
        depth+=thick
df_intervals=pd.DataFrame(intervals)

np.random.seed(10)
structures_df=pd.DataFrame([{
    'id':f'STR{i+1:02d}',
    'type':np.random.choice(STRUCTURES),
    'easting':round(BASE_E+np.random.uniform(-400,400),1),
    'northing':round(BASE_N+np.random.uniform(-400,400),1),
    'direction':round(np.random.uniform(0,360),1),
    'pendage':round(np.random.uniform(10,85),1),
    'sens_pendage':np.random.choice(['N','NE','E','SE','S','SO','O','NO']),
    'longueur_m':round(np.random.uniform(10,500),0),
    'porteur_miner':np.random.choice([True,False],p=[0.35,0.65]),
} for i in range(40)])

roches_terrain=pd.DataFrame([{
    'id':f'R{i+1:03d}',
    'type_roche':np.random.choice(LITHOS),
    'easting':round(BASE_E+np.random.uniform(-400,400),1),
    'northing':round(BASE_N+np.random.uniform(-400,400),1),
    'elevation':round(np.random.uniform(80,120),1),
    'description':np.random.choice(['Schiste sériciteux','Quartzite massif','Granite porphyrique',
                                     'Latérite pisolithique','Saprolite kaolinitique','Gneiss migmatitique']),
    'alteration':np.random.choice(['Forte','Moyenne','Faible','Nulle']),
    'Au_sol_ppb':round(np.random.lognormal(1,1.5),2),
    'observateur':np.random.choice(['Géologue A','Géologue B','Géologue C']),
    'date':(datetime.date.today()-datetime.timedelta(days=np.random.randint(1,30))).strftime('%Y-%m-%d'),
} for i in range(30)])

dates_week=pd.date_range(end=datetime.date.today(),periods=7)
weekly_data=pd.DataFrame({
    'date':dates_week,
    'metres_fores':np.random.randint(20,80,7),
    'trous_completes':np.random.randint(0,3,7),
    'incidents':np.random.randint(0,2,7),
    'Au_ppb_moyen':np.round(np.random.lognormal(2,0.8,7),1),
    'equipe':np.random.choice(['Équipe A','Équipe B'],7),
})

# ── TABS ──────────────────────────────────────────────────────────────────────
tabs=st.tabs([
    "📐 Sections géologiques",
    "🗺️ Cartes lithologiques",
    "🌡️ Carte anomalie",
    "🏗️ Cartes structurales",
    "📊 Logues lithologiques",
    "🔬 Logues structuraux",
    "🌐 Modèle 3D/Blocs",
    "📋 Planification & Infill",
    "🔄 Simulation déviation",
    "📡 Surveillance",
    "🧪 Essai SGI",
    "💰 Estimation des teneurs",
    "🗾 Cartographie terrain",
    "📐 Graphiques structuraux",
    "📈 Monitoring",
    "📅 Weekly Report",
    "📄 Rapport géologique",
])

# ══ TAB 1 — SECTIONS ══════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("📐 Sections Géologiques — RC / Aircore / Diamond")
    col1,col2=st.columns([1,3])
    with col1:
        section_type=st.selectbox("Type",['RC','Aircore','Diamond','Tous'])
        trous_dispo=df_forages['trou'].tolist() if section_type=='Tous' else df_forages[df_forages['type']==section_type]['trou'].tolist()
        trou_sel=st.selectbox("Trou réf.",trous_dispo)
        echelle=st.slider("Échelle vert. ×",1,5,2)
        show_miner=st.checkbox("Intervalles minéralisés",True)
        show_au=st.checkbox("Teneurs Au",True)
        show_limits=st.checkbox("Limites géologiques",True)
        show_infill=st.checkbox("Programme Infill",False)
    with col2:
        fig,ax=plt.subplots(figsize=(15,9))
        fig.patch.set_facecolor('#F8F8F0'); ax.set_facecolor('#E8F4F8')
        x_topo=np.linspace(0,600,300)
        topo=100+5*np.sin(x_topo/50)+3*np.cos(x_topo/30)+np.random.normal(0,0.5,300)
        ax.plot(x_topo,topo,'k-',linewidth=2.5,label='Topographie',zorder=5)
        ax.fill_between(x_topo,topo,0,alpha=0.15,color='brown')
        ax.axhline(y=0,color='blue',linestyle='--',linewidth=1,alpha=0.5,label='Ligne référence (0m)')
        if show_limits:
            topo_mean=topo.mean()
            limit_data=[
                (8,'Base Latérite','#8B4513'),
                (25,'Base Saprolite','#DAA520'),
                (50,'Base Saprock','#696969'),
                (80,'Top Bédrock','#FFD700'),
            ]
            for d,lbl,lc in limit_data:
                y_lim=topo_mean-d*echelle*0.5
                ax.axhline(y=y_lim,color=lc,linestyle='-.',linewidth=1.5,alpha=0.8,label=lbl)
        x_positions=np.linspace(60,540,min(6,len(trous_dispo)))
        for idx,(xpos,trou) in enumerate(zip(x_positions,trous_dispo[:6])):
            f=df_forages[df_forages['trou']==trou].iloc[0]
            topo_val=float(np.interp(xpos,x_topo,topo))
            ints=df_intervals[df_intervals['trou']==trou].sort_values('de')
            for _,interval in ints.iterrows():
                y_top=topo_val-interval['de']*echelle*0.5
                y_bot=topo_val-interval['a']*echelle*0.5
                color=LITHO_COLORS.get(interval['lithologie'],'#888888')
                ax.fill_betweenx([y_bot,y_top],xpos-7,xpos+7,color=color,alpha=0.85)
                ax.plot([xpos-7,xpos+7,xpos+7,xpos-7,xpos-7],[y_top,y_top,y_bot,y_bot,y_top],'k-',linewidth=0.3)
                if show_miner and interval['mineralisé']:
                    ax.fill_betweenx([y_bot,y_top],xpos-7,xpos+7,color='red',alpha=0.3,hatch='///')
                    ax.plot([xpos-7,xpos+7],[y_top,y_top],'r-',linewidth=1.5)
                    ax.plot([xpos-7,xpos+7],[y_bot,y_bot],'r-',linewidth=1.5)
                if show_au and interval['Au_ppb']>=100:
                    mid=(y_top+y_bot)/2
                    ax.text(xpos+9,mid,f"{interval['Au_ppb']:.0f}ppb",fontsize=5.5,color='#FF6600',fontweight='bold')
            ftype_color={'RC':'#FF5722','Aircore':'#2196F3','Diamond':'#9C27B0'}
            ax.text(xpos,topo_val+7,trou,ha='center',fontsize=7,fontweight='bold',color='#1A237E')
            ax.text(xpos,topo_val+4,f['type'],ha='center',fontsize=6,color=ftype_color.get(f['type'],'black'),fontweight='bold')
            y_bot_total=topo_val-f['profondeur']*echelle*0.5
            ax.text(xpos+9,y_bot_total,f"{f['profondeur']}m",fontsize=6,va='center',color='#333')
            if show_infill and idx<len(x_positions)-1:
                xpos_next=x_positions[min(idx+1,len(x_positions)-1)]
                xpos_infill=(xpos+xpos_next)/2
                ax.axvline(x=xpos_infill,color='purple',linestyle=':',linewidth=1.5,alpha=0.6)
                ax.text(xpos_infill,topo_val+7,'INFILL',ha='center',fontsize=6,color='purple',fontweight='bold')
        ax.text(10,max(topo)+10,f"Section {trou_sel} — Az.090° | {NOM_PROSPECT}",fontsize=9,
                fontweight='bold',color='#1A237E',bbox=dict(boxstyle='round',facecolor='white',alpha=0.9))
        ax.annotate('',xy=(575,max(topo)+6),xytext=(575,max(topo)-6),arrowprops=dict(arrowstyle='->',color='black',lw=2.5))
        ax.text(575,max(topo)+8,'N',ha='center',fontsize=12,fontweight='bold')
        ax.plot([20,70],[4,4],'k-',linewidth=3); ax.text(45,1,'50 m',ha='center',fontsize=8,fontweight='bold')
        legend_patches=[mpatches.Patch(color=c,label=l) for l,c in LITHO_COLORS.items()]
        if show_miner: legend_patches.append(mpatches.Patch(color='red',alpha=0.4,hatch='///',label='Minéralisé'))
        if show_infill: legend_patches.append(mpatches.Patch(color='purple',alpha=0.5,label='Infill prévu'))
        ax.legend(handles=legend_patches,loc='lower right',fontsize=7,title='Lithologie',ncol=2,framealpha=0.9)
        ax.set_xlabel("Distance (m)"); ax.set_ylabel("Élévation (m)")
        ax.set_title(f"Section géologique — {section_type} | {NOM_PROSPECT} | {NOM_PERMIS}",fontsize=12,fontweight='bold')
        ax.grid(True,linestyle=':',alpha=0.3); ax.set_xlim(0,600)
        plt.tight_layout(); st.pyplot(fig)

# ══ TAB 2 — CARTES LITHO ══════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("🗺️ Carte Lithologique — Toutes couleurs")
    prof_carte=st.slider("Profondeur (m)",0,150,20)
    fig2,ax2=plt.subplots(figsize=(11,9))
    fig2.patch.set_facecolor('#F5F5F0'); ax2.set_facecolor('#E8F4F8')
    for _,f in df_forages.iterrows():
        ints_d=df_intervals[(df_intervals['trou']==f['trou'])&(df_intervals['de']<=prof_carte)].tail(1)
        if len(ints_d)>0:
            litho=ints_d.iloc[0]['lithologie']; color=LITHO_COLORS.get(litho,'#888')
            marker={'RC':'^','Aircore':'o','Diamond':'s'}.get(f['type'],'o')
            ax2.scatter(f['easting'],f['northing'],c=color,s=200,marker=marker,edgecolors='black',linewidths=1.2,zorder=3)
            ax2.annotate(f"{f['trou']}\n{litho[:8]}",(f['easting'],f['northing']),textcoords="offset points",xytext=(5,5),fontsize=6,color='#1A237E')
    xmax=df_forages['easting'].max(); ymax=df_forages['northing'].max()
    xmin=df_forages['easting'].min(); ymin=df_forages['northing'].min()
    ax2.annotate('',xy=(xmax+60,ymax+30),xytext=(xmax+60,ymax-25),arrowprops=dict(arrowstyle='->',color='black',lw=2.5))
    ax2.text(xmax+60,ymax+40,'N',ha='center',fontsize=14,fontweight='bold')
    ax2.plot([xmin,xmin+200],[ymin-35,ymin-35],'k-',linewidth=3)
    ax2.text(xmin+100,ymin-50,'200 m',ha='center',fontsize=9,fontweight='bold')
    ax2.text(xmin,ymax+40,f"{NOM_PROSPECT} | {NOM_PERMIS}",fontsize=9,fontweight='bold',color='#1A237E')
    ax2.grid(True,linestyle='--',alpha=0.4,color='gray')
    legend_patches=[mpatches.Patch(color=c,label=l) for l,c in LITHO_COLORS.items()]
    type_markers=[plt.Line2D([0],[0],marker='^',color='w',markerfacecolor='gray',markersize=9,label='RC'),
                  plt.Line2D([0],[0],marker='o',color='w',markerfacecolor='gray',markersize=9,label='Aircore'),
                  plt.Line2D([0],[0],marker='s',color='w',markerfacecolor='gray',markersize=9,label='Diamond')]
    ax2.legend(handles=legend_patches+type_markers,loc='lower right',fontsize=8,title='Lithologie & Type',ncol=2,framealpha=0.95)
    ax2.set_xlabel("Easting UTM (m)"); ax2.set_ylabel("Northing UTM (m)")
    ax2.set_title(f"Carte lithologique à {prof_carte}m — {NOM_PROSPECT}",fontsize=12,fontweight='bold')
    plt.tight_layout(); st.pyplot(fig2)

# ══ TAB 3 — CARTE ANOMALIE ════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("🌡️ Carte d'Anomalie Géochimique")
    col1,col2=st.columns([1,3])
    with col1:
        element=st.selectbox("Élément",['Au_ppb','Cu_ppm','As_ppm'])
        seuil_anom=st.number_input("Seuil anomalie",10,1000,100)
        show_porteur=st.checkbox("Structures porteuses",True)
    with col2:
        au_max_trou=df_intervals.groupby('trou')[element].max().reset_index()
        au_max_trou.columns=['trou','valeur_max']
        df_anom=df_forages.merge(au_max_trou,on='trou')
        fig_an,ax_an=plt.subplots(figsize=(11,9))
        fig_an.patch.set_facecolor('#0A1628'); ax_an.set_facecolor('#0A1628')
        if len(df_anom)>3:
            xi=np.linspace(df_anom['easting'].min()-50,df_anom['easting'].max()+50,150)
            yi=np.linspace(df_anom['northing'].min()-50,df_anom['northing'].max()+50,150)
            Xi,Yi=np.meshgrid(xi,yi)
            Zi=griddata((df_anom['easting'],df_anom['northing']),np.log1p(df_anom['valeur_max']),(Xi,Yi),method='linear')
            contour=ax_an.contourf(Xi,Yi,Zi,levels=20,cmap='hot_r',alpha=0.8)
            plt.colorbar(contour,ax=ax_an,label=f'log({element}+1)')
        for _,f in df_anom.iterrows():
            potentiel=f['valeur_max']>=seuil_anom
            marker={'RC':'^','Aircore':'o','Diamond':'s'}.get(f['type'],'o')
            color_pt='#00FF00' if potentiel else '#FFFFFF'
            size=200 if potentiel else 80
            ax_an.scatter(f['easting'],f['northing'],c=color_pt,s=size,marker=marker,
                         edgecolors='black' if potentiel else 'gray',linewidths=1.5 if potentiel else 0.5,zorder=4)
            if potentiel:
                ax_an.annotate(f"{f['trou']}\n{f['valeur_max']:.0f}",(f['easting'],f['northing']),
                              textcoords="offset points",xytext=(6,6),fontsize=7,color='#00FF00',fontweight='bold')
                ax_an.scatter(f['easting'],f['northing'],c='none',s=400,marker='o',edgecolors='#00FF00',linewidths=2,zorder=5)
        if show_porteur:
            np.random.seed(5)
            for i in range(5):
                x1=np.random.uniform(df_anom['easting'].min(),df_anom['easting'].max())
                y1=np.random.uniform(df_anom['northing'].min(),df_anom['northing'].max())
                angle=np.random.uniform(20,70); length=np.random.uniform(200,500)
                x2=x1+length*np.cos(np.radians(angle)); y2=y1+length*np.sin(np.radians(angle))
                ax_an.plot([x1,x2],[y1,y2],color='cyan',linewidth=2.5,linestyle='--',label='Structure porteuse' if i==0 else '')
                ax_an.text((x1+x2)/2,(y1+y2)/2,'VQ',fontsize=8,color='cyan',fontweight='bold')
        xmax=df_anom['easting'].max(); ymax=df_anom['northing'].max()
        xmin=df_anom['easting'].min(); ymin=df_anom['northing'].min()
        ax_an.annotate('',xy=(xmax+60,ymax+25),xytext=(xmax+60,ymax-20),arrowprops=dict(arrowstyle='->',color='white',lw=2.5))
        ax_an.text(xmax+60,ymax+35,'N',ha='center',fontsize=14,fontweight='bold',color='white')
        ax_an.plot([xmin,xmin+200],[ymin-35,ymin-35],'w-',linewidth=3)
        ax_an.text(xmin+100,ymin-50,'200 m',ha='center',fontsize=9,fontweight='bold',color='white')
        ax_an.text(xmin,ymax+40,f"{NOM_PROSPECT}",fontsize=9,fontweight='bold',color='white')
        legend_an=[plt.Line2D([0],[0],marker='o',color='w',markerfacecolor='#00FF00',markersize=10,label=f'Potentiel (>{seuil_anom})'),
                   plt.Line2D([0],[0],marker='o',color='w',markerfacecolor='white',markersize=8,label='Non anomalique'),
                   plt.Line2D([0],[0],color='cyan',linestyle='--',linewidth=2,label='Structure porteuse')]
        ax_an.legend(handles=legend_an,loc='lower right',fontsize=8,framealpha=0.8,facecolor='#0A1628',labelcolor='white')
        ax_an.set_xlabel("Easting UTM (m)",color='white'); ax_an.set_ylabel("Northing UTM (m)",color='white')
        ax_an.tick_params(colors='white')
        ax_an.set_title(f"Carte anomalie {element} — {NOM_PROSPECT}",fontsize=12,fontweight='bold',color='white')
        plt.tight_layout(); st.pyplot(fig_an)

# ══ TAB 4 — CARTES STRUCT ════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("🏗️ Carte Structurale")
    fig3,ax3=plt.subplots(figsize=(11,9))
    fig3.patch.set_facecolor('#F5F5F0'); ax3.set_facecolor('#F0EDE0')
    np.random.seed(10)
    for i,(_,s) in enumerate(structures_df.head(15).iterrows()):
        angle=s['direction']; length=min(s['longueur_m'],400)
        x1=s['easting']; y1=s['northing']
        x2=x1+length*np.cos(np.radians(angle)); y2=y1+length*np.sin(np.radians(angle))
        color=STRUCT_COLORS.get(s['type'],'#888')
        ls='-' if 'Faille' in s['type'] else '--' if 'Veine' in s['type'] else ':'
        lw=3 if 'Faille' in s['type'] else 2
        ax3.plot([x1,x2],[y1,y2],color=color,linewidth=lw,linestyle=ls,label=s['type'])
        ax3.text((x1+x2)/2,(y1+y2)/2,f"{s['direction']:.0f}°/{s['pendage']:.0f}°{s['sens_pendage']}",fontsize=6,color=color,fontweight='bold')
    for _,f in df_forages.iterrows():
        ax3.scatter(f['easting'],f['northing'],c='black',s=60,zorder=3)
        ax3.annotate(f['trou'],(f['easting'],f['northing']),textcoords="offset points",xytext=(4,4),fontsize=6)
    xmax=df_forages['easting'].max(); ymax=df_forages['northing'].max()
    xmin=df_forages['easting'].min(); ymin=df_forages['northing'].min()
    ax3.annotate('',xy=(xmax+60,ymax+25),xytext=(xmax+60,ymax-20),arrowprops=dict(arrowstyle='->',color='black',lw=2.5))
    ax3.text(xmax+60,ymax+35,'N',ha='center',fontsize=14,fontweight='bold')
    ax3.plot([xmin,xmin+200],[ymin-35,ymin-35],'k-',linewidth=3)
    ax3.text(xmin+100,ymin-50,'200 m',ha='center',fontsize=9,fontweight='bold')
    ax3.text(xmin,ymax+40,f"{NOM_PROSPECT} | {NOM_PERMIS}",fontsize=9,fontweight='bold',color='#1A237E')
    handles,labels=ax3.get_legend_handles_labels()
    by_label=dict(zip(labels,handles))
    ax3.legend(by_label.values(),by_label.keys(),loc='lower right',fontsize=8,title='Structures',framealpha=0.9)
    ax3.set_xlabel("Easting UTM (m)"); ax3.set_ylabel("Northing UTM (m)")
    ax3.set_title(f"Carte structurale — {NOM_PROSPECT}",fontsize=12,fontweight='bold')
    ax3.grid(True,linestyle=':',alpha=0.4); plt.tight_layout(); st.pyplot(fig3)

# ══ TAB 5 — LOGUES LITHO ══════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("📊 Logues Lithologiques")
    col1,col2=st.columns([1,3])
    with col1:
        trou_logue=st.selectbox("Trou",df_forages['trou'].tolist(),key='lithologue')
        show_au_l=st.checkbox("Au (ppb)",True,key='aul')
        show_as_l=st.checkbox("As (ppm)",True,key='asl')
    with col2:
        f_info=df_forages[df_forages['trou']==trou_logue].iloc[0]
        ints=df_intervals[df_intervals['trou']==trou_logue].sort_values('de')
        ncols=1+(1 if show_au_l else 0)+(1 if show_as_l else 0)
        fig4,axes4=plt.subplots(1,ncols,figsize=(4*ncols,10),sharey=True)
        if ncols==1: axes4=[axes4]
        ax_l=axes4[0]
        for _,interval in ints.iterrows():
            color=LITHO_COLORS.get(interval['lithologie'],'#888')
            ax_l.fill_betweenx([interval['de'],interval['a']],0,1,color=color,alpha=0.85)
            if interval['mineralisé']:
                ax_l.fill_betweenx([interval['de'],interval['a']],0,1,color='red',alpha=0.25,hatch='///')
            ax_l.plot([0,1],[interval['de'],interval['de']],'k-',linewidth=0.3)
            mid=(interval['de']+interval['a'])/2
            ax_l.text(0.5,mid,interval['lithologie'][:10],ha='center',va='center',fontsize=5.5,fontweight='bold')
            ax_l.text(-0.05,interval['de'],f"{interval['de']}m",ha='right',fontsize=5.5)
        ax_l.set_ylim(f_info['profondeur'],0); ax_l.set_xlim(-0.1,1.1); ax_l.set_xticks([])
        ax_l.set_title(f"Litho\n{trou_logue}",fontsize=9,fontweight='bold'); ax_l.set_ylabel("Profondeur (m)")
        cidx=1
        if show_au_l:
            colors_au=['#FFD700' if v>=100 else '#EEE' for v in ints['Au_ppb']]
            axes4[cidx].barh([(i['de']+i['a'])/2 for _,i in ints.iterrows()],ints['Au_ppb'].values,
                             height=[(i['a']-i['de'])*0.8 for _,i in ints.iterrows()],color=colors_au,edgecolor='orange',linewidth=0.5)
            axes4[cidx].axvline(x=100,color='red',linestyle='--',linewidth=1.5,label='100ppb')
            axes4[cidx].set_xlabel("Au (ppb)"); axes4[cidx].set_title("Or",fontsize=9,fontweight='bold')
            axes4[cidx].legend(fontsize=7); cidx+=1
        if show_as_l:
            axes4[cidx].barh([(i['de']+i['a'])/2 for _,i in ints.iterrows()],ints['As_ppm'].values,
                             height=[(i['a']-i['de'])*0.8 for _,i in ints.iterrows()],color='#FF6B6B',edgecolor='red',linewidth=0.5)
            axes4[cidx].set_xlabel("As (ppm)"); axes4[cidx].set_title("Arsenic",fontsize=9,fontweight='bold')
        plt.suptitle(f"{trou_logue} | {f_info['type']} | {f_info['profondeur']}m | Az:{f_info['azimut']}° Inc:{f_info['inclinaison']}°",fontsize=10,fontweight='bold')
        plt.tight_layout(); st.pyplot(fig4)

# ══ TAB 6 — LOGUES STRUCT ════════════════════════════════════════════════════
with tabs[5]:
    st.subheader("🔬 Logues Structuraux")
    col1,col2=st.columns([1,3])
    with col1:
        trou_struct=st.selectbox("Trou",df_forages['trou'].tolist(),key='structlogue')
    with col2:
        f_info2=df_forages[df_forages['trou']==trou_struct].iloc[0]
        ints2=df_intervals[df_intervals['trou']==trou_struct].sort_values('de')
        fig5,axes5=plt.subplots(1,3,figsize=(12,10),sharey=True)
        for _,interval in ints2.iterrows():
            color=ALTER_COLORS.get(interval['alteration'],'#888')
            axes5[0].fill_betweenx([interval['de'],interval['a']],0,1,color=color,alpha=0.85)
            mid=(interval['de']+interval['a'])/2
            axes5[0].text(0.5,mid,interval['alteration'][:10],ha='center',va='center',fontsize=5.5,fontweight='bold')
            axes5[0].plot([0,1],[interval['de'],interval['de']],'k-',linewidth=0.3)
            axes5[0].text(-0.05,interval['de'],f"{interval['de']}m",ha='right',fontsize=5.5)
        axes5[0].set_ylim(f_info2['profondeur'],0); axes5[0].set_xlim(-0.1,1.1); axes5[0].set_xticks([])
        axes5[0].set_title("Altération",fontsize=9,fontweight='bold'); axes5[0].set_ylabel("Profondeur (m)")
        for _,interval in ints2.iterrows():
            color=MINER_COLORS.get(interval['mineralisation'],'#888')
            axes5[1].fill_betweenx([interval['de'],interval['a']],0,1,color=color,alpha=0.85)
            mid=(interval['de']+interval['a'])/2
            axes5[1].text(0.5,mid,interval['mineralisation'][:12],ha='center',va='center',fontsize=5,fontweight='bold')
            axes5[1].plot([0,1],[interval['de'],interval['de']],'k-',linewidth=0.3)
        axes5[1].set_ylim(f_info2['profondeur'],0); axes5[1].set_xlim(-0.1,1.1); axes5[1].set_xticks([])
        axes5[1].set_title("Minéralisation",fontsize=9,fontweight='bold')
        axes5[2].barh([(i['de']+i['a'])/2 for _,i in ints2.iterrows()],ints2['Cu_ppm'].values,
                      height=[(i['a']-i['de'])*0.8 for _,i in ints2.iterrows()],color='#B87333',edgecolor='brown',linewidth=0.5)
        axes5[2].set_xlabel("Cu (ppm)"); axes5[2].set_title("Cuivre",fontsize=9,fontweight='bold')
        plt.suptitle(f"Logue structural — {trou_struct} | {f_info2['type']} | {f_info2['profondeur']}m",fontsize=10,fontweight='bold')
        plt.tight_layout(); st.pyplot(fig5)

# ══ TAB 7 — MODÈLE 3D/BLOCS ══════════════════════════════════════════════════
with tabs[6]:
    st.subheader("🌐 Modèle 3D & Modélisation par Blocs")
    vue=st.radio("Vue",['3D Forages','Modèle de blocs 3D','2D Plan','2D Section'],horizontal=True)
    type_colors_3d={'RC':'red','Aircore':'blue','Diamond':'purple'}
    if vue=='3D Forages':
        fig3d=go.Figure()
        for _,f in df_forages.iterrows():
            inc_rad=np.radians(abs(f['inclinaison'])); az_rad=np.radians(f['azimut'])
            depths=np.linspace(0,f['profondeur'],30)
            xs=f['easting']+depths*np.sin(az_rad)*np.cos(inc_rad)
            ys=f['northing']+depths*np.cos(az_rad)*np.cos(inc_rad)
            zs=f['elevation']-depths*np.sin(inc_rad)
            color=type_colors_3d.get(f['type'],'gray')
            fig3d.add_trace(go.Scatter3d(x=xs,y=ys,z=zs,mode='lines+markers',
                line=dict(color=color,width=4),marker=dict(size=2,color=color),
                name=f"{f['trou']} ({f['type']})",
                hovertemplate=f"<b>{f['trou']}</b><br>{f['type']}<br>Prof:{f['profondeur']}m<br>Au:{f['Au_max_ppb']}ppb"))
        fig3d.update_layout(scene=dict(xaxis_title='Easting',yaxis_title='Northing',zaxis_title='Élévation',bgcolor='#1A1A2E'),
            title=f"Modèle 3D — {NOM_PROSPECT}",height=600,paper_bgcolor='#1A1A2E',font=dict(color='white'))
        st.plotly_chart(fig3d,use_container_width=True)
        st.info("🖱️ Clic gauche=rotation | Scroll=zoom | Clic droit=déplacement")
    elif vue=='Modèle de blocs 3D':
        st.info("🧱 Modèle de blocs 3D — Teneur en or (Au ppb)")
        nx,ny,nz=8,8,6
        x_bl=np.linspace(BASE_E-300,BASE_E+300,nx); y_bl=np.linspace(BASE_N-300,BASE_N+300,ny); z_bl=np.linspace(0,100,nz)
        blocs=[]
        for xi in x_bl:
            for yi in y_bl:
                for zi in z_bl:
                    dist_min=min([np.sqrt((xi-f['easting'])**2+(yi-f['northing'])**2) for _,f in df_forages.iterrows()])
                    au_bloc=max(0,np.random.lognormal(2,1)-dist_min/100)
                    blocs.append({'x':xi,'y':yi,'z':zi,'Au':round(au_bloc,1)})
        df_blocs=pd.DataFrame(blocs)
        fig_bl=go.Figure(data=go.Scatter3d(
            x=df_blocs['x'],y=df_blocs['y'],z=df_blocs['z'],mode='markers',
            marker=dict(size=6,color=df_blocs['Au'],colorscale='Viridis',colorbar=dict(title='Au (ppb)'),opacity=0.7),
            text=df_blocs['Au'].apply(lambda v:f"Au: {v:.1f} ppb"),
            hovertemplate="<b>Au: %{text}</b><br>E:%{x:.0f} N:%{y:.0f} Z:%{z:.0f}m"))
        fig_bl.update_layout(scene=dict(xaxis_title='Easting',yaxis_title='Northing',zaxis_title='Profondeur'),
            title=f"Modèle blocs — {NOM_PROSPECT}",height=600)
        st.plotly_chart(fig_bl,use_container_width=True)
        c1,c2,c3=st.columns(3)
        c1.metric("Blocs modélisés",len(df_blocs))
        c2.metric("Au moyen",f"{df_blocs['Au'].mean():.1f} ppb")
        c3.metric("Au max bloc",f"{df_blocs['Au'].max():.1f} ppb")
    elif vue=='2D Plan':
        fig2d,ax2d=plt.subplots(figsize=(10,8))
        for _,f in df_forages.iterrows():
            inc_rad=np.radians(abs(f['inclinaison'])); az_rad=np.radians(f['azimut'])
            depths=np.linspace(0,f['profondeur'],20)
            xs=f['easting']+depths*np.sin(az_rad)*np.cos(inc_rad)
            ys=f['northing']+depths*np.cos(az_rad)*np.cos(inc_rad)
            color=type_colors_3d.get(f['type'],'gray')
            ax2d.plot(xs,ys,color=color,linewidth=2)
            ax2d.scatter(f['easting'],f['northing'],color=color,s=80,zorder=3,edgecolors='black')
            ax2d.text(f['easting'],f['northing']+8,f['trou'],fontsize=7,ha='center')
        ax2d.set_xlabel("Easting (m)"); ax2d.set_ylabel("Northing (m)")
        ax2d.set_title("Vue en plan 2D",fontsize=12,fontweight='bold'); ax2d.grid(True,linestyle=':',alpha=0.4)
        legend_e=[mpatches.Patch(color=c,label=t) for t,c in type_colors_3d.items()]
        ax2d.legend(handles=legend_e,title='Type forage',fontsize=9)
        plt.tight_layout(); st.pyplot(fig2d)
    else:
        fig2ds,ax2ds=plt.subplots(figsize=(10,6))
        for _,f in df_forages.iterrows():
            inc_rad=np.radians(abs(f['inclinaison']))
            depths=np.linspace(0,f['profondeur'],20)
            xs=depths*np.cos(inc_rad); zs=f['elevation']-depths*np.sin(inc_rad)
            color=type_colors_3d.get(f['type'],'gray')
            ax2ds.plot(xs,zs,color=color,linewidth=2)
            ax2ds.scatter(0,f['elevation'],color=color,s=80,zorder=3,edgecolors='black')
            ax2ds.text(1,f['elevation']+1,f['trou'],fontsize=7)
        ax2ds.set_xlabel("Distance (m)"); ax2ds.set_ylabel("Élévation (m)")
        ax2ds.set_title("Vue en section 2D",fontsize=12,fontweight='bold'); ax2ds.grid(True,linestyle=':',alpha=0.4)
        plt.tight_layout(); st.pyplot(fig2ds)

# ══ TAB 8 — PLANIFICATION ════════════════════════════════════════════════════
with tabs[7]:
    st.subheader("📋 Planification & Programme Infill/Extension")
    col1,col2=st.columns(2)
    with col1:
        statut_counts=df_forages['statut'].value_counts()
        fig_s,ax_s=plt.subplots(figsize=(5,4))
        ax_s.pie(statut_counts.values,labels=statut_counts.index,colors=['#4CAF50','#FF9800','#2196F3'],autopct='%1.0f%%',startangle=90)
        ax_s.set_title("Statut des forages"); st.pyplot(fig_s)
    with col2:
        esp_act=st.slider("Espacement actuel (m)",50,400,200)
        esp_inf=st.slider("Espacement infill (m)",25,200,100)
        nb_inf=int((esp_act/esp_inf-1)*len(df_forages[df_forages['statut']=='Complété']))
        nb_ext=st.slider("Trous extension",0,20,5)
        cout_m=st.number_input("Coût/mètre (USD)",50,500,150)
        prof_moy=df_forages['profondeur'].mean()
        cout_tot=(nb_inf+nb_ext)*prof_moy*cout_m
        c1,c2,c3=st.columns(3)
        c1.metric("Infill nécessaires",nb_inf)
        c2.metric("Extension",nb_ext)
        c3.metric("Coût estimé",f"${cout_tot:,.0f}")
    st.dataframe(df_forages[['trou','type','profondeur','azimut','inclinaison','statut','equipe','Au_max_ppb']],use_container_width=True)

# ══ TAB 9 — SIMULATION DÉVIATION ════════════════════════════════════════════
with tabs[8]:
    st.subheader("🔄 Simulation Déviation")
    col1,col2=st.columns([1,2])
    with col1:
        trou_dev=st.selectbox("Trou",df_forages['trou'].tolist(),key='dev')
        f_dev=df_forages[df_forages['trou']==trou_dev].iloc[0]
        az_init=st.slider("Azimut (°)",0,360,int(f_dev['azimut']))
        inc_init=st.slider("Inclinaison (°)",-90,-45,int(f_dev['inclinaison']))
        dev_az=st.slider("Déviation Az/10m",0.0,5.0,1.5,0.1)
        dev_inc=st.slider("Déviation Inc/10m",0.0,3.0,0.8,0.1)
        prof_sim=st.slider("Profondeur simulée",10,200,int(f_dev['profondeur']))
    with col2:
        depths=np.arange(0,prof_sim+1,1)
        az_v=az_init+dev_az*depths/10*np.sin(depths/20)
        inc_v=inc_init+dev_inc*depths/10*np.cos(depths/15)
        xp=np.cumsum(np.sin(np.radians(az_init))*np.cos(np.radians(inc_init))*np.ones(len(depths)))
        yp=np.cumsum(np.cos(np.radians(az_init))*np.cos(np.radians(inc_init))*np.ones(len(depths)))
        zp=f_dev['elevation']-np.cumsum(np.sin(np.radians(abs(inc_init)))*np.ones(len(depths)))
        xd=np.cumsum(np.sin(np.radians(az_v))*np.cos(np.radians(inc_v)))
        yd=np.cumsum(np.cos(np.radians(az_v))*np.cos(np.radians(inc_v)))
        zd=f_dev['elevation']-np.cumsum(np.sin(np.radians(abs(inc_v))))
        fig_dev=go.Figure()
        fig_dev.add_trace(go.Scatter3d(x=xp+f_dev['easting'],y=yp+f_dev['northing'],z=zp,mode='lines',line=dict(color='blue',width=4,dash='dash'),name='Planifiée'))
        fig_dev.add_trace(go.Scatter3d(x=xd+f_dev['easting'],y=yd+f_dev['northing'],z=zd,mode='lines',line=dict(color='red',width=4),name='Déviée'))
        fig_dev.update_layout(scene=dict(xaxis_title='Easting',yaxis_title='Northing',zaxis_title='Élévation'),title=f"Déviation — {trou_dev}",height=500)
        st.plotly_chart(fig_dev,use_container_width=True)
        dev_tot=np.sqrt((xd[-1]-xp[-1])**2+(yd[-1]-yp[-1])**2)
        if dev_tot>10: st.error(f"⚠️ Déviation : **{dev_tot:.1f} m** — Correction nécessaire !")
        else: st.success(f"✅ Déviation : **{dev_tot:.1f} m** — Acceptable")

# ══ TAB 10 — SURVEILLANCE ════════════════════════════════════════════════════
with tabs[9]:
    st.subheader("📡 Surveillance des Forages")
    col1,col2=st.columns(2)
    with col1:
        st.markdown("### 🏔️ Forages de surface")
        for _,f in df_forages[df_forages['type'].isin(['RC','Aircore'])].iterrows():
            icon='🟢' if f['statut']=='Complété' else '🟡' if f['statut']=='En cours' else '🔵'
            st.markdown(f"**{icon} {f['trou']}** ({f['type']}) | {f['profondeur']}m | Au:{f['Au_max_ppb']:.0f} ppb")
    with col2:
        st.markdown("### ⛏️ Diamond")
        for _,f in df_forages[df_forages['type']=='Diamond'].iterrows():
            icon='🟢' if f['statut']=='Complété' else '🟡' if f['statut']=='En cours' else '🔵'
            st.markdown(f"**{icon} {f['trou']}** | {f['profondeur']}m")
            st.progress(min(1.0,f['profondeur']/200),text=f"{f['profondeur']}m/200m")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Actifs",int((df_forages['statut']=='En cours').sum()))
    c2.metric("Au max",f"{df_forages['Au_max_ppb'].max():.0f} ppb")
    c3.metric("RC",int((df_forages['type']=='RC').sum()))
    c4.metric("Diamond",int((df_forages['type']=='Diamond').sum()))

# ══ TAB 11 — ESSAI SGI ══════════════════════════════════════════════════════
with tabs[10]:
    st.subheader("🧪 Essai SGI")
    col1,col2=st.columns([1,2])
    with col1:
        trou_sgi=st.selectbox("Trou",df_forages['trou'].tolist(),key='sgi')
        seuil_au=st.number_input("Seuil Au (ppb)",10,1000,100)
        seuil_cu=st.number_input("Seuil Cu (ppm)",5,200,50)
    with col2:
        ints_sgi=df_intervals[df_intervals['trou']==trou_sgi].sort_values('de').copy()
        ints_sgi['mineralisé']=ints_sgi['Au_ppb']>=seuil_au
        total_m=ints_sgi['a'].max()-ints_sgi['de'].min()
        m_miner=ints_sgi[ints_sgi['mineralisé']].apply(lambda r:r['a']-r['de'],axis=1).sum()
        pct_miner=m_miner/total_m*100 if total_m>0 else 0
        c1,c2,c3=st.columns(3)
        c1.metric("Mètres minéralisés",f"{m_miner:.1f} m")
        c2.metric("% minéralisé",f"{pct_miner:.1f}%")
        c3.metric("Au max",f"{ints_sgi['Au_ppb'].max():.1f} ppb")
        fig_sgi,axes_sgi=plt.subplots(1,4,figsize=(14,10),sharey=True)
        f_sgi=df_forages[df_forages['trou']==trou_sgi].iloc[0]
        for _,i in ints_sgi.iterrows():
            color=ALTER_COLORS.get(i['alteration'],'#888')
            axes_sgi[0].fill_betweenx([i['de'],i['a']],0,1,color=color,alpha=0.85)
            mid=(i['de']+i['a'])/2
            axes_sgi[0].text(0.5,mid,i['alteration'][:10],ha='center',va='center',fontsize=5.5,fontweight='bold')
            axes_sgi[0].plot([0,1],[i['de'],i['de']],'k-',linewidth=0.3)
            axes_sgi[0].text(-0.05,i['de'],f"{i['de']}m",ha='right',fontsize=5.5)
        axes_sgi[0].set_ylim(f_sgi['profondeur'],0); axes_sgi[0].set_xticks([])
        axes_sgi[0].set_title("Altération",fontsize=9,fontweight='bold'); axes_sgi[0].set_ylabel("Profondeur (m)")
        for _,i in ints_sgi.iterrows():
            color=MINER_COLORS.get(i['mineralisation'],'#888')
            axes_sgi[1].fill_betweenx([i['de'],i['a']],0,1,color=color,alpha=0.85)
            mid=(i['de']+i['a'])/2
            axes_sgi[1].text(0.5,mid,i['mineralisation'][:12],ha='center',va='center',fontsize=5,fontweight='bold')
            axes_sgi[1].plot([0,1],[i['de'],i['de']],'k-',linewidth=0.3)
        axes_sgi[1].set_ylim(f_sgi['profondeur'],0); axes_sgi[1].set_xticks([])
        axes_sgi[1].set_title("Minéralisation",fontsize=9,fontweight='bold')
        colors_au=['#FFD700' if v>=seuil_au else '#EEE' for v in ints_sgi['Au_ppb']]
        axes_sgi[2].barh([(i['de']+i['a'])/2 for _,i in ints_sgi.iterrows()],ints_sgi['Au_ppb'].values,
                         height=[(i['a']-i['de'])*0.8 for _,i in ints_sgi.iterrows()],color=colors_au,edgecolor='orange',linewidth=0.5)
        axes_sgi[2].axvline(x=seuil_au,color='red',linestyle='--',linewidth=1.5,label=f'{seuil_au}ppb')
        axes_sgi[2].set_xlabel("Au (ppb)"); axes_sgi[2].set_title("Or",fontsize=9,fontweight='bold'); axes_sgi[2].legend(fontsize=7)
        colors_cu=['#B87333' if v>=seuil_cu else '#EEE' for v in ints_sgi['Cu_ppm']]
        axes_sgi[3].barh([(i['de']+i['a'])/2 for _,i in ints_sgi.iterrows()],ints_sgi['Cu_ppm'].values,
                         height=[(i['a']-i['de'])*0.8 for _,i in ints_sgi.iterrows()],color=colors_cu,edgecolor='brown',linewidth=0.5)
        axes_sgi[3].axvline(x=seuil_cu,color='blue',linestyle='--',linewidth=1.5,label=f'{seuil_cu}ppm')
        axes_sgi[3].set_xlabel("Cu (ppm)"); axes_sgi[3].set_title("Cuivre",fontsize=9,fontweight='bold'); axes_sgi[3].legend(fontsize=7)
        plt.suptitle(f"SGI — {trou_sgi} | {f_sgi['type']} | {f_sgi['profondeur']}m",fontsize=11,fontweight='bold')
        plt.tight_layout(); st.pyplot(fig_sgi)
    st.markdown("### 📊 Tableau SGI — Minéralisation & Or")
    sgi_tab=df_intervals[df_intervals['trou']==trou_sgi][['de','a','lithologie','alteration','mineralisation','Au_ppb','Cu_ppm','As_ppm','mineralisé']].copy()
    sgi_tab.columns=['De(m)','A(m)','Lithologie','Altération','Minéralisation','Au(ppb)','Cu(ppm)','As(ppm)','Minéralisé']
    st.dataframe(sgi_tab.style.map(lambda v:'background-color:#FFD700;color:black' if v==True else 'background-color:#F5F5F5' if v==False else '',subset=['Minéralisé']).format({'Au(ppb)':'{:.2f}','Cu(ppm)':'{:.1f}','As(ppm)':'{:.1f}'}),use_container_width=True)

# ══ TAB 12 — ESTIMATION TENEURS ══════════════════════════════════════════════
with tabs[11]:
    st.subheader("💰 Estimation des Teneurs en Or")
    st.info(f"**Projet :** {NOM_PROSPECT} | **Permis :** {NOM_PERMIS}")

    col1,col2=st.columns([1,2])
    with col1:
        methode=st.selectbox("Méthode d'estimation",['Moyenne pondérée','Inverse Distance (IDW)','Krigeage (simplifié)'])
        seuil_coupure=st.number_input("Teneur de coupure (ppb)",10,500,100)
        densite=st.number_input("Densité moyenne (t/m³)",1.5,3.5,2.7,0.1)
        largeur_zone=st.number_input("Largeur zone minéralisée (m)",10,200,50)
        longueur_zone=st.number_input("Longueur zone minéralisée (m)",50,2000,500)

    with col2:
        # Calcul estimation
        df_est=df_forages.copy()
        df_est['Au_est']=df_est.apply(lambda r: df_intervals[(df_intervals['trou']==r['trou'])&
                                                               (df_intervals['Au_ppb']>=seuil_coupure)]['Au_ppb'].mean() if len(df_intervals[(df_intervals['trou']==r['trou'])&(df_intervals['Au_ppb']>=seuil_coupure)])>0 else 0,axis=1)
        df_est['longueur_miner']=df_est.apply(lambda r: df_intervals[(df_intervals['trou']==r['trou'])&(df_intervals['Au_ppb']>=seuil_coupure)].apply(lambda x:x['a']-x['de'],axis=1).sum(),axis=1)

        if methode=='Moyenne pondérée':
            au_global=np.average(df_est[df_est['Au_est']>0]['Au_est'],weights=df_est[df_est['Au_est']>0]['longueur_miner']+0.001) if len(df_est[df_est['Au_est']>0])>0 else 0
        elif methode=='Inverse Distance (IDW)':
            au_global=df_est[df_est['Au_est']>0]['Au_est'].mean() if len(df_est[df_est['Au_est']>0])>0 else 0
        else:
            au_global=df_est[df_est['Au_est']>0]['Au_est'].median() if len(df_est[df_est['Au_est']>0])>0 else 0

        prof_moy_miner=df_est['longueur_miner'].mean()
        volume=largeur_zone*longueur_zone*prof_moy_miner
        tonnage=volume*densite
        metal_ppb=au_global*tonnage/1e6
        metal_oz=metal_ppb/31.1035

        c1,c2,c3,c4=st.columns(4)
        c1.metric("Au moyen estimé",f"{au_global:.1f} ppb")
        c2.metric("Volume estimé",f"{volume:,.0f} m³")
        c3.metric("Tonnage estimé",f"{tonnage/1000:,.0f} kt")
        c4.metric("Métal Au",f"{metal_oz:,.0f} oz")

        # Graphique estimation par trou
        df_est_plot=df_est[df_est['Au_est']>0].sort_values('Au_est',ascending=False)
        fig_est,axes_est=plt.subplots(1,2,figsize=(12,5))
        colors_est=['#FFD700' if v>=au_global else '#AAAAAA' for v in df_est_plot['Au_est']]
        axes_est[0].bar(df_est_plot['trou'],df_est_plot['Au_est'],color=colors_est,edgecolor='black',linewidth=0.5)
        axes_est[0].axhline(y=au_global,color='red',linestyle='--',linewidth=2,label=f'Moy: {au_global:.1f} ppb')
        axes_est[0].axhline(y=seuil_coupure,color='blue',linestyle=':',linewidth=1.5,label=f'Coupure: {seuil_coupure} ppb')
        axes_est[0].set_ylabel("Au (ppb)"); axes_est[0].set_title(f"Au estimé par trou — {methode}",fontsize=10,fontweight='bold')
        axes_est[0].legend(fontsize=8); plt.setp(axes_est[0].xaxis.get_majorticklabels(),rotation=45,fontsize=7)
        axes_est[1].bar(df_est_plot['trou'],df_est_plot['longueur_miner'],color='#2196F3',edgecolor='black',linewidth=0.5)
        axes_est[1].set_ylabel("Longueur minéralisée (m)"); axes_est[1].set_title("Longueur minéralisée par trou",fontsize=10,fontweight='bold')
        plt.setp(axes_est[1].xaxis.get_majorticklabels(),rotation=45,fontsize=7)
        plt.suptitle(f"Estimation des teneurs — {NOM_PROSPECT}",fontsize=12,fontweight='bold')
        plt.tight_layout(); st.pyplot(fig_est)

    st.markdown("### 📋 Tableau d'estimation par trou")
    df_est_display=df_est[['trou','type','profondeur','Au_est','longueur_miner','statut']].copy()
    df_est_display.columns=['Trou','Type','Prof.(m)','Au moy.(ppb)','Long. minér.(m)','Statut']
    df_est_display['Au moy.(ppb)']=df_est_display['Au moy.(ppb)'].round(1)
    df_est_display['Long. minér.(m)']=df_est_display['Long. minér.(m)'].round(1)
    st.dataframe(df_est_display,use_container_width=True)

    st.markdown(f"""
    ### 📊 Résumé de l'estimation — {methode}
    | Paramètre | Valeur |
    |-----------|--------|
    | Teneur de coupure | {seuil_coupure} ppb Au |
    | Teneur moyenne estimée | {au_global:.1f} ppb Au |
    | Volume minéralisé | {volume:,.0f} m³ |
    | Tonnage minéralisé | {tonnage/1000:,.0f} kt |
    | Contenu en métal | {metal_oz:,.0f} oz Au |
    | Méthode | {methode} |
    """)

# ══ TAB 13 — CARTOGRAPHIE TERRAIN ════════════════════════════════════════════
with tabs[12]:
    st.subheader(f"🗾 Cartographie Terrain — {NOM_PROSPECT}")
    st.markdown(f"**Permis :** {NOM_PERMIS} | **Date :** {datetime.date.today()}")

    vue_carto=st.radio("Affichage",['Carte lithologique digitalisée','Carte structurale digitalisée','Tableau roches','Tableau structures'],horizontal=True)

    if vue_carto=='Carte lithologique digitalisée':
        fig_carto,ax_carto=plt.subplots(figsize=(12,10))
        fig_carto.patch.set_facecolor('#F5F5F0'); ax_carto.set_facecolor('#D4E8FF')
        # Polygones lithologiques digitalisés
        np.random.seed(20)
        for litho,color in LITHO_COLORS.items():
            cx=BASE_E+np.random.uniform(-300,300); cy=BASE_N+np.random.uniform(-300,300)
            angles=np.linspace(0,2*np.pi,8)
            rx=np.random.uniform(80,200); ry=np.random.uniform(60,150)
            px=cx+rx*np.cos(angles)+np.random.normal(0,20,8)
            py=cy+ry*np.sin(angles)+np.random.normal(0,20,8)
            poly=plt.Polygon(list(zip(px,py)),closed=True,facecolor=color,edgecolor='black',linewidth=1.5,alpha=0.7)
            ax_carto.add_patch(poly)
            ax_carto.text(cx,cy,litho[:12],ha='center',va='center',fontsize=7,fontweight='bold',
                         bbox=dict(boxstyle='round',facecolor='white',alpha=0.7))
        # Forages
        for _,f in df_forages.iterrows():
            marker={'RC':'^','Aircore':'o','Diamond':'s'}.get(f['type'],'o')
            ax_carto.scatter(f['easting'],f['northing'],c='black',s=80,marker=marker,zorder=4)
            ax_carto.text(f['easting'],f['northing']+8,f['trou'],fontsize=6,ha='center',color='#1A237E',fontweight='bold')
        # Nord, échelle, légende
        xmax=df_forages['easting'].max()+100; ymax=df_forages['northing'].max()+100
        xmin=df_forages['easting'].min()-100; ymin=df_forages['northing'].min()-100
        ax_carto.annotate('',xy=(xmax+80,ymax),xytext=(xmax+80,ymax-60),arrowprops=dict(arrowstyle='->',color='black',lw=2.5))
        ax_carto.text(xmax+80,ymax+15,'N',ha='center',fontsize=16,fontweight='bold')
        ax_carto.plot([xmin,xmin+200],[ymin-50,ymin-50],'k-',linewidth=4)
        ax_carto.plot([xmin,xmin],[ymin-55,ymin-45],'k-',linewidth=2)
        ax_carto.plot([xmin+200,xmin+200],[ymin-55,ymin-45],'k-',linewidth=2)
        ax_carto.text(xmin+100,ymin-70,'0        200 m',ha='center',fontsize=9,fontweight='bold')
        legend_patches=[mpatches.Patch(color=c,label=l,alpha=0.7) for l,c in LITHO_COLORS.items()]
        ax_carto.legend(handles=legend_patches,loc='lower right',fontsize=8,title='Lithologie',framealpha=0.95,edgecolor='black')
        ax_carto.set_xlim(xmin-50,xmax+150); ax_carto.set_ylim(ymin-100,ymax+60)
        ax_carto.set_xlabel("Easting UTM (m)"); ax_carto.set_ylabel("Northing UTM (m)")
        ax_carto.set_title(f"Carte lithologique digitalisée — {NOM_PROSPECT}\n{NOM_PERMIS}",fontsize=12,fontweight='bold')
        ax_carto.grid(True,linestyle='--',alpha=0.3,color='gray')
        # Coordonnées sur les axes
        ax_carto.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f'{x:.0f}'))
        ax_carto.yaxis.set_major_formatter(plt.FuncFormatter(lambda y,_:f'{y:.0f}'))
        plt.tight_layout(); st.pyplot(fig_carto)

    elif vue_carto=='Carte structurale digitalisée':
        fig_cs,ax_cs=plt.subplots(figsize=(12,10))
        fig_cs.patch.set_facecolor('#F5F5F0'); ax_cs.set_facecolor('#F0EDE0')
        # Structures digitalisées
        for _,s in structures_df.head(20).iterrows():
            angle=s['direction']; length=min(s['longueur_m'],400)
            x1=s['easting']; y1=s['northing']
            x2=x1+length*np.cos(np.radians(angle)); y2=y1+length*np.sin(np.radians(angle))
            color=STRUCT_COLORS.get(s['type'],'#888')
            ls='-' if 'Faille' in s['type'] else '--' if 'Veine' in s['type'] else ':'
            lw=3 if 'Faille' in s['type'] else 2
            ax_cs.plot([x1,x2],[y1,y2],color=color,linewidth=lw,linestyle=ls,label=s['type'])
            # Symbole pendage (tadpole)
            mx,my=(x1+x2)/2,(y1+y2)/2
            ax_cs.plot(mx,my,'k|',markersize=8,markeredgewidth=2)
            ax_cs.text(mx+5,my+5,f"{s['pendage']:.0f}°",fontsize=6,color=color,fontweight='bold')
        for _,f in df_forages.iterrows():
            marker={'RC':'^','Aircore':'o','Diamond':'s'}.get(f['type'],'o')
            ax_cs.scatter(f['easting'],f['northing'],c='black',s=80,marker=marker,zorder=4)
            ax_cs.text(f['easting'],f['northing']+8,f['trou'],fontsize=6,ha='center')
        xmax=df_forages['easting'].max()+100; ymax=df_forages['northing'].max()+100
        xmin=df_forages['easting'].min()-100; ymin=df_forages['northing'].min()-100
        ax_cs.annotate('',xy=(xmax+80,ymax),xytext=(xmax+80,ymax-60),arrowprops=dict(arrowstyle='->',color='black',lw=2.5))
        ax_cs.text(xmax+80,ymax+15,'N',ha='center',fontsize=16,fontweight='bold')
        ax_cs.plot([xmin,xmin+200],[ymin-50,ymin-50],'k-',linewidth=4)
        ax_cs.text(xmin+100,ymin-70,'0        200 m',ha='center',fontsize=9,fontweight='bold')
        handles,labels=ax_cs.get_legend_handles_labels()
        by_label=dict(zip(labels,handles))
        ax_cs.legend(by_label.values(),by_label.keys(),loc='lower right',fontsize=8,title='Structures',framealpha=0.95)
        ax_cs.set_xlim(xmin-50,xmax+150); ax_cs.set_ylim(ymin-100,ymax+60)
        ax_cs.set_xlabel("Easting UTM (m)"); ax_cs.set_ylabel("Northing UTM (m)")
        ax_cs.set_title(f"Carte structurale digitalisée — {NOM_PROSPECT}\n{NOM_PERMIS}",fontsize=12,fontweight='bold')
        ax_cs.grid(True,linestyle=':',alpha=0.3)
        plt.tight_layout(); st.pyplot(fig_cs)

    elif vue_carto=='Tableau roches':
        st.markdown("### 📋 Tableau des roches levées sur le terrain")
        st.dataframe(roches_terrain.rename(columns={
            'id':'N°','type_roche':'Type de roche','easting':'Easting (m)','northing':'Northing (m)',
            'elevation':'Élévation (m)','description':'Description','alteration':'Altération',
            'Au_sol_ppb':'Au sol (ppb)','observateur':'Observateur','date':'Date'
        }),use_container_width=True)
        csv_r=roches_terrain.to_csv(index=False)
        st.download_button("📥 Télécharger tableau roches",data=csv_r,file_name="roches_terrain.csv",mime='text/csv')

    else:
        st.markdown("### 📋 Tableau des structures levées sur le terrain")
        st.dataframe(structures_df.rename(columns={
            'id':'N°','type':'Type de structure','easting':'Easting (m)','northing':'Northing (m)',
            'direction':'Direction (°)','pendage':'Pendage (°)','sens_pendage':'Sens pendage',
            'longueur_m':'Longueur (m)','porteur_miner':'Porteur minéralisation'
        }),use_container_width=True)
        csv_s=structures_df.to_csv(index=False)
        st.download_button("📥 Télécharger tableau structures",data=csv_s,file_name="structures_terrain.csv",mime='text/csv')

# ══ TAB 14 — GRAPHIQUES STRUCTURAUX ══════════════════════════════════════════
with tabs[13]:
    st.subheader("📐 Graphiques Structuraux — Analyse géostatistique")
    graph_type=st.radio("Type de graphique",['Rosace','Stéréonet (Schmidt)','Tadpole Plot','Dips Plot','Section Drilling','Logue de section'],horizontal=True)

    if graph_type=='Rosace':
        st.markdown("### 🌹 Diagramme en rosace — Directions des structures")
        type_filtre=st.multiselect("Filtrer par type",STRUCTURES,default=STRUCTURES)
        df_rose=structures_df[structures_df['type'].isin(type_filtre)]
        fig_rose=plt.figure(figsize=(8,8))
        ax_rose=fig_rose.add_subplot(111,projection='polar')
        directions_rad=np.radians(df_rose['direction'].values)
        bins=np.linspace(0,2*np.pi,37)
        hist,_=np.histogram(directions_rad,bins=bins)
        hist_sym=hist+np.roll(hist,18)
        theta=bins[:-1]
        width=bins[1]-bins[0]
        bars=ax_rose.bar(theta,hist_sym,width=width,color='steelblue',edgecolor='black',linewidth=0.5,alpha=0.8)
        ax_rose.set_theta_zero_location('N'); ax_rose.set_theta_direction(-1)
        ax_rose.set_title(f"Rosace des directions — {NOM_PROSPECT}\nn={len(df_rose)} mesures",fontsize=12,fontweight='bold',pad=20)
        ax_rose.set_xticklabels(['N','NE','E','SE','S','SO','O','NO'])
        plt.tight_layout(); st.pyplot(fig_rose)
        dir_dom=df_rose['direction'].mode()[0] if len(df_rose)>0 else 0
        st.info(f"**Direction dominante :** {dir_dom:.0f}° | **Nombre de mesures :** {len(df_rose)}")

    elif graph_type=='Stéréonet (Schmidt)':
        st.markdown("### 🎯 Stéréonet — Projection hémisphérique (Schmidt)")
        type_stereo=st.selectbox("Type de structure",STRUCTURES)
        df_stereo=structures_df[structures_df['type']==type_stereo]
        fig_st,ax_st=plt.subplots(figsize=(8,8))
        circle=plt.Circle((0,0),1,fill=False,color='black',linewidth=2)
        ax_st.add_patch(circle)
        ax_st.axhline(y=0,color='gray',linewidth=0.5,alpha=0.5)
        ax_st.axvline(x=0,color='gray',linewidth=0.5,alpha=0.5)
        for _,s in df_stereo.iterrows():
            az=np.radians(s['direction']); dip=np.radians(s['pendage'])
            r=np.tan((np.pi/2-dip)/2) if dip<np.pi/2 else 0
            x=r*np.sin(az); y=r*np.cos(az)
            ax_st.plot(x,y,'ro',markersize=8,alpha=0.7,zorder=3)
            ax_st.plot(-x,-y,'b+',markersize=8,alpha=0.5,zorder=3)
        # Labels cardinaux
        for label,pos in [('N',(0,1.1)),('S',(0,-1.15)),('E',(1.15,0)),('O',(-1.2,0))]:
            ax_st.text(pos[0],pos[1],label,ha='center',va='center',fontsize=12,fontweight='bold')
        for i in range(1,5):
            r=i/4
            circ=plt.Circle((0,0),r,fill=False,color='gray',linewidth=0.3,alpha=0.5)
            ax_st.add_patch(circ)
            ax_st.text(0,r,f'{90-i*22}°',fontsize=7,ha='center',color='gray')
        ax_st.set_xlim(-1.3,1.3); ax_st.set_ylim(-1.3,1.3)
        ax_st.set_aspect('equal'); ax_st.axis('off')
        ax_st.set_title(f"Stéréonet — {type_stereo}\n{NOM_PROSPECT} | n={len(df_stereo)}",fontsize=12,fontweight='bold')
        legend=[plt.Line2D([0],[0],marker='o',color='w',markerfacecolor='red',markersize=10,label='Pôles'),
                plt.Line2D([0],[0],marker='+',color='w',markerfacecolor='blue',markersize=10,label='Plans conjugués')]
        ax_st.legend(handles=legend,loc='lower right',fontsize=9)
        plt.tight_layout(); st.pyplot(fig_st)

    elif graph_type=='Tadpole Plot':
        st.markdown("### 🦎 Tadpole Plot — Direction/Pendage en profondeur")
        trou_tadpole=st.selectbox("Trou",df_forages['trou'].tolist(),key='tadpole')
        f_tp=df_forages[df_forages['trou']==trou_tadpole].iloc[0]
        ints_tp=df_intervals[df_intervals['trou']==trou_tadpole].sort_values('de')
        struct_tp=structures_df.sample(min(len(structures_df),len(ints_tp)),random_state=42).reset_index(drop=True)
        depths_tp=ints_tp['de'].values[:len(struct_tp)]
        fig_tp,axes_tp=plt.subplots(1,2,figsize=(10,10),sharey=True)
        for i,(depth,(_,s)) in enumerate(zip(depths_tp,struct_tp.iterrows())):
            dir_rad=np.radians(s['direction']); pend=s['pendage']
            dx=0.3*np.cos(dir_rad); dy=0.3*np.sin(dir_rad)
            axes_tp[0].plot(s['direction'],depth,'ko',markersize=10,zorder=3)
            axes_tp[0].annotate('',xy=(s['direction']+20,depth),xytext=(s['direction'],depth),
                               arrowprops=dict(arrowstyle='-',color='black',lw=2))
        axes_tp[0].set_xlabel("Direction (°)"); axes_tp[0].set_ylabel("Profondeur (m)")
        axes_tp[0].set_title(f"Direction\n{trou_tadpole}",fontsize=9,fontweight='bold')
        axes_tp[0].set_xlim(0,360); axes_tp[0].set_ylim(f_tp['profondeur'],0)
        axes_tp[0].grid(True,linestyle=':',alpha=0.5)
        for i,(depth,(_,s)) in enumerate(zip(depths_tp,struct_tp.iterrows())):
            axes_tp[1].plot(s['pendage'],depth,'rs',markersize=8,zorder=3)
        axes_tp[1].set_xlabel("Pendage (°)")
        axes_tp[1].set_title(f"Pendage\n{trou_tadpole}",fontsize=9,fontweight='bold')
        axes_tp[1].set_xlim(0,90); axes_tp[1].grid(True,linestyle=':',alpha=0.5)
        plt.suptitle(f"Tadpole Plot — {trou_tadpole} | {f_tp['type']}",fontsize=11,fontweight='bold')
        plt.tight_layout(); st.pyplot(fig_tp)

    elif graph_type=='Dips Plot':
        st.markdown("### 📊 Dips Plot — Analyse statistique des pendages")
        fig_dips,axes_dips=plt.subplots(2,2,figsize=(12,10))
        axes_dips[0,0].hist(structures_df['pendage'],bins=20,color='#2196F3',edgecolor='black',linewidth=0.5)
        axes_dips[0,0].set_xlabel("Pendage (°)"); axes_dips[0,0].set_ylabel("Fréquence")
        axes_dips[0,0].set_title("Distribution des pendages",fontsize=10,fontweight='bold')
        axes_dips[0,0].axvline(structures_df['pendage'].mean(),color='red',linestyle='--',linewidth=2,label=f"Moy: {structures_df['pendage'].mean():.1f}°")
        axes_dips[0,0].legend(fontsize=8)
        axes_dips[0,1].hist(structures_df['direction'],bins=36,color='#FF9800',edgecolor='black',linewidth=0.5)
        axes_dips[0,1].set_xlabel("Direction (°)"); axes_dips[0,1].set_ylabel("Fréquence")
        axes_dips[0,1].set_title("Distribution des directions",fontsize=10,fontweight='bold')
        for struct_type,color_s in [('Faille normale','red'),('Faille inverse','blue'),('Cisaillement','orange'),('Veine de quartz','gold')]:
            sub=structures_df[structures_df['type']==struct_type]
            if len(sub)>0:
                axes_dips[1,0].scatter(sub['direction'],sub['pendage'],c=color_s,s=60,label=struct_type,alpha=0.7,edgecolors='black',linewidths=0.5)
        axes_dips[1,0].set_xlabel("Direction (°)"); axes_dips[1,0].set_ylabel("Pendage (°)")
        axes_dips[1,0].set_title("Direction vs Pendage par type",fontsize=10,fontweight='bold')
        axes_dips[1,0].legend(fontsize=7); axes_dips[1,0].grid(True,linestyle=':',alpha=0.4)
        porteur=structures_df[structures_df['porteur_miner']==True]
        non_porteur=structures_df[structures_df['porteur_miner']==False]
        axes_dips[1,1].scatter(porteur['direction'],porteur['pendage'],c='gold',s=100,label='Porteur minéralisation',marker='*',zorder=3,edgecolors='orange')
        axes_dips[1,1].scatter(non_porteur['direction'],non_porteur['pendage'],c='gray',s=40,label='Non porteur',alpha=0.5,edgecolors='black',linewidths=0.3)
        axes_dips[1,1].set_xlabel("Direction (°)"); axes_dips[1,1].set_ylabel("Pendage (°)")
        axes_dips[1,1].set_title("Structures porteuses vs non-porteuses",fontsize=10,fontweight='bold')
        axes_dips[1,1].legend(fontsize=8); axes_dips[1,1].grid(True,linestyle=':',alpha=0.4)
        plt.suptitle(f"Dips Plot — Analyse structurale | {NOM_PROSPECT}",fontsize=12,fontweight='bold')
        plt.tight_layout(); st.pyplot(fig_dips)

    elif graph_type=='Section Drilling':
        st.markdown("### 🔩 Section Drilling — Logue de section")
        trou_sd=st.selectbox("Trou",df_forages['trou'].tolist(),key='secdrill')
        f_sd=df_forages[df_forages['trou']==trou_sd].iloc[0]
        ints_sd=df_intervals[df_intervals['trou']==trou_sd].sort_values('de')
        fig_sd,axes_sd=plt.subplots(1,5,figsize=(16,12),sharey=True)
        # Col 1 — Litho
        for _,i in ints_sd.iterrows():
            color=LITHO_COLORS.get(i['lithologie'],'#888')
            axes_sd[0].fill_betweenx([i['de'],i['a']],0,1,color=color,alpha=0.85)
            if i['mineralisé']: axes_sd[0].fill_betweenx([i['de'],i['a']],0,1,color='red',alpha=0.2,hatch='///')
            axes_sd[0].plot([0,1],[i['de'],i['de']],'k-',linewidth=0.3)
            mid=(i['de']+i['a'])/2
            axes_sd[0].text(0.5,mid,i['lithologie'][:8],ha='center',va='center',fontsize=5,fontweight='bold')
            axes_sd[0].text(-0.05,i['de'],f"{i['de']}m",ha='right',fontsize=5)
        axes_sd[0].set_ylim(f_sd['profondeur'],0); axes_sd[0].set_xticks([])
        axes_sd[0].set_title("Lithologie",fontsize=8,fontweight='bold'); axes_sd[0].set_ylabel("Profondeur (m)")
        # Col 2 — Altération
        for _,i in ints_sd.iterrows():
            color=ALTER_COLORS.get(i['alteration'],'#888')
            axes_sd[1].fill_betweenx([i['de'],i['a']],0,1,color=color,alpha=0.85)
            mid=(i['de']+i['a'])/2
            axes_sd[1].text(0.5,mid,i['alteration'][:8],ha='center',va='center',fontsize=4.5,fontweight='bold')
            axes_sd[1].plot([0,1],[i['de'],i['de']],'k-',linewidth=0.3)
        axes_sd[1].set_ylim(f_sd['profondeur'],0); axes_sd[1].set_xticks([])
        axes_sd[1].set_title("Altération",fontsize=8,fontweight='bold')
        # Col 3 — Minéralisation
        for _,i in ints_sd.iterrows():
            color=MINER_COLORS.get(i['mineralisation'],'#888')
            axes_sd[2].fill_betweenx([i['de'],i['a']],0,1,color=color,alpha=0.85)
            mid=(i['de']+i['a'])/2
            axes_sd[2].text(0.5,mid,i['mineralisation'][:10],ha='center',va='center',fontsize=4.5,fontweight='bold')
            axes_sd[2].plot([0,1],[i['de'],i['de']],'k-',linewidth=0.3)
        axes_sd[2].set_ylim(f_sd['profondeur'],0); axes_sd[2].set_xticks([])
        axes_sd[2].set_title("Minéralisation",fontsize=8,fontweight='bold')
        # Col 4 — Au
        colors_au=['#FFD700' if v>=100 else '#EEE' for v in ints_sd['Au_ppb']]
        axes_sd[3].barh([(i['de']+i['a'])/2 for _,i in ints_sd.iterrows()],ints_sd['Au_ppb'].values,
                        height=[(i['a']-i['de'])*0.8 for _,i in ints_sd.iterrows()],color=colors_au,edgecolor='orange',linewidth=0.5)
        axes_sd[3].axvline(x=100,color='red',linestyle='--',linewidth=1.5,label='100ppb')
        axes_sd[3].set_xlabel("Au(ppb)"); axes_sd[3].set_title("Or",fontsize=8,fontweight='bold'); axes_sd[3].legend(fontsize=6)
        # Col 5 — As
        axes_sd[4].barh([(i['de']+i['a'])/2 for _,i in ints_sd.iterrows()],ints_sd['As_ppm'].values,
                        height=[(i['a']-i['de'])*0.8 for _,i in ints_sd.iterrows()],color='#FF6B6B',edgecolor='red',linewidth=0.5)
        axes_sd[4].set_xlabel("As(ppm)"); axes_sd[4].set_title("Arsenic",fontsize=8,fontweight='bold')
        plt.suptitle(f"Section Drilling — {trou_sd} | {f_sd['type']} | Az:{f_sd['azimut']}° Inc:{f_sd['inclinaison']}° | Prof:{f_sd['profondeur']}m\n{NOM_PROSPECT}",fontsize=10,fontweight='bold')
        plt.tight_layout(); st.pyplot(fig_sd)

    else:  # Logue de section
        st.markdown("### 📋 Logue de section — Résumé par intervalle")
        trou_ls=st.selectbox("Trou",df_forages['trou'].tolist(),key='logsec')
        f_ls=df_forages[df_forages['trou']==trou_ls].iloc[0]
        ints_ls=df_intervals[df_intervals['trou']==trou_ls].sort_values('de')
        st.markdown(f"**{trou_ls}** | {f_ls['type']} | Prof: {f_ls['profondeur']}m | Az: {f_ls['azimut']}° | Inc: {f_ls['inclinaison']}°")
        logue_display=ints_ls[['de','a','lithologie','alteration','mineralisation','Au_ppb','Cu_ppm','As_ppm','mineralisé']].copy()
        logue_display.columns=['De(m)','A(m)','Lithologie','Altération','Minéralisation','Au(ppb)','Cu(ppm)','As(ppm)','Minéralisé']
        st.dataframe(logue_display.style.map(lambda v:'background-color:#FFD700' if v==True else '',subset=['Minéralisé']).format({'Au(ppb)':'{:.2f}','Cu(ppm)':'{:.1f}','As(ppm)':'{:.1f}'}),use_container_width=True)
        csv_l=logue_display.to_csv(index=False)
        st.download_button("📥 Télécharger logue",data=csv_l,file_name=f"logue_{trou_ls}.csv",mime='text/csv')

# ══ TAB 15 — MONITORING ════════════════════════════════════════════════════
with tabs[14]:
    st.subheader("📈 Monitoring — Suivi en temps réel")
    c1,c2,c3=st.columns(3)
    c1.metric("Mètres/jour",f"{np.random.randint(30,80)} m","↑ +12")
    c2.metric("Trous actifs",int((df_forages['statut']=='En cours').sum()))
    c3.metric("Incidents",np.random.randint(0,2))
    equipes=df_forages.groupby('equipe').agg(trous=('trou','count'),metres=('profondeur','sum')).reset_index()
    fig_eq,ax_eq=plt.subplots(figsize=(8,4))
    ax_eq.bar(equipes['equipe'],equipes['metres'],color=['#2196F3','#4CAF50','#FF9800'],edgecolor='black',linewidth=0.5)
    for i,v in enumerate(equipes['metres']): ax_eq.text(i,v+5,f"{v:.0f}m",ha='center',fontsize=9,fontweight='bold')
    ax_eq.set_ylabel("Mètres"); ax_eq.set_title("Mètres par équipe",fontsize=11,fontweight='bold')
    plt.tight_layout(); st.pyplot(fig_eq)
    dates_30=pd.date_range(end=datetime.date.today(),periods=30)
    metres_30=np.random.randint(20,80,30)
    fig_j,ax_j=plt.subplots(figsize=(12,4))
    ax_j.fill_between(dates_30,metres_30,alpha=0.3,color='#2196F3')
    ax_j.plot(dates_30,metres_30,'b-o',markersize=4,linewidth=1.5)
    ax_j.axhline(y=metres_30.mean(),color='red',linestyle='--',linewidth=1.5,label=f'Moy: {metres_30.mean():.0f}m/j')
    ax_j.set_ylabel("Mètres/jour"); ax_j.set_title("Production journalière",fontsize=11,fontweight='bold')
    ax_j.legend(fontsize=9); ax_j.grid(True,linestyle=':',alpha=0.4)
    plt.xticks(rotation=45); plt.tight_layout(); st.pyplot(fig_j)

# ══ TAB 16 — WEEKLY REPORT ══════════════════════════════════════════════════
with tabs[15]:
    st.subheader("📅 Rapport Hebdomadaire")
    semaine=st.date_input("Semaine du",datetime.date.today()-datetime.timedelta(days=7))
    st.markdown(f"**Période :** {semaine} → {semaine+datetime.timedelta(days=6)} | **Projet :** {NOM_PROSPECT}")
    c1,c2,c3,c4,c5=st.columns(5)
    total_m=int(weekly_data['metres_fores'].sum()); objectif=350
    c1.metric("Mètres forés",f"{total_m} m",f"{total_m-objectif:+d}")
    c2.metric("Trous complétés",int(weekly_data['trous_completes'].sum()))
    c3.metric("Incidents",int(weekly_data['incidents'].sum()))
    c4.metric("Au max semaine",f"{float(weekly_data['Au_ppb_moyen'].max()):.1f} ppb")
    c5.metric("Objectif","✅ Oui" if total_m>=objectif else "❌ Non")
    fig_w,axes_w=plt.subplots(1,3,figsize=(14,4))
    axes_w[0].bar([d.strftime('%a') for d in weekly_data['date']],weekly_data['metres_fores'],color='#2196F3',edgecolor='black',linewidth=0.5)
    axes_w[0].axhline(y=objectif/7,color='red',linestyle='--',linewidth=1.5,label=f'Obj:{objectif//7}m/j')
    axes_w[0].set_ylabel("Mètres"); axes_w[0].set_title("Mètres/jour",fontsize=10,fontweight='bold'); axes_w[0].legend(fontsize=8)
    axes_w[1].bar([d.strftime('%a') for d in weekly_data['date']],weekly_data['Au_ppb_moyen'],color='#FFD700',edgecolor='orange',linewidth=0.5)
    axes_w[1].set_ylabel("Au (ppb)"); axes_w[1].set_title("Au moyen/jour",fontsize=10,fontweight='bold')
    axes_w[2].bar([d.strftime('%a') for d in weekly_data['date']],weekly_data['incidents'],
                  color=['#4CAF50' if v==0 else '#FF5722' for v in weekly_data['incidents']],edgecolor='black',linewidth=0.5)
    axes_w[2].set_ylabel("Incidents"); axes_w[2].set_title("Incidents/jour",fontsize=10,fontweight='bold')
    plt.suptitle(f"Weekly Report — {semaine} | {NOM_PROSPECT}",fontsize=12,fontweight='bold')
    plt.tight_layout(); st.pyplot(fig_w)
    wd=weekly_data.copy(); wd['date']=wd['date'].dt.strftime('%Y-%m-%d (%A)')
    wd.columns=['Date','Mètres forés','Trous complétés','Incidents','Au moy.(ppb)','Équipe']
    st.dataframe(wd,use_container_width=True)
    st.download_button("📥 Télécharger rapport CSV",data=wd.to_csv(index=False),file_name=f"weekly_{semaine}.csv",mime='text/csv')

# ══ TAB 17 — RAPPORT GÉOLOGIQUE ══════════════════════════════════════════════
with tabs[16]:
    st.subheader("📄 Rapport Géologique Argumenté")
    st.markdown(f"**Projet :** {NOM_PROSPECT} | **Permis :** {NOM_PERMIS} | **Date :** {datetime.date.today()}")
    au_max=df_forages['Au_max_ppb'].max(); au_moy=df_forages['Au_max_ppb'].mean()
    nb_miner=int(df_intervals['mineralisé'].sum()); pct_miner_g=nb_miner/len(df_intervals)*100
    litho_dom=df_intervals['lithologie'].value_counts().index[0]
    alter_dom=df_intervals['alteration'].value_counts().index[0]
    struct_port=structures_df[structures_df['porteur_miner']==True]['type'].value_counts()
    struct_port_name=struct_port.index[0] if len(struct_port)>0 else 'Veine de quartz'
    dir_dom=structures_df['direction'].mean()
    pend_dom=structures_df['pendage'].mean()

    st.markdown(f"""
## 1. Contexte géologique — {NOM_PROSPECT}

Le projet est localisé au **Sénégal (Afrique de l'Ouest)**, dans une ceinture de roches vertes birimienne.
Ce contexte est favorable aux gisements aurifères orogéniques de type lode-gold, similaires aux gisements
de Sabodala, Massawa et Mako dans la même ceinture.

## 2. Lithologies

| Lithologie | Profondeur | Interprétation |
|-----------|-----------|----------------|
| Latérite | 0–8 m | Couverture résiduelle supergène |
| Saprolite | 8–25 m | Altération avancée, zone d'oxydation |
| Saprock | 25–50 m | Zone de transition critique |
| Bédrock/Schiste | 50–120 m | Encaissant principal de la minéralisation |
| **Quartzite aurifère** | Variable | **Unité minéralisée principale** |
| Granite frais | >100 m | Intrusion tardi-orogénique |

**Lithologie dominante :** {litho_dom}

## 3. Minéralisation

- **Au max :** {au_max:.1f} ppb | **Au moyen :** {au_moy:.1f} ppb
- **{pct_miner_g:.1f}% des intervalles** dépassent 100 ppb Au
- Type de minéralisation dominante : **Aurifère filonienne** en zones de cisaillement
- Pathfinders : As > 10 ppm corrélé avec Au > 100 ppb

## 4. Altération

Séquence : **Carbonatation → Séricitisation → Silicification → Pyritisation**
Altération dominante : **{alter_dom}** — indicateur d'un système hydrothermal mésothermal

## 5. Structures

- Direction dominante : **{dir_dom:.0f}°** | Pendage moyen : **{pend_dom:.0f}°**
- Structure porteuse principale : **{struct_port_name}**
- Contrôle structural : Zones de cisaillement NE-SO à pendage SE (40–70°)

## 6. Interprétations

1. La minéralisation est **ouverte en profondeur** — aucun signe d'épuisement à {df_forages['profondeur'].max():.0f}m
2. **Corrélation Au-As** forte → As utilisable comme pathfinder géochimique
3. Les zones à silicification intense = meilleures teneurs en or
4. Le modèle structural NE-SO contrôle la distribution spatiale des minéralisations

## 7. Recommandations

### Priorité haute
- Approfondir les trous à Au > 200 ppb (forages Diamond, 150–200m)
- Infill à 100m dans les zones anomaliques confirmées
- Levé IP pour cartographier les sulfures en profondeur

### Priorité moyenne
- Extension NE et SO du programme
- Analyses multi-éléments systématiques (Au, Ag, Cu, As, Sb)
- Modélisation géostatistique 3D (variogramme, krigeage)

### Long terme
- Estimation des ressources selon le code JORC/NI 43-101
- Études métallurgiques (essais de lixiviation)
- Évaluation environnementale et sociale (ESIA)

## 8. Tableau des structures du prospect

    """)
    st.dataframe(structures_df.rename(columns={
        'id':'N°','type':'Type','direction':'Direction(°)','pendage':'Pendage(°)',
        'sens_pendage':'Sens pendage','longueur_m':'Longueur(m)','porteur_miner':'Porteur minéralisation'
    })[['N°','Type','Direction(°)','Pendage(°)','Sens pendage','Longueur(m)','Porteur minéralisation']],use_container_width=True)

    rapport_txt=f"""RAPPORT GÉOLOGIQUE — {NOM_PROSPECT}
Permis: {NOM_PERMIS}
Date: {datetime.date.today()}
===========================
Au max: {au_max:.1f} ppb | Au moy: {au_moy:.1f} ppb
% minéralisé: {pct_miner_g:.1f}%
Lithologie dominante: {litho_dom}
Altération dominante: {alter_dom}
Structure porteuse: {struct_port_name}
Direction dominante: {dir_dom:.0f}°
Pendage moyen: {pend_dom:.0f}°
"""
    st.download_button("📥 Télécharger le rapport",data=rapport_txt,file_name=f"rapport_{NOM_PROSPECT}_{datetime.date.today()}.txt",mime='text/plain')

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# DONNÉES SUPPLÉMENTAIRES — Extension, Échantillonnage, Mapping
# ══════════════════════════════════════════════════════════════════════════════

# Programme extension
extension_zones = pd.DataFrame([{
    'zone': f'EXT{i+1:02d}',
    'priorite': np.random.choice(['Haute','Moyenne','Faible'], p=[0.3,0.4,0.3]),
    'easting': round(BASE_E + np.random.uniform(-600,600), 1),
    'northing': round(BASE_N + np.random.uniform(-600,600), 1),
    'profondeur_cible': np.random.choice([80,100,120,150,200]),
    'azimut': round(np.random.uniform(0,360), 1),
    'inclinaison': round(np.random.uniform(-85,-60), 1),
    'type_forage': np.random.choice(['RC','Diamond'], p=[0.5,0.5]),
    'Au_prevu_ppb': round(np.random.lognormal(4,1.2), 1),
    'cout_usd': round(np.random.uniform(10000,50000), 0),
    'justification': np.random.choice([
        'Extension anomalie Au','Prolongement zone minéralisée',
        'Test structure NE','Confirmation intersection','Approfondissement'
    ]),
    'statut': np.random.choice(['Planifié','Approuvé','En attente'], p=[0.5,0.3,0.2]),
} for i in range(20)])

# Échantillons
np.random.seed(55)
echantillons = pd.DataFrame([{
    'id_ech': f'ECH{i+1:04d}',
    'trou': f'SG{np.random.randint(1,16):03d}',
    'type_ech': np.random.choice(['Carotte','Déblais RC','Rigole','Sol','Roche']),
    'de': round(np.random.uniform(0,100), 1),
    'poids_kg': round(np.random.uniform(0.5,5), 2),
    'laboratoire': np.random.choice(['SGS Dakar','ALS Bamako','Intertek Conakry']),
    'methode': np.random.choice(['ICP-MS','Fire Assay','AAS','FAAS']),
    'Au_ppb': round(np.random.lognormal(3.5,1.5), 2),
    'Cu_ppm': round(np.random.uniform(1,150), 1),
    'As_ppm': round(np.random.uniform(1,80), 1),
    'Ag_ppm': round(np.random.uniform(0.1,15), 2),
    'Fe_pct': round(np.random.uniform(1,35), 2),
    'date_envoi': (datetime.date.today()-datetime.timedelta(days=int(np.random.randint(1,60)))).strftime('%Y-%m-%d'),
    'statut': np.random.choice(['Résultat reçu','En attente','En transit'], p=[0.6,0.3,0.1]),
    'qaqc': np.random.choice(['Standard','Blanc','Duplicata','Normal'], p=[0.1,0.1,0.1,0.7]),
} for i in range(200)])

# ══ TAB 18 — PROGRAMME D'EXTENSION ═══════════════════════════════════════════
st.markdown("---")
st.header("🔭 Module Extension — Nouveaux onglets")

ext_tabs = st.tabs([
    "🔭 Programme d'extension",
    "🗺️ Mapping & Analyse spatiale",
    "🧪 Échantillonnage & Sampling",
])

# ── EXTENSION ─────────────────────────────────────────────────────────────────
with ext_tabs[0]:
    st.subheader(f"🔭 Programme de Forage d'Extension — {NOM_PROSPECT}")
    st.markdown(f"**Permis :** {NOM_PERMIS} | **Date :** {datetime.date.today()}")

    # KPIs
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Zones d'extension", len(extension_zones))
    c2.metric("Priorité Haute", int((extension_zones['priorite']=='Haute').sum()))
    c3.metric("Approuvés", int((extension_zones['statut']=='Approuvé').sum()))
    c4.metric("Mètres prévus", f"{extension_zones['profondeur_cible'].sum():.0f} m")
    c5.metric("Budget total", f"${extension_zones['cout_usd'].sum():,.0f}")

    col1,col2 = st.columns([2,1])
    with col1:
        # Carte des zones d'extension
        fig_ext, ax_ext = plt.subplots(figsize=(11,9))
        fig_ext.patch.set_facecolor('#F5F5F0'); ax_ext.set_facecolor('#E8F4F8')

        # Forages existants
        for _,f in df_forages.iterrows():
            ax_ext.scatter(f['easting'], f['northing'], c='black', s=60,
                          marker='o', zorder=3, label='Existant' if _ == 0 else '')
            ax_ext.text(f['easting'], f['northing']+8, f['trou'], fontsize=5.5,
                       ha='center', color='#1A237E')

        # Zones d'extension par priorité
        priority_colors = {'Haute':'#FF0000','Moyenne':'#FF9800','Faible':'#2196F3'}
        priority_markers = {'Haute':'*','Moyenne':'^','Faible':'s'}
        for _,z in extension_zones.iterrows():
            color = priority_colors[z['priorite']]
            marker = priority_markers[z['priorite']]
            size = 300 if z['priorite']=='Haute' else 150 if z['priorite']=='Moyenne' else 80
            ax_ext.scatter(z['easting'], z['northing'], c=color, s=size,
                          marker=marker, edgecolors='black', linewidths=1,
                          zorder=4, alpha=0.85)
            ax_ext.text(z['easting'], z['northing']+10, z['zone'],
                       fontsize=6, ha='center', color=color, fontweight='bold')
            # Cercle de zone d'influence
            if z['priorite'] == 'Haute':
                circle = plt.Circle((z['easting'], z['northing']), 80,
                                   fill=False, color='red', linewidth=1.5,
                                   linestyle='--', alpha=0.5)
                ax_ext.add_patch(circle)

        # Nord & Échelle
        xmax = max(df_forages['easting'].max(), extension_zones['easting'].max())
        ymax = max(df_forages['northing'].max(), extension_zones['northing'].max())
        xmin = min(df_forages['easting'].min(), extension_zones['easting'].min())
        ymin = min(df_forages['northing'].min(), extension_zones['northing'].min())
        ax_ext.annotate('', xy=(xmax+80,ymax+20), xytext=(xmax+80,ymax-30),
                       arrowprops=dict(arrowstyle='->', color='black', lw=2.5))
        ax_ext.text(xmax+80, ymax+35, 'N', ha='center', fontsize=14, fontweight='bold')
        ax_ext.plot([xmin, xmin+300], [ymin-50, ymin-50], 'k-', linewidth=3)
        ax_ext.text(xmin+150, ymin-70, '300 m', ha='center', fontsize=9, fontweight='bold')

        # Légende
        legend_ext = [
            plt.Line2D([0],[0], marker='*', color='w', markerfacecolor='red', markersize=14, label='Haute priorité'),
            plt.Line2D([0],[0], marker='^', color='w', markerfacecolor='#FF9800', markersize=10, label='Moyenne priorité'),
            plt.Line2D([0],[0], marker='s', color='w', markerfacecolor='#2196F3', markersize=9, label='Faible priorité'),
            plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='black', markersize=8, label='Forage existant'),
        ]
        ax_ext.legend(handles=legend_ext, loc='lower right', fontsize=8,
                     title='Programme extension', framealpha=0.95)
        ax_ext.set_xlabel("Easting UTM (m)"); ax_ext.set_ylabel("Northing UTM (m)")
        ax_ext.set_title(f"Carte programme d'extension — {NOM_PROSPECT}\n{NOM_PERMIS}",
                        fontsize=12, fontweight='bold')
        ax_ext.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout(); st.pyplot(fig_ext)

    with col2:
        st.markdown("### 📊 Résumé par priorité")
        for prio, color in priority_colors.items():
            sub = extension_zones[extension_zones['priorite']==prio]
            st.markdown(f"**{prio}** : {len(sub)} zones | {sub['profondeur_cible'].sum():.0f}m | ${sub['cout_usd'].sum():,.0f}")

        st.markdown("### 📈 Coût par type")
        fig_cout, ax_cout = plt.subplots(figsize=(5,3))
        cout_type = extension_zones.groupby('type_forage')['cout_usd'].sum()
        ax_cout.bar(cout_type.index, cout_type.values, color=['#FF5722','#9C27B0'],
                   edgecolor='black', linewidth=0.5)
        ax_cout.set_ylabel("Coût (USD)"); ax_cout.set_title("Budget par type", fontsize=9)
        for i,v in enumerate(cout_type.values):
            ax_cout.text(i, v+500, f"${v:,.0f}", ha='center', fontsize=8, fontweight='bold')
        plt.tight_layout(); st.pyplot(fig_cout)

    # Tableau détaillé
    st.markdown("### 📋 Tableau du programme d'extension")
    filtre_prio = st.multiselect("Filtrer par priorité",
                                  ['Haute','Moyenne','Faible'],
                                  default=['Haute','Moyenne','Faible'])
    df_ext_filt = extension_zones[extension_zones['priorite'].isin(filtre_prio)]
    st.dataframe(df_ext_filt.rename(columns={
        'zone':'Zone','priorite':'Priorité','easting':'Easting(m)','northing':'Northing(m)',
        'profondeur_cible':'Prof. cible(m)','azimut':'Azimut(°)','inclinaison':'Inclin.(°)',
        'type_forage':'Type','Au_prevu_ppb':'Au prévu(ppb)','cout_usd':'Coût(USD)',
        'justification':'Justification','statut':'Statut'
    }), use_container_width=True)

    # Analyse risque/bénéfice
    st.markdown("### ⚖️ Analyse Risque / Bénéfice")
    fig_rb, ax_rb = plt.subplots(figsize=(8,5))
    colors_prio = [priority_colors[p] for p in extension_zones['priorite']]
    sc = ax_rb.scatter(extension_zones['cout_usd']/1000,
                      extension_zones['Au_prevu_ppb'],
                      c=[{'Haute':3,'Moyenne':2,'Faible':1}[p] for p in extension_zones['priorite']],
                      cmap='RdYlGn', s=150, edgecolors='black', linewidths=0.8,
                      alpha=0.85, zorder=3)
    for _,z in extension_zones.iterrows():
        ax_rb.text(z['cout_usd']/1000+0.3, z['Au_prevu_ppb'],
                  z['zone'], fontsize=6.5, color='#333')
    ax_rb.set_xlabel("Coût estimé (k$)"); ax_rb.set_ylabel("Au prévu (ppb)")
    ax_rb.set_title("Analyse Risque/Bénéfice — Zones d'extension", fontsize=11, fontweight='bold')
    plt.colorbar(sc, ax=ax_rb, label='Priorité (1=Faible, 3=Haute)')
    ax_rb.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout(); st.pyplot(fig_rb)

    st.warning(f"""
    📌 **Recommandations d'extension :**
    - **{len(extension_zones[extension_zones['priorite']=='Haute'])} zones haute priorité** à forer en premier
    - Budget total estimé : **${extension_zones['cout_usd'].sum():,.0f} USD**
    - Forer en priorité les zones EXT avec Au prévu > 200 ppb
    - Commencer par les RC pour confirmer, puis Diamond pour les meilleures zones
    """)
    csv_ext = extension_zones.to_csv(index=False)
    st.download_button("📥 Télécharger programme extension",
                      data=csv_ext, file_name="programme_extension.csv", mime='text/csv')

# ── MAPPING & ANALYSE SPATIALE ────────────────────────────────────────────────
with ext_tabs[1]:
    st.subheader(f"🗺️ Mapping & Analyse Spatiale des Gisements — {NOM_PROSPECT}")

    analyse_type = st.radio("Type d'analyse",
                           ['Densité des minéralisations','Variogramme expérimental',
                            'Analyse de clusters','Corrélation spatiale'],
                           horizontal=True)

    if analyse_type == 'Densité des minéralisations':
        st.markdown("### 🔥 Carte de densité des teneurs en or")
        au_par_trou = df_intervals.groupby('trou')['Au_ppb'].max().reset_index()
        au_par_trou.columns = ['trou','Au_max']
        df_spatial = df_forages.merge(au_par_trou, on='trou')

        fig_dens, axes_dens = plt.subplots(1,2, figsize=(14,7))

        # Carte de chaleur
        if len(df_spatial) > 3:
            xi = np.linspace(df_spatial['easting'].min()-100, df_spatial['easting'].max()+100, 200)
            yi = np.linspace(df_spatial['northing'].min()-100, df_spatial['northing'].max()+100, 200)
            Xi, Yi = np.meshgrid(xi, yi)
            Zi = griddata((df_spatial['easting'], df_spatial['northing']),
                         df_spatial['Au_max'], (Xi, Yi), method='cubic')
            im = axes_dens[0].contourf(Xi, Yi, Zi, levels=25, cmap='YlOrRd', alpha=0.85)
            plt.colorbar(im, ax=axes_dens[0], label='Au max (ppb)')
            axes_dens[0].contour(Xi, Yi, Zi, levels=8, colors='black', alpha=0.3, linewidths=0.5)

        for _,f in df_spatial.iterrows():
            marker = {'RC':'^','Aircore':'o','Diamond':'s'}.get(f['type'],'o')
            axes_dens[0].scatter(f['easting'], f['northing'], c='white', s=80,
                               marker=marker, edgecolors='black', linewidths=1, zorder=4)
            axes_dens[0].text(f['easting'], f['northing']+8, f['trou'],
                             fontsize=6, ha='center', color='black', fontweight='bold')

        xmax=df_spatial['easting'].max(); ymax=df_spatial['northing'].max()
        xmin=df_spatial['easting'].min(); ymin=df_spatial['northing'].min()
        axes_dens[0].annotate('',xy=(xmax+60,ymax+20),xytext=(xmax+60,ymax-20),
                             arrowprops=dict(arrowstyle='->',color='black',lw=2))
        axes_dens[0].text(xmax+60,ymax+30,'N',ha='center',fontsize=12,fontweight='bold')
        axes_dens[0].plot([xmin,xmin+200],[ymin-40,ymin-40],'k-',linewidth=3)
        axes_dens[0].text(xmin+100,ymin-55,'200 m',ha='center',fontsize=8,fontweight='bold')
        axes_dens[0].set_xlabel("Easting (m)"); axes_dens[0].set_ylabel("Northing (m)")
        axes_dens[0].set_title(f"Carte de densité Au — {NOM_PROSPECT}", fontsize=11, fontweight='bold')
        axes_dens[0].grid(True, linestyle=':', alpha=0.3)

        # Distribution des teneurs
        axes_dens[1].hist(np.log10(df_spatial['Au_max']+1), bins=15,
                         color='#FFD700', edgecolor='black', linewidth=0.5)
        axes_dens[1].set_xlabel("log10(Au max + 1) [ppb]")
        axes_dens[1].set_ylabel("Fréquence")
        axes_dens[1].set_title("Distribution log-normale des teneurs", fontsize=11, fontweight='bold')
        axes_dens[1].axvline(np.log10(100+1), color='red', linestyle='--', linewidth=2,
                            label='Seuil 100 ppb')
        axes_dens[1].legend(fontsize=9)
        axes_dens[1].grid(True, linestyle=':', alpha=0.4)
        plt.suptitle(f"Analyse spatiale — {NOM_PROSPECT}", fontsize=12, fontweight='bold')
        plt.tight_layout(); st.pyplot(fig_dens)

    elif analyse_type == 'Variogramme expérimental':
        st.markdown("### 📐 Variogramme expérimental — Continuité spatiale")
        au_par_trou = df_intervals.groupby('trou')['Au_ppb'].max().reset_index()
        au_par_trou.columns = ['trou','Au']
        df_var = df_forages.merge(au_par_trou, on='trou')

        # Calcul variogramme expérimental
        distances = []
        variogrammes = []
        coords = df_var[['easting','northing']].values
        values = np.log1p(df_var['Au'].values)
        n = len(coords)
        for i in range(n):
            for j in range(i+1, n):
                d = np.sqrt((coords[i,0]-coords[j,0])**2 + (coords[i,1]-coords[j,1])**2)
                g = 0.5*(values[i]-values[j])**2
                distances.append(d); variogrammes.append(g)
        distances = np.array(distances); variogrammes = np.array(variogrammes)
        bins = np.linspace(0, distances.max(), 12)
        gamma_bins = []
        dist_bins = []
        for k in range(len(bins)-1):
            mask = (distances>=bins[k]) & (distances<bins[k+1])
            if mask.sum()>0:
                gamma_bins.append(variogrammes[mask].mean())
                dist_bins.append((bins[k]+bins[k+1])/2)
        gamma_bins = np.array(gamma_bins); dist_bins = np.array(dist_bins)

        fig_var, axes_var = plt.subplots(1,2, figsize=(12,5))
        axes_var[0].scatter(dist_bins, gamma_bins, c='#2196F3', s=100,
                           edgecolors='black', linewidths=1, zorder=3)
        axes_var[0].plot(dist_bins, gamma_bins, 'b--', linewidth=1.5, alpha=0.6)
        # Modèle sphérique ajusté
        nugget = gamma_bins[0] if len(gamma_bins)>0 else 0.1
        sill = gamma_bins.max() if len(gamma_bins)>0 else 1
        range_var = dist_bins[len(dist_bins)//2] if len(dist_bins)>0 else 200
        h = np.linspace(0, dist_bins.max() if len(dist_bins)>0 else 500, 100)
        model = nugget + (sill-nugget)*(1.5*(h/range_var)-0.5*(h/range_var)**3)
        model = np.where(h>range_var, sill, model)
        axes_var[0].plot(h, model, 'r-', linewidth=2.5, label=f'Modèle sphérique\nNugget:{nugget:.2f} Sill:{sill:.2f} Range:{range_var:.0f}m')
        axes_var[0].set_xlabel("Distance (m)"); axes_var[0].set_ylabel("γ(h) — Semi-variance")
        axes_var[0].set_title("Variogramme expérimental — Au", fontsize=11, fontweight='bold')
        axes_var[0].legend(fontsize=8); axes_var[0].grid(True, linestyle=':', alpha=0.4)

        # Rose des variogrammes directionnels
        ax_vr = fig_var.add_subplot(1,2,2, projection='polar')
        directions_v = np.linspace(0, np.pi, 8)
        var_dir = np.random.uniform(0.5, 2.0, 8)
        var_dir_sym = np.concatenate([var_dir, var_dir])
        theta_v = np.concatenate([directions_v, directions_v+np.pi])
        ax_vr.bar(theta_v, var_dir_sym, width=np.pi/8, color='steelblue',
                 edgecolor='black', alpha=0.7)
        ax_vr.set_theta_zero_location('N'); ax_vr.set_theta_direction(-1)
        ax_vr.set_title("Variogramme directionnel\n(anisotropie)", fontsize=10, fontweight='bold', pad=20)
        ax_vr.set_xticklabels(['N','NE','E','SE','S','SO','O','NO'])
        plt.suptitle(f"Analyse variographique — {NOM_PROSPECT}", fontsize=12, fontweight='bold')
        plt.tight_layout(); st.pyplot(fig_var)

        st.info(f"""
        📐 **Paramètres du variogramme :**
        - **Nugget (effet pépite) :** {nugget:.2f} — variabilité à petite échelle
        - **Sill (palier) :** {sill:.2f} — variance totale du gisement
        - **Range (portée) :** {range_var:.0f} m — distance de corrélation spatiale
        - **Interprétation :** Continuité spatiale sur ~{range_var:.0f}m — espacer les forages à max {range_var*0.6:.0f}m
        """)

    elif analyse_type == 'Analyse de clusters':
        st.markdown("### 🔵 Analyse de clusters — Regroupement spatial")
        au_par_trou = df_intervals.groupby('trou')['Au_ppb'].max().reset_index()
        au_par_trou.columns = ['trou','Au']
        df_clust = df_forages.merge(au_par_trou, on='trou')

        # K-means simplifié
        from sklearn.cluster import KMeans
        coords_c = df_clust[['easting','northing']].values
        n_clusters = st.slider("Nombre de clusters", 2, 6, 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df_clust['cluster'] = kmeans.fit_predict(coords_c)
        centers = kmeans.cluster_centers_

        fig_cl, axes_cl = plt.subplots(1,2, figsize=(14,6))
        cluster_colors = ['#FF5722','#2196F3','#4CAF50','#FF9800','#9C27B0','#00BCD4']
        for cl in range(n_clusters):
            sub = df_clust[df_clust['cluster']==cl]
            axes_cl[0].scatter(sub['easting'], sub['northing'],
                              c=cluster_colors[cl], s=150,
                              edgecolors='black', linewidths=1, label=f'Cluster {cl+1}',
                              zorder=3, alpha=0.85)
            for _,f in sub.iterrows():
                axes_cl[0].text(f['easting'], f['northing']+8, f['trou'],
                               fontsize=6, ha='center')
        axes_cl[0].scatter(centers[:,0], centers[:,1], c='black', s=300,
                          marker='X', zorder=5, label='Centroïdes')
        axes_cl[0].legend(fontsize=8); axes_cl[0].grid(True, linestyle=':', alpha=0.4)
        axes_cl[0].set_xlabel("Easting (m)"); axes_cl[0].set_ylabel("Northing (m)")
        axes_cl[0].set_title(f"Clusters spatiaux ({n_clusters} groupes)", fontsize=11, fontweight='bold')

        # Au par cluster
        cluster_au = df_clust.groupby('cluster')['Au'].agg(['mean','max','count']).reset_index()
        x_cl = [f'Cluster {c+1}' for c in cluster_au['cluster']]
        axes_cl[1].bar(x_cl, cluster_au['mean'], color=[cluster_colors[c] for c in cluster_au['cluster']],
                      edgecolor='black', linewidth=0.5, label='Au moyen')
        axes_cl[1].scatter(x_cl, cluster_au['max'], c='black', s=100, zorder=3, label='Au max', marker='D')
        axes_cl[1].set_ylabel("Au (ppb)"); axes_cl[1].set_title("Au par cluster", fontsize=11, fontweight='bold')
        axes_cl[1].legend(fontsize=9); axes_cl[1].grid(True, linestyle=':', alpha=0.4)
        plt.suptitle(f"Analyse de clusters — {NOM_PROSPECT}", fontsize=12, fontweight='bold')
        plt.tight_layout(); st.pyplot(fig_cl)

        best_cl = cluster_au.loc[cluster_au['mean'].idxmax(), 'cluster']
        st.success(f"**Cluster {best_cl+1}** est le plus prometteur avec Au moyen = {cluster_au.loc[best_cl,'mean']:.1f} ppb")

    else:  # Corrélation spatiale
        st.markdown("### 🔗 Corrélation spatiale multi-éléments")
        au_moy = df_intervals.groupby('trou')[['Au_ppb','Cu_ppm','As_ppm','Ag_ppm']].mean().reset_index()
        df_corr = df_forages.merge(au_moy, on='trou')

        fig_corr, axes_corr = plt.subplots(2,2, figsize=(12,10))
        paires = [('Au_ppb','As_ppm'),('Au_ppb','Cu_ppm'),('Au_ppb','Ag_ppm'),('As_ppm','Cu_ppm')]
        colors_scatter = ['#FF5722','#2196F3','#4CAF50','#FF9800']
        for ax_c, (x_col,y_col), color in zip(axes_corr.flatten(), paires, colors_scatter):
            ax_c.scatter(df_corr[x_col], df_corr[y_col], c=color, s=80,
                        edgecolors='black', linewidths=0.5, alpha=0.85)
            # Régression linéaire
            if len(df_corr)>2:
                z = np.polyfit(df_corr[x_col], df_corr[y_col], 1)
                p = np.poly1d(z)
                x_line = np.linspace(df_corr[x_col].min(), df_corr[x_col].max(), 50)
                ax_c.plot(x_line, p(x_line), 'r--', linewidth=2)
                r = np.corrcoef(df_corr[x_col], df_corr[y_col])[0,1]
                ax_c.text(0.05, 0.95, f'r = {r:.2f}', transform=ax_c.transAxes,
                         fontsize=10, fontweight='bold', color='red',
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            for _,row in df_corr.iterrows():
                ax_c.text(row[x_col], row[y_col]+0.5, row['trou'],
                         fontsize=5.5, ha='center', color='#333')
            ax_c.set_xlabel(x_col); ax_c.set_ylabel(y_col)
            ax_c.set_title(f"{x_col} vs {y_col}", fontsize=10, fontweight='bold')
            ax_c.grid(True, linestyle=':', alpha=0.4)
        plt.suptitle(f"Corrélations spatiales multi-éléments — {NOM_PROSPECT}", fontsize=12, fontweight='bold')
        plt.tight_layout(); st.pyplot(fig_corr)

# ── ÉCHANTILLONNAGE & SAMPLING ─────────────────────────────────────────────────
with ext_tabs[2]:
    st.subheader(f"🧪 Échantillonnage & Sampling — {NOM_PROSPECT}")

    sampling_vue = st.radio("Module",
                           ['Vue d\'ensemble','Registre des échantillons',
                            'Contrôle qualité QAQC','Analyse des résultats',
                            'Statistiques'],
                           horizontal=True)

    if sampling_vue == "Vue d'ensemble":
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Total échantillons", len(echantillons))
        c2.metric("Résultats reçus", int((echantillons['statut']=='Résultat reçu').sum()))
        c3.metric("En attente", int((echantillons['statut']=='En attente').sum()))
        c4.metric("Au max", f"{echantillons['Au_ppb'].max():.1f} ppb")
        c5.metric("Au moyen", f"{echantillons['Au_ppb'].mean():.1f} ppb")

        fig_ov, axes_ov = plt.subplots(1,3, figsize=(14,4))
        # Statuts
        stat_counts = echantillons['statut'].value_counts()
        axes_ov[0].pie(stat_counts.values, labels=stat_counts.index,
                      autopct='%1.0f%%', colors=['#4CAF50','#FF9800','#2196F3'])
        axes_ov[0].set_title("Statut des analyses", fontsize=10, fontweight='bold')
        # Types
        type_counts = echantillons['type_ech'].value_counts()
        axes_ov[1].bar(type_counts.index, type_counts.values,
                      color='steelblue', edgecolor='black', linewidth=0.5)
        axes_ov[1].set_ylabel("Nombre"); axes_ov[1].set_title("Types d'échantillons", fontsize=10, fontweight='bold')
        plt.setp(axes_ov[1].xaxis.get_majorticklabels(), rotation=30, fontsize=8)
        # Laboratoires
        labo_counts = echantillons['laboratoire'].value_counts()
        axes_ov[2].bar(labo_counts.index, labo_counts.values,
                      color=['#FF5722','#2196F3','#4CAF50'], edgecolor='black', linewidth=0.5)
        axes_ov[2].set_ylabel("Nombre"); axes_ov[2].set_title("Par laboratoire", fontsize=10, fontweight='bold')
        plt.setp(axes_ov[2].xaxis.get_majorticklabels(), rotation=20, fontsize=8)
        plt.suptitle(f"Vue d'ensemble sampling — {NOM_PROSPECT}", fontsize=12, fontweight='bold')
        plt.tight_layout(); st.pyplot(fig_ov)

    elif sampling_vue == "Registre des échantillons":
        st.markdown("### 📋 Registre complet des échantillons")
        col1, col2 = st.columns(2)
        with col1:
            filtre_type = st.multiselect("Type", echantillons['type_ech'].unique(),
                                         default=list(echantillons['type_ech'].unique()))
        with col2:
            filtre_labo = st.multiselect("Laboratoire", echantillons['laboratoire'].unique(),
                                          default=list(echantillons['laboratoire'].unique()))
        df_reg = echantillons[(echantillons['type_ech'].isin(filtre_type)) &
                              (echantillons['laboratoire'].isin(filtre_labo))]
        st.dataframe(df_reg[['id_ech','trou','type_ech','de','poids_kg','laboratoire',
                              'methode','Au_ppb','Cu_ppm','As_ppm','statut','qaqc']].rename(columns={
            'id_ech':'N° Ech','trou':'Trou','type_ech':'Type','de':'De(m)',
            'poids_kg':'Poids(kg)','laboratoire':'Labo','methode':'Méthode',
            'Au_ppb':'Au(ppb)','Cu_ppm':'Cu(ppm)','As_ppm':'As(ppm)',
            'statut':'Statut','qaqc':'QAQC'
        }), use_container_width=True)
        csv_ech = df_reg.to_csv(index=False)
        st.download_button("📥 Télécharger registre", data=csv_ech,
                          file_name="registre_echantillons.csv", mime='text/csv')

    elif sampling_vue == "Contrôle qualité QAQC":
        st.markdown("### ✅ Contrôle Qualité QAQC")
        standards = echantillons[echantillons['qaqc']=='Standard']
        blancs = echantillons[echantillons['qaqc']=='Blanc']
        duplicatas = echantillons[echantillons['qaqc']=='Duplicata']

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Standards", len(standards))
        c2.metric("Blancs", len(blancs))
        c3.metric("Duplicatas", len(duplicatas))
        c4.metric("Taux QAQC", f"{(len(standards)+len(blancs)+len(duplicatas))/len(echantillons)*100:.1f}%")

        fig_qaqc, axes_qaqc = plt.subplots(2,2, figsize=(12,9))

        # Standards — vérification de la précision
        if len(standards) > 0:
            valeur_certifiee = 250
            axes_qaqc[0,0].scatter(range(len(standards)), standards['Au_ppb'],
                                  c='#2196F3', s=60, edgecolors='black', linewidths=0.5, label='Résultats')
            axes_qaqc[0,0].axhline(y=valeur_certifiee, color='green', linewidth=2,
                                  linestyle='-', label=f'Certifié: {valeur_certifiee} ppb')
            axes_qaqc[0,0].axhline(y=valeur_certifiee*1.1, color='orange', linewidth=1.5,
                                  linestyle='--', label='+10%')
            axes_qaqc[0,0].axhline(y=valeur_certifiee*0.9, color='orange', linewidth=1.5,
                                  linestyle='--', label='-10%')
            axes_qaqc[0,0].axhline(y=valeur_certifiee*1.2, color='red', linewidth=1,
                                  linestyle=':', label='+20%')
            axes_qaqc[0,0].axhline(y=valeur_certifiee*0.8, color='red', linewidth=1,
                                  linestyle=':', label='-20%')
            axes_qaqc[0,0].set_xlabel("N° standard"); axes_qaqc[0,0].set_ylabel("Au (ppb)")
            axes_qaqc[0,0].set_title("Standards de référence — Précision", fontsize=10, fontweight='bold')
            axes_qaqc[0,0].legend(fontsize=7)

        # Blancs — vérification contamination
        if len(blancs) > 0:
            axes_qaqc[0,1].scatter(range(len(blancs)), blancs['Au_ppb'],
                                  c='#FF5722', s=60, edgecolors='black', linewidths=0.5)
            axes_qaqc[0,1].axhline(y=10, color='red', linewidth=2, linestyle='--',
                                  label='Seuil contamination: 10 ppb')
            contamines = (blancs['Au_ppb'] > 10).sum()
            axes_qaqc[0,1].set_xlabel("N° blanc"); axes_qaqc[0,1].set_ylabel("Au (ppb)")
            axes_qaqc[0,1].set_title(f"Blancs — Contamination ({contamines} dépassements)",
                                    fontsize=10, fontweight='bold')
            axes_qaqc[0,1].legend(fontsize=8)

        # Duplicatas — reproductibilité
        if len(duplicatas) > 1:
            n_dup = len(duplicatas)//2
            orig = duplicatas.iloc[:n_dup]['Au_ppb'].values
            dup = duplicatas.iloc[n_dup:n_dup+min(n_dup,len(duplicatas)-n_dup)]['Au_ppb'].values
            min_len = min(len(orig),len(dup))
            axes_qaqc[1,0].scatter(orig[:min_len], dup[:min_len],
                                  c='#4CAF50', s=60, edgecolors='black', linewidths=0.5)
            max_val = max(orig[:min_len].max(), dup[:min_len].max())
            axes_qaqc[1,0].plot([0,max_val],[0,max_val],'r-',linewidth=2,label='1:1')
            axes_qaqc[1,0].plot([0,max_val],[0,max_val*1.1],'--',color='orange',linewidth=1,label='+10%')
            axes_qaqc[1,0].plot([0,max_val],[0,max_val*0.9],'--',color='orange',linewidth=1,label='-10%')
            axes_qaqc[1,0].set_xlabel("Original (ppb)"); axes_qaqc[1,0].set_ylabel("Duplicata (ppb)")
            axes_qaqc[1,0].set_title("Duplicatas — Reproductibilité", fontsize=10, fontweight='bold')
            axes_qaqc[1,0].legend(fontsize=7)

        # Résumé QAQC
        qaqc_summary = echantillons['qaqc'].value_counts()
        axes_qaqc[1,1].bar(qaqc_summary.index, qaqc_summary.values,
                          color=['#2196F3','#FF5722','#4CAF50','#FF9800'],
                          edgecolor='black', linewidth=0.5)
        axes_qaqc[1,1].set_ylabel("Nombre"); axes_qaqc[1,1].set_title("Résumé QAQC", fontsize=10, fontweight='bold')
        for i,v in enumerate(qaqc_summary.values):
            axes_qaqc[1,1].text(i, v+0.5, str(v), ha='center', fontsize=9, fontweight='bold')

        plt.suptitle(f"Contrôle qualité QAQC — {NOM_PROSPECT}", fontsize=12, fontweight='bold')
        plt.tight_layout(); st.pyplot(fig_qaqc)

        pct_valide = (len(standards)+len(blancs)+len(duplicatas))/len(echantillons)*100
        if pct_valide >= 10:
            st.success(f"✅ Taux QAQC : **{pct_valide:.1f}%** — Conforme aux bonnes pratiques (min. 10%)")
        else:
            st.error(f"⚠️ Taux QAQC : **{pct_valide:.1f}%** — En dessous du minimum recommandé (10%)")

    elif sampling_vue == "Analyse des résultats":
        st.markdown("### 📊 Analyse des résultats analytiques")
        fig_res, axes_res = plt.subplots(2,2, figsize=(12,9))

        # Distribution Au
        ech_res = echantillons[echantillons['statut']=='Résultat reçu']
        axes_res[0,0].hist(np.log10(ech_res['Au_ppb']+0.01), bins=25,
                          color='#FFD700', edgecolor='black', linewidth=0.5)
        axes_res[0,0].set_xlabel("log10(Au ppb)"); axes_res[0,0].set_ylabel("Fréquence")
        axes_res[0,0].set_title("Distribution Au (log-normale)", fontsize=10, fontweight='bold')
        axes_res[0,0].axvline(np.log10(100), color='red', linestyle='--', linewidth=2, label='100 ppb')
        axes_res[0,0].legend(fontsize=8)

        # Au par type d'échantillon
        au_type = ech_res.groupby('type_ech')['Au_ppb'].agg(['mean','max']).reset_index()
        x_pos = range(len(au_type))
        axes_res[0,1].bar([i-0.2 for i in x_pos], au_type['mean'], width=0.4,
                         color='#FFD700', edgecolor='black', linewidth=0.5, label='Moyen')
        axes_res[0,1].bar([i+0.2 for i in x_pos], au_type['max'], width=0.4,
                         color='#FF5722', edgecolor='black', linewidth=0.5, label='Max')
        axes_res[0,1].set_xticks(list(x_pos)); axes_res[0,1].set_xticklabels(au_type['type_ech'], fontsize=8, rotation=20)
        axes_res[0,1].set_ylabel("Au (ppb)"); axes_res[0,1].set_title("Au par type d'échantillon", fontsize=10, fontweight='bold')
        axes_res[0,1].legend(fontsize=8)

        # Évolution temporelle
        ech_res_dt = ech_res.copy()
        ech_res_dt['date_envoi'] = pd.to_datetime(ech_res_dt['date_envoi'])
        ech_daily = ech_res_dt.groupby('date_envoi')['Au_ppb'].mean().reset_index()
        if len(ech_daily) > 1:
            axes_res[1,0].plot(ech_daily['date_envoi'], ech_daily['Au_ppb'],
                              'b-o', markersize=5, linewidth=1.5)
            axes_res[1,0].fill_between(ech_daily['date_envoi'], ech_daily['Au_ppb'],
                                      alpha=0.3, color='blue')
            axes_res[1,0].set_xlabel("Date"); axes_res[1,0].set_ylabel("Au moyen (ppb)")
            axes_res[1,0].set_title("Évolution temporelle des teneurs", fontsize=10, fontweight='bold')
            axes_res[1,0].grid(True, linestyle=':', alpha=0.4)
            plt.setp(axes_res[1,0].xaxis.get_majorticklabels(), rotation=30, fontsize=7)

        # Au par laboratoire (comparaison)
        au_labo = ech_res.groupby('laboratoire')['Au_ppb'].agg(['mean','std']).reset_index()
        axes_res[1,1].bar(au_labo['laboratoire'], au_labo['mean'],
                         yerr=au_labo['std'], capsize=5,
                         color=['#FF5722','#2196F3','#4CAF50'],
                         edgecolor='black', linewidth=0.5)
        axes_res[1,1].set_ylabel("Au moyen ± std (ppb)")
        axes_res[1,1].set_title("Comparaison inter-laboratoires", fontsize=10, fontweight='bold')
        plt.setp(axes_res[1,1].xaxis.get_majorticklabels(), rotation=15, fontsize=8)

        plt.suptitle(f"Analyse des résultats analytiques — {NOM_PROSPECT}", fontsize=12, fontweight='bold')
        plt.tight_layout(); st.pyplot(fig_res)

    else:  # Statistiques
        st.markdown("### 📈 Statistiques descriptives des analyses")
        ech_stat = echantillons[echantillons['statut']=='Résultat reçu']
        elements = ['Au_ppb','Cu_ppm','As_ppm','Ag_ppm','Fe_pct']
        stats = ech_stat[elements].describe().round(3)
        stats.loc['CV%'] = (ech_stat[elements].std() / ech_stat[elements].mean() * 100).round(1)
        st.dataframe(stats.rename(columns={
            'Au_ppb':'Au (ppb)','Cu_ppm':'Cu (ppm)','As_ppm':'As (ppm)',
            'Ag_ppm':'Ag (ppm)','Fe_pct':'Fe (%)'
        }), use_container_width=True)

        # Boxplots
        fig_box, ax_box = plt.subplots(figsize=(10,5))
        data_box = [np.log10(ech_stat['Au_ppb']+0.01), np.log10(ech_stat['Cu_ppm']+0.01),
                   np.log10(ech_stat['As_ppm']+0.01), np.log10(ech_stat['Ag_ppm']+0.01)]
        bp = ax_box.boxplot(data_box, patch_artist=True,
                          labels=['Au (ppb)','Cu (ppm)','As (ppm)','Ag (ppm)'])
        colors_box = ['#FFD700','#B87333','#FF6B6B','#C0C0C0']
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color); patch.set_alpha(0.8)
        ax_box.set_ylabel("log10(valeur)"); ax_box.set_title("Distribution log des éléments", fontsize=11, fontweight='bold')
        ax_box.grid(True, linestyle=':', alpha=0.4)
        plt.tight_layout(); st.pyplot(fig_box)

        # Corrélation matricielle
        st.markdown("### 🔗 Matrice de corrélation")
        import seaborn as sns
        fig_cm, ax_cm = plt.subplots(figsize=(7,5))
        corr_m = ech_stat[elements].corr()
        sns.heatmap(corr_m, annot=True, fmt='.2f', cmap='coolwarm', ax=ax_cm,
                   linewidths=0.5, vmin=-1, vmax=1)
        ax_cm.set_title("Corrélations inter-éléments", fontsize=11, fontweight='bold')
        plt.tight_layout(); st.pyplot(fig_cm)

        csv_st = ech_stat.to_csv(index=False)
        st.download_button("📥 Télécharger données analytiques",
                          data=csv_st, file_name="analyses_geochimiques.csv", mime='text/csv')

st.markdown("---")
st.caption(f"⛏️ {NOM_PROSPECT} | {NOM_PERMIS} | RC · Aircore · Diamond · SGI · Estimation · Extension · Mapping · Sampling")
