import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
import plotly.express as px
from scipy.interpolate import griddata
import datetime

np.random.seed(42)

st.set_page_config(page_title="Dashboard Géologie Minière — Sénégal", layout="wide", page_icon="⛏️")

st.markdown("""
<style>
.main-header{background:linear-gradient(90deg,#1A237E,#0D47A1);color:white;
  padding:15px 20px;border-radius:10px;margin-bottom:20px;}
.kpi-card{background:#E3F2FD;border-radius:8px;padding:10px;text-align:center;}
</style>""", unsafe_allow_html=True)

st.markdown("""<div class='main-header'>
<h2>⛏️ Dashboard Géologie Minière — Projet Sénégal</h2>
<p>Sections · Cartes · 3D/2D · Logues · Planification · SGI · Monitoring · Weekly Report</p>
</div>""", unsafe_allow_html=True)

# ── DONNÉES ───────────────────────────────────────────────────────────────────
BASE_E, BASE_N = 350000.0, 1480000.0
LITHOS = ['Latérite','Saprolite','Granite altéré','Schiste','Quartzite aurifère','Granite frais']
LITHO_COLORS = {'Latérite':'#8B4513','Saprolite':'#DAA520','Granite altéré':'#CD853F',
                'Schiste':'#696969','Quartzite aurifère':'#FFD700','Granite frais':'#708090'}
STRUCTURES = ['Faille normale','Faille inverse','Cisaillement','Veine de quartz','Zone altérée']
STRUCT_COLORS = {'Faille normale':'#FF0000','Faille inverse':'#0000FF',
                 'Cisaillement':'#FF6600','Veine de quartz':'#FFFFFF','Zone altérée':'#4CAF50'}
ALTERATIONS = ['Silicification','Argilisation','Séricitisation','Carbonatation','Chloritisation','Épidotisation']
ALTER_COLORS = {'Silicification':'#FF6B35','Argilisation':'#A8D8EA','Séricitisation':'#AA96DA',
                'Carbonatation':'#FCBAD3','Chloritisation':'#B8F0B8','Épidotisation':'#FFE66D'}
MINERALISATIONS = ['Aurifère disséminée','Aurifère filonienne','Sulfures disséminés',
                   'Magnétite','Pyrite massive','Stérile']
MINER_COLORS = {'Aurifère disséminée':'#FFD700','Aurifère filonienne':'#FFA500',
                'Sulfures disséminés':'#808080','Magnétite':'#2F4F4F',
                'Pyrite massive':'#B8860B','Stérile':'#F5F5F5'}

# Forages
forages = []
types_forage = ['RC','Aircore','Diamond']
for i in range(12):
    ftype = np.random.choice(types_forage, p=[0.4,0.3,0.3])
    prof = np.random.choice([60,80,100,120,150,200]) if ftype=='Diamond' else np.random.choice([30,40,50,60])
    forages.append({
        'trou':f'SG{i+1:03d}','type':ftype,
        'easting':round(BASE_E+np.random.uniform(-300,300),1),
        'northing':round(BASE_N+np.random.uniform(-300,300),1),
        'elevation':round(np.random.uniform(80,120),1),
        'profondeur':prof,
        'azimut':round(np.random.uniform(0,360),1),
        'inclinaison':round(np.random.uniform(-85,-60),1),
        'statut':np.random.choice(['Complété','En cours','Planifié'],p=[0.6,0.2,0.2]),
        'Au_max_ppb':round(np.random.lognormal(2,1.5),1),
        'equipe':np.random.choice(['Équipe A','Équipe B','Équipe C']),
        'date_debut':(datetime.date.today()-datetime.timedelta(days=np.random.randint(1,60))).strftime('%Y-%m-%d'),
    })
df_forages = pd.DataFrame(forages)

# Intervalles
intervals = []
for _,f in df_forages.iterrows():
    depth=0
    while depth < f['profondeur']:
        thick=np.random.uniform(2,15)
        litho_idx=min(int(depth/f['profondeur']*len(LITHOS)),len(LITHOS)-1)
        litho=LITHOS[litho_idx] if np.random.random()>0.3 else np.random.choice(LITHOS)
        alter=np.random.choice(ALTERATIONS)
        miner=np.random.choice(MINERALISATIONS,p=[0.15,0.10,0.15,0.10,0.15,0.35])
        au=round(np.random.lognormal(-1,2),2) if litho=='Quartzite aurifère' else round(np.random.lognormal(-3,1),3)
        intervals.append({
            'trou':f['trou'],'type':f['type'],
            'de':round(depth,1),'a':round(min(depth+thick,f['profondeur']),1),
            'lithologie':litho,'alteration':alter,'mineralisation':miner,
            'Au_ppb':au,'Cu_ppm':round(np.random.uniform(1,80),1),
            'As_ppm':round(np.random.uniform(1,50),1),
            'Ag_ppm':round(np.random.uniform(0.1,10),2),
        })
        depth+=thick
df_intervals=pd.DataFrame(intervals)

# Données weekly
dates_week = pd.date_range(end=datetime.date.today(), periods=7)
weekly_data = pd.DataFrame({
    'date': dates_week,
    'metres_fores': np.random.randint(20,80,7),
    'trous_completes': np.random.randint(0,3,7),
    'incidents': np.random.randint(0,2,7),
    'Au_ppb_moyen': np.round(np.random.lognormal(2,0.8,7),1),
    'equipe': np.random.choice(['Équipe A','Équipe B'],7),
})

# ── TABS ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📐 Sections géologiques",
    "🗺️ Cartes lithologiques",
    "🏗️ Cartes structurales",
    "📊 Logues lithologiques",
    "🔬 Logues structuraux",
    "🌐 Modèle 3D",
    "📋 Planification",
    "🔄 Simulation déviation",
    "📡 Surveillance",
    "🧪 Essai SGI",
    "📈 Monitoring",
    "📅 Weekly Report",
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
    with col2:
        fig,ax=plt.subplots(figsize=(14,8))
        fig.patch.set_facecolor('#F8F8F0'); ax.set_facecolor('#E8F4F8')
        x_topo=np.linspace(0,500,200)
        topo=100+5*np.sin(x_topo/50)+3*np.cos(x_topo/30)+np.random.normal(0,0.5,200)
        ax.plot(x_topo,topo,'k-',linewidth=2,label='Topographie',zorder=5)
        ax.fill_between(x_topo,topo,0,alpha=0.15,color='brown')
        ax.axhline(y=0,color='blue',linestyle='--',linewidth=1,alpha=0.5,label='Ligne référence (0m)')
        x_positions=np.linspace(50,450,min(5,len(trous_dispo)))
        for idx,(xpos,trou) in enumerate(zip(x_positions,trous_dispo[:5])):
            f=df_forages[df_forages['trou']==trou].iloc[0]
            topo_val=float(np.interp(xpos,x_topo,topo))
            ints=df_intervals[df_intervals['trou']==trou].sort_values('de')
            for _,interval in ints.iterrows():
                y_top=topo_val-interval['de']*echelle*0.5
                y_bot=topo_val-interval['a']*echelle*0.5
                color=LITHO_COLORS.get(interval['lithologie'],'#888888')
                ax.fill_betweenx([y_bot,y_top],xpos-6,xpos+6,color=color,alpha=0.85)
                ax.plot([xpos-6,xpos+6,xpos+6,xpos-6,xpos-6],[y_top,y_top,y_bot,y_bot,y_top],'k-',linewidth=0.3)
            ax.text(xpos,topo_val+3,trou,ha='center',fontsize=7,fontweight='bold',color='#1A237E')
            ftype_color={'RC':'#FF5722','Aircore':'#2196F3','Diamond':'#9C27B0'}
            ax.text(xpos,topo_val+6,f['type'],ha='center',fontsize=6,color=ftype_color.get(f['type'],'black'),fontweight='bold')
            y_bot_total=topo_val-f['profondeur']*echelle*0.5
            ax.text(xpos+8,y_bot_total,f"{f['profondeur']}m",fontsize=6,va='center',color='#333')
        ax.text(10,max(topo)+8,f"Section {trou_sel} — Az.090°",fontsize=9,fontweight='bold',color='#1A237E',bbox=dict(boxstyle='round',facecolor='white',alpha=0.8))
        ax.annotate('',xy=(480,max(topo)+5),xytext=(480,max(topo)-5),arrowprops=dict(arrowstyle='->',color='black',lw=2))
        ax.text(480,max(topo)+7,'N',ha='center',fontsize=11,fontweight='bold')
        ax.plot([20,70],[5,5],'k-',linewidth=3); ax.text(45,2,'50 m',ha='center',fontsize=8,fontweight='bold')
        legend_patches=[mpatches.Patch(color=c,label=l) for l,c in LITHO_COLORS.items()]
        ax.legend(handles=legend_patches,loc='lower right',fontsize=7,title='Lithologie',ncol=2,framealpha=0.9)
        ax.set_xlabel("Distance (m)"); ax.set_ylabel("Élévation (m)")
        ax.set_title(f"Section géologique — {section_type} | Projet Sénégal",fontsize=12,fontweight='bold')
        ax.grid(True,linestyle=':',alpha=0.4); ax.set_xlim(0,500)
        plt.tight_layout(); st.pyplot(fig)
    st.info("🔴 RC — forage rapide | 🔵 Aircore — zone sapolitique | 🟣 Diamond — carotté haute précision")

# ══ TAB 2 — CARTES LITHO ══════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("🗺️ Carte Lithologique")
    prof_carte=st.slider("Profondeur (m)",0,150,20)
    fig2,ax2=plt.subplots(figsize=(10,8))
    fig2.patch.set_facecolor('#F5F5F0'); ax2.set_facecolor('#E8F4F8')
    for _,f in df_forages.iterrows():
        ints_d=df_intervals[(df_intervals['trou']==f['trou'])&(df_intervals['de']<=prof_carte)].tail(1)
        if len(ints_d)>0:
            litho=ints_d.iloc[0]['lithologie']; color=LITHO_COLORS.get(litho,'#888')
            marker={'RC':'^','Aircore':'o','Diamond':'s'}.get(f['type'],'o')
            ax2.scatter(f['easting'],f['northing'],c=color,s=150,marker=marker,edgecolors='black',linewidths=1,zorder=3)
            ax2.annotate(f['trou'],(f['easting'],f['northing']),textcoords="offset points",xytext=(5,5),fontsize=6)
    xmax=df_forages['easting'].max(); ymax=df_forages['northing'].max()
    ax2.annotate('',xy=(xmax+50,ymax+30),xytext=(xmax+50,ymax-20),arrowprops=dict(arrowstyle='->',color='black',lw=2.5))
    ax2.text(xmax+50,ymax+40,'N',ha='center',fontsize=14,fontweight='bold')
    xmin=df_forages['easting'].min(); ymin=df_forages['northing'].min()
    ax2.plot([xmin,xmin+200],[ymin-30,ymin-30],'k-',linewidth=3)
    ax2.text(xmin+100,ymin-45,'200 m',ha='center',fontsize=9,fontweight='bold')
    legend_patches=[mpatches.Patch(color=c,label=l) for l,c in LITHO_COLORS.items()]
    type_markers=[plt.Line2D([0],[0],marker='^',color='w',markerfacecolor='gray',markersize=8,label='RC'),
                  plt.Line2D([0],[0],marker='o',color='w',markerfacecolor='gray',markersize=8,label='Aircore'),
                  plt.Line2D([0],[0],marker='s',color='w',markerfacecolor='gray',markersize=8,label='Diamond')]
    ax2.legend(handles=legend_patches+type_markers,loc='lower right',fontsize=8,title='Lithologie & Type',ncol=2,framealpha=0.9)
    ax2.set_xlabel("Easting UTM (m)"); ax2.set_ylabel("Northing UTM (m)")
    ax2.set_title(f"Carte lithologique à {prof_carte}m — Projet Sénégal",fontsize=12,fontweight='bold')
    ax2.grid(True,linestyle=':',alpha=0.4); plt.tight_layout(); st.pyplot(fig2)

# ══ TAB 3 — CARTES STRUCT ═════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("🏗️ Carte Structurale")
    fig3,ax3=plt.subplots(figsize=(10,8))
    fig3.patch.set_facecolor('#F5F5F0'); ax3.set_facecolor('#F0EDE0')
    np.random.seed(10)
    for i in range(8):
        x1=np.random.uniform(df_forages['easting'].min(),df_forages['easting'].max())
        y1=np.random.uniform(df_forages['northing'].min(),df_forages['northing'].max())
        angle=np.random.uniform(0,180); length=np.random.uniform(100,400)
        x2=x1+length*np.cos(np.radians(angle)); y2=y1+length*np.sin(np.radians(angle))
        struct=np.random.choice(STRUCTURES); color=STRUCT_COLORS[struct]
        ls='-' if 'Faille' in struct else '--' if 'Veine' in struct else ':'
        ax3.plot([x1,x2],[y1,y2],color=color,linewidth=2.5,linestyle=ls,label=struct)
        ax3.text((x1+x2)/2,(y1+y2)/2,f'{int(angle)}°',fontsize=7,color=color,fontweight='bold')
    for _,f in df_forages.iterrows():
        ax3.scatter(f['easting'],f['northing'],c='black',s=60,zorder=3)
        ax3.annotate(f['trou'],(f['easting'],f['northing']),textcoords="offset points",xytext=(4,4),fontsize=6)
    xmax=df_forages['easting'].max(); ymax=df_forages['northing'].max()
    ax3.annotate('',xy=(xmax+50,ymax+30),xytext=(xmax+50,ymax-20),arrowprops=dict(arrowstyle='->',color='black',lw=2.5))
    ax3.text(xmax+50,ymax+40,'N',ha='center',fontsize=14,fontweight='bold')
    xmin=df_forages['easting'].min(); ymin=df_forages['northing'].min()
    ax3.plot([xmin,xmin+200],[ymin-30,ymin-30],'k-',linewidth=3)
    ax3.text(xmin+100,ymin-45,'200 m',ha='center',fontsize=9,fontweight='bold')
    handles,labels=ax3.get_legend_handles_labels()
    by_label=dict(zip(labels,handles))
    ax3.legend(by_label.values(),by_label.keys(),loc='lower right',fontsize=8,title='Structures',framealpha=0.9)
    ax3.set_xlabel("Easting UTM (m)"); ax3.set_ylabel("Northing UTM (m)")
    ax3.set_title("Carte structurale — Projet Sénégal",fontsize=12,fontweight='bold')
    ax3.grid(True,linestyle=':',alpha=0.4); plt.tight_layout(); st.pyplot(fig3)

# ══ TAB 4 — LOGUES LITHO ══════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("📊 Logues Lithologiques")
    col1,col2=st.columns([1,3])
    with col1:
        trou_logue=st.selectbox("Trou",df_forages['trou'].tolist(),key='lithologue')
        show_au=st.checkbox("Afficher Au (ppb)",True)
        show_as=st.checkbox("Afficher As (ppm)",True)
    with col2:
        f_info=df_forages[df_forages['trou']==trou_logue].iloc[0]
        ints=df_intervals[df_intervals['trou']==trou_logue].sort_values('de')
        ncols=1+(1 if show_au else 0)+(1 if show_as else 0)
        fig4,axes4=plt.subplots(1,ncols,figsize=(4*ncols,10),sharey=True)
        if ncols==1: axes4=[axes4]
        ax_litho=axes4[0]
        for _,interval in ints.iterrows():
            color=LITHO_COLORS.get(interval['lithologie'],'#888')
            ax_litho.fill_betweenx([interval['de'],interval['a']],0,1,color=color,alpha=0.85)
            ax_litho.plot([0,1],[interval['de'],interval['de']],'k-',linewidth=0.3)
            mid=(interval['de']+interval['a'])/2
            ax_litho.text(0.5,mid,interval['lithologie'],ha='center',va='center',fontsize=6,fontweight='bold')
            ax_litho.text(-0.05,interval['de'],f"{interval['de']}m",ha='right',fontsize=6)
        ax_litho.set_ylim(f_info['profondeur'],0); ax_litho.set_xlim(-0.1,1.1)
        ax_litho.set_title(f"Litho\n{trou_logue}",fontsize=9,fontweight='bold')
        ax_litho.set_xlabel("Lithologie"); ax_litho.set_ylabel("Profondeur (m)"); ax_litho.set_xticks([])
        cidx=1
        if show_au:
            axes4[cidx].barh([(i['de']+i['a'])/2 for _,i in ints.iterrows()],
                             ints['Au_ppb'].values,height=[(i['a']-i['de'])*0.8 for _,i in ints.iterrows()],
                             color='gold',edgecolor='orange',linewidth=0.5)
            axes4[cidx].axvline(x=100,color='red',linestyle='--',linewidth=1,label='Seuil 100ppb')
            axes4[cidx].set_xlabel("Au (ppb)"); axes4[cidx].set_title("Or",fontsize=9,fontweight='bold')
            axes4[cidx].legend(fontsize=7); cidx+=1
        if show_as:
            axes4[cidx].barh([(i['de']+i['a'])/2 for _,i in ints.iterrows()],
                             ints['As_ppm'].values,height=[(i['a']-i['de'])*0.8 for _,i in ints.iterrows()],
                             color='#FF6B6B',edgecolor='red',linewidth=0.5)
            axes4[cidx].set_xlabel("As (ppm)"); axes4[cidx].set_title("Arsenic",fontsize=9,fontweight='bold')
        plt.suptitle(f"{trou_logue} | {f_info['type']} | {f_info['profondeur']}m | Az:{f_info['azimut']}° Inc:{f_info['inclinaison']}°",fontsize=10,fontweight='bold')
        plt.tight_layout(); st.pyplot(fig4)

# ══ TAB 5 — LOGUES STRUCT ═════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("🔬 Logues Structuraux")
    col1,col2=st.columns([1,3])
    with col1:
        trou_struct=st.selectbox("Trou",df_forages['trou'].tolist(),key='structlogue')
    with col2:
        f_info2=df_forages[df_forages['trou']==trou_struct].iloc[0]
        ints2=df_intervals[df_intervals['trou']==trou_struct].sort_values('de')
        fig5,axes5=plt.subplots(1,3,figsize=(12,10),sharey=True)
        # Logue altération
        for _,interval in ints2.iterrows():
            color=ALTER_COLORS.get(interval['alteration'],'#888')
            axes5[0].fill_betweenx([interval['de'],interval['a']],0,1,color=color,alpha=0.85)
            mid=(interval['de']+interval['a'])/2
            axes5[0].text(0.5,mid,interval['alteration'],ha='center',va='center',fontsize=5.5,fontweight='bold')
            axes5[0].plot([0,1],[interval['de'],interval['de']],'k-',linewidth=0.3)
            axes5[0].text(-0.05,interval['de'],f"{interval['de']}m",ha='right',fontsize=5.5)
        axes5[0].set_ylim(f_info2['profondeur'],0); axes5[0].set_xlim(-0.1,1.1); axes5[0].set_xticks([])
        axes5[0].set_title("Altération",fontsize=9,fontweight='bold'); axes5[0].set_ylabel("Profondeur (m)")
        # Logue minéralisation
        for _,interval in ints2.iterrows():
            color=MINER_COLORS.get(interval['mineralisation'],'#888')
            axes5[1].fill_betweenx([interval['de'],interval['a']],0,1,color=color,alpha=0.85)
            mid=(interval['de']+interval['a'])/2
            axes5[1].text(0.5,mid,interval['mineralisation'],ha='center',va='center',fontsize=5,fontweight='bold')
            axes5[1].plot([0,1],[interval['de'],interval['de']],'k-',linewidth=0.3)
        axes5[1].set_ylim(f_info2['profondeur'],0); axes5[1].set_xlim(-0.1,1.1); axes5[1].set_xticks([])
        axes5[1].set_title("Minéralisation",fontsize=9,fontweight='bold')
        # Logue Cu
        axes5[2].barh([(i['de']+i['a'])/2 for _,i in ints2.iterrows()],
                      ints2['Cu_ppm'].values,height=[(i['a']-i['de'])*0.8 for _,i in ints2.iterrows()],
                      color='#B87333',edgecolor='brown',linewidth=0.5)
        axes5[2].set_xlabel("Cu (ppm)"); axes5[2].set_title("Cuivre",fontsize=9,fontweight='bold')
        plt.suptitle(f"Logue structural — {trou_struct} | {f_info2['type']} | {f_info2['profondeur']}m",fontsize=10,fontweight='bold')
        plt.tight_layout(); st.pyplot(fig5)
        # Légendes
        col_a,col_b=st.columns(2)
        with col_a:
            st.markdown("**Légende altération :**")
            leg_a=[mpatches.Patch(color=c,label=l) for l,c in ALTER_COLORS.items()]
            fig_la,ax_la=plt.subplots(figsize=(6,1)); ax_la.legend(handles=leg_a,loc='center',ncol=3,fontsize=7); ax_la.axis('off'); st.pyplot(fig_la)
        with col_b:
            st.markdown("**Légende minéralisation :**")
            leg_m=[mpatches.Patch(color=c,label=l) for l,c in MINER_COLORS.items()]
            fig_lm,ax_lm=plt.subplots(figsize=(6,1)); ax_lm.legend(handles=leg_m,loc='center',ncol=3,fontsize=7); ax_lm.axis('off'); st.pyplot(fig_lm)

# ══ TAB 6 — MODÈLE 3D ════════════════════════════════════════════════════════
with tabs[5]:
    st.subheader("🌐 Modèle 3D des forages")
    vue=st.radio("Vue",['3D interactif','2D plan','2D section'],horizontal=True)
    type_colors_3d={'RC':'red','Aircore':'blue','Diamond':'purple'}
    if vue=='3D interactif':
        fig3d=go.Figure()
        for _,f in df_forages.iterrows():
            inc_rad=np.radians(abs(f['inclinaison'])); az_rad=np.radians(f['azimut'])
            depths=np.linspace(0,f['profondeur'],20)
            xs=f['easting']+depths*np.sin(az_rad)*np.cos(inc_rad)
            ys=f['northing']+depths*np.cos(az_rad)*np.cos(inc_rad)
            zs=f['elevation']-depths*np.sin(inc_rad)
            color=type_colors_3d.get(f['type'],'gray')
            fig3d.add_trace(go.Scatter3d(x=xs,y=ys,z=zs,mode='lines+markers',
                line=dict(color=color,width=4),marker=dict(size=2,color=color),
                name=f"{f['trou']} ({f['type']})",
                hovertemplate=f"<b>{f['trou']}</b><br>{f['type']}<br>Prof:{f['profondeur']}m"))
        fig3d.update_layout(scene=dict(xaxis_title='Easting',yaxis_title='Northing',zaxis_title='Élévation',bgcolor='#F0F0F0'),
            title="Modèle 3D — Projet Sénégal",height=600)
        st.plotly_chart(fig3d,use_container_width=True)
        st.info("🖱️ Clic gauche=rotation | Scroll=zoom | Clic droit=déplacement")
    elif vue=='2D plan':
        fig2d,ax2d=plt.subplots(figsize=(10,8))
        for _,f in df_forages.iterrows():
            inc_rad=np.radians(abs(f['inclinaison'])); az_rad=np.radians(f['azimut'])
            depths=np.linspace(0,f['profondeur'],20)
            xs=f['easting']+depths*np.sin(az_rad)*np.cos(inc_rad)
            ys=f['northing']+depths*np.cos(az_rad)*np.cos(inc_rad)
            color=type_colors_3d.get(f['type'],'gray')
            ax2d.plot(xs,ys,color=color,linewidth=2)
            ax2d.scatter(f['easting'],f['northing'],color=color,s=80,zorder=3,edgecolors='black')
            ax2d.text(f['easting'],f['northing']+5,f['trou'],fontsize=7,ha='center')
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
            ax2ds.text(0,f['elevation']+1,f['trou'],fontsize=7)
        ax2ds.set_xlabel("Distance (m)"); ax2ds.set_ylabel("Élévation (m)")
        ax2ds.set_title("Vue en section 2D",fontsize=12,fontweight='bold'); ax2ds.grid(True,linestyle=':',alpha=0.4)
        plt.tight_layout(); st.pyplot(fig2ds)

# ══ TAB 7 — PLANIFICATION ════════════════════════════════════════════════════
with tabs[6]:
    st.subheader("📋 Planification des Forages")
    col1,col2=st.columns(2)
    with col1:
        st.markdown("### Statut")
        statut_counts=df_forages['statut'].value_counts()
        fig_s,ax_s=plt.subplots(figsize=(5,4))
        ax_s.pie(statut_counts.values,labels=statut_counts.index,colors=['#4CAF50','#FF9800','#2196F3'],autopct='%1.0f%%',startangle=90)
        ax_s.set_title("Répartition par statut"); st.pyplot(fig_s)
    with col2:
        st.markdown("### Tableau")
        df_plan=df_forages[['trou','type','profondeur','azimut','inclinaison','statut','equipe','Au_max_ppb']].copy()
        df_plan.columns=['Trou','Type','Prof.(m)','Az°','Inc°','Statut','Équipe','Au max(ppb)']
        st.dataframe(df_plan,use_container_width=True)
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Total forages",len(df_forages))
    c2.metric("Mètres totaux",f"{df_forages['profondeur'].sum():.0f} m")
    c3.metric("Au max",f"{df_forages['Au_max_ppb'].max():.1f} ppb")
    c4.metric("Avancement",f"{int((df_forages['statut']=='Complété').sum())/len(df_forages)*100:.0f}%")

# ══ TAB 8 — SIMULATION DÉVIATION ════════════════════════════════════════════
with tabs[7]:
    st.subheader("🔄 Simulation Déviation — Azimut & Inclinaison")
    col1,col2=st.columns([1,2])
    with col1:
        trou_dev=st.selectbox("Trou",df_forages['trou'].tolist(),key='dev')
        f_dev=df_forages[df_forages['trou']==trou_dev].iloc[0]
        az_init=st.slider("Azimut initial (°)",0,360,int(f_dev['azimut']))
        inc_init=st.slider("Inclinaison initiale (°)",-90,-45,int(f_dev['inclinaison']))
        deviation_az=st.slider("Déviation azimut/10m (°)",0.0,5.0,1.5,0.1)
        deviation_inc=st.slider("Déviation inclin./10m (°)",0.0,3.0,0.8,0.1)
        prof_sim=st.slider("Profondeur simulée (m)",10,200,int(f_dev['profondeur']))
    with col2:
        depths=np.arange(0,prof_sim+1,1)
        az_vals=az_init+deviation_az*depths/10*np.sin(depths/20)
        inc_vals=inc_init+deviation_inc*depths/10*np.cos(depths/15)
        xs_plan=np.cumsum(np.sin(np.radians(az_init))*np.cos(np.radians(inc_init))*np.ones(len(depths)))
        ys_plan=np.cumsum(np.cos(np.radians(az_init))*np.cos(np.radians(inc_init))*np.ones(len(depths)))
        zs_plan=f_dev['elevation']-np.cumsum(np.sin(np.radians(abs(inc_init)))*np.ones(len(depths)))
        xs_dev=np.cumsum(np.sin(np.radians(az_vals))*np.cos(np.radians(inc_vals)))
        ys_dev=np.cumsum(np.cos(np.radians(az_vals))*np.cos(np.radians(inc_vals)))
        zs_dev=f_dev['elevation']-np.cumsum(np.sin(np.radians(abs(inc_vals))))
        fig_dev=go.Figure()
        fig_dev.add_trace(go.Scatter3d(x=xs_plan+f_dev['easting'],y=ys_plan+f_dev['northing'],z=zs_plan,
            mode='lines',line=dict(color='blue',width=4,dash='dash'),name='Planifiée'))
        fig_dev.add_trace(go.Scatter3d(x=xs_dev+f_dev['easting'],y=ys_dev+f_dev['northing'],z=zs_dev,
            mode='lines',line=dict(color='red',width=4),name='Déviée'))
        fig_dev.update_layout(scene=dict(xaxis_title='Easting',yaxis_title='Northing',zaxis_title='Élévation'),
            title=f"Simulation déviation — {trou_dev}",height=500)
        st.plotly_chart(fig_dev,use_container_width=True)
        dev_totale=np.sqrt((xs_dev[-1]-xs_plan[-1])**2+(ys_dev[-1]-ys_plan[-1])**2)
        if dev_totale>10:
            st.error(f"⚠️ Déviation : **{dev_totale:.1f} m** — Correction nécessaire !")
        else:
            st.success(f"✅ Déviation : **{dev_totale:.1f} m** — Acceptable")

# ══ TAB 9 — SURVEILLANCE ════════════════════════════════════════════════════
with tabs[8]:
    st.subheader("📡 Surveillance des Forages")
    col1,col2=st.columns(2)
    with col1:
        st.markdown("### 🏔️ Forages de surface (RC & Aircore)")
        for _,f in df_forages[df_forages['type'].isin(['RC','Aircore'])].iterrows():
            icon='🟢' if f['statut']=='Complété' else '🟡' if f['statut']=='En cours' else '🔵'
            st.markdown(f"**{icon} {f['trou']}** ({f['type']}) | {f['profondeur']}m | {f['equipe']} | Au:{f['Au_max_ppb']} ppb")
    with col2:
        st.markdown("### ⛏️ Forages souterrains (Diamond)")
        for _,f in df_forages[df_forages['type']=='Diamond'].iterrows():
            icon='🟢' if f['statut']=='Complété' else '🟡' if f['statut']=='En cours' else '🔵'
            prog=min(1.0,f['profondeur']/200)
            st.markdown(f"**{icon} {f['trou']}** | {f['profondeur']}m")
            st.progress(prog,text=f"{f['profondeur']}m / 200m cible")
    st.markdown("---")
    c1,c2,c3,c4,c5=st.columns(5)
    c1.metric("Actifs",int((df_forages['statut']=='En cours').sum()))
    c2.metric("Mètres forés",f"{df_forages[df_forages['statut']=='Complété']['profondeur'].sum():.0f}m")
    c3.metric("Au max",f"{df_forages['Au_max_ppb'].max():.0f} ppb")
    c4.metric("Trous RC",int((df_forages['type']=='RC').sum()))
    c5.metric("Trous Diamond",int((df_forages['type']=='Diamond').sum()))
    st.markdown("### ⚠️ Alertes")
    prob=df_forages[df_forages['Au_max_ppb']>500]
    if len(prob)>0:
        for _,f in prob.iterrows():
            st.warning(f"🔔 **{f['trou']}** — Anomalie Au: {f['Au_max_ppb']:.0f} ppb → Priorité !")
    else:
        st.success("✅ Aucune anomalie critique")

# ══ TAB 10 — ESSAI SGI ══════════════════════════════════════════════════════
with tabs[9]:
    st.subheader("🧪 Essai SGI — Minéralisation & Altération")

    col1,col2=st.columns([1,2])
    with col1:
        trou_sgi=st.selectbox("Trou",df_forages['trou'].tolist(),key='sgi')
        seuil_au=st.number_input("Seuil Au minéralisé (ppb)",min_value=10,max_value=1000,value=100)
        seuil_cu=st.number_input("Seuil Cu significatif (ppm)",min_value=5,max_value=200,value=50)

    with col2:
        ints_sgi=df_intervals[df_intervals['trou']==trou_sgi].sort_values('de').copy()
        ints_sgi['mineralisé']=ints_sgi['Au_ppb']>=seuil_au
        ints_sgi['Cu_significatif']=ints_sgi['Cu_ppm']>=seuil_cu

        # Résumé SGI
        total_m=ints_sgi['a'].max()-ints_sgi['de'].min()
        m_miner=ints_sgi[ints_sgi['mineralisé']].apply(lambda r: r['a']-r['de'],axis=1).sum()
        pct_miner=m_miner/total_m*100 if total_m>0 else 0

        c1,c2,c3=st.columns(3)
        c1.metric("Intervalles minéralisés",f"{m_miner:.1f} m")
        c2.metric("% minéralisé",f"{pct_miner:.1f}%")
        c3.metric("Au max",f"{ints_sgi['Au_ppb'].max():.1f} ppb")

        # Graphique SGI
        fig_sgi,axes_sgi=plt.subplots(1,4,figsize=(14,10),sharey=True)
        f_sgi=df_forages[df_forages['trou']==trou_sgi].iloc[0]

        # Logue altération
        for _,i in ints_sgi.iterrows():
            color=ALTER_COLORS.get(i['alteration'],'#888')
            axes_sgi[0].fill_betweenx([i['de'],i['a']],0,1,color=color,alpha=0.85)
            mid=(i['de']+i['a'])/2
            axes_sgi[0].text(0.5,mid,i['alteration'][:10],ha='center',va='center',fontsize=5.5,fontweight='bold')
            axes_sgi[0].plot([0,1],[i['de'],i['de']],'k-',linewidth=0.3)
            axes_sgi[0].text(-0.05,i['de'],f"{i['de']}m",ha='right',fontsize=5.5)
        axes_sgi[0].set_ylim(f_sgi['profondeur'],0); axes_sgi[0].set_xticks([])
        axes_sgi[0].set_title("Altération",fontsize=9,fontweight='bold'); axes_sgi[0].set_ylabel("Profondeur (m)")

        # Logue minéralisation
        for _,i in ints_sgi.iterrows():
            color=MINER_COLORS.get(i['mineralisation'],'#888')
            axes_sgi[1].fill_betweenx([i['de'],i['a']],0,1,color=color,alpha=0.85)
            mid=(i['de']+i['a'])/2
            axes_sgi[1].text(0.5,mid,i['mineralisation'][:12],ha='center',va='center',fontsize=5,fontweight='bold')
            axes_sgi[1].plot([0,1],[i['de'],i['de']],'k-',linewidth=0.3)
        axes_sgi[1].set_ylim(f_sgi['profondeur'],0); axes_sgi[1].set_xticks([])
        axes_sgi[1].set_title("Minéralisation",fontsize=9,fontweight='bold')

        # Au ppb
        colors_au=['#FFD700' if v>=seuil_au else '#EEEEEE' for v in ints_sgi['Au_ppb']]
        axes_sgi[2].barh([(i['de']+i['a'])/2 for _,i in ints_sgi.iterrows()],
                         ints_sgi['Au_ppb'].values,
                         height=[(i['a']-i['de'])*0.8 for _,i in ints_sgi.iterrows()],
                         color=colors_au,edgecolor='orange',linewidth=0.5)
        axes_sgi[2].axvline(x=seuil_au,color='red',linestyle='--',linewidth=1.5,label=f'Seuil {seuil_au}ppb')
        axes_sgi[2].set_xlabel("Au (ppb)"); axes_sgi[2].set_title("Or — SGI",fontsize=9,fontweight='bold')
        axes_sgi[2].legend(fontsize=7)

        # Cu ppm
        colors_cu=['#B87333' if v>=seuil_cu else '#EEEEEE' for v in ints_sgi['Cu_ppm']]
        axes_sgi[3].barh([(i['de']+i['a'])/2 for _,i in ints_sgi.iterrows()],
                         ints_sgi['Cu_ppm'].values,
                         height=[(i['a']-i['de'])*0.8 for _,i in ints_sgi.iterrows()],
                         color=colors_cu,edgecolor='brown',linewidth=0.5)
        axes_sgi[3].axvline(x=seuil_cu,color='blue',linestyle='--',linewidth=1.5,label=f'Seuil {seuil_cu}ppm')
        axes_sgi[3].set_xlabel("Cu (ppm)"); axes_sgi[3].set_title("Cuivre — SGI",fontsize=9,fontweight='bold')
        axes_sgi[3].legend(fontsize=7)

        plt.suptitle(f"Essai SGI — {trou_sgi} | {f_sgi['type']} | {f_sgi['profondeur']}m",fontsize=11,fontweight='bold')
        plt.tight_layout(); st.pyplot(fig_sgi)

    st.info(f"""
    🧪 **Interprétation SGI — {trou_sgi} :**
    - **{pct_miner:.1f}%** du forage est minéralisé (Au ≥ {seuil_au} ppb)
    - Altérations dominantes : silicification et séricitisation → contexte aurifère favorable
    - Les zones minéralisées coïncident avec les veines de quartz et la silicification
    - **Recommandation :** Approfondir les investigations aux intervalles > {seuil_au} ppb
    """)

# ══ TAB 11 — MONITORING ════════════════════════════════════════════════════
with tabs[10]:
    st.subheader("📈 Monitoring — Suivi en temps réel")

    col1,col2,col3=st.columns(3)
    col1.metric("Mètres forés aujourd'hui",f"{np.random.randint(30,80)} m","↑ +12 vs hier")
    col2.metric("Trous actifs",int((df_forages['statut']=='En cours').sum()))
    col3.metric("Incidents",np.random.randint(0,2),"0 critique")

    # Avancement par équipe
    st.markdown("### 🏗️ Avancement par équipe")
    equipes=df_forages.groupby('equipe').agg(
        trous=('trou','count'),
        metres=('profondeur','sum'),
        au_max=('Au_max_ppb','max')
    ).reset_index()
    fig_eq,ax_eq=plt.subplots(figsize=(8,4))
    ax_eq.bar(equipes['equipe'],equipes['metres'],color=['#2196F3','#4CAF50','#FF9800'],edgecolor='black',linewidth=0.5)
    ax_eq.set_ylabel("Mètres forés totaux"); ax_eq.set_title("Mètres forés par équipe",fontsize=11,fontweight='bold')
    for i,(v,e) in enumerate(zip(equipes['metres'],equipes['equipe'])):
        ax_eq.text(i,v+5,f"{v:.0f}m",ha='center',fontsize=9,fontweight='bold')
    plt.tight_layout(); st.pyplot(fig_eq)

    # Suivi journalier
    st.markdown("### 📅 Suivi journalier — 30 derniers jours")
    dates_30=pd.date_range(end=datetime.date.today(),periods=30)
    metres_30=np.random.randint(20,80,30)
    fig_jour,ax_jour=plt.subplots(figsize=(12,4))
    ax_jour.fill_between(dates_30,metres_30,alpha=0.3,color='#2196F3')
    ax_jour.plot(dates_30,metres_30,'b-o',markersize=4,linewidth=1.5)
    ax_jour.axhline(y=metres_30.mean(),color='red',linestyle='--',linewidth=1.5,label=f'Moyenne: {metres_30.mean():.0f}m/j')
    ax_jour.set_ylabel("Mètres forés/jour"); ax_jour.set_title("Production journalière",fontsize=11,fontweight='bold')
    ax_jour.legend(fontsize=9); ax_jour.grid(True,linestyle=':',alpha=0.4)
    plt.xticks(rotation=45); plt.tight_layout(); st.pyplot(fig_jour)

    # Anomalies géochimiques
    st.markdown("### 🔬 Suivi des anomalies géochimiques")
    fig_anom,axes_anom=plt.subplots(1,2,figsize=(12,4))
    au_vals=df_forages['Au_max_ppb'].sort_values(ascending=False)
    axes_anom[0].bar(df_forages.sort_values('Au_max_ppb',ascending=False)['trou'],
                     au_vals,color=['#FFD700' if v>200 else '#DDDDDD' for v in au_vals],
                     edgecolor='orange',linewidth=0.5)
    axes_anom[0].axhline(y=200,color='red',linestyle='--',linewidth=1.5,label='Seuil 200 ppb')
    axes_anom[0].set_ylabel("Au max (ppb)"); axes_anom[0].set_title("Au max par trou",fontsize=10,fontweight='bold')
    axes_anom[0].legend(fontsize=8); plt.setp(axes_anom[0].xaxis.get_majorticklabels(),rotation=45,fontsize=7)

    type_counts=df_forages['type'].value_counts()
    axes_anom[1].pie(type_counts.values,labels=type_counts.index,colors=['#FF5722','#2196F3','#9C27B0'],
                     autopct='%1.0f%%',startangle=90)
    axes_anom[1].set_title("Répartition par type",fontsize=10,fontweight='bold')
    plt.tight_layout(); st.pyplot(fig_anom)

# ══ TAB 12 — WEEKLY REPORT ══════════════════════════════════════════════════
with tabs[11]:
    st.subheader("📅 Rapport Hebdomadaire — Weekly Report")

    semaine=st.date_input("Semaine du",datetime.date.today()-datetime.timedelta(days=7))
    st.markdown(f"**Période :** {semaine} → {semaine+datetime.timedelta(days=6)}")

    # KPIs semaine
    c1,c2,c3,c4,c5=st.columns(5)
    total_m_week=int(weekly_data['metres_fores'].sum())
    total_trous=int(weekly_data['trous_completes'].sum())
    total_inc=int(weekly_data['incidents'].sum())
    au_max_week=float(weekly_data['Au_ppb_moyen'].max())
    objectif=350
    c1.metric("Mètres forés",f"{total_m_week} m",f"{total_m_week-objectif:+d} vs objectif {objectif}m")
    c2.metric("Trous complétés",total_trous)
    c3.metric("Incidents",total_inc,"0 grave" if total_inc==0 else "⚠️ vérifier")
    c4.metric("Au max semaine",f"{au_max_week:.1f} ppb")
    c5.metric("Objectif atteint","✅ Oui" if total_m_week>=objectif else "❌ Non")

    # Graphique weekly
    fig_week,axes_week=plt.subplots(1,3,figsize=(14,4))
    axes_week[0].bar([d.strftime('%a') for d in weekly_data['date']],weekly_data['metres_fores'],
                     color='#2196F3',edgecolor='black',linewidth=0.5)
    axes_week[0].axhline(y=objectif/7,color='red',linestyle='--',linewidth=1.5,label=f'Obj/jour: {objectif//7}m')
    axes_week[0].set_ylabel("Mètres"); axes_week[0].set_title("Mètres forés/jour",fontsize=10,fontweight='bold')
    axes_week[0].legend(fontsize=8)

    axes_week[1].bar([d.strftime('%a') for d in weekly_data['date']],weekly_data['Au_ppb_moyen'],
                     color='#FFD700',edgecolor='orange',linewidth=0.5)
    axes_week[1].set_ylabel("Au (ppb)"); axes_week[1].set_title("Au moyen/jour",fontsize=10,fontweight='bold')

    axes_week[2].bar([d.strftime('%a') for d in weekly_data['date']],weekly_data['incidents'],
                     color=['#4CAF50' if v==0 else '#FF5722' for v in weekly_data['incidents']],
                     edgecolor='black',linewidth=0.5)
    axes_week[2].set_ylabel("Incidents"); axes_week[2].set_title("Incidents/jour",fontsize=10,fontweight='bold')
    plt.suptitle(f"Rapport hebdomadaire — Semaine du {semaine}",fontsize=12,fontweight='bold')
    plt.tight_layout(); st.pyplot(fig_week)

    # Tableau hebdomadaire
    st.markdown("### 📋 Détail journalier")
    weekly_display=weekly_data.copy()
    weekly_display['date']=weekly_display['date'].dt.strftime('%Y-%m-%d (%A)')
    weekly_display.columns=['Date','Mètres forés','Trous complétés','Incidents','Au moy.(ppb)','Équipe']
    st.dataframe(weekly_display,use_container_width=True)

    # Commentaires & recommandations
    st.markdown("### 📝 Commentaires & Recommandations")
    col1,col2=st.columns(2)
    with col1:
        st.success(f"""
        ✅ **Points positifs :**
        - {total_m_week} m forés cette semaine
        - Au max détecté : {au_max_week:.1f} ppb
        - {total_trous} trous complétés
        - Équipes mobilisées : {', '.join(weekly_data['equipe'].unique())}
        """)
    with col2:
        st.warning(f"""
        ⚠️ **Points à surveiller :**
        - {total_inc} incident(s) reporté(s) cette semaine
        - {"Objectif non atteint — revoir la planification" if total_m_week<objectif else "Objectif atteint — maintenir le rythme"}
        - Vérifier les trous stoppés
        - Planifier les forages de la semaine prochaine
        """)

    # Bouton export
    st.markdown("### 💾 Export")
    csv=weekly_display.to_csv(index=False)
    st.download_button("📥 Télécharger le rapport CSV",data=csv,
                       file_name=f"weekly_report_{semaine}.csv",mime='text/csv')

st.markdown("---")
st.caption("⛏️ Dashboard Géologie Minière — Projet Sénégal | RC · Aircore · Diamond · SGI · Monitoring · Weekly")
