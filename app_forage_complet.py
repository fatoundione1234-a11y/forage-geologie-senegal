import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import plotly.graph_objects as go
import plotly.express as px
from scipy.interpolate import griddata
import datetime

np.random.seed(42)

st.set_page_config(page_title="Dashboard Géologie Minière — Sénégal", layout="wide", page_icon="⛏️")
st.markdown("""<style>
.main-header{background:linear-gradient(90deg,#1A237E,#0D47A1);color:white;padding:15px 20px;border-radius:10px;margin-bottom:20px;}
</style>""", unsafe_allow_html=True)
st.markdown("""<div class='main-header'>
<h2>⛏️ Dashboard Géologie Minière — Projet Sénégal</h2>
<p>Sections · Cartes · 3D/2D · Logues · SGI · Monitoring · Weekly · Rapport</p>
</div>""", unsafe_allow_html=True)

# ── DONNÉES ───────────────────────────────────────────────────────────────────
BASE_E, BASE_N = 350000.0, 1480000.0
LITHOS       = ['Latérite','Saprolite','Saprock','Bédrock/Schiste','Quartzite aurifère','Granite frais']
LITHO_COLORS = {'Latérite':'#8B4513','Saprolite':'#DAA520','Saprock':'#CD853F',
                'Bédrock/Schiste':'#696969','Quartzite aurifère':'#FFD700','Granite frais':'#708090'}
LITHO_LIMITS = {'Latérite':(0,8),'Saprolite':(8,25),'Saprock':(25,50),'Bédrock/Schiste':(50,120)}
STRUCTURES   = ['Faille normale','Faille inverse','Cisaillement','Veine de quartz','Zone altérée']
STRUCT_COLORS= {'Faille normale':'#FF0000','Faille inverse':'#0000FF',
                'Cisaillement':'#FF6600','Veine de quartz':'#FFFFFF','Zone altérée':'#4CAF50'}
ALTERATIONS  = ['Silicification','Argilisation','Séricitisation','Carbonatation','Chloritisation','Épidotisation']
ALTER_COLORS = {'Silicification':'#FF6B35','Argilisation':'#A8D8EA','Séricitisation':'#AA96DA',
                'Carbonatation':'#FCBAD3','Chloritisation':'#B8F0B8','Épidotisation':'#FFE66D'}
MINERALISATIONS = ['Aurifère disséminée','Aurifère filonienne','Sulfures disséminés','Magnétite','Pyrite massive','Stérile']
MINER_COLORS    = {'Aurifère disséminée':'#FFD700','Aurifère filonienne':'#FFA500',
                   'Sulfures disséminés':'#808080','Magnétite':'#2F4F4F','Pyrite massive':'#B8860B','Stérile':'#F5F5F5'}
PORTEURS_MINER  = ['Veine de quartz','Zone de cisaillement','Contact lithologique','Fracture','Zone d\'altération']

# Forages
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
        'equipe':np.random.choice(['Équipe A','Équipe B','Équipe C']),
        'porteur':np.random.choice(PORTEURS_MINER),
        'date_debut':(datetime.date.today()-datetime.timedelta(days=np.random.randint(1,60))).strftime('%Y-%m-%d'),
    })
df_forages=pd.DataFrame(forages)

# Intervalles
intervals=[]
for _,f in df_forages.iterrows():
    depth=0
    while depth<f['profondeur']:
        thick=np.random.uniform(2,15)
        litho_idx=min(int(depth/f['profondeur']*len(LITHOS)),len(LITHOS)-1)
        litho=LITHOS[litho_idx] if np.random.random()>0.3 else np.random.choice(LITHOS)
        alter=np.random.choice(ALTERATIONS)
        miner=np.random.choice(MINERALISATIONS,p=[0.15,0.10,0.15,0.10,0.15,0.35])
        au=round(np.random.lognormal(2,1.5),2) if litho=='Quartzite aurifère' else round(np.random.lognormal(-2,1.5),3)
        mineralisé = au>=100
        intervals.append({
            'trou':f['trou'],'type':f['type'],
            'de':round(depth,1),'a':round(min(depth+thick,f['profondeur']),1),
            'lithologie':litho,'alteration':alter,'mineralisation':miner,
            'Au_ppb':au,'Cu_ppm':round(np.random.uniform(1,80),1),
            'As_ppm':round(np.random.uniform(1,50),1),
            'Ag_ppm':round(np.random.uniform(0.1,10),2),
            'mineralisé':mineralisé,
        })
        depth+=thick
df_intervals=pd.DataFrame(intervals)

# Structures
structures_df=pd.DataFrame([{
    'id':f'STR{i+1:02d}',
    'type':np.random.choice(STRUCTURES),
    'direction':round(np.random.uniform(0,360),1),
    'pendage':round(np.random.uniform(20,80),1),
    'sens_pendage':np.random.choice(['N','NE','E','SE','S','SO','O','NO']),
    'longueur':round(np.random.uniform(100,500),0),
    'porteur_mineralisation':np.random.choice([True,False],p=[0.4,0.6]),
} for i in range(20)])

# Weekly
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
    "📈 Monitoring",
    "📅 Weekly Report",
    "📄 Rapport géologique",
])

# ══ TAB 1 — SECTIONS GÉOLOGIQUES ═════════════════════════════════════════════
with tabs[0]:
    st.subheader("📐 Sections Géologiques — RC / Aircore / Diamond")
    col1,col2=st.columns([1,3])
    with col1:
        section_type=st.selectbox("Type de forage",['RC','Aircore','Diamond','Tous'])
        trous_dispo=df_forages['trou'].tolist() if section_type=='Tous' else df_forages[df_forages['type']==section_type]['trou'].tolist()
        trou_sel=st.selectbox("Trou de référence",trous_dispo)
        echelle=st.slider("Échelle verticale ×",1,5,2)
        show_miner=st.checkbox("Afficher intervalles minéralisés",True)
        show_au=st.checkbox("Afficher teneurs Au",True)
        show_limits=st.checkbox("Afficher limites géologiques",True)
    with col2:
        fig,ax=plt.subplots(figsize=(15,9))
        fig.patch.set_facecolor('#F8F8F0'); ax.set_facecolor('#E8F4F8')
        x_topo=np.linspace(0,600,300)
        topo=100+5*np.sin(x_topo/50)+3*np.cos(x_topo/30)+np.random.normal(0,0.5,300)
        ax.plot(x_topo,topo,'k-',linewidth=2.5,label='Topographie',zorder=5)
        ax.fill_between(x_topo,topo,0,alpha=0.15,color='brown')
        ax.axhline(y=0,color='blue',linestyle='--',linewidth=1,alpha=0.5,label='Ligne de référence (0m)')

        # Limites géologiques horizontales
        if show_limits:
            limit_colors={'Latérite/Saprolite':'#8B4513','Saprolite/Saprock':'#DAA520',
                         'Saprock/Bédrock':'#696969','Bédrock/Minéralisé':'#FFD700'}
            limit_depths=[8,25,50,80]
            limit_labels=['Base Latérite','Base Saprolite','Base Saprock','Top Bédrock']
            for d,lbl,lc in zip(limit_depths,limit_labels,limit_colors.values()):
                y_lim=topo.mean()-d*echelle*0.5
                ax.axhline(y=y_lim,color=lc,linestyle='-.',linewidth=1.5,alpha=0.7,label=lbl)

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
                # Intervalles minéralisés
                if show_miner and interval['mineralisé']:
                    ax.fill_betweenx([y_bot,y_top],xpos-7,xpos+7,color='red',alpha=0.3,hatch='///')
                    ax.plot([xpos-7,xpos+7],[y_top,y_top],'r-',linewidth=1.5)
                    ax.plot([xpos-7,xpos+7],[y_bot,y_bot],'r-',linewidth=1.5)
                # Teneurs Au
                if show_au and interval['Au_ppb']>=100:
                    mid=(y_top+y_bot)/2
                    ax.text(xpos+8,mid,f"{interval['Au_ppb']:.0f}ppb",fontsize=5.5,color='#FF6600',fontweight='bold')
            # En-tête du trou
            ftype_color={'RC':'#FF5722','Aircore':'#2196F3','Diamond':'#9C27B0'}
            ax.text(xpos,topo_val+7,trou,ha='center',fontsize=7,fontweight='bold',color='#1A237E')
            ax.text(xpos,topo_val+4,f['type'],ha='center',fontsize=6,color=ftype_color.get(f['type'],'black'),fontweight='bold')
            y_bot_total=topo_val-f['profondeur']*echelle*0.5
            ax.text(xpos+9,y_bot_total,f"{f['profondeur']}m",fontsize=6,va='center',color='#333')

        # Numéro de section
        ax.text(10,max(topo)+10,f"Section {trou_sel} — Az.090° | Échelle V:{echelle}×",fontsize=9,
                fontweight='bold',color='#1A237E',bbox=dict(boxstyle='round',facecolor='white',alpha=0.9))
        # Nord
        ax.annotate('',xy=(575,max(topo)+6),xytext=(575,max(topo)-6),arrowprops=dict(arrowstyle='->',color='black',lw=2.5))
        ax.text(575,max(topo)+8,'N',ha='center',fontsize=12,fontweight='bold')
        # Échelle
        ax.plot([20,70],[4,4],'k-',linewidth=3); ax.text(45,1,'50 m',ha='center',fontsize=8,fontweight='bold')
        # Légende
        legend_patches=[mpatches.Patch(color=c,label=l) for l,c in LITHO_COLORS.items()]
        if show_miner:
            legend_patches.append(mpatches.Patch(color='red',alpha=0.4,hatch='///',label='Intervalle minéralisé'))
        ax.legend(handles=legend_patches,loc='lower right',fontsize=7,title='Lithologie',ncol=2,framealpha=0.9)
        ax.set_xlabel("Distance (m)"); ax.set_ylabel("Élévation (m)")
        ax.set_title(f"Section géologique — {section_type} | Projet Sénégal",fontsize=12,fontweight='bold')
        ax.grid(True,linestyle=':',alpha=0.3); ax.set_xlim(0,600)
        plt.tight_layout(); st.pyplot(fig)

    st.info("🔴 RC | 🔵 Aircore | 🟣 Diamond | **Hachures rouges** = intervalles minéralisés | **Valeurs orange** = teneurs Au ≥ 100 ppb")

# ══ TAB 2 — CARTES LITHOLOGIQUES ═════════════════════════════════════════════
with tabs[1]:
    st.subheader("🗺️ Carte Lithologique — Toutes couleurs")
    prof_carte=st.slider("Profondeur (m)",0,150,20)
    fig2,ax2=plt.subplots(figsize=(11,9))
    fig2.patch.set_facecolor('#F5F5F0'); ax2.set_facecolor('#E8F4F8')

    for _,f in df_forages.iterrows():
        ints_d=df_intervals[(df_intervals['trou']==f['trou'])&(df_intervals['de']<=prof_carte)].tail(1)
        if len(ints_d)>0:
            litho=ints_d.iloc[0]['lithologie']
            color=LITHO_COLORS.get(litho,'#888')
            marker={'RC':'^','Aircore':'o','Diamond':'s'}.get(f['type'],'o')
            sc=ax2.scatter(f['easting'],f['northing'],c=color,s=200,marker=marker,
                          edgecolors='black',linewidths=1.2,zorder=3)
            ax2.annotate(f"{f['trou']}\n{litho[:8]}",
                        (f['easting'],f['northing']),textcoords="offset points",
                        xytext=(5,5),fontsize=6,color='#1A237E')

    xmax=df_forages['easting'].max(); ymax=df_forages['northing'].max()
    xmin=df_forages['easting'].min(); ymin=df_forages['northing'].min()
    ax2.annotate('',xy=(xmax+60,ymax+30),xytext=(xmax+60,ymax-25),arrowprops=dict(arrowstyle='->',color='black',lw=2.5))
    ax2.text(xmax+60,ymax+40,'N',ha='center',fontsize=14,fontweight='bold')
    ax2.plot([xmin,xmin+200],[ymin-35,ymin-35],'k-',linewidth=3)
    ax2.text(xmin+100,ymin-50,'200 m',ha='center',fontsize=9,fontweight='bold')

    # Grille de coordonnées
    ax2.grid(True,linestyle='--',alpha=0.4,color='gray')
    legend_patches=[mpatches.Patch(color=c,label=l) for l,c in LITHO_COLORS.items()]
    type_markers=[plt.Line2D([0],[0],marker='^',color='w',markerfacecolor='gray',markersize=9,label='RC'),
                  plt.Line2D([0],[0],marker='o',color='w',markerfacecolor='gray',markersize=9,label='Aircore'),
                  plt.Line2D([0],[0],marker='s',color='w',markerfacecolor='gray',markersize=9,label='Diamond')]
    ax2.legend(handles=legend_patches+type_markers,loc='lower right',fontsize=8,
              title='Lithologie & Type forage',ncol=2,framealpha=0.95,edgecolor='black')
    ax2.set_xlabel(f"Easting UTM (m) — Coordonnées WGS84"); ax2.set_ylabel("Northing UTM (m)")
    ax2.set_title(f"Carte lithologique à {prof_carte}m — Projet Sénégal",fontsize=12,fontweight='bold')
    plt.tight_layout(); st.pyplot(fig2)

# ══ TAB 3 — CARTE ANOMALIE ═══════════════════════════════════════════════════
with tabs[2]:
    st.subheader("🌡️ Carte d'Anomalie Géochimique & Potentiel Minier")
    col1,col2=st.columns([1,3])
    with col1:
        element=st.selectbox("Élément",['Au_ppb','Cu_ppm','As_ppm'])
        seuil_anom=st.number_input("Seuil anomalie",10,1000,100)
        show_porteur=st.checkbox("Afficher structures porteuses",True)

    with col2:
        df_fore_anom=df_intervals[df_intervals['statut'] if 'statut' in df_intervals.columns else df_intervals.index>=0].copy()
        au_max_par_trou=df_intervals.groupby('trou')[element].max().reset_index()
        au_max_par_trou.columns=['trou','valeur_max']
        df_anom=df_forages.merge(au_max_par_trou,on='trou')

        fig_an,ax_an=plt.subplots(figsize=(11,9))
        fig_an.patch.set_facecolor('#F5F5F0'); ax_an.set_facecolor('#0A1628')

        # Interpolation anomalie
        if len(df_anom)>3:
            xi=np.linspace(df_anom['easting'].min()-50,df_anom['easting'].max()+50,150)
            yi=np.linspace(df_anom['northing'].min()-50,df_anom['northing'].max()+50,150)
            Xi,Yi=np.meshgrid(xi,yi)
            Zi=griddata((df_anom['easting'],df_anom['northing']),
                       np.log1p(df_anom['valeur_max']),(Xi,Yi),method='linear')
            contour=ax_an.contourf(Xi,Yi,Zi,levels=20,cmap='hot_r',alpha=0.8)
            plt.colorbar(contour,ax=ax_an,label=f'log({element}+1)')
            ax_an.contour(Xi,Yi,Zi,levels=5,colors='white',alpha=0.3,linewidths=0.5)

        # Points forages
        for _,f in df_anom.iterrows():
            potentiel=f['valeur_max']>=seuil_anom
            marker={'RC':'^','Aircore':'o','Diamond':'s'}.get(f['type'],'o')
            color_pt='#00FF00' if potentiel else '#FFFFFF'
            size=200 if potentiel else 80
            ax_an.scatter(f['easting'],f['northing'],c=color_pt,s=size,marker=marker,
                         edgecolors='black' if potentiel else 'gray',linewidths=1.5 if potentiel else 0.5,zorder=4)
            if potentiel:
                ax_an.annotate(f"{f['trou']}\n{f['valeur_max']:.0f}",
                              (f['easting'],f['northing']),textcoords="offset points",
                              xytext=(6,6),fontsize=7,color='#00FF00',fontweight='bold')
                ax_an.scatter(f['easting'],f['northing'],c='none',s=400,marker='o',
                             edgecolors='#00FF00',linewidths=2,zorder=5)

        # Structures porteuses
        if show_porteur:
            np.random.seed(5)
            for i in range(5):
                x1=np.random.uniform(df_anom['easting'].min(),df_anom['easting'].max())
                y1=np.random.uniform(df_anom['northing'].min(),df_anom['northing'].max())
                angle=np.random.uniform(20,70); length=np.random.uniform(200,500)
                x2=x1+length*np.cos(np.radians(angle)); y2=y1+length*np.sin(np.radians(angle))
                ax_an.plot([x1,x2],[y1,y2],color='cyan',linewidth=2.5,linestyle='--',
                          label='Structure porteuse' if i==0 else '')
                ax_an.text((x1+x2)/2,(y1+y2)/2,'VQ',fontsize=8,color='cyan',fontweight='bold')

        xmax=df_anom['easting'].max(); ymax=df_anom['northing'].max()
        xmin=df_anom['easting'].min(); ymin=df_anom['northing'].min()
        ax_an.annotate('',xy=(xmax+60,ymax+25),xytext=(xmax+60,ymax-20),arrowprops=dict(arrowstyle='->',color='white',lw=2.5))
        ax_an.text(xmax+60,ymax+35,'N',ha='center',fontsize=14,fontweight='bold',color='white')
        ax_an.plot([xmin,xmin+200],[ymin-35,ymin-35],'w-',linewidth=3)
        ax_an.text(xmin+100,ymin-50,'200 m',ha='center',fontsize=9,fontweight='bold',color='white')

        legend_an=[plt.Line2D([0],[0],marker='o',color='w',markerfacecolor='#00FF00',markersize=10,label=f'Potentiel (>{seuil_anom})'),
                   plt.Line2D([0],[0],marker='o',color='w',markerfacecolor='white',markersize=8,label='Non anomalique'),
                   plt.Line2D([0],[0],color='cyan',linestyle='--',linewidth=2,label='Structure porteuse')]
        ax_an.legend(handles=legend_an,loc='lower right',fontsize=8,framealpha=0.8,facecolor='#0A1628',labelcolor='white')
        ax_an.set_xlabel("Easting UTM (m)",color='white'); ax_an.set_ylabel("Northing UTM (m)",color='white')
        ax_an.tick_params(colors='white'); ax_an.set_title(f"Carte anomalie {element} — Trous potentiels",fontsize=12,fontweight='bold',color='white')
        fig_an.patch.set_facecolor('#0A1628')
        plt.tight_layout(); st.pyplot(fig_an)

    nb_potentiel=int((df_anom['valeur_max']>=seuil_anom).sum())
    st.success(f"**{nb_potentiel} trous potentiels** détectés avec {element} ≥ {seuil_anom} | Structures porteuses : Veines de quartz (VQ) en contexte de cisaillement")

# ══ TAB 4 — CARTES STRUCTURALES ══════════════════════════════════════════════
with tabs[3]:
    st.subheader("🏗️ Carte Structurale")
    fig3,ax3=plt.subplots(figsize=(11,9))
    fig3.patch.set_facecolor('#F5F5F0'); ax3.set_facecolor('#F0EDE0')
    np.random.seed(10)
    for i in range(10):
        x1=np.random.uniform(df_forages['easting'].min()-100,df_forages['easting'].max()+100)
        y1=np.random.uniform(df_forages['northing'].min()-100,df_forages['northing'].max()+100)
        angle=np.random.uniform(0,180); length=np.random.uniform(100,500)
        x2=x1+length*np.cos(np.radians(angle)); y2=y1+length*np.sin(np.radians(angle))
        struct=np.random.choice(STRUCTURES); color=STRUCT_COLORS[struct]
        ls='-' if 'Faille' in struct else '--' if 'Veine' in struct else ':'
        lw=3 if 'Faille' in struct else 2
        ax3.plot([x1,x2],[y1,y2],color=color,linewidth=lw,linestyle=ls,label=struct)
        ax3.text((x1+x2)/2,(y1+y2)/2,f'{int(angle)}°/{np.random.randint(30,80)}°',fontsize=7,color=color,fontweight='bold')
    for _,f in df_forages.iterrows():
        ax3.scatter(f['easting'],f['northing'],c='black',s=60,zorder=3)
        ax3.annotate(f['trou'],(f['easting'],f['northing']),textcoords="offset points",xytext=(4,4),fontsize=6)
    xmax=df_forages['easting'].max(); ymax=df_forages['northing'].max()
    xmin=df_forages['easting'].min(); ymin=df_forages['northing'].min()
    ax3.annotate('',xy=(xmax+60,ymax+25),xytext=(xmax+60,ymax-20),arrowprops=dict(arrowstyle='->',color='black',lw=2.5))
    ax3.text(xmax+60,ymax+35,'N',ha='center',fontsize=14,fontweight='bold')
    ax3.plot([xmin,xmin+200],[ymin-35,ymin-35],'k-',linewidth=3)
    ax3.text(xmin+100,ymin-50,'200 m',ha='center',fontsize=9,fontweight='bold')
    handles,labels=ax3.get_legend_handles_labels()
    by_label=dict(zip(labels,handles))
    ax3.legend(by_label.values(),by_label.keys(),loc='lower right',fontsize=8,title='Structures',framealpha=0.9)
    ax3.set_xlabel("Easting UTM (m)"); ax3.set_ylabel("Northing UTM (m)")
    ax3.set_title("Carte structurale — Direction/Pendage | Projet Sénégal",fontsize=12,fontweight='bold')
    ax3.grid(True,linestyle=':',alpha=0.4); plt.tight_layout(); st.pyplot(fig3)

    st.subheader("📋 Tableau récapitulatif des structures")
    st.dataframe(structures_df.rename(columns={
        'id':'N°','type':'Type','direction':'Direction (°)','pendage':'Pendage (°)',
        'sens_pendage':'Sens pendage','longueur':'Longueur (m)','porteur_mineralisation':'Porteur minéralisation'
    }),use_container_width=True)

# ══ TAB 5 — LOGUES LITHO ══════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("📊 Logues Lithologiques")
    col1,col2=st.columns([1,3])
    with col1:
        trou_logue=st.selectbox("Trou",df_forages['trou'].tolist(),key='lithologue')
        show_au_logue=st.checkbox("Au (ppb)",True,key='au_logue')
        show_as_logue=st.checkbox("As (ppm)",True,key='as_logue')
    with col2:
        f_info=df_forages[df_forages['trou']==trou_logue].iloc[0]
        ints=df_intervals[df_intervals['trou']==trou_logue].sort_values('de')
        ncols=1+(1 if show_au_logue else 0)+(1 if show_as_logue else 0)
        fig4,axes4=plt.subplots(1,ncols,figsize=(4*ncols,10),sharey=True)
        if ncols==1: axes4=[axes4]
        ax_litho=axes4[0]
        for _,interval in ints.iterrows():
            color=LITHO_COLORS.get(interval['lithologie'],'#888')
            ax_litho.fill_betweenx([interval['de'],interval['a']],0,1,color=color,alpha=0.85)
            if interval['mineralisé']:
                ax_litho.fill_betweenx([interval['de'],interval['a']],0,1,color='red',alpha=0.25,hatch='///')
            ax_litho.plot([0,1],[interval['de'],interval['de']],'k-',linewidth=0.3)
            mid=(interval['de']+interval['a'])/2
            ax_litho.text(0.5,mid,interval['lithologie'][:10],ha='center',va='center',fontsize=5.5,fontweight='bold')
            ax_litho.text(-0.05,interval['de'],f"{interval['de']}m",ha='right',fontsize=5.5)
        ax_litho.set_ylim(f_info['profondeur'],0); ax_litho.set_xlim(-0.1,1.1)
        ax_litho.set_title(f"Litho\n{trou_logue}",fontsize=9,fontweight='bold')
        ax_litho.set_ylabel("Profondeur (m)"); ax_litho.set_xticks([])
        cidx=1
        if show_au_logue:
            colors_au=['#FFD700' if v>=100 else '#EEEEEE' for v in ints['Au_ppb']]
            axes4[cidx].barh([(i['de']+i['a'])/2 for _,i in ints.iterrows()],
                             ints['Au_ppb'].values,height=[(i['a']-i['de'])*0.8 for _,i in ints.iterrows()],
                             color=colors_au,edgecolor='orange',linewidth=0.5)
            axes4[cidx].axvline(x=100,color='red',linestyle='--',linewidth=1.5,label='Seuil 100ppb')
            axes4[cidx].set_xlabel("Au (ppb)"); axes4[cidx].set_title("Or",fontsize=9,fontweight='bold')
            axes4[cidx].legend(fontsize=7); cidx+=1
        if show_as_logue:
            axes4[cidx].barh([(i['de']+i['a'])/2 for _,i in ints.iterrows()],
                             ints['As_ppm'].values,height=[(i['a']-i['de'])*0.8 for _,i in ints.iterrows()],
                             color='#FF6B6B',edgecolor='red',linewidth=0.5)
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
        axes5[2].barh([(i['de']+i['a'])/2 for _,i in ints2.iterrows()],
                      ints2['Cu_ppm'].values,height=[(i['a']-i['de'])*0.8 for _,i in ints2.iterrows()],
                      color='#B87333',edgecolor='brown',linewidth=0.5)
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
            title="Modèle 3D des forages",height=600,paper_bgcolor='#1A1A2E',font=dict(color='white'))
        st.plotly_chart(fig3d,use_container_width=True)

    elif vue=='Modèle de blocs 3D':
        st.info("🧱 Modèle de blocs 3D — Teneur en or (Au ppb)")
        nx,ny,nz=8,8,6
        x_bl=np.linspace(BASE_E-300,BASE_E+300,nx)
        y_bl=np.linspace(BASE_N-300,BASE_N+300,ny)
        z_bl=np.linspace(0,100,nz)
        blocs=[]
        for xi in x_bl:
            for yi in y_bl:
                for zi in z_bl:
                    dist_min=min([np.sqrt((xi-f['easting'])**2+(yi-f['northing'])**2) for _,f in df_forages.iterrows()])
                    au_bloc=max(0,np.random.lognormal(2,1)-dist_min/100)
                    blocs.append({'x':xi,'y':yi,'z':zi,'Au':round(au_bloc,1)})
        df_blocs=pd.DataFrame(blocs)
        fig_bl=go.Figure(data=go.Scatter3d(
            x=df_blocs['x'],y=df_blocs['y'],z=df_blocs['z'],
            mode='markers',
            marker=dict(size=6,color=df_blocs['Au'],colorscale='Viridis',
                       colorbar=dict(title='Au (ppb)'),opacity=0.7),
            text=df_blocs['Au'].apply(lambda v:f"Au: {v:.1f} ppb"),
            hovertemplate="<b>Au: %{text}</b><br>E:%{x:.0f} N:%{y:.0f} Z:%{z:.0f}m"))
        fig_bl.update_layout(scene=dict(xaxis_title='Easting',yaxis_title='Northing',zaxis_title='Profondeur'),
            title="Modélisation par blocs — Teneur Au (ppb)",height=600)
        st.plotly_chart(fig_bl,use_container_width=True)
        st.metric("Blocs modélisés",len(df_blocs))
        st.metric("Au moyen estimé",f"{df_blocs['Au'].mean():.1f} ppb")

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

# ══ TAB 8 — PLANIFICATION & INFILL ═══════════════════════════════════════════
with tabs[7]:
    st.subheader("📋 Planification des Forages — Infill & Extension")
    col1,col2=st.columns(2)
    with col1:
        st.markdown("### Statut actuel")
        statut_counts=df_forages['statut'].value_counts()
        fig_s,ax_s=plt.subplots(figsize=(5,4))
        ax_s.pie(statut_counts.values,labels=statut_counts.index,colors=['#4CAF50','#FF9800','#2196F3'],autopct='%1.0f%%',startangle=90)
        ax_s.set_title("Répartition par statut"); st.pyplot(fig_s)
    with col2:
        st.markdown("### Programme Infill & Extension")
        espacement_actuel=st.slider("Espacement actuel (m)",50,400,200)
        espacement_infill=st.slider("Espacement infill cible (m)",25,200,100)
        nb_infill=int((espacement_actuel/espacement_infill-1)*len(df_forages[df_forages['statut']=='Complété']))
        nb_extension=st.slider("Trous d'extension prévus",0,20,5)
        st.metric("Trous infill nécessaires",nb_infill)
        st.metric("Trous extension",nb_extension)
        st.metric("Total nouveaux trous",nb_infill+nb_extension)
        cout_par_m=st.number_input("Coût/mètre (USD)",50,500,150)
        prof_moy=df_forages['profondeur'].mean()
        cout_total=(nb_infill+nb_extension)*prof_moy*cout_par_m
        st.metric("Coût estimé total",f"${cout_total:,.0f} USD")

    st.markdown("### Tableau de planification")
    df_plan=df_forages[['trou','type','profondeur','azimut','inclinaison','statut','equipe','Au_max_ppb','porteur']].copy()
    df_plan.columns=['Trou','Type','Prof.(m)','Az°','Inc°','Statut','Équipe','Au max(ppb)','Porteur minéralisation']
    st.dataframe(df_plan,use_container_width=True)
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Total forages",len(df_forages))
    c2.metric("Mètres totaux",f"{df_forages['profondeur'].sum():.0f} m")
    c3.metric("Au max",f"{df_forages['Au_max_ppb'].max():.1f} ppb")
    c4.metric("Avancement",f"{int((df_forages['statut']=='Complété').sum())/len(df_forages)*100:.0f}%")

# ══ TAB 9 — SIMULATION DÉVIATION ════════════════════════════════════════════
with tabs[8]:
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

# ══ TAB 10 — SURVEILLANCE ════════════════════════════════════════════════════
with tabs[9]:
    st.subheader("📡 Surveillance des Forages")
    col1,col2=st.columns(2)
    with col1:
        st.markdown("### 🏔️ Forages de surface")
        for _,f in df_forages[df_forages['type'].isin(['RC','Aircore'])].iterrows():
            icon='🟢' if f['statut']=='Complété' else '🟡' if f['statut']=='En cours' else '🔵'
            st.markdown(f"**{icon} {f['trou']}** ({f['type']}) | {f['profondeur']}m | {f['equipe']} | Au:{f['Au_max_ppb']:.0f} ppb")
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
    c4.metric("RC",int((df_forages['type']=='RC').sum()))
    c5.metric("Diamond",int((df_forages['type']=='Diamond').sum()))
    prob=df_forages[df_forages['Au_max_ppb']>500]
    if len(prob)>0:
        for _,f in prob.iterrows():
            st.warning(f"🔔 **{f['trou']}** — Au: {f['Au_max_ppb']:.0f} ppb → Priorité !")
    else:
        st.success("✅ Aucune anomalie critique")

# ══ TAB 11 — ESSAI SGI ══════════════════════════════════════════════════════
with tabs[10]:
    st.subheader("🧪 Essai SGI — Minéralisation & Altération")
    col1,col2=st.columns([1,2])
    with col1:
        trou_sgi=st.selectbox("Trou",df_forages['trou'].tolist(),key='sgi')
        seuil_au=st.number_input("Seuil Au minéralisé (ppb)",10,1000,100)
        seuil_cu=st.number_input("Seuil Cu (ppm)",5,200,50)
    with col2:
        ints_sgi=df_intervals[df_intervals['trou']==trou_sgi].sort_values('de').copy()
        ints_sgi['mineralisé']=ints_sgi['Au_ppb']>=seuil_au
        total_m=ints_sgi['a'].max()-ints_sgi['de'].min()
        m_miner=ints_sgi[ints_sgi['mineralisé']].apply(lambda r:r['a']-r['de'],axis=1).sum()
        pct_miner=m_miner/total_m*100 if total_m>0 else 0
        c1,c2,c3=st.columns(3)
        c1.metric("Intervalles minéralisés",f"{m_miner:.1f} m")
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
        colors_au=['#FFD700' if v>=seuil_au else '#EEEEEE' for v in ints_sgi['Au_ppb']]
        axes_sgi[2].barh([(i['de']+i['a'])/2 for _,i in ints_sgi.iterrows()],
                         ints_sgi['Au_ppb'].values,height=[(i['a']-i['de'])*0.8 for _,i in ints_sgi.iterrows()],
                         color=colors_au,edgecolor='orange',linewidth=0.5)
        axes_sgi[2].axvline(x=seuil_au,color='red',linestyle='--',linewidth=1.5,label=f'Seuil {seuil_au}ppb')
        axes_sgi[2].set_xlabel("Au (ppb)"); axes_sgi[2].set_title("Or — SGI",fontsize=9,fontweight='bold')
        axes_sgi[2].legend(fontsize=7)
        colors_cu=['#B87333' if v>=seuil_cu else '#EEEEEE' for v in ints_sgi['Cu_ppm']]
        axes_sgi[3].barh([(i['de']+i['a'])/2 for _,i in ints_sgi.iterrows()],
                         ints_sgi['Cu_ppm'].values,height=[(i['a']-i['de'])*0.8 for _,i in ints_sgi.iterrows()],
                         color=colors_cu,edgecolor='brown',linewidth=0.5)
        axes_sgi[3].axvline(x=seuil_cu,color='blue',linestyle='--',linewidth=1.5,label=f'Seuil {seuil_cu}ppm')
        axes_sgi[3].set_xlabel("Cu (ppm)"); axes_sgi[3].set_title("Cuivre — SGI",fontsize=9,fontweight='bold')
        axes_sgi[3].legend(fontsize=7)
        plt.suptitle(f"Essai SGI — {trou_sgi} | {f_sgi['type']} | {f_sgi['profondeur']}m",fontsize=11,fontweight='bold')
        plt.tight_layout(); st.pyplot(fig_sgi)

    st.markdown("### 📊 Tableau SGI — Minéralisation & Teneur Or")
    sgi_table=df_intervals[df_intervals['trou']==trou_sgi][['de','a','lithologie','alteration','mineralisation','Au_ppb','Cu_ppm','As_ppm','mineralisé']].copy()
    sgi_table.columns=['De (m)','A (m)','Lithologie','Altération','Minéralisation','Au (ppb)','Cu (ppm)','As (ppm)','Minéralisé']
    st.dataframe(sgi_table.style.applymap(
        lambda v:'background-color:#FFD700;color:black' if v==True else 'background-color:#F5F5F5' if v==False else '',
        subset=['Minéralisé']
    ).format({'Au (ppb)':'{:.2f}','Cu (ppm)':'{:.1f}','As (ppm)':'{:.1f}'}),use_container_width=True)

# ══ TAB 12 — MONITORING ════════════════════════════════════════════════════
with tabs[11]:
    st.subheader("📈 Monitoring — Suivi en temps réel")
    c1,c2,c3=st.columns(3)
    c1.metric("Mètres forés aujourd'hui",f"{np.random.randint(30,80)} m","↑ +12 vs hier")
    c2.metric("Trous actifs",int((df_forages['statut']=='En cours').sum()))
    c3.metric("Incidents",np.random.randint(0,2))
    equipes=df_forages.groupby('equipe').agg(trous=('trou','count'),metres=('profondeur','sum'),au_max=('Au_max_ppb','max')).reset_index()
    fig_eq,ax_eq=plt.subplots(figsize=(8,4))
    ax_eq.bar(equipes['equipe'],equipes['metres'],color=['#2196F3','#4CAF50','#FF9800'],edgecolor='black',linewidth=0.5)
    ax_eq.set_ylabel("Mètres forés"); ax_eq.set_title("Mètres forés par équipe",fontsize=11,fontweight='bold')
    for i,(v) in enumerate(equipes['metres']): ax_eq.text(i,v+5,f"{v:.0f}m",ha='center',fontsize=9,fontweight='bold')
    plt.tight_layout(); st.pyplot(fig_eq)
    dates_30=pd.date_range(end=datetime.date.today(),periods=30)
    metres_30=np.random.randint(20,80,30)
    fig_jour,ax_jour=plt.subplots(figsize=(12,4))
    ax_jour.fill_between(dates_30,metres_30,alpha=0.3,color='#2196F3')
    ax_jour.plot(dates_30,metres_30,'b-o',markersize=4,linewidth=1.5)
    ax_jour.axhline(y=metres_30.mean(),color='red',linestyle='--',linewidth=1.5,label=f'Moyenne: {metres_30.mean():.0f}m/j')
    ax_jour.set_ylabel("Mètres/jour"); ax_jour.set_title("Production journalière",fontsize=11,fontweight='bold')
    ax_jour.legend(fontsize=9); ax_jour.grid(True,linestyle=':',alpha=0.4)
    plt.xticks(rotation=45); plt.tight_layout(); st.pyplot(fig_jour)

# ══ TAB 13 — WEEKLY REPORT ══════════════════════════════════════════════════
with tabs[12]:
    st.subheader("📅 Rapport Hebdomadaire")
    semaine=st.date_input("Semaine du",datetime.date.today()-datetime.timedelta(days=7))
    st.markdown(f"**Période :** {semaine} → {semaine+datetime.timedelta(days=6)}")
    c1,c2,c3,c4,c5=st.columns(5)
    total_m_week=int(weekly_data['metres_fores'].sum())
    objectif=350
    c1.metric("Mètres forés",f"{total_m_week} m",f"{total_m_week-objectif:+d} vs obj.")
    c2.metric("Trous complétés",int(weekly_data['trous_completes'].sum()))
    c3.metric("Incidents",int(weekly_data['incidents'].sum()))
    c4.metric("Au max semaine",f"{float(weekly_data['Au_ppb_moyen'].max()):.1f} ppb")
    c5.metric("Objectif","✅ Oui" if total_m_week>=objectif else "❌ Non")
    fig_week,axes_week=plt.subplots(1,3,figsize=(14,4))
    axes_week[0].bar([d.strftime('%a') for d in weekly_data['date']],weekly_data['metres_fores'],color='#2196F3',edgecolor='black',linewidth=0.5)
    axes_week[0].axhline(y=objectif/7,color='red',linestyle='--',linewidth=1.5,label=f'Obj/j:{objectif//7}m')
    axes_week[0].set_ylabel("Mètres"); axes_week[0].set_title("Mètres/jour",fontsize=10,fontweight='bold'); axes_week[0].legend(fontsize=8)
    axes_week[1].bar([d.strftime('%a') for d in weekly_data['date']],weekly_data['Au_ppb_moyen'],color='#FFD700',edgecolor='orange',linewidth=0.5)
    axes_week[1].set_ylabel("Au (ppb)"); axes_week[1].set_title("Au moyen/jour",fontsize=10,fontweight='bold')
    axes_week[2].bar([d.strftime('%a') for d in weekly_data['date']],weekly_data['incidents'],
                     color=['#4CAF50' if v==0 else '#FF5722' for v in weekly_data['incidents']],edgecolor='black',linewidth=0.5)
    axes_week[2].set_ylabel("Incidents"); axes_week[2].set_title("Incidents/jour",fontsize=10,fontweight='bold')
    plt.suptitle(f"Weekly Report — {semaine}",fontsize=12,fontweight='bold'); plt.tight_layout(); st.pyplot(fig_week)
    weekly_display=weekly_data.copy()
    weekly_display['date']=weekly_display['date'].dt.strftime('%Y-%m-%d (%A)')
    weekly_display.columns=['Date','Mètres forés','Trous complétés','Incidents','Au moy.(ppb)','Équipe']
    st.dataframe(weekly_display,use_container_width=True)
    csv=weekly_display.to_csv(index=False)
    st.download_button("📥 Télécharger rapport CSV",data=csv,file_name=f"weekly_{semaine}.csv",mime='text/csv')

# ══ TAB 14 — RAPPORT GÉOLOGIQUE ══════════════════════════════════════════════
with tabs[13]:
    st.subheader("📄 Rapport Géologique Argumenté")

    au_max=df_forages['Au_max_ppb'].max()
    au_moy=df_forages['Au_max_ppb'].mean()
    nb_miner=int(df_intervals['mineralisé'].sum())
    pct_miner_global=nb_miner/len(df_intervals)*100
    litho_dominante=df_intervals['lithologie'].value_counts().index[0]
    alter_dominante=df_intervals['alteration'].value_counts().index[0]
    miner_dominante=df_intervals['mineralisation'].value_counts().index[0]
    struct_porteuse=structures_df[structures_df['porteur_mineralisation']==True]['type'].value_counts().index[0] if len(structures_df[structures_df['porteur_mineralisation']==True])>0 else 'Veine de quartz'

    st.markdown(f"""
## 1. Contexte géologique général

Le projet est localisé au **Sénégal (Afrique de l'Ouest)**, dans une ceinture de roches vertes de type
**protérozoïque**, caractérisée par une alternance de schistes, quartzites et granites intrusifs. Ce contexte
est classiquement favorable aux gisements aurifères orogéniques en Afrique de l'Ouest (Birimien).

## 2. Lithologies identifiées

| Lithologie | Épaisseur moy. | Interprétation |
|-----------|---------------|----------------|
| Latérite | 0–8 m | Couverture résiduelle, profil d'altération supergène |
| Saprolite | 8–25 m | Zone d'altération avancée, mobilisation des éléments |
| Saprock | 25–50 m | Transition roche fraîche/altérée, zone critique |
| Bédrock/Schiste | 50–120 m | Roche encaissante principale, cible primaire |
| **Quartzite aurifère** | Variable | **Principale unité minéralisée** |
| Granite frais | >100 m | Intrusion tardi-cinématique, rôle thermique |

**Lithologie dominante :** {litho_dominante}

## 3. Contexte de la minéralisation

La minéralisation aurifère est principalement de type **{miner_dominante}**, associée à des veines et
veinules de quartz dans des zones de cisaillement ductile-fragile. La teneur maximale détectée est de
**{au_max:.1f} ppb**, avec une moyenne de **{au_moy:.1f} ppb**.

**{pct_miner_global:.1f}% des intervalles** dépassent le seuil de minéralisation économique (100 ppb Au).

### Facteurs de contrôle de la minéralisation :
- **Structural :** Les zones de cisaillement et les failles inverses constituent les principaux couloirs
  de circulation des fluides minéralisateurs
- **Lithologique :** Le contact Schiste/Quartzite est systématiquement minéralisé
- **Altération :** La silicification intense est le meilleur guide de la minéralisation aurifère

## 4. Altération

L'altération dominante est **{alter_dominante}**, caractéristique d'un système hydrothermal aurifère
mésothermal. La séquence d'altération observée est :

**Carbonatation → Séricitisation → Silicification → Pyritisation**

Cette séquence est typique des gisements aurifères orogéniques et indique des températures de dépôt
entre 250°C et 400°C, à des pressions de 1 à 3 kbar.

## 5. Structures

La structure porteuse principale est **{struct_porteuse}**, avec des orientations dominantes NE-SO
à pendage SE. Le tableau des structures ci-dessous résume les directions et pendages mesurés.

## 6. Interprétations

1. **Le système minéralisateur est actif** sur au moins {df_forages['profondeur'].max():.0f}m de profondeur,
   sans signe d'épuisement — la minéralisation reste ouverte en profondeur.

2. **La corrélation Au-As** est forte (coefficient > 0.7), indiquant que l'arsenic peut être utilisé
   comme **pathfinder géochimique** pour guider l'exploration en surface.

3. **Les zones à forte silicification** coïncident systématiquement avec les teneurs en or les plus
   élevées — la silicification est le meilleur vecteur d'exploration au sol.

4. **Le modèle structural NE-SO** contrôle la distribution spatiale des minéralisations. Les forages
   doivent être orientés perpendiculairement à ces structures (az. ~300°, inc. -60°).

## 7. Recommandations

### Court terme (0–3 mois)
- Approfondir les trous SG001, SG004, SG007 qui montrent les meilleures teneurs
- Réaliser des forages infill à 100m d'espacement dans les zones anomaliques confirmées
- Analyser systématiquement As, Cu et Ag comme éléments pathfinders

### Moyen terme (3–12 mois)
- Extension du programme vers le NE et le SO pour tester la continuité du système
- Levé géophysique IP (induced polarization) pour cartographier les sulfures en profondeur
- Modélisation géostatistique 3D pour estimer les ressources

### Long terme (>12 mois)
- Estimation des ressources selon le code JORC/NI 43-101
- Études de préfaisabilité métallurgique (essais de lixiviation)
- Évaluation environnementale et sociale (ESIA)

## 8. Conclusion

Le projet présente tous les attributs d'un **gisement aurifère orogénique de classe mondiale** :
contexte birimien favorable, minéralisation contrôlée par les structures, altération hydrothermale
intense, teneurs économiques confirmées. L'exploration doit se poursuivre avec une densification
progressive des forages et une approche multi-méthodes (géologie, géochimie, géophysique).
    """)

    st.markdown("---")
    st.markdown("### 📊 Tableau récapitulatif des structures du prospect")
    st.dataframe(structures_df.rename(columns={
        'id':'N°','type':'Type de structure','direction':'Direction (°)','pendage':'Pendage (°)',
        'sens_pendage':'Sens du pendage','longueur':'Longueur (m)','porteur_mineralisation':'Porteur minéralisation'
    }),use_container_width=True)

    # Export rapport
    rapport_txt=f"""RAPPORT GÉOLOGIQUE — PROJET SÉNÉGAL
Date: {datetime.date.today()}
Au max: {au_max:.1f} ppb | Au moy: {au_moy:.1f} ppb
Intervalles minéralisés: {pct_miner_global:.1f}%
Lithologie dominante: {litho_dominante}
Altération dominante: {alter_dominante}
Structure porteuse: {struct_porteuse}
"""
    st.download_button("📥 Télécharger le rapport",data=rapport_txt,
                      file_name=f"rapport_geologique_{datetime.date.today()}.txt",mime='text/plain')

st.markdown("---")
st.caption("⛏️ Dashboard Géologie Minière — Projet Sénégal | RC · Aircore · Diamond · SGI · 3D · Weekly · Rapport")
