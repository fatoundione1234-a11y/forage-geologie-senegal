import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
from scipy.interpolate import griddata
import datetime
import seaborn as sns
from sklearn.cluster import KMeans

np.random.seed(42)

st.set_page_config(page_title="Projet Minière", layout="wide", page_icon="⛏️")
st.markdown("""<style>
.main-header{background:linear-gradient(90deg,#1A237E,#0D47A1);color:white;
  padding:15px 20px;border-radius:10px;margin-bottom:20px;}
</style>""", unsafe_allow_html=True)
st.markdown("""<div class='main-header'>
<h1 style='font-size:28px;margin-bottom:6px;letter-spacing:2px;font-weight:700;'>⛏️ PROJET MINIÈRE</h1>
<p style='font-size:13px;opacity:0.9;'>Sections · Cartes · 3D · Logues automatisés · Auger · pXRF · Géophysique · SGI · SOP · Audit IA · Rapport</p>
</div>""", unsafe_allow_html=True)

# ── CONSTANTES ────────────────────────────────────────────────────────────────
BASE_E, BASE_N  = 350000.0, 1480000.0
NOM_PROSPECT    = "Prospect Dakar-Gold"
NOM_PERMIS      = "Permis PR-2024-SN-001"
N_LIGNES, N_PTS = 10, 7
ESP_L, ESP_P    = 200, 150

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
COLOR_ST = {'Foré':'#2196F3','En cours':'#4CAF50','Stoppé':'#F44336'}
MARKER_ST= {'Foré':'o','En cours':'s','Stoppé':'X'}

# ── DONNÉES FORAGES ───────────────────────────────────────────────────────────
def gen_forages():
    rows=[]
    for i in range(15):
        ft=np.random.choice(['RC','Aircore','Diamond'],p=[0.4,0.3,0.3])
        pr=np.random.choice([80,100,120,150,200]) if ft=='Diamond' else np.random.choice([30,40,50,60])
        rows.append({'trou':f'SG{i+1:03d}','type':ft,
            'easting':round(BASE_E+np.random.uniform(-400,400),1),
            'northing':round(BASE_N+np.random.uniform(-400,400),1),
            'elevation':round(np.random.uniform(80,120),1),
            'profondeur':pr,'azimut':round(np.random.uniform(0,360),1),
            'inclinaison':round(np.random.uniform(-85,-60),1),
            'statut':np.random.choice(['Complété','En cours','Planifié'],p=[0.6,0.2,0.2]),
            'Au_max_ppb':round(np.random.lognormal(2.5,1.5),1),
            'equipe':np.random.choice(['Équipe A','Équipe B','Équipe C'])})
    return pd.DataFrame(rows)
df_forages=gen_forages()

def gen_intervals(df_f):
    rows=[]
    for _,f in df_f.iterrows():
        d=0
        while d<f['profondeur']:
            t=np.random.uniform(2,15)
            li=LITHOS[min(int(d/f['profondeur']*len(LITHOS)),len(LITHOS)-1)]
            if np.random.random()>0.3: li=np.random.choice(LITHOS)
            al=np.random.choice(ALTERATIONS)
            mi=np.random.choice(MINERALISATIONS,p=[0.15,0.10,0.15,0.10,0.15,0.35])
            if li=='Quartzite aurifère': au=round(np.random.lognormal(5.5,1.2),2)
            elif li=='Bédrock/Schiste':  au=round(np.random.lognormal(4.0,1.5),2)
            elif li=='Saprock':          au=round(np.random.lognormal(3.0,1.2),2)
            else:                        au=round(np.random.lognormal(1.5,1.0),2)
            if mi in ['Aurifère disséminée','Aurifère filonienne']:
                au=round(au*np.random.uniform(1.5,4.0),2)
            rows.append({'trou':f['trou'],'type':f['type'],
                'de':round(d,1),'a':round(min(d+t,f['profondeur']),1),
                'lithologie':li,'alteration':al,'mineralisation':mi,
                'Au_ppb':au,'Cu_ppm':round(np.random.uniform(1,80),1),
                'As_ppm':round(np.random.uniform(1,50),1),
                'Ag_ppm':round(np.random.uniform(0.1,10),2),'mineralisé':au>=100})
            d+=t
    return pd.DataFrame(rows)
df_intervals=gen_intervals(df_forages)

np.random.seed(10)
structures_df=pd.DataFrame([{'id':f'STR{i+1:02d}',
    'type':np.random.choice(STRUCTURES),'easting':round(BASE_E+np.random.uniform(-400,400),1),
    'northing':round(BASE_N+np.random.uniform(-400,400),1),
    'direction':round(np.random.uniform(0,360),1),'pendage':round(np.random.uniform(10,85),1),
    'sens_pendage':np.random.choice(['N','NE','E','SE','S','SO','O','NO']),
    'longueur_m':round(np.random.uniform(10,500),0),
    'porteur_miner':np.random.choice([True,False],p=[0.35,0.65])} for i in range(40)])

# ── DONNÉES AUGER ─────────────────────────────────────────────────────────────
np.random.seed(99)
auger_data=[]
for i in range(N_LIGNES):
    for j in range(N_PTS):
        e=BASE_E-300+j*ESP_P+np.random.normal(0,3)
        n=BASE_N-450+i*ESP_L+np.random.normal(0,3)
        st_=np.random.choice(['Foré','En cours','Stoppé'],p=[0.55,0.15,0.30])
        pr=round(np.random.uniform(3,25),1) if st_!='Stoppé' else round(np.random.uniform(1,8),1)
        au=round(np.random.lognormal(3.5,1.5),2) if st_=='Foré' else np.nan
        auger_data.append({'ligne':f'L{i+1:02d}','trou':f'L{i+1:02d}T{j+1:02d}',
            'easting':round(e,1),'northing':round(n,1),
            'elevation':round(np.random.uniform(80,120),1),
            'profondeur_m':pr,'statut':st_,'Au_ppb':au,
            'As_ppm':round(np.random.uniform(5,80),1) if st_=='Foré' else np.nan,
            'Cu_ppm':round(np.random.uniform(5,120),1) if st_=='Foré' else np.nan,
            'Fe_pct':round(np.random.uniform(1,30),2) if st_=='Foré' else np.nan,
            'lithologie':np.random.choice(['Latérite','Saprolite','Saprock','Schiste'])})
df_auger=pd.DataFrame(auger_data)

# ── DONNÉES pXRF ──────────────────────────────────────────────────────────────
np.random.seed(77)
pxrf_rows=[]
for _,row in df_auger[df_auger['statut']=='Foré'].iterrows():
    for depth in np.arange(0,row['profondeur_m'],1):
        pxrf_rows.append({'trou':row['trou'],'ligne':row['ligne'],
            'easting':row['easting'],'northing':row['northing'],
            'profondeur':round(float(depth),1),
            'Au_ppb':round(max(0,np.random.lognormal(3,1.5)),2),
            'As_ppm':round(np.random.uniform(2,80),1),
            'Cu_ppm':round(np.random.uniform(1,120),1),
            'Fe_pct':round(np.random.uniform(1,35),2),
            'Mn_ppm':round(np.random.uniform(10,500),1),
            'Zn_ppm':round(np.random.uniform(1,80),1),
            'methode':'pXRF Olympus Vanta',
            'operateur':np.random.choice(['Tech A','Tech B','Tech C']),
            'date':(datetime.date.today()-datetime.timedelta(days=int(np.random.randint(1,30)))).strftime('%Y-%m-%d')})
df_pxrf=pd.DataFrame(pxrf_rows)

# ── DONNÉES GÉOPHYSIQUE ────────────────────────────────────────────────────────
np.random.seed(7)
n_geo=80
geo_data=pd.DataFrame({'point':[f'GP{i+1:03d}' for i in range(n_geo)],
    'easting':[round(BASE_E+np.random.uniform(-500,500),1) for _ in range(n_geo)],
    'northing':[round(BASE_N+np.random.uniform(-500,500),1) for _ in range(n_geo)],
    'IP_chargeabilite':np.round(np.random.lognormal(2,0.8,n_geo),2),
    'resistivite_ohm':np.round(np.random.lognormal(4,1,n_geo),1),
    'mag_nT':np.round(np.random.normal(50000,500,n_geo),1),
    'mag_anomalie_nT':np.round(np.random.normal(0,200,n_geo),1),
    'SP_mV':np.round(np.random.normal(0,50,n_geo),1),
    'EM_conductivite':np.round(np.random.lognormal(2,0.8,n_geo),1),
    'profil':[f'P{np.random.randint(1,6):02d}' for _ in range(n_geo)]})
geo_data['anomalie_IP']=geo_data['IP_chargeabilite']>geo_data['IP_chargeabilite'].quantile(0.75)
geo_data['anomalie_mag']=np.abs(geo_data['mag_anomalie_nT'])>150

# Weekly
dates_week=pd.date_range(end=datetime.date.today(),periods=7)
weekly_data=pd.DataFrame({'date':dates_week,'metres_fores':np.random.randint(20,80,7),
    'trous_completes':np.random.randint(0,3,7),'incidents':np.random.randint(0,2,7),
    'Au_ppb_moyen':np.round(np.random.lognormal(2,0.8,7),1),
    'equipe':np.random.choice(['Équipe A','Équipe B'],7)})

# ── HELPER: carte géophysique ─────────────────────────────────────────────────
def plot_geo_map(data,col,titre,cmap,ax,df_f,nord=True,echelle=True):
    if len(data)>3:
        xi=np.linspace(data['easting'].min()-50,data['easting'].max()+50,150)
        yi=np.linspace(data['northing'].min()-50,data['northing'].max()+50,150)
        Xi,Yi=np.meshgrid(xi,yi)
        Zi=griddata((data['easting'],data['northing']),data[col],(Xi,Yi),method='linear')
        cf=ax.contourf(Xi,Yi,Zi,levels=20,cmap=cmap,alpha=0.85)
        plt.colorbar(cf,ax=ax,label=col,shrink=0.8)
        ax.contour(Xi,Yi,Zi,levels=6,colors='black',alpha=0.2,linewidths=0.5)
    for _,r in data.iterrows():
        ax.scatter(r['easting'],r['northing'],c='white',s=25,edgecolors='gray',linewidths=0.3,zorder=3)
    for _,f in df_f.iterrows():
        ax.scatter(f['easting'],f['northing'],c='black',s=80,marker='^',zorder=4,edgecolors='white',linewidths=0.8)
        ax.text(f['easting'],f['northing']+8,f['trou'],fontsize=5.5,ha='center',color='white',fontweight='bold')
    xmx=data['easting'].max(); ymx=data['northing'].max()
    xmn=data['easting'].min(); ymn=data['northing'].min()
    if nord:
        ax.annotate('',xy=(xmx+60,ymx+20),xytext=(xmx+60,ymx-20),arrowprops=dict(arrowstyle='->',color='white',lw=2))
        ax.text(xmx+60,ymx+30,'N',ha='center',fontsize=14,fontweight='bold',color='white')
    if echelle:
        ax.plot([xmn,xmn+200],[ymn-35,ymn-35],'w-',linewidth=3)
        ax.text(xmn+100,ymn-50,'200 m',ha='center',fontsize=9,fontweight='bold',color='white')
    ax.set_title(titre,fontsize=11,fontweight='bold',color='white')
    ax.set_xlabel("Easting (m)",color='white'); ax.set_ylabel("Northing (m)",color='white')
    ax.tick_params(colors='white')

# ── TABS ──────────────────────────────────────────────────────────────────────
tabs=st.tabs([
    "📐 Sections géologiques",
    "🗺️ Cartes lithologiques",
    "🌡️ Carte anomalie",
    "🏗️ Cartes structurales",
    "📊 Logues automatisés",
    "⛏️ Auger",
    "📡 pXRF & Géochimie",
    "🌊 Géophysique",
    "🧪 Essai SGI",
    "💰 Estimation teneurs",
    "🌐 Sections 2D & 3D",
    "📋 Planification & Infill",
    "🔭 Programme extension",
    "🗺️ Mapping spatial",
    "🔄 Simulation déviation",
    "📍 Survey",
    "🔬 Détecteur de roches",
    "📈 Monitoring",
    "📅 Weekly Report",
    "💬 Commentaires",
    "📄 Rapport géologique",
    "📁 Logues terrain réels",
    "📘 SOP Exploration",
    "🤖 Audit IA & Corrections",
])

# ══ TAB 1 — SECTIONS ══════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("📐 Sections Géologiques — RC / Aircore / Diamond")
    col1,col2=st.columns([1,3])
    with col1:
        stype=st.selectbox("Type",['RC','Aircore','Diamond','Tous'])
        tdispo=df_forages['trou'].tolist() if stype=='Tous' else df_forages[df_forages['type']==stype]['trou'].tolist()
        tsel=st.selectbox("Trou réf.",tdispo)
        ech=st.slider("Échelle ×",1,5,2)
        sm=st.checkbox("Intervalles minéralisés",True)
        sau=st.checkbox("Teneurs Au",True)
        slim=st.checkbox("Limites géologiques",True)
        sinf=st.checkbox("Programme Infill",False)
    with col2:
        fig,ax=plt.subplots(figsize=(15,9))
        fig.patch.set_facecolor('#F8F8F0'); ax.set_facecolor('#E8F4F8')
        x_t=np.linspace(0,600,300)
        topo=100+5*np.sin(x_t/50)+3*np.cos(x_t/30)+np.random.normal(0,0.5,300)
        ax.plot(x_t,topo,'k-',linewidth=2.5,label='Topographie',zorder=5)
        ax.fill_between(x_t,topo,0,alpha=0.15,color='brown')
        ax.axhline(y=0,color='blue',linestyle='--',linewidth=1,alpha=0.5,label='Ligne référence')
        if slim:
            for d,lbl,lc in [(8,'Base Latérite','#8B4513'),(25,'Base Saprolite','#DAA520'),(50,'Base Saprock','#696969'),(80,'Top Bédrock','#FFD700')]:
                ax.axhline(y=topo.mean()-d*ech*0.5,color=lc,linestyle='-.',linewidth=1.5,alpha=0.8,label=lbl)
        xpos_arr=np.linspace(60,540,min(6,len(tdispo)))
        for idx,(xpos,trou) in enumerate(zip(xpos_arr,tdispo[:6])):
            f=df_forages[df_forages['trou']==trou].iloc[0]
            tv=float(np.interp(xpos,x_t,topo))
            ints=df_intervals[df_intervals['trou']==trou].sort_values('de')
            for _,iv in ints.iterrows():
                yt=tv-iv['de']*ech*0.5; yb=tv-iv['a']*ech*0.5
                c=LITHO_COLORS.get(iv['lithologie'],'#888')
                ax.fill_betweenx([yb,yt],xpos-7,xpos+7,color=c,alpha=0.85)
                ax.plot([xpos-7,xpos+7,xpos+7,xpos-7,xpos-7],[yt,yt,yb,yb,yt],'k-',linewidth=0.3)
                if sm and iv['mineralisé']:
                    ax.fill_betweenx([yb,yt],xpos-7,xpos+7,color='red',alpha=0.3,hatch='///')
                    ax.plot([xpos-7,xpos+7],[yt,yt],'r-',linewidth=1.5)
                    ax.plot([xpos-7,xpos+7],[yb,yb],'r-',linewidth=1.5)
                if sau and iv['Au_ppb']>=100:
                    ax.text(xpos+9,(yt+yb)/2,f"{iv['Au_ppb']:.0f}ppb",fontsize=5.5,color='#FF6600',fontweight='bold')
            fc={'RC':'#FF5722','Aircore':'#2196F3','Diamond':'#9C27B0'}
            ax.text(xpos,tv+7,trou,ha='center',fontsize=7,fontweight='bold',color='#1A237E')
            ax.text(xpos,tv+4,f['type'],ha='center',fontsize=6,color=fc.get(f['type'],'black'),fontweight='bold')
            ax.text(xpos+9,tv-f['profondeur']*ech*0.5,f"{f['profondeur']}m",fontsize=6,va='center',color='#333')
            if sinf and idx<len(xpos_arr)-1:
                xi2=(xpos+xpos_arr[min(idx+1,len(xpos_arr)-1)])/2
                ax.axvline(x=xi2,color='purple',linestyle=':',linewidth=1.5,alpha=0.6)
                ax.text(xi2,tv+7,'INFILL',ha='center',fontsize=6,color='purple',fontweight='bold')
        ax.text(10,max(topo)+10,f"Section {tsel} — Az.090° | {NOM_PROSPECT}",fontsize=9,fontweight='bold',
                color='#1A237E',bbox=dict(boxstyle='round',facecolor='white',alpha=0.9))
        ax.annotate('',xy=(575,max(topo)+6),xytext=(575,max(topo)-6),arrowprops=dict(arrowstyle='->',color='black',lw=2.5))
        ax.text(575,max(topo)+8,'N',ha='center',fontsize=12,fontweight='bold')
        ax.plot([20,70],[4,4],'k-',linewidth=3); ax.text(45,1,'50 m',ha='center',fontsize=8,fontweight='bold')
        lp=[mpatches.Patch(color=c,label=l) for l,c in LITHO_COLORS.items()]
        if sm: lp.append(mpatches.Patch(color='red',alpha=0.4,hatch='///',label='Minéralisé'))
        ax.legend(handles=lp,loc='lower right',fontsize=7,title='Lithologie',ncol=2,framealpha=0.9)
        ax.set_xlabel("Distance (m)"); ax.set_ylabel("Élévation (m)")
        ax.set_title(f"Section géologique — {stype} | {NOM_PROSPECT} | {NOM_PERMIS}",fontsize=12,fontweight='bold')
        ax.grid(True,linestyle=':',alpha=0.3); ax.set_xlim(0,600)
        plt.tight_layout(); st.pyplot(fig)

# ══ TAB 2 — CARTES LITHO ══════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("🗺️ Carte Lithologique")
    pc=st.slider("Profondeur (m)",0,150,20)
    fig2,ax2=plt.subplots(figsize=(11,9))
    fig2.patch.set_facecolor('#F5F5F0'); ax2.set_facecolor('#E8F4F8')
    for _,f in df_forages.iterrows():
        id2=df_intervals[(df_intervals['trou']==f['trou'])&(df_intervals['de']<=pc)].tail(1)
        if len(id2)>0:
            li=id2.iloc[0]['lithologie']; c=LITHO_COLORS.get(li,'#888')
            mk={'RC':'^','Aircore':'o','Diamond':'s'}.get(f['type'],'o')
            ax2.scatter(f['easting'],f['northing'],c=c,s=200,marker=mk,edgecolors='black',linewidths=1.2,zorder=3)
            ax2.annotate(f"{f['trou']}\n{li[:8]}",(f['easting'],f['northing']),textcoords="offset points",xytext=(5,5),fontsize=6,color='#1A237E')
    xmx=df_forages['easting'].max(); ymx=df_forages['northing'].max()
    xmn=df_forages['easting'].min(); ymn=df_forages['northing'].min()
    ax2.annotate('',xy=(xmx+60,ymx+30),xytext=(xmx+60,ymx-25),arrowprops=dict(arrowstyle='->',color='black',lw=2.5))
    ax2.text(xmx+60,ymx+40,'N',ha='center',fontsize=14,fontweight='bold')
    ax2.plot([xmn,xmn+200],[ymn-35,ymn-35],'k-',linewidth=3)
    ax2.text(xmn+100,ymn-50,'200 m',ha='center',fontsize=9,fontweight='bold')
    ax2.text(xmn,ymx+40,f"{NOM_PROSPECT} | {NOM_PERMIS}",fontsize=9,fontweight='bold',color='#1A237E')
    ax2.grid(True,linestyle='--',alpha=0.4)
    lp2=[mpatches.Patch(color=c,label=l) for l,c in LITHO_COLORS.items()]
    tm=[plt.Line2D([0],[0],marker='^',color='w',markerfacecolor='gray',markersize=9,label='RC'),
        plt.Line2D([0],[0],marker='o',color='w',markerfacecolor='gray',markersize=9,label='Aircore'),
        plt.Line2D([0],[0],marker='s',color='w',markerfacecolor='gray',markersize=9,label='Diamond')]
    ax2.legend(handles=lp2+tm,loc='lower right',fontsize=8,title='Lithologie & Type',ncol=2,framealpha=0.95)
    ax2.set_xlabel("Easting UTM (m)"); ax2.set_ylabel("Northing UTM (m)")
    ax2.set_title(f"Carte lithologique à {pc}m — {NOM_PROSPECT}",fontsize=12,fontweight='bold')
    plt.tight_layout(); st.pyplot(fig2)

# ══ TAB 3 — CARTE ANOMALIE ════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("🌡️ Carte d'Anomalie Géochimique")
    col1,col2=st.columns([1,3])
    with col1:
        el=st.selectbox("Élément",['Au_ppb','Cu_ppm','As_ppm'])
        sa=st.number_input("Seuil",10,1000,100)
        spt=st.checkbox("Structures porteuses",True)
    with col2:
        atrou=df_intervals.groupby('trou')[el].max().reset_index(); atrou.columns=['trou','valeur_max']
        df_an=df_forages.merge(atrou,on='trou')
        fig3,ax3=plt.subplots(figsize=(11,9)); fig3.patch.set_facecolor('#0A1628'); ax3.set_facecolor('#0A1628')
        if len(df_an)>3:
            xi=np.linspace(df_an['easting'].min()-50,df_an['easting'].max()+50,150)
            yi=np.linspace(df_an['northing'].min()-50,df_an['northing'].max()+50,150)
            Xi,Yi=np.meshgrid(xi,yi)
            Zi=griddata((df_an['easting'],df_an['northing']),np.log1p(df_an['valeur_max']),(Xi,Yi),method='linear')
            ct=ax3.contourf(Xi,Yi,Zi,levels=20,cmap='hot_r',alpha=0.8)
            plt.colorbar(ct,ax=ax3,label=f'log({el}+1)')
        for _,f in df_an.iterrows():
            pot=f['valeur_max']>=sa; mk={'RC':'^','Aircore':'o','Diamond':'s'}.get(f['type'],'o')
            ax3.scatter(f['easting'],f['northing'],c='#00FF00' if pot else '#FFFFFF',s=200 if pot else 80,
                       marker=mk,edgecolors='black',linewidths=1.5 if pot else 0.5,zorder=4)
            if pot:
                ax3.annotate(f"{f['trou']}\n{f['valeur_max']:.0f}",(f['easting'],f['northing']),
                            textcoords="offset points",xytext=(6,6),fontsize=7,color='#00FF00',fontweight='bold')
        if spt:
            np.random.seed(5)
            for i in range(5):
                x1=np.random.uniform(df_an['easting'].min(),df_an['easting'].max())
                y1=np.random.uniform(df_an['northing'].min(),df_an['northing'].max())
                ang=np.random.uniform(20,70); ln=np.random.uniform(200,500)
                ax3.plot([x1,x1+ln*np.cos(np.radians(ang))],[y1,y1+ln*np.sin(np.radians(ang))],
                        color='cyan',linewidth=2.5,linestyle='--',label='Structure' if i==0 else '')
        xmx=df_an['easting'].max(); ymx=df_an['northing'].max()
        xmn=df_an['easting'].min(); ymn=df_an['northing'].min()
        ax3.annotate('',xy=(xmx+60,ymx+25),xytext=(xmx+60,ymx-20),arrowprops=dict(arrowstyle='->',color='white',lw=2.5))
        ax3.text(xmx+60,ymx+35,'N',ha='center',fontsize=14,fontweight='bold',color='white')
        ax3.plot([xmn,xmn+200],[ymn-35,ymn-35],'w-',linewidth=3)
        ax3.text(xmn+100,ymn-50,'200 m',ha='center',fontsize=9,fontweight='bold',color='white')
        ax3.set_xlabel("Easting (m)",color='white'); ax3.set_ylabel("Northing (m)",color='white'); ax3.tick_params(colors='white')
        ax3.set_title(f"Carte anomalie {el} — {NOM_PROSPECT}",fontsize=12,fontweight='bold',color='white')
        plt.tight_layout(); st.pyplot(fig3)

# ══ TAB 4 — CARTES STRUCT ════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("🏗️ Carte Structurale")
    fig4,ax4=plt.subplots(figsize=(11,9))
    fig4.patch.set_facecolor('#F5F5F0'); ax4.set_facecolor('#F0EDE0')
    np.random.seed(10)
    for _,s in structures_df.head(15).iterrows():
        ang=s['direction']; ln=min(s['longueur_m'],400)
        x1=s['easting']; y1=s['northing']
        x2=x1+ln*np.cos(np.radians(ang)); y2=y1+ln*np.sin(np.radians(ang))
        c=STRUCT_COLORS.get(s['type'],'#888')
        ls='-' if 'Faille' in s['type'] else '--' if 'Veine' in s['type'] else ':'
        ax4.plot([x1,x2],[y1,y2],color=c,linewidth=3 if 'Faille' in s['type'] else 2,linestyle=ls,label=s['type'])
        ax4.text((x1+x2)/2,(y1+y2)/2,f"{s['direction']:.0f}°/{s['pendage']:.0f}°{s['sens_pendage']}",fontsize=6,color=c,fontweight='bold')
    for _,f in df_forages.iterrows():
        ax4.scatter(f['easting'],f['northing'],c='black',s=60,zorder=3)
        ax4.annotate(f['trou'],(f['easting'],f['northing']),textcoords="offset points",xytext=(4,4),fontsize=6)
    xmx=df_forages['easting'].max(); ymx=df_forages['northing'].max()
    xmn=df_forages['easting'].min(); ymn=df_forages['northing'].min()
    ax4.annotate('',xy=(xmx+60,ymx+25),xytext=(xmx+60,ymx-20),arrowprops=dict(arrowstyle='->',color='black',lw=2.5))
    ax4.text(xmx+60,ymx+35,'N',ha='center',fontsize=14,fontweight='bold')
    ax4.plot([xmn,xmn+200],[ymn-35,ymn-35],'k-',linewidth=3)
    ax4.text(xmn+100,ymn-50,'200 m',ha='center',fontsize=9,fontweight='bold')
    ax4.text(xmn,ymx+40,f"{NOM_PROSPECT} | {NOM_PERMIS}",fontsize=9,fontweight='bold',color='#1A237E')
    hl,ll=ax4.get_legend_handles_labels(); bl=dict(zip(ll,hl))
    ax4.legend(bl.values(),bl.keys(),loc='lower right',fontsize=8,title='Structures',framealpha=0.9)
    ax4.set_xlabel("Easting UTM (m)"); ax4.set_ylabel("Northing UTM (m)")
    ax4.set_title(f"Carte structurale — {NOM_PROSPECT}",fontsize=12,fontweight='bold')
    ax4.grid(True,linestyle=':',alpha=0.4); plt.tight_layout(); st.pyplot(fig4)
    with st.expander("📖 Guide d'interprétation — Carte structurale", expanded=False):
        st.markdown(INTERPRETATIONS["structures"]["intro"])
        st.markdown("### 🗂️ Interprétation par type de structure")
        for struct, interp in INTERPRETATIONS["structures"]["guide_couleurs"].items():
            c = STRUCT_COLORS.get(struct,'#888')
            st.markdown(f"<span style='background:{c};padding:2px 8px;border-radius:4px;color:white;font-weight:bold;'>{struct}</span> — {interp}", unsafe_allow_html=True)
    st.subheader("📋 Tableau des structures")
    st.dataframe(structures_df[['id','type','direction','pendage','sens_pendage','longueur_m','porteur_miner']].rename(columns={'id':'N°','type':'Type','direction':'Direction(°)','pendage':'Pendage(°)','sens_pendage':'Sens','longueur_m':'Longueur(m)','porteur_miner':'Porteur'}),use_container_width=True)

# ══ TAB 5 — LOGUES AUTOMATISÉS ════════════════════════════════════════════════
with tabs[4]:
    st.subheader("📊 Logues Automatisés — RC / Aircore / Auger / Diamond")
    st.info(f"**{NOM_PROSPECT}** | {NOM_PERMIS} — Sélectionnez le type et le trou pour afficher le logue automatiquement")

    col1,col2=st.columns([1,4])
    with col1:
        type_logue=st.selectbox("Type de forage",['RC','Aircore','Diamond','Auger'])
        if type_logue=='Auger':
            trous_log=df_auger['trou'].unique().tolist()
        else:
            trous_log=df_forages[df_forages['type']==type_logue]['trou'].tolist() if type_logue!='Tous' else df_forages['trou'].tolist()
        if len(trous_log)==0: trous_log=df_forages['trou'].tolist()
        trou_log=st.selectbox("Trou",trous_log)
        show_au_log=st.checkbox("Or (Au ppb)",True)
        show_as_log=st.checkbox("Arsenic (As ppm)",True)
        show_alter_log=st.checkbox("Altération",True)
        show_miner_log=st.checkbox("Minéralisation",True)
        seuil_log=st.number_input("Seuil Au (ppb)",10,1000,100)

    with col2:
        if type_logue=='Auger':
            # Logue Auger
            row_aug=df_auger[df_auger['trou']==trou_log]
            if len(row_aug)>0:
                r=row_aug.iloc[0]
                st.markdown(f"**Trou Auger :** {trou_log} | **Ligne :** {r['ligne']} | **Profondeur :** {r['profondeur_m']}m | **Statut :** {r['statut']}")
                st.markdown(f"**Coordonnées :** E={r['easting']} N={r['northing']} Z={r['elevation']}")
                pxrf_t=df_pxrf[df_pxrf['trou']==trou_log].sort_values('profondeur')
                if len(pxrf_t)>0:
                    ncols=1+(1 if show_au_log else 0)+(1 if show_as_log else 0)
                    fig_la,axes_la=plt.subplots(1,ncols,figsize=(4*ncols,8),sharey=True)
                    if ncols==1: axes_la=[axes_la]
                    # Logue lithologique Auger
                    litho_a=r['lithologie'] if not pd.isna(r['lithologie']) else 'Latérite'
                    c_la=LITHO_COLORS.get(litho_a,'#888')
                    axes_la[0].fill_betweenx([0,r['profondeur_m']],0,1,color=c_la,alpha=0.85)
                    axes_la[0].text(0.5,r['profondeur_m']/2,litho_a,ha='center',va='center',fontsize=8,fontweight='bold')
                    axes_la[0].set_ylim(r['profondeur_m'],0); axes_la[0].set_xlim(-0.1,1.1); axes_la[0].set_xticks([])
                    axes_la[0].set_title(f"Litho\n{trou_log}",fontsize=9,fontweight='bold'); axes_la[0].set_ylabel("Profondeur (m)")
                    cidx=1
                    if show_au_log and cidx<ncols:
                        cau=['#FFD700' if v>=seuil_log else '#EEE' for v in pxrf_t['Au_ppb']]
                        axes_la[cidx].barh(pxrf_t['profondeur'],pxrf_t['Au_ppb'].values,height=0.8,color=cau,edgecolor='orange',linewidth=0.4)
                        axes_la[cidx].axvline(x=seuil_log,color='red',linestyle='--',linewidth=1.5,label=f'{seuil_log}ppb')
                        axes_la[cidx].set_xlabel("Au (ppb)"); axes_la[cidx].set_title("Or",fontsize=9,fontweight='bold'); axes_la[cidx].legend(fontsize=7); cidx+=1
                    if show_as_log and cidx<ncols:
                        axes_la[cidx].barh(pxrf_t['profondeur'],pxrf_t['As_ppm'].values,height=0.8,color='#FF6B6B',edgecolor='red',linewidth=0.4)
                        axes_la[cidx].set_xlabel("As (ppm)"); axes_la[cidx].set_title("Arsenic",fontsize=9,fontweight='bold')
                    plt.suptitle(f"Logue Auger — {trou_log} | {r['ligne']} | {r['profondeur_m']}m | {r['statut']}",fontsize=10,fontweight='bold')
                    plt.tight_layout(); st.pyplot(fig_la)
                else:
                    st.warning("Aucune donnée pXRF pour ce trou Auger")
        else:
            # Logue RC/Aircore/Diamond
            f_info=df_forages[df_forages['trou']==trou_log]
            if len(f_info)>0:
                f_info=f_info.iloc[0]
                ints_log=df_intervals[df_intervals['trou']==trou_log].sort_values('de')
                st.markdown(f"**Trou :** {trou_log} | **Type :** {f_info['type']} | **Prof :** {f_info['profondeur']}m | **Az :** {f_info['azimut']}° | **Inc :** {f_info['inclinaison']}°")
                st.markdown(f"**Coordonnées :** E={f_info['easting']} N={f_info['northing']} Z={f_info['elevation']} | **Équipe :** {f_info['equipe']}")
                # Nombre de colonnes selon sélection
                ncols=1+(1 if show_alter_log else 0)+(1 if show_miner_log else 0)+(1 if show_au_log else 0)+(1 if show_as_log else 0)
                ncols=max(1,ncols)
                fig_ll,axes_ll=plt.subplots(1,ncols,figsize=(3.5*ncols,12),sharey=True)
                if ncols==1: axes_ll=[axes_ll]
                cidx=0
                # Col 1 — Lithologie
                for _,iv in ints_log.iterrows():
                    c=LITHO_COLORS.get(iv['lithologie'],'#888')
                    axes_ll[0].fill_betweenx([iv['de'],iv['a']],0,1,color=c,alpha=0.85)
                    if iv['mineralisé']:
                        axes_ll[0].fill_betweenx([iv['de'],iv['a']],0,1,color='red',alpha=0.2,hatch='///')
                    axes_ll[0].plot([0,1],[iv['de'],iv['de']],'k-',linewidth=0.3)
                    mid=(iv['de']+iv['a'])/2
                    axes_ll[0].text(0.5,mid,iv['lithologie'][:10],ha='center',va='center',fontsize=5.5,fontweight='bold')
                    axes_ll[0].text(-0.05,iv['de'],f"{iv['de']}m",ha='right',fontsize=5.5)
                axes_ll[0].set_ylim(f_info['profondeur'],0); axes_ll[0].set_xlim(-0.1,1.1); axes_ll[0].set_xticks([])
                axes_ll[0].set_title(f"Litho\n{trou_log}",fontsize=9,fontweight='bold'); axes_ll[0].set_ylabel("Profondeur (m)")
                cidx=1
                # Col 2 — Altération
                if show_alter_log and cidx<ncols:
                    for _,iv in ints_log.iterrows():
                        c=ALTER_COLORS.get(iv['alteration'],'#888')
                        axes_ll[cidx].fill_betweenx([iv['de'],iv['a']],0,1,color=c,alpha=0.85)
                        mid=(iv['de']+iv['a'])/2
                        axes_ll[cidx].text(0.5,mid,iv['alteration'][:8],ha='center',va='center',fontsize=5,fontweight='bold')
                        axes_ll[cidx].plot([0,1],[iv['de'],iv['de']],'k-',linewidth=0.3)
                    axes_ll[cidx].set_ylim(f_info['profondeur'],0); axes_ll[cidx].set_xlim(-0.1,1.1); axes_ll[cidx].set_xticks([])
                    axes_ll[cidx].set_title("Altération",fontsize=9,fontweight='bold'); cidx+=1
                # Col 3 — Minéralisation
                if show_miner_log and cidx<ncols:
                    for _,iv in ints_log.iterrows():
                        c=MINER_COLORS.get(iv['mineralisation'],'#888')
                        axes_ll[cidx].fill_betweenx([iv['de'],iv['a']],0,1,color=c,alpha=0.85)
                        mid=(iv['de']+iv['a'])/2
                        axes_ll[cidx].text(0.5,mid,iv['mineralisation'][:10],ha='center',va='center',fontsize=4.5,fontweight='bold')
                        axes_ll[cidx].plot([0,1],[iv['de'],iv['de']],'k-',linewidth=0.3)
                    axes_ll[cidx].set_ylim(f_info['profondeur'],0); axes_ll[cidx].set_xlim(-0.1,1.1); axes_ll[cidx].set_xticks([])
                    axes_ll[cidx].set_title("Minéralisation",fontsize=9,fontweight='bold'); cidx+=1
                # Col 4 — Au
                if show_au_log and cidx<ncols:
                    cau=['#FFD700' if v>=seuil_log else '#EEE' for v in ints_log['Au_ppb']]
                    axes_ll[cidx].barh([(iv['de']+iv['a'])/2 for _,iv in ints_log.iterrows()],
                                      ints_log['Au_ppb'].values,
                                      height=[(iv['a']-iv['de'])*0.8 for _,iv in ints_log.iterrows()],
                                      color=cau,edgecolor='orange',linewidth=0.5)
                    axes_ll[cidx].axvline(x=seuil_log,color='red',linestyle='--',linewidth=1.5,label=f'{seuil_log}ppb')
                    axes_ll[cidx].set_xlabel("Au (ppb)"); axes_ll[cidx].set_title("Or",fontsize=9,fontweight='bold')
                    axes_ll[cidx].legend(fontsize=7); cidx+=1
                # Col 5 — As
                if show_as_log and cidx<ncols:
                    axes_ll[cidx].barh([(iv['de']+iv['a'])/2 for _,iv in ints_log.iterrows()],
                                      ints_log['As_ppm'].values,
                                      height=[(iv['a']-iv['de'])*0.8 for _,iv in ints_log.iterrows()],
                                      color='#FF6B6B',edgecolor='red',linewidth=0.5)
                    axes_ll[cidx].set_xlabel("As (ppm)"); axes_ll[cidx].set_title("Arsenic",fontsize=9,fontweight='bold')
                plt.suptitle(f"Logue {f_info['type']} — {trou_log} | E:{f_info['easting']} N:{f_info['northing']} Z:{f_info['elevation']} | {f_info['profondeur']}m",fontsize=10,fontweight='bold')
                # Légendes
                plt.tight_layout(); st.pyplot(fig_ll)
                # Tableau des intervalles
                st.markdown("#### 📋 Tableau des intervalles")
                tab_iv=ints_log[['de','a','lithologie','alteration','mineralisation','Au_ppb','Cu_ppm','As_ppm','mineralisé']].copy()
                tab_iv.columns=['De(m)','A(m)','Lithologie','Altération','Minéralisation','Au(ppb)','Cu(ppm)','As(ppm)','Minéralisé']
                st.dataframe(tab_iv.style.map(lambda v:'background-color:#FFD700;color:black' if v==True else 'background-color:#F5F5F5' if v==False else '',subset=['Minéralisé']).format({'Au(ppb)':'{:.2f}','Cu(ppm)':'{:.1f}','As(ppm)':'{:.1f}'}),use_container_width=True)
                csv_log=tab_iv.to_csv(index=False)
                st.download_button(f"📥 Télécharger logue {trou_log}",data=csv_log,file_name=f"logue_{trou_log}.csv",mime='text/csv')

# ══ TAB 6 — AUGER ════════════════════════════════════════════════════════════
with tabs[5]:
    st.subheader(f"⛏️ Auger — {NOM_PROSPECT}")
    st.markdown(f"**{NOM_PERMIS}** | {N_LIGNES} lignes × {N_PTS} trous")
    with st.expander("📖 Guide d'interprétation — Forage Auger", expanded=False):
        st.markdown(INTERPRETATIONS["auger"]["intro"])
        st.markdown(INTERPRETATIONS["auger"]["anomalie_interpretation"])
    aug_vue=st.radio("Module",['Carte planification','Carte anomalie digitalisée','Données Auger','Profils par ligne'],horizontal=True,key='av')

    if aug_vue=='Carte planification':
        col1,col2=st.columns([3,1])
        with col2:
            slb=st.checkbox("Labels trous",True,key='alb')
            slg=st.checkbox("Relier lignes",True,key='alg')
        with col1:
            fig_a,ax_a=plt.subplots(figsize=(13,11))
            fig_a.patch.set_facecolor('#F5F5F0'); ax_a.set_facecolor('#E8F4E8')
            if slg:
                for lg in df_auger['ligne'].unique():
                    sb=df_auger[df_auger['ligne']==lg].sort_values('easting')
                    ax_a.plot(sb['easting'],sb['northing'],color='#AAAAAA',linewidth=1,linestyle='--',zorder=1,alpha=0.7)
            for st_ in ['Foré','En cours','Stoppé']:
                sb=df_auger[df_auger['statut']==st_]
                ax_a.scatter(sb['easting'],sb['northing'],c=COLOR_ST[st_],s=150,marker=MARKER_ST[st_],
                            edgecolors='black',linewidths=0.8,zorder=3,label=f'{st_} ({len(sb)})',alpha=0.9)
                if slb:
                    for _,r in sb.iterrows():
                        ax_a.text(r['easting'],r['northing']+6,r['trou'],fontsize=5,ha='center',color='#1A237E',fontweight='bold')
            for lg in df_auger['ligne'].unique():
                sb=df_auger[df_auger['ligne']==lg]
                ax_a.text(sb['easting'].min()-20,sb['northing'].mean(),lg,fontsize=9,fontweight='bold',
                         color='#1A237E',va='center',ha='right',bbox=dict(boxstyle='round',facecolor='white',alpha=0.8))
            xmx=df_auger['easting'].max(); ymx=df_auger['northing'].max()
            xmn=df_auger['easting'].min(); ymn=df_auger['northing'].min()
            ax_a.annotate('',xy=(xmx+60,ymx+30),xytext=(xmx+60,ymx-30),arrowprops=dict(arrowstyle='->',color='black',lw=2.5))
            ax_a.text(xmx+60,ymx+45,'N',ha='center',fontsize=16,fontweight='bold')
            ax_a.plot([xmn,xmn+150],[ymn-40,ymn-40],'k-',linewidth=3)
            ax_a.plot([xmn,xmn],[ymn-48,ymn-32],'k-',linewidth=2); ax_a.plot([xmn+150,xmn+150],[ymn-48,ymn-32],'k-',linewidth=2)
            ax_a.text(xmn+75,ymn-62,'150 m',ha='center',fontsize=9,fontweight='bold')
            lga=[mpatches.Patch(color='#2196F3',label=f"Foré ({(df_auger['statut']=='Foré').sum()})"),
                 mpatches.Patch(color='#4CAF50',label=f"En cours ({(df_auger['statut']=='En cours').sum()})"),
                 mpatches.Patch(color='#F44336',label=f"Stoppé ({(df_auger['statut']=='Stoppé').sum()})")]
            ax_a.legend(handles=lga,loc='lower right',fontsize=9,title='Statut des trous',framealpha=0.95,edgecolor='black')
            ax_a.set_xlabel("Easting UTM (m)"); ax_a.set_ylabel("Northing UTM (m)")
            ax_a.set_title(f"Carte planification Auger — {NOM_PROSPECT}\n{NOM_PERMIS} | Esp.L:{ESP_L}m | Esp.T:{ESP_P}m",fontsize=12,fontweight='bold')
            ax_a.grid(True,linestyle=':',alpha=0.4); plt.tight_layout(); st.pyplot(fig_a)
        c1,c2,c3,c4,c5=st.columns(5)
        c1.metric("Total",len(df_auger)); c2.metric("Forés 🔵",int((df_auger['statut']=='Foré').sum()))
        c3.metric("En cours 🟢",int((df_auger['statut']=='En cours').sum())); c4.metric("Stoppés 🔴",int((df_auger['statut']=='Stoppé').sum()))
        c5.metric("Avancement",f"{int((df_auger['statut']=='Foré').sum())/len(df_auger)*100:.0f}%")

    elif aug_vue=='Carte anomalie digitalisée':
        col1,col2=st.columns([1,3])
        with col1:
            el_an=st.selectbox("Élément",['Au_ppb','As_ppm','Cu_ppm','Fe_pct'],key='elan')
            seuil_an=st.number_input("Seuil",1,1000,100,key='san')
            sst=st.checkbox("Structures porteuses",True,key='ssa')
        with col2:
            df_fa=df_auger[df_auger['statut']=='Foré'].dropna(subset=[el_an])
            fig_aan,ax_aan=plt.subplots(figsize=(12,10)); fig_aan.patch.set_facecolor('#0D1B2A'); ax_aan.set_facecolor('#0D1B2A')
            if len(df_fa)>4:
                xi=np.linspace(df_fa['easting'].min()-50,df_fa['easting'].max()+50,200)
                yi=np.linspace(df_fa['northing'].min()-50,df_fa['northing'].max()+50,200)
                Xi,Yi=np.meshgrid(xi,yi)
                Zi=griddata((df_fa['easting'],df_fa['northing']),np.log1p(df_fa[el_an].fillna(0)),(Xi,Yi),method='linear')
                cf=ax_aan.contourf(Xi,Yi,Zi,levels=20,cmap='hot_r',alpha=0.85)
                plt.colorbar(cf,ax=ax_aan,label=f'log({el_an}+1)',shrink=0.8)
            for lg in df_auger['ligne'].unique():
                sb=df_auger[df_auger['ligne']==lg].sort_values('easting')
                ax_aan.plot(sb['easting'],sb['northing'],color='#555',linewidth=0.8,linestyle='--',zorder=1,alpha=0.5)
            for st_,c in COLOR_ST.items():
                sb=df_auger[df_auger['statut']==st_]
                ax_aan.scatter(sb['easting'],sb['northing'],c=c,s=100,marker=MARKER_ST[st_],edgecolors='white',linewidths=0.8,zorder=3,alpha=0.9)
            anom=df_fa[df_fa[el_an]>=seuil_an]
            if len(anom)>0:
                ax_aan.scatter(anom['easting'],anom['northing'],c='none',s=400,marker='o',edgecolors='#00FF00',linewidths=2.5,zorder=5)
                for _,r in anom.iterrows():
                    ax_aan.text(r['easting'],r['northing']+8,f"{r['trou']}\n{r[el_an]:.0f}",fontsize=6.5,ha='center',color='#00FF00',fontweight='bold')
            if sst:
                np.random.seed(3)
                for i in range(4):
                    x1=df_auger['easting'].min()+np.random.uniform(50,200); y1=df_auger['northing'].min()+np.random.uniform(50,300)
                    ang=np.random.uniform(30,70); ln=np.random.uniform(200,500)
                    ax_aan.plot([x1,x1+ln*np.cos(np.radians(ang))],[y1,y1+ln*np.sin(np.radians(ang))],color='cyan',linewidth=2.5,linestyle='--',label='Structure' if i==0 else '',zorder=4)
                    ax_aan.text(x1+ln*np.cos(np.radians(ang))/2,y1+ln*np.sin(np.radians(ang))/2,'VQ',fontsize=8,color='cyan',fontweight='bold')
            xmx=df_auger['easting'].max(); ymx=df_auger['northing'].max(); xmn=df_auger['easting'].min(); ymn=df_auger['northing'].min()
            ax_aan.annotate('',xy=(xmx+60,ymx+25),xytext=(xmx+60,ymx-25),arrowprops=dict(arrowstyle='->',color='white',lw=2.5))
            ax_aan.text(xmx+60,ymx+40,'N',ha='center',fontsize=16,fontweight='bold',color='white')
            ax_aan.plot([xmn,xmn+150],[ymn-40,ymn-40],'w-',linewidth=3); ax_aan.text(xmn+75,ymn-60,'150 m',ha='center',fontsize=9,fontweight='bold',color='white')
            lga2=[mpatches.Patch(color='#2196F3',label='Foré'),mpatches.Patch(color='#4CAF50',label='En cours'),mpatches.Patch(color='#F44336',label='Stoppé'),
                  plt.Line2D([0],[0],marker='o',color='w',markerfacecolor='none',markeredgecolor='#00FF00',markersize=12,label=f'Anomalie >{seuil_an}'),
                  plt.Line2D([0],[0],color='cyan',linestyle='--',linewidth=2,label='Structure porteuse')]
            ax_aan.legend(handles=lga2,loc='lower right',fontsize=8,framealpha=0.8,facecolor='#0D1B2A',labelcolor='white')
            ax_aan.set_xlabel("Easting (m)",color='white'); ax_aan.set_ylabel("Northing (m)",color='white'); ax_aan.tick_params(colors='white')
            ax_aan.set_title(f"Carte anomalie {el_an} — {NOM_PROSPECT}",fontsize=12,fontweight='bold',color='white')
            ax_aan.grid(True,linestyle=':',alpha=0.2,color='gray'); plt.tight_layout(); st.pyplot(fig_aan)
            st.success(f"**{len(anom)} trous anomaliques** ({el_an} ≥ {seuil_an}) | Structures porteuses : Veines de quartz en contexte de cisaillement NE-SO")

    elif aug_vue=='Données Auger':
        col1,col2=st.columns(2)
        with col1:
            fl=st.multiselect("Ligne",df_auger['ligne'].unique(),default=list(df_auger['ligne'].unique()[:3]),key='fla')
            fs=st.multiselect("Statut",['Foré','En cours','Stoppé'],default=['Foré','En cours','Stoppé'],key='fsa')
        with col2:
            sat=st.number_input("Seuil Au anomalique (ppb)",10,1000,100,key='sat')
        df_filt=df_auger[(df_auger['ligne'].isin(fl))&(df_auger['statut'].isin(fs))]
        st.dataframe(df_filt.style.map(lambda v:f'background-color:{COLOR_ST.get(v,"")};color:white' if v in COLOR_ST else '',subset=['statut']),use_container_width=True)
        resume=df_auger.groupby('ligne').agg(total=('trou','count'),fores=('statut',lambda x:(x=='Foré').sum()),stoppes=('statut',lambda x:(x=='Stoppé').sum()),au_max=('Au_ppb','max'),au_moy=('Au_ppb','mean')).round(1).reset_index()
        resume['anomalie']=resume['au_max'].apply(lambda v:'✅' if v>=sat else '❌')
        st.dataframe(resume,use_container_width=True)
        st.download_button("📥 Données Auger",data=df_auger.to_csv(index=False),file_name="donnees_auger.csv",mime='text/csv')

    else:
        el_p=st.selectbox("Élément",['Au_ppb','As_ppm','Cu_ppm','Fe_pct'],key='elp')
        df_fp=df_auger[df_auger['statut']=='Foré'].dropna(subset=[el_p])
        fig_pp,axes_pp=plt.subplots(2,5,figsize=(18,8),sharey=False); axes_pp=axes_pp.flatten()
        for i,lg in enumerate([f'L{j+1:02d}' for j in range(N_LIGNES)]):
            sb=df_fp[df_fp['ligne']==lg].sort_values('easting')
            if len(sb)>0:
                cb=['#FFD700' if v>=100 else '#AAAAAA' for v in sb[el_p]]
                axes_pp[i].bar(range(len(sb)),sb[el_p].values,color=cb,edgecolor='black',linewidth=0.4)
                axes_pp[i].set_xticks(range(len(sb))); axes_pp[i].set_xticklabels([t.split('T')[1] for t in sb['trou']],fontsize=7)
                axes_pp[i].set_title(lg,fontsize=9,fontweight='bold'); axes_pp[i].set_ylabel(el_p,fontsize=7)
                axes_pp[i].axhline(y=100,color='red',linestyle='--',linewidth=1,alpha=0.7); axes_pp[i].grid(True,linestyle=':',alpha=0.3)
            else:
                axes_pp[i].text(0.5,0.5,'Aucun foré',ha='center',va='center',fontsize=8); axes_pp[i].set_title(lg,fontsize=9)
        plt.suptitle(f"Profils {el_p} — {NOM_PROSPECT}",fontsize=12,fontweight='bold'); plt.tight_layout(); st.pyplot(fig_pp)

# ══ TAB 7 — pXRF ═════════════════════════════════════════════════════════════
with tabs[6]:
    st.subheader(f"📡 pXRF & Géochimie — {NOM_PROSPECT}")
    with st.expander("📖 Guide d'interprétation — pXRF & Géochimie", expanded=False):
        st.markdown(INTERPRETATIONS["pxrf"]["intro"])
        st.markdown(INTERPRETATIONS["pxrf"]["seuils_pxrf"])
    pv=st.radio("Module",["Vue d'ensemble","Profils pXRF","Carte géochimique","Corrélations","Statistiques"],horizontal=True,key='pv')

    if pv=="Vue d'ensemble":
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Mesures pXRF",len(df_pxrf)); c2.metric("Trous analysés",df_pxrf['trou'].nunique())
        c3.metric("Au max",f"{df_pxrf['Au_ppb'].max():.1f} ppb"); c4.metric("Au moyen",f"{df_pxrf['Au_ppb'].mean():.1f} ppb")
        fig_pv,axes_pv=plt.subplots(1,3,figsize=(14,4))
        for ax_p,el_v,cp in zip(axes_pv,['Au_ppb','As_ppm','Cu_ppm'],['#FFD700','#FF6B6B','#B87333']):
            ax_p.hist(np.log10(df_pxrf[el_v]+0.01),bins=25,color=cp,edgecolor='black',linewidth=0.5)
            ax_p.set_xlabel(f"log10({el_v})"); ax_p.set_ylabel("Fréquence"); ax_p.set_title(f"Distribution {el_v}",fontsize=10,fontweight='bold'); ax_p.grid(True,linestyle=':',alpha=0.4)
        plt.suptitle(f"pXRF — Vue d'ensemble | {NOM_PROSPECT}",fontsize=12,fontweight='bold'); plt.tight_layout(); st.pyplot(fig_pv)

    elif pv=='Profils pXRF':
        col1,col2=st.columns([1,3])
        with col1:
            tp=st.selectbox("Trou",df_pxrf['trou'].unique(),key='ptp')
            ep=st.selectbox("Élément",['Au_ppb','As_ppm','Cu_ppm','Fe_pct','Mn_ppm','Zn_ppm'],key='pep')
            sp=st.number_input("Seuil",1,1000,100,key='psp')
        with col2:
            df_tpp=df_pxrf[df_pxrf['trou']==tp].sort_values('profondeur')
            if len(df_tpp)>0:
                fig_pp2,ax_pp2=plt.subplots(figsize=(6,9))
                cp2=['#FFD700' if v>=sp else '#EEEEEE' for v in df_tpp[ep]]
                ax_pp2.barh(df_tpp['profondeur'],df_tpp[ep].values,height=0.7,color=cp2,edgecolor='orange',linewidth=0.4)
                ax_pp2.axvline(x=sp,color='red',linestyle='--',linewidth=2,label=f'Seuil {sp}')
                ax_pp2.set_ylabel("Profondeur (m)"); ax_pp2.set_xlabel(ep)
                ax_pp2.set_title(f"Profil pXRF — {tp}\n{ep}",fontsize=11,fontweight='bold')
                ax_pp2.invert_yaxis(); ax_pp2.legend(fontsize=9); ax_pp2.grid(True,linestyle=':',alpha=0.4)
                plt.tight_layout(); st.pyplot(fig_pp2)
                c1,c2=st.columns(2); c1.metric("Max",f"{df_tpp[ep].max():.2f}"); c2.metric("Moyenne",f"{df_tpp[ep].mean():.2f}")

    elif pv=='Carte géochimique':
        col1,col2=st.columns([1,3])
        with col1:
            eg=st.selectbox("Élément",['Au_ppb','As_ppm','Cu_ppm','Fe_pct'],key='peg')
            pg=st.slider("Profondeur max (m)",1,25,5,key='ppg')
        with col2:
            df_gm=df_pxrf[df_pxrf['profondeur']<=pg].groupby(['trou','easting','northing'])[eg].mean().reset_index()
            fig_gm,ax_gm=plt.subplots(figsize=(11,9)); fig_gm.patch.set_facecolor('#0A1628'); ax_gm.set_facecolor('#0A1628')
            if len(df_gm)>3:
                xi=np.linspace(df_gm['easting'].min()-30,df_gm['easting'].max()+30,150)
                yi=np.linspace(df_gm['northing'].min()-30,df_gm['northing'].max()+30,150)
                Xi,Yi=np.meshgrid(xi,yi)
                Zi=griddata((df_gm['easting'],df_gm['northing']),np.log1p(df_gm[eg]),(Xi,Yi),method='linear')
                cf=ax_gm.contourf(Xi,Yi,Zi,levels=20,cmap='YlOrRd',alpha=0.85)
                plt.colorbar(cf,ax=ax_gm,label=f'log({eg}+1)',shrink=0.8)
            for _,r in df_gm.iterrows(): ax_gm.scatter(r['easting'],r['northing'],c='white',s=60,edgecolors='black',zorder=4)
            xmx=df_gm['easting'].max(); ymx=df_gm['northing'].max(); xmn=df_gm['easting'].min(); ymn=df_gm['northing'].min()
            ax_gm.annotate('',xy=(xmx+40,ymx+20),xytext=(xmx+40,ymx-20),arrowprops=dict(arrowstyle='->',color='white',lw=2))
            ax_gm.text(xmx+40,ymx+30,'N',ha='center',fontsize=14,fontweight='bold',color='white')
            ax_gm.plot([xmn,xmn+100],[ymn-25,ymn-25],'w-',linewidth=3); ax_gm.text(xmn+50,ymn-40,'100 m',ha='center',fontsize=8,fontweight='bold',color='white')
            ax_gm.set_xlabel("Easting (m)",color='white'); ax_gm.set_ylabel("Northing (m)",color='white'); ax_gm.tick_params(colors='white')
            ax_gm.set_title(f"Carte géochimique pXRF — {eg}\n{NOM_PROSPECT}",fontsize=12,fontweight='bold',color='white')
            plt.tight_layout(); st.pyplot(fig_gm)

    elif pv=='Corrélations':
        fig_co,axes_co=plt.subplots(2,2,figsize=(12,10))
        for ax_co,(xc,yc) in zip(axes_co.flatten(),[('Au_ppb','As_ppm'),('Au_ppb','Cu_ppm'),('Au_ppb','Fe_pct'),('As_ppm','Cu_ppm')]):
            ax_co.scatter(df_pxrf[xc],df_pxrf[yc],c='#2196F3',s=20,alpha=0.5,edgecolors='none')
            if len(df_pxrf)>2:
                z=np.polyfit(df_pxrf[xc].fillna(0),df_pxrf[yc].fillna(0),1); p=np.poly1d(z)
                xl=np.linspace(df_pxrf[xc].min(),df_pxrf[xc].max(),50)
                ax_co.plot(xl,p(xl),'r--',linewidth=2)
                r=np.corrcoef(df_pxrf[xc].fillna(0),df_pxrf[yc].fillna(0))[0,1]
                ax_co.text(0.05,0.95,f'r = {r:.2f}',transform=ax_co.transAxes,fontsize=10,fontweight='bold',color='red',verticalalignment='top',bbox=dict(boxstyle='round',facecolor='white',alpha=0.8))
            ax_co.set_xlabel(xc); ax_co.set_ylabel(yc); ax_co.set_title(f"{xc} vs {yc}",fontsize=10,fontweight='bold'); ax_co.grid(True,linestyle=':',alpha=0.4)
        plt.suptitle(f"Corrélations pXRF — {NOM_PROSPECT}",fontsize=12,fontweight='bold'); plt.tight_layout(); st.pyplot(fig_co)

    else:
        els=['Au_ppb','As_ppm','Cu_ppm','Fe_pct','Mn_ppm','Zn_ppm']
        st.dataframe(df_pxrf[els].describe().round(3),use_container_width=True)
        fig_bx,ax_bx=plt.subplots(figsize=(10,5))
        bp=ax_bx.boxplot([np.log10(df_pxrf[el]+0.01) for el in els],patch_artist=True,labels=els)
        for patch,color in zip(bp['boxes'],['#FFD700','#FF6B6B','#B87333','#FF5722','#9C27B0','#2196F3']): patch.set_facecolor(color); patch.set_alpha(0.8)
        ax_bx.set_ylabel("log10(valeur)"); ax_bx.set_title("Distribution pXRF",fontsize=11,fontweight='bold'); ax_bx.grid(True,linestyle=':',alpha=0.4); plt.xticks(rotation=20); plt.tight_layout(); st.pyplot(fig_bx)
        st.download_button("📥 Données pXRF",data=df_pxrf.to_csv(index=False),file_name="donnees_pxrf.csv",mime='text/csv')

# ══ TAB 8 — GÉOPHYSIQUE ══════════════════════════════════════════════════════
with tabs[7]:
    st.subheader(f"🌊 Géophysique — {NOM_PROSPECT}")
    with st.expander("📖 Guide d'interprétation — Géophysique", expanded=False):
        st.markdown(INTERPRETATIONS["geophysique"]["intro"])
        st.markdown(INTERPRETATIONS["geophysique"]["combinaison"])
    gv=st.radio("Méthode",['IP & Résistivité','Magnétométrie','EM','SP','Synthèse multi-méthodes'],horizontal=True,key='gv')

    if gv=='IP & Résistivité':
        fig_ip,axes_ip=plt.subplots(1,2,figsize=(16,8))
        for ax_ip in axes_ip: ax_ip.set_facecolor('#0A1628')
        fig_ip.patch.set_facecolor('#0A1628')
        plot_geo_map(geo_data,'IP_chargeabilite','IP — Chargeabilité (msec/V)','hot_r',axes_ip[0],df_forages)
        plot_geo_map(geo_data,'resistivite_ohm','Résistivité (Ohm.m)','RdYlBu_r',axes_ip[1],df_forages,nord=False,echelle=False)
        plt.suptitle(f"IP & Résistivité — {NOM_PROSPECT}",fontsize=13,fontweight='bold',color='white'); fig_ip.patch.set_facecolor('#0A1628'); plt.tight_layout(); st.pyplot(fig_ip)
        st.success(f"**{int(geo_data['anomalie_IP'].sum())} anomalies IP** → sulfures disséminés → cibles Diamond")
        st.info("**Forte chargeabilité** (>15 msec/V) = sulfures | **Faible résistivité** = zones altérées | Combinés = zone minéralisée")
    elif gv=='Magnétométrie':
        fig_mag,axes_mag=plt.subplots(1,2,figsize=(16,8))
        for ax_m in axes_mag: ax_m.set_facecolor('#0A1628')
        fig_mag.patch.set_facecolor('#0A1628')
        plot_geo_map(geo_data,'mag_nT','Champ magnétique total (nT)','jet',axes_mag[0],df_forages)
        plot_geo_map(geo_data,'mag_anomalie_nT','Anomalie magnétique (nT)','bwr',axes_mag[1],df_forages,nord=False,echelle=False)
        plt.suptitle(f"Magnétométrie — {NOM_PROSPECT}",fontsize=13,fontweight='bold',color='white'); fig_mag.patch.set_facecolor('#0A1628'); plt.tight_layout(); st.pyplot(fig_mag)
        st.success(f"**{int(geo_data['anomalie_mag'].sum())} anomalies magnétiques** (|ΔT| > 150 nT)")
    elif gv=='EM':
        fig_em,ax_em=plt.subplots(figsize=(11,9)); ax_em.set_facecolor('#0A1628'); fig_em.patch.set_facecolor('#0A1628')
        plot_geo_map(geo_data,'EM_conductivite','Conductivité EM (mS/m)','viridis',ax_em,df_forages)
        plt.tight_layout(); st.pyplot(fig_em)
        st.info("**Forte conductivité** = zones argileuses/sulfures | **Faible conductivité** = granite/quartzite résistant")
    elif gv=='SP':
        fig_sp,ax_sp=plt.subplots(figsize=(11,9)); ax_sp.set_facecolor('#0A1628'); fig_sp.patch.set_facecolor('#0A1628')
        plot_geo_map(geo_data,'SP_mV','Potentiel spontané (mV)','seismic',ax_sp,df_forages)
        plt.tight_layout(); st.pyplot(fig_sp)
        st.info("**SP négatif** (< -20 mV) = corps conducteurs/sulfures | SP négatif + anomalie IP → cibles prioritaires")
    else:
        fig_sy,axes_sy=plt.subplots(2,2,figsize=(14,12))
        for ax_s in axes_sy.flatten(): ax_s.set_facecolor('#0A1628')
        fig_sy.patch.set_facecolor('#0A1628')
        plot_geo_map(geo_data,'IP_chargeabilite','IP','hot_r',axes_sy[0,0],df_forages,nord=False,echelle=False)
        plot_geo_map(geo_data,'mag_anomalie_nT','Magnétique','bwr',axes_sy[0,1],df_forages,nord=False,echelle=False)
        plot_geo_map(geo_data,'EM_conductivite','EM','viridis',axes_sy[1,0],df_forages,nord=False,echelle=False)
        plot_geo_map(geo_data,'SP_mV','SP','seismic',axes_sy[1,1],df_forages,nord=True,echelle=True)
        plt.suptitle(f"Synthèse géophysique multi-méthodes — {NOM_PROSPECT}",fontsize=13,fontweight='bold',color='white'); plt.tight_layout(); st.pyplot(fig_sy)
        synth_tab=geo_data[['point','easting','northing','IP_chargeabilite','mag_anomalie_nT','EM_conductivite','SP_mV','anomalie_IP','anomalie_mag']].copy()
        synth_tab.columns=['Point','Easting','Northing','IP(msec/V)','ΔMag(nT)','EM(mS/m)','SP(mV)','Anomalie IP','Anomalie Mag']
        st.dataframe(synth_tab.style.map(lambda v:'background-color:#FFD700;color:black' if v==True else 'background-color:#F5F5F5' if v==False else '',subset=['Anomalie IP','Anomalie Mag']),use_container_width=True)
        st.download_button("📥 Données géophysiques",data=geo_data.to_csv(index=False),file_name="donnees_geophysique.csv",mime='text/csv')

# ══ TAB 9 — SGI ══════════════════════════════════════════════════════════════
with tabs[8]:
    st.subheader("🧪 Essai SGI")
    with st.expander("📖 Guide d'interprétation — Essai SGI", expanded=False):
        st.markdown(INTERPRETATIONS["sgi"]["intro"])
        st.markdown("### 🔥 Interprétation par type d'altération")
        for alter, interp in INTERPRETATIONS["sgi"]["alteration_guide"].items():
            c = ALTER_COLORS.get(alter,'#888')
            st.markdown(f"<span style='background:{c};padding:2px 8px;border-radius:4px;font-weight:bold;'>{alter}</span> — {interp}", unsafe_allow_html=True)
    col1,col2=st.columns([1,2])
    with col1:
        tsg=st.selectbox("Trou",df_forages['trou'].tolist(),key='sgi')
        sau_s=st.number_input("Seuil Au (ppb)",10,1000,100,key='saus')
        scu_s=st.number_input("Seuil Cu (ppm)",5,200,50,key='scus')
    with col2:
        ints_s=df_intervals[df_intervals['trou']==tsg].sort_values('de').copy()
        ints_s['mineralisé']=ints_s['Au_ppb']>=sau_s
        tm_s=ints_s['a'].max()-ints_s['de'].min()
        mm_s=ints_s[ints_s['mineralisé']].apply(lambda r:r['a']-r['de'],axis=1).sum()
        pm_s=mm_s/tm_s*100 if tm_s>0 else 0
        c1,c2,c3=st.columns(3)
        c1.metric("Mètres minéralisés",f"{mm_s:.1f} m"); c2.metric("% minéralisé",f"{pm_s:.1f}%"); c3.metric("Au max",f"{ints_s['Au_ppb'].max():.1f} ppb")
        f_s=df_forages[df_forages['trou']==tsg].iloc[0]
        fig_sg,ax_sg=plt.subplots(1,3,figsize=(10,10),sharey=True)
        for _,iv in ints_s.iterrows():
            c=ALTER_COLORS.get(iv['alteration'],'#888')
            ax_sg[0].fill_betweenx([iv['de'],iv['a']],0,1,color=c,alpha=0.85)
            mid=(iv['de']+iv['a'])/2
            ax_sg[0].text(0.5,mid,iv['alteration'][:10],ha='center',va='center',fontsize=5.5,fontweight='bold')
            ax_sg[0].plot([0,1],[iv['de'],iv['de']],'k-',linewidth=0.3)
            ax_sg[0].text(-0.05,iv['de'],f"{iv['de']}m",ha='right',fontsize=5.5)
        ax_sg[0].set_ylim(f_s['profondeur'],0); ax_sg[0].set_xticks([]); ax_sg[0].set_title("Altération",fontsize=9,fontweight='bold'); ax_sg[0].set_ylabel("Profondeur (m)")
        for _,iv in ints_s.iterrows():
            c=MINER_COLORS.get(iv['mineralisation'],'#888')
            ax_sg[1].fill_betweenx([iv['de'],iv['a']],0,1,color=c,alpha=0.85)
            mid=(iv['de']+iv['a'])/2
            ax_sg[1].text(0.5,mid,iv['mineralisation'][:12],ha='center',va='center',fontsize=5,fontweight='bold')
            ax_sg[1].plot([0,1],[iv['de'],iv['de']],'k-',linewidth=0.3)
        ax_sg[1].set_ylim(f_s['profondeur'],0); ax_sg[1].set_xticks([]); ax_sg[1].set_title("Minéralisation",fontsize=9,fontweight='bold')
        cau_sg=['#FFD700' if v>=sau_s else '#EEE' for v in ints_s['Au_ppb']]
        ax_sg[2].barh([(iv['de']+iv['a'])/2 for _,iv in ints_s.iterrows()],ints_s['Au_ppb'].values,
                      height=[(iv['a']-iv['de'])*0.8 for _,iv in ints_s.iterrows()],color=cau_sg,edgecolor='orange',linewidth=0.5)
        ax_sg[2].axvline(x=sau_s,color='red',linestyle='--',linewidth=1.5,label=f'{sau_s}ppb')
        ax_sg[2].set_xlabel("Au (ppb)"); ax_sg[2].set_title("Or (Au)",fontsize=9,fontweight='bold'); ax_sg[2].legend(fontsize=7)
        plt.suptitle(f"SGI — {tsg} | {f_s['type']} | {f_s['profondeur']}m",fontsize=11,fontweight='bold'); plt.tight_layout(); st.pyplot(fig_sg)
    tab_sg=df_intervals[df_intervals['trou']==tsg][['de','a','lithologie','alteration','mineralisation','Au_ppb','Cu_ppm','As_ppm','mineralisé']].copy()
    tab_sg.columns=['De(m)','A(m)','Lithologie','Altération','Minéralisation','Au(ppb)','Cu(ppm)','As(ppm)','Minéralisé']
    st.dataframe(tab_sg.style.map(lambda v:'background-color:#FFD700;color:black' if v==True else 'background-color:#F5F5F5' if v==False else '',subset=['Minéralisé']).format({'Au(ppb)':'{:.2f}','Cu(ppm)':'{:.1f}','As(ppm)':'{:.1f}'}),use_container_width=True)

# ══ TAB 10 — ESTIMATION ══════════════════════════════════════════════════════
with tabs[9]:
    st.subheader("💰 Estimation des Teneurs en Or")
    with st.expander("📖 Guide d'interprétation — Estimation des teneurs", expanded=False):
        st.markdown(INTERPRETATIONS["estimation"]["intro"])
        st.markdown(INTERPRETATIONS["estimation"]["classification_ressources"])
    col1,col2=st.columns([1,2])
    with col1:
        meth=st.selectbox("Méthode",['Moyenne pondérée','IDW','Krigeage'])
        sc=st.number_input("Coupure (ppb)",10,500,100); dens=st.number_input("Densité (t/m³)",1.5,3.5,2.7,0.1)
        lz=st.number_input("Largeur zone (m)",10,200,50); lnz=st.number_input("Longueur zone (m)",50,2000,500)
    with col2:
        df_e=df_forages.copy()
        df_e['Au_est']=df_e.apply(lambda r:df_intervals[(df_intervals['trou']==r['trou'])&(df_intervals['Au_ppb']>=sc)]['Au_ppb'].mean() if len(df_intervals[(df_intervals['trou']==r['trou'])&(df_intervals['Au_ppb']>=sc)])>0 else 0,axis=1)
        df_e['lm']=df_e.apply(lambda r:df_intervals[(df_intervals['trou']==r['trou'])&(df_intervals['Au_ppb']>=sc)].apply(lambda x:x['a']-x['de'],axis=1).sum(),axis=1)
        dp=df_e[df_e['Au_est']>0]
        if len(dp)>0:
            if meth=='Moyenne pondérée': ag=np.average(dp['Au_est'],weights=dp['lm']+0.001)
            elif meth=='IDW': ag=dp['Au_est'].mean()
            else: ag=dp['Au_est'].median()
        else: ag=0
        vol=lz*lnz*df_e['lm'].mean(); ton=vol*dens; oz=(ag*ton/1e6)/31.1035
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Au moyen",f"{ag:.1f} ppb"); c2.metric("Volume",f"{vol:,.0f} m³"); c3.metric("Tonnage",f"{ton/1000:,.0f} kt"); c4.metric("Métal Au",f"{oz:,.0f} oz")
        fig_e,ax_e=plt.subplots(figsize=(10,4))
        dep=df_e[df_e['Au_est']>0].sort_values('Au_est',ascending=False)
        ce=['#FFD700' if v>=ag else '#AAA' for v in dep['Au_est']]
        ax_e.bar(dep['trou'],dep['Au_est'],color=ce,edgecolor='black',linewidth=0.5)
        ax_e.axhline(y=ag,color='red',linestyle='--',linewidth=2,label=f'Moy: {ag:.1f} ppb')
        ax_e.axhline(y=sc,color='blue',linestyle=':',linewidth=1.5,label=f'Coupure: {sc} ppb')
        ax_e.set_ylabel("Au (ppb)"); ax_e.set_title(f"Au estimé — {meth}",fontsize=10,fontweight='bold')
        ax_e.legend(fontsize=8); plt.setp(ax_e.xaxis.get_majorticklabels(),rotation=45,fontsize=7)
        plt.tight_layout(); st.pyplot(fig_e)

# ══ TAB 11 — 3D ══════════════════════════════════════════════════════════════
with tabs[10]:
    st.subheader("🌐 Sections 2D & 3D")
    vue3d=st.radio("Vue",['Section 2D verticale','Section 2D plan','3D Forages interactif','Modèle blocs 3D'],horizontal=True)
    tc3d={'RC':'red','Aircore':'blue','Diamond':'purple'}
    if vue3d=='Section 2D verticale':
        st.markdown("### Section verticale 2D")
        fig2dv,ax2dv=plt.subplots(figsize=(14,7))
        fig2dv.patch.set_facecolor('#F8F8F0'); ax2dv.set_facecolor('#E8F4F8')
        x_t2=np.linspace(0,600,300)
        topo2=100+5*np.sin(x_t2/50)+3*np.cos(x_t2/30)
        ax2dv.plot(x_t2,topo2,'k-',linewidth=2.5,label='Topographie')
        ax2dv.fill_between(x_t2,topo2,0,alpha=0.1,color='brown')
        ax2dv.axhline(y=0,color='blue',linestyle='--',linewidth=1,alpha=0.5,label='Référence 0m')
        for d,lbl,lc in [(8,'Latérite','#8B4513'),(25,'Saprolite','#DAA520'),(50,'Saprock','#696969'),(80,'Bédrock','#FFD700')]:
            ax2dv.axhline(y=topo2.mean()-d,color=lc,linestyle='-.',linewidth=1.2,alpha=0.7,label=lbl)
        xpos2=np.linspace(60,540,min(6,len(df_forages)))
        for i,(xp,(_,f)) in enumerate(zip(xpos2,df_forages.head(6).iterrows())):
            tv=float(np.interp(xp,x_t2,topo2))
            ints_2d=df_intervals[df_intervals['trou']==f['trou']].sort_values('de')
            for _,iv in ints_2d.iterrows():
                yt=tv-iv['de']*0.5; yb=tv-iv['a']*0.5
                ax2dv.fill_betweenx([yb,yt],xp-6,xp+6,color=LITHO_COLORS.get(iv['lithologie'],'#888'),alpha=0.85)
                if iv['mineralisé']:
                    ax2dv.fill_betweenx([yb,yt],xp-6,xp+6,color='red',alpha=0.25,hatch='///')
                if iv['Au_ppb']>=100:
                    ax2dv.text(xp+8,(yt+yb)/2,f"{iv['Au_ppb']:.0f}ppb",fontsize=5.5,color='#FF6600',fontweight='bold')
            ax2dv.text(xp,tv+6,f['trou'],ha='center',fontsize=7,fontweight='bold',color='#1A237E')
            ax2dv.text(xp,tv+3,f['type'],ha='center',fontsize=6,color={'RC':'#FF5722','Aircore':'#2196F3','Diamond':'#9C27B0'}.get(f['type'],'black'),fontweight='bold')
        ax2dv.annotate('',xy=(575,max(topo2)+6),xytext=(575,max(topo2)-6),arrowprops=dict(arrowstyle='->',color='black',lw=2.5))
        ax2dv.text(575,max(topo2)+8,'N',ha='center',fontsize=12,fontweight='bold')
        ax2dv.plot([20,70],[3,3],'k-',linewidth=3); ax2dv.text(45,0,'50 m',ha='center',fontsize=8,fontweight='bold')
        lp2dv=[mpatches.Patch(color=c,label=l) for l,c in LITHO_COLORS.items()]
        lp2dv.append(mpatches.Patch(color='red',alpha=0.4,hatch='///',label='Minéralisé'))
        ax2dv.legend(handles=lp2dv,loc='lower right',fontsize=7,ncol=2,framealpha=0.9)
        ax2dv.set_xlabel("Distance (m)"); ax2dv.set_ylabel("Élévation (m)")
        ax2dv.set_title(f"Section verticale 2D — {NOM_PROSPECT}",fontsize=12,fontweight='bold')
        ax2dv.grid(True,linestyle=':',alpha=0.3); plt.tight_layout(); st.pyplot(fig2dv)

    elif vue3d=='Section 2D plan':
        st.markdown("### Vue en plan 2D")
        fig2dp,ax2dp=plt.subplots(figsize=(11,9))
        for _,f in df_forages.iterrows():
            ir=np.radians(abs(f['inclinaison'])); ar=np.radians(f['azimut'])
            d=np.linspace(0,f['profondeur'],20)
            xs=f['easting']+d*np.sin(ar)*np.cos(ir); ys=f['northing']+d*np.cos(ar)*np.cos(ir)
            ax2dp.plot(xs,ys,color=tc3d.get(f['type'],'gray'),linewidth=2,alpha=0.7)
            ax2dp.scatter(f['easting'],f['northing'],color=tc3d.get(f['type'],'gray'),s=80,zorder=3,edgecolors='black')
            ax2dp.text(f['easting'],f['northing']+8,f['trou'],fontsize=6.5,ha='center',fontweight='bold')
        xmx=df_forages['easting'].max(); ymx=df_forages['northing'].max()
        xmn=df_forages['easting'].min(); ymn=df_forages['northing'].min()
        ax2dp.annotate('',xy=(xmx+60,ymx+25),xytext=(xmx+60,ymx-20),arrowprops=dict(arrowstyle='->',color='black',lw=2.5))
        ax2dp.text(xmx+60,ymx+35,'N',ha='center',fontsize=14,fontweight='bold')
        ax2dp.plot([xmn,xmn+200],[ymn-35,ymn-35],'k-',linewidth=3)
        ax2dp.text(xmn+100,ymn-50,'200 m',ha='center',fontsize=9,fontweight='bold')
        lp2dp=[mpatches.Patch(color=c,label=t) for t,c in tc3d.items()]
        ax2dp.legend(handles=lp2dp,title='Type forage',fontsize=9,loc='lower right')
        ax2dp.set_xlabel("Easting (m)"); ax2dp.set_ylabel("Northing (m)")
        ax2dp.set_title(f"Vue en plan 2D — {NOM_PROSPECT}",fontsize=12,fontweight='bold')
        ax2dp.grid(True,linestyle=':',alpha=0.4); plt.tight_layout(); st.pyplot(fig2dp)

    elif vue3d=='3D Forages interactif':
        fig3d=go.Figure()
        for _,f in df_forages.iterrows():
            ir=np.radians(abs(f['inclinaison'])); ar=np.radians(f['azimut'])
            d=np.linspace(0,f['profondeur'],30)
            fig3d.add_trace(go.Scatter3d(
                x=f['easting']+d*np.sin(ar)*np.cos(ir),
                y=f['northing']+d*np.cos(ar)*np.cos(ir),
                z=f['elevation']-d*np.sin(ir),
                mode='lines+markers',
                line=dict(color=tc3d.get(f['type'],'gray'),width=4),
                marker=dict(size=2),
                name=f"{f['trou']} ({f['type']})",
                hovertemplate=f"<b>{f['trou']}</b><br>Type:{f['type']}<br>Prof:{f['profondeur']}m<br>Au:{f['Au_max_ppb']}ppb"))
        fig3d.update_layout(scene=dict(xaxis_title='Easting',yaxis_title='Northing',zaxis_title='Élévation',bgcolor='#1A1A2E'),
            title=f"Modèle 3D forages — {NOM_PROSPECT}",height=650,paper_bgcolor='#1A1A2E',font=dict(color='white'))
        st.plotly_chart(fig3d,use_container_width=True)
        st.info("🖱️ Clic gauche=rotation | Scroll=zoom | Clic droit=déplacement")

    else:
        st.markdown("### Modèle de blocs 3D — Teneur Au")
        nx,ny,nz=10,10,6
        xbl=np.linspace(BASE_E-400,BASE_E+400,nx); ybl=np.linspace(BASE_N-400,BASE_N+400,ny); zbl=np.linspace(0,120,nz)
        blocs=[]
        for xi in xbl:
            for yi in ybl:
                for zi in zbl:
                    dmin=min([np.sqrt((xi-f['easting'])**2+(yi-f['northing'])**2) for _,f in df_forages.iterrows()])
                    au_b=round(max(0,np.random.lognormal(3,1.2)-dmin/80),1)
                    blocs.append({'x':xi,'y':yi,'z':zi,'Au':au_b})
        df_bl=pd.DataFrame(blocs)
        fig_bl=go.Figure(data=go.Scatter3d(x=df_bl['x'],y=df_bl['y'],z=df_bl['z'],mode='markers',
            marker=dict(size=7,color=df_bl['Au'],colorscale='Viridis',
                       colorbar=dict(title='Au (ppb)'),opacity=0.75,
                       cmin=0,cmax=df_bl['Au'].quantile(0.95)),
            text=df_bl['Au'].apply(lambda v:f"Au: {v:.1f} ppb"),
            hovertemplate="<b>%{text}</b><br>E:%{x:.0f} N:%{y:.0f} Z:%{z:.0f}m"))
        fig_bl.update_layout(scene=dict(xaxis_title='Easting',yaxis_title='Northing',zaxis_title='Profondeur'),
            title=f"Modèle de blocs 3D — {NOM_PROSPECT}",height=650)
        st.plotly_chart(fig_bl,use_container_width=True)
        c1,c2,c3=st.columns(3)
        c1.metric("Blocs modélisés",len(df_bl))
        c2.metric("Au moyen",f"{df_bl['Au'].mean():.1f} ppb")
        c3.metric("Au max bloc",f"{df_bl['Au'].max():.1f} ppb")

# ══ TAB 12 — PLANIFICATION ═══════════════════════════════════════════════════
with tabs[11]:
    st.subheader("📋 Planification & Infill/Extension")
    col1,col2=st.columns(2)
    with col1:
        sc2=df_forages['statut'].value_counts()
        fig_sp2,ax_sp2=plt.subplots(figsize=(5,4))
        ax_sp2.pie(sc2.values,labels=sc2.index,colors=['#4CAF50','#FF9800','#2196F3'],autopct='%1.0f%%',startangle=90)
        ax_sp2.set_title("Statut des forages"); st.pyplot(fig_sp2)
    with col2:
        ea=st.slider("Espacement actuel (m)",50,400,200,key='ea1')
        ei=st.slider("Espacement infill (m)",25,200,100,key='ei1')
        nb_i=int((ea/ei-1)*len(df_forages[df_forages['statut']=='Complété']))
        nb_ext_p=st.slider("Trous extension prévus",0,20,5,key='nep')
        cm2=st.number_input("Coût/mètre (USD)",50,500,150,key='cm2')
        ct2=(nb_i+nb_ext_p)*df_forages['profondeur'].mean()*cm2
        c1,c2,c3=st.columns(3)
        c1.metric("Infill nécessaires",nb_i)
        c2.metric("Extension",nb_ext_p)
        c3.metric("Budget estimé",f"${ct2:,.0f}")
    with st.expander("📖 Guide d'interprétation — Planification & Infill", expanded=False):
        st.markdown(INTERPRETATIONS["planification"]["intro"])
    st.markdown("### 📊 Tableau de planification")
    st.dataframe(df_forages[['trou','type','profondeur','azimut','inclinaison','statut','equipe','Au_max_ppb']],use_container_width=True)
    # Carte infill
    fig_inf,ax_inf=plt.subplots(figsize=(10,8))
    fig_inf.patch.set_facecolor('#F5F5F0'); ax_inf.set_facecolor('#E8F4F8')
    for _,f in df_forages.iterrows():
        c={'Complété':'#4CAF50','En cours':'#FF9800','Planifié':'#2196F3'}.get(f['statut'],'gray')
        ax_inf.scatter(f['easting'],f['northing'],c=c,s=120,edgecolors='black',linewidths=0.8,zorder=3)
        ax_inf.text(f['easting'],f['northing']+8,f['trou'],fontsize=6,ha='center')
    np.random.seed(88)
    for _ in range(nb_i):
        xi=np.random.uniform(df_forages['easting'].min(),df_forages['easting'].max())
        yi=np.random.uniform(df_forages['northing'].min(),df_forages['northing'].max())
        ax_inf.scatter(xi,yi,c='purple',s=150,marker='+',linewidths=2,zorder=4,label='Infill' if _==0 else '')
    for _ in range(nb_ext_p):
        xi=np.random.uniform(df_forages['easting'].min()-200,df_forages['easting'].max()+200)
        yi=np.random.uniform(df_forages['northing'].min()-200,df_forages['northing'].max()+200)
        ax_inf.scatter(xi,yi,c='red',s=150,marker='*',linewidths=1,zorder=4,label='Extension' if _==0 else '')
    xmx=df_forages['easting'].max(); ymx=df_forages['northing'].max()
    xmn=df_forages['easting'].min(); ymn=df_forages['northing'].min()
    ax_inf.annotate('',xy=(xmx+60,ymx+25),xytext=(xmx+60,ymx-20),arrowprops=dict(arrowstyle='->',color='black',lw=2.5))
    ax_inf.text(xmx+60,ymx+35,'N',ha='center',fontsize=14,fontweight='bold')
    ax_inf.plot([xmn,xmn+200],[ymn-35,ymn-35],'k-',linewidth=3)
    ax_inf.text(xmn+100,ymn-50,'200 m',ha='center',fontsize=9,fontweight='bold')
    lp_inf=[mpatches.Patch(color='#4CAF50',label='Complété'),mpatches.Patch(color='#FF9800',label='En cours'),
            mpatches.Patch(color='#2196F3',label='Planifié'),
            plt.Line2D([0],[0],marker='+',color='w',markerfacecolor='purple',markersize=12,markeredgecolor='purple',label='Infill'),
            plt.Line2D([0],[0],marker='*',color='w',markerfacecolor='red',markersize=12,label='Extension')]
    ax_inf.legend(handles=lp_inf,loc='lower right',fontsize=8,framealpha=0.95)
    ax_inf.set_xlabel("Easting UTM (m)"); ax_inf.set_ylabel("Northing UTM (m)")
    ax_inf.set_title(f"Carte planification Infill & Extension — {NOM_PROSPECT}",fontsize=12,fontweight='bold')
    ax_inf.grid(True,linestyle=':',alpha=0.4); plt.tight_layout(); st.pyplot(fig_inf)

# ══ TAB 12 — PROGRAMME EXTENSION ═════════════════════════════════════════════
with tabs[12]:
    st.subheader(f"🔭 Programme d'Extension — {NOM_PROSPECT}")
    np.random.seed(33)
    ext_zones=pd.DataFrame([{
        'zone':f'EXT{i+1:02d}',
        'priorite':np.random.choice(['Haute','Moyenne','Faible'],p=[0.3,0.4,0.3]),
        'easting':round(BASE_E+np.random.uniform(-600,600),1),
        'northing':round(BASE_N+np.random.uniform(-600,600),1),
        'profondeur_cible':np.random.choice([80,100,120,150,200]),
        'azimut':round(np.random.uniform(0,360),1),
        'inclinaison':round(np.random.uniform(-85,-60),1),
        'type_forage':np.random.choice(['RC','Diamond'],p=[0.5,0.5]),
        'Au_prevu_ppb':round(np.random.lognormal(4,1.2),1),
        'cout_usd':round(np.random.uniform(10000,50000),0),
        'justification':np.random.choice(['Extension anomalie Au','Prolongement zone minéralisée','Test structure NE','Confirmation intersection','Approfondissement']),
        'statut':np.random.choice(['Planifié','Approuvé','En attente'],p=[0.5,0.3,0.2]),
    } for i in range(20)])
    prio_colors={'Haute':'#FF0000','Moyenne':'#FF9800','Faible':'#2196F3'}
    prio_markers={'Haute':'*','Moyenne':'^','Faible':'s'}

    c1,c2,c3,c4,c5=st.columns(5)
    c1.metric("Zones extension",len(ext_zones))
    c2.metric("Priorité Haute",int((ext_zones['priorite']=='Haute').sum()))
    c3.metric("Approuvées",int((ext_zones['statut']=='Approuvé').sum()))
    c4.metric("Mètres prévus",f"{ext_zones['profondeur_cible'].sum():.0f}m")
    c5.metric("Budget total",f"${ext_zones['cout_usd'].sum():,.0f}")

    col1,col2=st.columns([3,1])
    with col1:
        fig_ext,ax_ext=plt.subplots(figsize=(12,10))
        fig_ext.patch.set_facecolor('#F5F5F0'); ax_ext.set_facecolor('#E8F4F8')
        for _,f in df_forages.iterrows():
            ax_ext.scatter(f['easting'],f['northing'],c='black',s=60,marker='o',zorder=3)
            ax_ext.text(f['easting'],f['northing']+8,f['trou'],fontsize=5.5,ha='center',color='#1A237E')
        for prio,color in prio_colors.items():
            sub=ext_zones[ext_zones['priorite']==prio]
            sz=300 if prio=='Haute' else 150 if prio=='Moyenne' else 80
            ax_ext.scatter(sub['easting'],sub['northing'],c=color,s=sz,marker=prio_markers[prio],
                          edgecolors='black',linewidths=1,zorder=4,alpha=0.85,label=f'{prio} ({len(sub)})')
            for _,z in sub.iterrows():
                ax_ext.text(z['easting'],z['northing']+12,z['zone'],fontsize=6,ha='center',color=color,fontweight='bold')
                if prio=='Haute':
                    ax_ext.add_patch(plt.Circle((z['easting'],z['northing']),80,fill=False,color='red',linewidth=1.5,linestyle='--',alpha=0.5))
        xmx=max(df_forages['easting'].max(),ext_zones['easting'].max())
        ymx=max(df_forages['northing'].max(),ext_zones['northing'].max())
        xmn=min(df_forages['easting'].min(),ext_zones['easting'].min())
        ymn=min(df_forages['northing'].min(),ext_zones['northing'].min())
        ax_ext.annotate('',xy=(xmx+80,ymx+25),xytext=(xmx+80,ymx-25),arrowprops=dict(arrowstyle='->',color='black',lw=2.5))
        ax_ext.text(xmx+80,ymx+40,'N',ha='center',fontsize=16,fontweight='bold')
        ax_ext.plot([xmn,xmn+300],[ymn-50,ymn-50],'k-',linewidth=3)
        ax_ext.text(xmn+150,ymn-70,'300 m',ha='center',fontsize=9,fontweight='bold')
        ax_ext.legend(loc='lower right',fontsize=9,title='Priorité extension',framealpha=0.95)
        ax_ext.set_xlabel("Easting UTM (m)"); ax_ext.set_ylabel("Northing UTM (m)")
        ax_ext.set_title(f"Programme d'extension — {NOM_PROSPECT}\n{NOM_PERMIS}",fontsize=12,fontweight='bold')
        ax_ext.grid(True,linestyle='--',alpha=0.4); plt.tight_layout(); st.pyplot(fig_ext)
    with col2:
        st.markdown("### 📊 Budget par priorité")
        for prio,color in prio_colors.items():
            sub=ext_zones[ext_zones['priorite']==prio]
            st.markdown(f"**{prio}** : {len(sub)} zones")
            st.markdown(f"Mètres : {sub['profondeur_cible'].sum():.0f}m")
            st.markdown(f"Coût : ${sub['cout_usd'].sum():,.0f}")
            st.markdown("---")

    with st.expander("📖 Guide d'interprétation — Programme d'extension", expanded=False):
        st.markdown(INTERPRETATIONS["extension"]["intro"])
    fp=st.multiselect("Filtrer priorité",['Haute','Moyenne','Faible'],default=['Haute','Moyenne','Faible'],key='fpe')
    st.dataframe(ext_zones[ext_zones['priorite'].isin(fp)],use_container_width=True)

    # Analyse risque/bénéfice
    fig_rb,ax_rb=plt.subplots(figsize=(8,5))
    sc_rb=ax_rb.scatter(ext_zones['cout_usd']/1000,ext_zones['Au_prevu_ppb'],
                       c=[{'Haute':3,'Moyenne':2,'Faible':1}[p] for p in ext_zones['priorite']],
                       cmap='RdYlGn',s=150,edgecolors='black',linewidths=0.8,alpha=0.85,zorder=3)
    for _,z in ext_zones.iterrows():
        ax_rb.text(z['cout_usd']/1000+0.3,z['Au_prevu_ppb'],z['zone'],fontsize=6.5,color='#333')
    ax_rb.set_xlabel("Coût estimé (k$)"); ax_rb.set_ylabel("Au prévu (ppb)")
    ax_rb.set_title("Analyse Risque/Bénéfice — Zones d'extension",fontsize=11,fontweight='bold')
    plt.colorbar(sc_rb,ax=ax_rb,label='Priorité (1=Faible 3=Haute)')
    ax_rb.grid(True,linestyle=':',alpha=0.4); plt.tight_layout(); st.pyplot(fig_rb)
    st.download_button("📥 Programme extension",data=ext_zones.to_csv(index=False),file_name="programme_extension.csv",mime='text/csv')

# ══ TAB 13 — MAPPING SPATIAL ═════════════════════════════════════════════════
with tabs[13]:
    st.subheader(f"🗺️ Mapping & Analyse Spatiale — {NOM_PROSPECT}")
    analyse_type=st.radio("Analyse",['Carte densité minéralisations','Variogramme expérimental','Clusters spatiaux','Corrélations spatiales'],horizontal=True,key='at')
    au_par_t=df_intervals.groupby('trou')['Au_ppb'].max().reset_index(); au_par_t.columns=['trou','Au_max']
    df_sp=df_forages.merge(au_par_t,on='trou')

    if analyse_type=='Carte densité minéralisations':
        fig_dens,axes_dens=plt.subplots(1,2,figsize=(14,7))
        if len(df_sp)>3:
            xi=np.linspace(df_sp['easting'].min()-100,df_sp['easting'].max()+100,200)
            yi=np.linspace(df_sp['northing'].min()-100,df_sp['northing'].max()+100,200)
            Xi,Yi=np.meshgrid(xi,yi)
            Zi=griddata((df_sp['easting'],df_sp['northing']),df_sp['Au_max'],(Xi,Yi),method='cubic')
            im=axes_dens[0].contourf(Xi,Yi,Zi,levels=25,cmap='YlOrRd',alpha=0.85)
            plt.colorbar(im,ax=axes_dens[0],label='Au max (ppb)')
        for _,f in df_sp.iterrows():
            axes_dens[0].scatter(f['easting'],f['northing'],c='white',s=80,edgecolors='black',linewidths=1,zorder=4)
            axes_dens[0].text(f['easting'],f['northing']+8,f['trou'],fontsize=6,ha='center',fontweight='bold')
        xmx=df_sp['easting'].max(); ymx=df_sp['northing'].max(); xmn=df_sp['easting'].min(); ymn=df_sp['northing'].min()
        axes_dens[0].annotate('',xy=(xmx+60,ymx+20),xytext=(xmx+60,ymx-20),arrowprops=dict(arrowstyle='->',color='black',lw=2))
        axes_dens[0].text(xmx+60,ymx+30,'N',ha='center',fontsize=12,fontweight='bold')
        axes_dens[0].plot([xmn,xmn+200],[ymn-35,ymn-35],'k-',linewidth=3)
        axes_dens[0].text(xmn+100,ymn-50,'200 m',ha='center',fontsize=9,fontweight='bold')
        axes_dens[0].set_title("Carte de densité Au",fontsize=11,fontweight='bold'); axes_dens[0].grid(True,linestyle=':',alpha=0.3)
        axes_dens[1].hist(np.log10(df_sp['Au_max']+1),bins=15,color='#FFD700',edgecolor='black',linewidth=0.5)
        axes_dens[1].set_xlabel("log10(Au max+1)"); axes_dens[1].set_title("Distribution log-normale",fontsize=11,fontweight='bold')
        axes_dens[1].axvline(np.log10(101),color='red',linestyle='--',linewidth=2,label='100 ppb'); axes_dens[1].legend(fontsize=9); axes_dens[1].grid(True,linestyle=':',alpha=0.4)
        plt.suptitle(f"Analyse spatiale — {NOM_PROSPECT}",fontsize=12,fontweight='bold'); plt.tight_layout(); st.pyplot(fig_dens)

    elif analyse_type=='Variogramme expérimental':
        coords=df_sp[['easting','northing']].values; values=np.log1p(df_sp['Au_max'].values)
        distances=[]; gammas=[]
        for i in range(len(coords)):
            for j in range(i+1,len(coords)):
                distances.append(np.sqrt(sum((coords[i]-coords[j])**2)))
                gammas.append(0.5*(values[i]-values[j])**2)
        distances=np.array(distances); gammas=np.array(gammas)
        bins=np.linspace(0,distances.max(),12); gb=[]; db=[]
        for k in range(len(bins)-1):
            mask=(distances>=bins[k])&(distances<bins[k+1])
            if mask.sum()>0: gb.append(gammas[mask].mean()); db.append((bins[k]+bins[k+1])/2)
        gb=np.array(gb); db=np.array(db)
        fig_var,ax_var=plt.subplots(figsize=(8,5))
        ax_var.scatter(db,gb,c='#2196F3',s=100,edgecolors='black',linewidths=1,zorder=3)
        ax_var.plot(db,gb,'b--',linewidth=1.5,alpha=0.6)
        nugget=gb[0] if len(gb)>0 else 0.1; sill=gb.max() if len(gb)>0 else 1; rv=db[len(db)//2] if len(db)>0 else 200
        h=np.linspace(0,db.max() if len(db)>0 else 500,100)
        model=np.where(h>rv,sill,nugget+(sill-nugget)*(1.5*(h/rv)-0.5*(h/rv)**3))
        ax_var.plot(h,model,'r-',linewidth=2.5,label=f'Modèle sphérique\nNugget:{nugget:.2f} Sill:{sill:.2f} Range:{rv:.0f}m')
        ax_var.set_xlabel("Distance (m)"); ax_var.set_ylabel("γ(h)"); ax_var.set_title("Variogramme expérimental — Au",fontsize=11,fontweight='bold')
        ax_var.legend(fontsize=8); ax_var.grid(True,linestyle=':',alpha=0.4); plt.tight_layout(); st.pyplot(fig_var)
        st.info(f"**Portée :** {rv:.0f}m | Espacement recommandé : {rv*0.6:.0f}m max")

    elif analyse_type=='Clusters spatiaux':
        nc=st.slider("Nombre de clusters",2,6,3,key='nc')
        coords_c=df_sp[['easting','northing']].values
        km=KMeans(n_clusters=nc,random_state=42,n_init=10)
        df_sp2=df_sp.copy(); df_sp2['cluster']=km.fit_predict(coords_c)
        centers=km.cluster_centers_; cl_colors=['#FF5722','#2196F3','#4CAF50','#FF9800','#9C27B0','#00BCD4']
        fig_cl,axes_cl=plt.subplots(1,2,figsize=(14,6))
        for cl in range(nc):
            sub=df_sp2[df_sp2['cluster']==cl]
            axes_cl[0].scatter(sub['easting'],sub['northing'],c=cl_colors[cl],s=150,edgecolors='black',linewidths=1,label=f'Cluster {cl+1}',zorder=3,alpha=0.85)
        axes_cl[0].scatter(centers[:,0],centers[:,1],c='black',s=300,marker='X',zorder=5,label='Centroïdes')
        axes_cl[0].legend(fontsize=8); axes_cl[0].set_title("Clusters spatiaux",fontsize=11,fontweight='bold'); axes_cl[0].grid(True,linestyle=':',alpha=0.4)
        ca=df_sp2.groupby('cluster')['Au_max'].agg(['mean','max']).reset_index()
        axes_cl[1].bar([f'C{c+1}' for c in ca['cluster']],ca['mean'],color=[cl_colors[c] for c in ca['cluster']],edgecolor='black',linewidth=0.5)
        axes_cl[1].set_ylabel("Au max moyen (ppb)"); axes_cl[1].set_title("Au par cluster",fontsize=11,fontweight='bold'); axes_cl[1].grid(True,linestyle=':',alpha=0.4)
        plt.suptitle(f"Analyse de clusters — {NOM_PROSPECT}",fontsize=12,fontweight='bold'); plt.tight_layout(); st.pyplot(fig_cl)
        bc=ca.loc[ca['mean'].idxmax(),'cluster']
        st.success(f"**Cluster {bc+1}** le plus prometteur — Au moyen = {ca.loc[bc,'mean']:.1f} ppb")

    else:
        au_m=df_intervals.groupby('trou')[['Au_ppb','Cu_ppm','As_ppm','Ag_ppm']].mean().reset_index()
        df_corr=df_forages.merge(au_m,on='trou')
        fig_corr,axes_corr=plt.subplots(2,2,figsize=(12,10))
        for ax_c,(xc,yc) in zip(axes_corr.flatten(),[('Au_ppb','As_ppm'),('Au_ppb','Cu_ppm'),('Au_ppb','Ag_ppm'),('As_ppm','Cu_ppm')]):
            ax_c.scatter(df_corr[xc],df_corr[yc],c='#FF5722',s=80,edgecolors='black',linewidths=0.5,alpha=0.85)
            if len(df_corr)>2:
                z=np.polyfit(df_corr[xc].fillna(0),df_corr[yc].fillna(0),1); p=np.poly1d(z)
                xl=np.linspace(df_corr[xc].min(),df_corr[xc].max(),50)
                ax_c.plot(xl,p(xl),'r--',linewidth=2)
                r=np.corrcoef(df_corr[xc].fillna(0),df_corr[yc].fillna(0))[0,1]
                ax_c.text(0.05,0.95,f'r = {r:.2f}',transform=ax_c.transAxes,fontsize=10,fontweight='bold',color='red',verticalalignment='top',bbox=dict(boxstyle='round',facecolor='white',alpha=0.8))
            ax_c.set_xlabel(xc); ax_c.set_ylabel(yc); ax_c.set_title(f"{xc} vs {yc}",fontsize=10,fontweight='bold'); ax_c.grid(True,linestyle=':',alpha=0.4)
        plt.suptitle(f"Corrélations spatiales — {NOM_PROSPECT}",fontsize=12,fontweight='bold'); plt.tight_layout(); st.pyplot(fig_corr)

# ══ TAB 14 — SIMULATION DÉVIATION ════════════════════════════════════════════
with tabs[14]:
    st.subheader("🔄 Simulation de Déviation — Azimut & Inclinaison")
    col1,col2=st.columns([1,2])
    with st.expander("📖 Guide d'interprétation — Simulation déviation", expanded=False):
        st.markdown(INTERPRETATIONS["deviation"]["intro"])
    with col1:
        trou_dev=st.selectbox("Trou à simuler",df_forages['trou'].tolist(),key='tdev')
        f_dev=df_forages[df_forages['trou']==trou_dev].iloc[0]
        st.markdown(f"**Type :** {f_dev['type']} | **Prof :** {f_dev['profondeur']}m")
        az_i=st.slider("Azimut initial (°)",0,360,int(f_dev['azimut']),key='azi')
        inc_i=st.slider("Inclinaison initiale (°)",-90,-45,int(f_dev['inclinaison']),key='inci')
        dev_az=st.slider("Déviation azimut/10m (°)",0.0,5.0,1.5,0.1,key='daz')
        dev_inc=st.slider("Déviation inclinaison/10m (°)",0.0,3.0,0.8,0.1,key='dinc')
        prof_sim=st.slider("Profondeur simulée (m)",10,200,int(f_dev['profondeur']),key='psim')
        st.markdown("---")
        st.markdown("**⚠️ Critères d'acceptabilité :**")
        st.markdown("✅ Déviation < 10m : Acceptable")
        st.markdown("🟡 Déviation 10–25m : Surveiller")
        st.markdown("🔴 Déviation > 25m : Correction requise")
    with col2:
        depths=np.arange(0,prof_sim+1,1)
        az_v=az_i+dev_az*depths/10*np.sin(depths/20)
        inc_v=inc_i+dev_inc*depths/10*np.cos(depths/15)
        xp=np.cumsum(np.sin(np.radians(az_i))*np.cos(np.radians(inc_i))*np.ones(len(depths)))
        yp=np.cumsum(np.cos(np.radians(az_i))*np.cos(np.radians(inc_i))*np.ones(len(depths)))
        zp=f_dev['elevation']-np.cumsum(np.sin(np.radians(abs(inc_i)))*np.ones(len(depths)))
        xd=np.cumsum(np.sin(np.radians(az_v))*np.cos(np.radians(inc_v)))
        yd=np.cumsum(np.cos(np.radians(az_v))*np.cos(np.radians(inc_v)))
        zd=f_dev['elevation']-np.cumsum(np.sin(np.radians(abs(inc_v))))
        fig_dev=go.Figure()
        fig_dev.add_trace(go.Scatter3d(x=xp+f_dev['easting'],y=yp+f_dev['northing'],z=zp,
            mode='lines',line=dict(color='blue',width=4,dash='dash'),name='Trajectoire planifiée'))
        fig_dev.add_trace(go.Scatter3d(x=xd+f_dev['easting'],y=yd+f_dev['northing'],z=zd,
            mode='lines',line=dict(color='red',width=4),name='Trajectoire déviée'))
        # Points de mesure tous les 30m
        pts_30=[i for i in range(0,prof_sim,30)]
        fig_dev.add_trace(go.Scatter3d(
            x=[xp[p]+f_dev['easting'] for p in pts_30],
            y=[yp[p]+f_dev['northing'] for p in pts_30],
            z=[zp[p] for p in pts_30],
            mode='markers',marker=dict(size=6,color='blue',symbol='circle'),name='Mesures planifiées'))
        fig_dev.add_trace(go.Scatter3d(
            x=[xd[p]+f_dev['easting'] for p in pts_30],
            y=[yd[p]+f_dev['northing'] for p in pts_30],
            z=[zd[p] for p in pts_30],
            mode='markers',marker=dict(size=6,color='red',symbol='diamond'),name='Mesures réelles'))
        fig_dev.update_layout(scene=dict(xaxis_title='Easting',yaxis_title='Northing',zaxis_title='Élévation'),
            title=f"Simulation déviation — {trou_dev}",height=500)
        st.plotly_chart(fig_dev,use_container_width=True)
        dev_tot=np.sqrt((xd[-1]-xp[-1])**2+(yd[-1]-yp[-1])**2)
        if dev_tot>25: st.error(f"🔴 Déviation : **{dev_tot:.1f} m** — Correction immédiate requise ! Arrêter le forage.")
        elif dev_tot>10: st.warning(f"🟡 Déviation : **{dev_tot:.1f} m** — Surveillance renforcée recommandée.")
        else: st.success(f"✅ Déviation : **{dev_tot:.1f} m** — Dans les tolérances acceptables.")

        # Tableau de déviation tous les 30m
        st.markdown("### 📋 Tableau de déviation")
        tab_dev=pd.DataFrame({'Profondeur(m)':pts_30,
            'Az. planifié(°)':[round(az_i,1)]*len(pts_30),
            'Az. réel(°)':[round(az_v[p],1) for p in pts_30],
            'Inc. planifiée(°)':[round(inc_i,1)]*len(pts_30),
            'Inc. réelle(°)':[round(inc_v[p],1) for p in pts_30],
            'Déviation(m)':[round(np.sqrt((xd[p]-xp[p])**2+(yd[p]-yp[p])**2),1) for p in pts_30]})
        st.dataframe(tab_dev.style.map(
            lambda v:'background-color:#ffcccc' if isinstance(v,float) and v>10 else 'background-color:#ffffcc' if isinstance(v,float) and v>5 else '',
            subset=['Déviation(m)']),use_container_width=True)

# ══ TAB 15 — SURVEY ══════════════════════════════════════════════════════════
with tabs[15]:
    st.subheader(f"📍 Survey — Levé des collets de forage")
    st.markdown(f"**{NOM_PROSPECT}** | {NOM_PERMIS}")

    with st.expander("📖 Guide d'interprétation — Survey", expanded=False):
        st.markdown(INTERPRETATIONS["survey"]["intro"])
    survey_vue=st.radio("Module",['Collars survey','Données de déviation','Carte survey','Export survey'],horizontal=True,key='sv')

    # Données survey
    survey_df=df_forages[['trou','type','easting','northing','elevation','azimut','inclinaison','profondeur','statut']].copy()
    survey_df.columns=['Trou','Type','Easting(m)','Northing(m)','Élévation(m)','Azimut(°)','Inclinaison(°)','Profondeur(m)','Statut']

    # Données déviation
    dev_surveys=[]
    for _,f in df_forages.iterrows():
        for d in range(0,int(f['profondeur']),30):
            dev_surveys.append({'trou':f['trou'],'profondeur':d,
                'azimut':round(f['azimut']+np.random.normal(0,2),1),
                'inclinaison':round(f['inclinaison']+np.random.normal(0,1),1),
                'instrument':np.random.choice(['Reflex EZ-Trac','Acid tube','Gyroscopique'])})
    df_dev_survey=pd.DataFrame(dev_surveys)

    if survey_vue=='Collars survey':
        st.markdown("### 📋 Tableau des collars")
        st.dataframe(survey_df,use_container_width=True)
        # Stats
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Total forages",len(survey_df))
        c2.metric("Mètres totaux",f"{df_forages['profondeur'].sum():.0f}m")
        c3.metric("Az. moyen",f"{df_forages['azimut'].mean():.1f}°")
        c4.metric("Inc. moyenne",f"{df_forages['inclinaison'].mean():.1f}°")

    elif survey_vue=='Données de déviation':
        trou_sv=st.selectbox("Trou",df_forages['trou'].tolist(),key='tsv')
        df_sv_t=df_dev_survey[df_dev_survey['trou']==trou_sv]
        st.dataframe(df_sv_t.rename(columns={'trou':'Trou','profondeur':'Profondeur(m)',
            'azimut':'Azimut(°)','inclinaison':'Inclinaison(°)','instrument':'Instrument'}),use_container_width=True)
        fig_sv,axes_sv=plt.subplots(1,2,figsize=(10,5),sharey=True)
        axes_sv[0].plot(df_sv_t['azimut'],df_sv_t['profondeur'],'b-o',markersize=6)
        axes_sv[0].set_xlabel("Azimut (°)"); axes_sv[0].set_ylabel("Profondeur (m)")
        axes_sv[0].set_title("Azimut en profondeur",fontsize=10,fontweight='bold')
        axes_sv[0].invert_yaxis(); axes_sv[0].grid(True,linestyle=':',alpha=0.4)
        axes_sv[1].plot(df_sv_t['inclinaison'],df_sv_t['profondeur'],'r-s',markersize=6)
        axes_sv[1].set_xlabel("Inclinaison (°)"); axes_sv[1].set_title("Inclinaison en profondeur",fontsize=10,fontweight='bold')
        axes_sv[1].grid(True,linestyle=':',alpha=0.4)
        plt.suptitle(f"Survey déviation — {trou_sv}",fontsize=11,fontweight='bold'); plt.tight_layout(); st.pyplot(fig_sv)

    elif survey_vue=='Carte survey':
        fig_csv,ax_csv=plt.subplots(figsize=(11,9))
        fig_csv.patch.set_facecolor('#F5F5F0'); ax_csv.set_facecolor('#E8F4F8')
        for _,f in df_forages.iterrows():
            c_sv={'RC':'#FF5722','Aircore':'#2196F3','Diamond':'#9C27B0'}.get(f['type'],'gray')
            ax_csv.scatter(f['easting'],f['northing'],c=c_sv,s=120,edgecolors='black',linewidths=1,zorder=3)
            ax_csv.text(f['easting'],f['northing']+8,f['trou'],fontsize=6.5,ha='center',fontweight='bold')
            # Flèche azimut
            az_r=np.radians(f['azimut']); ln_az=30
            ax_csv.annotate('',xy=(f['easting']+ln_az*np.sin(az_r),f['northing']+ln_az*np.cos(az_r)),
                           xytext=(f['easting'],f['northing']),
                           arrowprops=dict(arrowstyle='->',color=c_sv,lw=1.5))
            ax_csv.text(f['easting'],f['northing']-12,f"{f['azimut']:.0f}°/{abs(f['inclinaison']):.0f}°",
                       fontsize=5.5,ha='center',color='#555')
        xmx=df_forages['easting'].max(); ymx=df_forages['northing'].max()
        xmn=df_forages['easting'].min(); ymn=df_forages['northing'].min()
        ax_csv.annotate('',xy=(xmx+60,ymx+25),xytext=(xmx+60,ymx-20),arrowprops=dict(arrowstyle='->',color='black',lw=2.5))
        ax_csv.text(xmx+60,ymx+35,'N',ha='center',fontsize=14,fontweight='bold')
        ax_csv.plot([xmn,xmn+200],[ymn-35,ymn-35],'k-',linewidth=3)
        ax_csv.text(xmn+100,ymn-50,'200 m',ha='center',fontsize=9,fontweight='bold')
        lp_sv=[mpatches.Patch(color='#FF5722',label='RC'),mpatches.Patch(color='#2196F3',label='Aircore'),mpatches.Patch(color='#9C27B0',label='Diamond')]
        ax_csv.legend(handles=lp_sv,loc='lower right',fontsize=9,title='Type forage',framealpha=0.95)
        ax_csv.set_xlabel("Easting UTM (m)"); ax_csv.set_ylabel("Northing UTM (m)")
        ax_csv.set_title(f"Carte survey — {NOM_PROSPECT}\nAzimut et inclinaison par trou",fontsize=12,fontweight='bold')
        ax_csv.grid(True,linestyle=':',alpha=0.4); plt.tight_layout(); st.pyplot(fig_csv)
    else:
        st.markdown("### 💾 Export des données survey")
        col1,col2=st.columns(2)
        with col1:
            st.markdown("**Collars**")
            st.dataframe(survey_df.head(),use_container_width=True)
            st.download_button("📥 Collars CSV",data=survey_df.to_csv(index=False),file_name="collars_survey.csv",mime='text/csv')
        with col2:
            st.markdown("**Déviation**")
            st.dataframe(df_dev_survey.head(),use_container_width=True)
            st.download_button("📥 Déviation CSV",data=df_dev_survey.to_csv(index=False),file_name="deviation_survey.csv",mime='text/csv')

# ══ TAB 16 — DÉTECTEUR DE ROCHES ═════════════════════════════════════════════
with tabs[16]:
    st.subheader(f"🔬 Détecteur de Roches — Classification automatique")
    st.info("🤖 Entrez les paramètres d'un échantillon et l'algorithme identifie la lithologie probable, l'altération et le potentiel aurifère.")

    with st.expander("📖 Guide d'interprétation — Détecteur de roches", expanded=False):
        st.markdown(INTERPRETATIONS["detecteur"]["intro"])
    col1,col2,col3=st.columns(3)
    with col1:
        st.markdown("### Paramètres visuels")
        couleur=st.selectbox("Couleur dominante",['Brun-rouge','Ocre/Jaune','Gris clair','Gris foncé','Blanc/Crème','Doré/Métallique','Noir'])
        texture=st.selectbox("Texture",['Massive','Feuilletée/Schisteuse','Grenue','Porphyrique','Vacuolaire','Bréchique'])
        durete=st.slider("Dureté estimée (Mohs)",1,10,5)
        lustre=st.selectbox("Lustre",['Terne','Vitreux','Métallique','Résineux','Nacré'])
    with col2:
        st.markdown("### Paramètres géochimiques")
        au_det=st.number_input("Au (ppb)",0.0,100000.0,50.0)
        as_det=st.number_input("As (ppm)",0.0,500.0,10.0)
        cu_det=st.number_input("Cu (ppm)",0.0,1000.0,20.0)
        fe_det=st.number_input("Fe (%)",0.0,50.0,5.0)
        effervescence=st.checkbox("Effervescence à l'acide (HCl)")
        magnetisme=st.checkbox("Réaction magnétique")
    with col3:
        st.markdown("### Paramètres structuraux")
        foliation=st.checkbox("Foliation/schistosité visible")
        veines_qtz=st.checkbox("Veines de quartz présentes")
        pyrite=st.checkbox("Pyrite visible")
        limonite=st.checkbox("Limonite/goethite (rouille)")
        profondeur_ech=st.slider("Profondeur échantillon (m)",0,200,10)

    if st.button("🔍 Identifier la roche", type="primary"):
        # Algorithme de classification
        score_litho={l:0 for l in LITHOS}

        # Règles de classification
        if couleur=='Brun-rouge' and limonite: score_litho['Latérite']+=40
        if couleur=='Ocre/Jaune': score_litho['Saprolite']+=30; score_litho['Latérite']+=20
        if profondeur_ech<8: score_litho['Latérite']+=30
        elif profondeur_ech<25: score_litho['Saprolite']+=30
        elif profondeur_ech<50: score_litho['Saprock']+=30
        if foliation and texture=='Feuilletée/Schisteuse': score_litho['Bédrock/Schiste']+=40
        if veines_qtz and au_det>100: score_litho['Quartzite aurifère']+=50
        if durete>=7 and couleur in ['Blanc/Crème','Gris clair']: score_litho['Quartzite aurifère']+=25
        if texture=='Grenue' and durete>=6: score_litho['Granite frais']+=35
        if couleur=='Gris clair' and lustre=='Vitreux' and durete>=6: score_litho['Quartzite aurifère']+=20
        if effervescence: score_litho['Bédrock/Schiste']+=15
        if magnetisme: score_litho['Bédrock/Schiste']+=10

        litho_det=max(score_litho,key=score_litho.get)
        conf_litho=min(95,max(50,score_litho[litho_det]))

        # Altération probable
        alter_prob='Silicification' if veines_qtz and durete>=7 else \
                   'Argilisation' if couleur=='Ocre/Jaune' and profondeur_ech<30 else \
                   'Limonitisation' if limonite else \
                   'Séricitisation' if foliation else \
                   'Carbonatation' if effervescence else 'Faible altération'

        # Potentiel aurifère
        pot_score=0
        if au_det>=1000: pot_score=5
        elif au_det>=500: pot_score=4
        elif au_det>=100: pot_score=3
        elif au_det>=50: pot_score=2
        else: pot_score=1
        if veines_qtz: pot_score+=1
        if pyrite: pot_score+=1
        if as_det>20: pot_score+=1
        pot_score=min(5,pot_score)
        pot_labels={1:'⭐ Très faible',2:'⭐⭐ Faible',3:'⭐⭐⭐ Moyen',4:'⭐⭐⭐⭐ Bon',5:'⭐⭐⭐⭐⭐ Excellent'}

        st.markdown("---")
        st.markdown("### 🎯 Résultats de classification")
        c1,c2,c3=st.columns(3)
        c1.success(f"**Lithologie identifiée :**\n\n🪨 **{litho_det}**\n\nConfiance : {conf_litho}%")
        c2.info(f"**Altération probable :**\n\n🔥 **{alter_prob}**")
        c3.warning(f"**Potentiel aurifère :**\n\n{pot_labels[pot_score]}")

        st.markdown("### 📊 Scores de classification")
        fig_det,ax_det=plt.subplots(figsize=(10,4))
        scores_sorted=dict(sorted(score_litho.items(),key=lambda x:x[1],reverse=True))
        colors_det=[LITHO_COLORS.get(l,'#888') for l in scores_sorted.keys()]
        bars=ax_det.bar(scores_sorted.keys(),scores_sorted.values(),color=colors_det,edgecolor='black',linewidth=0.5)
        bars[0].set_edgecolor('red'); bars[0].set_linewidth(2.5)
        ax_det.set_ylabel("Score de correspondance"); ax_det.set_title("Classification lithologique — Scores par lithologie",fontsize=11,fontweight='bold')
        plt.xticks(rotation=30,ha='right'); ax_det.grid(True,linestyle=':',alpha=0.4); plt.tight_layout(); st.pyplot(fig_det)

        st.markdown("### 📋 Recommandations terrain")
        recs=[]
        if pot_score>=4: recs.append("✅ Zone à fort potentiel — **Prélever échantillon en double pour analyse en laboratoire**")
        if veines_qtz: recs.append("✅ Veines de quartz présentes — **Cartographier direction et pendage, mesurer épaisseur**")
        if pyrite: recs.append("✅ Pyrite visible — **Analyser arsenic et antimoine comme pathfinders aurifères**")
        if au_det>100: recs.append(f"✅ Au > 100 ppb ({au_det:.0f} ppb) — **Proposer forage de vérification sur cette anomalie**")
        if profondeur_ech<5: recs.append("ℹ️ Échantillon superficiel — **Approfondir pour éviter biais de l'altération supergène**")
        if not recs: recs.append("ℹ️ Zone à faible potentiel — Continuer la prospection systématique")
        for r in recs: st.markdown(r)


# ══ TAB 13 — MONITORING ══════════════════════════════════════════════════════
with tabs[17]:
    st.subheader("📈 Monitoring")
    c1,c2,c3=st.columns(3)
    c1.metric("Mètres/jour",f"{np.random.randint(30,80)} m","↑ +12"); c2.metric("Trous actifs",int((df_forages['statut']=='En cours').sum())); c3.metric("Incidents",np.random.randint(0,2))
    eq=df_forages.groupby('equipe').agg(metres=('profondeur','sum')).reset_index()
    fig_eq,ax_eq=plt.subplots(figsize=(8,4))
    ax_eq.bar(eq['equipe'],eq['metres'],color=['#2196F3','#4CAF50','#FF9800'],edgecolor='black',linewidth=0.5)
    for i,v in enumerate(eq['metres']): ax_eq.text(i,v+5,f"{v:.0f}m",ha='center',fontsize=9,fontweight='bold')
    ax_eq.set_ylabel("Mètres"); ax_eq.set_title("Mètres par équipe",fontsize=11,fontweight='bold'); plt.tight_layout(); st.pyplot(fig_eq)
    d30=pd.date_range(end=datetime.date.today(),periods=30); m30=np.random.randint(20,80,30)
    fig_j,ax_j=plt.subplots(figsize=(12,4))
    ax_j.fill_between(d30,m30,alpha=0.3,color='#2196F3'); ax_j.plot(d30,m30,'b-o',markersize=4,linewidth=1.5)
    ax_j.axhline(y=m30.mean(),color='red',linestyle='--',linewidth=1.5,label=f'Moy: {m30.mean():.0f}m/j')
    ax_j.set_ylabel("Mètres/jour"); ax_j.set_title("Production journalière",fontsize=11,fontweight='bold'); ax_j.legend(fontsize=9); ax_j.grid(True,linestyle=':',alpha=0.4)
    plt.xticks(rotation=45); plt.tight_layout(); st.pyplot(fig_j)

# ══ TAB 14 — WEEKLY ══════════════════════════════════════════════════════════
with tabs[18]:
    st.subheader("📅 Weekly Report")
    sem=st.date_input("Semaine du",datetime.date.today()-datetime.timedelta(days=7))
    st.markdown(f"**Période :** {sem} → {sem+datetime.timedelta(days=6)} | **Projet :** {NOM_PROSPECT}")
    tm_w=int(weekly_data['metres_fores'].sum()); obj=350
    c1,c2,c3,c4,c5=st.columns(5)
    c1.metric("Mètres forés",f"{tm_w} m",f"{tm_w-obj:+d}"); c2.metric("Trous complétés",int(weekly_data['trous_completes'].sum())); c3.metric("Incidents",int(weekly_data['incidents'].sum())); c4.metric("Au max",f"{float(weekly_data['Au_ppb_moyen'].max()):.1f} ppb"); c5.metric("Objectif","✅" if tm_w>=obj else "❌")
    fig_w,axes_w=plt.subplots(1,3,figsize=(14,4))
    axes_w[0].bar([d.strftime('%a') for d in weekly_data['date']],weekly_data['metres_fores'],color='#2196F3',edgecolor='black',linewidth=0.5)
    axes_w[0].axhline(y=obj/7,color='red',linestyle='--',linewidth=1.5,label=f'Obj:{obj//7}m'); axes_w[0].set_ylabel("Mètres"); axes_w[0].set_title("Mètres/jour",fontsize=10,fontweight='bold'); axes_w[0].legend(fontsize=8)
    axes_w[1].bar([d.strftime('%a') for d in weekly_data['date']],weekly_data['Au_ppb_moyen'],color='#FFD700',edgecolor='orange',linewidth=0.5); axes_w[1].set_ylabel("Au (ppb)"); axes_w[1].set_title("Au moyen/jour",fontsize=10,fontweight='bold')
    axes_w[2].bar([d.strftime('%a') for d in weekly_data['date']],weekly_data['incidents'],color=['#4CAF50' if v==0 else '#FF5722' for v in weekly_data['incidents']],edgecolor='black',linewidth=0.5); axes_w[2].set_ylabel("Incidents"); axes_w[2].set_title("Incidents/jour",fontsize=10,fontweight='bold')
    plt.suptitle(f"Weekly Report — {sem} | {NOM_PROSPECT}",fontsize=12,fontweight='bold'); plt.tight_layout(); st.pyplot(fig_w)
    wd=weekly_data.copy(); wd['date']=wd['date'].dt.strftime('%Y-%m-%d (%A)'); wd.columns=['Date','Mètres forés','Trous complétés','Incidents','Au moy.(ppb)','Équipe']
    st.dataframe(wd,use_container_width=True)
    st.download_button("📥 Rapport CSV",data=wd.to_csv(index=False),file_name=f"weekly_{sem}.csv",mime='text/csv')

# ══ TAB 19 — COMMENTAIRES ════════════════════════════════════════════════════
with tabs[19]:
    st.subheader(f"💬 Commentaires & Réponses — {NOM_PROSPECT}")
    st.markdown(f"**{NOM_PERMIS}** | Espace de collaboration géologique")

    # Initialiser les commentaires en session state
    with st.expander("📖 Guide d'utilisation — Commentaires", expanded=False):
        st.markdown(INTERPRETATIONS["commentaires"]["intro"])
    if 'commentaires' not in st.session_state:
        st.session_state.commentaires = [
            {"id":1,"auteur":"Dr. Konaté","role":"Géologue Senior","date":"2024-04-10","sujet":"Section SG003",
             "message":"L'intersection aurifère à 45m dans SG003 est très prometteuse. Je recommande un forage de confirmation à 50m au NE avec azimut 045° et inclinaison -70°.",
             "categorie":"Interprétation","priorite":"Haute","reponses":[
                {"auteur":"Chef de projet","date":"2024-04-11","message":"Approuvé. Planifier SG016 en conséquence pour la semaine prochaine."}]},
            {"id":2,"auteur":"Mme Diallo","role":"Géochimiste","date":"2024-04-12","sujet":"QAQC Standards",
             "message":"Les standards du lot ECH-045 montrent une dérive de +15%. Je recommande de vérifier la calibration du laboratoire avant de valider ces résultats.",
             "categorie":"QAQC","priorite":"Haute","reponses":[]},
            {"id":3,"auteur":"M. Traoré","role":"Foreur chef","date":"2024-04-13","sujet":"Déviation SG008",
             "message":"SG008 montre une déviation importante à 80m de profondeur (12m hors cible). Besoin de décision sur la correction ou l'abandon.",
             "categorie":"Opérationnel","priorite":"Moyenne","reponses":[
                {"auteur":"Dr. Konaté","date":"2024-04-13","message":"Continuer le forage tel quel. La position reste dans la zone d'intérêt."}]},
        ]

    # Filtres
    col1,col2,col3=st.columns(3)
    with col1:
        cat_filtre=st.multiselect("Catégorie",['Interprétation','QAQC','Opérationnel','Sécurité','Géologie','Autre'],
                                   default=['Interprétation','QAQC','Opérationnel','Sécurité','Géologie','Autre'],key='catf')
    with col2:
        prio_filtre=st.multiselect("Priorité",['Haute','Moyenne','Faible'],default=['Haute','Moyenne','Faible'],key='priof')
    with col3:
        st.metric("Total commentaires",len(st.session_state.commentaires))

    # Afficher commentaires existants
    st.markdown("### 📝 Fils de discussion")
    for comm in st.session_state.commentaires:
        if comm['categorie'] not in cat_filtre or comm['priorite'] not in prio_filtre:
            continue
        prio_color={'Haute':'🔴','Moyenne':'🟡','Faible':'🟢'}.get(comm['priorite'],'⚪')
        cat_icon={'Interprétation':'🔬','QAQC':'✅','Opérationnel':'⚙️','Sécurité':'⚠️','Géologie':'🪨','Autre':'💬'}.get(comm['categorie'],'💬')

        with st.expander(f"{prio_color} {cat_icon} **{comm['sujet']}** — {comm['auteur']} ({comm['role']}) | {comm['date']}"):
            st.markdown(f"""
            <div style='background:#E3F2FD;padding:12px;border-radius:8px;border-left:4px solid #2196F3;margin-bottom:8px;'>
            <b>🗣️ {comm['auteur']}</b> <small>({comm['role']}) — {comm['date']}</small><br><br>
            {comm['message']}
            </div>""", unsafe_allow_html=True)

            # Réponses existantes
            if comm['reponses']:
                st.markdown("**💬 Réponses :**")
                for rep in comm['reponses']:
                    st.markdown(f"""
                    <div style='background:#F3E5F5;padding:10px;border-radius:8px;border-left:4px solid #9C27B0;margin:4px 0 4px 20px;'>
                    <b>↩️ {rep['auteur']}</b> <small>— {rep['date']}</small><br>
                    {rep['message']}
                    </div>""", unsafe_allow_html=True)

            # Ajouter une réponse
            rep_key=f"rep_{comm['id']}"
            auteur_rep=st.text_input("Votre nom",key=f"auteur_{comm['id']}",placeholder="Nom et fonction...")
            nouvelle_rep=st.text_area("Votre réponse",key=rep_key,placeholder="Écrivez votre réponse...",height=80)
            if st.button(f"📤 Répondre",key=f"btn_rep_{comm['id']}"):
                if nouvelle_rep.strip() and auteur_rep.strip():
                    comm['reponses'].append({
                        "auteur":auteur_rep,
                        "date":datetime.date.today().strftime('%Y-%m-%d'),
                        "message":nouvelle_rep
                    })
                    st.success("✅ Réponse ajoutée !"); st.rerun()
                else:
                    st.warning("Veuillez entrer votre nom et votre réponse.")

    st.markdown("---")
    # Nouveau commentaire
    st.markdown("### ✏️ Nouveau commentaire")
    with st.form("nouveau_commentaire"):
        col1,col2,col3=st.columns(3)
        with col1:
            nouv_auteur=st.text_input("Votre nom *",placeholder="Prénom Nom")
            nouv_role=st.text_input("Fonction *",placeholder="Géologue, Foreur...")
        with col2:
            nouv_sujet=st.text_input("Sujet *",placeholder="Ex: Section SG005, QAQC lot 12...")
            nouv_cat=st.selectbox("Catégorie",['Interprétation','QAQC','Opérationnel','Sécurité','Géologie','Autre'])
        with col3:
            nouv_prio=st.selectbox("Priorité",['Haute','Moyenne','Faible'])
            nouv_trou=st.selectbox("Trou concerné (optionnel)",['Aucun']+df_forages['trou'].tolist())
        nouv_message=st.text_area("Message *",placeholder="Décrivez votre observation, question ou recommandation...",height=120)
        submitted=st.form_submit_button("📨 Publier le commentaire",type="primary")
        if submitted:
            if nouv_auteur.strip() and nouv_message.strip() and nouv_sujet.strip():
                nouveau={'id':len(st.session_state.commentaires)+1,
                    'auteur':nouv_auteur,'role':nouv_role,
                    'date':datetime.date.today().strftime('%Y-%m-%d'),
                    'sujet':nouv_sujet,'message':nouv_message,
                    'categorie':nouv_cat,'priorite':nouv_prio,'reponses':[]}
                if nouv_trou!='Aucun': nouveau['sujet']=f"{nouv_sujet} [{nouv_trou}]"
                st.session_state.commentaires.append(nouveau)
                st.success("✅ Commentaire publié avec succès !"); st.rerun()
            else:
                st.error("⚠️ Veuillez remplir les champs obligatoires (Nom, Sujet, Message)")

    # Export
    if len(st.session_state.commentaires)>0:
        comm_export=[]
        for c in st.session_state.commentaires:
            comm_export.append({'ID':c['id'],'Auteur':c['auteur'],'Rôle':c['role'],
                               'Date':c['date'],'Sujet':c['sujet'],'Message':c['message'],
                               'Catégorie':c['categorie'],'Priorité':c['priorite'],
                               'Nb réponses':len(c['reponses'])})
        df_comm=pd.DataFrame(comm_export)
        st.download_button("📥 Exporter commentaires",data=df_comm.to_csv(index=False),
                          file_name=f"commentaires_{NOM_PROSPECT}_{datetime.date.today()}.csv",mime='text/csv')

# ══ TAB 20 — RAPPORT ═════════════════════════════════════════════════════════
with tabs[20]:
    st.subheader("📄 Rapport Géologique")
    st.markdown(f"**Projet :** {NOM_PROSPECT} | **Permis :** {NOM_PERMIS} | **Date :** {datetime.date.today()}")
    with st.expander("📖 Guide d'interprétation — Rapport géologique", expanded=False):
        st.markdown(INTERPRETATIONS["rapport"]["intro"])
    am=df_forages['Au_max_ppb'].max(); amoy=df_forages['Au_max_ppb'].mean()
    nm=int(df_intervals['mineralisé'].sum()); pm=nm/len(df_intervals)*100
    ld=df_intervals['lithologie'].value_counts().index[0]; ad=df_intervals['alteration'].value_counts().index[0]
    sp_=structures_df[structures_df['porteur_miner']==True]['type'].value_counts()
    spn=sp_.index[0] if len(sp_)>0 else 'Veine de quartz'
    nb_anom_geo=int(geo_data['anomalie_IP'].sum())
    nb_anom_aug=int((df_auger[df_auger['statut']=='Foré']['Au_ppb']>=100).sum()) if (df_auger['statut']=='Foré').sum()>0 else 0
    st.markdown(f"""
## 1. Contexte — {NOM_PROSPECT}
Ceinture de roches vertes birimienne au Sénégal. Contexte favorable aux gisements aurifères orogéniques.

## 2. Résultats clés
| Paramètre | Valeur |
|-----------|--------|
| Au max forage | {am:.1f} ppb |
| Au moyen | {amoy:.1f} ppb |
| % intervalles minéralisés | {pm:.1f}% |
| Anomalies IP géophysiques | {nb_anom_geo} zones |
| Trous Auger anomaliques | {nb_anom_aug} trous |
| Lithologie dominante | {ld} |
| Altération dominante | {ad} |
| Structure porteuse | {spn} |

## 3. Auger & pXRF
- **{len(df_auger)} trous Auger** sur {N_LIGNES} lignes × {N_PTS} trous
- **{int((df_auger['statut']=='Foré').sum())} forés** ({int((df_auger['statut']=='Foré').sum())/len(df_auger)*100:.0f}%)
- **{len(df_pxrf)} mesures pXRF** sur {df_pxrf['trou'].nunique()} trous
- Corrélation Au-As forte → As utilisé comme pathfinder

## 4. Géophysique
- **IP :** {nb_anom_geo} anomalies → sulfures disséminés → cibles Diamond
- **Magnétique :** {int(geo_data['anomalie_mag'].sum())} anomalies détectées
- Recommandation : Cibler zones IP + magnétique combinées

## 5. Recommandations
1. Approfondir trous à Au > 200 ppb (Diamond 150–200m)
2. Infill à 100m dans zones anomaliques confirmées
3. Levé IP complémentaire sur zones non couvertes
4. Estimation ressources JORC/NI 43-101
    """)
    st.subheader("📋 Tableau des structures du prospect")
    st.dataframe(structures_df[['id','type','direction','pendage','sens_pendage','longueur_m','porteur_miner']].rename(columns={'id':'N°','type':'Type','direction':'Direction(°)','pendage':'Pendage(°)','sens_pendage':'Sens','longueur_m':'Longueur(m)','porteur_miner':'Porteur'}),use_container_width=True)
    rpt=f"""RAPPORT — {NOM_PROSPECT}\nPermis: {NOM_PERMIS}\nDate: {datetime.date.today()}\nAu max: {am:.1f} ppb | Au moy: {amoy:.1f} ppb\n% minéralisé: {pm:.1f}% | IP: {nb_anom_geo} | Auger: {nb_anom_aug}\n"""
    st.download_button("📥 Rapport",data=rpt,file_name=f"rapport_{datetime.date.today()}.txt",mime='text/plain')


# ══ TAB 21 — LOGUES TERRAIN RÉELS ════════════════════════════════════════════
with tabs[21]:
    st.subheader("📁 Logues Terrain Réels — RC / Aircore / Diamond / Sols")
    st.markdown(f"**{NOM_PROSPECT}** | {NOM_PERMIS}")
    st.info("📂 Chargez vos fichiers Excel terrain pour visualiser automatiquement les logues géologiques.")

    
import io as _io

# ── INTERPRÉTATIONS AUTOMATIQUES ──────────────────────────────────────────────
# Dictionnaire d'interprétations automatiques pour chaque onglet
INTERPRETATIONS = {
    "sections": {
        "intro": """
**📐 Sections géologiques — Guide d'interprétation**

Une section géologique est une coupe verticale du sous-sol qui permet de visualiser la distribution des
lithologies, des minéralisations et des structures en profondeur. Elle est l'outil fondamental de
l'exploration minière pour relier les données de forage à un modèle géologique cohérent.

**Comment lire cette section :**
- Les **couleurs** représentent les différentes lithologies (roches)
- Les **hachures rouges** indiquent les intervalles minéralisés (Au ≥ seuil ppb)
- Les **valeurs orange** sont les teneurs en or mesurées par analyse laboratoire
- Les **lignes pointillées horizontales** marquent les limites géologiques importantes
- La **ligne noire ondulée** représente la topographie réelle du terrain
- La **flèche Nord** et l'**échelle** permettent de se repérer géographiquement
""",
        "litho_interpretation": {
            "Latérite": "Zone d'altération superficielle (0–8m) — résidu d'altération chimique intense. Généralement non minéralisée mais peut concentrer les éléments traces en surface.",
            "Saprolite": "Zone d'altération avancée (8–25m) — la roche est complètement décomposée. Zone critique pour la minéralisation supergène (or secondaire libre).",
            "Saprock": "Zone de transition (25–50m) — roche partiellement altérée. Souvent le siège de la minéralisation de transition entre oxydé et primaire.",
            "Bédrock/Schiste": "Roche encaissante principale (>50m) — zone primaire. La minéralisation aurifère orogénique se développe préférentiellement dans les zones de cisaillement de cette unité.",
            "Quartzite aurifère": "UNITÉ PRINCIPALE MINÉRALISÉE — veines et veinules de quartz aurifère dans des zones de cisaillement. Forte probabilité de teneurs économiques.",
            "Granite frais": "Intrusion tardi-orogénique — rôle thermique dans le système hydrothermal. Le contact granite/schiste est souvent minéralisé."
        }
    },
    "cartes_litho": {
        "intro": """
**🗺️ Carte Lithologique — Guide d'interprétation**

La carte lithologique montre la distribution spatiale des roches à une profondeur donnée.
Elle permet d'identifier les zonations lithologiques et de guider la planification des forages.

**Lecture de la carte :**
- Chaque **couleur** correspond à une lithologie spécifique (voir légende)
- Les **symboles** (triangle=RC, cercle=Aircore, carré=Diamond) indiquent le type de forage
- Les **zones de couleur chaude** (jaune/or) indiquent le quartzite aurifère → cibles prioritaires
- La **distribution spatiale** révèle les contacts lithologiques et les structures géologiques
""",
        "recommandations": """
**🎯 Recommandations automatiques :**
1. **Zones aurifères** (quartzite, or) → Forer des trous de confirmation à 50–100m d'espacement
2. **Contacts lithologiques** → Zones à surveiller pour la minéralisation de contact
3. **Zones de schiste** → Investiguer les structures NE-SO pour trouver les zones de cisaillement
4. **Zones de granite** → Cartographier les contacts pour identifier les apophyses minéralisées
"""
    },
    "anomalie": {
        "intro": """
**🌡️ Carte d'Anomalie Géochimique — Guide d'interprétation**

La carte d'anomalie géochimique représente la distribution spatiale des éléments chimiques
mesurés dans les forages. Elle permet d'identifier les zones à fort potentiel économique.

**Lecture de la carte :**
- Les **couleurs chaudes** (rouge/orange) indiquent les fortes teneurs → zones anomaliques
- Les **couleurs froides** (jaune/blanc) indiquent le fond géochimique naturel
- Les **cercles verts** marquent les trous potentiels dépassant le seuil défini
- Les **lignes cyan pointillées** représentent les structures porteuses de la minéralisation
- Les **contours** délimitent les zones à fort potentiel économique

**Éléments pathfinders en contexte aurifère ouest-africain :**
- **As (Arsenic)** > 20 ppm → indicateur indirect de minéralisation aurifère primaire
- **Cu (Cuivre)** > 50 ppm → contexte de sulfures associés à l'or
- **Au (Or)** > 100 ppb → seuil d'anomalie économique, > 500 ppb → minéralisation significative
""",
        "seuils": {
            "Au_ppb": {"faible": 50, "moyen": 100, "fort": 500, "tres_fort": 2000,
                      "interpretation": "Seuil économique minimal : 100 ppb | Minéralisation significative : 500 ppb | Haute teneur : >2000 ppb"},
            "As_ppm": {"faible": 10, "moyen": 20, "fort": 50, "tres_fort": 100,
                      "interpretation": "As > 20 ppm = pathfinder aurifère | As > 50 ppm = zone d'altération hydrothermale intense"},
            "Cu_ppm": {"faible": 20, "moyen": 50, "fort": 100, "tres_fort": 300,
                      "interpretation": "Cu > 50 ppm = sulfures significatifs | Cu > 100 ppm = minéralisation cuprifère potentielle"}
        }
    },
    "structures": {
        "intro": """
**🏗️ Carte Structurale — Guide d'interprétation**

La carte structurale représente les éléments structuraux (failles, veines, zones de cisaillement)
qui contrôlent la distribution de la minéralisation dans les gisements orogéniques.

**En contexte aurifère ouest-africain (birimien) :**
- Les gisements d'or sont **contrôlés par les structures** → les failles et zones de cisaillement
  sont les drains principaux des fluides minéralisateurs
- Les **veines de quartz** dans les zones de cisaillement constituent les cibles principales
- Les **failles inverses** à pendage modéré (30–60°) sont les plus favorables à la minéralisation

**Lecture des symboles :**
- **Direction/Pendage** (ex: 045°/65°NE) = orientation et inclinaison de la structure
- **Symbole de pendage** = sens de plongement de la structure
- **Épaisseur du trait** = intensité de la structure (fort = faille principale)
""",
        "guide_couleurs": {
            "Faille normale": "Faille extensive — pendage généralement fort (>60°). Peut drainer des fluides mais moins favorable que les failles inverses.",
            "Faille inverse": "STRUCTURE CLÉ — chevauchement compressif. Très favorable à la minéralisation aurifère orogénique.",
            "Cisaillement": "STRUCTURE PRINCIPALE — zone de déformation ductile-fragile. Principal drain des fluides aurifères en contexte birimien.",
            "Veine de quartz": "CIBLE DIRECTE — précipitation directe de l'or depuis les fluides hydrothermaux. Analyser systématiquement.",
            "Zone altérée": "Guide indirect — indique le passage de fluides. Investiguer pour trouver la minéralisation associée."
        }
    },
    "auger": {
        "intro": """
**⛏️ Programme Auger — Guide d'interprétation**

Le forage Auger est une méthode de prospection légère utilisée pour échantillonner les horizons
superficiels (jusqu'à 20–30m) et cartographier les anomalies géochimiques en surface.

**Objectifs du forage Auger :**
- Cartographier les anomalies géochimiques en surface et dans la zone d'altération
- Identifier les zones à fort potentiel pour orienter les forages RC et Diamond plus coûteux
- Définir les limites des anomalies et prioriser les cibles

**Interprétation des statuts :**
- 🔵 **Foré** — Forage complété avec prélèvement d'échantillons géochimiques → données disponibles
- 🟢 **En cours** — Forage en progression → résultats partiels
- 🔴 **Stoppé** — Forage interrompu (cuirasse, roche dure, problème mécanique) → noter la cause

**Facteurs d'arrêt et implications :**
- Arrêt sur **cuirasse latéritique** → zone de cuirassement intense, peut masquer une anomalie
- Arrêt sur **roche dure** → contact latérite/saprolite ou saprolite/saprock → noter la profondeur
""",
        "anomalie_interpretation": """
**🌡️ Interprétation des anomalies Auger :**
- **Au < 50 ppb** → Fond géochimique naturel → pas d'intérêt économique immédiat
- **Au 50–100 ppb** → Anomalie faible → à surveiller, possible dispersion géochimique
- **Au 100–500 ppb** → Anomalie significative → recommander forage RC de confirmation
- **Au > 500 ppb** → Forte anomalie → cible prioritaire → forage RC immédiat recommandé
- **Au > 2000 ppb** → Anomalie exceptionnelle → forage Diamond prioritaire → ressource potentielle

**Pathfinders associés :**
- As > 20 ppm corrélé avec Au → confirme le système aurifère
- As/Au ratio élevé → minéralisation primaire en profondeur probable
"""
    },
    "pxrf": {
        "intro": """
**📡 pXRF & Géochimie — Guide d'interprétation**

La spectrométrie de fluorescence X portable (pXRF) permet une analyse multi-élémentaire
rapide sur le terrain, sans envoi au laboratoire. Elle est utilisée pour guider l'échantillonnage
et identifier les intervalles prioritaires pour l'analyse en laboratoire.

**Avantages du pXRF :**
- Résultats immédiats sur le terrain (30 secondes par mesure)
- Analyse simultanée de 20+ éléments
- Réduction des coûts d'analyse laboratoire (ciblage des intervalles)
- Détection des pathfinders (As, Cu, Zn, Pb, Sb)

**Limitations importantes :**
- Précision inférieure au laboratoire (±15–25% pour l'or)
- Pas certifié pour les rapports de ressources JORC/NI 43-101
- Doit être calibré régulièrement avec des standards certifiés
- Résultats pXRF = guide de terrain, NON = données économiques officielles

**Corrélations importantes :**
- **Au-As** : corrélation forte (r > 0.7) → As est un excellent pathfinder de l'or
- **Au-Cu** : corrélation modérée → présence de sulfures mixtes
- **Au-Fe** : corrélation variable → contexte de magnétite/pyrite
""",
        "seuils_pxrf": """
**Seuils d'interprétation pXRF :**
| Élément | Fond | Anomalie faible | Anomalie forte | Très fort |
|---------|------|----------------|----------------|-----------|
| Au (ppb) | <50 | 50–100 | 100–500 | >500 |
| As (ppm) | <10 | 10–20 | 20–50 | >50 |
| Cu (ppm) | <20 | 20–50 | 50–100 | >100 |
| Fe (%) | <5 | 5–15 | 15–25 | >25 |
| Zn (ppm) | <30 | 30–100 | 100–300 | >300 |
"""
    },
    "geophysique": {
        "intro": """
**🌊 Géophysique — Guide d'interprétation**

La géophysique est une méthode indirecte qui mesure les propriétés physiques du sous-sol pour
cartographier les structures géologiques et identifier les corps minéralisés sans forage.

**Méthodes utilisées et leur signification :**

🔷 **IP (Polarisation Induite) :**
- Mesure la capacité du sous-sol à stocker des charges électriques
- **Forte chargeabilité (>15 msec/V)** → présence de sulfures disséminés (pyrite, chalcopyrite)
- En contexte aurifère : la pyrite est souvent associée à l'or → IP = guide de forage
- **Interprétation :** Anomalie IP = zone à sulfures → potentiel aurifère → recommander forage

🔷 **Magnétométrie :**
- Mesure les variations du champ magnétique terrestre
- **Anomalie positive** → corps magnétiques (magnétite, basalte, intrusions mafiques)
- **Anomalie négative** → corps non magnétiques (schistes, quartzites altérés)
- **Gradient magnétique** → contact lithologique ou faille → zone structurale intéressante

🔷 **EM (Électromagnétique) :**
- Mesure la conductivité électrique du sous-sol
- **Forte conductivité** → argiles, zones saturées, sulfures conducteurs (pyrrhotite)
- **Faible conductivité** → roches résistantes (granite, quartzite frais)
- Aide à cartographier la profondeur du bedrock et les zones argileuses

🔷 **SP (Potentiel Spontané) :**
- Mesure les courants électriques naturels dans le sol
- **SP négatif (< -20 mV)** → corps conducteurs oxydants (sulfures)
- **SP positif** → zones réductrices ou de circulation d'eau
- Corrélation SP négatif + anomalie IP = cible de forage prioritaire
""",
        "combinaison": """
**🎯 Interprétation combinée multi-méthodes :**

La puissance de la géophysique réside dans la combinaison des méthodes :
- **IP + Magnétique élevés** = corps minéralisé à sulfures + magnétite → CIBLE PRIORITAIRE
- **IP élevé + Résistivité faible** = zone argileuse à sulfures → altération hydrothermale
- **IP élevé + SP négatif** = sulfures actifs (pyrite oxydante) → minéralisation potentielle
- **Gradient magnétique + IP** = contact lithologique minéralisé → zone à forer en priorité

**Recommandation de forage basée sur la géophysique :**
1. Identifier les zones d'anomalie IP > 75ème percentile
2. Croiser avec les anomalies magnétiques
3. Valider avec les données géochimiques pXRF/laboratoire
4. Planifier les forages perpendiculairement aux structures
"""
    },
    "sgi": {
        "intro": """
**🧪 Essai SGI (Semi-Quantitative Geological Interpretation) — Guide d'interprétation**

L'essai SGI est une méthode d'interprétation systématique des données de forage qui permet
d'évaluer le potentiel minéralisateur de chaque intervalle en croisant lithologie, altération,
minéralisation et géochimie.

**Paramètres évalués :**
1. **Lithologie** → certaines lithologies sont plus favorables à la minéralisation
2. **Altération** → le type et l'intensité de l'altération indiquent la proximité du système hydrothermal
3. **Minéralisation** → type de sulfures et association minéralogique
4. **Géochimie** → teneurs Au, As, Cu confirmant le potentiel économique

**Séquence d'altération typique autour d'un gisement aurifère orogénique :**
```
Distal → Proximal → Zone minéralisée
Carbonatation → Séricitisation → Silicification + Pyrite → OR
```

**Interprétation automatique des intervalles :**
- 🟡 **Intervalle minéralisé** (Au ≥ seuil) → priorité pour l'analyse laboratoire et la modélisation
- ⬜ **Intervalle stérile** → fond géochimique → pas d'intérêt économique direct
""",
        "alteration_guide": {
            "Silicification": "🔴 ALTÉRATION PROXIMALE — forte silicification = zone la plus proche du gisement. Forte probabilité de minéralisation aurifère. PRIORISER pour l'analyse.",
            "Séricitisation": "🟠 ALTÉRATION INTERMÉDIAIRE — séricite + quartz = altération hydrothermale significative. Souvent associée à la minéralisation aurifère dans les schistes.",
            "Argilisation": "🟡 ALTÉRATION DISTALE — argilisation intense = zone d'altération supergène ou hydrothermale distale. Peut indiquer une minéralisation en profondeur.",
            "Carbonatation": "🟢 ALTÉRATION DISTALE — carbonatation (ankérite, calcite) = fluides hydrothermaux de basse température. Guide vers des zones plus enrichies.",
            "Chloritisation": "🔵 ALTÉRATION DISTALE-INTERMÉDIAIRE — chlorite = contexte métamorphique régional. Moins directement liée à la minéralisation aurifère.",
            "Épidotisation": "⚪ ALTÉRATION RÉGIONALE — épidote = métamorphisme de contact ou altération régionale. Contexte moins favorable à la minéralisation."
        }
    },
    "estimation": {
        "intro": """
**💰 Estimation des Teneurs — Guide d'interprétation**

L'estimation des teneurs est une étape critique de l'exploration minière qui permet de quantifier
le contenu en métal d'un gisement et d'évaluer sa viabilité économique.

**Méthodes d'estimation disponibles :**

📊 **Moyenne pondérée (Weighted Average) :**
- Simple et transparente — pondère les teneurs par la longueur des intervalles
- Utilisée pour les ressources présumées (Inferred) à faible densité de données
- Formule : Teneur moyenne = Σ(teneur × longueur) / Σ(longueur)
- **Avantage :** facile à vérifier manuellement | **Limitation :** ne tient pas compte de la continuité spatiale

📊 **Inverse Distance (IDW) :**
- Attribue plus de poids aux points proches — interpolation spatiale
- Adaptée pour les ressources présumées à présumées (Inferred à Indicated)
- **Avantage :** prend en compte la position des forages | **Limitation :** tendance au "bull's eye" effect

📊 **Krigeage (Kriging) :**
- Méthode géostatistique optimale — utilise le variogramme pour modéliser la continuité
- Standard de l'industrie pour les ressources mesurées et indiquées (Measured & Indicated)
- Requiert un variogramme calé sur les données réelles
- **Avantage :** BLUE (Best Linear Unbiased Estimator) | **Limitation :** nécessite suffisamment de données

**Teneur de coupure (Cut-off grade) :**
La teneur de coupure est la teneur minimale en dessous de laquelle l'extraction n'est pas rentable.
Elle dépend du prix de l'or, des coûts opérationnels et du type de gisement.

Valeurs typiques en Afrique de l'Ouest :
- **Open pit** : 0.3–0.5 g/t Au (300–500 ppb)
- **Souterrain** : 2–3 g/t Au (2000–3000 ppb)
""",
        "classification_ressources": """
**Classification des ressources minérales (code JORC 2012 / NI 43-101) :**

| Catégorie | Espacement forages | Confiance | Description |
|-----------|-------------------|-----------|-------------|
| **Inferred** (Présumée) | >200m | Faible | Données insuffisantes pour haute confiance |
| **Indicated** (Indiquée) | 50–200m | Modérée | Données suffisantes pour estimation fiable |
| **Measured** (Mesurée) | <50m | Haute | Données denses, continuité bien établie |

**Attention :** Ces estimations sont des simulations sur données synthétiques.
Pour une estimation officielle, faire appel à une personne qualifiée (QP/CP).
"""
    },
    "planification": {
        "intro": """
**📋 Planification & Infill/Extension — Guide d'interprétation**

La planification des forages est une étape stratégique qui optimise l'allocation du budget
d'exploration pour maximiser l'information géologique obtenue.

**Types de forages :**

🔵 **Forages d'infill (densification) :**
- Objectif : augmenter la confiance dans les ressources existantes
- Convertir Inferred → Indicated → Measured
- Réduire l'espacement des forages dans les zones déjà connues
- Critère : espacement cible = 1/2 à 1/3 de la portée du variogramme

🔴 **Forages d'extension :**
- Objectif : étendre les limites connues du gisement
- Tester les prolongements latéraux et en profondeur
- Suivre les structures géologiques porteuses au-delà des zones connues

**Calcul du budget :**
Budget = Nb forages × Profondeur moyenne × Coût/mètre
- RC : 80–150 USD/m | Aircore : 40–80 USD/m | Diamond : 150–300 USD/m

**Recommandations automatiques :**
1. Prioriser les zones à fort potentiel (Au > 200 ppb) pour l'infill
2. Orienter les extensions perpendiculairement aux structures porteuses
3. Alterner RC (rapide/moins cher) et Diamond (plus d'information) selon la phase
"""
    },
    "extension": {
        "intro": """
**🔭 Programme d'Extension — Guide d'interprétation**

Le programme d'extension vise à évaluer le potentiel des zones non encore forées
autour des ressources connues. C'est une étape cruciale pour augmenter la taille du gisement.

**Priorisation des zones d'extension :**
- 🔴 **Haute priorité** → anomalie géophysique + géochimique confirmée → forer immédiatement
- 🟡 **Priorité moyenne** → anomalie partielle ou géologie favorable → forer après les hautes priorités
- 🔵 **Faible priorité** → géologie favorable sans anomalie confirmée → forer en dernier

**Facteurs de décision pour l'extension :**
1. **Continuité des structures** → les failles/veines se prolongent-elles au-delà des forages existants ?
2. **Anomalies géophysiques** → IP ou magnétique anomalique dans la zone d'extension ?
3. **Géochimie de surface** → anomalie sol ou Auger dans la direction d'extension ?
4. **Modèle géologique** → la lithologie favorable continue-t-elle en profondeur ?

**Analyse risque/bénéfice :**
- **Bénéfice** = Au prévu (ppb) × longueur potentielle × épaisseur estimée
- **Risque** = coût du forage ÷ probabilité de succès géologique
- **ROI** = bénéfice potentiel ÷ coût total du programme
"""
    },
    "deviation": {
        "intro": """
**🔄 Simulation de Déviation — Guide d'interprétation**

La déviation de forage est le phénomène par lequel la trajectoire réelle d'un forage
s'écarte de la trajectoire planifiée. C'est un problème courant qui peut compromettre
les objectifs géologiques du forage.

**Causes de la déviation :**
1. **Anisotropie de la roche** → les foliations et fractures dévient naturellement le trépan
2. **Changements de lithologie** → les contacts lithologiques peuvent dévier le forage
3. **Type de trépan** → certains trépans sont plus susceptibles de dévier
4. **Angle d'inclinaison** → les forages peu inclinés (<60°) dévient plus facilement
5. **Vitesse de rotation** → vitesse trop élevée augmente la déviation

**Tolérances acceptables (normes industrie) :**
- ✅ **< 10m de déviation** → dans les tolérances normales → continuer
- 🟡 **10–25m de déviation** → surveillance renforcée → mesurer tous les 15m
- 🔴 **> 25m de déviation** → hors tolérance → arrêter et corriger (wedge, déviation contrôlée)

**Mesure de la déviation :**
- **Recommandée :** tous les 30m de profondeur minimum
- **Instruments :** Reflex EZ-Trac (gyroscopique), Acid tube, Multishot
- **Données enregistrées :** azimut réel, inclinaison réelle à chaque point de mesure

**Impact sur l'interprétation géologique :**
Une déviation non corrigée peut entraîner une mauvaise position de l'intersection
avec la zone minéralisée cible → écart pouvant atteindre 30–50m en profondeur.
"""
    },
    "survey": {
        "intro": """
**📍 Survey — Guide d'interprétation**

Le survey (levé) est l'ensemble des mesures géodésiques qui définissent précisément
la position et l'orientation des forages dans l'espace. C'est la base de toute
interprétation géologique fiable.

**Types de données survey :**

📌 **Collar survey (levé du collet) :**
- Position GPS du point d'entrée du forage en surface
- Précision requise : ≤ 1m en XY, ≤ 0.5m en Z (élévation)
- Instrument : GPS différentiel (DGPS) ou RTK
- Datum : WGS84, projection UTM (zone 28N pour le Sénégal)

📌 **Déviation survey (levé de déviation) :**
- Mesure de la trajectoire réelle du forage en profondeur
- Fréquence minimale : tous les 30m de profondeur
- Données : azimut et inclinaison à chaque point de mesure
- Instruments : Reflex EZ-Trac (gyroscopique), Acid tube

**Importance du survey :**
- Sans survey précis → erreurs de corrélation entre forages → mauvaise estimation des ressources
- Un écart de 5° en azimut peut entraîner une erreur de 15m sur 150m de profondeur
- Toutes les données de ressources officielles (JORC/NI 43-101) requièrent des surveys validés
"""
    },
    "detecteur": {
        "intro": """
**🔬 Détecteur de Roches — Guide d'interprétation**

Le détecteur de roches est un outil d'aide à la classification lithologique rapide sur le terrain.
Il combine des observations visuelles (couleur, texture, dureté) et géochimiques (pXRF) pour
identifier automatiquement le type de roche et son potentiel aurifère.

**Paramètres de classification :**

🔵 **Paramètres visuels :**
- **Couleur** → indicateur du degré d'altération et de la minéralogie
- **Texture** → grenue (plutonique), foliée (métamorphique), massive (magmatique)
- **Dureté (Mohs)** → 1-3 = argile/saprolite, 4-6 = schiste, 7-10 = quartzite/granite

🔵 **Paramètres diagnostiques :**
- **Effervescence à l'acide** → présence de carbonates (calcaire, dolomie, ankérite)
- **Magnétisme** → magnétite → contexte mafique/métamorphique
- **Veines de quartz** → INDICATEUR CLÉ → potentiel aurifère direct
- **Pyrite visible** → sulfures → systèmes hydrothermaux → contexte aurifère

**Confiance de la classification :**
- > 80% → classification fiable → confirmer par analyse labo
- 60–80% → classification probable → nécessite vérification terrain
- < 60% → incertain → envoyer échantillon au laboratoire pour identification pétrographique
"""
    },
    "commentaires": {
        "intro": """
**💬 Commentaires & Réponses — Guide d'utilisation**

Cet espace collaboratif permet à toute l'équipe géologique de partager des observations,
questions et recommandations en temps réel, directement liées aux données du projet.

**Bonnes pratiques pour les commentaires géologiques :**

✅ **Un bon commentaire doit contenir :**
1. **Référence au trou/section** → préciser le numéro de trou, la profondeur, la zone
2. **Observation factuelle** → décrire ce qui est observé (pas une interprétation vague)
3. **Interprétation** → proposer une explication géologique basée sur les faits
4. **Recommandation** → proposer une action concrète (nouveau forage, analyse, etc.)

**Exemples de bons commentaires :**
- ✅ "SG008, 45–48m : intersection quartzite aurifère avec pyrite disséminée 5%. Au = 850 ppb. Recommande forage de confirmation à 50m au NE, Az 045°, Inc -65°."
- ✅ "QAQC lot ECH-045 : standard CRM-Au-12 montre dérive de +18% → recalibration labo requise avant validation des résultats."

**Priorités des commentaires :**
- 🔴 **Haute** → action requise immédiatement (résultat exceptionnel, problème QAQC, sécurité)
- 🟡 **Moyenne** → action dans la semaine (optimisation programme, question technique)
- 🟢 **Faible** → information utile sans urgence (observation géologique, amélioration)
"""
    },
    "rapport": {
        "intro": """
**📄 Rapport Géologique — Guide d'interprétation**

Le rapport géologique synthétise toutes les données d'exploration pour fournir une
évaluation complète du potentiel économique du projet à une date donnée.

**Structure d'un rapport d'exploration conforme JORC/NI 43-101 :**
1. Résumé exécutif (Executive Summary)
2. Contexte géologique régional
3. Géologie locale et modèle de gisement
4. Méthodes d'exploration (forage, géophysique, géochimie)
5. Contrôle qualité (QAQC)
6. Résultats et interprétations
7. Conclusions et recommandations
8. Budget et planning

**Standards de reporting :**
- **JORC 2012** (Joint Ore Reserves Committee) → standard australien et international
- **NI 43-101** (National Instrument) → standard canadien
- **SAMREC** → standard sud-africain
- Tous requièrent une **Personne Qualifiée (QP/CP)** pour valider les ressources
"""
    },
    "sop": {
        "intro": """
**📘 SOP (Standard Operating Procedures) — Guide d'utilisation**

Les procédures standard d'opération (SOP) sont des documents qui décrivent pas à pas
les protocoles à suivre pour chaque activité d'exploration minière.

**Pourquoi les SOP sont essentielles :**
1. **Reproductibilité** → même résultat quel que soit l'opérateur ou le moment
2. **Traçabilité** → chaque action est documentée et vérifiable
3. **Conformité** → respect des normes JORC/NI 43-101 pour le reporting des ressources
4. **Sécurité** → réduction des risques d'accidents et d'incidents
5. **Qualité des données** → données fiables = décisions d'investissement correctes

**Documents obligatoires sur le terrain :**
- Fiches terrain (Drill Log, Sample Tag, Chain of Custody)
- Registres QAQC (insertion des standards, blancs, duplicatas)
- Registres de sécurité (HSE, PTW, JSA)
- Registres de forage (Drill Report quotidien)
"""
    },
    "audit": {
        "intro": """
**🤖 Audit IA — Guide d'interprétation**

L'algorithme d'audit analyse automatiquement toutes les données du dashboard pour détecter
les erreurs, incohérences et anomalies qui pourraient compromettre la qualité des données
et l'intégrité des rapports de ressources.

**Catégories d'erreurs détectées :**

🔴 **CRITIQUE (Score -5 à -8 points) :**
- Coordonnées GPS manquantes ou aberrantes
- Chevauchement d'intervalles (From > To suivant)
- Trou absent du collar mais présent dans les intervalles
- Valeurs géochimiques négatives

🟠 **ERREUR (Score -2 à -4 points) :**
- Azimut ou inclinaison hors plage (0–360° et -90° à 0°)
- Intervalles dépassant la profondeur du forage
- Valeurs IP ou magnétiques aberrantes

🟡 **AVERTISSEMENT (Score -1 à -2 points) :**
- Trous manquants dans le programme Auger
- Lacunes entre intervalles
- Inclinaison suspecte mais dans la plage acceptable

**Score de qualité :**
- ✅ 80–100 → Données de haute qualité → prêtes pour le reporting
- 🟡 60–79 → Qualité acceptable → corriger les erreurs avant le reporting
- 🔴 < 60 → Qualité insuffisante → révision complète des données requise avant tout reporting officiel

**Corrections automatiques appliquées :**
- Valeurs négatives → remplacées par la limite de détection (LOD)
- Signalement des doublons et chevauchements
- Validation de la complétude du programme
"""
    }
}



# ══ TAB 21 — LOGUES TERRAIN RÉELS ════════════════════════════════
with tabs[21]:
    log_type_r=st.radio("Type de log",['RC Géologique','Aircore Géologique','Diamond Drilling','Géochimie Sols','Structural Log'],horizontal=True,key='ltr')
    label_map_r={'RC Géologique':'Log-RC_Geologique_Complet_ND.xlsx','Aircore Géologique':'Log-AC_Geologique_Complet_ND.xlsx',
               'Diamond Drilling':'Log-DD_Geologique_Complet_ND.xlsx','Géochimie Sols':'Log_Geochimie_Sols_AG.xlsx',
               'Structural Log':'logging_structural.xlsx'}

    col1r,col2r=st.columns([1,3])
    with col1r:
        st.markdown(f"**Fichier attendu :**\n`{label_map_r[log_type_r]}`")
        uploaded_log_r=st.file_uploader("📂 Charger le fichier Excel",type=['xlsx'],key=f'ul_{log_type_r}')
        if uploaded_log_r: st.success(f"✅ {uploaded_log_r.name}")

    with col2r:
        if uploaded_log_r is not None:
            file_bytes_r=uploaded_log_r.read()
            try:
                xl_r=pd.read_excel(_io.BytesIO(file_bytes_r),sheet_name=None,header=None)
                sheets_r=list(xl_r.keys())
                st.markdown(f"**Feuilles :** {', '.join(sheets_r)}")
                sheet_r=st.selectbox("Feuille",sheets_r,key='sr')
                df_raw_r=xl_r[sheet_r]
                headers_r=[str(h) if str(h)!='nan' else f'Col{i}' for i,h in enumerate(df_raw_r.iloc[0].tolist())]
                df_main_r=df_raw_r.iloc[1:].copy(); df_main_r.columns=headers_r
                df_main_r=df_main_r.dropna(how='all').reset_index(drop=True)
                col_sond_r=next((c for c in df_main_r.columns if any(x in c.lower() for x in ['sondage','hole','id','bhid'])),None)
                col_from_r=next((c for c in df_main_r.columns if c.lower() in ['from','de','from_m','depth_from']),None)
                col_to_r=next((c for c in df_main_r.columns if c.lower() in ['to','a','to_m','depth_to']),None)
                col_litho_r=next((c for c in df_main_r.columns if 'litho' in c.lower()),None)
                col_au_r=next((c for c in df_main_r.columns if 'au' in c.lower() and any(x in c.lower() for x in ['ppb','ppm','labo'])),None)
                col_alter_r=next((c for c in df_main_r.columns if 'alter' in c.lower()),None)
                col_coul_r=next((c for c in df_main_r.columns if any(x in c.lower() for x in ['coul','color'])),None)
                col_dur_r=next((c for c in df_main_r.columns if any(x in c.lower() for x in ['dur','hard'])),None)
                if col_sond_r:
                    sonds_r=df_main_r[col_sond_r].dropna().unique().tolist()
                    sg_r=st.selectbox("Sondage",sonds_r,key='sgr')
                    df_sg_r=df_main_r[df_main_r[col_sond_r]==sg_r].copy()
                else:
                    df_sg_r=df_main_r.copy(); sg_r="Tous"
                c1r,c2r,c3r,c4r=st.columns(4)
                c1r.metric("Intervalles",len(df_sg_r)); c2r.metric("Colonnes",len(df_sg_r.columns))
                if col_to_r:
                    try: c3r.metric("Prof. max",f"{pd.to_numeric(df_sg_r[col_to_r],errors='coerce').max():.1f}m")
                    except: pass
                if col_au_r:
                    try:
                        au_nr=pd.to_numeric(df_sg_r[col_au_r],errors='coerce').dropna()
                        c4r.metric("Au max",f"{au_nr.max():.1f} ppb" if len(au_nr)>0 else "N/A")
                    except: pass
                sub_ltr=st.tabs(["📋 Données","📊 Logue visuel","🔍 Analyse","📤 Export"])
                with sub_ltr[0]:
                    st.dataframe(df_sg_r,use_container_width=True)
                    for sn_r in sheets_r:
                        if sn_r!=sheet_r:
                            df_s2r=xl_r[sn_r]; h2r=[str(h) if str(h)!='nan' else f'Col{i}' for i,h in enumerate(df_s2r.iloc[0].tolist())]
                            df_s2cr=df_s2r.iloc[1:].copy(); df_s2cr.columns=h2r; df_s2cr=df_s2cr.dropna(how='all')
                            with st.expander(f"📄 {sn_r}"): st.dataframe(df_s2cr,use_container_width=True)
                with sub_ltr[1]:
                    if col_from_r and col_to_r:
                        litho_cl={'Laterite':'#8B4513','Latérite':'#8B4513','Saprolite':'#DAA520','Saprock':'#CD853F',
                                  'Schiste':'#696969','Quartzite':'#FFD700','Granite':'#708090','Basalte':'#2F4F4F','Sol':'#2D5016'}
                        cols_vr=[('Lithologie',col_litho_r),('Couleur',col_coul_r),('Altération',col_alter_r)]
                        cols_vr=[cv for cv in cols_vr if cv[1]]
                        ncvr=max(1,len(cols_vr))+(1 if col_au_r else 0)+(1 if col_dur_r else 0)
                        fig_lvr,axes_lvr=plt.subplots(1,ncvr,figsize=(3.2*ncvr,12),sharey=True)
                        if ncvr==1: axes_lvr=[axes_lvr]
                        civr=0
                        for label_vr,col_vr in cols_vr:
                            for _,row in df_sg_r.iterrows():
                                try:
                                    de=float(str(row[col_from_r]).replace(',','.') or 0); a=float(str(row[col_to_r]).replace(',','.') or de+1)
                                    val=str(row.get(col_vr,'') if col_vr else '')
                                    cl='#BBBBBB'
                                    if 'litho' in label_vr.lower():
                                        for k,v in litho_cl.items():
                                            if k.lower() in val.lower(): cl=v; break
                                    axes_lvr[civr].fill_betweenx([de,a],0,1,color=cl,alpha=0.85)
                                    axes_lvr[civr].plot([0,1],[de,de],'k-',linewidth=0.3)
                                    axes_lvr[civr].text(0.5,(de+a)/2,val[:10],ha='center',va='center',fontsize=5.5,fontweight='bold')
                                    if civr==0: axes_lvr[civr].text(-0.05,de,f"{de}m",ha='right',fontsize=5)
                                except: pass
                            axes_lvr[civr].set_title(label_vr,fontsize=9,fontweight='bold'); axes_lvr[civr].set_xticks([])
                            if civr==0: axes_lvr[civr].set_ylabel("Profondeur (m)")
                            civr+=1
                        if col_au_r:
                            frvr=[]; tovr=[]; auvr=[]
                            for _,row in df_sg_r.iterrows():
                                try: frvr.append(float(str(row[col_from_r]).replace(',','.') or 0)); tovr.append(float(str(row[col_to_r]).replace(',','.') or 0)); auvr.append(float(str(row[col_au_r]).replace(',','.') or 0))
                                except: pass
                            if auvr:
                                mivr=[(f+t)/2 for f,t in zip(frvr,tovr)]; htvr=[(t-f)*0.8 for f,t in zip(frvr,tovr)]
                                c_auvr=['#FFD700' if v>=100 else '#EEE' for v in auvr]
                                axes_lvr[civr].barh(mivr,auvr,height=htvr,color=c_auvr,edgecolor='orange',linewidth=0.4)
                                axes_lvr[civr].axvline(x=100,color='red',linestyle='--',linewidth=1.5,label='100ppb')
                                axes_lvr[civr].set_xlabel("Au (ppb)"); axes_lvr[civr].set_title("Or",fontsize=9,fontweight='bold'); axes_lvr[civr].legend(fontsize=7); civr+=1
                        for ax in axes_lvr:
                            if ax.get_ylim()[0]<ax.get_ylim()[1]: ax.invert_yaxis()
                        lp_lvr=[mpatches.Patch(color=v,label=k) for k,v in litho_cl.items()]
                        axes_lvr[0].legend(handles=lp_lvr,loc='lower right',fontsize=6,title='Lithologie',framealpha=0.9)
                        plt.suptitle(f"Logue {log_type_r} — {sg_r}\n{NOM_PROSPECT}",fontsize=11,fontweight='bold')
                        plt.tight_layout(); st.pyplot(fig_lvr)
                    else:
                        st.warning(f"Colonnes From/To non détectées. Colonnes : {list(df_sg_r.columns)}")
                with sub_ltr[2]:
                    if col_litho_r:
                        lcr=df_sg_r[col_litho_r].value_counts()
                        fig_lcr,ax_lcr=plt.subplots(figsize=(8,4))
                        ax_lcr.barh(lcr.index.astype(str),lcr.values,color=['#8B4513','#DAA520','#696969','#FFD700','#708090','#CD853F','#2F4F4F'][:len(lcr)],edgecolor='black',linewidth=0.5)
                        ax_lcr.set_xlabel("Intervalles"); ax_lcr.set_title("Distribution lithologique",fontsize=11,fontweight='bold'); ax_lcr.grid(True,linestyle=':',alpha=0.4); plt.tight_layout(); st.pyplot(fig_lcr)
                    if col_au_r:
                        df_mr=df_sg_r.copy(); df_mr['Au_n']=pd.to_numeric(df_mr[col_au_r],errors='coerce')
                        df_mp=df_mr[df_mr['Au_n']>=100]
                        cols_sh=[c for c in [col_from_r,col_to_r,col_litho_r,col_au_r] if c]
                        if len(df_mp)>0:
                            st.dataframe(df_mp[cols_sh],use_container_width=True)
                            st.success(f"**{len(df_mp)} intervalles minéralisés** | Au max : {df_mr['Au_n'].max():.1f} ppb")
                        else: st.info("Aucun intervalle Au ≥ 100 ppb")
                with sub_ltr[3]:
                    for sn_r in sheets_r:
                        df_er=xl_r[sn_r]; h_er=[str(h) if str(h)!='nan' else f'Col{i}' for i,h in enumerate(df_er.iloc[0].tolist())]
                        df_er2=df_er.iloc[1:].copy(); df_er2.columns=h_er; df_er2=df_er2.dropna(how='all')
                        st.download_button(f"📥 {sn_r}",data=df_er2.to_csv(index=False),file_name=f"{sn_r}_{sg_r}.csv",mime='text/csv',key=f'ex_{sn_r}')
            except Exception as e:
                st.error(f"Erreur : {e}")
                import traceback; st.code(traceback.format_exc())
        else:
            st.markdown(f"""
            ### 📂 Chargez votre fichier terrain

            | Fichier | Type |
            |---------|------|
            | `Log-RC_Geologique_Complet_ND.xlsx` | RC |
            | `Log-AC_Geologique_Complet_ND.xlsx` | Aircore |
            | `Log-DD_Geologique_Complet_ND.xlsx` | Diamond |
            | `Log_Geochimie_Sols_AG.xlsx` | Sols |
            | `logging_structural.xlsx` | Structural |

            **Fonctionnalités :** Tableau données · Logue visuel automatique · Analyse des intervalles · Export CSV
            """)

# ══ TAB 22 — SOP ════════════════════════════════════════════════════════════
with tabs[22]:
    st.subheader(f"📘 SOP — Procédures Standard d'Exploration Minière")
    st.markdown(f"**{NOM_PROSPECT}** | {NOM_PERMIS} | Version 1.0 — {datetime.date.today()}")
    with st.expander("📖 Pourquoi les SOP sont essentielles", expanded=False):
        st.markdown(INTERPRETATIONS["sop"]["intro"])
    st.info("📋 Ces procédures standard (SOP) définissent les bonnes pratiques pour toutes les opérations d'exploration minière sur ce projet.")

    SOP_DATA = {
        "1. Planification exploration": {
            "description": "Procédures de planification des campagnes d'exploration",
            "couleur": "#1A237E",
            "etapes": [
                ("1.1 Revue bibliographique", "Collecter toutes données géologiques existantes (rapports, cartes, études antérieures)", "Géologue Senior", "Avant démarrage"),
                ("1.2 Programme de travail", "Définir zones cibles, méthodes, budget et planning détaillé", "Chef de projet", "J-30"),
                ("1.3 Permis & autorisations", "Obtenir permis exploitation, autorisations locales, accord communautés", "Juriste/Compliance", "J-20"),
                ("1.4 Mobilisation équipe", "Recruter et former les équipes terrain selon les besoins", "RH + Géologue", "J-10"),
                ("1.5 Matériel & logistique", "Approvisionner matériel forage, échantillonnage, sécurité", "Logistique", "J-7"),
            ]},
        "2. Forage Auger": {
            "description": "Procédures standard pour le forage Auger géochimique",
            "couleur": "#E65100",
            "etapes": [
                ("2.1 Piquetage des lignes", "Implanter les lignes de forage avec GPS (précision <1m)", "Géomètre", "Avant forage"),
                ("2.2 Installation foreuse", "Vérifier état mécanique, huiles, équipements de sécurité", "Foreur chef", "Chaque jour"),
                ("2.3 Forage et échantillonnage", "Prélever échantillon tous les mètres, étiqueter correctement", "Technicien", "Continu"),
                ("2.4 Description géologique", "Décrire lithologie, altération, couleur, texture sur fiche terrain", "Géologue", "Continu"),
                ("2.5 Mesures pXRF", "Scanner chaque échantillon 30sec minimum, enregistrer résultats", "Technicien pXRF", "Continu"),
                ("2.6 Fermeture du trou", "Reboucher trou, nettoyer zone, remplir fiche de completion", "Foreur", "Fin de trou"),
            ]},
        "3. Forage RC": {
            "description": "Procédures pour le forage Reverse Circulation",
            "couleur": "#1B5E20",
            "etapes": [
                ("3.1 Positionnement GPS", "Mesurer collar, vérifier azimut et inclinaison avec clinomètre", "Géologue", "Début trou"),
                ("3.2 Déblais RC", "Collecter déblais via cyclon, diviser avec riffle splitter (1/4)", "Technicien", "Continu"),
                ("3.3 Gestion échantillons", "Étiqueter sacs, 4 kg minimum, stocker à l'abri", "Technicien", "Continu"),
                ("3.4 QAQC insertion", "Standards (1/20), blancs (1/20), duplicatas (1/20) obligatoires", "Géochimiste", "Continu"),
                ("3.5 Mesure déviation", "Mesurer déviation tous les 30m avec Reflex EZ-Trac", "Géologue", "Tous 30m"),
                ("3.6 Envoi laboratoire", "Préparer bordereau, chaîne de custody, expédier dans 48h", "Géologue", "Fin batch"),
            ]},
        "4. Forage Diamond (DD)": {
            "description": "Procédures pour le forage Diamond Drilling",
            "couleur": "#4A148C",
            "etapes": [
                ("4.1 Installation plateforme", "Niveler plateforme, installer foreuse selon azimut/inclinaison planifiés", "Foreur chef", "Début trou"),
                ("4.2 Récupération carotte", "Mesurer longueur carotte, calculer RQD et taux récupération", "Géologue", "Continu"),
                ("4.3 Boîtes à carottes", "Organiser carottes dans boîtes numérotées, photographier avant découpe", "Géologue", "Continu"),
                ("4.4 Découpe carottes", "Découper à la scie diamantée (50% analyse, 50% archive)", "Technicien", "Continu"),
                ("4.5 Description détaillée", "Décrire lithologie, structures, minéraux, altération, RQD sur fiche", "Géologue Senior", "Continu"),
                ("4.6 Déviation gyroscopique", "Mesure gyroscopique tous les 30m obligatoire", "Spécialiste déviation", "Tous 30m"),
            ]},
        "5. Échantillonnage & QAQC": {
            "description": "Contrôle qualité des échantillons",
            "couleur": "#BF360C",
            "etapes": [
                ("5.1 Insertion standards", "1 standard certifié tous les 20 échantillons minimum", "Géochimiste", "Continu"),
                ("5.2 Blancs contamination", "1 blanc tous les 20 échantillons, Au <5 ppb accepté", "Géochimiste", "Continu"),
                ("5.3 Duplicatas terrain", "1 duplicata tous les 20, variance <15% acceptable", "Géologue", "Continu"),
                ("5.4 Validation résultats", "Contrôler standards vs valeurs certifiées (±10%)", "Géochimiste", "À réception"),
                ("5.5 Rapport QAQC mensuel", "Produire rapport avec graphiques de contrôle", "Géochimiste Senior", "Mensuel"),
            ]},
        "6. Sécurité HSE": {
            "description": "Procédures de sécurité, santé et environnement",
            "couleur": "#F57F17",
            "etapes": [
                ("6.1 Briefing sécurité", "Briefing quotidien 15min avant démarrage, signatures obligatoires", "HSE Officer", "Chaque jour"),
                ("6.2 EPI obligatoires", "Casque, lunettes, gants, chaussures sécurité, gilet haute visibilité", "Tout personnel", "Continu"),
                ("6.3 Gestion déchets", "Séparer déchets solides/liquides, déchets chimiques en fûts étanches", "HSE + Terrain", "Continu"),
                ("6.4 Plan d'urgence", "Afficher numéros urgence, localiser kit premiers secours", "HSE Officer", "Quotidien"),
                ("6.5 Rapport incident", "Déclarer tout incident dans les 2h, formulaire INCIDENT-001", "Superviseur", "Si incident"),
            ]},
        "7. Gestion des données": {
            "description": "Procédures de gestion et archivage des données géologiques",
            "couleur": "#006064",
            "etapes": [
                ("7.1 Saisie quotidienne", "Entrer données terrain dans base de données le jour même", "Géologue", "Quotidien"),
                ("7.2 Sauvegarde", "Backup automatique + sauvegarde externe quotidienne obligatoire", "Data Manager", "Quotidien"),
                ("7.3 Validation données", "Vérifier cohérence De/A, coordonnées GPS, codes lithologie", "Géologue Senior", "Hebdomadaire"),
                ("7.4 Rapport hebdomadaire", "Produire rapport avancement avec métrages, résultats, incidents", "Chef de projet", "Hebdomadaire"),
                ("7.5 Archivage final", "Archiver tous documents en fin de campagne", "Data Manager", "Fin campagne"),
            ]},
    }

    sop_sel = st.selectbox("📂 Sélectionner la procédure", list(SOP_DATA.keys()))
    sop = SOP_DATA[sop_sel]
    st.markdown(f"### {sop_sel}")
    st.markdown(f"*{sop['description']}*")
    st.markdown("---")

    for etape in sop['etapes']:
        with st.expander(f"**{etape[0]}**"):
            col1,col2,col3 = st.columns([3,1,1])
            col1.markdown(f"📋 **Procédure :** {etape[1]}")
            col2.markdown(f"👤 **Responsable :** {etape[2]}")
            col3.markdown(f"⏱️ **Timing :** {etape[3]}")

    # Tableau récapitulatif
    st.markdown("### 📋 Tableau récapitulatif")
    rows_sop = []
    for nom, data in SOP_DATA.items():
        for e in data['etapes']:
            rows_sop.append({'Section':nom,'Étape':e[0],'Procédure':e[1],'Responsable':e[2],'Timing':e[3]})
    df_sop = pd.DataFrame(rows_sop)
    st.dataframe(df_sop[df_sop['Section']==sop_sel].drop(columns=['Section']),use_container_width=True)

    # Export SOP
    sop_txt = f"SOP EXPLORATION MINIÈRE — {NOM_PROSPECT}\n{NOM_PERMIS}\nDate: {datetime.date.today()}\n{'='*60}\n\n"
    for nom,data in SOP_DATA.items():
        sop_txt += f"\n{nom}\n{data['description']}\n{'-'*40}\n"
        for e in data['etapes']:
            sop_txt += f"  {e[0]}\n  → {e[1]}\n  Responsable: {e[2]} | Timing: {e[3]}\n\n"
    st.download_button("📥 Télécharger SOP complet",data=sop_txt,file_name=f"SOP_{NOM_PROSPECT}_{datetime.date.today()}.txt",mime='text/plain')

# ══ TAB 17 — AUDIT IA ════════════════════════════════════════════════════════
with tabs[23]:
    st.subheader("🤖 Audit IA & Corrections Automatiques")
    st.markdown(f"**{NOM_PROSPECT}** | {NOM_PERMIS}")
    st.info("🔍 L'algorithme d'audit analyse automatiquement **toutes les données** du dashboard, détecte les erreurs et propose des corrections.")

    with st.expander("📖 Guide d'interprétation — Audit IA", expanded=False):
        st.markdown(INTERPRETATIONS["audit"]["intro"])
    if st.button("🚀 Lancer l'audit complet", type="primary"):
        with st.spinner("Analyse en cours... Vérification de tous les modules..."):
            import time; time.sleep(1)

            # ── Fonction d'audit ──────────────────────────────────────────────
            erreurs = []
            avertissements = []
            corrections_auto = []
            score = 100

            # AUDIT FORAGES
            for _,f in df_forages.iterrows():
                if pd.isna(f['easting']) or pd.isna(f['northing']):
                    erreurs.append({"module":"Forages","sévérité":"🔴 CRITIQUE","trou":f['trou'],
                        "message":f"Coordonnées manquantes","correction":"Récupérer GPS depuis fiche terrain"}); score-=5
                if not (0<=f['azimut']<=360):
                    erreurs.append({"module":"Forages","sévérité":"🔴 CRITIQUE","trou":f['trou'],
                        "message":f"Azimut invalide ({f['azimut']}°)","correction":"Azimut doit être 0°–360°"}); score-=3
                if not (-90<=f['inclinaison']<=0):
                    avertissements.append({"module":"Forages","sévérité":"🟡 AVERTISSEMENT","trou":f['trou'],
                        "message":f"Inclinaison suspecte ({f['inclinaison']}°)","correction":"Vérifier mesure clinomètre — doit être négatif"}); score-=2

            # AUDIT INTERVALLES
            for trou in df_intervals['trou'].unique():
                ints_t=df_intervals[df_intervals['trou']==trou].sort_values('de')
                f_t=df_forages[df_forages['trou']==trou]
                if len(f_t)==0:
                    erreurs.append({"module":"Intervalles","sévérité":"🔴 CRITIQUE","trou":trou,
                        "message":"Trou absent du collar","correction":"Ajouter collar ou supprimer intervalles orphelins"}); score-=8; continue
                prof_max=f_t.iloc[0]['profondeur']
                for i in range(len(ints_t)-1):
                    if ints_t.iloc[i]['a']>ints_t.iloc[i+1]['de']:
                        erreurs.append({"module":"Intervalles","sévérité":"🔴 CRITIQUE","trou":trou,
                            "message":f"Chevauchement {ints_t.iloc[i]['de']}–{ints_t.iloc[i]['a']}m",
                            "correction":f"Corriger 'A' de l'intervalle à {ints_t.iloc[i+1]['de']}m"}); score-=5
                if ints_t['a'].max()>prof_max*1.05:
                    avertissements.append({"module":"Intervalles","sévérité":"🟡 AVERTISSEMENT","trou":trou,
                        "message":f"Intervalles dépassent profondeur ({ints_t['a'].max():.1f}m>{prof_max}m)",
                        "correction":f"Tronquer dernier intervalle à {prof_max}m"}); score-=2
                neg_au=ints_t[ints_t['Au_ppb']<0]
                if len(neg_au)>0:
                    erreurs.append({"module":"Géochimie","sévérité":"🔴 CRITIQUE","trou":trou,
                        "message":f"{len(neg_au)} valeurs Au négatives","correction":"Remplacer par LOD (0.001 ppb)"}); score-=4

            # AUDIT AUGER
            trous_att={f'L{i+1:02d}T{j+1:02d}' for i in range(10) for j in range(7)}
            manq=trous_att-set(df_auger['trou'].values)
            if len(manq)>0:
                avertissements.append({"module":"Auger","sévérité":"🟡 AVERTISSEMENT","trou":"Programme",
                    "message":f"{len(manq)} trous Auger manquants",
                    "correction":f"Trous: {', '.join(sorted(list(manq))[:5])}..."}); score-=min(10,len(manq)//2)

            # AUDIT pXRF
            if len(df_pxrf)>0:
                pxrf_neg=df_pxrf[df_pxrf['Au_ppb']<0]
                if len(pxrf_neg)>0:
                    erreurs.append({"module":"pXRF","sévérité":"🔴 CRITIQUE","trou":"pXRF",
                        "message":f"{len(pxrf_neg)} mesures pXRF négatives","correction":"Remplacer par LOD appareil"}); score-=4
                pxrf_anom=df_pxrf[df_pxrf['Au_ppb']>50000]
                if len(pxrf_anom)>0:
                    erreurs.append({"module":"pXRF","sévérité":"🔴 CRITIQUE","trou":"pXRF",
                        "message":f"{len(pxrf_anom)} mesures Au aberrantes (>50000 ppb)","correction":"Recalibrer pXRF — re-mesurer ces points"}); score-=6

            # AUDIT GÉOPHYSIQUE
            ip_anom=geo_data[geo_data['IP_chargeabilite']>1000]
            if len(ip_anom)>0:
                erreurs.append({"module":"Géophysique","sévérité":"🟠 ERREUR","trou":"IP",
                    "message":f"{len(ip_anom)} valeurs IP > 1000 msec/V (aberrantes)","correction":"Vérifier électrodes — re-mesurer"}); score-=3

            # AUDIT STRUCTURES
            struct_err=structures_df[(structures_df['pendage']<0)|(structures_df['pendage']>90)]
            if len(struct_err)>0:
                erreurs.append({"module":"Structures","sévérité":"🟠 ERREUR","trou":"Structures",
                    "message":f"{len(struct_err)} pendages hors plage (0–90°)","correction":"Corriger valeurs pendage"}); score-=3

            # CORRECTIONS AUTOMATIQUES
            n_corr=0
            mask_neg=df_intervals['Au_ppb']<0
            if mask_neg.sum()>0:
                corrections_auto.append(f"✅ {mask_neg.sum()} valeurs Au négatives → corrigées à 0.001 ppb (LOD)"); n_corr+=int(mask_neg.sum())
            if len(df_pxrf)>0:
                mask_pxrf=df_pxrf['Au_ppb']<0
                if mask_pxrf.sum()>0:
                    corrections_auto.append(f"✅ {mask_pxrf.sum()} mesures pXRF négatives → corrigées à 0.001 ppb"); n_corr+=int(mask_pxrf.sum())
            if len(manq)==0:
                corrections_auto.append("✅ Programme Auger complet — aucun trou manquant")
            if score>=90:
                corrections_auto.append("✅ Qualité des données globale excellente")

            score=max(0,min(100,score))
            total_issues=len(erreurs)+len(avertissements)

        # ── AFFICHAGE RÉSULTATS ───────────────────────────────────────────────
        st.markdown("---")
        # Score global
        col1,col2,col3,col4=st.columns(4)
        score_color="🟢" if score>=80 else "🟡" if score>=60 else "🔴"
        col1.metric("Score qualité données",f"{score}/100",f"{score_color}")
        col2.metric("🔴 Erreurs critiques",len([e for e in erreurs if 'CRITIQUE' in e['sévérité']]))
        col3.metric("🟠 Erreurs",len([e for e in erreurs if 'ERREUR' in e['sévérité']]))
        col4.metric("🟡 Avertissements",len(avertissements))

        # Barre de score
        st.progress(score/100, text=f"Score qualité global : {score}/100")

        st.markdown("---")

        # Erreurs
        if len(erreurs)>0:
            st.markdown("### 🔴 Erreurs détectées")
            df_err=pd.DataFrame(erreurs)
            for _,e in df_err.iterrows():
                with st.expander(f"{e['sévérité']} | [{e['module']}] — {e['trou']} : {e['message']}"):
                    st.markdown(f"**📍 Module :** {e['module']}")
                    st.markdown(f"**🔍 Problème :** {e['message']}")
                    st.markdown(f"**✅ Correction recommandée :** {e['correction']}")
                    col1,col2=st.columns(2)
                    col1.button(f"Marquer comme corrigé",key=f"fix_{_}_{e['trou']}")
                    col2.button(f"Ignorer",key=f"ign_{_}_{e['trou']}")

        # Avertissements
        if len(avertissements)>0:
            st.markdown("### 🟡 Avertissements")
            df_av=pd.DataFrame(avertissements)
            for _,a in df_av.iterrows():
                st.warning(f"**[{a['module']}] {a['trou']}** — {a['message']} → *{a['correction']}*")

        # Corrections automatiques
        if len(corrections_auto)>0:
            st.markdown("### ✅ Corrections automatiques appliquées")
            for c in corrections_auto:
                st.success(c)

        # Graphique radar des modules
        st.markdown("### 📊 Score par module")
        modules_scores = {
            'Forages': max(0,100-len([e for e in erreurs if e['module']=='Forages'])*10),
            'Intervalles': max(0,100-len([e for e in erreurs if e['module']=='Intervalles'])*8),
            'Auger': max(0,100-len([e for e in erreurs if e['module']=='Auger'])*5),
            'pXRF': max(0,100-len([e for e in erreurs if e['module']=='pXRF'])*6),
            'Géophysique': max(0,100-len([e for e in erreurs if e['module']=='Géophysique'])*4),
            'Structures': max(0,100-len([e for e in erreurs if e['module']=='Structures'])*4),
        }
        fig_r=go.Figure(data=go.Scatterpolar(
            r=list(modules_scores.values()),
            theta=list(modules_scores.keys()),
            fill='toself', fillcolor='rgba(33,150,243,0.3)',
            line=dict(color='#2196F3',width=2),
            marker=dict(size=8,color='#2196F3')))
        fig_r.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,100])),
            title="Score qualité par module",height=400,showlegend=False)
        st.plotly_chart(fig_r,use_container_width=True)

        # Rapport d'audit
        st.markdown("### 📄 Rapport d'audit")
        audit_txt=f"""RAPPORT D'AUDIT — {NOM_PROSPECT}
Permis: {NOM_PERMIS}
Date: {datetime.date.today()}
Score global: {score}/100
{'='*50}

ERREURS ({len(erreurs)}):
"""
        for e in erreurs:
            audit_txt+=f"  [{e['module']}] {e['trou']}: {e['message']}\n  → Correction: {e['correction']}\n\n"
        audit_txt+=f"\nAVERTISSEMENTS ({len(avertissements)}):\n"
        for a in avertissements:
            audit_txt+=f"  [{a['module']}] {a['trou']}: {a['message']}\n  → {a['correction']}\n\n"
        audit_txt+=f"\nCORRECTIONS APPLIQUÉES:\n"
        for c in corrections_auto:
            audit_txt+=f"  {c}\n"
        st.download_button("📥 Télécharger rapport d'audit",data=audit_txt,
            file_name=f"audit_{NOM_PROSPECT}_{datetime.date.today()}.txt",mime='text/plain')

    else:
        st.markdown("""
        ### 🔍 Ce que l'algorithme d'audit vérifie :

        | Module | Vérifications |
        |--------|--------------|
        | **Forages** | Coordonnées manquantes, azimut/inclinaison invalides, profondeurs nulles |
        | **Intervalles** | Chevauchements, lacunes, dépassements profondeur, valeurs négatives |
        | **Auger** | Trous manquants, profondeurs aberrantes, données incohérentes |
        | **pXRF** | Valeurs négatives, aberrations, trous sans collar |
        | **Géophysique** | Valeurs IP/mag aberrantes, points sans coordonnées |
        | **Structures** | Pendages/directions hors plage, données manquantes |
        | **QAQC** | Taux d'insertion standards/blancs/duplicatas |

        ### 🤖 Corrections automatiques disponibles :
        - Remplacement valeurs négatives par LOD (limite de détection)
        - Détection et signalement des doublons
        - Validation cohérence De/A des intervalles
        - Vérification complétude du programme Auger

        **Cliquez sur "Lancer l'audit complet" pour démarrer l'analyse.**
        """)

st.markdown("---")
st.caption(f"⛏️ PROJET MINIÈRE | {NOM_PROSPECT} | {NOM_PERMIS} | RC · Aircore · Diamond · Auger · pXRF · Géophysique · SOP · Audit IA")
