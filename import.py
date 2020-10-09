#region Import Modules

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import datetime
import sympy
import scipy.stats
import seaborn as sns
import cartopy.crs as ccrs


#endregion

#region Functions

def get_latitude(latitude_string):
    coordinate_designation ,  coordinate_value_string = latitude_string.split(':')
    try:
        if coordinate_designation.lower()=='lat':
            if coordinate_value_string[-1].lower()=='n':
                latitude=float(coordinate_value_string[:-1])
            elif coordinate_value_string[-1].lower()=='s':
                latitude = -float(coordinate_value_string[:-1])
            else:
                print('Unknown latitude direction: '+coordinate_value_string[-1])
        return latitude #[°N]
    except:
        print('Unknown format! Expected:\'Lat:XXX.XX N\' or \'Lat:XXX.XX S\'')
        print('Text Received: ' + latitude_string)

def get_longitude(longitude_string):
    coordinate_designation , coordinate_value_string=longitude_string.split(':')
    try:
        if coordinate_designation.lower()=='long':
            if coordinate_value_string[-1].lower()=='e':
                longitude=float(coordinate_value_string[:-1])
            elif coordinate_value_string[-1].lower()=='w':
                longitude = -float(coordinate_value_string[:-1])
            else:
                print('Unknown longitude direction: '+coordinate_value_string[-1])
        return longitude #[°E]
    except:
        print('Unknown format! Expected:\'Long:XXX.XX N\' or \'Long:XXX.XX S\'')
        print('Text Received: '+longitude_string)

def get_period(period_string):
    coordinate_designation , coordinate_value_string=period_string.split(':')
    period_raw=coordinate_value_string.split('-')
    try:
        if coordinate_designation.lower()=='period':
            period=[float(x)+1900 if float(x)>=30 else float(x)+2000 for x in period_raw]
        return period #period of analysis: Ashrae
    except:
        print('Unknown format! Expected:\'Period: XX-XX \'')
        print('Text Received: '+period_string)

def get_climatic_summary(column_names,indexes,line,key_1,key_2,key_3,df):
    climatic_summary = pd.DataFrame(columns=column_names, index=indexes)

    for i,value in enumerate(key_1):
        if math.isnan(key_1[i]):
            climatic_summary[climatic_summary.columns[i]][climatic_summary.index[0]] = np.nan
        else:
            climatic_summary[climatic_summary.columns[i]][climatic_summary.index[0]] = df[df.columns[key_1[i]]][line]
        if math.isnan(key_2[i]):
            climatic_summary[climatic_summary.columns[i]][climatic_summary.index[1]] = np.nan
        else:
            climatic_summary[climatic_summary.columns[i]][climatic_summary.index[1]] = df[df.columns[key_2[i]]][line]
        if math.isnan(key_3[i]):
            climatic_summary[climatic_summary.columns[i]][climatic_summary.index[2]] = np.nan
        else:
            climatic_summary[climatic_summary.columns[i]][climatic_summary.index[2]] = df[df.columns[key_3[i]]][line]
    return climatic_summary

def get_climatic_summary_2(column_names,indexes,key_1,key_2,key_3,line_1,line_2,line_3,df):
    climatic_summary = pd.DataFrame(columns=column_names, index=indexes)
    for i,value in enumerate(key_1):
        if math.isnan(key_1[i]):
            climatic_summary[climatic_summary.columns[i]][climatic_summary.index[0]] = np.nan
        else:
            climatic_summary[climatic_summary.columns[i]][climatic_summary.index[0]] = df[df.columns[key_1[i]]][line_1[i]]
        if math.isnan(key_2[i]):
            climatic_summary[climatic_summary.columns[i]][climatic_summary.index[1]] = np.nan
        else:
            climatic_summary[climatic_summary.columns[i]][climatic_summary.index[1]] = df[df.columns[key_2[i]]][line_2[i]]
        if math.isnan(key_3[i]):
            climatic_summary[climatic_summary.columns[i]][climatic_summary.index[2]] = np.nan
        else:
            climatic_summary[climatic_summary.columns[i]][climatic_summary.index[2]] = df[df.columns[key_3[i]]][line_3[i]]
    return climatic_summary

def get_climatic_summary_3(column_names,indexes,line,key_1,key_2,key_3,key_4,df):
    climatic_summary = pd.DataFrame(columns=column_names, index=indexes)

    for i,value in enumerate(key_1):
        if math.isnan(key_1[i]):
            climatic_summary[climatic_summary.columns[i]][climatic_summary.index[0]] = np.nan
        else:
            climatic_summary[climatic_summary.columns[i]][climatic_summary.index[0]] = df[df.columns[key_1[i]]][line]
        if math.isnan(key_2[i]):
            climatic_summary[climatic_summary.columns[i]][climatic_summary.index[1]] = np.nan
        else:
            climatic_summary[climatic_summary.columns[i]][climatic_summary.index[1]] = df[df.columns[key_2[i]]][line]
        if math.isnan(key_3[i]):
            climatic_summary[climatic_summary.columns[i]][climatic_summary.index[2]] = np.nan
        else:
            climatic_summary[climatic_summary.columns[i]][climatic_summary.index[2]] = df[df.columns[key_3[i]]][line]
        if math.isnan(key_4[i]):
            climatic_summary[climatic_summary.columns[i]][climatic_summary.index[3]] = np.nan
        else:
            climatic_summary[climatic_summary.columns[i]][climatic_summary.index[3]] = df[df.columns[key_4[i]]][line]

    return climatic_summary

def get_daily_temperature(T_max,T_range,identifier):
    # daily temperature profile was determined by statistical analysis on different locations, it is not location-specific
    columns = ['AST', 'f_tr']
    index = list(range(1, 25))
    f = [0.88, 0.92, 0.95, 0.98, 1, 0.98, 0.91, 0.74, 0.55, 0.38, 0.23, 0.13, 0.05, 0, 0, 0.06, 0.14, 0.24, 0.39, 0.5,
         0.59, 0.68, 0.75, 0.82]
    fraction_of_daily_temperature_range = pd.DataFrame(columns=columns, index=index,
                                                       data=np.array([index, f]).transpose())
    T=fraction_of_daily_temperature_range.copy()
    T[identifier]=T_max-T['f_tr']*T_range
    del T['f_tr']
    return T #[°C]

def extract_ashrae_data(filename):
    df = pd.read_excel (filename)
    print (df)

    location=df[df.columns[0]][0]
    latitude=get_latitude(df[df.columns[0]][1])
    longitude=get_longitude(df[df.columns[2]][1])
    altitude=float(df[df.columns[4]][1].split(':')[1])
    time_zone=float(df[df.columns[9]][1].split(':')[1])
    period_of_analysis=get_period(df[df.columns[12]][1])



    column_names=['coldest month','T_db','V_wind @ T_db','WIND_DIRECTION @ T_db','T_dp','HR @ T_dp','T_db @ T_dp','V_wind','T_db @ V_wind']
    indexes=[99.6,99,98]
    heating=pd.DataFrame(columns=column_names, index=indexes)
    line=7
    key_1=[0,1,13,14,3,4,5,9,10]
    key_2=[0,2,np.nan,np.nan,6,7,8,11,12]
    key_3=[0,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]
    heating=get_climatic_summary(column_names,indexes,line,key_1,key_2,key_3,df)

    column_names=['hottest month','hottest month in T_db range','T_db','T_wb @ T_db','V_wind @ T_db','WIND_DIRECTION @ T_db','T_wb','T_db @ T_wb','T_dp','HR @ T_dp','T_db @ T_dp','Delta_h','T_db @ Delta_h']
    indexes=[99.6,99,98]
    cooling=pd.DataFrame(columns=column_names, index=indexes)
    line=[14,20]
    key_1=[0,1,2,3,14,15,8,9,0,1,2,9,10]
    line_1=[14,14,14,14,14,14,14,14,20,20,20,20,20]
    key_2=[0,1,4,5,np.nan,np.nan,10,11,3,4,5,11,12]
    line_2=[14,14,14,14,14,14,14,14,20,20,20,20,20]
    key_3=[0,1,6,7,np.nan,np.nan,12,13,6,7,8,13,14]
    line_3=[14,14,14,14,14,14,14,14,20,20,20,20,20]
    cooling=get_climatic_summary_2(column_names,indexes,key_1,key_2,key_3,line_1,line_2,line_3,df)

    extreme_V_wind={
        '99%': df[df.columns[0]][27],
        '97.5%': df[df.columns[1]][27],
        '95%': df[df.columns[1]][27]
    }


    extreme_T_wb={'Max':df[df.columns[3]][27]}

    extreme_T_db={
        'Min':df[df.columns[4]][27],
        'Max': df[df.columns[5]][27],
        'Std Min': df[df.columns[6]][27],
        'Std Max': df[df.columns[7]][27],
    }

    column_names=['Min','Max']
    indexes=[5,10,20,50]

    line=27
    key_1=[8,9]
    key_2=[10,11]
    key_3=[12,13]
    key_4=[14,15]
    return_periods_extreme_T_db=get_climatic_summary_3(column_names,indexes,line,key_1,key_2,key_3,key_4,df)

    indexes_names=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    columns_names=['T_db: mean','T_db: std','HDD10.0','HDD18.3','CDD10.0','CDD18.3','CDH23.3','CDH26.7']
    design_degree_days=pd.DataFrame(data=df[df.columns[4:16]][31:39].values.transpose(),columns=columns_names,index=indexes_names)

    indexes_names=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    columns_names=['T_db: 99.6%','T_wb @ T_db: 99.6%','T_db: 98%','T_wb @ T_db: 98%','T_db: 95%','T_wb @ T_db: 95%','T_db: 90%','T_wb @ T_db: 90%']
    T_db_monthly=pd.DataFrame(data=df[df.columns[4:16]][40:48].values.transpose(),columns=columns_names,index=indexes_names)

    indexes_names=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    columns_names=['T_wb: 99.6%','T_db @ T_wb: 99.6%','T_wb: 98%','T_db @ T_wb: 98%','T_wb: 95%','T_db @ T_wb: 95%','T_wb: 90%','T_db @ T_wb: 90%']
    T_wb_monthly=pd.DataFrame(data=df[df.columns[4:16]][49:57].values.transpose(),columns=columns_names,index=indexes_names)

    indexes_names=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    columns_names=['T_db: Range','T_db: Range @ 95% T_db','T_wb: Range @ 95% T_db','T_db: Range @ 95% T_wb','T_wb: Range @ 95% T_wb']
    T_Ranges_monthly=pd.DataFrame(data=df[df.columns[4:16]][58:63].values.transpose(),columns=columns_names,index=indexes_names)

    indexes_names=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    columns_names=['tau_b','tau_d','Ebn,noon','Edn,noon']
    radiation=pd.DataFrame(data=df[df.columns[4:16]][64:68].values.transpose(),columns=columns_names,index=indexes_names)

    return location,latitude,longitude, altitude, time_zone, period_of_analysis, heating, cooling, extreme_V_wind, extreme_T_wb, extreme_T_db, return_periods_extreme_T_db, design_degree_days, T_db_monthly,T_wb_monthly, T_Ranges_monthly, radiation

def add_subplot_to_annual_graph(list_of_axes,list_of_lines,i,x=None,y_avg=None,y_avg_err=None,
                                y_min=None,y_min_err=None,y_max=None,y_max_err=None,y_extreme_min=None,
                                y_extreme_min_err=None,y_extreme_max=None,y_extreme_max_err=None,
                                x_axis_title=None,y_axis_title=None,subplot_title=None):
    list_of_lines.append([])
    y_avg_upper = list(y_avg + y_avg_err)
    y_avg_lower = list(y_avg - y_avg_err)
    y_min_upper = list(y_min + y_min_err)
    y_min_lower = list(y_min - y_min_err)
    y_max_upper = list(y_max + y_max_err)
    y_max_lower = list(y_max - y_max_err)

    # Monthly Average
    list_of_lines[i].append(
        list_of_axes[i].plot(x, y_avg, color='black', label='Monthly: Average')
    )
    list_of_lines[i].append(
        list_of_axes[i].fill_between(x, y_avg_lower, y_avg_upper, label='Band: Average', color='gainsboro', alpha=0.9)
    )
    # Monthly Minimum
    list_of_lines[i].append(
        list_of_axes[i].plot(x, y_min, label='Monthly: Minimum', color='lightskyblue')
    )

    list_of_lines[i].append(
        list_of_axes[i].fill_between(x, y_min_lower, y_min_upper, label='Band: Minimum', color='skyblue', alpha=0.3)
    )
    # Monthly Maximum
    list_of_lines[i].append(
        list_of_axes[i].plot(x, y_max, label='Monthly: Maximum', color='indianred')
    )
    list_of_lines[i].append(
        list_of_axes[i].fill_between(x, y_max_lower, y_max_upper, label='Band: Maximum', color='lightcoral', alpha=0.3)
    )
    # Extreme Annual Maximum
    list_of_lines[i].append(
        list_of_axes[i].errorbar(x, y_extreme_max, yerr=y_extreme_max_err, label='Extreme Annual: Maximum',
                                 color='maroon', ecolor='indianred')
    )
    # Extreme Annual Minimum
    list_of_lines[i].append(
        list_of_axes[i].errorbar(x, y_extreme_min, yerr=y_extreme_min_err, label='Extreme Annual: Minimum',
                                 color='dodgerblue', ecolor='steelblue')
    )

    ax = list_of_axes[i]
    ax.set_xlabel(x_axis_title)
    ax.set_ylabel(y_axis_title)
    ax.set_title(subplot_title)
    ax.legend(loc='upper right')

    return list_of_axes,list_of_lines

def extraterrestrial_radiant_flux(n):
    E_sc=1367 #[W/m^2]
    E_0=E_sc*(1+0.033*math.cos(math.radians(360*(n-3)/365)))
    return E_0 #[W/m^2]

def equation_of_time(n):
    tau=math.radians(360*(n-1)/365)
    ET=2.2918*(0.0075+0.1868*math.cos(tau)-3.2077*math.sin(tau)-1.4615*math.cos(2*tau)-4.089*math.sin(2*tau))
    return ET #[min]

def solar_declination(n):
    delta=23.45*math.sin(math.radians(360*(n+284)/365))
    return delta #[°]

def apparent_solar_time(n,LST,longitude,time_zone,LSM=None,DST=None):
    if LSM==None:
        LSM=15*time_zone
    if DST!=None:
        LST-=1
    ET=equation_of_time(n)
    AST=LST+(ET/60)+(longitude-LSM)/15
    return AST #[h]

def hour_angle(AST):
    H=15*(AST-12)
    return H #[°]

def solar_altitude(latitude,delta,H):
    L=math.radians(latitude)
    delta=math.radians(delta)
    H=math.radians(H)

    beta=math.asin((math.cos(L)*math.cos(delta)*math.cos(H))+(math.sin(L)*math.sin(delta)))
    beta=math.degrees(beta)

    return beta #[°]

def azimuth_angle(H,delta,beta,latitude):
    L=math.radians(latitude)
    delta=math.radians(delta)
    H=math.radians(H)
    beta=math.radians(beta)

    sin_phi=(math.sin(H)*math.cos(delta))/math.cos(beta)
    cos_phi=(math.cos(H)*math.cos(delta)*math.sin(L)-math.sin(delta)*math.cos(L))/(math.cos(beta))

    # solution to a trigonometric equation has two solutions in [0,2*pi]:
    phi_sin=np.array([math.asin(sin_phi),math.pi-math.asin(sin_phi)])
    phi_cos=np.array([math.acos(cos_phi),2*math.pi-math.acos(cos_phi)])

    # solution for phi is the angle which minimizes the difference between cos(phi) and sin(phi)
    phi_aux1=np.concatenate((phi_sin,phi_sin))
    phi_aux2=reversed_arr = np.concatenate((phi_cos,phi_cos[::-1]))
    err_rel=abs(phi_aux1-phi_aux2)
    x=err_rel.argmin()
    phi=math.degrees(phi_aux1[x])

    return phi #[°]


def relative_air_mass(beta):
    beta=math.radians(beta)
    m=1/(math.sin(beta)+0.50572*(6.07995+math.degrees(beta))**(-1.6364))

    if m.imag!=0:
        m=np.nan

    return m #[]

def normal_irradiance(E_0,tau_b,tau_d,m):
    ab = 1.454 - 0.406*tau_b - 0.268*tau_d + 0.021*tau_b*tau_d
    ad=0.507 + 0.205*tau_b - 0.080*tau_d - 0.190*tau_b*tau_d

    E_b=E_0*math.exp(-tau_b*m**(ab))
    E_d=E_0*math.exp(-tau_d*m**(ad))

    return E_b, E_d #[W/m^2]

def direction_to_surface_azimuth(direction):
    directions_key={
        'N':180,
        'NE':-135,
        'E':-90,
        'SE':-45,
        'S':0,
        'SW':45,
        'W':90,
        'NW':135,
       }
    surface_azimuth=directions_key[direction]

    return surface_azimuth #[°]

def surface_azimuth_to_direction(surface_azimuth):
    directions_key={
        'N':180,
        'NE':-135,
        'E':-90,
        'SE':-45,
        'S':0,
        'SW':45,
        'W':90,
        'NW':135,
       }
    inv_map = {v: k for k, v in directions_key.items()}
    direction = inv_map[surface_azimuth]

    return direction #[]




def incidence_angle(beta,surface_azimuth,phi,tilt_angle):
    gama=phi-surface_azimuth
    beta=math.radians(beta)
    surface_azimuth=math.radians(surface_azimuth)
    phi=math.radians(phi)
    tilt_angle=math.radians(tilt_angle)
    gama=math.radians(gama)

    theta=math.acos(math.cos(beta)*math.cos(gama)*math.sin(tilt_angle)+math.sin(beta)*math.cos(tilt_angle))
    theta=math.degrees(theta)

    return theta #[°]

def on_surface_irradiance(E_b,E_d,theta,tilt_angle,beta,rho_g=0.2):
    tilt_angle=math.radians(tilt_angle)
    theta=math.radians(theta)
    beta=math.radians(beta)

    # beam component
    E_bt=E_b*math.cos(theta)

    # diffuse component
    Y=max(0.45,0.55+0.437*math.cos(theta)+0.313*(math.cos(theta)**2))
    if math.degrees(tilt_angle) <= 90:
        E_dt=E_d*(Y*math.sin(tilt_angle)+math.cos(tilt_angle))
    else:
        E_dt=E_d*Y*math.sin(tilt_angle)

    # reflected component
    E_rt=(E_b*math.sin(beta)+E_d)*rho_g*((1+math.cos(beta))/2)

    return E_bt,E_dt,E_rt #[W/m^2]


def generate_hourly_data(latitude,longitude,time_zone,n_day,surface_azimuth,tilt_angle,radiation_data):
    LST=np.arange(0,24,0.5)
    df = pd.DataFrame(index = LST)

    delta = solar_declination(n_day)
    E_0=extraterrestrial_radiant_flux(n_day)
    month=get_month_from_n(n_day)
    tau_d=radiation_data['tau_d'][month]
    tau_b = radiation_data['tau_b'][month]


    df['LST']=df.index
    for i,r in df.iterrows():
        df['AST']=apparent_solar_time(n_day, df.index, longitude, time_zone, LSM = None, DST = None)
        df.loc[i,'H']=hour_angle(df['AST'][i])
        df.loc[i,'beta']=solar_altitude(latitude,delta,df['H'][i])
        df.loc[i,'phi']=azimuth_angle(df['H'][i], delta, df['beta'][i], latitude)
        df.loc[i,'theta']=incidence_angle(df['beta'][i],surface_azimuth,df['phi'][i],tilt_angle)
        df.loc[i,'m']=relative_air_mass(df['beta'][i])
        df.loc[i,'Eb,n'],df.loc[i,'Ed,n']=normal_irradiance(E_0, tau_b, tau_d, df['m'][i])
        df.loc[i,'Eb,surface'], df.loc[i,'Ed,surface'],df.loc[i,'Er,surface']=on_surface_irradiance(df['Eb,n'][i],df['Ed,n'][i],df['theta'][i],tilt_angle,df['beta'][i],rho_g=0.2)

    return df

def get_month_from_n(n):
    n=n-1
    d0=datetime.date(2000,1,1)
    d2=d0+(datetime.timedelta(n))
    month=d2.strftime('%b')
    return month #[]

def get_n_from_date(day,month,year=2000):
    d0 = datetime.date(year, 1, 1)

    if type(month)=='str':
        month = datetime.datetime.strptime(month, "%b").month

    d2=datetime.date(year,month,day)

    Deltat=d2-d0
    n=Deltat.days+1
    return n

def generate_Tsample(month,n_sample,design_degree_days):
    mean=design_degree_days['T_db: mean'][month]
    std=design_degree_days['T_db: std'][month]
    dist=scipy.stats.norm(mean,std)
    T=dist.rvs(n_sample)
    return T

def generate_df_Tdistribution(n_sample,design_degree_days,plot=True):
    df=pd.DataFrame(columns=['T','Month'])
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for month in months:
        T=generate_Tsample(month,n_sample,design_degree_days)
        df_aux = pd.DataFrame(columns = ['T','Month'])
        df_aux['T']=T
        df_aux['Month']=month
        df=df.append(df_aux,ignore_index=True)
    if plot == True:
        ax=sns.displot(df,x='T',hue='Month', label = month,kind='kde',fill=True)
    plt.xlabel('Dry-Bulb Temperature - [°C]')
    plt.ylabel('Probability Density Function - f - [-]')
    plt.title('Dry-bulb Monthly Temperature Distribution',fontsize=14, weight='bold')
    return df,ax

def month_limits(year):
    d0 = datetime.date(year, 1, 1)
    df=pd.DataFrame(columns=['Month','n_start','n_end','1st_day','last_day'])
    for i in range(12-1):
        d_first=datetime.date(year,i+1,1)
        d_last = datetime.date(year, i + 2, 1)-datetime.timedelta(days=1)
        df_aux=pd.DataFrame(columns=['Month','n_start','n_end'],index=np.arange(1,2,1))
        df_aux['Month']=i+1
        df_aux['1st_day']=d_first
        df_aux['last_day'] = d_last
        df_aux['n_start']=(d_first-d0).days+1
        df_aux['n_end']=(d_last-d0).days+1
        df=df.append(df_aux,ignore_index=True)
    d_first = datetime.date(year, 12, 1)
    d_last = datetime.date(year, 12, 31)
    df_aux = pd.DataFrame(columns = ['Month', 'n_start', 'n_end'], index = np.arange(1, 2, 1))
    df_aux['Month'] = 12
    df_aux['1st_day'] = d_first
    df_aux['last_day'] = d_last
    df_aux['n_start'] = (d_first - d0).days + 1
    df_aux['n_end'] = (d_last - d0).days + 1
    df = df.append(df_aux, ignore_index = True)

    return df


def time_period_df(date_start,date_end):
    year_start=date_start.year
    year_finish=date_end.year

    dic_days=month_limits(year_start)

    df=pd.DataFrame(columns=['Year','Month','Number of days'],index=np.arange(1,2,1))
    n_end=dic_days[dic_days['Month']==date_start.month]['n_end'].values
    n_start=dic_days[dic_days['Month']==date_start.month]['n_start'].values
    df['Number of days']=(n_end-n_start)-date_start.day+1
    df['Year']=date_start.year
    df['Month']=date_start.month

    date_i=datetime.date(year_start,date_start.month,1)
    while date_i<date_end:
        month=date_i.month
        year=date_i.year
        try:
            date_i=datetime.date(year,month+1,1)
        except:
            date_i=datetime.date(year+1,1,1)
        df_aux=pd.DataFrame(columns=['Year','Month','Number of days'])
        n_end = dic_days[dic_days['Month'] == date_i.month]['n_end'].values
        n_start = dic_days[dic_days['Month'] == date_i.month]['n_start'].values
        df_aux['Number of days']=n_end-n_start+1
        df_aux['Year'] = date_i.year
        df_aux['Month'] = date_i.month
        df = df.append(df_aux, ignore_index = True)

    df=df.drop(df[(df['Month']>=date_end.month)&(df['Year']==date_end.year)].index)
    df_aux=pd.DataFrame(columns=['Year','Month','Number of days'],index=np.arange(1,2,1))
    n_end=dic_days[dic_days['Month']==date_end.month]['n_end'].values
    n_start=dic_days[dic_days['Month']==date_end.month]['n_start'].values
    df_aux['Number of days']=date_end.day
    df_aux['Year']=date_end.year
    df_aux['Month']=date_end.month
    df = df.append(df_aux, ignore_index = True)

    return df


def calculate_degree_days(date_start,date_end,T_heating,T_cooling,design_degree_days):
    df_time=time_period_df(date_start,date_end)

    df_sol=df_time.copy(deep=True)
    df_sol['Heating Degree Days']=np.nan
    df_sol['Cooling Degree Days']=np.nan

    n_sample=100000
    for row,column in df_sol.iterrows():
        month=df_sol.loc[row,'Month']
        d2=datetime.date(df_sol.loc[row,'Year'],month,1)
        month = d2.strftime('%b')


        mean = design_degree_days['T_db: mean'][month]
        std = design_degree_days['T_db: std'][month]
        dist = scipy.stats.norm(mean, std)
        T=dist.rvs(n_sample)

        x = np.arange(min(T), max(T), (max(T) - min(T)) / n_sample)

        cdf=dist.cdf(x)
        sf=dist.sf(x)


        cdf_a=pd.DataFrame(columns=['T','sum_p'],data=list(np.array([x,cdf]).transpose()))
        cdf_a['DeltaT']=T_heating-cdf_a['T']

        sf_a=pd.DataFrame(columns=['T','sum_p'],data=list(np.array([x,sf]).transpose()))
        sf_a['DeltaT']=sf_a['T']-T_cooling

        cdf_a=cdf_a[cdf_a['T']<T_heating]
        sf_a=sf_a[sf_a['T']>T_cooling]


        HD=np.trapz(cdf_a['DeltaT'],cdf_a['sum_p'])
        CD = np.trapz(sf_a['sum_p'],sf_a['DeltaT'])

        HDD=HD*df_sol.loc[row,'Number of days']
        CDD=CD*df_sol.loc[row,'Number of days']

        df_sol.loc[row,'Heating Degree Days']= HDD
        df_sol.loc[row, 'Cooling Degree Days'] = CDD
    return df_sol

def calculate_monthly_representative_radiation(radiation,latitude,longitude,time_zone,year=2020,surface_azimuth=0,tilt_angle=0):
    day=21

    df=pd.DataFrame()

    for i in range(12):
        n_day=get_n_from_date(day,i+1,year)
        df_day = generate_hourly_data(latitude, longitude, time_zone, n_day, surface_azimuth, tilt_angle, radiation)
        df_day['Month']=i+1
        df_day['Etotal,n']=df_day['Eb,n']+df_day['Ed,n']
        df_day['Etotal,surface'] = df_day['Eb,surface'] + df_day['Ed,surface'] + df_day['Er,surface']
        df_day=df_day[(df_day['LST'] == 9) | (df_day['LST'] == 12) | (df_day['LST'] == 15) | (df_day['LST'] == 17)]
        df=df.append(df_day,ignore_index=True)
    return df

#endregion

#region Data Treatment

# Import data from xls and parse it to dataframes
location,latitude,longitude, altitude, time_zone, period_of_analysis, heating, cooling, extreme_V_wind, extreme_T_wb,\
extreme_T_db, return_periods_extreme_T_db, design_degree_days, T_db_monthly,T_wb_monthly, T_Ranges_monthly, \
radiation = extract_ashrae_data(r'data_files/test.xlsx')

surface_azimuth=direction_to_surface_azimuth('S')
tilt_angle=0
n_day=60+15
df=generate_hourly_data(latitude,longitude,time_zone,n_day,surface_azimuth,tilt_angle,radiation)


#endregion

#region Plots
list_of_figures=[]
list_of_axes=[]
list_of_lines=[]

#region Figure 1
list_of_figures.append(plt.figure(len(list_of_figures)+1,figsize=(16,8)))
list_of_figures[0].suptitle('Temperature: Annual Limits',fontsize=14, weight='bold')
list_of_axes.append(list_of_figures[0].add_subplot(1,2,1))

#Subfigure1

x=T_db_monthly['T_db: 95%'].index
y_avg=T_db_monthly['T_db: 95%']
y_avg_err=design_degree_days['T_db: std']*2
y_min=y_avg-T_Ranges_monthly['T_db: Range @ 95% T_db']*0.5
y_min_err=extreme_T_db['Std Min']*2
y_max=y_avg+T_Ranges_monthly['T_db: Range @ 95% T_db']*0.5
y_max_err=extreme_T_db['Std Max']*2
y_extreme_min=np.ones(len(design_degree_days.index))*extreme_T_db['Min']
y_extreme_min_err=np.ones(len(design_degree_days.index))*extreme_T_db['Std Min']
y_extreme_max=np.ones(len(design_degree_days.index))*extreme_T_db['Max']
y_extreme_max_err=np.ones(len(design_degree_days.index))*extreme_T_db['Std Max']
x_axis_title='Month of the year'
y_axis_title='Dry-bulb Temperature - [°C]'
subplot_title='Dry-bulb Temperature'

add_subplot_to_annual_graph(list_of_axes,list_of_lines,0,x=x,y_avg=y_avg,y_avg_err=y_avg_err,y_min=y_min,
                            y_min_err=y_min_err,y_max=y_max,y_max_err=y_max_err,y_extreme_min=y_extreme_min,
                            y_extreme_min_err=y_extreme_min_err,y_extreme_max=y_extreme_max,
                            y_extreme_max_err=y_extreme_max_err,x_axis_title=x_axis_title,y_axis_title=y_axis_title,
                            subplot_title=subplot_title)

#Subfigure2
list_of_axes.append(list_of_figures[0].add_subplot(1,2,2,sharex=list_of_axes[0],sharey=list_of_axes[0]))

x=T_wb_monthly.index
y_avg=T_wb_monthly['T_wb: 95%']
y_avg_err=design_degree_days['T_db: std']*2
y_min=y_avg-T_Ranges_monthly['T_db: Range @ 95% T_wb']*0.5
y_min_err=extreme_T_db['Std Min']*2
y_max=y_avg+T_Ranges_monthly['T_db: Range @ 95% T_wb']*0.5
y_max_err=extreme_T_db['Std Max']*2
y_extreme_min=np.ones(len(design_degree_days.index)) * extreme_T_db['Min']
y_extreme_min_err=np.ones(len(design_degree_days.index)) * extreme_T_db['Std Min']
y_extreme_max=np.ones(len(design_degree_days.index))*extreme_T_wb['Max']
y_extreme_max_err=np.ones(len(design_degree_days.index))*extreme_T_db['Std Max']
x_axis_title='Month of the year'
y_axis_title='Wet-bulb Temperature - [°C]'
subplot_title='Wet-bulb Temperature'

add_subplot_to_annual_graph(list_of_axes,list_of_lines,1,x=x,y_avg=y_avg,y_avg_err=y_avg_err,y_min=y_min,
                            y_min_err=y_min_err,y_max=y_max,y_max_err=y_max_err,y_extreme_min=y_extreme_min,
                            y_extreme_min_err=y_extreme_min_err,y_extreme_max=y_extreme_max,
                            y_extreme_max_err=y_extreme_max_err,x_axis_title=x_axis_title,y_axis_title=y_axis_title,
                            subplot_title=subplot_title)
#endregion

#region Figure 2

df_T,ax=generate_df_Tdistribution(100000,design_degree_days,plot=True)
ax.fig.set_size_inches(16, 8)
list_of_axes.append(ax.ax)
list_of_figures.append(ax.fig)

#endregion

#region Figure 3

fig=plt.figure(figsize=(16,8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.stock_img()
list_of_axes.append(ax)
list_of_figures.append(fig)
ax.set_title('Details of the analysis',fontsize=14, weight='bold')

ny_lon, ny_lat = longitude, latitude

plt.plot([ny_lon], [ny_lat],
         color='indianred', linewidth=2, marker='o',
         transform=ccrs.Geodetic(),
         )


box = '\n'.join((
    "Location: " + str(location),
    "Latitude: " + str(latitude),
    "Longitude: " + str(longitude),
    "Altitude: " + str(altitude),
    "Period of the analysis: [" + "%4d"%(period_of_analysis[0])+" ; "+"%4d"%(period_of_analysis[1])+"]",
))
ax.text(0.05, 0.95, box, transform=ax.transAxes, fontsize=10, va='top', ha="left",
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))  ### boxplot


plt.show()

#endregion

#region Figure 4

fig=plt.figure(figsize=(16,8))
fig.suptitle('Irradiance at different daytime hours at the 21st of each month',fontsize=14, weight='bold')
list_of_figures.append(fig)
ax1=fig.add_subplot(121)
list_of_axes.append(ax1)

df_rad=calculate_monthly_representative_radiation(radiation,latitude,longitude,time_zone)

n_colors=len(df_rad['LST'].unique())
a=sns.color_palette("RdPu",n_colors)
l1=sns.lineplot(x='Month',y='Etotal,n',style='LST',hue='LST',data=df_rad,palette=a,ax=ax1)
ax1.set_title('Irradiance on a surface perpendicular to the solar rays')
ax1.set_ylabel('Irradiance - [W/m^2]')
ax1.set_xlabel('Month - [-]')

ax2=fig.add_subplot(122,sharey=ax1,sharex=ax1)
l2=sns.lineplot(x='Month',y='Etotal,surface',style='LST',hue='LST',data=df_rad,palette=a,ax=ax2)
ax2.set_title('Total Irradiance on the surface | Tilted at '+str(tilt_angle)+' [°] | Orientation = '+str(surface_azimuth_to_direction(surface_azimuth)))
ax2.set_ylabel('Irradiance - [W/m^2]')
ax2.set_xlabel('Month - [-]')

list_of_axes.append(ax2)



'radiation calculated at th 21st of each month at AST:8 | 12 | 16'

#endregion

#endregion