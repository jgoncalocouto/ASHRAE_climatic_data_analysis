#region Import Modules
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import math
import datetime
import sympy
import scipy.stats
import seaborn as sns


#endregion

#region Functions

def get_latitude(latitude_string):
    coordinate_designation , coordinate_value_string=latitude_string.split(':')
    try:
        if coordinate_designation.lower()=='lat':
            if coordinate_value_string[-1].lower()=='n':
                latitude=float(coordinate_value_string[:-1])
            elif coordinate_value_string[-1].lower()=='s':
                latitude = -float(coordinate_value_string[:-1])
            else:
                print('Unknown latitude direction: '+coordinate_value_string[-1])
        return latitude
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
        return longitude
    except:
        print('Unknown format! Expected:\'Long:XXX.XX N\' or \'Long:XXX.XX S\'')
        print('Text Received: '+longitude_string)

def get_period(period_string):
    coordinate_designation , coordinate_value_string=period_string.split(':')
    period_raw=coordinate_value_string.split('-')
    try:
        if coordinate_designation.lower()=='period':
            period=[float(x)+1900 if float(x)>=30 else float(x)+2000 for x in period_raw]
        return period
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
    columns = ['AST', 'f_tr']
    index = list(range(1, 25))
    f = [0.88, 0.92, 0.95, 0.98, 1, 0.98, 0.91, 0.74, 0.55, 0.38, 0.23, 0.13, 0.05, 0, 0, 0.06, 0.14, 0.24, 0.39, 0.5,
         0.59, 0.68, 0.75, 0.82]
    fraction_of_daily_temperature_range = pd.DataFrame(columns=columns, index=index,
                                                       data=np.array([index, f]).transpose())
    T=fraction_of_daily_temperature_range.copy()
    T[identifier]=T_max-T['f_tr']*T_range
    del T['f_tr']
    return T

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
    box = '\n'.join(("Location: " + str(location), "Latitude: " + str(latitude), "Longitude: " + str(longitude),
                     "Altitude: " + str(altitude)))
    ax.text(0.05, 0.95, box, transform=ax.transAxes, fontsize=10, va='top', ha="left",
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))  ### boxplot
    return list_of_axes,list_of_lines

def extraterrestrial_radiant_flux(n):
    E_sc=1367 #[W/m^2]
    E_0=E_sc*(1+0.033*math.cos(math.radians(360*(n-3)/365)))
    return E_0

def equation_of_time(n):
    tau=math.radians(360*(n-1)/365)
    ET=2.2918*(0.0075+0.1868*math.cos(tau)-3.2077*math.sin(tau)-1.4615*math.cos(2*tau)-4.089*math.sin(2*tau))
    return ET

def solar_declination(n):
    delta=23.45*math.sin(math.radians(360*(n+284)/365))
    return delta

def apparent_solar_time(n,LST,longitude,time_zone,LSM=None,DST=None):
    if LSM==None:
        LSM=15*time_zone
    if DST!=None:
        LST-=1
    ET=equation_of_time(n)
    AST=LST+(ET/60)+(longitude-LSM)/15
    return AST

def hour_angle(AST):
    H=15*(AST-12)

    return H

def solar_altitude(latitude,delta,H):
    L=math.radians(latitude)
    delta=math.radians(delta)
    H=math.radians(H)

    beta=math.asin((math.cos(L)*math.cos(delta)*math.cos(H))+(math.sin(L)*math.sin(delta)))
    beta=math.degrees(beta)

    return beta

def azimuth_angle(H,delta,beta,latitude):
    L=math.radians(latitude)
    delta=math.radians(delta)
    H=math.radians(H)
    beta=math.radians(beta)

    phi=sympy.Symbol('phi',real=True)

    e1=sympy.Eq(sympy.sin(phi),(math.sin(H)*math.cos(delta))/(math.cos(beta)))
    e2=sympy.Eq(sympy.cos(phi),((math.cos(H)*math.cos(delta)*math.sin(L))-(math.sin(delta)*math.cos(L)))/(math.cos(beta)))



    try:
        sol1 = sympy.solve([e1], phi)
        sol2 = sympy.solve([e2], phi)
        min=100
        for i,valuei in enumerate(sol1):
            for j,valuej in enumerate(sol2):
                err=abs(valuei[0]-valuej[0])
                if err<min:
                    sol=i
                    min=err
        min_error=(min/sol1[sol][0])*100
        phi=sol1[sol][0]
    except:
        print('Solution could not be obtained for azimuth angle')
        phi=np.nan
    phi=math.degrees(phi)

    return phi


def relative_air_mass(beta):
    beta=math.radians(beta)
    m=1/(math.sin(beta)+0.50572*(6.07995+math.degrees(beta))**(-1.6364))

    if m.imag!=0:
        m=np.nan


    return m

def normal_irradiance(E_0,tau_b,tau_d,m):
    ab = 1.454 - 0.406*tau_b - 0.268*tau_d + 0.021*tau_b*tau_d
    ad=0.507 + 0.205*tau_b - 0.080*tau_d - 0.190*tau_b*tau_d

    E_b=E_0*math.exp(-tau_b*m**(ab))
    E_d=E_0*math.exp(-tau_d*m**(ad))

    return E_b, E_d

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

    return surface_azimuth


def incidence_angle(beta,surface_azimuth,phi,tilt_angle):
    gama=phi-surface_azimuth
    beta=math.radians(beta)
    surface_azimuth=math.radians(surface_azimuth)
    phi=math.radians(phi)
    tilt_angle=math.radians(tilt_angle)
    gama=math.radians(gama)

    theta=math.acos(math.cos(beta)*math.cos(gama)*math.sin(tilt_angle)+math.sin(beta)*math.cos(tilt_angle))
    theta=math.degrees(theta)

    return theta

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

    return E_bt,E_dt,E_rt

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
    d0=datetime.date(2000,1,1)
    d2=d0+datetime.timedelta(n)
    month=d2.strftime('%b')
    return month

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
    plt.title('Dry-bulb Monthly Temperature Distribution')
    return df,ax





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
list_of_figures[0].suptitle('Temperature: Annual Limits')
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