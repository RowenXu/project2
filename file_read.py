#------读取文件必要库------------
import xarray as xr
import pandas as pd
import torch 
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy import interpolate
import datetime as dt
import cftime
#-------需要的数据库类--------
class dataset_new(Dataset):
    def __init__(self,data,label,transform=None):
        self.data=data
        self.label=label
        self.transform=transform
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):     
        sample=self.data[idx]
        sample_label=self.label[idx]
        if self.transform:
            sample=self.transform(sample)
        return sample,sample_label
    
#------实验设定不同输入区域的范围--------
region={
    'nino12':{
        'lat':[0,-10],
        'lon':[-90,-80]
    },
    'nino34':{
        'lat':[5,-5],
        'lon':[-150,-90]
    },
    'MC':{
        'lat':[20,-10],
        'lon':[100,110]
    },
    'Tropical':
    {
        'lat':[-20,20],
        'lon':[120,280]
    }
}

S=region['nino12']
#------设定lead time-------
n_lead=12
#-------设置输入高度--------
bar=200


#---------将cftime转换为datetime--------
def cftime_to_datetime(cftime_data):
    if isinstance(cftime_data, cftime.Datetime360Day):
        datetime_data=dt.datetime(cftime_data.year, cftime_data.month, cftime_data.day, cftime_data.hour, cftime_data.minute, cftime_data.second, cftime_data.microsecond)
        return datetime_data
    else:
        return cftime_data



#=========输入文件名============
#=========输入文件名============
#=========输入文件名============
input_file=['data/ERA5/data_50y.grib',#ERA5数据,
    #UKESM数据
      'data/UKESM/ua_Amon_UKESM1-0-LL_hist-1950HC_r1i1p1f2_gn_195001-201412.nc',
      'data/UKESM/va_Amon_UKESM1-0-LL_hist-1950HC_r1i1p1f2_gn_195001-201412.nc',
      'data/UKESM/zg_Amon_UKESM1-0-LL_hist-1950HC_r1i1p1f2_gn_195001-201412.nc',
    #GISS数据
        'data/GISS/ua_Amon_GISS-E2-1-G_historical_r11i1p1f2_gn_195101-200012.nc',
        'data/GISS/va_Amon_GISS-E2-1-G_historical_r11i1p1f2_gn_195101-200012.nc',
        'data/GISS/zg_Amon_GISS-E2-1-G_historical_r11i1p1f2_gn_195101-200012.nc',
    #BCC-ESM数据
        'data/BCC-ESM/ua_Amon_BCC-ESM1_histSST_r1i1p1f1_gn_185001-201412.nc',
        'data/BCC-ESM/va_Amon_BCC-ESM1_histSST_r1i1p1f1_gn_185001-201412.nc',
        'data/BCC-ESM/zg_Amon_BCC-ESM1_histSST_r1i1p1f1_gn_185001-201412.nc',
    #ACCESS-CM2数据
        'data/ACCESS-CM2/ua_Amon_ACCESS-CM2_historical_r1i1p1f1_gn_195001-201412.nc',
        'data/ACCESS-CM2/va_Amon_ACCESS-CM2_historical_r1i1p1f1_gn_195001-201412.nc',
        'data/ACCESS-CM2/zg_Amon_ACCESS-CM2_historical_r1i1p1f1_gn_195001-201412.nc',
      ]

#------导入文件--------
ERA5_u=xr.open_dataset(input_file[0]).u.sel(isobaricInhPa=bar).sel(time=slice('1950-01-01','2023-12-01'))
ERA5_v=xr.open_dataset(input_file[0]).v.sel(isobaricInhPa=bar).sel(time=slice('1950-01-01','2023-12-01'))
ERA5_z=xr.open_dataset(input_file[0]).z.sel(isobaricInhPa=bar).sel(time=slice('1950-01-01','2023-12-01'))

UKESM_u=xr.open_dataset(input_file[1]).ua.sel(plev=bar*100).sel(time=slice('1950-01-01','2014-12-01'))
UKESM_v=xr.open_dataset(input_file[2]).va.sel(plev=bar*100).sel(time=slice('1950-01-01','2014-12-01'))
UKESM_z=xr.open_dataset(input_file[3]).zg.sel(plev=bar*100).sel(time=slice('1950-01-01','2014-12-01'))

GISS_u=xr.open_dataset(input_file[4]).ua.sel(plev=bar*100).sel(time=slice('1951-01-01','2014-12-01'))
GISS_v=xr.open_dataset(input_file[5]).va.sel(plev=bar*100).sel(time=slice('1951-01-01','2014-12-01'))
GISS_z=xr.open_dataset(input_file[6]).zg.sel(plev=bar*100).sel(time=slice('1951-01-01','2014-12-01'))

BCC_u=xr.open_dataset(input_file[7]).ua.sel(plev=bar*100).sel(time=slice('1950-01-01','2014-12-01'))
BCC_v=xr.open_dataset(input_file[8]).va.sel(plev=bar*100).sel(time=slice('1950-01-01','2014-12-01'))
BCC_z=xr.open_dataset(input_file[9]).zg.sel(plev=bar*100).sel(time=slice('1950-01-01','2014-12-01'))

ACCESS_u=xr.open_dataset(input_file[10]).ua.sel(plev=bar*100,method='nearest').sel(time=slice('1950-01-01','2014-12-01'))
ACCESS_v=xr.open_dataset(input_file[11]).va.sel(plev=bar*100,method='nearest').sel(time=slice('1950-01-01','2014-12-01'))
ACCESS_z=xr.open_dataset(input_file[12]).zg.sel(plev=bar*100,method='nearest').sel(time=slice('1950-01-01','2014-12-01'))

time_ERA5=ERA5_u.time
time_UKESM=UKESM_u.time
time_GISS=GISS_u.time
time_BCC=BCC_u.time
time_ACCESS=ACCESS_u.time
#------对于CMIP数据进行经度重排序--------
def sort_lon(data):
    data['lon']=xr.where(data['lon']>180,data['lon']-360,data['lon'])
    data=data.sortby('lon')
    return data
UKESM_u=sort_lon(UKESM_u)
UKESM_v=sort_lon(UKESM_v)
UKESM_z=sort_lon(UKESM_z)

GISS_u=sort_lon(GISS_u)
GISS_v=sort_lon(GISS_v)
GISS_z=sort_lon(GISS_z)

BCC_u=sort_lon(BCC_u)
BCC_v=sort_lon(BCC_v)
BCC_z=sort_lon(BCC_z)

ACCESS_u=sort_lon(ACCESS_u)
ACCESS_v=sort_lon(ACCESS_v)
ACCESS_z=sort_lon(ACCESS_z)
#------对于CMIP数据进行纬度重排序--------
UKESM_u=UKESM_u.sortby('lat',ascending=False)
UKESM_v=UKESM_v.sortby('lat',ascending=False)
UKESM_z=UKESM_z.sortby('lat',ascending=False)

GISS_u=GISS_u.sortby('lat',ascending=False)
GISS_v=GISS_v.sortby('lat',ascending=False)
GISS_z=GISS_z.sortby('lat',ascending=False)

BCC_u=BCC_u.sortby('lat',ascending=False)
BCC_v=BCC_v.sortby('lat',ascending=False)
BCC_z=BCC_z.sortby('lat',ascending=False)

ACCESS_u=ACCESS_u.sortby('lat',ascending=False)
ACCESS_v=ACCESS_v.sortby('lat',ascending=False)
ACCESS_z=ACCESS_z.sortby('lat',ascending=False)
#------选定区域的经纬度--------
ERA5_u=ERA5_u.sel(latitude=slice(S['lat'][0],S['lat'][1]),longitude=slice(S['lon'][0],S['lon'][1]))
ERA5_v=ERA5_v.sel(latitude=slice(S['lat'][0],S['lat'][1]),longitude=slice(S['lon'][0],S['lon'][1]))
ERA5_z=ERA5_z.sel(latitude=slice(S['lat'][0],S['lat'][1]),longitude=slice(S['lon'][0],S['lon'][1]))

UKESM_u=UKESM_u.sel(lat=slice(S['lat'][0],S['lat'][1]),lon=slice(S['lon'][0],S['lon'][1]))
UKESM_v=UKESM_v.sel(lat=slice(S['lat'][0],S['lat'][1]),lon=slice(S['lon'][0],S['lon'][1]))
UKESM_z=UKESM_z.sel(lat=slice(S['lat'][0],S['lat'][1]),lon=slice(S['lon'][0],S['lon'][1]))

GISS_u=GISS_u.sel(lat=slice(S['lat'][0],S['lat'][1]),lon=slice(S['lon'][0],S['lon'][1]))
GISS_v=GISS_v.sel(lat=slice(S['lat'][0],S['lat'][1]),lon=slice(S['lon'][0],S['lon'][1]))
GISS_z=GISS_z.sel(lat=slice(S['lat'][0],S['lat'][1]),lon=slice(S['lon'][0],S['lon'][1]))

BCC_u=BCC_u.sel(lat=slice(S['lat'][0],S['lat'][1]),lon=slice(S['lon'][0],S['lon'][1]))
BCC_v=BCC_v.sel(lat=slice(S['lat'][0],S['lat'][1]),lon=slice(S['lon'][0],S['lon'][1]))
BCC_z=BCC_z.sel(lat=slice(S['lat'][0],S['lat'][1]),lon=slice(S['lon'][0],S['lon'][1]))

ACCESS_u=ACCESS_u.sel(lat=slice(S['lat'][0],S['lat'][1]),lon=slice(S['lon'][0],S['lon'][1]))
ACCESS_v=ACCESS_v.sel(lat=slice(S['lat'][0],S['lat'][1]),lon=slice(S['lon'][0],S['lon'][1]))
ACCESS_z=ACCESS_z.sel(lat=slice(S['lat'][0],S['lat'][1]),lon=slice(S['lon'][0],S['lon'][1]))
#------对于CMIP数据进行插值--------
def interp_func(data,ref):
    data_new=data.interp(lat=ref.latitude,lon=ref.longitude)
    return data_new


UKESM_u_new=interp_func(UKESM_u,ERA5_u)
UKESM_v_new=interp_func(UKESM_v,ERA5_v)
UKESM_z_new=interp_func(UKESM_z,ERA5_z)

GISS_u_new=interp_func(GISS_u,ERA5_u)
GISS_v_new=interp_func(GISS_v,ERA5_v)
GISS_z_new=interp_func(GISS_z,ERA5_z)

BCC_u_new=interp_func(BCC_u,ERA5_u)
BCC_v_new=interp_func(BCC_v,ERA5_v)
BCC_z_new=interp_func(BCC_z,ERA5_z)

ACCESS_u_new=interp_func(ACCESS_u,ERA5_u)
ACCESS_v_new=interp_func(ACCESS_v,ERA5_v)
ACCESS_z_new=interp_func(ACCESS_z,ERA5_z)
#------对于数据进行归一化--------
def normalize(data):
    data=(data-data.mean())/data.std()
    return data

ERA5_u=normalize(ERA5_u)
ERA5_v=normalize(ERA5_v)
ERA5_z=normalize(ERA5_z)

UKESM_u_new=normalize(UKESM_u_new)
UKESM_v_new=normalize(UKESM_v_new)
UKESM_z_new=normalize(UKESM_z_new)

GISS_u_new=normalize(GISS_u_new)
GISS_v_new=normalize(GISS_v_new)
GISS_z_new=normalize(GISS_z_new)

BCC_u_new=normalize(BCC_u_new)
BCC_v_new=normalize(BCC_v_new)
BCC_z_new=normalize(BCC_z_new)

ACCESS_u_new=normalize(ACCESS_u_new)
ACCESS_v_new=normalize(ACCESS_v_new)
ACCESS_z_new=normalize(ACCESS_z_new)
#------将数据转换为tensor--------
ERA5_u=torch.from_numpy(ERA5_u.values)
ERA5_v=torch.from_numpy(ERA5_v.values)
ERA5_z=torch.from_numpy(ERA5_z.values)

UKESM_u_new=torch.from_numpy(UKESM_u_new.values)
UKESM_v_new=torch.from_numpy(UKESM_v_new.values)
UKESM_z_new=torch.from_numpy(UKESM_z_new.values)

GISS_u_new=torch.from_numpy(GISS_u_new.values)
GISS_v_new=torch.from_numpy(GISS_v_new.values)
GISS_z_new=torch.from_numpy(GISS_z_new.values)

BCC_u_new=torch.from_numpy(BCC_u_new.values)
BCC_v_new=torch.from_numpy(BCC_v_new.values)
BCC_z_new=torch.from_numpy(BCC_z_new.values)

ACCESS_u_new=torch.from_numpy(ACCESS_u_new.values)
ACCESS_v_new=torch.from_numpy(ACCESS_v_new.values)
ACCESS_z_new=torch.from_numpy(ACCESS_z_new.values)
#------设置输入数据的大小--------
def set_input_data(u,v,z,time):
    full_input_data=torch.zeros(time.size-2,9,u.shape[1],u.shape[2])
    for i in range(time.size-2):
        full_input_data[i,:,:,:]=torch.stack([u[i],
                                    v[i],
                                    z[i],
                                    u[i+1],
                                    v[i+1],
                                    z[i+1],
                                    u[i+2],
                                    v[i+2],
                                    z[i+2]])
    return full_input_data

ERA5_input_data=set_input_data(ERA5_u,ERA5_v,ERA5_z,time_ERA5)
UKESM_input_data=set_input_data(UKESM_u_new,UKESM_v_new,UKESM_z_new,time_UKESM)
GISS_input_data=set_input_data(GISS_u_new,GISS_v_new,GISS_z_new,time_GISS)
BCC_input_data=set_input_data(BCC_u_new,BCC_v_new,BCC_z_new,time_BCC)
ACCESS_input_data=set_input_data(ACCESS_u_new,ACCESS_v_new,ACCESS_z_new,time_ACCESS)

#=========标签文件名============
#=========标签文件名============
#=========标签文件名============

label_file=[
'data/label/nino34.long.anom.data.txt'
]

#------导入标签数据,提取对应时间的标签,标准化--------
def read_label(file,time):
    data=[]
    #读取
    with open(file) as f:
        f.readline()
        for i in f.readlines():
            if len(i.split())<2:
                break
            year=int(i.split()[0])
            if year in time.dt.year.values:
                [data.append(float(item)) for item in i.split()[1:]]
    data=torch.tensor(data)
    print(data.shape)
    data=data/data.std()
    return data

ERA5_nino34_data=read_label(label_file[0],time_ERA5)
UKESM_nino34_data=read_label(label_file[0],time_UKESM)
GISS_nino34_data=read_label(label_file[0],time_GISS)
BCC_nino34_data=read_label(label_file[0],time_BCC)
ACCESS_nino34_data=read_label(label_file[0],time_ACCESS)

#-----根据lead time,给标签数据匹配对应的输入数据,直到输入数据全部用完--------
def set_output_data(inp,outp,n_lead):
    print(inp.shape)
    print(outp.shape)
    input_data=inp[0:inp.shape[0]-n_lead]
    output_data=outp[n_lead:outp.shape[0]-2]
    return input_data,output_data

#------设置输出数据的大小--------
def set_full_output_data(nino,time):
    print(nino.shape)
    print(time.shape)
    full_output_data=torch.zeros(time.size-2,3)
    for i in range(time.size-2):
        full_output_data[i,:]=torch.stack(
            [nino[i],
            nino[i+1],
            nino[i+2]]
        )
    return full_output_data

ERA5_nino34_data=set_full_output_data(ERA5_nino34_data,time_ERA5)
UKESM_nino34_data=set_full_output_data(UKESM_nino34_data,time_UKESM)
GISS_nino34_data=set_full_output_data(GISS_nino34_data,time_GISS)
BCC_nino34_data=set_full_output_data(BCC_nino34_data,time_BCC)
ACCESS_nino34_data=set_full_output_data(ACCESS_nino34_data,time_ACCESS)

ERA5_input_data,ERA5_nino34_data=set_output_data(ERA5_input_data,ERA5_nino34_data,n_lead)
UKESM_input_data,UKESM_nino34_data=set_output_data(UKESM_input_data,UKESM_nino34_data,n_lead)
GISS_input_data,GISS_nino34_data=set_output_data(GISS_input_data,GISS_nino34_data,n_lead)
BCC_input_data,BCC_nino34_data=set_output_data(BCC_input_data,BCC_nino34_data,n_lead)
ACCESS_input_data,ACCESS_nino34_data=set_output_data(ACCESS_input_data,ACCESS_nino34_data,n_lead)
print(ERA5_input_data[0])
print(UKESM_input_data[0])
print(GISS_input_data[0])
print(BCC_input_data[0])
print(ACCESS_input_data[0])

dataset_ERA5=dataset_new(ERA5_input_data,ERA5_nino34_data)
dataset_UKESM=dataset_new(UKESM_input_data,UKESM_nino34_data)
dataset_GISS=dataset_new(GISS_input_data,GISS_nino34_data)
dataset_BCC=dataset_new(BCC_input_data,BCC_nino34_data)

print(dataset_BCC)

dataset_ACCESS=dataset_new(ACCESS_input_data,ACCESS_nino34_data)

len_train=int(0.8*(len(dataset_ERA5)+len(dataset_UKESM)+len(dataset_GISS)+len(dataset_BCC)+len(dataset_ACCESS)))
len_valid=int(len(dataset_ERA5)+len(dataset_UKESM)+len(dataset_GISS)+len(dataset_BCC)+len(dataset_ACCESS))-len_train



train_dataset,valid_dataset=torch.utils.data.random_split(torch.utils.data.ConcatDataset([dataset_ERA5,dataset_UKESM,dataset_GISS,dataset_BCC,dataset_ACCESS]),[len_train,len_valid])
print(train_dataset[0])
train_loader=DataLoader(train_dataset,batch_size=128,shuffle=True)
valid_loader=DataLoader(valid_dataset,batch_size=128,shuffle=True)

torch.save(train_loader,'./dataset/'+str(n_lead)+'/'+str(bar)+'/train_loader.pth')
torch.save(valid_loader,'./dataset/'+str(n_lead)+'/'+str(bar)+'/valid_loader.pth')