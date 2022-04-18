import time as ti
import numpy as np
import scipy
import os
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator
from datetime import datetime,timedelta,timezone

from kamodo import Kamodo,kamodofy,gridify
from kamodo import unit_subs
from kamodo.util import get_unit_quantity

# each function is in its own shared library *.cpython-37m-darwin.so (Python 3.7.x, MacOS)
#import kamodo_ccmc.readers.OpenGGCM.read_grid as rg
#import kamodo_ccmc.readers.OpenGGCM.read_b_grids as rbg
#import kamodo_ccmc.readers.OpenGGCM.readmagfile3d as rmhd
# functions are now bundled in a single reader libraray
import kamodo_ccmc.readers.OpenGGCM.readOpenGGCM as ropgm

# this dictionary is not being used yet
model_varnames={
    ### 3D variables - to be aggregated into 4D ###
    'bx':['B_x','x component of magnetic field',0,'GSE','car',['time','x','y','z'],'nT'],
    'by':['B_y','y component of magnetic field',1,'GSE','car',['time','x','y','z'],'nT'],
    'bz':['B_z','z component of magnetic field',2,'GSE','car',['time','x','y','z'],'nT'],
    'bx1':['B1_x','x component of magnetic field (on grid cell faces)',3,'GSE','car',['time','gx_bx','gy_bx','gz_bx'],'nT'],
    'by1':['B1_y','y component of magnetic field (on grid cell faces)',4,'GSE','car',['time','gx_by','gy_by','gz_by'],'nT'],
    'bz1':['B1_z','z component of magnetic field (on grid cell faces)',5,'GSE','car',['time','gx_bz','gy_bz','gz_bz'],'nT'],
    'ex':['E_x','x component of electric field (on grid cell edges)',6,'GSE','car',['time','gx_ex','gy_ex','gz_ex'],'mV/m'],
    'ey':['E_y','y component of electric field (on grid cell edges)',7,'GSE','car',['time','gx_ey','gy_ey','gz_ey'],'mV/m'],
    'ez':['E_z','z component of electric field (on grid cell edges)',8,'GSE','car',['time','gx_ez','gy_ez','gz_ez'],'mV/m'],
    'vx':['V_x','x component of plasma velocity',9,'GSE','car',['time','x','y','z'],'km/s'],
    'vy':['V_y','y component of plasma velocity',10,'GSE','car',['time','x','y','z'],'km/s'],
    'vz':['V_z','z component of plasma velocity',11,'GSE','car',['time','x','y','z'],'km/s'],
    'rr':['N','plasma number denstity (hydrogen equivalent)',12,'GSE','car',['time','x','y','z'],'1/cm**3'],
    'resis':['eta','resistivity',13,'GSE','car',['time','x','y','z'],'m**2/s'],
    'pp':['P','plasma pressure',14,'GSE','car',['time','x','y','z'],'pPa'],
}

openggcm_varnames={
    'bx':{'kamodo_name':"B_x",'read_by_default':False,'dimensions':"3D",'x':"x",'y':"y",'z':"z",'description':"X component of magnetic field averaged to cell center",'unit':"nT"},
    'by':{'kamodo_name':"B_y",'read_by_default':False,'dimensions':"3D",'x':"x",'y':"y",'z':"z",'description':"Y component of magnetic field averaged to cell center",'unit':"nT"},
    'bz':{'kamodo_name':"B_z",'read_by_default':False,'dimensions':"3D",'x':"x",'y':"y",'z':"z",'description':"Z component of magnetic field averaged to cell center",'unit':"nT"},
    'bx1':{'kamodo_name':"B_x",'read_by_default':True,'dimensions':"3D",'x':"gx_bx",'y':"gy_bx",'z':"gz_bx",'description':"X component of magnetic field on cell faces",'unit':"nT"},
    'by1':{'kamodo_name':"B_y",'read_by_default':True,'dimensions':"3D",'x':"gx_by",'y':"gy_by",'z':"gz_by",'description':"X component of magnetic field on cell faces",'unit':"nT"},
    'bz1':{'kamodo_name':"B_z",'read_by_default':True,'dimensions':"3D",'x':"gx_bz",'y':"gy_bz",'z':"gz_bz",'description':"Z component of magnetic field on cell faces",'unit':"nT"},
    'ex':{'kamodo_name':"E_x",'read_by_default':True,'dimensions':"3D",'x':"gx_ex",'y':"gy_ex",'z':"gz_ex",'description':"X component of electric field on cell edges",'unit':"nT"},
    'ey':{'kamodo_name':"E_y",'read_by_default':True,'dimensions':"3D",'x':"gx_ey",'y':"gy_ey",'z':"gz_ey",'description':"Y component of electric field on cell edges",'unit':"nT"},
    'ez':{'kamodo_name':"E_z",'read_by_default':True,'dimensions':"3D",'x':"gx_ez",'y':"gy_ez",'z':"gz_ez",'description':"Z component of electric field on cell edges",'unit':"nT"},
    'vx':{'kamodo_name':"V_x",'read_by_default':True,'dimensions':"3D",'x':"x",'y':"y",'z':"z",'description':"X component of plasma velocity on cell centers",'unit':"nT"},
    'vy':{'kamodo_name':"V_y",'read_by_default':True,'dimensions':"3D",'x':"x",'y':"y",'z':"z",'description':"Y component of plasma velocity on cell centers",'unit':"nT"},
    'vz':{'kamodo_name':"V_z",'read_by_default':True,'dimensions':"3D",'x':"x",'y':"y",'z':"z",'description':"Z component of plasma velocity on cell centers",'unit':"nT"},
    'xjx':{'kamodo_name':"J_x",'read_by_default':True,'dimensions':"3D",'x':"x",'y':"y",'z':"z",'description':"X component of current density on cell centers",'unit':"muA/m**2"},
    'xjy':{'kamodo_name':"J_y",'read_by_default':True,'dimensions':"3D",'x':"x",'y':"y",'z':"z",'description':"X component of current density on cell centers",'unit':"muA/m**2"},
    'xjz':{'kamodo_name':"J_z",'read_by_default':True,'dimensions':"3D",'x':"x",'y':"y",'z':"z",'description':"X component of current density on cell centers",'unit':"muA/m**2"},
    'rr':{'kamodo_name':"rho",'read_by_default':True,'dimensions':"3D",'x':"x",'y':"y",'z':"z",'description':"plasma mass density on cell centers",'unit':"kg/m**3"},
    'pp':{'kamodo_name':"P",'read_by_default':True,'dimensions':"3D",'x':"x",'y':"y",'z':"z",'description':"plasma pressure",'unit':"pPa"},
    'resis': {'kamodo_name':"eta",'read_by_default':True,'dimensions':"3D",'x':"x",'y':"y",'z':"z",'description':"plasma resistivity on cell centers",'unit':"m**2/s"}
}

# variable linkage to grid position vectors are established during variable registration
# these are gx_bx, gy_bx, ... gz_ez affecting magnetic field (b1x,b1y,b1z) and electric field (ex,ey,ez)


def MODEL():
    from numpy import array, zeros, abs, NaN, unique, insert, diff, where
    from time import perf_counter
    from os.path import isfile, basename
    from kamodo import Kamodo
    print('KAMODO IMPORTED!')
    #    from netCDF4 import Dataset 
    from kamodo.readers.reader_utilities import regdef_4D_interpolators, regdef_3D_interpolators
    
    #    class OpenGGCM_GM(Kamodo):
    class MODEL(Kamodo):
        '''OpenGCM magnetosphere reader
           reads 4D NetCDF file of outputs aggragated into hourly 4D data.
           If hourly file isn't there attempts to convert 3D *.3df* output files.
        '''
        def __init__(self,filename,variables_requested=[],
                     runname = "noname", filetime=False,
                     printfiles=False, gridded_int=True, 
                     fulltime=True, verbose=False,
                     missing_value=np.NAN,**kwargs):
            super(MODEL, self).__init__()
            t0=perf_counter() # profiling time stamp
            file_prefix = basename(full_file_prefix)  # runname.3df.ssss
            file_dir = full_file_prefix.split(file_prefix)[0]
            if isfile(full_file_prefix+'.nc'):  #file already prepared!
                nc_file = full_file_prefix+'.nc'  # input file name: file_dir/YYYY-MM-DD_HH.nc
                self.conversion_test = True
            else:  #file not prepared, prepare it
                try:
                    from openggcm_to_nc import openggcm_combine_magnetosphere_files
                    nc_file = openggcm_combine_magnetosphere_files(full_file_prefix)
                    self.conversion_test = True
                except:
                    self.conversion_test = False
                    return 
        nc_data = Dataset(nc_file, 'r')      
        file_seconds=filename[-6:]
        grid_file_b=filename[:-11]+".grid"
        grid_file=filename[:-11]+".grid2"
        print('grid file: ',grid_file)
        print('seconds: ',file_seconds)
        openggcm_missing=1.e30
        self.missing_value=missing_value
        
        all_variables={"B_x":'bx',"B_y":'by',"B_z":'bz',
                       "B1_x":'bx1',"B1_y":'by1',"B1_z":'bz1',
           "V_x":'vx',"V_y":'vy',"V_z":'vz',
           "J_x":'xjx',"J_y":'xjy',"J_z":'xjz',
           "rho":'rr',"P":'pp',"eta":'resis'}
        all_variable_units={"B_x":'nT',"B_y":'nT',"B_z":'nT',
                            "B1_x":'nT',"B1_y":'nT',"B1_z":'nT',
           "V_x":'km/s',"V_y":'km/s',"V_z":'km/s',
           "J_x":'muA/m^2',"J_y":'muA/m^2',"J_z":'muA/m^2',
           "rho":'kg/m^3',"P":'pPa',"eta":'m^2/s'}
        all_variable_to_GSE_factors={"B_x":-1,"B_y":-1,"B_z":1,
                                     "B1_x":-1,"B1_y":-1,"B1_z":1,
           "V_x":-1,"V_y":-1,"V_z":1,
           "J_x":-1,"J_y":-1,"J_z":1,
           "rho":1,"P":1,"eta":1}
        if variables_requested is None:
            variables=all_variables.keys()
        else:
            variables=variables_requested
        self.verbose=True

        if os.path.exists(grid_file_b):            
            gx=np.zeros(5000)
            gy=np.zeros(5000)
            gz=np.zeros(5000)
            nx,ny,nz,gx,gy,gz = ropgm.read_grid_for_vector(10,grid_file_b,' ',gx,gy,gz)
            self.x=-np.flip(gx[0:nx])
            self.y=-np.flip(gy[0:ny])
            self.z=gz[0:nz]
            self.nx=nx
            self.ny=ny
            self.nz=nz
        else:
            if os.path.exists(grid_file):
                gx,gy,gz,nx,ny,nz=ropgm.read_grid_dir(grid_file) # python reader of ASCII file
                self.x=-np.flip(gx)
                self.y=-np.flip(gy)
                self.z=gz
                self.nx=nx
                self.ny=ny
                self.nz=nz
            else:
                raise IOError('no grid file {} or {} found'.format(grid_file,grid_file_b))
            
        fielddata=np.zeros(shape=(self.nx,self.ny,self.nz),dtype=float,order='F');
        self.variables=dict()
        self.Time=None
        self.symbol_registry=dict()
        self.signatures=dict()
        if os.path.exists(grid_file_b):
            gx_bx=np.zeros(nx)
            gy_bx=np.zeros(ny)
            gz_bx=np.zeros(nz)
            gx_by=np.zeros(nx)
            gy_by=np.zeros(ny)
            gz_by=np.zeros(nz)
            gx_bz=np.zeros(nx)
            gy_bz=np.zeros(ny)
            gz_bz=np.zeros(nz)
            gx_ex=np.zeros(nx)
            gy_ex=np.zeros(ny)
            gz_ex=np.zeros(nz)
            gx_ey=np.zeros(nx)
            gy_ey=np.zeros(ny)
            gz_ey=np.zeros(nz)
            gx_ez=np.zeros(nx)
            gy_ez=np.zeros(ny)
            gz_ez=np.zeros(nz)
            nx1,ny1,nz1,gx_bx,gy_bx,gz_bx = ropgm.read_grid_for_vector(10,grid_file_b,'bx',gx_bx,gy_bx,gz_bx)
            if nx1 > 0:
                self.gx_bx=-np.flip(gx_bx[0:nx1])
            if ny1 > 0:
                self.gy_bx=-np.flip(gy_bx[0:ny1])
            if nz1 > 0:
                self.gz_bx=gz_bx[0:nz1]
            nx1,ny1,nz1,gx_by,gy_by,gz_by = ropgm.read_grid_for_vector(10,grid_file_b,'by',gx_by,gy_by,gz_by)
            if nx1 > 0:
                self.gx_by=-np.flip(gx_by[0:nx1])
            if ny1 > 0:
                self.gy_by=-np.flip(gy_by[0:ny1])
            if nz1 > 0:
                self.gz_by=gz_by[0:nz1]
            nx1,ny1,nz1,gx_bz,gy_bz,gz_bz = ropgm.read_grid_for_vector(10,grid_file_b,'bz',gx_bz,gy_bz,gz_bz)
            if nz1 > 0:
                self.gx_bz=-np.flip(gx_bz[0:nx1])
            if nz1 > 0:
                self.gy_bz=-np.flip(gy_bz[0:ny1])
            if nz1 > 0:
                self.gz_bz=gz_bz[0:nz1]

            nx1,ny1,nz1,gx_ex,gy_ex,gz_ex = ropgm.read_grid_for_vector(10,grid_file_b,'ex',gx_ex,gy_ex,gz_ex)
            if nx1 > 0:
                self.gx_ex=-np.flip(gx_ex[0:nx1])
            if ny1 > 0:
                self.gy_ex=-np.flip(gy_ex[0:ny1])
            if nz1 > 0:
                self.gz_ex=gz_ex[0:nz1]
            nx1,ny1,nz1,gx_ey,gy_ey,gz_ey = ropgm.read_grid_for_vector(10,grid_file_b,'ey',gx_ey,gy_ey,gz_ey)
            if nx1 > 0:
                self.gx_ey=-np.flip(gx_ey[0:nx1])
            if ny1 > 0:
                self.gy_ey=-np.flip(gy_ey[0:ny1])
            if nz1 > 0:
                self.gz_ey=gz_ey[0:nz1]
            nx1,ny1,nz1,gx_ez,gy_ez,gz_ez = ropgm.read_grid_for_vector(10,grid_file_b,'ez',gx_ez,gy_ez,gz_ez)
            if nz1 > 0:
                self.gx_ez=-np.flip(gx_ez[0:nx1])
            if nz1 > 0:
                self.gy_ez=-np.flip(gy_ez[0:ny1])
            if nz1 > 0:
                self.gz_ez=gz_ez[0:nz1]
        

        for var in variables:
            units=all_variable_units[var]
            varname=all_variables[var]
            factor=all_variable_to_GSE_factors[var]
        
            fieldarray,asciitime,it = rmhd.read_3d_field(filename,
                    fielddata,varname,nx,ny,nz);
            fieldarray[(fieldarray == openggcm_missing)]=missing_value
# a date+time string comes as an metadata for each variable
# usually there is the same time for each variable in each file, so we do this only once
            print(varname,fieldarray.min(),fieldarray.max(),asciitime,it)
            if self.Time is None:
                asciitime_arr=asciitime.split()
                if len(asciitime_arr) > 4:
                    datetime_str_arr=(asciitime_arr[3]).split(b':')
                    year=int(datetime_str_arr[0])
                    mon=int(datetime_str_arr[1])
                    day=int(datetime_str_arr[2])
                    hour=int(datetime_str_arr[3])
                    minute=int(datetime_str_arr[4])
                    sec=int(np.floor(float(datetime_str_arr[5])))
                    self.Time=datetime(year,mon,day,hour,minute,sec,tzinfo=timezone.utc)
                    
                    self.time=np.float64((self.Time-datetime(year,mon,day,0,0,0,tzinfo=timezone.utc)).total_seconds())/3600.

            self.variables[var]=dict(units=units,data=factor*np.flip(np.flip(fieldarray,axis=0),axis=1))
            self.register_variable(var,units,gridded_int)
        
        
        
    def register_variable(self, varname, units,gridded_int):
        interpolator,x_,y_,z_ = self.get_grid_interpolator(varname)
        
        # store the interpolator
        self.variables[varname]['interpolator'] = interpolator

        def interpolate(xvec):  
            return self.variables[varname]['interpolator'](xvec)
            
        
        # update docstring for this variable
        interpolate.__doc__ = "A function that returns {} in [{}].".format(varname,units)

        self[varname] = kamodofy(interpolate, 
                           units = units, 
                           citation = "Rastaetter et al 2020",
                                 data = None ) # ,x=x_,y=y_,z=z_)
        if gridded_int:
            self[varname + '_ijk'] = kamodofy(gridify(self[varname],
                                                 # use variable-specific grid as defaults
                                                 # this is necessary if
                                                 # the interpolator called without arguments
                                                 # directly returned the 3D data
                                                      x_i = x_,  
                                                      y_j = y_, 
                                                      z_k = z_),
                                                 # use a global set of grid positions
                                                 # some edge locations may generate NaN values
                                                 #     x_i = self.x,  
                                                 #     y_j = self.y, 
                                                 #     z_k = self.z),
                                              units = units,
                                              citation = "Rastaetter et al 2020",
                                              data = self.variables[varname]['data'])

    def get_grid_interpolator(self, varname):
        """create a regular grid interpolator for this variable"""
        data =  self.variables[varname]['data']
        # (default) positions on plasma grid
        x_=self.x
        y_=self.y
        z_=self.z
        # x,y,z positions on staggered grid
        if varname == 'B1_x':
            x_=self.gx_bx
            y_=self.gy_bx
            z_=self.gz_bx
        if varname == 'B1_y':
            x_=self.gx_by
            y_=self.gy_by
            z_=self.gz_by
        if varname == 'B1_z':
            x_=self.gx_bz
            y_=self.gy_bz
            z_=self.gz_bz
        if varname == 'E1_x':
            x_=self.gx_ex
            y_=self.gy_ex
            z_=self.gz_ex
        if varname == 'E1_y':
            x_=self.gx_ey
            y_=self.gy_ey
            z_=self.gz_ey
        if varname == 'E1_z':
            x_=self.gx_ez
            y_=self.gy_ez
            z_=self.gz_ez
            
        interpolator = RegularGridInterpolator((x_, y_, z_), data, 
                                                bounds_error = False,
                                               fill_value = self.missing_value)
        return interpolator,x_,y_,z_

    
