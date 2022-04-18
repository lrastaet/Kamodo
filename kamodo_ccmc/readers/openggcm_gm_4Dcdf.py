import time as ti
import numpy as np
import scipy
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator
from datetime import datetime,timedelta,timezone

from kamodo import Kamodo,kamodofy,gridify
from kamodo import unit_subs
from kamodo.util import get_unit_quantity

# each function is in its own shared library *.cpython-37m-darwin.so (Python 3.7.x, MacOS)

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
    'rr':['Np','plasma number denstity (hydrogen equivalent)',12,'GSE','car',['time','x','y','z'],'1/cm**3'],
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
    from netCDF4 import Dataset
    from os.path import isfile, basename
    from kamodo import Kamodo
    from kamodo_ccmc.readers.reader_utilities import regdef_4D_interpolators
    print('KAMODO IMPORTED!')
    #    from netCDF4 import Dataset 
    from kamodo_ccmc.readers.reader_utilities import regdef_4D_interpolators, regdef_3D_interpolators
    
    #    class OpenGGCM_GM(Kamodo):
    class MODEL(Kamodo):
        '''OpenGCM magnetosphere reader'''
        def __init__(self,full_file_prefix, variables_requested=[], runname = "noname",
                     filetime=False, verbose=False, gridded_int=True, 
                     printfiles=False, fulltime=True, 
                     missing_value=np.NAN,**kwargs):
            super(MODEL, self).__init__()
            file_prefix = basename(full_file_prefix)  # runname.3df.ssss
            file_dir = full_file_prefix.split(file_prefix)[0]
            if isfile(full_file_prefix+'.nc'):  #file already prepared!
                nc_file = full_file_prefix+'.nc'  # input file name: file_dir/YYYY-MM-DD_HH.nc
                self.conversion_test = True
            else:  #file not prepared, prepare it
                from openggcm_gm_tocdf import openggcm_combine_magnetosphere_files
                conversion_suceeeded = openggcm_combine_magnetosphere_files(full_file_prefix)
                self.conversion_test = conversion_suceeded
                if not conversion_suceeded:
                    self.conversion_test = False
                    return 
                
            t0=perf_counter() # profiling time stamp
            #determine type of prefix: for a day or for a hour
            if '-' in file_prefix: day_flag=False
            else: day_flag=True
            
            cdf_data = Dataset(nc_file, 'r')      
            files = cdf_data.file.split(',')
            modelname=cdf_data.model

            self.near_Earth_boundary_radius=cdf_data.near_Earth_boundary_radius
            self.near_Earth_boundary_radius_unit=cdf_data.near_Earth_boundary_radius_units
            self.files=files
            self.missing_value=NaN
        
            if variables_requested is None:
                variables=all_variables.keys()
            else:
                variables=variables_requested
            self.verbose=verbose

            self.modelname = modelname
            self._registered=0

            t = array(cdf_data.variables['_time'])  #hours since midnight
            if len(t)>1: self.dt = diff(t).max()*3600.  #t is in hours since midnight
            else: self.dt = 0
            if verbose:
                print("dt:",self.dt)  # this dt may not be constant
    
#establish time attributes first for file searching
            self.filedate=cdf_data.filedate
            self.filetimes=[datetime.strptime(self.filedate,'%Y-%m-%d').replace(tzinfo=timezone.utc)+timedelta(seconds=time*3600) for time in t]
            self.datetimes=[datetime.strftime(filetime,'%Y-%m-%d %H:%M:%S') for filetime in self.filetimes]

            if filetime and not fulltime: #(used when searching for neighboring files below)
                return  #return times as is to prevent recursion


            add_boundary=False # now needed since NetCDF files include first data from next hour
            if fulltime and add_boundary:  #add boundary time (default value)
                #find other files with same pattern
                from glob import glob

                list_file=full_file_prefix[:full_file_prefix.index('3df')+3]+'_list'
                f=open(list_file)
                list_data=f.readlines()
                f.close()
                files=[list_line.split()[0] for list_line in list_data[1:]]
                if day_flag: 
                    file_prefixes = unique([basename(f)[:11] for f in files\
                                            if '.nc' not in basename(f)])
                else:  #give prefix for hourly files
                    file_prefixes = unique([basename(f)[:14] for f in files\
                                            if '.nc' not in basename(f)])
                
                #find closest file by utc timestamp
                #swmf_ie has an open time at the end, so need a beginning time from the next file
                #files are automatically sorted by YYMMDD, so next file is next in the list
                current_idx = where(file_prefixes==file_prefix)[0]
                if current_idx+1==len(file_prefixes):
                    print('No later file available.')
                    filecheck = False  
                    if filetime:
                        return   
                else:
                    min_file_prefix = file_dir+file_prefixes[current_idx+1][0]  #+1 for adding an end time
                    kamodo_test = MODEL(min_file_prefix, filetime=True, fulltime=False)
                    if not kamodo_test.conversion_test: 
                        print('No later file available.')
                        filecheck = False  
                        if filetime:
                            return        
                    else:
                        time_test = abs(kamodo_test.filetimes[0]-self.filetimes[1])  
                        if time_test<=self.dt:  #if nearest file time at least within one timestep (hrs)
                            filecheck = True
                        
                            #time only version if returning time for searching
                            if filetime:
                                kamodo_neighbor = MODEL(min_file_prefix, fulltime=False, filetime=True)
                                self.datetimes[1] = kamodo_neighbor.datetimes[0]
                                self.filetimes[1] = kamodo_neighbor.filetimes[0]
                                return  #return object with additional time (for SF code) 
                            
                            #get kamodo object with same requested variables to add to each array below
                            if verbose: print(f'Took {perf_counter()-t0:.3f}s to find closest file.')
                            kamodo_neighbor = MODEL(min_file_prefix, 
                                                    variables_requested=variables_requested, 
                                                    fulltime=False)
                            self.datetimes[1] = kamodo_neighbor.datetimes[0]
                            self.filetimes[1] = kamodo_neighbor.filetimes[0]
                            short_data = kamodo_neighbor.short_data                                
                            if verbose: print(f'Took {perf_counter()-t0:.3f}s to get data from closest file.')
                        else:
                            print(f'No later file found within {self.dt:.1f}s.')
                            filecheck = False 
                            if filetime:
                                return                    
            else:
                filecheck = False  
            #perform initial check on variables_requested list
            if len(variables_requested)>0 and fulltime:
                test_list = [value[0] for key, value in model_varnames.items()]
                err_list = [item for item in variables_requested if item not in test_list]
                if len(err_list)>0: print('Variable name(s) not recognized:', err_list)
                
            #get list of variables possible in these files using first file
            if len(variables_requested)>0:
                gvar_list = [key for key, value in model_varnames.items() \
                                 if value[0] in variables_requested and \
                                     key in cdf_data.variables.keys()]  # file variable names
                #check for variables requested but not available
                if len(gvar_list)!=len(variables_requested):
                    err_list = [value[0] for key, value in model_varnames.items() \
                                 if value[0] in variables_requested and \
                                     key not in cdf_data.variables.keys()]
                    if len(err_list)>0: print('Some requested variables are not available:', err_list)
            else:
                avoid_list = []
                gvar_list = [key for key in cdf_data.variables.keys() \
                             if key in model_varnames.keys() and \
                                 key not in avoid_list]
                                   
            # Store variable's data and units, transposing the 2D+time array.
            variables = {model_varnames[key][0]:{'units':model_varnames[key][-1],
                                   'data':array(cdf_data.variables[key])}\
                              for key in gvar_list} 
                
            #prepare and return data only for first timestamp
            if not fulltime:  
                cdf_data.close()
                variables['time'] = self.filetimes[0]
                self.short_data = variables
                return            
    
            #return if only one file found because interpolator code will break
            if len(self.files)<2:
                print('Not enough files found with given file prefix.')
                return 
    
            #store variables
#            self.filename = files
            self.runname = runname
            self.missing_value = NaN
            self.modelname = 'OpenGGCM_GM'
            self._registered = 0
            if printfiles: 
                print('Files:')
                for file in self.filename: print(file)
    
            #### Store coordinate data as class attributes   
            if filecheck:
                new_time = ts_to_hrs(short_data['time'], self.filedate)  #new time in hours since midnight
                self._time = append(t, new_time) 
            else: 
                self._time = t
                
            #store coordinate data
            #self._radius = array(cdf_data.variables['radius'])
            self._x = array(cdf_data.variables['_x'])  
            self._y = array(cdf_data.variables['_y'])
            self._z = array(cdf_data.variables['_z'])
            cdf_data.close()
            if verbose: print(f'Took {perf_counter()-t0:.6f}s to read in data')
    
            #register interpolators for each variable
            varname_list, self.variables = [key for key in variables.keys()], {}  #store original list b/c gridded interpolators
            t_reg = perf_counter()
            for varname in varname_list:  #all are 3D variables
                if filecheck:  #if neighbor found
                    #append data for first time stamp, transpose and register
                    data_shape = list(variables[varname]['data'].shape)
                    data_shape[0]+=1  #add space for time
                    new_data = zeros(data_shape)                
                    new_data[:-1,:,:,:] = variables[varname]['data']  #put in current data
                    new_data[-1,:,:,:] = short_data[varname]['data'][0,:,:,:]  #add in data for additional time
                else:
                    new_data = variables[varname]['data']
                self.variables[varname] = dict(units = variables[varname]['units'], data = new_data)           
                self.register_variable(self.variables[varname]['units'], 
                                       self.variables[varname]['data'],
                                       varname,
                                       gridded_int)
            if verbose: print(f'Took {perf_counter()-t_reg:.5f}s to register '+\
                              f'{len(varname_list)} variables.')
            if verbose: print(f'Took a total of {perf_counter()-t0:.5f}s to kamodofy '+\
                              f'{len(gvar_list)} variables.')
        
        #define and register a 4D variable-----------------------------------------
        def register_variable(self, units, variable, varname, gridded_int):
            x_,y_,z_ = self.get_grid(varname) # variable may have different grid positions in the staggered grid of the model
            xvec_dependencies = {'time':'hr','x':'R_E','y':'R_E','z':'R_E'}

            self = regdef_4D_interpolators(self, units, variable, self._time, 
                                           x_, y_, z_ , varname, 
                                           xvec_dependencies, gridded_int)             
        def get_grid(self, varname):
            """fetch the grid positon for this variable"""
            data =  self.variables[varname]['data']
            # (default) positions on plasma grid
            x_=self._x
            y_=self._y
            z_=self._z
            # x,y,z positions on staggered grid
            if varname == 'B1_x':
                x_=self._x_bx
                y_=self._y_bx
                z_=self._z_bx
            if varname == 'B1_y':
                x_=self._x_by
                y_=self._y_by
                z_=self._z_by
            if varname == 'B1_z':
                x_=self._x_bz
                y_=self._y_bz
                z_=self._z_bz
            if varname == 'E1_x':
                x_=self._x_ex
                y_=self._y_ex
                z_=self._z_ex
            if varname == 'E1_y':
                x_=self._x_ey
                y_=self._y_ey
                z_=self._z_ey
            if varname == 'E1_z':
                x_=self._x_ez
                y_=self._y_ez
                z_=self._z_ez
            
            return x_,y_,z_

    
    return MODEL
        
        
