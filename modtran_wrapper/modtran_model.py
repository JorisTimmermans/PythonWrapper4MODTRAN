__author__ = "J Timmermans"
__copyright__ = "Copyright 2017 J Timmermans"
__version__ = "1.0 (13.05.2017)"
__license__ = "GPLv3"
__email__ = "j.timmermans@ucl.ac.uk"


# load same modules as the 'stronconstraint_utils.py
import cPickle

import json
import os
import glob
import sys
import matplotlib.pyplot as plt
import numpy as np

import gp_emulator

import platform

import subprocess
import multiprocessing

import time                             # required for StandardStateACRMF and grab_acrmf_emulators
from collections import OrderedDict     # required for StandardStateACRMF
from eoldas_ng import State             # required for StandardStateACRMF
import copy
import datetime

import shutil
import random

class modtran():
    def __init__(self, path='/home/ucfajti/Simulations/MODTRAN/Mod5.2.1/'):

        # acquired output of MODTRAN run
        data = np.zeros([36,3])
        # data[ 0]=[   0.2000,  989.756, 293.30]
        data[ 0]=[   0.0000, 1001.325, 293.30]#  7.590E-01 4.260E+20 3.161E+22 9.097E-01 2.611E-04 3.383E-03 3.337E+04     0.000     0.000     0.000
        data[ 1]=[   1.1667,  884.661, 288.99]#  6.246E-01 1.919E+20 1.931E+22 8.252E-01 2.370E-04 3.380E-03 2.906E+04     0.000     0.000     0.000
        data[ 2]=[   2.1333,  789.215, 284.45]#  5.131E-01 7.733E+19 1.115E+22 7.479E-01 2.149E-04 3.394E-03 2.534E+04     0.000     0.000     0.000
        data[ 3]=[   3.1000,  701.385, 278.60]#  4.224E-01 2.488E+19 5.752E+21 6.787E-01 1.951E-04 3.508E-03 2.217E+04     0.000     0.000     0.000
        data[ 4]=[   4.0667,  622.830, 272.80]#  3.474E-01 8.516E+18 3.056E+21 6.155E-01 1.770E-04 3.616E-03 1.943E+04     0.000     0.000     0.000
        data[ 5]=[   5.0333,  551.658, 267.00]#  2.845E-01 2.464E+18 1.489E+21 5.570E-01 1.602E-04 3.726E-03 1.702E+04     0.000     0.000     0.000
        data[ 6]=[   6.0000,  486.998, 261.20]#  2.317E-01 9.532E+17 8.364E+20 5.026E-01 1.446E-04 3.891E-03 1.489E+04     0.000     0.000     0.000
        data[ 7]=[   7.0000,  426.000, 254.70]#  1.865E-01 3.500E+17 4.548E+20 4.509E-01 1.297E-04 4.229E-03 1.295E+04     0.000     0.000     0.000
        data[ 8]=[   8.0000,  371.999, 248.20]#  1.497E-01 1.129E+17 2.315E+20 4.040E-01 1.162E-04 4.454E-03 1.128E+04     0.000     0.000     0.000
        data[ 9]=[   9.0000,  323.999, 241.70]#  1.198E-01 3.684E+16 1.183E+20 3.614E-01 1.039E-04 4.850E-03 9.818E+03     0.000     0.000     0.000
        data[10]=[  10.0000,  280.999, 235.30]#  9.505E-02 1.048E+16 5.623E+19 3.219E-01 9.260E-05 5.071E-03 8.526E+03     0.000     0.000     0.000
        data[11]=[  11.0000,  243.000, 228.80]#  7.518E-02 1.239E+15 1.719E+19 2.863E-01 8.236E-05 6.201E-03 7.406E+03     0.000     0.000     0.000
        data[12]=[  12.0000,  208.999, 222.30]#  5.891E-02 9.213E+13 4.151E+18 2.534E-01 7.290E-05 6.828E-03 6.411E+03     0.000     0.000     0.000
        data[13]=[  13.0000,  179.001, 215.80]#  4.586E-02 5.296E+12 8.780E+17 2.236E-01 6.432E-05 8.104E-03 5.541E+03     0.000     0.000     0.000
        data[14]=[  14.0000,  153.000, 215.70]#  3.353E-02 1.513E+12 4.013E+17 1.912E-01 5.500E-05 1.016E-02 4.631E+03     0.000     0.000     0.000
        data[15]=[  15.0000,  129.999, 215.70]#  2.421E-02 5.050E+11 1.970E+17 1.625E-01 4.673E-05 9.814E-03 3.854E+03     0.000     0.000     0.000
        data[16]=[  16.0000,  111.000, 215.70]#  1.765E-02 3.468E+11 1.394E+17 1.387E-01 3.990E-05 1.006E-02 3.233E+03     0.000     0.000     0.000
        data[17]=[  17.0000,   95.000, 215.70]#  1.293E-02 2.389E+11 9.902E+16 1.187E-01 3.415E-05 1.004E-02 2.726E+03     0.000     0.000     0.000
        data[18]=[  18.0000,   81.200, 216.80]#  9.350E-03 1.674E+11 7.049E+16 1.010E-01 2.904E-05 1.220E-02 2.287E+03     0.000     0.000     0.000
        data[19]=[  19.0000,   69.500, 217.90]#  6.780E-03 1.253E+11 5.193E+16 8.598E-02 2.473E-05 1.558E-02 1.925E+03     0.000     0.000     0.000
        data[20]=[  20.0000,   59.500, 219.20]#  4.911E-03 9.650E+10 3.879E+16 7.317E-02 2.105E-05 1.768E-02 1.622E+03     0.000     0.000     0.000
        data[21]=[  21.0000,   51.000, 220.40]#  3.569E-03 7.664E+10 2.947E+16 6.238E-02 1.794E-05 1.809E-02 1.371E+03     0.000     0.000     0.000
        data[22]=[  22.0000,   43.700, 221.60]#  2.592E-03 6.061E+10 2.233E+16 5.316E-02 1.529E-05 1.862E-02 1.160E+03     0.000     0.000     0.000
        data[23]=[  23.0000,   37.600, 222.80]#  1.898E-03 5.077E+10 1.749E+16 4.549E-02 1.309E-05 1.869E-02 9.867E+02     0.000     0.000     0.000
        data[24]=[  24.0000,   32.200, 223.90]#  1.379E-03 3.980E+10 1.320E+16 3.877E-02 1.115E-05 1.873E-02 8.364E+02     0.000     0.000     0.000
        data[25]=[  25.0000,   27.700, 225.10]#  1.009E-03 3.212E+10 1.015E+16 3.317E-02 9.542E-06 1.924E-02 7.124E+02     0.000     0.000     0.000
        data[26]=[  30.0000,   13.200, 233.70]#  2.126E-04 8.475E+09 2.392E+15 1.523E-02 4.380E-06 1.288E-02 3.223E+02     0.000     0.000     0.000
        data[27]=[  35.0000, 6.519971, 245.20]#  4.713E-05 2.083E+09 5.583E+14 7.168E-03 2.062E-06 7.707E-03 1.507E+02     0.000     0.000     0.000
        data[28]=[  40.0000, 3.329993, 257.50]#  1.115E-05 5.231E+08 1.361E+14 3.486E-03 1.003E-06 3.180E-03 7.307E+01     0.000     0.000     0.000
        data[29]=[  45.0000, 1.759993, 269.90]#  2.834E-06 1.519E+08 3.697E+13 1.758E-03 5.057E-07 9.556E-04 3.679E+01     0.000     0.000     0.000
        data[30]=[  50.0000, 0.951001, 275.70]#  7.930E-07 4.329E+07 1.044E+13 9.299E-04 2.675E-07 3.145E-04 1.945E+01     0.000     0.000     0.000
        data[31]=[  55.0000, 0.514999, 269.30]#  2.437E-07 1.259E+07 3.121E+12 5.155E-04 1.483E-07 1.121E-04 1.078E+01     0.000     0.000     0.000
        data[32]=[  60.0000, 0.272001, 257.10]#  7.460E-08 3.365E+06 8.927E+11 2.852E-04 8.204E-08 4.479E-05 5.962E+00     0.000     0.000     0.000
        data[33]=[  70.0000, 0.067000, 218.10]#  6.290E-09 1.554E+05 5.570E+10 8.281E-05 2.382E-08 4.002E-06 1.731E+00     0.000     0.000     0.000
        data[34]=[  80.0000, 0.012000, 174.10]#  3.166E-10 2.520E+03 1.591E+09 1.858E-05 5.345E-09 4.489E-07 3.883E-01     0.000     0.000     0.000
        data[35]=[ 100.0000, 0.000258, 190.50]#  1.223E-13 3.529E-02 1.170E+05 3.651E-07 1.050E-10 1.764E-08 5.842E-03     0.000     0.000     0.000

        H                                       =   data[:,0]#[]
        P                                       =   data[:,1]#[isort]
        T                                       =   data[:,2]#[isort]


        isort                                   =   np.argsort(P)
        H                                       =   H[isort]#[]
        P                                       =   P[isort]#[isort]
        T                                       =   T[isort]#[isort]

        # P_ECMWF                                 =   [1000, 	975, 	950, 	925, 	900, 	875, 	850, 	825, 	800, 	775,
        #                                               750, 	700, 	650, 	600, 	550,    500, 	450, 	400, 	350, 	300,
        #                                               250, 	225, 	200, 	175, 	150, 	125, 	100, 	 70, 	 50,     30,
        #                                                20, 	 10,      7, 	  5, 	  3, 	  2, 	  1];

        Pi_ECMWF                                =   [1000.0,  700.0, 400.0, 175.0, 100.0, 7.0]
        Pi_MODTRAN                              =   [1.0, 0.515, 0.272, 0.067, 0.012, 0.000258]  # sampling
        Pi                                      =   np.append(Pi_ECMWF, Pi_MODTRAN)
        Hi                                      =   np.interp(Pi,P,H)
        Ti                                      =   np.interp(Pi,P,T)


        # plt.figure()
        # plt.subplot(2,1,1)
        # plt.plot(P,H)
        # plt.plot(Pi,Hi,'o')
        # plt.subplot(2,1,2)
        # plt.plot(T,H)
        # plt.plot(Ti,Hi,'o')


        # create default dict
        default_dict                            =   dict()
        default_dict['H1str']                   =   814.          # sensor altitude (in km) (SLSTR height)
        default_dict['H2str']                   =   0.2           # sensor altitude (in km)
        default_dict['H20str']                  =   4.0           # total column water vapor (in [g/cm2])
        default_dict['Rhostr']                  =   0.4           # reflection coefficient
        default_dict['Tempstr']                 =   283.15        # lower atmospheric boundary temperature
        default_dict['AOTstr']                  =   0.2            # aerosol optical thickness
        default_dict['O3str']                   =   0.4            # vertical ozone ATM.cm
        default_dict['VZAstr']                  =   20.           # viewing zenith angle
        default_dict['SZAstr']                  =   10.           # viewing zenith angle
        default_dict['RAAstr']                  =   30.           # viewing zenith angle
        default_dict['isometric']               =   0.5           # viewing zenith angle
        default_dict['ross']                    =   0.5           # viewing zenith angle
        default_dict['li']                      =   0.5           # viewing zenith angle

        for i in xrange(len(Pi)):
            default_dict['press%02.0f'%(i+1)]   =   Pi[i]
            default_dict['tatm%02.0f'%(i+1)]    =   Ti[i]
            default_dict['height%02.0f'%(i+1)]  =   Hi[i]

        # plt.figure(figsize=[15,4])
        # plt.subplot(2,1,1)
        # plt.plot(P,H)
        # plt.plot(Pi,Hi,'o')
        # plt.subplot(2,1,2)
        # plt.plot(H,T)
        # plt.plot(Hi,Ti,'o')


        #
        # default_dict['press01']                =   1.00e+03
        # default_dict['press02']                =   7.00e+02
        # default_dict['press03']                =   4.00e+02
        # default_dict['press04']                =   7.00e+01
        # default_dict['press05']                =   7.00e+00
        # default_dict['press06']                =   1.00e+00
        # default_dict['press07']                =   2.72e-01
        # default_dict['press08']                =   6.70e-02
        # default_dict['press09']                =   1.20e-02
        # default_dict['press10']                =   2.58e-04
        #
        # default_dict['height01']                =   2.30e-01
        # default_dict['height02']                =   2.91e+00
        # default_dict['height03']                =   7.07e+00
        # default_dict['height04']                =   1.82e+01
        # default_dict['height05']                =   3.22e+01
        # default_dict['height06']                =   4.76e+01
        # default_dict['height07']                =   6.00e+01
        # default_dict['height08']                =   7.00e+01
        # default_dict['height09']                =   8.00e+01
        # default_dict['height10']                =   1.00e+02
        #
        # default_dict['tatm01']                  =   317.294
        # default_dict['tatm02']                  =   292.299
        # default_dict['tatm03']                  =   266.753
        # default_dict['tatm04']                  =   235.915
        # default_dict['tatm05']                  =   263.315
        # default_dict['tatm06']                  =   297.960
        # default_dict['tatm07']                  =   257.100
        # default_dict['tatm08']                  =   218.100
        # default_dict['tatm09']                  =   174.100
        # default_dict['tatm10']                  =   190.500


        parameters                               =   ['H1str', 'H2str', 'SZAstr', 'VZAstr', 'RAAstr', 'H20str', 'Tempstr', 'AOTstr', 'O3str'] # 'Rhostr',
        min_vals                                 =   [777.,    0. ,     0.0,      0.0,        0.0,      0.5,     263.15,      0.0,     0.23]
        max_vals                                 =   [999.,    0.4,     45.,     45.0,      180.0,      8.0,     323.15,      1.8,     0.55]

        # parameters                               =   ['H1str', 'H2str', 'H20str', 'Tempstr', 'AOTstr', 'O3str', 'Lmeas_TOA', 'SZAstr', 'VZAstr','RAAstr'] # 'Rhostr',
        # min_vals                                 =   [777.,    0. ,     0.5,       263.15,     0.0,     0.23,      0.0,      0.0,      0.0,     0.0]
        # max_vals                                 =   [999.,    0.4,     8.0,       323.15,     1.8,     0.55,    333.3,      45.,     45.0,   180.0]

        self.default_dict                       =   default_dict
        self.data                               =   data

        self.modtran_dir                        =   path
        self.homedir                            =   '/home/ucfajti/Simulations/Python/modtran/'

        self.parameters                         =   parameters
        self.min_vals                           =   min_vals
        self.max_vals                           =   max_vals

    def UpdateTP5(self,parameter_dict, outputstr='Synergy/', scenariostr=''):

        # set parameters
        inputdir                                =   self.homedir
        inputfile                               =   'MODTRAN_master.tp5' #self.inputfile   #"/tmp/acrmf_runfile"

        # modtran_dir                           =   '/home/ucfajti/Simulations/Python/modtran/' #self.acrm_dir    #"/home/ucfajti/Simulations/Fortran/acrm/acrmf122012/"
        modtran_dir                             =   self.modtran_dir
        outputdir                               =   modtran_dir + outputstr
        if len(scenariostr)==0:
            scenariostr                         =   '%06.0f' % (random.random()*1e6)
        outputfile                              =   'MODTRAN_simulation_' + scenariostr + '.tp5'

        try:
            os.stat(outputdir)
        except:
            os.mkdir(outputdir)

        # update default variables
        self.default_dict.update ( parameter_dict )

        # copy to unique name
        shutil.copy(inputdir + inputfile, outputdir +outputfile)

        #Read template file ('master_config')
        fp                                      =   open (outputdir + outputfile, 'r')
        config_file                             =   fp.read()
        fp.close()

        fp = open(outputdir + outputfile, 'w')
        for line in config_file.split('\n'):
            for words in line.split(' '):
                for varname,v in self.default_dict.iteritems():


                    if (varname in words):
                        wordstrs                =   words.split('_')
                        word                    =   wordstrs[0] # word
                        prefix                  =   wordstrs[1] # prefix
                        format                  =   wordstrs[2] # number-string
                        string                  =   prefix + format % v

                        line                    =   line.replace (words, string)
            fp.write ( line+"\n" )
        fp.close()


        return outputfile

    def UpdateTP5_kernel(self,parameter_dict, outputstr='Synergy/', scenariostr=''):

        # set parameters
        inputdir                                =   self.homedir
        inputfile                               =   'MODTRAN_master_kernels.tp5' #self.inputfile   #"/tmp/acrmf_runfile"

        modtran_dir                             =   self.modtran_dir
        outputdir                               =   modtran_dir + outputstr
        if len(scenariostr)==0:
            scenariostr                         =   '%06.0f' % (random.random()*1e6)
        outputfile                              =   'MODTRAN_kernels_simulation_' + scenariostr + '.tp5'

        try:
            os.stat(outputdir)
        except:
            os.mkdir(outputdir)

        # update default variables
        self.default_dict.update ( parameter_dict )

        # self.default_dict['VISstr']           =   3.912 / (default_dict['AOTstr'] + 0.01159)

        # copy to unique name
        shutil.copy(inputdir + inputfile, outputdir +outputfile)

        #Read template file ('master_config')
        fp                                      =   open (outputdir + outputfile, 'r')
        config_file                             =   fp.read()
        fp.close()

        fp = open(outputdir + outputfile, 'w')
        for line in config_file.split('\n'):
            for words in line.split(' '):
                for varname,v in self.default_dict.iteritems():


                    if (varname in words):
                        wordstrs                =   words.split('_')
                        word                    =   wordstrs[0] # word
                        prefix                  =   wordstrs[1] # prefix
                        format                  =   wordstrs[2] # number-string
                        string                  =   prefix + format % v

                        line                    =   line.replace (words, string)
            fp.write ( line+"\n" )
        fp.close()


        return outputfile

    def UpdateTP5_Tprofile(self,parameter_dict, outputstr='Synergy/', scenariostr=''):

        H                                       =   self.data[:,0]#[]
        P                                       =   self.data[:,1]#[isort]
        T                                       =   self.data[:,2]#[isort]

        # set parameters
        inputdir                                =   self.homedir
        inputfile                               =   'MODTRAN_master_Tprofile.tp5' #self.inputfile   #"/tmp/acrmf_runfile"

        modtran_dir                             =   self.modtran_dir
        outputdir                               =   modtran_dir + outputstr
        if len(scenariostr)==0:
            scenariostr                         =   '%06.0f' % (random.random()*1e6)
        outputfile                              =   'MODTRAN_Tprofile_simulation_' + scenariostr + '.tp5'

        try:
            os.stat(outputdir)
        except:
            os.mkdir(outputdir)

        # preprocessing
        x                                       =   []
        y1                                      =   []
        y2                                      =   []

        # update default variables
        self.default_dict.update ( parameter_dict )

        # use updated specified heights to define full atmospheric profile
        i                                       =   0
        while i>=0:
            try:
                x.append(self.default_dict['press%02.0f' % (i+1)])
                y1.append(self.default_dict['height%02.0f' % (i+1)])
                y2.append(self.default_dict['tatm%02.0f' % (i+1)])
            except:
                i                               =   -999
            i                                   =   i+1


        # Pressure at surface =
        Hs                                      =   self.default_dict['H2str']
        H2m                                     =   Hs
        Ps                                      =   np.interp(H2m,y1,x)

        # Create atmospheric pressure profile from surface to TOA
        xi_top                                  =   [280.999000,   243.000000,  208.999000,  179.001000,  153.000000,  129.999000,  111.000000,    95.000000,   81.200000,   69.500000,
                                                      59.500000,    51.000000,   43.700000,   37.600000,   32.200000,   27.700000,   13.200000,    6.519971,     3.329993,    1.759993,
                                                       0.951001,     0.514999,    0.272001,    0.067000,    0.012000,    0.000258]
        xi_bottom                               =   np.linspace(Ps,323.999000, 36-len(xi_top))
        xi                                      =   np.append(xi_bottom, xi_top)
        xi                                      =   np.array(xi)

        # interpolate profile variables
        # x                                       =   [1000, 175, 30, 5]
        # y1                                      =   [self.default_dict['Height01'], self.default_dict['Height02'], self.default_dict['Height03'], self.default_dict['Height04']]
        # y2                                      =   [self.default_dict['Tatm01'], self.default_dict['Tatm02'], self.default_dict['Tatm03'], self.default_dict['Tatm04']]


        # xi2                                     =   [989.756000,   884.661000,  789.215000,  701.385000,  622.830000,  551.658000,  486.998000,   426.000000,  371.999000,  323.999000,


        # interpolate temperature and pressure's from MODTRAN-pressure levels to newly defined pressure-levels
        isx                                     =   np.argsort(x)
        x                                       =   np.array([x[i] for i in isx])
        y1                                      =   np.array([y1[i] for i in isx])
        y2                                      =   np.array([y2[i] for i in isx])

        # transformation function acquired by fitting in matlab over original data
        xiprime                                 =   -336.3*xi**0.0204 + 385.9
        xprime                                  =   -336.3*x**0.0204 + 385.9
        iprime                                  =   np.argsort(xprime)

        # yli[i1]                                 =   -np.log(np.interp(xi[i1],x, np.exp(-y1)))

        i1                                      =   xi<200
        i2                                      =   xi>=200
        y1i                                     =   np.zeros_like(xi)
        y1i[i1]                                 =   np.interp(xiprime[i1],xprime[iprime], y1[iprime])
        y1i[i2]                                 =   np.interp(xi[i2],x, y1)

        # plt.figure()
        # plt.plot( -336.3*P**(0.0204) + 385.9,H,'.')
        # plt.plot(xprime,y1)

        # from scipy.interpolate import interp1d
        # f2                                      =   interp1d(x,y1)
        # y1i                                     =   f2(xi)

        #
        i1                                      =   (xi>1) * (xi<=100)
        i2                                      =   (xi<2) + (xi>100)

        # transformation function acquired by fitting in matlab over original data
        xiprime                                 =   96.96*xi**-0.3153 + 193.6
        xprime                                  =   96.96*x**-0.3153 + 193.6
        iprime                                  =   np.argsort(xprime)

        y2i                                     =   np.zeros_like(xi)
        y2i[i1]                                 =   np.interp(xiprime[i1],xprime[iprime], y2[iprime])
        y2i[i2]                                 =   np.interp(xi[i2],x,y2)

        ###############################################################
        #evaluate the transformation against simple linear interpolation
        ###############################################################
        # old interpolations
        # y1i_old                                 =   np.interp(xi,x,y1)
        # y2i_old                                 =   np.interp(xi,x,y2)
        # #
        # #
        #
        # plt.figure(figsize=[15,10])
        # plt.subplot(2,1,1)
        # plt.plot(P,T)
        # plt.plot(x,y2,'d')
        # plt.plot(xi,y2i,'o')
        # plt.plot(xi,y2i_old,'.')
        #
        #
        # plt.xlim([x[-6],x[-4]])
        # plt.legend(['data','sampling','new interp','old interp'])
        #
        #
        # plt.figure(figsize=[15,10])
        # plt.subplot(2,1,1)
        # plt.semilogx(P,T)
        # plt.semilogx(x,y2,'d')
        # plt.semilogx(xi,y2i,'o')
        # plt.semilogx(xi,y2i_old,'.')
        # # plt.xlim([x[-5],x[-4]])
        # plt.legend(['data','sampling','new interp','old interp'])
        #
        # plt.subplot(2,1,2)
        # plt.plot(T,H)
        # plt.plot(y2i,y1i,'.')
        # plt.legend(['data','new interp','old interp'])
        #



        # # update default variables
        # self.default_dict.update ( parameter_dict )

        # print self.default_dict
        # self.default_dict['VISstr']           =   3.912 / (default_dict['AOTstr'] + 0.01159)
        # print self.default_dict['height%02.0f' % (1)]

        for i in xrange(len(xi)):
            self.default_dict['Height%02.0f' % (i+1)]   =   y1i[i]
            self.default_dict['Tatm%02.0f'   % (i+1)]     =   y2i[i]
            self.default_dict['Press%02.0f'  % (i+1)]    =   xi[i]

        # plt.figure(figsize=[15,10])
        # plt.subplot(2,1,1)
        # plt.semilogx(x, y1)
        # plt.semilogx(self.data[:,1],self.data[:,0])
        # plt.legend(['Training_set','Mid-Latitude-Summer'])
        # plt.subplot(2,1,2)
        # plt.semilogx(y2, y1)
        # plt.semilogx(self.data[:,2],self.data[:,0])
        # plt.legend(['Training_set','Mid-Latitude-Summer'])


        ################################################################################################################
        # copy to unique name
        shutil.copy(inputdir + inputfile, outputdir +outputfile)

        #Read template file ('master_config')
        fp                                      =   open (outputdir + outputfile, 'r')
        config_file                             =   fp.read()
        fp.close()

        fp = open(outputdir + outputfile, 'w')
        for line in config_file.split('\n'):
            for words in line.split(' '):
                for varname,v in self.default_dict.iteritems():


                    if (varname in words):
                        wordstrs                =   words.split('_')
                        word                    =   wordstrs[0] # word
                        prefix                  =   wordstrs[1] # prefix
                        format                  =   wordstrs[2] # number-string
                        string                  =   prefix + format % v

                        line                    =   line.replace (words, string)
            fp.write ( line+"\n" )
        fp.close()


        return outputfile

    def Wait4MODTRAN2Finish(self, scenarios, waittime =0.5, verbose = 0):
        exec_str_seq                            =   'nohup ./' + path2modtran_exe + '>/dev/null 2>&1 &' # ' &'
        ncores                                  =   multiprocessing.cpu_count()
        print self.modtran_dir
        # change workingdir
        os.chdir(self.modtran_dir)

        ntrain                                  =   len(scenarios)

        pass_error                              =   1
        time_per_run                            =   np.NaN
        counter                                 =   0
        tic                                     =   datetime.datetime.now()
        time_reinitialize                       =   30
        print '- Waiting for simulations to finish, please wait'
        while pass_error:
            # check if run is finished based on presence of output files (Tp7, WRN)
            not_finished                        =   np.ones(ntrain)
            not_run                             =   np.zeros(ntrain)
            for i,scenario in enumerate(scenarios):
                filename_tp7                    =   scenario.replace('.tp5','.tp7')
                filename_wrn                    =   scenario.replace('.tp5','.wrn')

                path2file_tp7                       =   modtran_model.modtran_dir + filename_tp7
                path2file_wrn                   =   modtran_model.modtran_dir + filename_wrn

                try:
                    info_tp7                    =   os.stat(path2file_tp7)
                    info_wrn                    =   os.stat(path2file_wrn)

                    if (info_tp7.st_size>0) + (info_wrn.st_size>0):
                        not_finished[i]         =   0
                        not_run[i]              =   0
                    else:
                        not_finished[i]         =   1
                        not_run[i]              =   0

                except:
                    not_run[i]                  =   1
                    not_finished[i]             =   1

            # Determine how many instances of Modtran are still running
            os.system('ps aux | grep Mod90_5.2.1.exe > log.txt')
            with open('log.txt') as f:
                list_processes                  =   f.readlines()
            nr_processes                        =   len(list_processes)

            # start some un-started runs
            if (np.mod(int(counter*waittime),time_reinitialize)==(time_reinitialize-1)) * (np.sum(not_run)>0):
                print '     > Reinitialize some simulations (%02.0f)' % np.sum(not_run)
                scenarios_not_run               =   [scenarios[i] for i in np.where(not_run)[0]]
                modtran_model.ExecuteScenarios(scenarios_not_run, exec_str_par)
                print '     > Waiting for (reinitialized) simulations to finish, please wait'
                time_reinitialize               =   time_reinitialize*3

            # total passed time since execution
            toc                                 =   datetime.datetime.now()
            time_total                          =   (toc - tic).total_seconds()
            time_cut                            =   ((time_per_run) * ntrain /  ncores)*3

            # derive run-time of 1st number of simulations to be finished (this is set only once)
            if np.isnan(time_per_run)* np.any(1-not_finished):
                time_per_run                    =   time_total / (np.max([sum(1-not_finished),1.]))
                time_runup                      =   ntrain * 2.0                                                            #time since 1st launch to execution of wait module
                time_per_run                    =   time_per_run + time_runup
                counter                         =   0


            # Determine if another wait-module needs to be continued
            if np.any([sum(not_finished)/ntrain <=0.2, sum(not_finished)<8, time_total>time_cut, nr_processes>2]): # & ntrain>20:
                if counter==0:
                    print '     >Almost all simulations are finished, still %2.0f simulations to go, please wait' % sum(not_finished)

                # Define new wait-period for iteration to break (4)
                waittime_max                    =   60.
                if (counter*waittime)<=waittime_max:
                    pass_error                  =   np.any(not_finished)

                else:
                    pass_error                  =   0

                # Define number of left over instances before to break (2+i)
                if nr_processes>2:
                    pass_error                  =   0

            else:
                pass_error                      =   np.sum(not_finished)

            counter                             =   counter +1
            time.sleep(waittime)



        os.chdir(self.homedir)


        # wait till all processes are finished running
        nr_processes                          =   100
        counter                                 =   0
        print '- Waiting for processes to finish'
        while (nr_processes>2)*(counter*1.<60*5):
            time.sleep(1.)
            os.system('ps aux | grep Mod90_5.2.1.exe > log.txt')
            with open('log.txt') as f:
                list_processes                  =   f.readlines()
            nr_processes                        =   len(list_processes)

        # check if all simulations were completed, or if out of time was performd
        I_unfinished                            =   np.where(not_finished)
        print '- Simulations Finished'


        # scenario_missing                        =   modtran_model.FindMissingScenarios(scenarios_s)
        # if len(scenario_missing)>10:
        #
        #     print sum(not_finished)/ntrain <=0.2
        #     print sum(not_finished)<8
        #     print time_total>time_cut
        #
        #     print I_unfinished
        #
        #     print scenario_missing
        #
        #     import pdb
        #     pdb.set_trace()



        os.chdir(self.homedir)
        return I_unfinished, not_finished

    def ListFinishedModtranSimulations(self):
        # dirs                                            =   [x[0] for x in os.walk('.')]
        os.chdir(self.modtran_dir)
        names                                           =   os.listdir('.')
        files                                           =   []
        directories                                     =   []
        for name in names:
            file                                        =   glob.glob(name + "/mod5root_.in")
            if len(file)>0:
                directories.append(name)

        os.chdir(self.homedir)
        return directories

    def ReadTP7(self, scenarios):
        Values2                                 =   []
        for scenario in scenarios:
            path2file                           =   scenario.replace('.tp5','.tp7')


            # read output
            os.chdir(self.modtran_dir)
            with open(path2file) as f:
                content = f.readlines()
            os.chdir(self.homedir)

            values                      =   []
            for i,line in enumerate(content):
                if i==10:
                    varnames            =   line.split()
                elif i>10:

                    if line <> ' -9999.\n':
                        words           =   line.split()

                        V               =   [eval(word) for word in words]

                        if len(V)<>len(varnames):
                            V2          =   np.zeros(len(varnames))*np.NaN
                            V2[0:3]     =   V[0:3]
                            V2[4:-1]    =   V[3:-1]
                        else:
                            V2          =   V
                        values.append(V2)
            if len(values)>0:
                Values                      =   np.array(values)
            else:
                Values                      =   Values*np.NaN

            # except:
            #
            #     # in case of erroneous data
            #     Values                          =   Values*np.NaN




            # store Values
            Values2.append(Values)
        Values2                                 =   np.array(Values2)



        if 'FREQ'in varnames:

            # calculate wavelength
            ifreq                               =   [i for i,name in enumerate(varnames) if name=='FREQ'][0]
            freq                                =   Values2[:,:,ifreq]                           # [cm-1]
            wl                                  =   1.e7/freq                                # [nm]

            #update data
            varnames[ifreq]                     =   'WL'
            Values2[:,:,ifreq]                  =   wl

            # sort the data acoording to WL
            isort                               =   np.argsort(wl[0])
            Values2                             =   Values2[:,isort,:]
            freq                                =   freq[0,isort]
            wl                                  =   wl[0,isort]

        Values_dict                             =   dict(zip(varnames,np.transpose(Values2,[2, 1, 0])))

        # # change units to # [W cm-2 sr-1 um-1]
        # dummy                                 =   Values2[:,0,0]
        # Dummy, Freq                           =   np.meshgrid(dummy, freq)
        # for name,Values in Values_dict.iteritems():
        #     if name in ['PTH_THRML','SING_SCAT','TOTAL_RAD','DRCT_RFLT','SURF_EMIS','TOTAL_RAD','GRND_RFLT']:
        #         b=1
        #         # pdb.set_trace()
        #         # Values                      =   Values* (Freq**2) *1e-4        # [W cm-2 sr-1 um-1]
        #         # values                      =   values*1e-4            # [W cm-2 sr-1 um-1]
        #     elif name in ['TOA_SUN','SOL@OBS','REF_SOL']:
        #         a=1
        #         # pdb.set_trace()
        #         # Values                      =   Values* (Freq**2) *1e-4        # [W cm-2 sr-1 um-1]
        #     Values_dict[name]                 =   Values


        WL                                      =   Values_dict['WL'][:,0]
        freq                                    =   1.e7/WL
        for name,Values in Values_dict.iteritems():
            for isc in xrange(np.shape(Values)[1]):
                if name in ['PTH_THRML','SING_SCAT','TOTAL_RAD','DRCT_RFLT','SURF_EMIS','TOTAL_RAD','GRND_RFLT']:
                    Values[:,isc]               =   Values[:,isc]* freq**2 * 1e-4 *1e4              # [W m-2 sr-1 um-1]
                elif name in ['TOA_SUN','SOL@OBS','REF_SOL']:
                    Values[:,isc]               =   Values[:,isc]* freq**2 * 1e-4 *1e4              # [W m-2 sr-1 um-1]
            Values_dict[name]                   =   Values

        # plt.semilogy(wl,Values_dict['TOTAL_RAD']* Freq**2 * 1e-4 * 1e-3)





        return Values_dict

    def ReadTP6(self, scenarios):
        values                                  =   []
        values2                                  =   []

        values                                  =   np.ones(len(scenarios))*np.NaN
        values2                                 =   np.ones(len(scenarios))*np.NaN

        for isc,scenario in enumerate(scenarios):
            path2file                           =   scenario.replace('.tp5','.tp6')

            # read output
            os.chdir(self.modtran_dir)
            with open(path2file) as f:
                content = f.readlines()
            os.chdir(self.homedir)

            for i,line in enumerate(content):
                if 'THE WATER COLUMN IS BEING SET TO THE MAXIMUM,' in line:
                    bline       =   line
                    v           =   float(bline.split()[-4])
                    v2          =   np.NaN

                if ('*** THE WATER PROFILE WAS DECREASED TO FIT THE INPUT WATER COLUMN' in line) + ('THE WATER PROFILE WAS INCREASED TO FIT THE INPUT WATER COLUMN' in line):
                    v            =   np.NaN
                    v2           =   -float(content[i+3].split()[-4])


            # values.append(v)
            # values2.append(v2)
            values[isc]         =   v
            values2[isc]         =   v2
            # values2.append(v2)


        # print np.all([np.isnan(values2), np.isnan(values)], axis=0)
        if np.all((values2<0) * (values>0)==0):
            V                               =   np.nansum([values2, values],axis=0)
            V                               =   V[0:-1:3]


        else:
            V                               =   np.nansum([np.abs(values2), np.abs(values)],axis=0)
            V                               =   V[0:-1:3]
            import pdb
            pdb.set_trace()

        # V                                   =   np.nansum([np.abs(values2), np.abs(values)],axis=0)
        # V                                   =   np.nansum([np.abs(values2), np.abs(values)],axis=0)
        # V                                   =   V[0:-1:3]

        return V

    def ReadDifferentScenarios(self, filename_modtroot='mod5root_.in'):

        # change workingdir
        os.chdir(self.modtran_dir)

        fp                                      =   open (filename_modtroot, 'r')
        scenarios                               =   fp.read()
        scenarios                               =   scenarios.split('\n')
        scenarios                               =   scenarios[0:-1]             # to remove the last (empty) line
        fp.close()

        os.chdir(self.homedir)

        return scenarios

    def T18(self,Values_dict,I, tts=None, tto=None):
        anomalyfix                              =   0.

        # post process output
        WL                                      =   Values_dict['WL'][:,0]
        Nwl                                     =   np.shape(WL)[0]
        Nscen                                   =   np.shape(I)[0]
        rho_dd                                  =   np.zeros([Nwl,Nscen])
        tau_oo                                  =   np.zeros([Nwl,Nscen])
        tau_do                                  =   np.zeros([Nwl,Nscen])
        L_u_TOA                                 =   np.zeros([Nwl,Nscen])
        L_d_BOA                                 =   np.zeros([Nwl,Nscen])
        for i in xrange(Nscen):
            # translate from MODTRAN outputs to T18 nomenclature
            TRAN                                =   Values_dict['TOT_TRANS'][:,I[i,0]]

            PTEM_000                            =   Values_dict['PTH_THRML'][:,I[i,0]]
            PTEM_050                            =   Values_dict['PTH_THRML'][:,I[i,1]]
            PTEM_100                            =   Values_dict['PTH_THRML'][:,I[i,2]]

            SFEM_000                            =   Values_dict['SURF_EMIS'][:,I[i,0]]
            SFEM_050                            =   Values_dict['SURF_EMIS'][:,I[i,1]]
            SFEM_100                            =   Values_dict['SURF_EMIS'][:,I[i,2]]

            GRFL_000                            =   Values_dict['GRND_RFLT'][:,I[i,0]]
            GRFL_050                            =   Values_dict['GRND_RFLT'][:,I[i,1]]
            GRFL_100                            =   Values_dict['GRND_RFLT'][:,I[i,2]]

            GSUN_000                            =   Values_dict['DRCT_RFLT'][:,I[i,0]]      # for optical
            GSUN_050                            =   Values_dict['DRCT_RFLT'][:,I[i,1]]      # for optical
            GSUN_100                            =   Values_dict['DRCT_RFLT'][:,I[i,2]]      # for optical

            # TOA_000                           =   Values_dict['TOA_SUN'][:,I[i,0]]      # for optical
            # TOA_050                           =   Values_dict['TOA_SUN'][:,I[i,1]]      # for optical
            # TOA_100                           =   Values_dict['TOA_SUN'][:,I[i,2]]      # for optical

            if Values_dict.has_key('SOL_SCAT'):
                PATH_000                        =   Values_dict['SOL_SCAT'][:,I[i,0]]
                PATH_050                        =   Values_dict['SOL_SCAT'][:,I[i,1]]
                PATH_100                        =   Values_dict['SOL_SCAT'][:,I[i,2]]
            else:
                SING_000                        =   Values_dict['SING_SCAT'][:,I[i,0]]
                SING_050                        =   Values_dict['SING_SCAT'][:,I[i,1]]
                SING_100                        =   Values_dict['SING_SCAT'][:,I[i,2]]

                MULT_000                        =   Values_dict['MULT_SCAT'][:,I[i,0]]
                MULT_050                        =   Values_dict['MULT_SCAT'][:,I[i,1]]
                MULT_100                        =   Values_dict['MULT_SCAT'][:,I[i,2]]

                PATH_000                        =   SING_000 + MULT_000
                PATH_050                        =   SING_050 + MULT_050
                PATH_100                        =   SING_100 + MULT_100

            # combine variables
            GTOT_000                            =   GRFL_000 + SFEM_000
            GTOT_050                            =   GRFL_050 + SFEM_050
            GTOT_100                            =   GRFL_100 + SFEM_100

            ATMO_000                            =   PATH_000 + PTEM_000
            ATMO_050                            =   PATH_050 + PTEM_050
            ATMO_100                            =   PATH_100 + PTEM_100


            #
            # plt.close()
            # plt.semilogy(WL, PTEM_050,'y')
            # plt.semilogy(WL, SFEM_050,'r')
            # plt.semilogy(WL, PATH_050,'b')
            # plt.semilogy(WL, GRFL_050,'g')
            # plt.xlim([2000, 12500])


            # First stage of processing
            Delta_ATMO_100                      =   ATMO_100 - ATMO_000

            Delta_GTOT_100                      =   GTOT_100 - GTOT_000
            Delta_GTOT_050                      =   GTOT_050 - GTOT_000

            Delta_PTEM_100                      =   PTEM_100-PTEM_000

            # Second stage of processing

            with np.errstate(divide='ignore', invalid='ignore'):
                taudo                           =   (Delta_ATMO_100/(Delta_GTOT_100+anomalyfix)) * TRAN                     # T7
                rhodd                           =   (Delta_GTOT_100 - 2*Delta_GTOT_050)/(Delta_GTOT_100 - Delta_GTOT_050    + anomalyfix)
                # rhodd2                          =   (Delta_GTOT_100 - 2*Delta_GTOT_050)/(Delta_GTOT_100-Delta_GTOT_050)
                rhodd[rhodd>1]                 =   1
                rhodd[rhodd<0]                 =   0

                Ttau_oo                         =   (Delta_PTEM_100/(Delta_ATMO_100 + anomalyfix))*(1-rhodd)*Delta_GTOT_100

                Ldown                           =   (Ttau_oo + (1-rhodd)*SFEM_000)/(TRAN+anomalyfix)
                Lup                             =   PTEM_000 - (Delta_ATMO_100/(Delta_GTOT_100+anomalyfix)) * SFEM_000      # T15

            # Atmospheric baseline functions according to T18
            tau_oo[:,i]                         =   np.max([TRAN,   TRAN*0.],   axis=0)
            tau_do[:,i]                         =   np.max([taudo,  taudo*0.],  axis=0)
            L_u_TOA[:,i]                        =   np.max([Lup,    Lup*0.],    axis=0)         # W cm-2 sr-1 / cm-1
            L_d_BOA[:,i]                        =   np.max([Ldown,  Ldown*0.],  axis=0)

            # plt.plot(WL, taudo)

            # note (as by conversation with Wout verhoef)
            # please note that both Lup and Ldown are (in the 10um domain) not sensitive to Tair (2m). This is because at 10um,
            #   1) (Delta_PTEM_100/(Delta_ATMO_100) is about 1, leading to Ttauoo ~ (SFEM100-SFEM000)(1-rho_dd)
            #   2) (Delta_ATMO_100/(Delta_GTOT_100) is about 0.01

            # plt.subplot(3,1,1)
            # plt.plot(WL,Delta_ATMO_100/Delta_GTOT_100)
            # plt.ylim([0,1])
            # plt.subplot(3,1,2)
            # plt.plot(WL,SFEM_000)
            # plt.subplot(3,1,3)
            # plt.plot(WL,PTEM_000)
            # plt.ylim([0,0.1])
            ######################################################################
            ########## Optical ##################################################$
            ######################################################################
            # if not(tts==None):
            #     # preprocessing
            #     EOS                           =   Values_dict['TOA_SUN'][:,I[i,0]]*np.cos(tts)/np.pi                         # T1
            #
            #     tausstauoo                    =   GSUN_100/EOS                                                            # T8
            #     b                             =   -np.log(tausstauoo) * np.cos(tts) * np.cos(tto) / (np.cos(tts) + np.cos(tto))
            #
            #     # reflectances (spherical albedo + atmospheric bidirectional reflectance)
            #     rhodd                         =   rhodd
            #     rhoso                         =   PATH_000/EOS                                                            # T2
            #
            #     # transmissivities
            #     tauss                         =   np.exp(-b/np.cos(tts))
            #     tauoo                         =   np.exp(-b/np.cos(tto))
            #
            #     taudo                         =   taudo
            #     tausd                         =   (GTOT_100*(1-rhodd)/GSUN_100-1)*tauss
            #
            #     #
            #     rho_dd[:,i]                   =   np.max([rhodd,  rhodd*0.],  axis=0)


        return WL, TRAN, tau_oo, tau_do, L_u_TOA, L_d_BOA

    def ShowOutput(self, Values_dict, tau_oo=None, tau_do=None, L_u_TOA=None, L_d_BOA=None):
        varnames                                =   [name for name in Values_dict.iterkeys()]
        Nplots                                  =   len(varnames)-1.
        Nc                                      =   2.
        Nr                                      =   np.ceil(Nplots/Nc)
        wl                                      =   Values_dict['WL']
        plt.ion()


        # # show GRND_RFLT for all scenarios
        # string                                =   []
        # plt.figure(figsize=[15,4])
        # for i in xrange(len(training_set)):
        #     plt.plot(WL,Values_dict['GRND_RFLT'][:,1+3*i])
        #     str                               =   '%4.2f' % training_set[i]
        #     string.append(str)
        # plt.legend(string)
        # plt.xlim([3e3,4.5e3])

        # show All outputs for all scenarios
        h1                                      =   plt.figure(figsize=[15,15])
        counter                                 =   0
        for name, value in Values_dict.iteritems():
            if name<>'WL':
                counter                         =   counter+1
                plt.subplot(Nr,Nc,counter)
                plt.plot(wl,value)

                if name in ['PTH_THRML','SING_SCAT','TOTAL_RAD','DRCT_RFLT','SURF_EMIS','TOTAL_RAD','GRND_RFLT']:
                    plt.title('Unfiltered [W cm-2 sr-1 nm-1/]')
                elif name in ['TOA_SUN','SOL@OBS','REF_SOL']:
                    plt.title('Unfiltered [ W cm-2 nm-1]')

                plt.xlabel('wavelength [nm]')
                plt.ylabel(name)
                plt.xlim([3000, 4500])

        # show T18 outputs
        if not(tau_oo==None):
            h1                                  =   plt.figure(figsize=[15,15])
            plt.subplot(2,2,1)
            plt.plot(wl,tau_oo)
            plt.title('Unfiltered')
            plt.ylabel('tau_oo')
            plt.subplot(2,2,2)
            plt.plot(wl,tau_do)
            plt.title('Unfiltered [W cm-2 sr-1 nm-1/]')
            plt.ylabel('tau_do')
            plt.subplot(2,2,3)
            plt.plot(wl,L_u_TOA)
            plt.title('Unfiltered [W cm-2 sr-1 nm-1/]')
            plt.ylabel('L_up_TOA')
            plt.subplot(2,2,4)
            plt.plot(wl,L_d_BOA)
            plt.title('Unfiltered')
            plt.ylabel('L_do_BOA')

    def ExecuteScenarios(self, scenarios_s, exec_str):
        error                                   =   False
        # change workingdir
        os.chdir(self.modtran_dir)
        print '- Start scenarios (%02.0f)' % len(scenarios_s)
        for scenario in scenarios_s:
            print '\t*' + scenario
            try:
                # delete previous mod5root file
                os.system('rm -rf mod5root.in')
            except:
                a=1


            # write mod5root file with single scenario
            bufsize                             =   0
            fp                                  =   open ('mod5root.in', 'w',bufsize)
            fp.write ( scenario+"\n" )
            fp.flush()
            os.fsync(fp)
            fp.close()

            # Provide system time to flush file from mem to disk
            error                               =   True
            while error:
                time.sleep(.5)
                try:
                    a                           =   os.stat('mod5root.in')
                    error                       =   a.st_size==0
                except:
                    error                       =   True

            # remove any (potential) outputs of (previous) erroneous runs
            try:
                filename_tp7                    =   scenario.replace('.tp5','.tp7')
                info                            =   os.stat(filename_tp7)
                os.system('rm -rf ' + filename_tp7)
            except:
                a=1

            # Run one MODTRAN instance for each scenario
            time.sleep(0.5)
            os.system(exec_str)

            # wait some time untill Modtran has started running (to be identified by the creation of a tp7 file)
            counter                             =   0
            error                               =   True
            while error * (counter*0.1)<20:
                counter                         =   counter +1
                filename_tp7                    =   scenario.replace('.tp5','.tp7')
                try:
                    info                        =   os.stat(filename_tp7)
                except:
                    time.sleep(0.1)

        os.chdir(self.homedir)
        return error

    def IdentifyErroneousRuns(self,I,scenarios,trainingset,Values_dict=[]):
        training_set                            =   trainingset*1.,
        '- Remove erroneous MODTRAN runs'
        # Identify erroneous scenarios
        scenario_missing                        =   modtran_model.FindMissingScenarios(scenarios)

        # identify index of erroneous scenarios
        irow_                                   =    np.zeros(len(scenario_missing))
        for isc,sc_m in enumerate(scenario_missing):
            ii                                  =   np.where([sc_m == scenario for i,scenario in enumerate(scenarios)])[0][0]
            irow_[isc]                          =   np.where(np.array(I)==ii)[0][0]
        irow_                                   =   np.unique(irow_)
        irow_                                   =   sorted(irow_,reverse=True)

        # remove unavailable scenarios from the index
        for irow in irow_:
            # delete row from index
            I                                   =   np.delete(I,irow, axis=0)
            training_set                        =   np.delete(training_set, irow, axis = 0)

        if len(Values_dict)>0:
            # remove erroneous scenarios from the index
            irow_                               =   np.where(np.any(np.any(np.isnan(Values_dict['TOT_TRANS'][:,I]),axis =2), axis=0))[0]
            irow_                               =   sorted(irow_,reverse=True)
            for irow in irow_:
                # delete row from index
                I                               =   np.delete(I,irow, axis=0)
                training_set                    =   np.delete(training_set, irow, axis = 0)

        ntrain                                  =   np.shape(I)[0]
        return I,ntrain

    def FindMissingScenarios(self, scenarios):
        os.chdir(self.modtran_dir)

        scenario_missing                        =   []
        for scenario in scenarios:
            # print scenario
            filename_tp7                        =   scenario.replace('.tp5','.tp7')

            error                               =   1
            if os.path.isfile(filename_tp7):
                info                            =   os.stat(filename_tp7)
                if (info.st_size > 0):
                    error                       =   0
            if error:
                scenario_missing.append(scenario)
        os.chdir(self.homedir)
        return scenario_missing

    def ReRunErroneous(self, scenarios, Groupsize):
        # time.sleep(60)
        os.chdir(self.modtran_dir)
        exec_str_seq                            =   './' + path2modtran_exe
        exec_str_seq                            =   'nohup ./' + path2modtran_exe + '>/dev/null 2>&1 &' # ' &'

        # Find unfinished scenarios
        scenario_missing                        =   self.FindMissingScenarios(scenarios)

        Nsims                                   =   len(scenario_missing)
        Ngroups                                 =   int(np.ceil(float(Nsims)/Groupsize))

        for igroup in xrange(Ngroups):
            index_min                           =   igroup*Groupsize
            index_max                           =   np.min([igroup*Groupsize+Groupsize,Nsims])
            index                               =   list(np.arange(index_min, index_max))

            scenarios_s                         =   [scenario_missing[ii] for ii in index]                                     # scenarios[index]

            # run left overs in sequential mode
            error                               =   self.ExecuteScenarios(scenarios_s, exec_str_seq)

            # wait till it finished
            I_unfinished, not_finished          =   self.Wait4MODTRAN2Finish(scenarios_s,waittime=2.)


        # change back to original directory
        os.chdir(self.homedir)


        scenario_missing                        =   self.FindMissingScenarios(scenarios)

    def CreateKernelsScenarios(self, ntrain, filename_modtroot='mod5root_kernels_.in'):
        ####################################################################################
        ############################# Preprocessing4kernels       ##########################
        ####################################################################################

        # create test scenarios
        print '-Creating test scenarios'

        # create test scenarios
        parameters                               =   ['H1str', 'H2str', 'H20str', 'Tempstr', 'AOTstr', 'O3str', 'Lmeas_TOA', 'SZAstr', 'VZAstr','RAAstr', 'isotropic', 'ross', 'li'] # 'Rhostr',
        min_vals                                 =   [777.,    0. ,     0.5,       263.15,     0.0,     0.23,      0.0,        0.0,      0.0,     0.0,     0.0,        0.0,    0.0]
        max_vals                                 =   [999.,    0.4,     8.0,       323.15,     1.8,     0.55,    333.3,       45.,     45.0,   180.0,     1.0,        1.0,    1.0]

        # Reinforce low AOT and low refl
        min_vals2                                =   [777.,    0. ,     0.5,       263.15,     0.0,     0.23,      0.0,        0.0,      0.0,     0.0,     0.0,        0.0,    0.0]
        max_vals2                                =   [999.,    0.4,     8.0,       323.15,     0.5,     0.55,    333.3,       45.,     45.0,   180.0,     1.0,        1.0,    1.0]


        ntrain2                                  =   int(ntrain/3)
        ntrain1                                  =   ntrain-ntrain2
        training_set1,distributions1             =   gp_emulator.create_training_set( parameters, min_vals,  max_vals,  n_train=ntrain1)
        training_set2,distributions2             =   gp_emulator.create_training_set( parameters, min_vals2, max_vals2, n_train=ntrain2)

        trainn                                   =   np.r_[training_set1, training_set2]
        training_set                             =   trainn*1.

        # create TP5 files
        datestr                                  =   (datetime.datetime.now()).strftime('%Y%m%d-%H%M%S')
        outputstr                                =   'Synergy_' + datestr + '/'
        filenames                                =   []
        I                                        =   np.zeros([len(training_set),3]).astype('int')
        for i,xx in enumerate(training_set):
            parameter_dict                       =   dict(zip(parameters,xx))
            scenariostr                          =   '%06.0f' % i

            # print parameter_dict
            filename                             =   modtran_model.UpdateTP5_kernel(parameter_dict, outputstr, scenariostr)
            filenames.append(filename)


        # write mod5root.in
        fp = open(modtran_dir + filename_modtroot, 'w')
        for filename in filenames:
            string                               =   outputstr + filename
            fp.write ( string+"\n" )
        fp.close()

        # create backup of mod5root in the directory itself
        shutil.copy(modtran_dir + filename_modtroot, modtran_dir + outputstr + filename_modtroot)

        return filename_modtroot, training_set, parameters

    def CreateIsotropicScenarios(self, ntrain, filename_modtroot='mod5root_.in'):
        ####################################################################################
        ############################# Preprocessing4isotropic     ##########################
        ####################################################################################


        print '-Creating test scenarios'
        # create test scenarios
        parameters                               =   ['H1str', 'H2str', 'H20str', 'Tempstr', 'AOTstr', 'O3str', 'Lmeas_TOA', 'SZAstr', 'VZAstr','RAAstr', 'height01', 'height02', 'height03', 'height04', 'tatm01',  'tatm02',   'tatm03',   'tatm04'] # 'Rhostr',
        min_vals                                 =   [777.,    0. ,     0.5,       263.15,     0.0,     0.23,      0.0,      0.0,      0.0,     0.0,        ]
        max_vals                                 =   [999.,    0.4,     8.0,       323.15,     1.8,     0.55,    333.3,      45.,     45.0,   180.0,        ]

        # Reinforce low AOT and low refl
        min_vals2                                =   [777.,    0. ,     0.5,       263.15,     0.0,     0.23,      0.0,      0.0,      0.0,     0.0]
        max_vals2                                =   [999.,    0.4,     8.0,       323.15,     0.5,     0.55,    333.3,      45.,     45.0,   180.0]

        # create test scenarios
        parameters                               =   ['H1str', 'H2str', 'SZAstr', 'VZAstr', 'RAAstr', 'H20str', 'Tempstr', 'AOTstr', 'O3str'] # 'Rhostr',
        min_vals                                 =   [777.,    0. ,     0.0,      0.0,        0.0,      0.5,     263.15,      0.0,     0.23]
        max_vals                                 =   [999.,    0.4,     45.,     45.0,      180.0,      8.0,     323.15,      1.8,     0.55]

        # Reinforce low AOT and low refl
        min_vals2                                =   [777.,    0. ,     0.0,      0.0,        0.0,      0.5,     263.15,      0.0,     0.23]
        max_vals2                                =   [999.,    0.4,     45.,     65.0,      180.0,      8.0,     323.15,      0.5,     0.55]

        parameters                               =   ['height01', 'height02', 'height03', 'height04']
        min_vals                                 =   [-6.35e-01,   1.96e+00,  5.47e+00, 1.52e+01]
        max_vals                                 =   [ 1.75e-01,   3.87e+00,  8.67e+00, 2.12e+01]

        min_vals2                                 =   [-6.35e-01,   1.96e+00,  5.47e+00, 1.52e+01]
        max_vals2                                 =   [ 1.75e-01,   3.87e+00,  8.67e+00, 2.12e+01]

        parameters                               =   ['height01']#, 'height02', 'height03', 'height04']
        min_vals                                 =   [ 0.35e-01]#,   1.96e+00,  5.47e+00, 1.52e+01]
        max_vals                                 =   [ 1.75e-01]#,   3.87e+00,  8.67e+00, 2.12e+01]

        min_vals2                                 =   [ 0.35e-01]#,   1.96e+00,  5.47e+00, 1.52e+01]
        max_vals2                                 =   [ 1.75e-01]#,   3.87e+00,  8.67e+00, 2.12e+01]

        # -6.35e-01, 1.96e+00, 5.47e+00, 1.52e+01, 2.72e+01, 3.90e+01, 6.00e+01, 7.00e+01, 8.00e+01, 1.00e+02,
        # 1.75e-01, 3.87e+00, 8.67e+00, 2.12e+01, 3.71e+01, 5.61e+01, 6.00e+01, 7.00e+01, 8.00e+01, 1.00e+02,

        # parameters                               =   ['Height01']#, 'Height02', 'Height03', 'Height04'] #, 'Tatm01',  'Tatm02',   'Tatm03',   'Tatm04'] # 'Rhostr',
        # min_vals                                 =   [ 0.00e+00]#,   1.07e+01,   1.92e+01,   3.02e+01]#,   222.103,  195.128,   190.365,    211.779]
        # max_vals                                 =   [ 1.75e-01]#,   1.44e+01,   2.70e+01,   4.11e+01]#,   317.294,  233.443,   240.905,    272.475]
        #
        # min_vals2                                =   [ 0.00e+00]#,   1.07e+01,   1.92e+01,   3.02e+01]#,   222.103,  195.128,   190.365,    211.779]
        # max_vals2                                =   [ 1.75e-01]#,   1.44e+01,   2.70e+01,   4.11e+01]#,   317.294,  233.443,   240.905,    272.475]
        #
        #
        # parameters                               =   ['Height02'] #, 'Tatm01',  'Tatm02',   'Tatm03',   'Tatm04'] # 'Rhostr',
        # min_vals                                 =   [1.07e+01]#,   222.103,  195.128,   190.365,    211.779]
        # max_vals                                 =   [1.44e+01]#,   317.294,  233.443,   240.905,    272.475]
        #
        # min_vals2                                =   [1.07e+01]#,   222.103,  195.128,   190.365,    211.779]
        # max_vals2                                =   [1.44e+01]#,   317.294,  233.443,   240.905,    272.475]

        # parameters                               =   ['Height03'] #, 'Tatm01',  'Tatm02',   'Tatm03',   'Tatm04'] # 'Rhostr',
        # min_vals                                 =   [1.92e+01]#,   222.103,  195.128,   190.365,    211.779]
        # max_vals                                 =   [2.70e+01]#,   317.294,  233.443,   240.905,    272.475]
        #
        # min_vals2                                =   [1.92e+01,   3.02e+01]#,   222.103,  195.128,   190.365,    211.779]
        # max_vals2                                =   [2.70e+01,   4.11e+01]#,   317.294,  233.443,   240.905,    272.475]
        #
        # parameters                               =   ['Height04'] #, 'Tatm01',  'Tatm02',   'Tatm03',   'Tatm04'] # 'Rhostr',
        # min_vals                                 =   [3.02e+01]#,   222.103,  195.128,   190.365,    211.779]
        # max_vals                                 =   [4.11e+01]#,   317.294,  233.443,   240.905,    272.475]
        #
        # min_vals2                                =   [3.02e+01]#,   222.103,  195.128,   190.365,    211.779]
        # max_vals2                                =   [4.11e+01]#,   317.294,  233.443,   240.905,    272.475]

        # parameters                               =   ['H20str', 'Tempstr', 'AOTstr', 'O3str'] # 'Rhostr',
        # min_vals                                 =   [0.5,     263.15,      0.0,     0.23]
        # max_vals                                 =   [8.0,     323.15,      1.8,     0.55]
        #
        # # Reinforce low AOT and low refl
        # min_vals2                                =   [0.5,     263.15,      0.0,     0.23]
        # max_vals2                                =   [8.0,     323.15,      0.5,     0.55]

        # parameters                               =   ['H20str', 'AOTstr', ] # 'Rhostr',
        # min_vals                                 =   [0.5,      0.0]
        # max_vals                                 =   [8.0,      0.8]
        #
        # # Reinforce low AOT and low refl
        # min_vals2                                =   [0.5,      0.0]
        # max_vals2                                =   [8.0,      0.5]

        #
        # parameters                               =   ['AOTstr', 'O3str'] # 'Rhostr',
        # min_vals                                 =   [0.0,     0.23]
        # max_vals                                 =   [1.8,     0.55]
        #
        # # Reinforce low AOT and low refl
        # min_vals2                                =   [0.0,     0.23]
        # max_vals2                                =   [0.5,     0.55]
        #
        # parameters                               =   ['Tempstr'] # 'Rhostr',
        # min_vals                                 =   [263.15]
        # max_vals                                 =   [323.15]
        #
        # # Reinforce low AOT and low refl
        # min_vals2                                =   [263.15]
        # max_vals2                                =   [323.15]

        ntrain2                                  =   int(ntrain/3)
        ntrain1                                  =   ntrain-ntrain2
        training_set1,distributions1             =   gp_emulator.create_training_set( parameters, min_vals,  max_vals,  n_train=ntrain1)
        training_set2,distributions2             =   gp_emulator.create_training_set( parameters, min_vals2, max_vals2, n_train=ntrain2)

        trainn                                   =   np.r_[training_set1, training_set2]
        training_set                             =   trainn*1.


        # create TP5 files
        datestr                                  =   (datetime.datetime.now()).strftime('%Y%m%d-%H%M%S')
        outputstr                                =   'Synergy_' + datestr + '/'
        filenames                                =   []
        I                                        =   np.zeros([len(training_set),3]).astype('int')

        for i,xx in enumerate(training_set):
            parameter_dict                       =   dict(zip(parameters,xx))
            #
            parameter_dict['Rhostr']             =   0.0
            scenariostr                          =   '%06.0f' % i + '_000'
            filename_000                         =   modtran_model.UpdateTP5(parameter_dict, outputstr, scenariostr)
            #
            parameter_dict['Rhostr']             =   0.5
            scenariostr                          =   '%06.0f' % i + '_050'
            filename_050                         =   modtran_model.UpdateTP5(parameter_dict, outputstr, scenariostr)
            #
            parameter_dict['Rhostr']             =   1.0
            scenariostr                          =   '%06.0f' % i + '_100'
            filename_100                         =   modtran_model.UpdateTP5(parameter_dict, outputstr, scenariostr)
            #
            filenames.append(filename_000)
            I[i,0]                               =   len(filenames)-1
            filenames.append(filename_050)
            I[i,1]                               =   len(filenames)-1
            filenames.append(filename_100)
            I[i,2]                               =   len(filenames)-1


        # write mod5root.in
        fp = open(modtran_dir + filename_modtroot, 'w')
        for filename in filenames:
            string                               =   outputstr + filename
            fp.write ( string+"\n" )
        fp.close()

        # create backup of mod5root in the directory itself
        shutil.copy(modtran_dir + filename_modtroot, modtran_dir + outputstr + filename_modtroot)

        self.min_vals                            =   min_vals
        self.min_vals                            =   min_vals
        self.parameters                          =   parameters

        return filename_modtroot, training_set, parameters, I

    def CreateIsotropicScenarios_Tprofile(self, ntrain, filename_modtroot='mod5root_.in'):
        ####################################################################################
        ############################# Preprocessing4isotropic     ##########################
        ####################################################################################


        print '- Creating test scenarios'
        # create test scenarios
        parameters                               =   ['H1str', 'H2str', 'H20str', 'Tempstr', 'AOTstr', 'O3str', 'Lmeas_TOA', 'SZAstr', 'VZAstr','RAAstr'] # 'Rhostr',
        min_vals                                 =   [777.,    0. ,     0.5,       263.15,     0.0,     0.23,      0.0,      0.0,      0.0,     0.0,    ]
        max_vals                                 =   [999.,    0.4,     8.0,       323.15,     1.8,     0.55,    333.3,      45.,     45.0,   180.0,    ]

        # Reinforce low AOT and low refl
        min_vals2                                =   [777.,    0. ,     0.5,       263.15,     0.0,     0.23,      0.0,      0.0,      0.0,     0.0]
        max_vals2                                =   [999.,    0.4,     8.0,       323.15,     0.5,     0.55,    333.3,      45.,     45.0,   180.0]

        # create test scenarios
        parameters                               =   ['H1str', 'H2str', 'VZAstr', 'H20str', 'AOTstr', 'O3str', 'tatm01',     'tatm02',   'tatm03'] # 'Rhostr',
        min_vals                                 =   [777.,    0. ,     0.0,      0.5,     0,        0.23,    222.103,      235.533,    214.285]
        max_vals                                 =   [999.,    0.4,     45.0,      8.0,     1.8,      0.55,    317.294,      292.299,    266.753]
        min_vals2                                =   [777.,    0. ,     0.0,      0.5,     0.0,      0.23,    222.103,      235.533,    214.285]
        max_vals2                                =   [999.,    0.4,     65.0,      8.0,     0.5,      0.55,    317.294,      292.299,    266.753]

        parameters                               =   ['H2str', 'VZAstr', 'H20str', 'AOTstr',    'tatm01',     'tatm02',   'tatm03'] # 'Rhostr',
        min_vals                                 =   [0.0 ,     0.0,      0.5,      0.0,         222.103,      235.533,    214.285]
        max_vals                                 =   [0.4,     45.0,      8.0,      1.8,         317.294,      292.299,    266.753]
        min_vals2                                =   [0.0 ,     0.0,      0.5,      0.0,         222.103,      235.533,    214.285]
        max_vals2                                =   [0.4,     65.0,      8.0,      0.5,         317.294,      292.299,    266.753]

        parameters                               =   ['H2str', 'VZAstr', 'AOTstr',    'tatm01',     'tatm02',   'tatm03'] # 'Rhostr',
        min_vals                                 =   [0.0,      0.0,     0.0,         222.103,      235.533,    214.285]
        max_vals                                 =   [0.4,     45.0,     1.8,         317.294,      292.299,    266.753]
        min_vals2                                =   [0.0 ,     0.0,     0.0,         222.103,      235.533,    214.285]
        max_vals2                                =   [0.4,     65.0,     0.5,         317.294,      292.299,    266.753]


        parameters                               =   ['tatm01',     'tatm02',   'tatm03'] # 'Rhostr',
        min_vals                                 =   [260.103,      235.533,    214.285]
        max_vals                                 =   [317.294,      292.299,    266.753]
        min_vals2                                =   [260.103,      235.533,    214.285]
        max_vals2                                =   [317.294,      292.299,    266.753]

        parameters                               =   ['tatm01',     'tatm02'] # RMSE_tau_max  =3.49%, RMSE_L_max = 6.43% (Band 2/3) ntrain =30
        min_vals                                 =   [260.103,      235.533]
        max_vals                                 =   [317.294,      292.299]
        min_vals2                                =   [260.103,      235.533]
        max_vals2                                =   [317.294,      292.299]

        parameters                               =   ['tatm01',     'tatm02'] # RMSE_tau_max  = 2.22%, RMSE_L_max =  1.90% (Band 2/3) ntrain = 100
        min_vals                                 =   [260.103,      235.533]
        max_vals                                 =   [317.294,      292.299]
        min_vals2                                =   [260.103,      235.533]
        max_vals2                                =   [317.294,      292.299]

        parameters                               =   ['tatm01',     'dtatm02'] # RMSE_tau_max  = 0.89%, RMSE_L_max =  1.22% (Band 2/3) ntrain = 100 [actual Vh20 is capped!)
        min_vals                                 =   [260.103,      -32.2]
        max_vals                                 =   [317.294,       13.7]
        min_vals2                                =   [260.103,      -32.2]
        max_vals2                                =   [317.294,       13.7]

        parameters                               =   ['H20str', 'tatm01',     'dtatm02'] # RMSE_tau_max  = 0.65%, RMSE_L_max =  0.01% (Band 2/3) ntrain = 100
        min_vals                                 =   [0.5,      260.103,      -32.2]
        max_vals                                 =   [8.0,      317.294,       13.7]
        min_vals2                                =   [0.5,      260.103,      -32.2]
        max_vals2                                =   [8.0,      317.294,       13.7]

        parameters                               =   ['H20str', 'tatm01',     'dtatm02'] # RMSE_tau_max  = 0.60%, RMSE_L_max =  0.80% (Band 2/3) ntrain = 400
        min_vals                                 =   [0.5,      260.103,      -32.2]
        max_vals                                 =   [8.0,      317.294,       13.7]
        min_vals2                                =   [0.5,      260.103,      -32.2]
        max_vals2                                =   [8.0,      317.294,       13.7]

        parameters                               =   ['H20str', 'tatm01',     'dtatm02', 'VZAstr'] # RMSE_tau_max  = 0.97%, RMSE_L_max =  1.39% (Band 2/3) ntrain = 150 + ttransformations
        min_vals                                 =   [0.5,      260.103,      -32.2,      0.0]
        max_vals                                 =   [8.0,      317.294,       13.7,     45.0]
        min_vals2                                =   [0.5,      260.103,      -32.2,      0.0]
        max_vals2                                =   [8.0,      317.294,       13.7,     65.0]



        parameters                               =   ['H20str', 'tatm01',     'dtatm02', 'VZAstr'] # RMSE_tau_max  = 0.92%, RMSE_L_max =  1.28% (Band 2/3) ntrain = 400 + ttransformations
        min_vals                                 =   [0.5,      260.103,      -32.2,      0.0]
        max_vals                                 =   [8.0,      317.294,       13.7,     45.0]
        min_vals2                                =   [0.5,      260.103,      -32.2,      0.0]
        max_vals2                                =   [8.0,      317.294,       13.7,     65.0]

        parameters                               =   ['H20str', 'tatm01',     'dtatm02', 'VZAstr', 'AOTstr'] # RMSE_tau_max  = 0.96%, RMSE_L_max =  1.17% (Band 2/3) ntrain = 400 + ttransformations
        min_vals                                 =   [0.5,      260.103,      -32.2,      0.0,      0.0]
        max_vals                                 =   [8.0,      317.294,       13.7,     45.0,      1.8]
        min_vals2                                =   [0.5,      260.103,      -32.2,      0.0,      0.0]
        max_vals2                                =   [8.0,      317.294,       13.7,     65.0,      0.5]

        parameters                               =   ['H2str', 'VZAstr', 'H20str', 'tatm01',     'dtatm02', 'AOTstr'] # RMSE_tau_max  = 1.04%, RMSE_L_max =  1.25% (Band 2/3) ntrain = 400 + ttransformations
        min_vals                                 =   [0.0,       0.0,      0.5,      260.103,      -32.2,      0.0]
        max_vals                                 =   [0.4,      45.0,      8.0,      317.294,       13.7,     1.8]
        min_vals2                                =   [0.0,       0.0,      0.5,      260.103,      -32.2,      0.0]
        max_vals2                                =   [0.4,      65.0,      8.0,      317.294,       13.7,     0.5]

        parameters                               =   ['H2str', 'VZAstr', 'H20str', 'tatm01',     'dtatm02', 'AOTstr'] # RMSE_tau_max  = 1.04%, RMSE_L_max =  1.25% (Band 2/3) ntrain = 500 + ttransformations
        min_vals                                 =   [0.0,       0.0,      0.5,      260.103,      -32.2,      0.0]
        max_vals                                 =   [0.4,      45.0,      8.0,      317.294,       13.7,     1.8]
        min_vals2                                =   [0.0,       0.0,      0.5,      260.103,      -32.2,      0.0]
        max_vals2                                =   [0.4,      65.0,      8.0,      317.294,       13.7,     0.5]

        parameters                               =   ['H2str', 'VZAstr', 'H20str', 'tatm01',     'dtatm02', 'AOTstr',   'O3str' ] # RMSE_tau_max  = 0.55%, RMSE_L_max =  1.22% (Band 2/3) ntrain = 600 + ttransformations
        min_vals                                 =   [0.0,       0.0,      0.5,      260.103,      -32.2,      0.0,     0.23    ]
        max_vals                                 =   [0.4,      45.0,      8.0,      317.294,       13.7,     1.8,      0.55    ]
        min_vals2                                =   [0.0,       0.0,      0.5,      260.103,      -32.2,      0.0,     0.23    ]
        max_vals2                                =   [0.4,      65.0,      8.0,      317.294,       13.7,     0.5,      0.55    ]

        parameters                               =   ['H2str', 'VZAstr', 'H20str', 'tatm01',     'dtatm02',     'dtatm03', 'AOTstr',   'O3str' ] # RMSE_tau_max  = 1.01%, RMSE_L_max =  1.5% (Band 2/3) ntrain = 600 + ttransformations
        min_vals                                 =   [0.0,       0.0,      0.5,      260.103,      -32.2,      -38.642,     0.0,     0.23    ]
        max_vals                                 =   [0.4,      45.0,      8.0,      317.294,       13.7,       -0.329,     1.8,     0.55    ]
        min_vals2                                =   [0.0,       0.0,      0.5,      260.103,      -32.2,      -38.642,     0.0,     0.23    ]
        max_vals2                                =   [0.4,      65.0,      8.0,      317.294,       13.7,       -0.329,     0.5,     0.55    ]


        parameters                               =   ['H2str', 'VZAstr', 'H20str', 'tatm01',     'dtatm02', 'AOTstr',   'O3str' ] # RMSE_tau_max  = 1.0%, RMSE_L_max =  1.5% (Band 2/3) ntrain = 800 + ttransformations
        min_vals                                 =   [0.0,       0.0,      0.5,      260.103,      -32.2,      0.0,     0.23    ]
        max_vals                                 =   [0.4,      45.0,      8.0,      317.294,       13.7,     1.8,      0.55    ]
        min_vals2                                =   [0.0,       0.0,      0.5,      260.103,      -32.2,      0.0,     0.23    ]
        max_vals2                                =   [0.4,      65.0,      8.0,      317.294,       13.7,     0.5,      0.55    ]


        parameters                               =   ['H2str', 'VZAstr', 'RHstr', 'tatm01',     'dtatm02', 'AOTstr',   'O3str' ] # RMSE_tau_max  = 1.0%, RMSE_L_max =  1.5% (Band 2/3) ntrain = 800 + ttransformations
        min_vals                                 =   [0.0,       0.0,      0.0,      260.103,      -32.2,      0.0,     0.23    ]
        max_vals                                 =   [0.4,      45.0,     99.0,      317.294,       13.7,     1.8,      0.55    ]
        min_vals2                                =   [0.0,       0.0,      0.5,      260.103,      -32.2,      0.0,     0.23    ]
        max_vals2                                =   [0.4,      65.0,     99.0,      317.294,       13.7,     0.5,      0.55    ]


        parameters                               =   ['RH01']   # RMSE_tau_max  = 0.02%, RMSE_L_max =  0.04% (Band 2/3) ntrain = 30
        min_vals                                 =   [ 0.0]
        max_vals                                 =   [99.0]
        min_vals2                                =   [ 0.0]
        max_vals2                                =   [99.0]

        parameters                               =   ['H20str', 'tatm01',     'dtatm02'] # RMSE_tau_max  = 0.65%, RMSE_L_max =  0.01% (Band 2/3) ntrain = 100
        min_vals                                 =   [0.5,      260.103,      -32.2]
        max_vals                                 =   [8.0,      317.294,       13.7]
        min_vals2                                =   [0.5,      260.103,      -32.2]
        max_vals2                                =   [8.0,      317.294,       13.7]

        parameters                               =   ['RH01', 'tatm01',     'AOTstr'] # using RH route (for input only): RMSE_tau_max  = 0.19%, RMSE_L_max =  0.76% (Band 2/3) ntrain = 400
        min_vals                                 =   [0.0,      260.103,      0.0]    # using RH route (for In&Out):     RMSE_tau_max  = 0.19%, RMSE_L_max =  0.98% (Band 2/3) ntrain = 400
        max_vals                                 =   [80.0,      317.294,    1.8]
        min_vals2                                =   [0.0,      260.103,      0.0]
        max_vals2                                =   [80.0,      278.294,    0.5]

        parameters                               =   ['RH01', 'tatm01',     'dtatm02', 'AOTstr', 'VZAstr'] # using RH route (for input only), RMSE_tau_max  = 1.46%, RMSE_L_max =  1.58% (Band 2/3) ntrain = 400
        min_vals                                 =   [0.1,      260.103,      -32.2,     0.0,       0.0]    # using RH route (for In&Out):     RMSE_tau_max  = 39.3%, RMSE_L_max =  1.45% (Band 2/3) ntrain = 400
        max_vals                                 =   [99.0,      317.294,      13.7,     1.8,      65.0]
        min_vals2                                =   [80.1,      260.103,     -32.2,     0.0,       0.0]
        max_vals2                                =   [90.0,      278.294,      13.7,     0.5,      65.0]

        parameters                               =   ['RH01', 'tatm01',     'dtatm02', 'AOTstr', 'VZAstr'] # using RH route (for input only), RMSE_tau_max  =  %, RMSE_L_max =  % (Band 2/3) ntrain = 400
        min_vals                                 =   [0.1,      260.103,      -32.2,     0.0,       0.0]    # using RH route (for In&Out):     RMSE_tau_max  = %, RMSE_L_max =  % (Band 2/3) ntrain = 400
        max_vals                                 =   [90.0,     317.294,      13.7,      1.8,      65.0]
        min_vals2                                =   [0.1,      260.103,     -32.2,      0.0,       0.0]
        max_vals2                                =   [90.0,     317.294,      13.7,      0.5,      65.0]

        # {'Latm': [1.8506872810280368, 1.344542824218746, 1.5898479137918824],
        #  'tau': [0.22344371494854445, 0.52818734961725045, 1.4610773112592674]}




        #
        #
        # parameters                               =   ['H2str', 'VZAstr', 'AOTstr',    'tatm01',     'tatm03'] # RMSE_tau_max  =0.79%, RMSE_L_max = 0.82% (Band 2/3)
        # min_vals                                 =   [0.0 ,     0.0,     0.0,         222.103,      214.285]
        # max_vals                                 =   [0.4,     45.0,     1.8,         317.294,      266.753]
        # min_vals2                                =   [0.0 ,     0.0,     0.0,         222.103,      214.285]
        # max_vals2                                =   [0.4,     65.0,     0.5,         317.294,      266.753]
        #
        #
        # parameters                               =   ['H2str', 'VZAstr', 'H20str', 'tatm01',     'tatm03',    'AOTstr'] # RMSE_tau_max  =0.61%, RMSE_L_max = 1.0% (Band 2/3)
        # min_vals                                 =   [0.0 ,     0.0,      0.5,      222.103,      214.285,     0.0]
        # max_vals                                 =   [0.4,     45.0,      8.0,      317.294,      266.753,     1.8]
        # min_vals2                                =   [0.0 ,     0.0,      0.5,      222.103,      214.285,     0.0]
        # max_vals2                                =   [0.4,     65.0,      8.0,      317.294,      266.753,     0.5]


        # parameters                               =   ['H20str', 'AOTstr', 'tatm01'] # 'Rhostr',
        # min_vals                                 =   [0.5,     0,        222.103]
        # max_vals                                 =   [8.0,     1.8,      317.294]
        # min_vals2                                =   [0.5,     0.0,      222.103]
        # max_vals2                                =   [8.0,     0.5,      317.294]
        #
        #
        # parameters                               =   ['H20str', 'tatm01'] # 'Rhostr',
        # min_vals                                 =   [0.5,     260.103]
        # max_vals                                 =   [8.0,     317.294]
        # min_vals2                                =   [0.5,     300.103]
        # max_vals2                                =   [8.0,     317.294]
        #
        # parameters                               =   ['tatm01'] # 'Rhostr',
        # min_vals                                 =   [260.103]
        # max_vals                                 =   [317.294]
        # min_vals2                                =   [300.103]
        # max_vals2                                =   [317.294]

        #
        # parameters                               =   ['H20str'] # 'Rhostr',
        # min_vals                                 =   [0.5     ]
        # max_vals                                 =   [8.0     ]
        # min_vals2                                =   [0.5     ]
        # max_vals2                                =   [8.0     ]

        # parameters                               =   ['height01', 'height02', 'height03', 'height04', 'height05', 'height06', 'tatm01', 'tatm02', 'tatm03', 'tatm04', 'tatm05', 'tatm06']
        # min_vals                                 =   [-6.35e-01,    1.96e+00,   5.47e+00,   1.07e+01,   1.44e+01,   2.72e+01, 222.103,      235.533,    214.285,    195.128,    184.190,    207.521]
        # max_vals                                 =   [1.75e-01,     3.87e+00,   8.67e+00,   1.36e+01,   1.88e+01,   3.71e+01, 317.294,      292.299,    266.753,    233.443,    235.072,    263.315]
        # min_vals2                                =   [-6.35e-01,    1.96e+00,   5.47e+00,   1.07e+01,   1.44e+01,   2.72e+01, 222.103,      235.533,    214.285,    195.128,    184.190,    207.521]
        # max_vals2                                =   [1.75e-01,     3.87e+00,   8.67e+00,   1.36e+01,   1.88e+01,   3.71e+01, 317.294,      292.299,    266.753,    233.443,    235.072,    263.315]

        # parameters                               =   ['height01', 'height02', 'height03', 'height04', 'height05', 'height06']
        # min_vals                                 =   [-6.35e-01,    1.96e+00,   5.47e+00,   1.07e+01,   1.44e+01,   2.72e+01]
        # max_vals                                 =   [1.75e-01,     3.87e+00,   8.67e+00,   1.36e+01,   1.88e+01,   3.71e+01]
        # min_vals2                                =   [-6.35e-01,    1.96e+00,   5.47e+00,   1.07e+01,   1.44e+01,   2.72e+01]
        # max_vals2                                =   [1.75e-01,     3.87e+00,   8.67e+00,   1.36e+01,   1.88e+01,   3.71e+01]
        #
        # parameters                               =   ['tatm01',     'tatm02',   'tatm03']#,   'tatm04',   'tatm05', 'tatm06']
        # min_vals                                 =   [222.103,      235.533,    214.285]#,    195.128,    184.190,    207.521]
        # max_vals                                 =   [317.294,      292.299,    266.753]#,    233.443,    235.072,    263.315]
        # min_vals2                                 =   [222.103,      235.533,    214.285]#,    195.128,    184.190,    207.521]
        # max_vals2                                 =   [317.294,      292.299,    266.753]#,    233.443,    235.072,    263.315]

        # parameters                               =    ['tatm04',   'tatm05', 'tatm06']
        # min_vals                                 =    [195.128,    184.190,    207.521]
        # max_vals                                 =    [233.443,    235.072,    263.315]
        # min_vals2                                 =   [195.128,    184.190,    207.521]
        # max_vals2                                 =   [233.443,    235.072,    263.315]

        # parameters                               =   ['H20str','AOTstr', 'O3str']
        # min_vals                                 =   [0.5,     0.0,     0.23]
        # max_vals                                 =   [8.0,     1.8,     0.55]
        # min_vals2                                =   [0.5,     0.0,     0.23]
        # max_vals2                                =   [8.0,     0.5,     0.55]


        ntrain2                                  =   int(ntrain/3)
        ntrain1                                  =   ntrain-ntrain2
        training_set1,distributions1             =   gp_emulator.create_training_set( parameters, min_vals,  max_vals,  n_train=ntrain1)
        training_set2,distributions2             =   gp_emulator.create_training_set( parameters, min_vals2, max_vals2, n_train=ntrain2)

        trainn                                   =   np.r_[training_set1, training_set2]
        training_set                             =   trainn*1.


        # import pdb
        # pdb.set_trace()
        # we can define tatm02 as a linear combination of tatm01. This will negate values that are completely unrealistic and consequently will(?)
        # improve the training of the emulator  s
        ##############################################################################################################
        irh01                                   =   [i for i,name in enumerate(parameters) if name=='RH01']

        if (len(irh01)>0):
            irh01                               =   irh01[0]
            RH                                  =   training_set[:,irh01]

            itatm01                             =   [i for i,name in enumerate(parameters) if name=='tatm01']
            if (len(itatm01)>0):
                itatm01                         =   itatm01[0]
                T01                             =   training_set[:,itatm01]
            else:
                T01                             =   np.ones_like(RH) * self.default_dict['tatm01']

            WV                                  =   WV_transformation()
            H20                                 =   WV.inverse(T01,RH)

            RH_check                            =   WV.forward(T01,H20)
            # import pdb
            # pdb.set_trace()

            training_set[:,irh01]               =   H20
            parameters[irh01]                   =   'H20str'

        ##############################################################################################################
        itatm01                                 =    [i for i,name in enumerate(parameters) if name=='tatm01']
        idT21                                   =    [i for i,name in enumerate(parameters) if name=='dtatm02']
        if (len(itatm01)>0) *(len(idT21)>0):
            itatm01                             =   itatm01[0]
            idT21                               =   idT21[0]
            dT                                  =   training_set[:,idT21]
            training_set[:,idT21]               =   training_set[:,itatm01] + dT
            parameters[idT21]                   =   'tatm02'

        ##############################################################################################################
        itatm02                                 =    [i for i,name in enumerate(parameters) if name=='tatm02']
        idT32                                   =    [i for i,name in enumerate(parameters) if name=='dtatm03']
        if (len(itatm02)>0) *(len(idT32)>0):
            itatm02                             =   itatm02[0]
            idT32                               =   idT32[0]
            dT                                  =   training_set[:,idT32]
            training_set[:,idT32]               =   training_set[:,itatm02] + dT
            parameters[idT32]                   =   'tatm03'

        # create TP5 files
        datestr                                  =   (datetime.datetime.now()).strftime('%Y%m%d-%H%M%S')
        outputstr                                =   'Synergy_' + datestr + '/'
        filenames                                =   []
        I                                        =   np.zeros([len(training_set),3]).astype('int')

        ##############################################################################################################
        for i,xx in enumerate(training_set):
            parameter_dict                       =   dict(zip(parameters,xx))
            #
            parameter_dict['Rhostr']             =   0.0
            scenariostr                          =   '%06.0f' % i + '_000'
            filename_000                         =   modtran_model.UpdateTP5_Tprofile(parameter_dict, outputstr, scenariostr)

            #
            parameter_dict['Rhostr']             =   0.5
            scenariostr                          =   '%06.0f' % i + '_050'
            filename_050                         =   modtran_model.UpdateTP5_Tprofile(parameter_dict, outputstr, scenariostr)

            #
            parameter_dict['Rhostr']             =   1.0
            scenariostr                          =   '%06.0f' % i + '_100'
            filename_100                         =   modtran_model.UpdateTP5_Tprofile(parameter_dict, outputstr, scenariostr)

            #
            filenames.append(filename_000)
            I[i,0]                               =   len(filenames)-1
            filenames.append(filename_050)
            I[i,1]                               =   len(filenames)-1
            filenames.append(filename_100)
            I[i,2]                               =   len(filenames)-1


        # write mod5root.in
        fp = open(modtran_dir + filename_modtroot, 'w')
        for filename in filenames:
            string                               =   outputstr + filename
            fp.write ( string+"\n" )
        fp.close()

        # create backup of mod5root in the directory itself
        shutil.copy(modtran_dir + filename_modtroot, modtran_dir + outputstr + filename_modtroot)

        self.min_vals                            =   min_vals
        self.min_vals                            =   min_vals
        self.parameters                          =   parameters

        return filename_modtroot, training_set, parameters, I


def CreatePlanckEmulator():
    Tb          =       planck_model.T
    Lmeas       =       planck_model.Lmeas[:,0]
    Tb2         =       np.reshape(Tb,[len(Tb),1])
    Lmeas2      =       np.reshape(Lmeas,[len(Lmeas),1])

    imin1            =       np.where(Tb>100)[0][0]
    imin2            =       np.where(Tb>100)[0][0]
    imax1            =       np.where(Tb<400)[0][-1]
    imax2            =       np.where(Tb<400)[0][-1]

    i1              =       [int(i) for i in np.linspace(imin1,imax1, 300)]
    i2              =       [int(i) for i in np.linspace(imin2,imax2, 300)]

    planck_em       =       gp_emulator.GaussianProcess(Tb2[i1],Lmeas[i1])
    planck_inv_em   =       gp_emulator.GaussianProcess(Lmeas2[i2],Tb[i2])
    planck_em.learn_hyperparameters(verbose=False)
    planck_inv_em.learn_hyperparameters(verbose=False)

    Lmeas_em,Lmeas_em_var,dLmeas_em =   planck_em.predict(Tb2,do_unc=True)
    Tb_em,Tb_em_var,dTb_em          =   planck_inv_em.predict(Lmeas2,do_unc=True)

    plt.plot(Lmeas2[i2],Tb[i2])
    plt.plot(Lmeas2[i2],Tb_em[i2])
    plt.legend(['traing','em'])
    plt.xlabel('L')
    plt.ylabel('Tb')

    i = list(np.where(np.all([Tb>250,Tb<390],axis=0))[0])
    i2= list(np.where(np.all([Tb_em>250,Tb_em<390],axis=0))[0])
    i   =   i[0:-1:100]
    plt.subplot(2,2,1)

    plt.plot(Tb[i],Lmeas[i])
    plt.plot(Tb2[i],Lmeas_em[i],'.')
    plt.plot(Tb_em[i2],Lmeas2[i2],'o')

    # plt.subplot(2,2,2)
    # plt.plot(Tb[i],Lmeas_em_var[i],'.')
    # plt.plot(Tb_em[i],Lmeas_var[i],'o')

    plt.subplot(2,2,3)
    plt.plot(Tb[i],dLmeas_em[i],'.')

    plt.subplot(2,2,4)
    plt.plot(dTb_em[i],Lmeas[i],'.')


class SensorSimulator():
    def __init__(self, name='SLSTR'):
        self.sensor                             =   name

    def SensorAtt(self):
        sensor                                  =   self.sensor

        band                                    =   dict()
        if sensor=='SLSTR':
            # apply Sentinel-3 SLTSTR sensor sensitivity
            # band['name']                       =   ['S1',  'S2',   'S3',   'S4',   'S5',   'S6',   'S7',   'S8',   'S9'] #,   'F1',   'F2']
            # band['function']                   =   ['vegetation monitoring','NDVI','NDVI','Cirrus-detection','vegetation monitoring','vegetation state','LST','LST','LST'] #,'Active fire','Active fire']
            # band['wv']                         =   1.e3*np.array([0.555, 0.659,  0.865,  1.375,  1.61,   2.25,   3.74,   10.85,  12.00]) #,  3.74,   10.85])
            # band['width']                      =   1.e3*np.array([0.02,  0.02,   0.02,   0.015,  0.06,   0.05,   0.38,   0.90,    1.00]) #,   0.38,   0.90])
            # band['min']                        =   band_wv - band_width/2
            # band['max']                        =   band_wv + band_width/2

            band['name']                        =   ['S7',   'S8',   'S9'] #,   'F1',   'F2']
            band['function']                    =   ['LST','LST','LST'] #,'Active fire','Active fire']
            band['wv']                          =   1.e3*np.array([3.74,   10.85,  12.00]) #,  3.74,   10.85])
            band['width']                       =   1.e3*np.array([0.38,   0.90,    1.00]) #,   0.38,   0.90])
            band['min']                         =   band['wv'] - band['width']/2
            band['max']                         =   band['wv'] + band['width']/2
        else:
            error

        return band

    def Apply2Sensor(self,wl,values):
        sensor                                  =   self.sensor
        band                                    =   self.SensorAtt()
        nbands                                  =   len(band['wv'])

        # Filter the output for anomalies
        nwv                                     =   np.shape(values)[0]
        ntrain                                  =   np.shape(values)[1]

        values2                                 =   values*1.
        for i in xrange(nwv-5):
            ii                                  =   list(np.arange(0,5)+i)
            values2[ii,:]                       =    np.nanmedian(values[ii,:],axis=0)

        ierror                                  =   np.where(np.isnan(values2))
        values2[ierror]                         =   0
        values                                  =   values2 *1.

        # apply sensor attributes
        band_values_av                          =   np.zeros([len(band['wv']),ntrain])*np.nan
        band_values_std                         =   np.zeros([len(band['wv']),ntrain])*np.nan

        for i,wv in enumerate(band['wv']):
            iwl                                 =   np.all([wl>=band['min'][i] , wl<=band['max'][i]], axis=0)

            if np.any(iwl):
                band_values_av[i,:]             =   np.mean(values[iwl,:],  axis=0)
                band_values_std[i,:]            =   np.std(values[iwl,:],  axis=0)

        return band_values_av, band_values_std

class Planck():
    def __init__(self, sensors=None):
        if sensors<>None:
            self.T, self.Lmeas                  =   self.constructLUT(sensors)

        #
    def constructLUT(self,sensors):
        # create LUT to easily convert from L to Tb
        T                                       =   np.arange(0.,800.,0.01)
        WL                                      =   np.arange(2.0e3, 15.0e3,50)
        #
        WL1, T1                                 =   np.meshgrid(WL,T)
        L                                       =   self.run_forward(T1,WL1).T
        #
        Lmeas, Lms                              =   sensors.Apply2Sensor(WL,L)                          # [W m-2 sr-1 um-1]
        #
        self.T                                  =   T
        self.Lmeas                              =   Lmeas.T
        return T, Lmeas.T

    def run_forward(self,Tb, wl_nm):
        # Verhoef and Bach, 2012, Rem. Sen. Env 120, 197-207
        T                                       =   Tb* np.ones([1,1])

        wl_um                                   =   wl_nm*1e-3      #  [um]
        c1                                      =   1.191066e-22      # [W m3 um-1 sr-1] = [W m2 sr-1]*1e6
        c2                                      =   1.438833e4        # [um K]
        Lb                                      =   c1*(wl_um*1e-6)**(-5) / (np.exp(c2 /(wl_um * T))-1)       #    [w m-2 um-1]
        L                                       =   Lb** np.ones([1,1])
        #
        return L

    def LUT_inverse(self,Li):
        nbands                                  =   np.shape(Li)[1]
        ntrain                                  =   np.shape(Li)[0]

        Ti                                      =   np.zeros_like(Li)
        for itrain in xrange(ntrain):
            for iband in xrange(nbands):
                Ti[itrain,iband]                =   np.interp(Li[itrain,iband], self.Lmeas[:,iband], self.T)

        return Ti.T

    def LUT_forward(self,Ti):
        # to be forced with Ti = F [ntrain x 1], or by Ti = F [ntrain x nbands]





        #check for Ti is 1D vector

        if Ti.ndim==1:
            dummy, Ti2                          =   np.meshgrid(self.Lmeas[0,:], Ti)
        else:
            Ti2                                 =   Ti*1.

        nbands                                  =   np.shape(self.Lmeas)[1]
        ntrain                                  =   np.shape(Ti)[0]
        Li                                      =   np.zeros([ntrain, nbands])

        for itrain in xrange(ntrain):
            for iband in xrange(nbands):
                Li[itrain,iband]                =   np.interp(Ti2[itrain,iband], self.T, self.Lmeas[:,iband])

        return Li.T

# calculate BOA simulations
def PltData(isc, Values_dict, parameters, training_set, planck_model,WL, band):
    Lmin                                        =   planck_model.run_forward(230,WL).T                            # [W m-2 sr-1 um-1]
    Lmax                                        =   planck_model.run_forward(330,WL).T                            # [W m-2 sr-1 um-1]

    plt.close()
    # plt.plot(WL, L_u_TOA[:,0:-1:100],'b')
    plt.plot(WL,Values_dict['TOTAL_RAD'][:,(0+3*isc):(3+3*isc)],'c')
    plt.plot(WL, Lmin,'g')
    plt.plot(WL, Lmax,'g')
    plt.vlines(band['min'], np.min(Lmin), np.max(Lmax),'r')
    plt.vlines(band['max'], np.min(Lmin), np.max(Lmax),'r')
    plt.ylim([np.min(Lmin), np.max(Lmax)])
    zip(parameters, training_set[isc,:])

class WV_transformation():
    def __init__(self):
        q_default_g_m3                              =     [1.29E+01, 8.66E+00, 5.49E+00, 3.12E+00, 1.82E+00, 9.81E-01,
                                                           6.10E-01, 3.70E-01, 2.10E-01, 1.20E-01, 6.40E-02, 2.20E-02,
                                                           6.00E-03, 1.44E-03, 7.69E-04, 4.44E-04, 3.68E-04, 3.05E-04,
                                                           2.56E-04, 2.21E-04, 1.94E-04, 1.73E-04, 1.54E-04, 1.41E-04,
                                                           1.25E-04, 1.12E-04, 5.75E-05, 2.85E-05, 1.43E-05, 7.70E-06,
                                                           4.11E-06, 2.22E-06, 1.15E-06, 2.46E-07, 3.14E-08, 1.17E-10]      # [g/m3]
        q_default_g_m3                              =   np.array(q_default_g_m3)
        q_default                                   =   q_default_g_m3/1e4                                                  # [g/m / cm2]
        dl                                          =   np.ones_like(q_default)*1e3                                         # [m]
        q_default_tcwv                              =   np.sum(q_default*dl)                                                # [g/cm2]

        self.Rv                                     =   461.5;                                                              # [J/Kg K]
        self.dl_                                    =   dl
        self.q_default_                             =   q_default
        self.q_default_tcwv                         =   q_default_tcwv


    def forward(self,T_bottom, q_tcwv):
        # q_tcwv                                 is total column vapor [g/cm2]
        q_default_tcwv                              =   self.q_default_tcwv                                             # (g / cm2)
        q_default_                                  =   self.q_default_                                                 # (g / m / cm2)

        # import pdb
        # pdb.set_trace()

        q_default_tcwv                              =   np.abs(q_default_tcwv)
        q_default_                                  =   np.abs(q_default_)


        RH_bottom                                   =   np.ones_like(T_bottom)
        for iii,v in enumerate(T_bottom):
            ratio                                   =   (q_tcwv[iii]/q_default_tcwv)                                    # [-]
            q_new_                                  =   ratio*q_default_                                                # [g /m/ cm2]

            q_new_bottom                            =   q_new_[0]                                                       # [g /m /cm2]
            RH_bottom[iii]                          =   self.CalculateRH(T_bottom[iii], q_new_bottom)

        return RH_bottom

    def inverse(self, T_bottom, RH_bottom):
        dl_                                         =   self.dl_
        q_default_                                  =   self.q_default_                                                 # g/ m / cm2
        q_default_bottom                            =   q_default_[0]                                                   # g/ m / cm2

        q_tcwv                                      =   np.zeros_like((T_bottom))

        for i,v in enumerate(T_bottom):
            q_bottom                                =   self.CalculateWV(T_bottom[i], RH_bottom[i])                     # g/ m3
            q_bottom                                =   q_bottom*1e-4                                                   # g /m /cm2

            q_                                      =   q_default_* (q_bottom/q_default_bottom)                         # g/ m /cm2
            q_tcwv[i]                               =   np.sum(q_*dl_)                                                  # g/ cm2



        return q_tcwv

    def CalculateWV(self, T, RH):
        Rv                                          =   self.Rv

        # calculations
        T_C                                         =   T-273.15
        e_sat_hPa                                   =   6.112 * np.exp( (17.62*T_C)/(243.12+T_C))        # [hPa]
        e_sat                                       =   e_sat_hPa *100                                  # [Pa]

        e_act                                       =   e_sat * (RH/100)                                # [Pa]
        q                                           =   e_act / (Rv*T)                                  # [kg/m3]

        q_layer_modtran                             =   q*1e3                                           # g/m3

        # per ~1km layer (assuming constant temperature per layer)
        # dl                                          =   1e3
        # q_layer                                     =   q*dl                                           # [kg/m2]
        #
        #
        # # in modtran units
        # q_layer_modtran                             =   q_layer*1e3*1e-4                                    # [g/cm2]



        return q_layer_modtran
    def CalculateRH(self,T, q_modtran_layer):
        # q water vapor in bottom layer     [g /m /cm2]
        # T of bottom layer                 [K]

        Rv                                          =   self.Rv
        q                                           =   q_modtran_layer*1e4/1e3;                   # [kg/m3]

        e_act                                       =   q*(Rv*T)                                    # [Pa]

        T_C                                         =   T-273.15                                    # [C]
        e_sat_hPa                                   =   6.112 * np.exp( (17.62*T_C)/(243.12+T_C))   # [hPa]
        e_sat                                       =   e_sat_hPa *100                              # [Pa]

        RH                                          =   e_act/e_sat*100

        # import pdb
        # pdb.set_trace()
        return RH

def InvestigateSensitivities(Emulator, em_name,training_set2,parameters,option_value):
        isc                                         =   0
        x_train2                                    =   training_set2[isc]*1.
        nvar                                        =   np.shape(x_train2)[0]
        nc                                          =   3
        nr                                          =   int(np.ceil(nvar/float(nc)))
        emulators_Tmeas                             =   Emulator[em_name]

        miny                                        =   250
        maxy                                        =   370

        plt.figure(figsize=[15,10])
        for iband in xrange(3):
            em                                      =   emulators_Tmeas[iband]

            min_vals                                =   np.min(training_set2, axis=0)
            max_vals                                =   np.max(training_set2, axis=0)

            for i, name in enumerate(parameters):                               #! remove the [-1]
                # print name
                minx                                =   min_vals[i]
                maxx                                =   max_vals[i]
                x                                   =   np.linspace(minx,maxx,100)

                Xtrain2, X                          =   np.meshgrid(x_train2, x)
                Xtrain2[:,i]                        =   X[:,0]

                plt.subplot(nr,nc,i+1)
                if option_value=='V':
                    V,V_var, dV                     =   em.predict(Xtrain2, do_unc=False)

                    y                               =   V
                    string                          =   em_name
                    plt.plot(x,y)

                    if name=='Tmeas_TOA':
                        plt.ylim([miny, maxy])

                elif option_value=='dV':
                    V,V_var, dV                     =   em.predict(Xtrain2, do_unc=False)
                    y                               =   dV[:,i]
                    string                          =   'd' + em_name

                    plt.plot(x,y)

                elif option_value=='V_var':
                    V,V_var, dV                     =   em.predict(Xtrain2, do_unc=True)
                    y                               =   V_var
                    string                          =   em_name + '_var'

                    i1                              =   np.where(y>0)[0]
                    i2                              =   np.where(y<0)[0]
                    plt.semilogy(x,np.abs(y),'b')
                    plt.semilogy(x[i1],y[i1],'g.')
                    plt.semilogy(x[i2],-y[i2],'r.')

                plt.title(string + '-sensitivity to '+ name)

        plt.legend(['Band 1','Band 2','Band 3'],loc=0)
        plt.savefig('SLTSR TIR '+string+'-sensitivity to MODTRAN variables')

if __name__ == "__main__":
    plt.ion()
    plt.close('all')

    # things to investigate:
    # -> Tatm01 = 220? - 320!
    # when Tatm01 <<273, H20 is modified to lower values (yes)... so extracted H20 (from TP5 file) is not actual H20 value in use in Modtran!
    # according to MODTREAN TP5:
    # - "THE WATER COLUMN MAXIMUM is 2.18314 GM / CM2." (Tatm = 2.413E+02 K) (MODTRAN_Tprofile_simulation_000000_050.tp6 of Synergy_20170725-135415) (L152)
    # -

    ######################################################################################################
    # parameters
    ######################################################################################################
    sensor                                      =   'SLSTR'
    ntrain                                      =   400
    Groupsize                                   =   30      # MaX Number parallel processes
    tic                                         =   datetime.datetime.now()

    option_load                                 =   1
    option_resume                               =   0
    ######################################################################################################
    # do not touch
    ######################################################################################################
    homedir                                     =   '/home/ucfajti/Simulations/Python/modtran/'
    modtran_dir                                 =   '/home/ucfajti/Simulations/MODTRAN/Mod5.2.1/'
    # modtran_dir                                 =   '/home/ucfajti/Simulations/MODTRAN/Mod5.2.1_brdf/'
    modtran_dir_validate                        =   '/home/ucfajti/Simulations/MODTRAN/Mod5.2.1_brdf/'
    subdir                                      =   'simulation_outputs/SaveStates/'
    emulator_home                               =   "./emulators/"
    ##################################################################################################################
    # dir_training                                =   'Synergy_20170612-223505'               # 800     ['H1str', 'H2str', 'H20str', 'Tempstr', 'AOTstr', 'O3str', 'Lmeas_TOA', 'SZAstr', 'VZAstr', 'RAAstr']
    # dir_training                                =   'Synergy_20170704-152045'               # 800     ['H1str', 'H2str', 'SZAstr', 'VZAstr', 'RAAstr', 'H20str', 'Tempstr', 'AOTstr', 'O3str']
    # dir_training                                =   'Synergy_20170717-100959'               # 250     ['height01', 'height02', 'height03', 'height04', 'height05', 'height06']
    # dir_training                                =   'Synergy_20170718-135450'               # 100     ['SZAstr', 'VZAstr', 'RAAstr']
    dir_training                                =   'Synergy_20170718-191502'               # 150     ['tatm01', 'tatm02', 'tatm03']
    # dir_training                                =   'Synergy_20170718-222112'               # 150     ['tatm04', 'tatm05', 'tatm06']
    # dir_training                                =   'Synergy_20170719-122450'               # 200     ['H20str', 'AOTstr', 'O3str']
    # dir_training                                =   'Synergy_20170725-093234'               # 100     ['H20str']
    # dir_training                                =   'Synergy_20170725-135415'               # 150     ['H20str', 'tatm01']
    # dir_training                                =   'Synergy_20170725-155014'               # 030     [tatm01]
    # dir_training                                =   'Synergy_20170725-200827'               # 030     [tatm01]
    # dir_training                                =   'Synergy_20170725-213801'               # 010     ['tatm01']
    # dir_training                                =   'Synergy_20170725-224402'               # 1000    ['H2str', 'VZAstr', 'H20str', 'AOTstr', 'tatm01', 'tatm02', 'tatm03']
    # dir_training                                =   'Synergy_20170726-154339'               # 030     ['tatm01']  , with minvals = 260 and H20=2.5
    # dir_training                                =   'Synergy_20170726-161952'               # 030     ['tatmo1'] , with minvals = 260 and H20=1.0
    # dir_training                                =   'Synergy_20170726-165314'               # 030     ['tatm01']  , with minvals = 260 and H20=4.0
    # dir_training                                =   'Synergy_20170726-222617'               # 030     ['tatm01']
    # dir_training                                =   'Synergy_20170727-183858'               # 1000    ['H2str', 'VZAstr', 'AOTstr', 'tatm01', 'tatm02', 'tatm03']
    # dir_training                                =   'Synergy_20170727-183858'               # 1000    ['H2str', 'VZAstr', 'AOTstr', 'tatm01', 'tatm02', 'tatm03']

    # dir_training                                =   'Synergy_20170804-100844'               #  100    ['tatm01', 'tatm02']
    # dir_training                                =   'Synergy_20170804-102955'               #  100    ['tatm01', 'tatm02']
    # dir_training                                =   'Synergy_20170804-113517'               #  100    ['H20str', 'tatm01', 'tatm02']
    dir_training                                =   'Synergy_20170804-125316'               #  100    ['H20str', 'tatm01', 'tatm02']  'datm02'
    # dir_training                                =   'Synergy_20170804-161917'               #  150    ['H20str', 'tatm01', 'tatm02', 'tatm03'] 'datm02' 'datm03'
    # dir_training                                =   'Synergy_20170804-191134'               #  150    ['H20str', 'tatm01', 'tatm02', 'VZAstr'] 'datm02'

    # dir_training                                =   'Synergy_20170804-215451'               #  150      ['H20str', 'tatm01', 'tatm02', 'VZAstr']
    # dir_training                                =   'Synergy_20170804-221417'               #  400      ['H20str', 'tatm01', 'tatm02']
    # dir_training                                =   'Synergy_20170805-095154'               #  400      ['H20str', 'tatm01', 'tatm02', 'VZAstr']
    # dir_training                                =   'Synergy_20170807-100041'               #  500      ['H2str', 'VZAstr', 'H20str', 'tatm01', 'tatm02', 'AOTstr']
    # dir_training                                =   'Synergy_20170807-143609'               #  500      ['H2str', 'VZAstr', 'H20str', 'tatm01', 'tatm02', 'AOTstr']
    # dir_training                                =   'Synergy_20170808-075947'               #  600      ['H2str', 'VZAstr', 'H20str', 'tatm01', 'tatm02', 'AOTstr', 'O3str']
    # dir_training                                =   'Synergy_20170808-192856'               #  600      ['H2str', 'VZAstr', 'H20str', 'tatm01', 'tatm02', 'tatm03', 'AOTstr', 'O3str']

    # dir_training                                =   'Synergy_20170810-152214'               #   30      ['H20str'] using RH route
    dir_training                                =   'Synergy_20170810-192849'               #  100        ['H20str', 'tatm01', 'tatm02'] using RH route
    dir_training                                =   'Synergy_20170810-223516'               #  400        ['H20str', 'tatm01', 'AOT'] using RH route
    dir_training                                =   'Synergy_20170811-084215'               # 400       ['']
    dir_training                                =   'Synergy_20170811-203244'               # 400        ['H20str', 'tatm01', 'tatm02', 'AOTstr', 'VZAstr'] + H20 transformation

    # dir_validation                              =   'Synergy_20170612-224558'               # 800     ['H1str', 'H2str', 'H20str', 'Tempstr', 'AOTstr', 'O3str', 'Lmeas_TOA', 'SZAstr', 'VZAstr', 'RAAstr']
    # dir_validation                              =   'Synergy_20170613-164728'               # 800     ['H1str', 'H2str', 'H20str', 'Tempstr', 'AOTstr', 'O3str', 'Lmeas_TOA', 'SZAstr', 'VZAstr', 'RAAstr']
    # dir_validation                              =   'Synergy_20170703-221201'               # 800     ['H1str', 'H2str', 'SZAstr', 'VZAstr', 'RAAstr']
    # dir_validation                              =   'Synergy_20170704-092924'               # 100     ['H20str']
    # dir_validation                              =   'Synergy_20170704-110433'               # 100     ['AOTstr', 'O3str']
    # dir_validation                              =   'Synergy_20170705-112136'               # 100     ['H20str', 'Tempstr']
    # dir_validation                              =   'Synergy_20170705-132455'               # 020     ['Tempstr']
    # dir_validation                              =   'Synergy_20170711-101050'               # 010     ['Height01', 'Height02', 'Height03', 'Height04', 'Tatm01', 'Tatm02', 'Tatm03', 'Tatm04']
    # dir_validation                              =   'Synergy_20170711-101932'               # 010     ['Height01', 'Height02', 'Height03', 'Height04']
    # dir_validation                              =   'Synergy_20170712-162615'               # 010     ['height01']
    # dir_validation                              =   'Synergy_20170713-153024'               # 010     ['heyght01']
    # dir_validation                              =   'Synergy_20170717-231021'               # 500     ['height01', 'height02', 'height03', 'height04', 'height05', 'height06', 'tatm01', 'tatm02', 'tatm03', 'tatm04', 'tatm05', 'tatm06']
    # dir_validation                              =   'Synergy_20170718-092849'               # 250     ['SZAstr', 'VZAstr', 'RAAstr']
    # dir_validation                              =   'Synergy_20170719-222751'               # 800     ['H1str', 'H2str', 'VZAstr', 'H20str', 'AOTstr', 'O3str', 'tatm01', 'tatm02', 'tatm03']
    # dir_validation                              =   'Synergy_20170721-233619'               # 800     ['H1str', 'H2str', 'VZAstr', 'H20str', 'AOTstr', 'O3str', 'tatm01', 'tatm02', 'tatm03']
    # dir_validation                              =   'Synergy_20170725-002318'               # 800     ['H2str', 'VZAstr', 'H20str', 'AOTstr', 'O3str', 'tatm01', 'tatm02', 'tatm03']
    # dir_validation                              =   'Synergy_20170725-091649'               # 800     ['H2str', 'VZAstr', 'H20str', 'AOTstr', 'O3str', 'tatm01', 'tatm02', 'tatm03']
    dir_validation                              =   'Synergy_20170726-222720'               #1000     ['H2str', 'VZAstr', 'H20str', 'AOTstr', 'tatm01', 'tatm02', 'tatm03']

    # dir_training                                =   dir_validation
    # modtran_dir                                 =   modtran_dir_validate

    ##################################################################################################################
    simulationstr                               =   dir_training
    if option_load:
        try:
            modtran_dir                         =   cPickle.load(open(homedir + subdir + simulationstr+ '_modtran_dir.pkl', 'r'))
        except:
            cPickle.dump(modtran_dir,    open(homedir + subdir + simulationstr+ '_modtran_dir.pkl', 'wb'))


    ##################################################################################################################

    #specify default values in the desired library form
    modtran_model                               =   modtran(modtran_dir)
    modtran_model_validate                      =   modtran(modtran_dir_validate)

    sensors                                     =   SensorSimulator(sensor)
    band                                        =   sensors.SensorAtt()

    ##################################################################################################################
    homedir                                     =   modtran_model.homedir
    modtran_dir                                 =   modtran_model.modtran_dir

    os.chdir(modtran_dir)
    path2modtran_exe                            =   glob.glob('*.exe')[0]
    exec_str_par                                =   'nohup nice -n 17 ./' + path2modtran_exe + '>/dev/null 2>&1 &' # ' &'
    os.chdir(homedir)

    ##################################################################################################################
    # Parameters for loading (previously) finished results
    print '- Overview of finished MODTRAN runs:'
    directories                                 =   modtran_model.ListFinishedModtranSimulations()
    for directory in directories:
        print directory

    # outputname
    filestr_meas_BOA                            =   "_BOA_%s_emulators_%04d_2.pkl"%(sensor,ntrain)
    filestr_atm_BOA                             =   "_BOA_%s_emulators_%04d_2.pkl"%(sensor,ntrain)

    homedir                                     =   modtran_model.homedir

    # sudo pkill Mod90_5.2.1.exe
    ####################################################################################
    ############################# Load Simulation Parameters   #########################
    ####################################################################################
    if (option_load==0):
        ############################# create scenarios             #####################

        # filename_modtroot, training_set, parameters, I =   modtran_model.CreateIsotropicScenarios(ntrain)
        filename_modtroot, training_set, parameters, I =   modtran_model.CreateIsotropicScenarios_Tprofile(ntrain)
        # filename_modtroot, training_set, parameters, =   modtran_model.CreateKernelsScenarios(ntrain)

        # read in all scenarios
        scenarios                               =   modtran_model.ReadDifferentScenarios(filename_modtroot)

        # save stuff for restarting later on
        simulationstr                           =   scenarios[0].split('/')[0]
        cPickle.dump(parameters,    open(homedir + subdir + simulationstr+ '_Parameters.pkl', 'wb'))
        cPickle.dump(I,             open(homedir + subdir + simulationstr + '_I' + '.pkl', 'wb'))
        cPickle.dump(training_set,  open(homedir + subdir + simulationstr+ '_Training_set.pkl', 'wb'))
        cPickle.dump(modtran_dir,    open(homedir + subdir + simulationstr+ '_modtran_dir.pkl', 'wb'))

    elif (option_load)+ (option_resume):
        ##################################################################################################################
        try:
            print '- Selected modtran run: ' + dir_training
            os.chdir(modtran_model.modtran_dir)
            shutil.copyfile(dir_training+ "/mod5root_.in", 'mod5root_.in')
            os.chdir(modtran_model.homedir)
        except:
            print '- Loading of Specified -Trainingset not successfull'

        ############################# Load Previous               #####################
        # (ACTION TO be performed outside of the SCRIPT: first copy backup mod5root_.in file to (above-specified) MODTRAN folder)
        filename_modtroot                       =   'mod5root_.in'
        scenarios                               =   modtran_model.ReadDifferentScenarios(filename_modtroot)

        print '- Load Parameters'
        simulationstr                           =   scenarios[0].split('/')[0]
        training_set                            =   cPickle.load(open(homedir + subdir + simulationstr + '_Training_set' + '.pkl', 'r'))
        parameters                              =   cPickle.load(open(homedir + subdir + simulationstr + '_Parameters' + '.pkl', 'r'))
        I                                       =   cPickle.load(open(homedir + subdir + simulationstr + '_I' + '.pkl', 'r'))
        ntrain                                  =   np.shape(training_set)[0]

        print '     > Number of Scenarios: %5.2f' % ntrain
        print '     > State Variables:'
        print parameters
        print

    ####################################################################################
    ############################# Load Simulations Results     #########################
    ####################################################################################
    if option_resume + option_load:
        try:
            print '- Load (almost) finished Data'
            Values_dict                         =   cPickle.load(open(homedir + subdir + simulationstr+ 'Values_dict.pkl', 'r'))
            varnames                            =   cPickle.load(open(homedir + subdir + simulationstr+ 'varnames.pkl', 'r'))

            print '     > MODTRAN Output:'
            print varnames
            print

        except:
            option_resume                       =   1

    ####################################################################################
    ############################# Start/Continue Simulations   #########################
    ####################################################################################
    if option_resume + (option_load==0):
        scenario_missing                        =   modtran_model.FindMissingScenarios(scenarios)

        print '- Total of simulations (still) to run: %03.0f' % len(scenario_missing)
        # seperate processing in groups
        Nsims                                   =   len(scenario_missing)
        Ngroups                                 =   int(np.ceil(float(Nsims)/Groupsize))
        for igroup in xrange(Ngroups):
            index_min                           =   igroup*Groupsize
            index_max                           =   np.min([igroup*Groupsize+Groupsize,Nsims])
            index                               =   list(np.arange(index_min, index_max))

            scenarios_s                         =   [scenario_missing[ii] for ii in index]                                     # scenarios[index]

            # start executing scenarios
            error                               =   modtran_model.ExecuteScenarios(scenarios_s, exec_str_par)

            # wait until all simulations are finished
            I_unfinished, not_finished          =   modtran_model.Wait4MODTRAN2Finish(scenarios_s,1., 0)

            # investigate why simulations are not written to disk!
            scenario_missing2                   =   modtran_model.FindMissingScenarios(scenarios_s)

            print '- Run is %4.2f %% Completed' % (float(igroup+1)/(Ngroups+1)*100)
            os.system('clear')
        ###### Read Simulations or re-initialize MODTRAN of the unfinished scenarios    ####
        unread                                  =   1
        maxiter                                 =   5
        iter                                    =   0

        while unread*(iter<=maxiter):
            iter                                =   iter+1

            print '- Try reading MODTRAN runs'
            # read simulations (which files, tp7, tp6)
            Values_dict                         =   modtran_model.ReadTP7(scenarios)                            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            varnames                            =   [name for name in Values_dict.iterkeys()]
            if not(np.any(np.isnan(Values_dict[varnames[0]]))):
                # Values_dict                     =   modtran_model.ReadTP7(scenarios)

                unread                          =   0

                print '############################'
                print '- all MODTRAN runs read'
            else:
                print '- Found missing MODTRAN runs'
                scenario_missing                =   modtran_model.FindMissingScenarios(scenarios)

                print '- ReRun unfinished MODTRAN runs (%02.0f)' % len(scenario_missing)
                modtran_model.ReRunErroneous(scenario_missing, Groupsize)

        cPickle.dump(Values_dict,    open(homedir + subdir + simulationstr+ 'Values_dict.pkl', 'wb'))
        cPickle.dump(varnames,       open(homedir + subdir + simulationstr+ 'varnames.pkl', 'wb'))


    # Identify erroneous Runs
    I, ntrain                                   =   modtran_model.IdentifyErroneousRuns(I,scenarios,training_set,Values_dict)

    toc                                         =   datetime.datetime.now()
    print '-MODTRAN Simulations finished, duration was %f hours' % ((toc-tic).total_seconds()/60/60)

    ####################################################################################
    ############################# PreProcess Data             ##########################c
    ####################################################################################
    # MODTRAN rescales H20 if it is not physical.
    option_h20                                  =   0   # do nothing
    option_h20                                  =   1   #replace set-H20 value with actual value
    # option_h20                                  =   2   #replace set-H20 value with actual value + remove erroneous values
    # option_h20                                  =   3  #replace set-H20 value with actual value + remove temperatures below 250
    # option_h20                                  =   4   #replace set-H20 value with actual value + remove some of the temperatures below 250

    def CorrectH20Values(modtran_model, scenarios, parameters, training_set, I, option_h20):
        V_H20                                       =   modtran_model.ReadTP6(scenarios)                            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ih20                                        =   [i for i,name in enumerate(parameters) if name=='H20str']
        itatm01                                     =   [i for i,name in enumerate(parameters) if name=='tatm01']

        if len(ih20):
            E                                       =   np.abs((np.abs(V_H20)-np.abs(training_set[:,ih20].T))[0,:])
        else:
            E                                       =   0

        if len(itatm01)>0:
            tatm                                    =   training_set[:,itatm01][:,0]
        else:
            tatm                                    =    modtran_model.default_dict['tatm01']

        if len(ih20)>0:
            h20                                     =   training_set[:,ih20][:,0]
        else:
            h20                                     =    modtran_model.default_dict['H20str']


        # set values to the actual used by modtran
        if option_h20<>0:
            iremove_                                =   []
            if len(ih20)>0:
                ih20                                =   ih20[0]
                training_set_old                    =   training_set*1

                E                                   =   training_set[:,ih20]-V_H20
                Ih20                                =   np.where(E>1e-2)[0]

                training_set[:,ih20]                =   V_H20

        # remove all scenarios from the index
        if option_h20==2:
            Isame                                   =   E<1e-2
            Idiff                                   =   Isame==False
            iremove_                                =   np.where(Idiff)[0]
            iremove_                                =   sorted(iremove_,reverse=True)

        # remove some scenarios from the index, more from Tlow-samples than from Thigh-samples (saturation plays bigger role at low temperatures)
        if option_h20==3:
            Itemp                                   =   (tatm< 250.15)
            iremove_                                =   np.where(Itemp)[0]
            iremove_                                =   sorted(iremove_,reverse=True)

        # remove some scenarios from the index, more from Tlow-samples than from Thigh-samples (saturation plays bigger role at low temperatures)
        if option_h20==4:
            Isame                                   =   E<1e-2
            Idiff                                   =   Isame==False
            Idiff_Tlow                              =   Idiff * (tatm< 250.15)
            Idiff_Thigh                             =   Idiff * (tatm>= 250.15)

            iremove_                                =   np.append( np.where(Idiff_Tlow)[0][0:-2:2], np.where(Idiff_Thigh)[0][0:-2:10])
            iremove_                                =   np.where(Idiff_Tlow)[0]
            iremove_                                =   sorted(iremove_,reverse=True)



        # remove unavailable scenarios from the index
        for iremove in iremove_:
            # delete row from index
            V_H20                                   =   np.delete(V_H20,iremove, axis=0)
            I                                       =   np.delete(I,iremove, axis=0)
            training_set                            =   np.delete(training_set, iremove, axis = 0)
        ntrain                                      =   np.shape(training_set)[0]

        return V_H20, I, training_set, ntrain

    training_set_old                            =   training_set*1.
    (V_H20, I, training_set, ntrain)            =   CorrectH20Values(modtran_model, scenarios, parameters, training_set, I, option_h20)
    training_set[:,0]                           =   np.abs(training_set[:,0])
    ####################################################################################
    ############################# PreProcess Data-1           ##########################
    ####################################################################################
    option_Tmeas                                =   0
    option_BOA                                  =   0

    sensors                                     =   SensorSimulator(sensor)
    band                                        =   sensors.SensorAtt()

    planck_model                                =   Planck(sensors)

    print '- PreProcess the Data'
    tic                                         =   datetime.datetime.now()
    # extract atmospheric variables using T18 MODTRAN interoation technique (Verhoef and Bach, 2003, Verhoef et al, 2012)
    WL, TRAN,tau_oo_,tau_do_,L_u_TOA_,L_d_BOA_  =  modtran_model.T18(Values_dict,I)

    cPickle.dump(WL,    open(homedir + subdir + 'WL.pkl', 'wb'))
    cPickle.dump(L_u_TOA_,    open(homedir + subdir + simulationstr + 'L_u_TOA.pkl', 'wb'))
    cPickle.dump(L_d_BOA_,    open(homedir + subdir + simulationstr + 'L_d_BOA.pkl', 'wb'))

    # apply Sentinel-3 SLTSTR sensor sensitivity (to be replaced with simpler version)
    band_tau_oo, band_tau_oo_std                =   sensors.Apply2Sensor(WL,tau_oo_)                                                 # [-]       [nbands x ntrain]
    band_tau_do, band_tau_do_std                =   sensors.Apply2Sensor(WL,tau_do_)                                                 # [-]       [nbands x ntrain]
    band_L_u_TOA, band_L_u_TOA_std              =   sensors.Apply2Sensor(WL,L_u_TOA_)                                                # [W cm-2 sr-1 um-1]       [nbands x ntrain]    still modify units
    band_L_d_BOA, band_L_d_BOA_std              =   sensors.Apply2Sensor(WL,L_d_BOA_)                                                # [W cm-2 sr-1 um-1]       [nbands x ntrain]    still modify units

    # brightness temperatures
    band_T_u_TOA                                =   planck_model.LUT_inverse(band_L_u_TOA.T)
    band_tau                                    =   band_tau_oo + band_tau_do

    nTemp                                       =   int(np.ceil(1100./ntrain))
    nparam                                      =   np.shape(training_set)[1]
    nparam2                                     =   nparam+1
    parameters2                                 =   [name for name in parameters]

    if option_Tmeas:
        nbands                                  =   np.shape(band_T_u_TOA)[0]
        ntrain2                                 =   ntrain * nTemp


        # band_T_meas_TOA                       =   np.zeros([nbands,ntrain2])
        band_Tmeas_BOA2                         =   np.zeros([nbands,ntrain2])
        band_Lmeas_BOA2                         =   np.zeros([nbands,ntrain2])
        band_L_d_BOA2                           =   np.zeros([nbands,ntrain2])

        training_set2                           =   np.zeros([ntrain2, nparam2])
        for isc in xrange(ntrain):
            # extract scenario
            temp                                =   band_T_u_TOA[:,isc]
            training                            =   training_set[isc,:]
            tau                                 =   band_tau[:,isc]
            L_u_TOA                             =   band_L_u_TOA[:,isc]
            L_d_BOA                             =   band_L_d_BOA[:,isc]

            #
            randv                               =   np.random.rand(nTemp)
            deltaT                              =   randv*80

            # Define variables
            DeltaT, T                           =   np.meshgrid(deltaT, temp)
            Training,dummy                      =   np.meshgrid(training,randv)
            dummy, L_u_TOA                      =   np.meshgrid(randv, L_u_TOA)
            dummy, L_d_BOA                      =   np.meshgrid(randv, L_d_BOA)

            dummy, tau                          =   np.meshgrid(randv, tau)

            # estimate Tmeas_TOA (so that Lmeas_TOA>L_u_TOA is always larger)
            Tmeas_TOA                           =   T + DeltaT #(should be 1D)
            Tmeas_TOA_1D                        =   np.max(Tmeas_TOA, axis=0)
            Lmeas_TOA                           =   planck_model.LUT_forward(Tmeas_TOA_1D.T)

            # define atmospheric correction
            Lmeas_BOA                           =   (Lmeas_TOA - L_u_TOA) / (tau)
            Tmeas_BOA                           =   planck_model.LUT_inverse(Lmeas_BOA.T)

            # store expended T.
            index                               =   np.arange(0,len(randv)) + isc*len(randv)
            training_set2[index,:]              =   np.vstack([Training.T,Tmeas_TOA_1D]).T
            band_Tmeas_BOA2[:,index]            =   Tmeas_BOA
            band_Lmeas_BOA2[:,index]            =   Lmeas_BOA
            # band_L_d_BOA2[:,index]            =   L_d_BOA

        parameters2.append('Tmeas_TOA')
        training_set                            =   training_set
        band_Tmeas_BOA                          =   band_Tmeas_BOA2
        band_Lmeas_BOA                          =   band_Lmeas_BOA2
        band_Latm_BOA                           =   band_L_d_BOA

    else:
        ntrain2                                 =   ntrain * nTemp
        training_set2                           =   np.zeros([ntrain2, nparam2])
        for isc in xrange(ntrain):
            # define new temperature
            randv                               =   np.random.rand(nTemp)
            deltaT                              =   randv*80

            # Define variables
            DeltaT, T                           =   np.meshgrid(deltaT, band_T_u_TOA[:,isc])
            Training,dummy                      =   np.meshgrid(training_set[isc,:],deltaT)


            # estimate Tmeas_TOA (so that Lmeas_TOA>L_u_TOA is always larger)
            Tmeas_TOA                           =   T + DeltaT #(should be 1D)
            Tmeas_TOA_1D                        =   np.max(Tmeas_TOA, axis=0)

            # store expended T.
            index                               =   np.arange(0,len(randv)) + isc*len(randv)
            training_set2[index,:]              =   np.vstack([Training.T,Tmeas_TOA_1D]).T

        parameters2.append('Tmeas_TOA')

        band_tau                                =   band_tau
        band_Latm_TOA                           =   band_L_u_TOA
        band_Latm_BOA                           =   band_L_d_BOA
        band_Tatm_TOA                           =   planck_model.LUT_inverse(band_L_u_TOA.T)

    cPickle.dump(band_tau,          open(homedir + subdir + simulationstr+ '_band_tau_TOA.pkl', 'wb'))
    cPickle.dump(band_Latm_TOA,     open(homedir + subdir + simulationstr+ '_band_Latm_TOA.pkl', 'wb'))
    cPickle.dump(band_Latm_BOA,     open(homedir + subdir + simulationstr+ '_band_Latm_BOA.pkl', 'wb'))

    toc                                 =   datetime.datetime.now()
    print '- PreProcess finished, duration was %f hours' % ((toc-tic).total_seconds()/60/60)

    ####################################################################################
    ############################# Analysis H20-T dependency   ##########################
    ####################################################################################
    print '- Analyse H20-Tatm dependency'
    str                                                 =   []
    c                                                   =   ['r','g','b','c','m','k','y']
    nr                                                  =   len(c)
    x                                                   =   np.linspace(min(V_H20)*0.9, max(V_H20)*1.1, nr)
    tatm                                                =   [modtran_model.default_dict['tatm%02.0f' % (i+1)] for i in xrange(10)]


    itemp                                               =   [i for i,name in enumerate(parameters) if name=='tatm01']
    if len(itemp)>0:
        V_Tatm_default                                  =   training_set_old[:,itemp[0]]
    else:
        V_Tatm_default                                  =   modtran_model.default_dict['tatm01'] + np.zeros_like(training_set_old[:,0])


    ih20                                                =   [i for i,name in enumerate(parameters) if name=='H20str']
    if len(ih20)>0:
        V_H20_default                                   =   training_set_old[:,ih20]
    else:
        V_H20_default                                   =   modtran_model.default_dict['H20str'] + np.zeros_like(V_Tatm_default)

    if len(itemp)>0:
        plt.figure(figsize=[15,15])
        plt.subplot(3,1,1)
        plt.plot(V_Tatm_default,V_H20_default,'y.')
        str.append('old')

        for i in xrange(nr-1):
            ix                                          =   (V_H20>x[i])*(V_H20<=x[i+1])
            # print np.sum(ix)
            plt.subplot(3,1,1)
            plt.plot(training_set[ix,itemp],V_H20[ix],'o'+c[i])
            plt.xlim([min(training_set[:,itemp]),max(training_set[:,itemp])])
            plt.ylabel('Actual H20')

            plt.subplot(3,1,2)
            plt.plot(training_set[ix,itemp], band_tau[2,ix],'o'+c[i])
            plt.xlim([min(training_set[:,itemp]),max(training_set[:,itemp])])
            plt.ylabel('tau')

            plt.subplot(3,1,3)
            plt.plot(training_set[ix,itemp], band_Latm_TOA[2,ix],'o'+c[i])
            plt.xlim([min(training_set[:,itemp]),max(training_set[:,itemp])])
            plt.ylabel('Latm')

            str.append('H20: [%5.2f - %5.2f]' % (x[i], x[i+1]))

        plt.subplot(3,1,1)
        plt.legend(str,loc=0)
        plt.savefig('Sensitivity to H20 and Tatm01.png')

    # as can be observed, the water vapor cannot exceed the saturation curve (dependent on T). This has a direct
    # impact on tau and Latm, limiting these also to a specific lower/upper curve. The best way to deal with this
    # is to perform an aprior transformation of H20 using Tatm01 and the saturation curve. In that manner the
    # transformed space, fully covers the sampling space.

    # In effect this will create larger errors for the emulators, than if not corrected for. However this correction
    # is not integrated into the coding at this moment, due to time constraints.

    ####################################################################################
    ##################### Investigate possible transformationz    #######################
    ####################################################################################
    print '- Investigate possible transformation'
    tic                                         =   datetime.datetime.now()
    # create emulator
    training_set_transformed_tau                    =   training_set*1.
    training_set_transformed_Latm                   =   training_set*1.
    training_set2_transformed                   =   training_set2*1.


    # ih20                                        =   [i for i,name in enumerate(parameters) if name=='H20str']
    # if len(ih20)>0:
    #     ih20                                    =   ih20[0]
    #     # training_set_transformed_tau[:,ih20]        =   np.min([training_set[:,ih20], np.ones_like(training_set[:,ih20])*5], axis=0)
    #     training_set_transformed[:,ih20]        =   1/(1+np.exp(training_set[:,ih20]))
    #     training_set_transformed2[:,ih20]       =   1/(1+np.exp(-training_set[:,ih20])) #training_set[:,ih20]

    ivza                                        =   [i for i,name in enumerate(parameters) if name=='VZAstr']
    if len(ivza)>0:
        ivza                                    =   ivza[0]
        training_set_transformed_tau[:,ivza]    =   np.cos(training_set[:,ivza]*np.pi/180)
        training_set_transformed_Latm[:,ivza]   =   np.cos(training_set[:,ivza]*np.pi/180)


    ih20                                        =   [i for i,name in enumerate(parameters) if name=='H20str']
    iatm01                                      =   [i for i,name in enumerate(parameters) if name=='tatm01']
    if len(ih20)>0:
        ih20                                    =   ih20[0]
        H20                                     =   np.abs(training_set[:,ih20])

        if len(iatm01)>0:
            iatm01                              =   iatm01[0]
            T01                                 =   training_set[:,iatm01]
        else:
            T01                                 =   np.ones_like(H20) * modtran_model.default_dict['tatm01']
        WV                                      =   WV_transformation()
        RH01                                    =   WV.forward(T01, H20)
        training_set_transformed_tau[:,ih20]    =   RH01
        # training_set_transformed_Latm[:,ih20]   =   RH01



    # iatm01                                      =   [i for i,name in enumerate(parameters) if name=='tatm01']
    # iatm02                                      =   [i for i,name in enumerate(parameters) if name=='tatm02']
    # iatm03                                      =   [i for i,name in enumerate(parameters) if name=='tatm03']
    # if (len(iatm02)>0) * (len(iatm03)>0):
    #     training_set_transformed_tau[:,iatm03]  =   training_set_transformed_tau[:,iatm03] - training_set_transformed_tau[:,iatm02]
    #     training_set_transformed_Latm[:,iatm03] =   training_set_transformed_Latm[:,iatm03] - training_set_transformed_Latm[:,iatm02]
    #
    # if (len(iatm01)>0) * (len(iatm02)>0):
    #     training_set_transformed_tau[:,iatm02]  =   training_set_transformed_tau[:,iatm02] - training_set_transformed_tau[:,iatm01]
    #     training_set_transformed_Latm[:,iatm02] =   training_set_transformed_Latm[:,iatm02] - training_set_transformed_Latm[:,iatm01]


    # itatm                                       =   [i for i,name in enumerate(parameters) if 'tatm' in name]
    # training_set_transformed_tau[:,itatm]       =   5.67e-8*training_set[:,itatm]**4
    # training_set_transformed_Latm[:,itatm]      =   5.67e-8*training_set[:,itatm]**4


    # no tran
    # {'Latm': [0.6782861404232613, 1.4399082627753286, 1.0933442086586453],
    #  'tau': [0.13400131073218288, 0.56373384805355686, 1.3036985055243919]}


     # Only vza trans (best option sofar)
    # {'Latm': [0.61914245383908817, 1.303958141386194, 1.0949290213067],
    #  'tau': [0.13154371057087824, 0.43786701230418595, 0.97647971452700488]}

    # with vza and Latm trans
    # {'Latm': [0.67214047819796308, 1.3945031748327097, 1.2290756124483497],
    # 'tau': [0.13154358901038202, 0.43786680617836649, 0.97648036046139741]}

    # with vza, Latm and tau trans
    # {'Latm': [0.67214074246474764, 1.3945041886857819, 1.2290754021493764],
    #  'tau': [0.13034012672131534, 0.44272076472277544, 1.04482360645202]}



    #or maybe use a sigmoid function
    # ->  1/(1+e^-x)
    # -> arctan(x)

    print '- Transformation finished, duration was %f hours' % ((toc-tic).total_seconds()/60/60)
    ####################################################################################
    ############################# Post Processing Data        ##########################
    ####################################################################################
    tic                                         =   datetime.datetime.now()
    print '- Post Process Data '
    if option_Tmeas==1:
        string_output                           =   ['Tmeas', 'inv_tau', 'Latm_inv_tau']

        Training_output                         =   dict()
        Training_output[string_output[0]]       =   band_Tmeas_BOA.T
        Training_output[string_output[1]]       =   1/band_tau.T
        Training_output[string_output[2]]       =   (-band_L_u_TOA/band_tau).T

        Training_set                            =   dict()
        Training_set[string_output[0]]          =   training_set2
        Training_set[string_output[1]]          =   training_set
        Training_set[string_output[2]]          =   training_set
    elif option_Tmeas==2:
        string_output                           =   ['tau', 'Latm', 'inv_tau', 'Latm_inv_tau']

        Training_output                         =   dict()
        Training_output[string_output[0]]       =   (band_tau).T
        Training_output[string_output[1]]       =   band_L_u_TOA.T
        Training_output[string_output[2]]       =   1/(band_tau).T
        Training_output[string_output[3]]       =   (-band_L_u_TOA/band_tau).T

        Training_set                            =   dict()
        Training_set[string_output[0]]          =   training_set
        Training_set[string_output[1]]          =   training_set
        Training_set[string_output[2]]          =   training_set
        Training_set[string_output[3]]          =   training_set

    else:
        string_output                           =   ['tau', 'Latm']

        Training_output                         =   dict()
        Training_output[string_output[0]]       =   (band_tau).T
        Training_output[string_output[1]]       =   band_L_u_TOA.T

        Training_set                            =   dict()
        Training_set[string_output[0]]          =   training_set_transformed_tau
        Training_set[string_output[1]]          =   training_set_transformed_Latm

    if option_BOA :
        string_output.append('Latm_BOA')
        Training_output['Latm_BOA']             =   band_Latm_BOA.T
        Training_set[string_output[1]]          =   training_set

    toc                                         =   datetime.datetime.now()
    print '- Post Processing finished, duration was %f hours' % ((toc-tic).total_seconds()/60/60)

    ####################################################################################
    ############################# Create Emulators            ##########################
    ####################################################################################
    os.chdir(modtran_model.homedir)
    tic                                         =   datetime.datetime.now()

    os.chdir(modtran_model.homedir)
    thresh_                                         =   [0.5] #[2.0, 1.0, 0.5, 0.4, 0.3, 0.2, 0.1]
    ithresh                                         =   list(np.ones_like(thresh_)==1)



    RMSE_                                           =   dict()
    if option_load:
        print '- Loading Emulators'
        try:
            for thresh in thresh_:
                threshstr                           =   '_thresh=%05.2f' % thresh
                Emulator                            =   cPickle.load(open(emulator_home + simulationstr+ threshstr + '_Emulators'+'.pkl', 'r'))
                RMSE_                               =   cPickle.load(open(emulator_home + simulationstr+ threshstr + '_Emulator_RMSE'+'.pkl', 'r'))

                ifinished                           =   [i for i,t in enumerate(thresh_) if t==thresh][0]

                thresh_                             =   thresh_[(ifinished+1):]
        except:
            print

    thresh_                                         =   [0.7] #[2.0, 1.0, 0.5, 0.4, 0.3, 0.2, 0.1]
    option_resume=1
    bands                                           =   [2]
    if (option_load==0) + (option_resume==1):
        print '- Creating (Unfinished) Emulators'
        for thresh in thresh_:
            Emulator                                =   dict()
            RMSE_                                   =   dict()
            Q75_rel_                                =   dict()

            threshstr                               =   '_thresh=%05.2f' % thresh
            # cPickle.dump(Emulator, open(emulator_home + simulationstr+ threshstr + '_Emulators'+'.pkl', 'wb'))
            for i,string in enumerate(string_output):
                print
                print string + 'Thresh = %5.2f' % thresh

                Emulator[string]                    =   [0,1,2]
                RMSE_[string]                       =   [0,1,2]
                Q75_rel_[string]                    =   [0,1,2]
                for iband in bands: #range(3):
                    print '--- %1.0f' % iband

                    Q75_rel                         =   100
                    counter                         =   0
                    r                               =   []
                    while (Q75_rel>thresh)*(counter<2):
                        try:
                            V_reference             =   Training_output[string][:,iband]
                            x_train                 =   Training_set[string]
                            gp1                      =   gp_emulator.GaussianProcess(x_train, V_reference)
                            gp1.learn_hyperparameters(verbose=False)

                            V_emulated, V_var, dV   =   gp1.predict(x_train)

                            E                       =   V_reference-V_emulated
                            E_rel                   =   (E/(V_reference + 1e-80))*100

                            if string=='tau':
                                ithresh             =   V_reference>0.2
                                rmse                =   np.sqrt(np.mean(E_rel[ithresh]**2))
                                # rmse_alt            =   np.sqrt(np.mean(E[ithresh]**2)) / np.sqrt(np.mean(V_reference[ithresh]**2)) * 100
                                (q75_rel, q95_rel)  =   np.percentile(np.abs(E_rel[ithresh]),[75, 95])
                            elif string=='Latm':
                                ithresh             =   (training_set[:,0]>0.1) * (training_set[:,3]>0.1)
                                rmse                =   np.sqrt(np.mean(E_rel[ithresh]**2))
                                # rmse_alt            =   np.sqrt(np.mean(E[ithresh]**2)) / np.sqrt(np.mean(V_reference[ithresh]**2)) * 100
                                (q75_rel, q95_rel)  =   np.percentile(np.abs(E_rel[ithresh]),[75, 95])

                                # if rmse>1:
                                #     counter         =   1000

                            else:
                                rmse                =   np.sqrt(np.mean(E_rel**2))
                                # rmse_alt            =   np.sqrt(np.mean(E**2)) / np.sqrt(np.mean(V_reference**2)) * 100
                                (q75_rel, q95_rel)  =   np.percentile(np.abs(E_rel),[75, 95])


                            if q75_rel<Q75_rel:
                                gp                  =   gp1
                                RMSE                =   rmse
                                Q75_rel             =   q75_rel
                                Q95_rel             =   q95_rel


                            counter                 =   counter + 1

                        except:
                            counter                 =   counter + 1
                            print
                        r.append(RMSE)

                        print thresh, rmse, q75_rel, q95_rel

                    Emulator[string][iband]         =   gp
                    RMSE_[string]                   =   RMSE
                    Q75_rel_[string]                =   Q75_rel
            cPickle.dump(Emulator, open(emulator_home + simulationstr+ threshstr + '_Emulators'+'.pkl', 'wb'))
            cPickle.dump(RMSE_, open(emulator_home + simulationstr+ threshstr + '_Emulator_RMSE'+'.pkl', 'wb'))

    #######################################################
    # Investigate errors
    #######################################################
    from scipy.stats.stats import spearmanr
    def InvestigateE(parameters, x_train, training_set, V_reference, E, E_rel, string, iband=0):
        Q95_rel                                 =   np.percentile(np.abs(E_rel),[95])[0]
        RMSE                                    =   np.sqrt(np.mean(E_rel**2))

        thresh                                  =   0.
        ierror                                  =   np.abs(E_rel)>=thresh

        nvar                                    =   np.shape(x_train)[1]
        nc                                      =   3
        nr                                      =   int(np.ceil(nvar/float(nc)))
        plt.figure(figsize=[20,15])
        for ivar in xrange(nvar):
            (R2,P)                              =   spearmanr(x_train[ierror,ivar], np.abs(E[ierror]))
            plt.subplot(nr,nc,ivar+1)
            plt.plot(training_set[ierror,ivar], np.abs(E_rel[ierror]) + 1e-19,'.')
            plt.ylabel(string + '-band %02.0f' % iband)
            plt.xlabel(parameters[ivar])
            plt.legend(['[R = %5.2f, P=%5.2f]' % (R2,P)], loc=0)
            if ivar==1:
                plt.title('RMSE = %5.2f%%, Q95 = %5.2f%% ' % (RMSE, Q95_rel))
        plt.savefig('Investigation into Errors of ' + string + ' for band%02.0f.png' % iband)

        if string =='tau':
            tt = 0.0
        elif string=='Latm':
            tt = 0.2

        ithresh2 = (np.abs(x_train[:,2]-x_train[:,1])>=15); (x_train[:,3]>=tt);   (np.arccos(x_train[:,4])/np.pi*180 <=36.);
        ithresh1 = ithresh2==0
        ivar = 0

        plt.figure(figsize=[20,15])
        plt.subplot(3,1,1)
        plt.plot( training_set[ithresh1,ivar], np.abs(V_reference[ithresh1]),'r.')
        plt.plot( training_set[ithresh2,ivar], np.abs(V_reference[ithresh2]),'.')
        plt.ylabel(string + '-Values')
        plt.subplot(3,1,2)
        plt.plot( training_set[ithresh1,ivar], np.abs(E[ithresh1]),'r.')
        plt.plot( training_set[ithresh2,ivar], np.abs(E[ithresh2]),'.')
        plt.ylabel(string + '-Error')
        plt.subplot(3,1,3)
        plt.semilogy( training_set[ithresh1,ivar], np.abs(E_rel[ithresh1])+1e-19,'r.')
        plt.semilogy( training_set[ithresh2,ivar], np.abs(E_rel[ithresh2])+1e-19,'.')
        plt.xlabel(parameters[ivar])
        plt.ylabel(string + '-Rel Error')
        plt.legend(['AOT< 0.2 [rMSE=%4.3f]' % np.sqrt(np.mean(E_rel[ithresh1]**2)),'AOT>=0.2 [rMSE=%4.3f]' % np.sqrt(np.mean(E_rel[ithresh2]**2))], loc=0)
        plt.savefig('Investigation (deeper) into Errors of ' + string + ' .png')


    rmse_                                           =   dict()
    for string in [string_output[0]]:
        rmse_[string]                               =   []
        for iband in bands:
            x_train                                 =   Training_set[string]
            V_reference                             =   Training_output[string][:,iband]

            gp                                      =   Emulator[string][iband]
            V_emulated, V_var, dV                   =   gp.predict(x_train)
            E                                       =   Training_output[string][:,iband]-V_emulated
            E_rel                                   =   (E/(V_reference + 1e-80))*100

            if string=='tau':
                ithresh                             =   V_reference>0.2
            elif string=='Latm':
                ithresh                             =   (training_set[:,0]>0.1) * (training_set[:,3]>0.1)

            rmse                                    =   np.sqrt(np.mean(E_rel[ithresh]**2))
            rmse_[string].append(rmse)
            InvestigateE(parameters, x_train[ithresh], training_set[ithresh], V_reference[ithresh], E[ithresh], E_rel[ithresh], string, iband)


    toc                                         =   datetime.datetime.now()
    print '-Creating/Loading Emulators finished, duration was %f hours' % ((toc-tic).total_seconds()/60/60)


    # for path radiance
    #######################################################
    # observations
    #     when H20 (G) is low   (<1.0), the rel Error is very high.
    #     when AOT is low       (<0.1), the rel Error is very high.

    # explanations
    #     rel errors become very high if original values are low:
    #     tt = 0.0; plt.figure(); plt.plot( V_reference, np.abs(E_rel),'.')
    #
    #     1) when H20 (G) is very low the Latm becomes very low (~0.0). This can have the effect of blowing up the relative error.
    #     2) when AOT (G) is very low the Latm becomes very low (~0.0). This can have the effect of blowing up the relative error.
    #
    #     Hypothesis:
    #     AOT is a single parameter that describes the aggregated effects of the scattering and consequently
    #     is dependent on aerosol mass loading, scattering, and absorption efficiencies. In addition to these
    #     inherent properties AOT also depends on the local RH [1]. The hygroscopic property of aerosols is represented
    #     by the aerosol humidification factor (AHF). This f (RH) is dependent not only on RH but also on the chemical
    #     and optical properties of aerosols [1]. In modtran the effect of aerosols is captured by multiple aerosol models
    #     [2] (figure 1) in order to te different inherent properties (such as RH). It is not garantued that the
    #     sensitivities of these models to the humidity is the same for all models. This mixed behaviour might be more
    #     difficult for an gaussian process model to emulate. The relative errors in emulation particularly are apparent
    #     when the values of variable to be emulated is low. Any error will then be enhanced due to the low values which
    #     is used to estimate the relative humidity.

    #[1] Effect of aerosol humidification on the column aerosol optical thickness over the Atmospheric Radiation Measurement Southern, Great Plains site
    #Myeong-Jae Jeong,1,3 Zhanqing Li,1 Elisabeth Andrews,2 and Si-Chee Tsay3
    # [2] The Aerosol Models in MODTRAN: Incorporating Selected Measurements from Northern Australia. S. B. Carr

    # for path transmissivity
    #######################################################
    # observations
    #     It appears that H20-errors and Tatm(01)-errors are correlated. This might be because the emulators fail to
    #     reproduce the dependency of H20 and temperatures. Due to the 'correction procedure' implemented to get the actual H20 values for low air temperatures
    #     the training_set does not have combinations of high H20 values with low air temperaturs. This region leads this part of the hypervolume is undersampled
    #     while the region with 'low-temperatures and low H20 values(@100%RH)' is oversampled. On basis of this, the emulator will assume this dew-temperature curve


    #  explanation:
    #      when H20 (G) is very high (that means the combination of high RH + T) the atm. transmissivity becomes very low (~0.0)
    #     this has the effect that the absolute error between emulation and reference can become very high if E~0).  In short this means
    #     that we find a definitive threshold for which we cannot process the image (a haze threshold)



    ####################################################################################
    ############################# Check Sensitivity of Emulators #######################
    ####################################################################################

    def InvestigateSensitivities(Emulator, em_name,training_set2,parameters,option_value, bands):
        nsc                                         =   np.shape(training_set2)[0]
        nvar                                        =   np.shape(training_set2)[1]
        nc                                          =   3
        nr                                          =   int(np.ceil(nvar/float(nc)))
        emulators_Tmeas                             =   Emulator[em_name]

        miny                                        =   250
        maxy                                        =   370

        plt.figure(figsize=[15,10])
        for iband in bands: #xrange(len(Emulator['tau'])):#xrange(3):
            em                                      =   emulators_Tmeas[iband]

            min_vals                                =   np.min(training_set2, axis=0)
            max_vals                                =   np.max(training_set2, axis=0)

            for i, name in enumerate(parameters):                               #! remove the [-1]
                # print name
                minx                                =   min_vals[i]
                maxx                                =   max_vals[i]
                x                                   =   np.linspace(minx,maxx,100)

                isc                                 =   0

                # old (single scenario)
                x_train2                            =   training_set2[isc]*1.
                Xtrain2, X                          =   np.meshgrid(x_train2, x)
                Xtrain2[:,i]                        =   X[:,0]
                plt.subplot(nr,nc,i+1)
                if option_value=='V':
                    V,V_var, dV                     =   em.predict(Xtrain2, do_unc=True)
                    y                               =   V

                # new (multiple scenarios)
                V_                                  =   []
                V_var_                              =   []
                dV_                                 =   []
                for isc in xrange(nsc):
                    x_train2                        =   training_set2[isc]*1.
                    Xtrain2, X                      =   np.meshgrid(x_train2, x)
                    Xtrain2[:,i]                    =   X[:,0]
                    V,V_var, dV                     =   em.predict(Xtrain2, do_unc=False)
                    V_.append(V)
                    V_var_.append(V_var)
                    dV_.append(dV[:,i])

                plt.subplot(nr,nc,i+1)
                if option_value=='V':
                    V_                              =   np.array(V_)
                    (Q05,Q25, Q50, Q75, Q95)        =   np.percentile(V_,[5, 25, 50, 75, 95], axis=0)

                    plt.plot(x,Q50,'r')
                    plt.fill_between(x,Q05,Q95, color=[0.5, 0.5, 0.5])
                    plt.fill_between(x,Q25,Q75, color=[0.3, 0.3, 0.3])
                    string                          =   em_name

                    if name=='Tmeas_TOA':
                        plt.ylim([miny, maxy])


                elif option_value=='dV':
                    dV_                              =   np.array(dV_)
                    (Q05,Q25, Q50, Q75, Q95)        =   np.percentile(dV_,[5, 25, 50, 75, 95], axis=0)

                    plt.plot(x,Q50,'r')
                    plt.fill_between(x,Q05,Q95, color=[0.5, 0.5, 0.5])
                    plt.fill_between(x,Q25,Q75, color=[0.3, 0.3, 0.3])
                    string                          =   em_name

                    # V,V_var, dV                     =   em.predict(Xtrain2, do_unc=False)
                    # y                               =   dV[:,i]
                    string                          =   'd' + em_name

                elif option_value=='V_var':
                    V_var_                          =   np.array(V_var_)
                    (Q05,Q25, Q50, Q75, Q95)        =   np.percentile(V_var_,[5, 25, 50, 75, 95], axis=0)

                    plt.plot(x,Q50,'r')
                    plt.fill_between(x,Q05,Q95, color=[0.5, 0.5, 0.5])
                    plt.fill_between(x,Q25,Q75, color=[0.3, 0.3, 0.3])

                    string                          =   em_name + '_var'

                    # i1                              =   np.where(y>0)[0]
                    # i2                              =   np.where(y<0)[0]
                    # plt.semilogy(x,np.abs(y),'b')
                    # plt.semilogy(x[i1],y[i1],'g.')
                    # plt.semilogy(x[i2],-y[i2],'r.')

                plt.title(string + '-sensitivity to '+ name)

        plt.legend(['Band 1','Band 2','Band 3'],loc=0)
        plt.savefig('SLTSR TIR '+string+'-sensitivity to MODTRAN variables')
    # plt.close('all')
    for em_name in ['tau']:
        InvestigateSensitivities(Emulator, em_name,training_set_transformed_tau,parameters,'V', bands)        # working
        # InvestigateSensitivities(Emulator, em_name,training_set_transformed_tau,parameters,'dV')        # working

    for em_name in ['Latm']:
        InvestigateSensitivities(Emulator, em_name,training_set_transformed_Latm,parameters,'V', bands)        # working
        # InvestigateSensitivities(Emulator, em_name,training_set_transformed_Latm,parameters,'dV')        # working

    print RMSE_

    q
    ####################################################################################
    ############################# Validate Emulators (previously finished run)        ##
    ####################################################################################

    # load previously performed run
    filename_modtroot                           =   'mod5root_.in'
    os.chdir(modtran_model_validate.modtran_dir)
    shutil.copyfile(dir_validation+ "/mod5root_.in", filename_modtroot)

    simulationstr_validation                    =   dir_validation #scenarios_validation[0].split('/')[0]
    scenarios_validation                        =   modtran_model_validate.ReadDifferentScenarios(filename_modtroot)
    training_set_validation                     =   cPickle.load(open(homedir + subdir + simulationstr_validation + '_Training_set' + '.pkl', 'r'))
    parameters_validation                       =   cPickle.load(open(homedir + subdir + simulationstr_validation + '_Parameters' + '.pkl', 'r'))

    I_validation                                =   cPickle.load(open(homedir + subdir + simulationstr_validation + '_I' + '.pkl', 'r'))
    # I_validation, ntrain_validation             =   modtran_model.IdentifyErroneousRuns(I_validation,scenarios_validation,Values_dict_validation)
    # Values_dict_validation                      =   cPickle.load(open(homedir + subdir + simulationstr_validation+ 'Values_dict.pkl', 'r'))
    # varnames_validation                         =   cPickle.load(open(homedir + subdir + simulationstr_validation+ 'varnames.pkl', 'r'))

    band_tau                                    =   cPickle.load(open(homedir + subdir + simulationstr_validation+ '_band_tau_TOA.pkl', 'r'))
    band_Latm_TOA                               =   cPickle.load(open(homedir + subdir + simulationstr_validation+ '_band_Latm_TOA.pkl', 'r'))
    band_Latm_BOA                               =   cPickle.load(open(homedir + subdir + simulationstr_validation+ '_band_Latm_BOA.pkl', 'r'))

    # correct training_set H20-values
    (V_H20_validate, I_validation, training_set_validation, ntrain_validation) =   CorrectH20Values(modtran_model_validate, scenarios_validation, parameters_validation, training_set_validation, I_validation, option_h20)


    # Forward simulate the output with the emulator
    print simulationstr_validation
    os.chdir(modtran_model.homedir)
    Output                                      =   dict()
    for i,string in enumerate(string_output):
        print
        print string
        filename                                =   'SLSTR_' + string + '_emulator.pkl'

        Output[string]                          =   []
        for iband, wv in enumerate(band['wv']):
            print '--- %1.0f' % iband

            em                                  =   Emulator[string][iband]
            V_emulated, V_var, dV               =   em.predict(training_set_validation)

            InvestigateE(parameters, x_train, E, E_rel, string, iband)

            Output[string].append(V_emulated)
        Output[string]                          =   np.array(Output[string])

    # cross compare output
    E_tau                                       =   np.array(Output['tau'] -     band_tau)
    E_L                                         =   np.array(Output['Latm'] -     band_Latm_TOA)

    Erel_tau                                    =   np.abs(E_tau / (band_tau +1e-30)* 100)
    Erel_L                                      =   np.abs(E_L / (band_Latm_TOA+1e-30) * 100)

    for iband, wv in enumerate(band['wv']):
        InvestigateE(parameters, training_set_validation, E_tau, Erel_tau['tau'], 'tau', iband)
        InvestigateE(parameters, training_set_validation, E_tau, Erel_tau['Latm'], 'Latm', iband)


    #######################################################

    (Q05_tau,Q25_tau,Q50_tau,Q75_tau,Q95_tau)   =   np.percentile(E_tau,[05, 25, 50, 75, 95], axis=1)
    (Q05_L,Q25_L,Q50_L,Q75_L,Q95_L)             =   np.percentile(E_L,[05, 25, 50, 75, 95], axis=1)

    (Q05_tau,Q25_tau,Q50_tau,Q75_tau,Q95_tau)   =   np.percentile(Erel_tau,[05, 25, 50, 75, 95], axis=1)
    (Q05_L,Q25_L,Q50_L,Q75_L,Q95_L)             =   np.percentile(Erel_L,[05, 25, 50, 75, 95], axis=1)

    # analyse
    plt.close('all')
    plt.subplot(2,1,1)
    plt.semilogy(Erel_tau.T)
    plt.legend(['Band 1', 'Band 2', 'Band 3'])
    plt.ylabel('tau_em - tau_ref [%]')
    plt.subplot(2,1,2)
    plt.semilogy(Erel_L.T)
    plt.ylabel('L_em - L_ref [%]')
    plt.legend(['Band 1', 'Band 2', 'Band 3'])

    # check emulation!











    # coy

    # load scenarios and data

    # Process data

    # Run Emulators over scenarios

    # perform cross-comparison







    ####################################################################################
    ############################# Create Sensor Simulator        #######################
    ####################################################################################
    def InvestigateSensitivities(Emulator, em_name,training_set2,parameters,option_value):
        isc                                         =   0
        x_train2                                    =   training_set2[isc]*1.
        nvar                                        =   np.shape(x_train2)[0]
        nc                                          =   3
        nr                                          =   int(np.ceil(nvar/float(nc)))
        emulators_Tmeas                             =   Emulator[em_name]

        miny                                        =   250
        maxy                                        =   370

        plt.figure(figsize=[15,10])
        for iband in xrange(3):
            em                                      =   emulators_Tmeas[iband]

            min_vals                                =   np.min(training_set2, axis=0)
            max_vals                                =   np.max(training_set2, axis=0)

            for i, name in enumerate(parameters):                               #! remove the [-1]
                # print name
                minx                                =   min_vals[i]
                maxx                                =   max_vals[i]
                x                                   =   np.linspace(minx,maxx,100)

                Xtrain2, X                          =   np.meshgrid(x_train2, x)
                Xtrain2[:,i]                        =   X[:,0]

                plt.subplot(nr,nc,i+1)
                if option_value=='V':
                    V,V_var, dV                     =   em.predict(Xtrain2, do_unc=False)

                    y                               =   V
                    string                          =   em_name
                    plt.plot(x,y)

                    if name=='Tmeas_TOA':
                        plt.ylim([miny, maxy])

                elif option_value=='dV':
                    V,V_var, dV                     =   em.predict(Xtrain2, do_unc=False)
                    y                               =   dV[:,i]
                    string                          =   'd' + em_name

                    plt.plot(x,y)

                elif option_value=='V_var':
                    V,V_var, dV                     =   em.predict(Xtrain2, do_unc=True)
                    y                               =   V_var
                    string                          =   em_name + '_var'

                    i1                              =   np.where(y>0)[0]
                    i2                              =   np.where(y<0)[0]
                    plt.plot(x,np.abs(y),'b')
                    plt.plot(x[i1],y[i1],'g.')
                    plt.plot(x[i2],-y[i2],'r.')

                plt.title(string + '-sensitivity to '+ name)

        plt.legend(['Band 1','Band 2','Band 3'],loc=0)
        plt.savefig('SLTSR TIR '+string+'-sensitivity to MODTRAN variables')
    class Atcor():
        def __init__(self, em_tau, em_L, iband):
            self.em_tau                          =   em_tau[iband]
            self.em_L                           =   em_L[iband]
            self.iband                          =   iband

        def predict(self,training_set2, do_unc=False):
            do_mu                                   =   True
            do_deriv                                =   True

            iband                               =   self.iband
            # ntrain                                      =   np.shape(training_set3)[0]
            # nbands                                      =   len(self.em_invtau)



            if (do_deriv+do_deriv):
                training_set                        =   training_set2[:,0:-1]
                TTOA                                =   training_set2[:,-1]
                LTOA                                =   planck_model.LUT_forward(TTOA.T)[iband]

                #
                tau, tau_var, dtau                  =   self.em_tau.predict(training_set, do_unc=do_unc)
                Latm, Latm_var, dLatm               =   self.em_L.predict(training_set, do_unc=do_unc)

                # the actual atmospheric correction
                LBOA                                =   (LTOA - Latm)/tau

                # calcualte brightness temperate equivalent
                LBOA                                =   np.tile(LBOA,[3,1])
                TBOA                                =   planck_model.LUT_inverse(LBOA.T)[iband]
                V                                   =   TBOA


            # derivative
            if do_deriv == True:
                dTTOA                               =   1e-5
                dLTOA                               =   planck_model.LUT_forward((TTOA+dTTOA).T)[iband] - LTOA
                dLTOA_dTTOA                         =   dLTOA/dTTOA

                dLBOA_dX                            =   ((dLatm.T*tau - dtau.T*(LTOA-Latm))/ (tau**2)).T
                dLBOA_dLTOA                         =   1/tau
                dLBOA_dTTOA                         =   dLBOA_dLTOA*dLTOA_dTTOA

                dLBOA_dX                            =   np.vstack([dLBOA_dX.T, dLBOA_dTTOA]).T

                dLBOA                               =   1e-4
                dTBOA                               =   planck_model.LUT_inverse( (LBOA + dLBOA).T)[iband] - TBOA
                dTBOA_dLBOA                         =   dTBOA / dLBOA

                dTBOA_dX                            =   (dTBOA_dLBOA.T * dLBOA_dX.T).T
                dV                                  =    dTBOA_dX
            else:
                dV                                  =   tau_var


            if do_unc==True:
                # F(a, b)   =   F(0) +  dF/da*a             + dF/db *b
                # var(F)    =           (dF/da)^2*var(a)    + (dF/db)^2 * var(b) + 2*(dF/da)*(dF/db)*var(a)*rho*var(b)
                # rho = correlation coefficient between a and b

                # print 'not implemented yet, due to variance from emulators being negative'
                # import pdb
                # pdb.set_trace()

                V_var                               =   tau_var



            else:
                V_var                               =    tau_var

            return V,V_var, dV
    class Atcor_inv():
        def __init__(self, em_invtau, em_Linvtau, iband):
            self.em_invtau                              =   em_invtau[iband]
            self.em_Linvtau                             =   em_Linvtau[iband]
            self.iband                                  =   iband

        def predict(self,training_set2, do_unc=False):
            iband                                       =   self.iband
            # ntrain                                      =   np.shape(training_set3)[0]
            # nbands                                      =   len(self.em_invtau)

            training_set                            =   training_set2[:,0:-1]
            TTOA                                    =   training_set2[:,-1]
            dTTOA                                   =   1e-5
            LTOA                                    =   planck_model.LUT_forward(TTOA.T)[iband]
            dLTOA                                   =   planck_model.LUT_forward((TTOA+dTTOA).T)[iband] - LTOA
            dLTOA_dTTOA                             =   dLTOA/dTTOA

            #
            a, a_var, da                            =   self.em_invtau.predict(training_set, do_unc=do_unc)
            b, b_var, db                            =   self.em_Linvtau.predict(training_set, do_unc=do_unc)

            # mu
            LBOA                                    =   LTOA*a + b
            LBOA                                    =   np.tile(LBOA,[3,1])
            TBOA                                    =   planck_model.LUT_inverse(LBOA.T)[iband]
            V                                       =   TBOA

            do_deriv=True
            # derivative
            if do_deriv == True:
                dLBOA_dX                            =   (da.T*LTOA).T + db
                dLBOA_dLTOA                         =   a
                dLBOA_dTTOA                         =   dLBOA_dLTOA*dLTOA_dTTOA

                dLBOA_dX                            =   np.vstack([dLBOA_dX.T, dLBOA_dTTOA]).T

                dLBOA                               =   1e-4
                dTBOA                               =   planck_model.LUT_inverse( (LBOA + dLBOA).T)[iband] - TBOA
                dTBOA_dLBOA                         =   dTBOA / dLBOA

                dTBOA_dX                            =   (dTBOA_dLBOA.T * dLBOA_dX.T).T

                # if iband==2:
                    # import pdb
                    # pdb.set_trace()
                dV                                  =    dTBOA_dX
            else:
                dV                                  =   da


            if do_unc==True:
                # L2T_simple                          =   LTOA/TTOA
                # print np.shape(dTBOA_dLBOA)

                a_var                               =   a_var #/  (dLTOA_dTTOA**2)      # to compensate for the Temperature2Radiances transformation
                b_var                               =   b_var #/  (dLTOA_dTTOA**2)      # to compensate for the Temperature2Radiances transformation

                V_var                               =   (a_var.T*LTOA**2).T + b_var

                V_var                               =   V_var# * (dTBOA_dLBOA**2)        # to compensate for the Radiances2Temperature transformation

                import pdb
                pdb.set_trace()

                # dLBOA_dLTOA                         =   a
                # dLBOA_dTTOA                         =   dLBOA_dLTOA*dLTOA_dTTOA
                #
                # dLBOA_dX                            =   np.vstack([dLBOA_dX.T, dLBOA_dTTOA]).T
                #
                # dLBOA                               =   1e-4
                # dTBOA                               =   planck_model.LUT_inverse( (LBOA + dLBOA).T)[iband] - TBOA
                # dTBOA_dLBOA                         =   dTBOA / dLBOA
                #
                # dTBOA_dX                            =   (dTBOA_dLBOA.T * dLBOA_dX.T).T
                #
                # V_var                               =    dTBOA_dX

            else:
                V_var                               =    b_var
            #
            #
            # # LBOA0                                   =   (LBOA.T).T
            # # vl1                                   =   (vL.T + vL_var.T).T
            # LBOA2                                   =   (LBOA.T + dLBOA_dX.T).T
            #
            # # in [K] units
            # dV                                      =   np.zeros_like(dLBOA_dX)
            # for i in xrange(np.shape(vl2)[1]):
            #     LBOA0                               =   np.tile(LBOA,[3,1])
            #     # VL1                                 =   np.tile(vl1,[3,1])
            #     LBOA2                               =   np.tile(LBOA2[:,i],[3,1])
            #
            #     TBOA1                               =   planck_model.LUT_inverse(LBOA0.T)[iband]
            #     # V1                                  =   planck_model.LUT_inverse(VL1.T)[iband]
            #     TBOA2                               =   planck_model.LUT_inverse(LBOA2.T)[iband]
            #
            #     V                                   =   V0
            #     # V_var                               =   V1 - V0
            #     dV[:,i]                             =   (TBOA2 - TBOA0)

                # if i==3:
                #     print 'I am here'
                #     import pdb
                #     pdb.set_trace()

                # V_var                               =   b_var

            # import uncertainties as unc
            # import uncertainties.unumpy as unumpy

            return V,V_var, dV

    Atmos_em                                    =   {'Tmeas':[]}
    Atmos_em['Tmeas'].append(Atcor(Emulator['tau'], Emulator['Latm'], 0))
    Atmos_em['Tmeas'].append(Atcor(Emulator['tau'], Emulator['Latm'], 1))
    Atmos_em['Tmeas'].append(Atcor(Emulator['tau'], Emulator['Latm'], 2))


    ####################################################################################
    ############################# Analyise Sensor Simulator Sensitivity ################
    ####################################################################################

    # plt.close('all')
    em_names                                    =   ['Tmeas']
    for em_name in em_names:
        InvestigateSensitivities(Atmos_em, em_name,training_set2,parameters2,'V')                  # working
        # InvestigateSensitivities(Atmos_inv_em, em_name,training_set2,parameters2,'V')                   # working


    for em_name in em_names:
        InvestigateSensitivities(Atmos_em, em_name,training_set2,parameters2,'dV')                  # working
        # InvestigateSensitivities(Atmos_inv_em, em_name,training_set2,parameters2,'dV')                   # working


    # for em_name in em_names:
    #     InvestigateSensitivities(Atmos_inv_em, em_name,training_set2,parameters2,'V_var')            # working
    #     InvestigateSensitivities(Atmos_em, em_name,training_set2,parameters2,'V_var')            # working


    ####################################################################################


    # x_train                                     =   x_train
    # Lmeas                                       =   Lmeas_BOA
    # Latm                                        =   Latm_BOA
    # Tmeas                                       =   Tmeas_BOA
    # Tatm                                        =   Tatm_BOA
    # emulators_Tmeas                             =   emulators_meas_BOA
    # emulators_Latm                              =   emulators_atm_BOA


    ##########################################
    # create/execute/load  modtran runs (and if needed finish them)
    ######################################################################################################
    # x_train, Lmeas, Latm, Tmeas, Tatm, emulators_Tmeas, emulators_Latm = create_modtran_SLSTR_emulators ( modtran_model,      sensor, subdir, option_load, ntrain = ntrain, groupsize=Groupsize)
    # x_train,Lmeas_BOA, Latm_BOA, Tmeas_BOA, Tatm_BOA, emulators_Lmeas_BOA, emulators_Latm_BOA


    # em = Emulator[em_name][iband]
    # V,V_var, dV = em.predict(training_set,do_unc=False)
    ######################################################################################################
    # investigate sensitivities
    ######################################################################################################
    # emulators_Lmeas, emulators_Latm, Lmeas, Latm, Tmeas, Tatm= create_modtran_SLSTR_emulators ( modtran_model_validate, sensor, subdir, option_load, ntrain = ntrain, groupsize=Groupsize)


    # plt.close('all')


    ################################################################################################################
    ################################################################################################################
    ################################################################################################################
    ################################################################################################################
    ################################################################################################################

    diffV                                       =   Vvalidate-Vemulated
    Qa02                                        =   np.percentile ( diffV, [2], axis=1 )[0]
    Qa05                                        =   np.percentile ( diffV, [5], axis=1 )[0]
    Qa25                                        =   np.percentile ( diffV, [25], axis=1 )[0]
    Qa50                                        =   np.percentile ( diffV, [50], axis=1 )[0]
    Qa75                                        =   np.percentile ( diffV, [75], axis=1 )[0]
    Qa95                                        =   np.percentile ( diffV, [95], axis=1 )[0]
    Qa98                                        =   np.percentile ( diffV, [98], axis=1 )[0]


    diffV_rel                                   =   diffV/Vmodtran*100
    Qa02_rel                                    =   np.percentile ( diffV_rel, [2], axis=1 )[0]
    Qa05_rel                                    =   np.percentile ( diffV_rel, [5], axis=1 )[0]
    Qa25_rel                                    =   np.percentile ( diffV_rel, [25], axis=1 )[0]
    Qa50_rel                                    =   np.percentile ( diffV_rel, [50], axis=1 )[0]
    Qa75_rel                                    =   np.percentile ( diffV_rel, [75], axis=1 )[0]
    Qa95_rel                                    =   np.percentile ( diffV_rel, [95], axis=1 )[0]
    Qa98_rel                                    =   np.percentile ( diffV_rel, [98], axis=1 )[0]

    wl                                          =   band['wv']

    plt.figure(figsize=[15,5])
    plt.subplot(2,1,1)
    plt.plot(wl, diffV[:,0:-1:10],'b.')
    plt.plot(wl, Qa50,'b')
    plt.fill_between(wl, Qa05, Qa95, color=[0.8, 0.8, 0.8])
    plt.fill_between(wl, Qa25, Qa75, color=[0.3, 0.3, 0.3])
    plt.ylabel('Tdiff [K]')
    plt.xlim([np.min(wl)*0.99, np.max(wl)*1.01])
    plt.ylim([np.min(Qa02), np.max(Qa98)])

    plt.subplot(2,1,2)
    plt.plot(wl, diffV_rel,'.b')
    plt.plot(wl, Qa50_rel,'b')
    plt.fill_between(wl, Qa05_rel, Qa95_rel, color=[0.8, 0.8, 0.8])
    plt.fill_between(wl, Qa25_rel, Qa75_rel, color=[0.3, 0.3, 0.3])
    plt.title('Validation of Tmeas - emulators')
    plt.ylabel('Relative Tdiff [%]')
    plt.xlabel('Wavelength [nm]')
    plt.xlim([np.min(wl)*0.99, np.max(wl)*1.01])
    plt.ylim([np.min(Qa02_rel), np.max(Qa98_rel)])

    ####################################################################################
    ############################# Show                        ##########################
    ####################################################################################

    # show MODTRAN outputs
    # modtran_model.ShowOutput(Values_dict, tau_oo, tau_do, L_u_TOA, L_d_BOA)

    #  show Reduced band simulations
    plt.figure(figsize=[15,5])
    plt.subplot(2,2,1)
    plt.plot(WL,tau_oo[:,0:-1:100],'k', alpha=0.1)
    plt.vlines(band['min'],0,1,color=[1, 0, 0])
    plt.vlines(band['max'],0,1,color=[1, 0, 0])
    plt.plot(band['wv'],band_tau_oo[:,0:-1:100])
    plt.xlim([2000, 15000])
    plt.xlim([3500, 4000])

    # for i in xrange(ntrain):


    #
    # plt.subplot(2,2,2)
    # plt.plot(WL,tau_oo,'k', alpha=0.1)
    # for i in xrange(ntrain):
    #     plt.errorbar(band['wv'],band_tau_oo[:,i], yerr=band_tau_oo_std[:,i]/2,xerr=band['width']/2)
    #     plt.xlim([2000, 15000])
    #
    # plt.subplot(2,2,3)
    # plt.plot(WL,L_u_TOA[:,0],'k', alpha=0.1)
    # for i in xrange(ntrain):
    #     plt.errorbar(band['wv'],band_L_u_TOA[:,i], yerr=band_L_u_TOA_std[:,i]/2,xerr=band['width']/2)
    # plt.xlim([400, 2000])
    #
    # plt.subplot(2,2,4)
    # plt.plot(WL,L_u_TOA[:,0],'k', alpha=0.1)
    # for i in xrange(ntrain):
    #     plt.errorbar(band['wv'],band_L_u_TOA[:,i], yerr=band_L_u_TOA_std[:,i]/2,xerr=band['width']/2)
    # plt.xlim([2000, 15000])





        #     if i==10:
        #         varnames            =   line.split()
        #     elif i>10:
        #
        #         if line <> ' -9999.\n':
        #             words           =   line.split()
        #
        #             V               =   [eval(word) for word in words]
        #
        #             if len(V)<>len(varnames):
        #                 V2          =   np.zeros(len(varnames))*np.NaN
        #                 V2[0:3]     =   V[0:3]
        #                 V2[4:-1]    =   V[3:-1]
        #             else:
        #                 V2          =   V
        #             values.append(V2)
        # if len(values)>0:
        #     Values                      =   np.array(values)
        # else:
        #     Values                      =   Values*np.NaN

        # except:
        #
        #     # in case of erroneous data
        #     Values                          =   Values*np.NaN




        # store Values
        # Values2.append(Values)