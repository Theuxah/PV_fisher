#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 11:24:58 2024

@author: Laurent MAGRI-STELLA

C to Python translation of PV_fisher code by Cullan HOWLETT
"""

import numpy as np
import numba
import scipy as sc
import matplotlib.pyplot as plt
import os

# Global parameters
nparams = 4
Data = [0,1,3,4]   # A vector of flags for the parameters we are erested in (0=beta, 1=fsigma8, 2=r_g, 3=sigma_g, 4=sigma_u). MAKE SURE THE LENGTH OF THIS VECTOR, NPARAMS AND THE ENTRIES AGREE/MAKE SENSE, OR YOU MIGHT GET NONSENSE RESULTS!!
nziter = 10            # Now many bins in redshift between zmin and zmax we are considering
zmin = 0.0         # The minimum redshift to consider (You must have power spectra that are within this range or GSL spline will error out)
zmax = 0.5         # The maximum redshift to consider (You must have power spectra that are within this range or GSL spline will error out)
Om = 0.3121        # The matter density at z=0
c = 299792.458     # The speed of light in km/s
gammaval = 0.55    # The value of gammaval to use in the forecasts (where f(z) = Om(z)^gammaval)
r_g = 1.0          # The cross correlation coefficient between the velocity and density fields
beta0 = 0.393      # The value of beta (at z=0, we'll modify this by the redshift dependent value of bias and f as required)
sigma80 = 0.8150   # The value of sigma8 at z=0
sigma_u = 13.00    # The value of the velocity damping parameter in Mpc/h. I use the values from Jun Koda's paper
sigma_g = 4.24     # The value of the density damping parameter in Mpc/h. I use the values from Jun Koda's paper
kmax = 0.2         # The maximum k to evaluate for dd, dv and vv correlations (Typical values are 0.1 - 0.2, on smaller scales the models are likely to break down).
survey_area = [0.0, 0.0, 1.745]   # We need to know the survey area for each survey and the overlap area between the surveys (redshift survey only first, then PV survey only, then overlap. 
# For fully overlapping we would have {0, 0, size_overlap}. For redshift larger than PV, we would have {size_red-size_overlap, 0, size_overlap}). Units are pi steradians, such that full sky is 4.0, half sky is 2.0 etc.
error_rand = 300.0    # The observational error due to random non-linear velocities (I normally use 300km/s as in Jun Koda's paper)
error_dist = 0.05     # The percentage error on the distance indicator (Typically 0.05 - 0.10 for SNe IA, 0.2 or more for Tully-Fisher or Fundamental Plane) 
verbosity = 0         # How much output to give: 0 = only percentage errors on fsigma8, 1 = other useful info and nuisance parameters, 2 = full fisher and covariance matrices

# The number of redshifts and the redshifts themselves of the input matter and velocity divergence power spectra. 
# These numbers are multiplied by 100, converted to s and written in the form _z0p%02d which is then appended to the filename Pvel_file. See routine read_power. 
nzin = 11
zin = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
Pvel_file = "./example_files/example_pk" # The file containing the velocity divergence power spectrum. Don't include .dat as we'll append the redshifts on read in

# The files containing the number density of the surveys. First is the PV survey, then the redshift survey. These files MUST have the same binning and redshift range, 
# so that the sum over redshift bins works (would be fine if we used splines), i.e., if one survey is shallower then that file must contain rows with n(z)=0.
# I also typically save nbar x 10^6 in the input file to make sure I don't lose precision when outputting small nbar values to files. This is corrected when the nbar file
# is read in, so see the read_nz() routine!
nbar_file = ["./example_files/example_nbar_vel.dat",
"./example_files/example_nbar_red.dat"]    

# Other global parameters
pkkmin = 0        # The minimum kmin to egrate over, based on the input power spectrum file
pkkmax = 0        # The maximum k in the input power spectrum. The maximum k to egrate over is the smallest of this or kmax
pmmarray = []
pmtarray = []
pttarray = []
rarray = []
zarray = []
deltararray = []
growtharray = []
nbararray = []


def ezinv(x):
    Om = 1.0
    return 1.0 / np.sqrt(Om * (1.0 + x) * (1.0 + x) * (1.0 + x) + (1.0 - Om))

def rz(red):
    result,error = sc.integrate.quad(ezinv, 0, red)
    return c * result / 100.0

    
def growthfunc(x):
    red = 1.0/x - 1.0
    Omz = Om * ezinv(red) * ezinv(red) / (x * x * x)
    f = np.power(Omz, gammaval)
    return f / x


def growthz(red):
    a = 1.0 / (1.0 + red)
    result,error = sc.integrate.quad(growthfunc, a, 1.0)
    return np.exp(-result)

def read_power():
    global pkkmax, pkkmin, pmmarray, pmtarray, pttarray, nzin

    pmmarray = []
    pmtarray = []
    pttarray = []

    for i in range(nzin):
        Pvel_file_in = f"{Pvel_file}_z0p{int(100.0 * zin[i]):02d}.dat"

        try:
            with open(Pvel_file_in, "r") as fp:
                lines = fp.readlines()
        except IOError:
            print(f"\nERROR: Can't open power file '{Pvel_file_in}'.\n\n")
            exit(0)

        data = [list(map(float, line.split())) for line in lines if not line.startswith("#")]
        NK = len(data)

        if i == 0:
            karray = np.zeros(NK)
            deltakarray = np.zeros(NK - 1)

        pmmarray.append(np.zeros(NK))
        pmtarray.append(np.zeros(NK))
        pttarray.append(np.zeros(NK))

        for j in range(NK):
            tk, pkdelta, pkdeltavel, pkvel = data[j]
            if i == 0:
                karray[j] = tk
            pttarray[i][j] = pkvel
            pmmarray[i][j] = pkdelta
            pmtarray[i][j] = pkdeltavel

    for i in range(NK - 1):
        deltakarray[i] = karray[i + 1] - karray[i]

    pkkmin = karray[0]
    pkkmax = karray[NK - 1]

    if pkkmax < kmax:
        print("ERROR: The maximum k in the input power spectra is less than k_max")
        exit(0)

    return NK, karray, deltakarray, pmmarray, pmtarray, pttarray, pkkmin, pkkmax


def read_nz():
    NRED = np.zeros(2, dtype=int)
    nbararray = np.zeros((2,), dtype=object)
    zinarray = None

    for nsamp in range(2):
        with open(nbar_file[nsamp], 'r') as fp:
            lines = [line for line in fp if not line.startswith("#")]
            NRED[nsamp] = sum(1 for line in lines if float(line.split()[0]) <= zmax)
            if nsamp == 0:
                zinarray = np.zeros(NRED[nsamp])
            nbararray[nsamp] = np.zeros(NRED[nsamp])

        with open(nbar_file[nsamp], 'r') as fp:
            for i, line in enumerate(lines):
                if i >= NRED[nsamp]:
                    break
                tz, tnbar = map(float, line.split())
                if nsamp == 0:
                    zinarray[i] = tz
                nbararray[nsamp][i] = 1.0e-6 * tnbar

    if NRED[1] != NRED[0]:
        raise ValueError("The number of redshift bins for each sample must match")

    zarray = np.zeros(NRED[0])
    rarray = np.zeros(NRED[0])
    deltararray = np.zeros(NRED[0])
    growtharray = np.zeros(NRED[0])

    for i in range(NRED[0] - 1):
        zarray[i] = (zinarray[i+1] + zinarray[i]) / 2.0
        rarray[i] = rz(zarray[i])
        deltararray[i] = rz(zinarray[i+1]) - rz(zinarray[i])
        growtharray[i] = growthz(zarray[i]) / growthz(0.0)

    zarray[NRED[0]-1] = (zmax + zinarray[NRED[0]-1]) / 2.0
    rarray[NRED[0]-1] = rz(zarray[NRED[0]-1])
    deltararray[NRED[0]-1] = rz(zmax) - rz(zinarray[NRED[0]-1])
    growtharray[NRED[0]-1] = growthz(zarray[NRED[0]-1]) / growthz(0.0)

    growth_spline = sc.interpolate.CubicSpline(zarray, growtharray)

    nbins = 400
    REDMIN = 0.0
    REDMAX = 2.0
    redbinwidth = (REDMAX-REDMIN) / (nbins-1)
    ztemp = np.linspace(REDMIN, REDMAX, nbins)
    rtemp = np.array([rz(z) for z in ztemp])
    r_spline = sc.interpolate.CubicSpline(ztemp, rtemp)

    return r_spline, growth_spline, NRED, zarray, rarray, deltararray, growtharray, nbararray

def mu_integrand(mu, p):

    numk = int(p[0])
    k = p[1]
    zminval = p[4]
    zmaxval = p[5]

    Pmm_array = np.array([pmmarray[j][numk] for j in range(nzin)])
    Pmt_array = np.array([pmtarray[j][numk] for j in range(nzin)])
    Ptt_array = np.array([pttarray[j][numk] for j in range(nzin)])

    Pmm_spline = sc.interpolate.CubicSpline(zin, Pmm_array)
    Pmt_spline = sc.interpolate.CubicSpline(zin, Pmt_array)
    Ptt_spline = sc.interpolate.CubicSpline(zin, Ptt_array)

    dendamp = np.sqrt(1.0 / (1.0 + 0.5 * (k * k * mu * mu * sigma_g * sigma_g)))
    veldamp = np.sin(k * sigma_u) / (k * sigma_u)

    result_sum = 0.0
    for i in range(NRED[0]):

        zval = zarray[i]
        r_sum = 0.0
        r = rarray[i]
        deltar = deltararray[i]

        if zval < zminval:
            continue
        if zval > zmaxval:
            break

        dd_prefac=0.0
        dv_prefac=0.0
        vv_prefac=0.0
        P_gg=0.0
        P_ug=0.0
        P_uu=0.0

        sigma8 = sigma80 * growtharray[i]

        # First lets calculate the relevant power spectra. Interpolate the power spectra linearly in redshift
        Pmm = Pmm_spline(zval)
        Pmt = Pmt_spline(zval)
        Ptt = Ptt_spline(zval)

        Omz = Om*ezinv(zval)**2*(1.0+zval)**3
        f = Omz**gammaval
        beta = f*beta0*growtharray[i]/Om**0.55

        vv_prefac  = 1.0e2*f*mu*veldamp/k
        dd_prefac = (1.0/(beta*beta) + 2.0*r_g*mu*mu/beta + mu**4)*f*f*dendamp**2
        dv_prefac = (r_g/beta + mu**2)*f*dendamp
        P_gg = dd_prefac*Pmm
        P_ug = vv_prefac*dv_prefac*Pmt
        P_uu = vv_prefac**2*Ptt

        # And now the derivatives. Need to create a matrix of derivatives for each of the two parameters of interest
        dPdt1 = np.zeros((2, 2))
        dPdt2 = np.zeros((2, 2))

        if p[2] == 0:  # Differential w.r.t betaA
            value = -2.0*(1.0/beta + r_g*mu*mu)*f*f*dendamp**2*Pmm/(beta*beta)
            dPdt1[0, 0] = value
            value = -(vv_prefac*f*r_g*dendamp*Pmt)/(beta*beta)
            dPdt1[0, 1] = value
            dPdt1[1, 0] = value
        elif p[2] == 1:  # Differential w.r.t fsigma8
            value = 2.0*(f/(beta*beta) + 2.0*f*r_g*mu*mu/beta + f*mu**4)*dendamp**2*Pmm/sigma8
            dPdt1[0, 0] = value
            value = 2.0*vv_prefac*(r_g/beta + mu**2)*dendamp*Pmt/sigma8
            dPdt1[0, 1] = value
            dPdt1[1, 0] = value
            value = (2.0*P_uu)/(f*sigma8)
            dPdt1[1, 1] = value
        elif p[2] == 2:  # Differential w.r.t r_g
            value = 2.0*(1.0/beta)*mu*mu*f*f*dendamp**2*Pmm
            dPdt1[0, 0] = value
            value = vv_prefac*(1.0/beta)*f*dendamp*Pmt
            dPdt1[0, 1] = value
            dPdt1[1, 0] = value
        elif p[2] == 3:  # Differential w.r.t sigma_g
            value = -k*k*mu*mu*dendamp**2*sigma_g*P_gg
            dPdt1[0, 0] = value
            value = -0.5*k*k*mu*mu*dendamp**2*sigma_g*P_ug
            dPdt1[0, 1] = value
            dPdt1[1, 0] = value
        elif p[2] == 4:  # Differential w.r.t sigma_u
            value = P_ug*(k*np.cos(k*sigma_u)/np.sin(k*sigma_u) - 1.0/sigma_u)
            dPdt1[0, 1] = value
            dPdt1[1, 0] = value
            value = 2.0*P_uu*(k*np.cos(k*sigma_u)/np.sin(k*sigma_u) - 1.0/sigma_u)
            dPdt1[1, 1] = value

        #Now for the second parameter
        if p[3] == 0:
            dPdt2[0, 0] = -2.0*(1.0/beta + r_g*mu*mu)*f*f*dendamp*dendamp*Pmm/(beta*beta)
            dPdt2[0, 1] = -(vv_prefac*f*r_g*dendamp*Pmt)/(beta*beta)
            dPdt2[1, 0] = -(vv_prefac*f*r_g*dendamp*Pmt)/(beta*beta)
        elif p[3] == 1:
            dPdt2[0, 0] = 2.0*(f/(beta*beta) + 2.0*f*r_g*mu*mu/beta + f*mu*mu*mu*mu)*dendamp*dendamp*Pmm/sigma8
            dPdt2[0, 1] = 2.0*vv_prefac*(r_g/beta + mu*mu)*dendamp*Pmt/sigma8
            dPdt2[1, 0] = 2.0*vv_prefac*(r_g/beta + mu*mu)*dendamp*Pmt/sigma8
            dPdt2[1, 1] = (2.0*P_uu)/(f*sigma8)
        elif p[3] == 2:
            dPdt2[0, 0] = 2.0*(1.0/beta)*mu*mu*f*f*dendamp*dendamp*Pmm
            dPdt2[0, 1] = vv_prefac*(1.0/beta)*f*dendamp*Pmt
            dPdt2[1, 0] = vv_prefac*(1.0/beta)*f*dendamp*Pmt
        elif p[3] == 3:
            dPdt2[0, 0] = -k*k*mu*mu*dendamp*dendamp*sigma_g*P_gg
            dPdt2[0, 1] = -0.5*k*k*mu*mu*dendamp*dendamp*sigma_g*P_ug
            dPdt2[1, 0] = -0.5*k*k*mu*mu*dendamp*dendamp*sigma_g*P_ug
        elif p[3] == 4:
            dPdt2[0, 1] = P_ug*(k*np.cos(k*sigma_u)/np.sin(k*sigma_u) - 1.0/sigma_u)
            dPdt2[1, 0] = P_ug*(k*np.cos(k*sigma_u)/np.sin(k*sigma_u) - 1.0/sigma_u)
            dPdt2[1, 1] = 2.0*P_uu*(k*np.cos(k*sigma_u)/np.sin(k*sigma_u) - 1.0/sigma_u)

        r_sum = 0.0

        # We need to do the overlapping and non-overlapping parts of the surveys separately
        # Assuming the following variables are defined elsewhere in your code
        # survey_area, nbararray, error_dist, r, error_rand, P_gg, P_uu, P_ug, dPdt1, dPdt2, deltar

        r_sum = 0.0
        result_sum = 0.0

        # We need to do the overlapping and non-overlapping parts of the surveys separately
        for surv in range(3):
            surv_sum = 0.0
            if survey_area[surv] > 0.0:
                n_g = 0.0
                n_u = 0.0

                # Set the nbar for each section.
                if surv == 0:
                    n_g = nbararray[1][i]
                elif surv == 1 or surv == 2:
                    error_obs = 100.0 * error_dist * r
                    error_noise = error_rand**2 + error_obs**2
                    n_u = nbararray[0][i] / error_noise
                    if surv == 2:
                        n_g = nbararray[1][i]

                if not (n_u > 0.0 or n_g > 0.0):
                    continue

                # First we need the determinant.
                det = 1.0 + n_u * n_g * (P_gg * P_uu - P_ug * P_ug) + n_u * P_uu + n_g * P_gg

                # Now the inverse matrix.
                iP = np.zeros((2, 2))
                iP[0, 0] = n_u * n_g * P_uu + n_g
                iP[1, 1] = n_g * n_u * P_gg + n_u
                iP[0, 1] = - n_g * n_u * P_ug
                iP[1, 0] = - n_g * n_u * P_ug

                # Finally we need to compute the Fisher integrand by summing over the inverse and differential matrices
                for j in range(2):
                    for m in range(2):
                        for u in range(2):
                            for q in range(2):
                                surv_sum += dPdt1[j, q] * iP[q, u] * dPdt2[u, m] * iP[m, j]

                surv_sum /= det * det
                surv_sum *= survey_area[surv]
                r_sum += surv_sum

            result_sum += r * r * deltar * r_sum
        
        return result_sum

def zeff_integrand(mu, p):
    numk = int(p[0])
    k = p[1]
    zminval = p[2]
    zmaxval = p[3]

    if nzin > 1:
        Pmm_array = [pmmarray[j][numk] for j in range(nzin)]
        Pmt_array = [pmtarray[j][numk] for j in range(nzin)]
        Ptt_array = [pttarray[j][numk] for j in range(nzin)]

        Pmm_spline = sc.interpolate.CubicSpline(zin, Pmm_array)
        Pmt_spline = sc.interpolate.CubicSpline(zin, Pmt_array)
        Ptt_spline = sc.interpolate.CubicSpline(zin, Ptt_array)

    dendamp = np.sqrt(1.0/(1.0+0.5*(k*k*mu*mu*sigma_g*sigma_g)))     # This is unitless
    veldamp = np.sinc(k*sigma_u)                                     # This is unitless

    dVeff = 0.0
    zdVeff = 0.0
    for i in range(NRED[0]):

        zval = zarray[i]
        if zval < zminval:
            continue
        if zval > zmaxval:
            break

        r_sum = 0.0
        r = rarray[i]
        deltar = deltararray[i]

        dd_prefac=0.0
        vv_prefac=0.0
        P_gg=0.0
        P_uu=0.0

        sigma8 = sigma80 * growtharray[i]

        # First lets calculate the relevant power spectra. Interpolate the power spectra linearly in redshift
        Pmm = Pmm_spline(zval)
        Pmt = Pmt_spline(zval)
        Ptt = Ptt_spline(zval)

        Omz = Om*ezinv(zval)*ezinv(zval)*(1.0+zval)*(1.0+zval)*(1.0+zval)
        f = pow(Omz, gammaval)
        beta = f*beta0*growtharray[i]/pow(Om,0.55)

        vv_prefac  = 1.0e2*f*mu*veldamp/k
        dd_prefac = (1.0/(beta*beta) + 2.0*r_g*mu*mu/beta + mu*mu*mu*mu)*f*f*dendamp*dendamp
        P_gg = dd_prefac*Pmm
        P_uu = vv_prefac*vv_prefac*Ptt

        # We need to do the overlapping and non-overlapping parts of the redshifts and PV surveys separately
        for surv in range(3):
            surv_sum = 0.0
            if survey_area[surv] > 0.0:
                error_obs = 100.0*error_dist*r                              # Percentage error * distance * H0 in km/s (factor of 100.0 comes from hubble parameter)
                error_noise = error_rand*error_rand + error_obs*error_obs   # Error_noise is in km^{2}s^{-2}
                n_u = nbararray[0][i]/error_noise                   
                n_g = nbararray[1][i]

                value1 = n_g/(1.0 + n_g*P_gg)
                value2 = n_u/(1.0 + n_u*P_uu)
                surv_sum += value1*value1 + value2*value2

                surv_sum *= survey_area[surv]
                r_sum += surv_sum

        dVeff += r*r*deltar*r_sum
        zdVeff += zval*r*r*deltar*r_sum

    return zdVeff/dVeff

#Now all functions are defined, we can start the main part of the code

NK, karray, deltakarray, pmmarray, pmtarray, pttarray, pkkmin, pkkmax = read_power()
r_spline, growth_spline, NRED, zarray, rarray, deltararray, growtharray, nbararray = read_nz()

# Run some checks
if not ((survey_area[0] > 0.0) or (survey_area[2] > 0.0)):
    for i in range(nparams):
        if Data[i] == 2:
            print("ERROR: r_g is a free parameter, but there is no information in the density field (Fisher matrix will be singular)")
            raise SystemExit
        if Data[i] == 3:
            print("ERROR: sigma_g is a free parameter, but there is no information in the density field (Fisher matrix will be singular)")
            raise SystemExit

if not ((survey_area[1] > 0.0) or (survey_area[2] > 0.0)):
    for i in range(nparams):
        if Data[i] == 4:
            print("ERROR: sigma_u is a free parameter, but there is no information in the velocity field (Fisher matrix will be singular)")
            raise SystemExit

if len(Data) != nparams:
    print("ERROR: Size of Data vector for parameters of interest must be equal to nparams")
    raise SystemExit



Fisher_Tot = np.zeros((nparams, nparams))

print(f"Evaluating the Fisher Matrix for {nziter} bins between [z_min = {zmin}, z_max = {zmax}]")

if verbosity == 0:
    print("#     zmin         zmax         zeff      fsigma8(z_eff)   percentage error(z_eff)")

for ziter in range(nziter):
    zbinwidth = (zmax-zmin)/(nziter)
    zmin_iter = ziter*zbinwidth + zmin
    zmax_iter = (ziter+1.0)*zbinwidth + zmin

    rzmax = r_spline(zmax_iter)  # Assuming r_spline is a callable function
    kmin = np.pi/rzmax

    if verbosity > 0:
        print(f"Evaluating the Fisher Matrix for [k_min = {kmin}, k_max = {kmax}] and [z_min = {zmin_iter}, z_max = {zmax_iter}]")

    k_sum1 = 0.0
    k_sum2 = 0.0
    for numk in range(NK):
        
        if numk != NK-1:
            k = karray[numk]+0.5*deltakarray[numk]
            deltak = deltakarray[numk]
        
        else:
            k = karray[numk]
            deltak = 0
            
        if k < kmin or k > kmax:
            continue

        params = [numk, k, zmin_iter, zmax_iter]
        result, error = sc.integrate.quad(zeff_integrand, 0.0, 1.0, args=(params,))
        
        k_sum1 += k*k*deltak*result
        k_sum2 += k*k*deltak

    z_eff = k_sum1/k_sum2
    if verbosity > 0:
        print(f"Effective redshift z_eff = {z_eff}")

    growth_eff = growth_spline(z_eff)  # Assuming growth_spline is a callable function

    Fisher = np.zeros((nparams, nparams))
    for i in range(nparams):
        for j in range(i, nparams):
            k_sum = 0.0
            for numk in range(NK-1):
                k = karray[numk]+0.5*deltakarray[numk]
                deltak = deltakarray[numk]
                if k < kmin or k > kmax:
                    continue

                params = [numk, k, Data[i], Data[j], zmin_iter, zmax_iter]
                result,error = sc.integrate.quad(mu_integrand, 0.0, 1.0, args=(params,))

                k_sum += k*k*deltak*result

            Fisher[i, j] = k_sum/(4.0*np.pi)
            Fisher[j, i] = k_sum/(4.0*np.pi)

    Fisher_Tot += Fisher

    if verbosity == 2:
        print("Fisher Matrix\n======================")
        for row in Fisher:
            print(row)

    Covariance = sc.linalg.inv(Fisher)

    sigma8 = sigma80 * growth_eff
    Omz = Om*ezinv(z_eff)**2*(1.0+z_eff)**3
    f = Omz**gammaval
    beta = f*beta0*growth_eff/Om**0.55

    if verbosity == 0:
        for i in range(nparams):
            if Data[i] == 1:
                print(f"{zmin_iter}  {zmax_iter}  {z_eff}  {f*sigma8}  {100.0*np.sqrt(Covariance[i, i])/(f*sigma8)}")