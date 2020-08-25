# import modules
import sys
import os
import numpy as np
from sedpy.observate import load_filters
from prospect import prospect_args
from prospect.fitting import fit_model, lnprobfn
from prospect.io import write_results as writer
from prospect.models.sedmodel import PolySpecModel
from prospect.models.templates import TemplateLibrary, adjust_continuity_agebins, adjust_dirichlet_agebins
from prospect.models import priors
from astropy.cosmology import Planck15 as cosmo
from prospect.likelihood import NoiseModel
from prospect.likelihood.kernels import Uncorrelated

# define paths

path_wdir = os.environ['WDIR_halo7d']
filter_folder = path_wdir + '/data/filters/'
filter_lsf = path_wdir + '/data/galaxy_LSF_output/'


# define extra functions

def load_zp_offsets(field):
    # load ZP offsets from Skelton+, a la Leja
    filename = path_wdir+'data/zp_offsets_tbl11_skel14.txt'
    with open(filename, 'r') as f:
        for jj in range(1):
            hdr = f.readline().split()
    dtype = [np.dtype((str, 35)), np.dtype((str, 35)), np.float, np.float]
    dat = np.loadtxt(filename, comments='#', dtype=np.dtype([(hdr[n+1], dtype[n]) for n in xrange(len(hdr)-1)]))
    # load field data
    if field is not None:
        good = dat['Field'] == field
        if np.sum(good) == 0:
            print 'Not an acceptable field name! Returning None'
            print 1/0
        else:
            dat = dat[good]
    return dat


# function to build obs dictionary

def build_obs(objid=1, data_table=path_wdir+'data/halo7d_with_phot.fits', err_floor_phot=0.05, err_floor_spec=0.01, remove_zp_offsets=False,
              S2N_cut=1.0, remove_mips24=False, switch_off_phot=False, switch_off_spec=False, **kwargs):
    """Load photometry from an ascii file.  Assumes the following columns:
    `objid`, `filterset`, [`mag0`,....,`magN`] where N >= 11.  The User should
    modify this function (including adding keyword arguments) to read in their
    particular data format and put it in the required dictionary.
    :param objid:
        The object id for the row of the photomotery file to use.  Integer.
        Requires that there be an `objid` column in the ascii file.
    :param phottable:
        Name (and path) of the ascii file containing the photometry.
    :param luminosity_distance: (optional)
        The Johnson 2013 data are given as AB absolute magnitudes.  They can be
        turned into apparent magnitudes by supplying a luminosity distance.
    :returns obs:
        Dictionary of observational data.
    """
    # Read data
    from astropy.table import Table
    catalog = Table.read(data_table)
    idx_cat = objid-1

    # update field name for filters
    field_name_filters = catalog[idx_cat]['FIELD'].lower().replace("-", "")
    if ((field_name_filters == 'egs') | (field_name_filters == 'egs+')):
        field_name_filters = 'aegis'

    # extract mags from catalog
    filternames = []
    mags = []
    mags_err = []
    for ii in catalog.keys():
        if ('f_' in ii[:2]):
            filternames.append(ii[2:].lower() + '_' + field_name_filters)
            mags.append(catalog[idx_cat][ii])
            mags_err.append(catalog[idx_cat][ii.replace('f_', 'e_')])
    filternames = np.array(filternames)
    mags = np.array(mags)
    mags_err = np.array(mags_err)

    ### add correction to MIPS magnitudes (only MIPS 24 right now!)
    # due to weird MIPS filter conventions
    dAB_mips_corr = np.array([-0.03542, -0.07669, -0.03807])  # 24, 70, 160, in AB magnitudes
    dflux = 10**(-dAB_mips_corr/2.5)

    mips_idx = np.array(['mips_24' in f for f in filternames], dtype=bool)
    mags[mips_idx] *= dflux[0]
    mags_err[mips_idx] *= dflux[0]

    # remove MIPS 24um
    if remove_mips24:
        matching = [s for s in filternames if "mips_24" in s]
        choice_non_mips = (np.array(filternames) != matching)
        filternames = filternames[choice_non_mips]
        mags = mags[choice_non_mips]
        mags_err = mags_err[choice_non_mips]

    # ensure filters available
    choice_finite = np.isfinite(np.squeeze(mags)) & (np.squeeze(mags) != -99.0) & (np.squeeze(mags_err) > 0.0)
    filternames = filternames[choice_finite]
    mags = mags[choice_finite]
    mags_err = mags_err[choice_finite]

    # add error from zeropoint offsets in quadrature
    zp_offsets = load_zp_offsets(field_name_filters.upper())
    band_names = np.array([x['Band'].lower() + '_' + x['Field'].lower() for x in zp_offsets])
    for ii, f in enumerate(filternames):
        match = (band_names == f)
        if match.sum():
            if remove_zp_offsets:
                hst_bands = ['f435w', 'f606w', 'f606wcand', 'f775w', 'f814w',
                             'f814wcand', 'f850lp', 'f850lpcand', 'f125w', 'f140w', 'f160w']
                if f.split('_')[0] not in hst_bands:
                    mags[ii] = mags[ii] / zp_offsets[match]['Flux-Correction'][0]
                    mags_err[ii] = mags_err[ii] / zp_offsets[match]['Flux-Correction'][0]
            mags_err[ii] = ((mags_err[ii]**2) + (mags[ii]*(1-zp_offsets[match]['Flux-Correction'][0]))**2)**0.5

    ### load up obs dictionary for photometry
    obs = {}
    obs['filters'] = load_filters(filternames, directory=filter_folder)
    obs['wave_effective'] = [f.wave_effective for f in obs['filters']]
    obs['maggies'] = mags * 1e-10
    obs['maggies_unc'] = np.clip(mags_err * 1e-10, mags * 1e-10 * err_floor_phot, np.inf)
    obs['phot_mask'] = np.isfinite(np.squeeze(mags)) & (mags != mags_err) & (mags != -99.0) & (mags_err > 0)

    ### now spectra
    # We have a spectrum (should be units of maggies). wavelength in AA
    if np.isfinite(catalog[idx_cat]['f_F814W']) & (catalog[idx_cat]['f_F814W'] > 0.0):
        mag_norm = catalog[idx_cat]['f_F814W'] * 1e-10
    elif np.isfinite(catalog[idx_cat]['f_F775W']) & (catalog[idx_cat]['f_F775W'] > 0.0):
        mag_norm = catalog[idx_cat]['f_F775W'] * 1e-10
    elif np.isfinite(catalog[idx_cat]['f_I']) & (catalog[idx_cat]['f_I'] > 0.0):
        mag_norm = catalog[idx_cat]['f_I'] * 1e-10
    else:
        mag_norm = 20.0 * 1e-10
    idx_w = (catalog[idx_cat]['LAM'] > 8040) & (catalog[idx_cat]['LAM'] < 8240) & (catalog[idx_cat]['ERR'] < 6000.0)
    conversion_factor = mag_norm/np.median(catalog[idx_cat]['FLUX'][idx_w].data)
    obs['wavelength'] = catalog[idx_cat]['LAM'].data
    obs['spectrum'] = catalog[idx_cat]['FLUX'].data * conversion_factor
    obs['unc'] = np.clip(catalog[idx_cat]['ERR'].data * conversion_factor, catalog[idx_cat]['FLUX'].data * conversion_factor * err_floor_spec, np.inf)
    obs['mask'] = (obs['wavelength'] < 9150.0) & (catalog[idx_cat]['ERR'].data < 6000.0) & \
                  (obs['wavelength'] > (1.0 + catalog[idx_cat]['ZSPEC']) * 3525.0) & (obs['wavelength'] < (1.0 + catalog[idx_cat]['ZSPEC']) * 7500.0) & \
                   ~((obs['wavelength'] > 6860.0) & (obs['wavelength'] < 6920.0)) & \
                   ~((obs['wavelength'] > 7150.0) & (obs['wavelength'] < 7340.0)) & \
                   ~((obs['wavelength'] > 7575.0) & (obs['wavelength'] < 7725.0))

    # check S2N cut
    idx_w = (obs['wavelength'] > 7000.0) & (obs['wavelength'] < 9200.0)
    SN_calc = np.mean(catalog[idx_cat]['FLUX'].data[(obs['mask'] == 1) & idx_w]/catalog[idx_cat]['ERR'].data[(obs['mask'] == 1) & idx_w])/np.sqrt(np.mean(np.diff(obs['wavelength'])))
    if (SN_calc < S2N_cut):
        print 'S/N =', SN_calc
        sys.exit("Do not fit this galaxy bc SN!")
    if (catalog[idx_cat]['exclusion_flag'] == 1.0):
        sys.exit("Do not fit this galaxy bc exclusion flag!")

    # option for switching off spec and/or phot
    if switch_off_spec:
        obs['wavelength'] = None
        obs['spectrum'] = None
        obs['unc'] = None
        obs['mask'] = None
    if switch_off_phot:
        obs['phot_mask'] = (np.array(obs['wave_effective']) > 7640.0) & (np.array(obs['wave_effective']) < 7660.0)

    # add unessential bonus info.  This will be stored in output
    obs['cat_row'] = idx_cat
    obs['id_halo7d'] = catalog[idx_cat]['ID']
    obs['id_3dhst'] = catalog[idx_cat]['id_3dhst']
    obs['field'] = catalog[idx_cat]['FIELD']
    obs['RA'] = catalog[idx_cat]['RA']
    obs['DEC'] = catalog[idx_cat]['DEC']
    obs['redshift'] = catalog[idx_cat]['ZSPEC']
    obs['SN_calc'] = SN_calc

    # plot SED to ensure everything is on the same scale
    if False:
        import matplotlib.pyplot as plt
        smask = obs['mask']
        plt.plot(obs['wavelength'][smask], obs['spectrum'][smask], '-', lw=2, color='red')
        plt.plot(obs['wave_effective'], obs['maggies'], 'o')
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(obs['spectrum'].min()*0.5, obs['spectrum'].max()*2)
        plt.xlim(6000, 10000)
        plt.show()

    return obs


# --------------
# Model Definition
# --------------
def build_model(objid=1, non_param_sfh=False, dirichlet_sfh=False, add_duste=False, add_neb=False, add_agn=False, switch_off_mix=False, marginalize_neb=True,
                n_bins_sfh=8, use_eline_prior=False, add_jitter=False, fit_continuum=False, switch_off_phot=False, switch_off_spec=False, fixed_dust=False, **extras):
    """Construct a model.  This method defines a number of parameter
    specification dictionaries and uses them to initialize a
    `models.sedmodel.SedModel` object.
    :param object_redshift:
        If given, given the model redshift to this value.
    :param add_dust: (optional, default: False)
        Switch to add (fixed) parameters relevant for dust emission.
    :param add_neb: (optional, default: False)
        Switch to add (fixed) parameters relevant for nebular emission, and
        turn nebular emission on.
    """
    # read in data table
    obs = build_obs(objid=objid)

    # get SFH template
    if non_param_sfh and not dirichlet_sfh:
        model_params = TemplateLibrary["continuity_sfh"]
    elif dirichlet_sfh:
        model_params = TemplateLibrary["dirichlet_sfh"]
    else:
        model_params = TemplateLibrary["parametric_sfh"]

    # fit for redshift
    # use catalog value as center of the prior
    model_params["zred"]['isfree'] = True
    model_params["zred"]["init"] = obs['redshift']
    model_params["zred"]["prior"] = priors.TopHat(mini=obs['redshift']-0.005, maxi=obs['redshift']+0.005)

    # get SFH template
    if non_param_sfh:
        t_univ = cosmo.age(obs['redshift']).value
        tbinmax = 0.95 * t_univ * 1e9
        lim1, lim2, lim3, lim4 = 7.4772, 8.0, 8.5, 9.0
        agelims = [0, lim1, lim2, lim3] + np.log10(np.linspace(10**lim4, tbinmax, n_bins_sfh-4)).tolist() + [np.log10(t_univ*1e9)]
        if dirichlet_sfh:
            model_params = adjust_dirichlet_agebins(model_params, agelims=agelims)
            model_params["total_mass"]["prior"] = priors.LogUniform(mini=3e9, maxi=1e12)
        else:
            model_params = adjust_continuity_agebins(model_params, tuniv=t_univ, nbins=n_bins_sfh)
            agebins = np.array([agelims[:-1], agelims[1:]])
            model_params['agebins']['init'] = agebins.T
            model_params["logmass"]["prior"] = priors.TopHat(mini=9.5, maxi=12.0)
    else:
        model_params["tau"]["prior"] = priors.LogUniform(mini=1e-1, maxi=10)
        model_params["tage"]["prior"] = priors.TopHat(mini=1e-3, maxi=cosmo.age(obs['redshift']).value)
        model_params["mass"]["prior"] = priors.LogUniform(mini=3e9, maxi=1e12)

    # metallicity (no mass-metallicity prior yet!)
    if fixed_dust:
        model_params["logzsol"]["prior"] = priors.ClippedNormal(mini=-1.0, maxi=0.19, mean=0.0, sigma=0.15)
    else:
        model_params["logzsol"]["prior"] = priors.TopHat(mini=-1.0, maxi=0.19)

    # complexify the dust
    if fixed_dust:
        model_params['dust_type']['init'] = 2
        model_params["dust2"]["prior"] = priors.ClippedNormal(mini=0.0, maxi=4.0, mean=0.3, sigma=1)
        model_params['dust1'] = {"N": 1,
                                 "isfree": False,
                                 "init": 0.0, "units": "optical depth towards young stars",
                                 "prior": None}
    else:
        model_params['dust_type']['init'] = 4
        model_params["dust2"]["prior"] = priors.ClippedNormal(mini=0.0, maxi=4.0, mean=0.3, sigma=1)
        model_params["dust_index"] = {"N": 1,
                                      "isfree": True,
                                      "init": 0.0, "units": "power-law multiplication of Calzetti",
                                      "prior": priors.TopHat(mini=-1.0, maxi=0.4)}

        def to_dust1(dust1_fraction=None, dust1=None, dust2=None, **extras):
            return(dust1_fraction*dust2)

        model_params['dust1'] = {"N": 1,
                                 "isfree": False,
                                 'depends_on': to_dust1,
                                 "init": 0.0, "units": "optical depth towards young stars",
                                 "prior": None}
        model_params['dust1_fraction'] = {'N': 1,
                                          'isfree': True,
                                          'init': 1.0,
                                          'prior': priors.ClippedNormal(mini=0.0, maxi=2.0, mean=1.0, sigma=0.3)}

    # velocity dispersion
    model_params.update(TemplateLibrary['spectral_smoothing'])
    model_params["sigma_smooth"]["prior"] = priors.TopHat(mini=40.0, maxi=400.0)

    # Change the model parameter specifications based on some keyword arguments
    if add_duste:
        # Add dust emission (with fixed dust SED parameters)
        model_params.update(TemplateLibrary["dust_emission"])
        model_params['duste_gamma']['isfree'] = True
        model_params['duste_gamma']['init'] = 0.01
        model_params['duste_gamma']['prior'] = priors.LogUniform(mini=1e-4, maxi=0.1)
        model_params['duste_qpah']['isfree'] = True
        model_params['duste_qpah']['prior'] = priors.TopHat(mini=0.5, maxi=7.0)
        model_params['duste_umin']['isfree'] = True
        model_params['duste_umin']['init'] = 1.0
        model_params['duste_umin']['prior'] = priors.ClippedNormal(mini=0.1, maxi=15.0, mean=2.0, sigma=1.0)

    if add_agn:
        # Allow for the presence of an AGN in the mid-infrared
        model_params.update(TemplateLibrary["agn"])
        model_params['fagn']['isfree'] = True
        model_params['fagn']['prior'] = priors.LogUniform(mini=1e-5, maxi=3.0)
        model_params['agn_tau']['isfree'] = True
        model_params['agn_tau']['prior'] = priors.LogUniform(mini=5.0, maxi=150.)

    if add_neb:
        # Add nebular emission
        model_params.update(TemplateLibrary["nebular"])
        model_params['gas_logu']['isfree'] = True
        model_params['gas_logu']['init'] = -2.0
        model_params['gas_logz']['isfree'] = True
        _ = model_params["gas_logz"].pop("depends_on")
        model_params['nebemlineinspec'] = {'N': 1,
                                           'isfree': False,
                                           'init': False}

        if marginalize_neb:
            model_params.update(TemplateLibrary['nebular_marginalization'])
            #model_params.update(TemplateLibrary['fit_eline_redshift'])
            model_params['eline_prior_width']['init'] = 3.0
            model_params['use_eline_prior']['init'] = use_eline_prior

            # only marginalize over a few (strong) emission lines
            if True:
                #SPS_HOME = os.getenv('SPS_HOME')
                #emline_info = np.genfromtxt(SPS_HOME + '/data/emlines_info.dat', dtype=[('wave', 'f8'), ('name', 'S20')], delimiter=',')
                to_fit = ['[OII]3726', '[OII]3729', 'H 3798', 'H 3835', 'H 3889', 'H 3970', '[NeIII]3870', 'H delta 4102', 'H gamma 4340',
                          '[OIII]4364', 'H beta 4861', '[OIII]4960', '[OIII]5007', '[NII]6549', 'H alpha 6563', '[NII]6585']
                #idx = np.array([1 if name in to_fit else 0 for name in emline_info['name']], dtype=bool)
                model_params['lines_to_fit']['init'] = to_fit

            # model_params['use_eline_prior']['init'] = False
        else:
            model_params['nebemlineinspec']['init'] = True

    # This removes the continuum from the spectroscopy. Highly recommend
    # using when modeling both photometry & spectroscopy
    if fit_continuum:
        # order of polynomial that's fit to spectrum
        polyorder_estimate = int(np.clip(np.round((np.min([7500*(obs['redshift']+1), 9150.0])-np.max([3525.0*(obs['redshift']+1), 6000.0]))/(obs['redshift']+1)*100), 10, 30))
        model_params['polyorder'] = {'N': 1,
                                     'init': polyorder_estimate,
                                     'isfree': False}
        # fit for normalization of spectrum
        # model_params['spec_norm'] = {'N': 1,
        #                              'init': 0.8,
        #                              'isfree': True,
        #                              'prior': priors.Normal(sigma=0.2, mean=0.8),
        #                              'units': 'f_true/f_obs'}

    # This is a pixel outlier model. It helps to marginalize over
    # poorly modeled noise, such as residual sky lines or
    # even missing absorption lines
    if not switch_off_mix:
        model_params['f_outlier_spec'] = {"N": 1,
                                          "isfree": True,
                                          "init": 0.01,
                                          "prior": priors.TopHat(mini=1e-5, maxi=0.5)}
        model_params['nsigma_outlier_spec'] = {"N": 1,
                                               "isfree": False,
                                               "init": 50.0}

    # This is a multiplicative noise inflation term. It inflates the noise in
    # all spectroscopic pixels as necessary to get a good fit.
    if add_jitter:
        model_params['spec_jitter'] = {"N": 1,
                                       "isfree": True,
                                       "init": 1.0,
                                       "prior": priors.TopHat(mini=1.0, maxi=5.0)}

    # Now instantiate the model using this new dictionary of parameter specifications
    model = PolySpecModel(model_params)

    return model


# --------------
# SPS Object
# --------------


def set_halo7d_lsf(ssp, objid=1, data_table=path_wdir + 'data/halo7d_with_phot.fits', zred=0.0, **extras):
    """Method to make the SSPs have the same (rest-frame) resolution as the
    Halo7d spectrographs.  This is only correct if the redshift is fixed, but is
    a decent approximation as long as redshift does not change much.
    """
    # load spectrum
    from astropy.table import Table
    catalog = Table.read(data_table)
    # get LSF
    wave, delta_v = get_lsf(catalog[objid-1]['LAM'].data, catalog[objid-1]['FIELD'], zred=zred, **extras)
    assert ssp.libraries[1] == 'miles', "Please change FSPS to the MILES libraries."
    ssp.params['smooth_lsf'] = True
    ssp.set_lsf(wave, delta_v)


def get_lsf(wave_obs, field, miles_fwhm_aa=2.54, zred=0.0, **extras):
    """This method takes an Halo7d spec and returns the gaussian kernel required
    to convolve the restframe MILES spectra to have the observed frame
    instrumental dispersion.
    :returns wave_rest:
        The restframe wavelength. ndarray of shape (nwave,)
    :returns dsv:
        The quadrature difference between the instrumental dispersion and the
        MILES dispersion, in km/s, as a function of wavelength.  Same shape as
        `wave_rest`
    """
    lightspeed = 2.998e5  # km/s
    # load LSF
    if (field == 'COSMOS'):
        deltaLambPolyVals = np.load(filter_lsf + 'COSMOS.deltaLambdaFit.npy')
    elif (field == 'EGS'):
        deltaLambPolyVals = np.load(filter_lsf + 'EGS.deltaLambdaFit.npy')
    elif (field == 'EGS+'):
        deltaLambPolyVals = np.load(filter_lsf + 'EGS.deltaLambdaFit.npy')
    elif (field == 'GOODS-N'):
        deltaLambPolyVals = np.load(filter_lsf + 'GOODSN.deltaLambdaFit.npy')
    # keep only the parameter values (drop the uncertainties)
    deltaLambPolyVals = deltaLambPolyVals[:, 0]
    deltaLambFunc = np.poly1d(deltaLambPolyVals)  # this is the LSF
    lsf = deltaLambFunc(wave_obs)  # returns the gaussian width at the values of wavelength
    # This is the instrumental velocity resolution in the observed frame
    #sigma_v = np.log(10) * lightspeed * 1e-4 * lsf
    #sigma_v = (1 + zred) * lightspeed * lsf / wave_obs
    sigma_v = lightspeed * lsf / wave_obs
    # filter out some places where Halo7d reports zero dispersion
    good = sigma_v > 0
    wave_obs, sigma_v = wave_obs[good], sigma_v[good]
    # Get the miles velocity resolution function at the corresponding
    # *rest-frame* wavelength
    wave_rest = wave_obs / (1 + zred)
    sigma_v_miles = lightspeed * miles_fwhm_aa / 2.355 / wave_rest
    # Get the quadrature difference
    # (Zero and negative values are skipped by FSPS)
    dsv = np.sqrt(np.clip(sigma_v**2 - sigma_v_miles**2, 0, np.inf))
    # Restrict to regions where MILES is used
    good = (wave_rest > 3525.0) & (wave_rest < 7500)
    # return the broadening of the rest-frame library spectra required to match
    # the obserrved frame instrumental lsf
    return wave_rest[good], dsv[good]


def build_sps(zcontinuous=1, non_param_sfh=False, add_lsf=False, compute_vega_mags=False, zred=0.0, **extras):
    if non_param_sfh:
        from prospect.sources import FastStepBasis
        sps = FastStepBasis(zcontinuous=zcontinuous,
                            compute_vega_mags=compute_vega_mags,
                            reserved_params=['tage', 'sigma_smooth', 'zred'])
    else:
        from prospect.sources import CSPSpecBasis
        sps = CSPSpecBasis(zcontinuous=zcontinuous,
                           compute_vega_mags=compute_vega_mags,
                           reserved_params=['sigma_smooth', 'zred'])

    if add_lsf:
        set_halo7d_lsf(sps.ssp, zred=zred, **extras)

    return sps


# -----------------
# Noise Model
# ------------------

def build_noise(add_jitter=False, **extras):
    if add_jitter:
        jitter = Uncorrelated(parnames=['spec_jitter'])
        spec_noise = NoiseModel(kernels=[jitter], metric_name='unc', weight_by=['unc'])
        return spec_noise, None
    else:
        return None, None


# -----------
# Everything
# ------------
def build_all(**kwargs):
    return (build_obs(**kwargs), build_model(**kwargs),
            build_sps(**kwargs), build_noise(**kwargs))


if __name__ == '__main__':

    # - Parser with default arguments -
    parser = prospect_args.get_parser()

    # - Add custom arguments -
    parser.add_argument('--data_table', type=str, default=path_wdir+"data/halo7d_with_phot.fits",
                        help="Names of table from which to get photometry.")
    parser.add_argument('--objid', type=int, default=0,
                        help="Zero-index row number in the table to fit.")
    parser.add_argument('--S2N_cut', type=np.float, default=5.0,
                        help="Signal-to-noise cut applied to sample.")
    parser.add_argument('--non_param_sfh', action="store_true",
                        help="If set, fit non-parametric star-formation history model.")
    parser.add_argument('--dirichlet_sfh', action="store_true",
                        help="If set, fit non-parametric star-formation history model with dirichlet prior.")
    parser.add_argument('--n_bins_sfh', type=int, default=8,
                        help="Number of bins for SFH (non parametric).")
    parser.add_argument('--add_lsf', action="store_true",
                        help="If set, add realistic instrumental dispersion.")
    parser.add_argument('--add_neb', action="store_true",
                        help="If set, add nebular emission in the model (and mock).")
    parser.add_argument('--fit_continuum', action="store_true",
                        help="If set, fit continuum.")
    parser.add_argument('--add_duste', action="store_true",
                        help="If set, add dust emission to the model.")
    parser.add_argument('--add_agn', action="store_true",
                        help="If set, add agn emission to the model.")
    parser.add_argument('--remove_mips24', action="store_true",
                        help="If set, removes MIPS 24um flux.")
    parser.add_argument('--err_floor_phot', type=np.float, default=0.001,
                        help="Error floor for photometry.")
    parser.add_argument('--err_floor_spec', type=np.float, default=0.001,
                        help="Error floor for spectroscopy.")
    parser.add_argument('--use_eline_prior', action="store_true",
                        help="If set, use EL prior from cloudy.")
    parser.add_argument('--add_jitter', action="store_true",
                        help="If set, jitter noise.")
    parser.add_argument('--switch_off_spec', action="store_true",
                        help="If set, remove spectrum from obs.")
    parser.add_argument('--switch_off_phot', action="store_true",
                        help="If set, remove photometry from obs.")
    parser.add_argument('--switch_off_mix', action="store_true",
                        help="If set, switch off mixture model.")
    parser.add_argument('--fixed_dust', action="store_true",
                        help="If set, fix dust to Calzetti and add tight Z prior.")

    args = parser.parse_args()
    run_params = vars(args)

    # add in dynesty settings
    run_params['dynesty'] = True
    run_params['nested_weight_kwargs'] = {'pfrac': 1.0}
    run_params['nested_nlive_batch'] = 200
    run_params['nested_walks'] = 45  # sampling gets very inefficient w/ high S/N spectra
    run_params['nested_nlive_init'] = 250
    run_params['nested_dlogz_init'] = 0.02
    run_params['nested_maxcall'] = 10000000
    run_params['nested_maxcall_init'] = 10000000
    run_params['nested_method'] = 'rwalk'
    run_params['nested_maxbatch'] = None
    run_params['nested_save_bounds'] = False
    run_params['nested_posterior_thresh'] = 0.05
    run_params['nested_first_update'] = {'min_ncall': 20000, 'min_eff': 7.5}

    obs, model, sps, noise = build_all(**run_params)
    run_params["param_file"] = __file__

    if args.debug:
        sys.exit()

    hfile = path_wdir + "results/{0}_idx_{1}_mcmc.h5".format(args.outfile, int(args.objid)-1)
    output = fit_model(obs, model, sps, noise, lnprobfn=lnprobfn, **run_params)

    print('writing hdf5 file now...')

    writer.write_hdf5(hfile, run_params, model, obs,
                      output["sampling"][0], output["optimization"][0],
                      tsample=output["sampling"][1],
                      toptimize=output["optimization"][1])

    try:
        hfile.close()
    except(AttributeError):
        pass
