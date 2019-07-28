# import modules

import sys
import os

import numpy as np
from sedpy.observate import load_filters

from prospect import prospect_args
from prospect.fitting import fit_model, lnprobfn
from prospect.io import write_results as writer
from prospect.likelihood import NoiseModel
from prospect.likelihood.kernels import Uncorrelated


# define paths

path_wdir = os.environ['WDIR_halo7d']
filter_folder = path_wdir + '/data/filters/'
filter_lsf = path_wdir + '/data/galaxy_LSF_output/'


# emission line function

def get_boxed_mask(wavelength, mask):
    mask_val = ~mask
    mask_vec = []
    kk = 0
    for ii in range(len(wavelength)-1):
        if (ii <= kk+1):
            continue
        if mask_val[ii]:
            for kk in range(ii, len(wavelength)-1):
                if ~mask_val[kk]:
                    break
            mask_vec.append(ii)
            mask_vec.append(kk)
    mask_plot = np.array(mask_vec).reshape(len(mask_vec)/2, 2)
    mask_bool = np.diff(mask_plot, axis=1) > 20.0/np.diff(wavelength)[0]
    mask_blocks = np.array(len(mask_val)*[True])
    for ii_c in range(len(mask_bool)-1):
        if mask_bool[ii_c]:
            mask_blocks[mask_plot[ii_c][0]:mask_plot[ii_c][1]] = (mask_plot[ii_c][1]-mask_plot[ii_c][0])*[False]
    return(mask_blocks)


doublet_info = {}
doublet_info['[OII]'] = {'w1': 3727.09, 'w2': 3729.88, 'ratio': 1.0}
doublet_info['[OIII]'] = {'w1': 4960.30, 'w2': 5008.24, 'ratio': 0.33}
doublet_info['[NII]'] = {'w1': 6549.86, 'w2': 6585.27, 'ratio': 0.33}


def get_lines_to_fit(wavelength, mask, redshift):
    line_fit_name = np.array(['Ha', 'Hb', 'Hg', 'Hd', 'He', 'H8', 'H9', 'H10', 'HeI', '[NII]', '[NII]', '[OII]', '[OII]', '[OIII]', '[OIII]', '[NeIII]'])
    line_fit_rest_wave = np.array([6564.61, 4862.69, 4341.69, 4102.92, 3971.19, 3890.15, 3836.48, 3798.98, 3889.0, 6585.27, 6549.86, 3727.09, 3729.88, 5008.24, 4960.30, 3869.81])
    # get lines in wavelength range
    idx_line1 = (np.min(wavelength[mask]) < (redshift+1.0)*line_fit_rest_wave) & (np.max(wavelength[mask]) > (redshift+1.0)*line_fit_rest_wave)
    # mask lines in masked region
    mask_new = get_boxed_mask(wavelength, mask)
    idx_line2 = []
    for ii_line in range(len(line_fit_rest_wave)):
        diff = np.abs(wavelength-(redshift+1.0)*line_fit_rest_wave[ii_line])
        idx_close = diff.argmin()
        if (diff[idx_close] <= np.diff(wavelength)[0]):
            idx_line2 = np.append(idx_line2, mask_new[idx_close])
        else:
            idx_line2 = np.append(idx_line2, False)
    idx_line = idx_line1 & (idx_line2 == 1.0)
    return(line_fit_name[idx_line], line_fit_rest_wave[idx_line])


def load_zp_offsets(field):
    # load skelton zp offsets
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


def build_obs(objid=1, data_table=path_wdir + 'data/halo7d_with_phot.fits', err_floor_phot=0.001, err_floor_spec=0.001, S2N_cut=1.0, remove_mips24=False, switch_off_phot=False, switch_off_spec=False, **kwargs):
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
    # Writes your code here to read data.  Can use FITS, h5py, astropy.table,
    # sqlite, whatever.
    # e.g.:
    # import astropy.io.fits as pyfits
    # catalog = pyfits.getdata(phottable)
    # Here we will read in an ascii catalog of magnitudes as a numpy structured
    # array
    from astropy.table import Table
    catalog = Table.read(data_table)
    idx_cat = objid-1
    # Here we are dynamically choosing which filters to use based on the object
    # and a flag in the catalog.  Feel free to make this logic more (or less)
    # complicated.
    filternames = []
    mags = []
    mags_err = []
    # field name
    field_name_filters = catalog[idx_cat]['FIELD'].lower().replace("-", "")
    if ((field_name_filters == 'egs') | (field_name_filters == 'egs+')):
        field_name_filters = 'aegis'
    for ii in catalog.keys():
        if ('f_' in ii[:2]):
            filternames.append(ii[2:].lower() + '_' + field_name_filters)
            mags.append(catalog[idx_cat][ii])
            mags_err.append(catalog[idx_cat][ii.replace('f_', 'e_')])
    filternames = np.array(filternames)
    mags = np.array(mags)
    mags_err = np.array(mags_err)
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
    band_names = np.array([x['Band'].lower()+'_'+x['Field'].lower() for x in zp_offsets])
    for ii, f in enumerate(filternames):
        match = (band_names == f)
        if match.sum():
            mags_err[ii] = ((mags_err[ii]**2) + (mags[ii]*(1-zp_offsets[match]['Flux-Correction'][0]))**2)**0.5
    # Build output dictionary.
    obs = {}
    # This is a list of sedpy filter objects.    See the
    # sedpy.observate.load_filters command for more details on its syntax.
    obs['filters'] = load_filters(filternames, directory=filter_folder)
    obs['wave_effective'] = [f.wave_effective for f in obs['filters']]
    # This is a list of maggies, converted from mags.  It should have the same
    # order as `filters` above.
    obs['maggies'] = mags * 1e-10
    # You should use real flux uncertainties (incl. error floor)
    obs['maggies_unc'] = np.clip(mags_err * 1e-10, mags * 1e-10 * err_floor_phot, np.inf)
    idx_longw = (np.array(obs['wave_effective']) > 2e4)
    obs['maggies_unc'][idx_longw] = np.clip(mags_err[idx_longw] * 1e-10, mags[idx_longw] * 1e-10 * 2.0 * err_floor_phot, np.inf)
    # Here we mask out any NaNs or infs
    obs['phot_mask'] = np.isfinite(np.squeeze(mags)) & (mags != mags_err) & (mags != -99.0) & (mags_err > 0)
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
    # Add unessential bonus info.  This will be stored in output
    if switch_off_spec:
        obs['wavelength'] = None
        obs['spectrum'] = None
        obs['unc'] = None
        obs['mask'] = None
    if switch_off_phot:
        obs['phot_mask'] = (np.array(obs['wave_effective']) == 7646.0363672352305)
    obs['cat_row'] = idx_cat
    obs['id_halo7d'] = catalog[idx_cat]['ID']
    obs['id_3dhst'] = catalog[idx_cat]['id_3dhst']
    obs['field'] = catalog[idx_cat]['FIELD']
    obs['RA'] = catalog[idx_cat]['RA']
    obs['DEC'] = catalog[idx_cat]['DEC']
    obs['redshift'] = catalog[idx_cat]['ZSPEC']
    obs['SN_calc'] = SN_calc
    # get EL that will be fit
    try:
        line_names, line_wave = get_lines_to_fit(obs['wavelength'], obs['mask'], catalog[idx_cat]['ZSPEC'])
        obs['EL_names'] = line_names
        obs['EL_wave'] = line_wave
    except (ValueError, TypeError):
        obs['EL_names'] = []
        obs['EL_wave'] = []
    return obs


# --------------
# Model Definition
# --------------

from prospect.models.sedmodel import PolySedModel, gauss


def gauss_doublet(x, w1, a1, sigma, w2, ratio):
    return(gauss(x, w1, a1, sigma) + gauss(x, w2, ratio*a1, sigma))


class ElineMargSEDModel(PolySedModel):

    def mean_model(self, theta, obs, sps=None, EL_info=False, **extras):
        s, p, x = self.sed(theta, obs, sps=sps, **extras)
        self._speccal = self.spec_calibration(obs=obs, **extras)
        s *= self._speccal
        if obs['spectrum'] is not None:
            self._el = self.get_el(obs, s, EL_info=EL_info)
            s += self._el
        if obs.get('logify_spectrum', False):
            return np.log(s), p, x
        else:
            return s, p, x

    def mean_model_withoutEL(self, theta, obs, sps=None, EL_info=True, **extras):
        s, p, x = self.sed(theta, obs, sps=sps, **extras)
        self._speccal = self.spec_calibration(obs=obs, **extras)
        s *= self._speccal
        self._el, amplitdes_mle = self.get_el(obs, s, EL_info=EL_info)
        if obs.get('logify_spectrum', False):
            return np.log(s), p, x, self._el, amplitdes_mle, self._speccal
        else:
            return s, p, x, self._el, amplitdes_mle, self._speccal

    def get_el(self, obs, spec, EL_info):
        # set up residual spectrum to be fit
        mask = obs.get('mask', slice(None))
        residual_spec = obs['spectrum'][mask]-spec[mask]
        residual_spec *= 1e10
        error_spec = 1e10*obs["unc"][mask]  # that is not right, needs to be fixed
        eline_wavelength = np.log(self.params['eline_wavelength'] * (1.0 + self.params['zred'] + self.params["dzred_gas"]))
        eline_sigma = self.params['sigma_gas']/2.998e5
        eline_spec = np.zeros(len(obs['spectrum']))
        # iterate over all lines
        amplitdes_mle = []
        doublet_done = []
        for ii_line, ii_name in enumerate(self.params['eline_names']):
            if (ii_name in doublet_info.keys()):
                if (ii_name not in doublet_done):
                    doublet_done.append(ii_name)
                    amplitdes_mle.append(np.sum(residual_spec*gauss_doublet(np.log(obs["wavelength"]), np.log(doublet_info[ii_name]['w1'] * (1.0 + self.params['zred'] + self.params["dzred_gas"])), 1.0, eline_sigma, np.log(doublet_info[ii_name]['w2'] * (1.0 + self.params['zred'] + self.params["dzred_gas"])), doublet_info[ii_name]['ratio'])[mask]/error_spec**2) / np.sum(gauss_doublet(np.log(obs["wavelength"]), np.log(doublet_info[ii_name]['w1'] * (1.0 + self.params['zred'] + self.params["dzred_gas"])), 1.0, eline_sigma, np.log(doublet_info[ii_name]['w2'] * (1.0 + self.params['zred'] + self.params["dzred_gas"])), doublet_info[ii_name]['ratio'])[mask]**2/error_spec**2))
                    eline_spec += gauss_doublet(np.log(obs["wavelength"]), np.log(doublet_info[ii_name]['w1'] * (1.0 + self.params['zred'] + self.params["dzred_gas"])), max([0.0, amplitdes_mle[-1]]), eline_sigma, np.log(doublet_info[ii_name]['w2'] * (1.0 + self.params['zred'] + self.params["dzred_gas"])), doublet_info[ii_name]['ratio'])
                else:
                    continue
            else:
                amplitdes_mle.append(np.sum(residual_spec*gauss(np.log(obs["wavelength"]), eline_wavelength[ii_line], 1.0, eline_sigma)[mask]/error_spec**2) / np.sum(gauss(np.log(obs["wavelength"]), eline_wavelength[ii_line], 1.0, eline_sigma)[mask]**2/error_spec**2))
                eline_spec += gauss(np.log(obs["wavelength"]), eline_wavelength[ii_line], max([0.0, amplitdes_mle[-1]]), eline_sigma)
        # return best-fit EL spectrum
        amplitdes_mle = np.array(amplitdes_mle)
        amplitdes_mle[amplitdes_mle < 0.0] = 0.0
        if EL_info:
            return(eline_spec*1e-10, amplitdes_mle*1e-10)
        else:
            return(eline_spec*1e-10)


def build_model(objid=1, data_table=path_wdir + 'data/halo7d_with_phot.fits', non_param_sfh=False, add_duste=False, add_neb=False, add_agn=False,
                add_jitter=False, fit_continuum=False, switch_off_phot=False, switch_off_spec=False, **extras):
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
    from prospect.models.templates import TemplateLibrary, adjust_continuity_agebins
    from prospect.models import priors
    #from prospect.models import transforms
    from astropy.table import Table
    from astropy.cosmology import Planck15 as cosmo
    # read in data table
    catalog = Table.read(data_table)
    idx_cat = objid-1

    # get SFH template
    if non_param_sfh:
        t_univ = cosmo.age(catalog[idx_cat]['ZSPEC']).value
        model_params = TemplateLibrary["continuity_sfh"]
        model_params = adjust_continuity_agebins(model_params, tuniv=t_univ, nbins=12)
    else:
        model_params = TemplateLibrary["parametric_sfh"]
        model_params["tau"]["prior"] = priors.LogUniform(mini=1e-1, maxi=10)
        model_params["tage"]["prior"] = priors.TopHat(mini=0.0, maxi=cosmo.age(catalog[idx_cat]['ZSPEC']).value)

    # adjust priors
    model_params["dust2"]["prior"] = priors.TopHat(mini=0.0, maxi=3.0)
    model_params["mass"]["prior"] = priors.LogUniform(mini=1e10, maxi=1e12)
    model_params["logzsol"]["prior"] = priors.TopHat(mini=-1.0, maxi=0.3)
    model_params["dust_type"]["init"] = 4
    model_params["dust_index"] = {"N": 1, "isfree": True,
                                  "init": 0.0, "units": "power-law multiplication of Calzetti",
                                  "prior": priors.TopHat(mini=-2.0, maxi=0.5)}

    # fit for redshift
    model_params["zred"]['isfree'] = True
    model_params["zred"]["init"] = catalog[idx_cat]['ZSPEC']
    model_params["zred"]["prior"] = priors.TopHat(mini=catalog[idx_cat]['ZSPEC']-0.01, maxi=catalog[idx_cat]['ZSPEC']+0.01)
    model_params["dzred_gas"] = {"N": 1, "isfree": True,
                                 "init": 0.0, "units": "redshift of gas relative to stars",
                                 "prior": priors.ClippedNormal(mean=0.0, sigma=0.0005, mini=-0.002, maxi=0.002)}

    # velocity dispersion
    model_params.update(TemplateLibrary['spectral_smoothing'])
    model_params["sigma_smooth"]["prior"] = priors.TopHat(mini=50.0, maxi=350.0)
    model_params["sigma_smooth"]["init"] = 200.0
    model_params["sigma_gas"] = {"N": 1, "isfree": True,
                                 "init": 100.0, "units": "velocity dispersion of gas",
                                 "prior": priors.TopHat(mini=10.0, maxi=300.0)}

    # modeling noise
    model_params['f_outlier_spec'] = {"N": 1,
                                      "isfree": True,
                                      "init": 0.01,
                                      "prior": priors.TopHat(mini=1e-5, maxi=0.3)}

    model_params['nsigma_outlier_spec'] = {"N": 1,
                                           "isfree": False,
                                           "init": 5.0}

    # noise jitter
    if add_jitter:
        model_params['spec_jitter'] = {"N": 1,
                                       "isfree": True,
                                       "init": 1.0,
                                       "prior": priors.TopHat(mini=1.0, maxi=4.0)}

    # Change the model parameter specifications based on some keyword arguments
    if add_duste:
        # Add dust emission (with fixed dust SED parameters)
        model_params.update(TemplateLibrary["dust_emission"])
        model_params['duste_gamma']['isfree'] = False
        #model_params['duste_gamma']['init'] = 1e-2
        #model_params['duste_gamma']['prior'] = priors.LogUniform(mini=1e-3, maxi=1e-1)
        model_params['duste_qpah']['isfree'] = True
        model_params['duste_qpah']['prior'] = priors.TopHat(mini=0.5, maxi=7.0)
        model_params['duste_umin']['isfree'] = False
        #model_params['duste_umin']['prior'] = priors.ClippedNormal(mean=1.0, sigma=0.5, mini=0.1, maxi=25)

    if add_agn:
        # Add dust emission (with fixed dust SED parameters)
        model_params.update(TemplateLibrary["agn"])
        model_params['fagn']['isfree'] = True
        model_params['fagn']['prior'] = priors.LogUniform(mini=1e-5, maxi=3.0)
        model_params['agn_tau']['isfree'] = True
        model_params['agn_tau']['prior'] = priors.LogUniform(mini=5.0, maxi=150.)

    if add_neb:
        # Add nebular emission (with fixed parameters)
        model_params.update(TemplateLibrary["nebular"])
        model_params['gas_logu']['isfree'] = True
        model_params['gas_logz']['isfree'] = True
        _ = model_params["gas_logz"].pop("depends_on")
        #model_params["gas_logz"]["depends_on"] = transforms.stellar_logzsol

    if fit_continuum:
        # order of polynomial that's fit to spectrum
        model_params['polyorder'] = {'N': 1,
                                     'init': 10,
                                     'isfree': False}
        # fit for normalization of spectrum
        model_params['spec_norm'] = {'N': 1,
                                     'init': 0.8,
                                     'isfree': True,
                                     'prior': priors.Normal(sigma=0.2, mean=0.8),
                                     'units': 'f_true/f_obs'}

    # list of all emission lines

    if switch_off_spec:
        model_params['eline_wavelength'] = {'N': 0,
                                            'init': np.array([]),
                                            'isfree': False}
        model_params['eline_names'] = {'N': 0,
                                       'init': np.array([]),
                                       'isfree': False}

    else:
        wave = catalog[idx_cat]['LAM'].data
        mask = (wave < 9150.0) & (catalog[idx_cat]['ERR'].data < 6000.0) & \
               (wave > (1.0 + catalog[idx_cat]['ZSPEC']) * 3525.0) & (wave < (1.0 + catalog[idx_cat]['ZSPEC']) * 7500.0) & \
              ~((wave > 6860.0) & (wave < 6920.0)) & \
              ~((wave > 7150.0) & (wave < 7340.0)) & \
              ~((wave > 7575.0) & (wave < 7725.0))
        line_names, rest_waves = get_lines_to_fit(wave, mask, catalog[idx_cat]['ZSPEC'])

        # choose EL that are in wavelength range => add to obs
        model_params['eline_wavelength'] = {'N': len(rest_waves),
                                            'init': np.array(rest_waves),
                                            'isfree': False}
        model_params['eline_names'] = {'N': len(line_names),
                                       'init': np.array(line_names),
                                       'isfree': False}

    # Now instantiate the model using this new dictionary of parameter specifications
    #model = sedmodel.PolySedModel(model_params)
    model = ElineMargSEDModel(model_params)

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
    #dsv = np.sqrt(np.clip(sigma_v**2 - sigma_v_miles**2, 0, np.inf))
    dsv = sigma_v**2 - sigma_v_miles**2
    # Restrict to regions where MILES is used
    good = (wave_rest > 3525.0) & (wave_rest < 7500)
    # return the broadening of the rest-frame library spectra required to match
    # the obserrved frame instrumental lsf
    return wave_rest[good], dsv[good]


def build_sps(zcontinuous=1, non_param_sfh=False, add_lsf=False, compute_vega_mags=False, **extras):
    if non_param_sfh:
        from prospect.sources import FastStepBasis
        sps = FastStepBasis(zcontinuous=zcontinuous,
                            compute_vega_mags=compute_vega_mags)
    else:
        from prospect.sources import CSPSpecBasis
        sps = CSPSpecBasis(zcontinuous=zcontinuous,
                           compute_vega_mags=compute_vega_mags)

    if add_lsf:
        set_halo7d_lsf(sps.ssp, **extras)

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
    parser.add_argument('--non_param_sfh', action="store_true",
                        help="If set, fit non-parametric star-formation history model.")
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
    parser.add_argument('--data_table', type=str, default=path_wdir+"data/halo7d_with_phot.fits",
                        help="Names of table from which to get photometry.")
    parser.add_argument('--objid', type=int, default=0,
                        help="Zero-index row number in the table to fit.")
    parser.add_argument('--err_floor_phot', type=np.float, default=0.001,
                        help="Error floor for photometry.")
    parser.add_argument('--err_floor_spec', type=np.float, default=0.001,
                        help="Error floor for spectroscopy.")
    parser.add_argument('--S2N_cut', type=np.float, default=5.0,
                        help="Signal-to-noise cut applied to sample.")
    parser.add_argument('--add_jitter', action="store_true",
                        help="If set, jitter noise.")
    parser.add_argument('--switch_off_spec', action="store_true",
                        help="If set, remove spectrum from obs.")
    parser.add_argument('--switch_off_phot', action="store_true",
                        help="If set, remove photometry from obs.")

    args = parser.parse_args()
    run_params = vars(args)
    print run_params

    obs, model, sps, noise = build_all(**run_params)
    run_params["param_file"] = __file__

    if args.debug:
        sys.exit()

    #hfile = setup_h5(model=model, obs=obs, **run_params)
    hfile = path_wdir + "results/{0}_idx_{1}_mcmc.h5".format(args.outfile, int(args.objid)-1)
    output = fit_model(obs, model, sps, noise, lnprobfn=lnprobfn, **run_params)

    writer.write_hdf5(hfile, run_params, model, obs,
                      output["sampling"][0], output["optimization"][0],
                      tsample=output["sampling"][1],
                      toptimize=output["optimization"][1])

    try:
        hfile.close()
    except(AttributeError):
        pass
