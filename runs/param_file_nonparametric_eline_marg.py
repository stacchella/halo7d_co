# import modules

import sys
import os

import numpy as np
from sedpy.observate import load_filters

from prospect import prospect_args
from prospect.fitting import fit_model
from prospect.io import write_results as writer

from astropy.io import fits


# define paths

path_wdir = os.environ['WDIR_halo7d']
filter_folder = path_wdir + '/data/filters/'


# emission line function

def get_lines_to_fit(wave_min, wave_max, redshift):
    line_fit_name = np.array(['Ha', 'Hb', 'Hg', 'Hd', 'He', 'H8', 'H9', 'H10', 'HeI', '[NII]', '[NII]', '[OII]', '[OII]', '[OIII]', '[OIII]', '[NeIII]'])
    line_fit_rest_wave = np.array([6564.61, 4862.69, 4341.69, 4102.92, 3971.19, 3890.15, 3836.48, 3798.98, 3889.0, 6585.27, 6549.86, 3727.09, 3729.88, 5008.24, 4960.30, 3869.81])
    idx_line = (wave_min < (redshift+1.0)*line_fit_rest_wave) & (wave_max > (redshift+1.0)*line_fit_rest_wave)
    return(line_fit_name[idx_line], line_fit_rest_wave[idx_line])


def build_obs(objid=1, data_table=path_wdir + 'data/halo7d_with_phot.fits', err_floor_phot=0.05, err_floor_spec=0.1, S2N_cut=1.0, remove_mips24=False, switch_off_phot=False, switch_off_spec=False, **kwargs):
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
    choice_finite = np.isfinite(np.squeeze(mags)) & (mags != -99.0) & (mags_err > 0.0)
    filternames = filternames[choice_finite]
    mags = mags[choice_finite]
    mags_err = mags_err[choice_finite]
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
    obs['mask'] = (catalog[idx_cat]['ERR'].data < 6000.0) & (catalog[idx_cat]['LAM'].data > (1.0 + catalog[idx_cat]['ZSPEC']) * 3550)
    # check S2N cut
    idx_w = (obs['wavelength'] > 7000.0) & (obs['wavelength'] < 9500.0)
    SN_calc = np.mean(obs['spectrum'][(obs['mask'] == 1) & idx_w]/obs['unc'][(obs['mask'] == 1) & idx_w])/np.sqrt(np.mean(np.diff(obs['wavelength'])))
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
        line_names, line_wave = get_lines_to_fit(np.min(obs['wavelength'][obs['mask']]), np.max(obs['wavelength'][obs['mask']]), catalog[idx_cat]['ZSPEC'])
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
from scipy import optimize


class ElineMargSEDModel(PolySedModel):

    def mean_model(self, theta, obs, sps=None, EL_info=False, **extras):
        s, p, x = self.sed(theta, obs, sps=sps, **extras)
        self._speccal = self.spec_calibration(obs=obs, **extras)
        s *= self._speccal
        if obs['spectrum'] is not None:
            self._el = self.get_el(obs, EL_info=EL_info)
            s += self._el
        if obs.get('logify_spectrum', False):
            return np.log(s), p, x
        else:
            return s, p, x

    def mean_model_withoutEL(self, theta, obs, sps=None, EL_info=True, **extras):
        s, p, x = self.sed(theta, obs, sps=sps, **extras)
        self._speccal = self.spec_calibration(obs=obs, **extras)
        s *= self._speccal
        self._el, popt_EL, pcov_EL = self.get_el(obs, EL_info=EL_info)
        if obs.get('logify_spectrum', False):
            return np.log(s), p, x, popt_EL, pcov_EL
        else:
            return s, p, x, self._el, popt_EL, pcov_EL

    def get_el(self, obs, EL_info):
        mask = obs.get('mask', slice(None))
        residual_spec = obs['spectrum']-self._spec
        eline_wavelength = np.log(self.params['eline_wavelength'] * (1.0 + self.params['zred']))

        def gaussian_wrap(x, *vec):
            return(gauss(x, eline_wavelength, vec[1:], vec[0]))

        popt_gauss, pcov_gauss = optimize.curve_fit(gaussian_wrap, np.log(obs["wavelength"][mask]), residual_spec[mask], sigma=obs["unc"][mask], p0=np.append(100.0/2.998e5, np.median(np.abs(residual_spec[mask]))*np.ones(len(self.params['eline_wavelength']))), ftol=1e-2, xtol=1e-2, maxfev=int(1e5))

        if EL_info:
            return(gaussian_wrap(np.log(obs["wavelength"]), *popt_gauss), popt_gauss, pcov_gauss)
        else:
            return(gaussian_wrap(np.log(obs["wavelength"]), *popt_gauss))


def build_model(objid=1, data_table=path_wdir + 'data/halo7d_with_phot.fits', add_duste=False, add_neb=False, add_agn=False, fit_continuum=False, switch_off_phot=False, switch_off_spec=False, **extras):
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
    t_univ = cosmo.age(catalog[idx_cat]['ZSPEC']).value
    # --- Get a basic delay-tau SFH parameter set. ---
    # This has 5 free parameters:
    #   "mass", "logzsol", "dust2", "tage", "tau"
    # And two fixed parameters
    #   "zred"=0.1, "sfh"=4
    # See the python-FSPS documentation for details about most of these
    # parameters.  Also, look at `TemplateLibrary.describe("parametric_sfh")` to
    # view the parameters, their initial values, and the priors in detail.
    model_params = TemplateLibrary["continuity_sfh"]

    # adjust priors
    model_params["dust2"]["prior"] = priors.TopHat(mini=0.0, maxi=3.0)
    model_params = adjust_continuity_agebins(model_params, tuniv=t_univ, nbins=7)
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
    #model_params["zred_gas"] = {"N": 1, "isfree": True,
    #                            "init": catalog[idx_cat]['ZSPEC'], "units": "redshift of gas",
    #                            "prior": priors.TopHat(mini=catalog[idx_cat]['ZSPEC']-0.005, maxi=catalog[idx_cat]['ZSPEC']+0.005)}

    # velocity dispersion
    model_params.update(TemplateLibrary['spectral_smoothing'])
    model_params["sigma_smooth"]["prior"] = priors.TopHat(mini=50.0, maxi=350.0)
    model_params["sigma_smooth"]["init"] = 200.0

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
        model_params['fagn']['prior'] = priors.LogNormal(mode=np.log(1e-4), sigma=0.3)
	#model_params['fagn']['prior'] = priors.LogUniform(mini=1e-5, maxi=3.0)
        model_params['agn_tau']['isfree'] = False
        #model_params['agn_tau']['prior'] = priors.LogUniform(mini=5.0, maxi=150.)

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
                                     'init': 1.0,
                                     'isfree': True,
                                     'prior': priors.Normal(sigma=0.2, mean=1.0),
                                     'units': 'f_true/f_obs'}

    # list of all emission lines

    if switch_off_spec:
        model_params['eline_wavelength'] = {'N': 0,
                                            'init': np.array([]),
                                            'isfree': False}

    else:
        mask = (catalog[idx_cat]['ERR'].data < 6000.0) & (catalog[idx_cat]['LAM'].data > (1.0 + catalog[idx_cat]['ZSPEC']) * 3550)
        wave = catalog[idx_cat]['LAM'].data
        line_names, rest_waves = get_lines_to_fit(np.min(wave[mask]), np.max(wave[mask]), catalog[idx_cat]['ZSPEC'])

        # choose EL that are in wavelength range => add to obs
        model_params['eline_wavelength'] = {'N': len(rest_waves),
                                            'init': np.array(rest_waves),
                                            'isfree': False}

    # Now instantiate the model using this new dictionary of parameter specifications
    #model = sedmodel.PolySedModel(model_params)
    model = ElineMargSEDModel(model_params)

    return model


# --------------
# SPS Object
# --------------

def build_sps(zcontinuous=1, compute_vega_mags=False, **extras):
    from prospect.sources import CSPSpecBasis
    sps = CSPSpecBasis(zcontinuous=zcontinuous,
                       compute_vega_mags=compute_vega_mags)
    return sps

# -----------------
# Noise Model
# ------------------

def build_noise(**extras):
    return None, None

# -----------
# Everything
# ------------

def build_all(**kwargs):

    return (build_obs(**kwargs), build_model(**kwargs),
            build_sps(**kwargs), build_noise(**kwargs))


if __name__=='__main__':

    # - Parser with default arguments -
    parser = prospect_args.get_parser()
    # - Add custom arguments -
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
    parser.add_argument('--err_floor_phot', type=np.float, default=0.05,
                        help="Error floor for photometry.")
    parser.add_argument('--err_floor_spec', type=np.float, default=0.1,
                        help="Error floor for spectroscopy.")
    parser.add_argument('--S2N_cut', type=np.float, default=5.0,
                        help="Signal-to-noise cut applied to sample.")
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
    output = fit_model(obs, model, sps, noise, **run_params)

    writer.write_hdf5(hfile, run_params, model, obs,
                      output["sampling"][0], output["optimization"][0],
                      tsample=output["sampling"][1],
                      toptimize=output["optimization"][1])

    try:
        hfile.close()
    except(AttributeError):
        pass
