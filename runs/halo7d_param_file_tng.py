# import modules
import sys
import os
import numpy as np
import h5py
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


# function to build mock

def build_mock(sps, model,
               filterset=None,
               wavelength=None,
               snr_spec=10.0, snr_phot=20.0, add_noise=False,
               seed=101, **kwargs):
    """Make a mock dataset.  Feel free to add more complicated kwargs, and put
    other things in the run_params dictionary to control how the mock is
    generated.
    :param filterset:
        A list of `sedpy` filter names.  Mock photometry will be generated
        for these filters.
    :param wavelength:
        A vector
    :param snr_phot:
        The S/N of the phock photometry.  This can also be a vector of same
        lngth as the number of filters, for heteroscedastic noise.
    :param snr_spec:
        The S/N of the phock spectroscopy.  This can also be a vector of same
        lngth as `wavelength`, for heteroscedastic noise.
    :param add_noise: (optional, boolean, default: True)
        If True, add a realization of the noise to the mock photometry.
    :param seed: (optional, int, default: 101)
        If greater than 0, use this seed in the RNG to get a deterministic
        noise for adding to the mock data.
    """
    # We'll put the mock data in this dictionary, just as we would for real
    # data.  But we need to know which filters (and wavelengths if doing
    # spectroscopy) with which to generate mock data.

    mock = {"filters": None, "maggies": None, "wavelength": None, "spectrum": None}
    mock['wavelength'] = wavelength
    if filterset is not None:
        mock['filters'] = load_filters(filterset)

    # Now we get any mock params from the kwargs dict
    params = {}
    for p in model.params.keys():
        if p in kwargs:
            params[p] = np.atleast_1d(kwargs[p])

    # And build the mock
    model.params.update(params)
    spec, phot, mfrac = model.predict(model.theta, mock, sps=sps)

    # Now store some output
    mock['true_spectrum'] = spec.copy()
    mock['true_maggies'] = np.copy(phot)
    mock['mock_params'] = deepcopy(model.params)

    # store the mock photometry
    if filterset is not None:
        pnoise_sigma = phot / snr_phot
        mock['maggies'] = phot.copy()
        mock['maggies_unc'] = pnoise_sigma
        mock['mock_snr_phot'] = snr_phot
        # And add noise
        if add_noise:
            if int(seed) > 0:
                np.random.seed(int(seed))
            pnoise = np.random.normal(0, 1, size=len(phot)) * pnoise_sigma
            mock['maggies'] += pnoise

        mock['phot_wave'] = np.array([f.wave_effective for f in mock['filters']])

    # store the mock spectrum
    if wavelength is not None:
        snoise_sigma = spec / snr_spec
        mock['spectrum'] = spec.copy()
        mock['unc'] = snoise_sigma
        mock['mock_snr_spec'] = snr_spec
        # And add noise
        if add_noise:
            if int(seed) > 0:
                np.random.seed(int(seed))
            snoise = np.random.normal(0, 1, size=len(spec)) * snoise_sigma
            mock['spectrum'] += snoise

    return mock




# function to build obs dictionary

def build_obs(index_galaxy=0, filterset=None,
              dlambda_spec=2.0, wave_lo=3800, wave_hi=7000.,
              snr_spec=20., snr_phot=20., add_noise=False, seed=101, **kwargs):
    """Load a mock
    :param wave_lo:
        The (restframe) minimum wavelength of the spectrum.
    :param wave_hi:
        The (restframe) maximum wavelength of the spectrum.
    :param dlambda_spec:
        The (restframe) wavelength sampling or spacing of the spectrum.
    :param filterset:
        A list of `sedpy` filter names.  Mock photometry will be generated
        for these filters.
    :param snr_spec:
        S/N ratio for the spectroscopy per pixel.  scalar.
    :param snr_phot:
        The S/N of the phock photometry.  This can also be a vector of same
        lngth as the number of filters, for heteroscedastic noise.
    :param add_noise: (optional, boolean, default: True)
        Whether to add a noise reealization to the spectroscopy.
    :param seed: (optional, int, default: 101)
        If greater than 0, use this seed in the RNG to get a deterministic
        noise for adding to the mock data.
    :returns obs:
        Dictionary of observational data.
    """

    # get redshifted wavelength
    a = 1 + kwargs.get("zred", 0.0)
    wavelength = np.arange(wave_lo, wave_hi, dlambda_spec) * a

    # filter list
    if filterset is None:
        filterset = ['f160w_goodsn', 'u_goodsn', 'f435w_goodsn', 'b_goodsn', 'g_goodsn', 'v_goodsn', 'f606w_goodsn',
                     'r_goodsn', 'rs_goodsn', 'i_goodsn', 'f775w_goodsn', 'z_goodsn', 'f850lp_goodsn', 'f125w_goodsn',
                     'j_goodsn', 'f140w_goodsn', 'h_goodsn', 'ks_goodsn', 'irac1_goodsn', 'irac2_goodsn', 'irac3_goodsn',
                     'irac4_goodsn', 'mips_24um_goodsn', 'rp_goodsn', 'ip_goodsn', 'f814w_goodsn', 'zp_goodsn', 'uvista_y_goodsn',
                     'j1_goodsn', 'j2_goodsn', 'j3_goodsn', 'uvista_j_goodsn', 'h1_goodsn', 'h2_goodsn', 'uvista_h_goodsn', 'k_goodsn',
                     'uvista_ks_goodsn', 'ia427_goodsn', 'ia464_goodsn', 'ia484_goodsn', 'ia505_goodsn', 'ia527_goodsn', 'ia574_goodsn',
                     'ia624_goodsn', 'ia679_goodsn', 'ia709_goodsn', 'ia738_goodsn', 'ia767_goodsn', 'ia827_goodsn']

    # we need the models to make a mock.
    # for the SPS we use the Tabular SFH from TNG
    sps = build_sps(index_galaxy, **kwargs)
    assert len(sps.tabular_time) > 0
    model = build_model(**kwargs)

    mock = build_mock(sps, model, filterset=filterset, snr_phot=snr_phot,
                      wavelength=wavelength, snr_spec=snr_spec,
                      add_noise=add_noise, seed=seed)
    mock['index_galaxy'] = index_galaxy
    mock['tabular_time'] = sps.tabular_time.copy()
    mock['tabular_sfr'] = sps.tabular_sfr.copy()
    mock['tabular_mtot'] = sps.mtot

    return mock


# --------------
# Model Definition
# --------------

def build_model(zred=0.0, non_param_sfh=False, dirichlet_sfh=False, add_duste=False, add_neb=False, add_agn=False, switch_off_mix=False, marginalize_neb=True,
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
    model_params["zred"]["init"] = zred
    model_params["zred"]["prior"] = priors.TopHat(mini=zred-0.005, maxi=zred+0.005)

    # get SFH template
    if non_param_sfh:
        t_univ = cosmo.age(zred).value
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
        model_params["tage"]["prior"] = priors.TopHat(mini=1e-3, maxi=cosmo.age(zred).value)
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
        polyorder_estimate = int(np.clip(np.round((np.min([7500*(zred+1), 9150.0])-np.max([3525.0*(zred+1), 6000.0]))/(zred+1)*100), 10, 30))
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

from prospect.sources import SSPBasis

class TabularBasis(SSPBasis):
    """Subclass of :py:class:`SSPBasis` that implements a fixed tabular SFH.
    The user must add the `tabular_time`, `tabular_sfr`, and `mtot` attributes
    """

    def get_galaxy_spectrum(self, **params):
        """Construct the tabular SFH and feed it to the ``ssp``.
        """
        self.update(**params)
        self.ssp.params["sfh"] = 3  # Hack to avoid rewriting the superclass
        self.ssp.set_tabular_sfh(self.tabular_time, self.tabular_sfr)
        wave, spec = self.ssp.get_spectrum(tage=-99, peraa=False)
        return wave, spec / self.mtot, self.ssp.stellar_mass / self.mtot


def extract_sfh(illustris_sfh_file="", index_galaxy=0):
    data = h5py.File(illustris_sfh_file, 'r')
    time = data['info']['sfh_tbins'][:]
    sfr = data['sfh_insitu_sfr'][:][index_galaxy] + data['sfh_exsitu_sfr'][:][index_galaxy]
    return time, sfr


def build_sps(zcontinuous=1, compute_vega_mags=False, illustris_sfh_file="", index_galaxy=0, zred=0.0, **extras):

    # extract SFH
    time, sfr = extract_sfh(illustris_sfh_file, index_galaxy=index_galaxy)
    tuniv = cosmo.age(zred).value
    inds = slice(0, np.argmin(np.abs(tuniv - time)))

    # poplulate sps object
    sps = TabularBasis(zcontinuous=zcontinuous,
                       compute_vega_mags=compute_vega_mags)
    sps.tabular_time = tuniv - time[inds]
    sps.tabular_sfr = sfr[inds]
    sps.mtot = np.trapz(sfr[inds], time[inds]) * 1e9

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
    parser.add_argument('--illustris_sfh_file', type=str, default=path_wdir + "data/galaxies_tng100_059_SFH30.hdf5",
                        help="Name of TNG file.")
    parser.add_argument('--index_galaxy', type=int, default=0,
                        help="Zero-index row number in the table to fit.")
    parser.add_argument('--zred', type=np.float, default=0.7,
                        help="Redshift of mock.")
    parser.add_argument('--non_param_sfh', action="store_true",
                        help="If set, fit non-parametric star-formation history model.")
    parser.add_argument('--dirichlet_sfh', action="store_true",
                        help="If set, fit non-parametric star-formation history model with dirichlet prior.")
    parser.add_argument('--n_bins_sfh', type=int, default=8,
                        help="Number of bins for SFH (non parametric).")
    parser.add_argument('--add_neb', action="store_true",
                        help="If set, add nebular emission in the model (and mock).")
    parser.add_argument('--fit_continuum', action="store_true",
                        help="If set, fit continuum.")
    parser.add_argument('--add_duste', action="store_true",
                        help="If set, add dust emission to the model.")
    parser.add_argument('--add_agn', action="store_true",
                        help="If set, add agn emission to the model.")
    parser.add_argument('--snr_spec', type=np.float, default=10.0,
                        help="SNR of spectroscopy.")
    parser.add_argument('--snr_phot', type=np.float, default=20.0,
                        help="SNR of photometry.")
    parser.add_argument('--use_eline_prior', action="store_true",
                        help="If set, use EL prior from cloudy.")
    parser.add_argument('--add_jitter', action="store_true",
                        help="If set, jitter noise.")
    parser.add_argument('--fixed_dust', action="store_true",
                        help="If set, fix dust to Calzetti and add tight Z prior.")
    parser.add_argument('--add_noise', action="store_true",
                        help="Add noise to mock.")

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

    hfile = path_wdir + "results/{0}_idx_{1}_mcmc.h5".format(args.outfile, int(args.index_galaxy)-1)
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
