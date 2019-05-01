# import modules

import time
import sys
import os

import numpy as np
from sedpy.observate import load_filters

from prospect import prospect_args
from prospect.fitting import fit_model
from prospect.io import write_results as writer


# define paths

path_wdir = os.environ['WDIR_halo7d']
filter_folder = path_wdir + '/data/filters/'


def build_obs(objid=1, data_table=path_wdir + 'data/halo7d_with_phot.fits', err_floor=0.05, remove_mips24=False, **kwargs):
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
    obs['maggies_unc'] = np.clip(mags_err * 1e-10, mags * 1e-10 * err_floor, np.inf)
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
    obs['unc'] = np.clip(catalog[idx_cat]['ERR'].data * conversion_factor, catalog[idx_cat]['FLUX'].data * conversion_factor * err_floor, np.inf)
    # mask emission lines
    dA_line = 5.0  # in Angstrom
    rest_waves = np.array([4862.69, 4341.69, 4102.92, 3971.19, 3890.15, 3836.48, 3798.98, 3869.81, 3727.09, 3729.88, 5008.24, 4960.30])
    mask = np.ones(len(catalog[idx_cat]['LAM']), dtype=bool)
    for ii_line in rest_waves:
        mask = mask & (catalog[idx_cat]['LAM'].data > (1.0 + catalog[idx_cat]['ZSPEC']) * ii_line + dA_line) & (catalog[idx_cat]['LAM'].data < (1.0 + catalog[idx_cat]['ZSPEC']) * ii_line - dA_line)
    obs['mask'] = (catalog[idx_cat]['ERR'].data < 6000.0) & (catalog[idx_cat]['LAM'].data > (1.0 + catalog[idx_cat]['ZSPEC']) * 3550)
    # Add unessential bonus info.  This will be stored in output
    #obs['dmod'] = catalog[ind]['dmod']
    obs['cat_row'] = idx_cat
    obs['id_halo7d'] = catalog[idx_cat]['ID']
    obs['id_3dhst'] = catalog[idx_cat]['id_3dhst']
    obs['field'] = catalog[idx_cat]['FIELD']
    obs['RA'] = catalog[idx_cat]['RA']
    obs['DEC'] = catalog[idx_cat]['DEC']
    obs['redshift'] = catalog[idx_cat]['ZSPEC']
    return obs


# --------------
# Model Definition
# --------------

def build_model(objid=1, data_table=path_wdir + 'data/halo7d_with_phot.fits', add_duste=False, add_neb=False, add_agn=False, fit_continuum=False, **extras):
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
    from prospect.models.templates import TemplateLibrary
    from prospect.models import priors, sedmodel
    #from prospect.models import transforms
    from astropy.table import Table
    from astropy.cosmology import Planck15 as cosmo
    # read in data table
    catalog = Table.read(data_table)
    idx_cat = objid-1
    # --- Get a basic delay-tau SFH parameter set. ---
    # This has 5 free parameters:
    #   "mass", "logzsol", "dust2", "tage", "tau"
    # And two fixed parameters
    #   "zred"=0.1, "sfh"=4
    # See the python-FSPS documentation for details about most of these
    # parameters.  Also, look at `TemplateLibrary.describe("parametric_sfh")` to
    # view the parameters, their initial values, and the priors in detail.
    model_params = TemplateLibrary["parametric_sfh"]

    # adjust priors
    model_params["dust2"]["prior"] = priors.TopHat(mini=0.0, maxi=2.0)
    model_params["tau"]["prior"] = priors.LogUniform(mini=1e-1, maxi=10)
    model_params["tage"]["prior"] = priors.LogUniform(mini=0.0, maxi=cosmo.age(catalog[idx_cat]['ZSPEC']).value)
    model_params["mass"]["prior"] = priors.LogUniform(mini=1e10, maxi=1e12)
    model_params["logzsol"]["prior"] = priors.TopHat(mini=-1.0, maxi=0.2)

    # fit for redshift
    model_params["zred"]['isfree'] = True
    model_params["zred"]["prior"] = priors.TopHat(mini=catalog[idx_cat]['ZSPEC']-0.01, maxi=catalog[idx_cat]['ZSPEC']+0.01)

    # velocity dispersion
    model_params.update(TemplateLibrary['spectral_smoothing'])
    model_params["sigma_smooth"]["prior"] = priors.TopHat(mini=100.0, maxi=350.0)

    # Change the model parameter specifications based on some keyword arguments
    if add_duste:
        # Add dust emission (with fixed dust SED parameters)
        model_params.update(TemplateLibrary["dust_emission"])
        model_params['duste_gamma']['isfree'] = False
        model_params['duste_qpah']['isfree'] = False
        model_params['duste_umin']['isfree'] = False

    if add_agn:
        # Add dust emission (with fixed dust SED parameters)
        model_params.update(TemplateLibrary["agn"])
        model_params['fagn']['isfree'] = True
        model_params['agn_tau']['isfree'] = True

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
                                     'prior': priors.Normal(sigma=0.5, mean=1.0),
                                     'units': 'f_true/f_obs'}

    # Now instantiate the model using this new dictionary of parameter specifications
    model = sedmodel.PolySedModel(model_params)

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
    parser.add_argument('--err_floor', type=np.float, default=0.05,
                        help="Error floor for photometry and spectroscopy.")

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
