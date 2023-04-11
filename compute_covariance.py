import os
import sys
from pathlib import Path

import bilby
import cupy as cp
try:
    cp.cuda.Device(1).use()
except:
    pass
import dill
import h5py
import numpy as np
from attrs import define
from gwpopulation.conversions import mu_var_max_to_alpha_beta_max
from gwpopulation_pipe import vt_helper
from gwpopulation_pipe.data_analysis import load_model
from scipy.interpolate import interp1d
from scipy.spatial.distance import jensenshannon
from scipy.stats import beta
from tqdm.auto import trange

bilby.core.utils.log.logger.setLevel("ERROR")
FITS = dict()
PLOT = False


@define
class Dump:
    models: list
    max_redshift: float = 1.9
    minimum_mass: int = 2
    maximum_mass: int = 100


def compute_pop_weights(samples, model):
    return model.prob(samples) / samples["prior"]


def compute_sel_weights(vt_data, vt_model):
    weights = vt_model.prob(vt_data) / vt_data["prior"]
    return vt_model.prob(vt_data) / vt_data["prior"]


def load_samples(base):
    try:
        samples = dict()
        with h5py.File(f"{base}_samples.hdf5", "r") as ff:
            for key in ff["original"].keys():
                samples[key] = cp.asarray(ff["original"][key][()])
    except OSError:
        with open(f"{base}_samples.pkl", "rb") as ff:
            samples = dill.load(ff)["original"]
        for key in samples:
            samples[key] = cp.asarray(samples[key])
    print(f"Loading sample file for {base}")
    return samples


def load_result(base, n_samples=0, nest=False):
    if os.path.exists(base):
        result = bilby.core.result.read_in_result(base)
    else:
        result = bilby.core.result.read_in_result(f"{base}_result.json")
    print(f"Loading result file for {base}")
    posterior = result.nested_samples.copy()
    if nest:
        print(f"Using nested samples from {base}_result.json")
        posterior = posterior[posterior["weights"] > 0]
    else:
        print(f"Using posterior samples from {base}_result.json")
        posterior["weights"] /= max(posterior["weights"])
        keep = np.random.uniform(0, 1, len(posterior)) <= posterior["weights"]
        posterior = posterior[keep].copy()
        posterior["weights"] = 1
    for key in result.priors:
        if key not in posterior:
            posterior.loc[:, key] = result.priors[key].sample(len(posterior))
    if "mu_chi" in posterior:
        posterior.loc[:, "alpha_chi"], posterior.loc[:, "beta_chi"], _ = mu_var_max_to_alpha_beta_max(
            posterior["mu_chi"],
            posterior["sigma_chi"],
            posterior["amax"],
        )
    if n_samples > 0 and n_samples < len(posterior):
        posterior = posterior.sample(n_samples)
    posterior.loc[:, "weights"] /= sum(posterior["weights"])
    return posterior, result.meta_data["models"]


def load_injections(base):
    if "tuned" in outdir:
    # if "tuned" in base or "scaling" in base:
        inj_file = "/home/jacob.golomb/montecarlo_uncertainties/injectionset/jan3/O3_jacob_tuned.hdf5"
    elif "more_injections" in outdir:
    # elif "more_injections" in base or "scaling" in base or True:
        inj_file = "/home/jacob.golomb/SpinTest/InjectionSet/jul21/O3_jacob.hdf5"
    elif outdir == "production":
        inj_file = "/home/reed.essick/rates+pop/o1+o2+o3-sensitivity-estimates/LIGO-T2100377-v2/o1+o2+o3_bbhpop_real+semianalytic-LIGO-T2100377-v2.hdf5"
    else:
        inj_file = "/home/reed.essick/rates+pop/o1+o2+o3-sensitivity-estimates/LIGO-T2100377-v2/o1+o2+o3_bbhpop_real+semianalytic-LIGO-T2100377-v2.hdf5"
        # raise ValueError(f"Unable to determine VT file for {inj_file}")
    print(f"Reading {inj_file}")
    # inj_file = "/home/reed.essick/rates+pop/o1+o2+o3-sensitivity-estimates/LIGO-T2100377-v2/o1+o2+o3_bbhpop_real+semianalytic-LIGO-T2100377-v2.hdf5"
    vt_data = vt_helper.load_injection_data(inj_file, snr_threshold=10)
    return vt_data


def optimized(
    samples,
    posterior,
    vt_data,
    model,
    vt_model,
    reduce_vt=1,
    reduce_population=1,
    outdir=".",
    nest=False,
    n_events=69,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(outdir, reduce_vt, reduce_population)

    if reduce_population > 1:
        original_n_samples = samples["prior"].shape[1]
        idxs = np.random.choice(
            original_n_samples, original_n_samples // reduce_population, replace=False
        )
        samples = {key: samples[key][:, idxs] for key in samples}

    if reduce_vt > 1:
        idxs = np.random.choice(
            len(vt_data["prior"]), len(vt_data["prior"]) // reduce_vt, replace=False
        )
        temp = dict()
        for key in vt_data:
            try:
                temp[key] = vt_data[key][idxs]
            except (TypeError, IndexError):
                temp[key] = vt_data[key]
        vt_data = temp
        vt_data["total_generated"] /= reduce_vt
    vt_eval = vt_helper.ResamplingVT(vt_model, vt_data, n_events=0)

    # final shape (n_samples, )
    all_weights = list()
    for ii in trange(len(posterior)):
        model.parameters = dict(posterior.iloc[ii])
        weights = compute_pop_weights(samples, model)
        all_weights.append(weights)
    all_weights = cp.asarray(all_weights)

    # final shape (n_samples, )
    mus = list()
    # final shape (n_samples, n_samples)
    sel_covar = np.empty((len(posterior), len(posterior)))
    all_sel_weights = list()
    for ii in trange(len(posterior)):
        vt_model.parameters.update(dict(posterior.iloc[ii]))
        weights = compute_sel_weights(vt_data, vt_model)
        all_sel_weights.append(weights)
    all_sel_weights = cp.asarray(all_sel_weights)

    post_weights = cp.asarray(posterior["weights"].values)

    weights = all_weights
    norm = weights.shape[-1]
    means = cp.mean(weights, axis=-1)
    mean_products = cp.einsum("ik,jk->ijk", means, means)
    # this is the covariance between ln likelihood biases for each pair of points
    # due to the per-event integrals
    pop_covar = cp.sum(
        cp.einsum("ikl,jkl->ijk", weights, weights) / norm / mean_products,
        axis=-1,
    ) / norm
    weights = all_sel_weights
    norm = weights.shape[-1]
    norm2 = vt_eval.total_injections
    means = cp.sum(weights, axis=-1) / norm2
    mean_products = cp.outer(means, means)
    # this is the covariance between ln likelihood biases for each pair of points
    # due to the selection function integral
    sel_covar = n_events**2 * (
        cp.einsum("ik,jk->ij", weights, weights) / norm2
    ) / (norm2 * mean_products)
    cov = 1 * pop_covar + sel_covar
    u, s, v = cp.linalg.svd(cov)
    if nest:
        fname = f"{outdir}/covariance_nest"
    else:
        fname = f"{outdir}/covariance"
    np.savez(fname, pop=pop_covar, sel=sel_covar, u=u, s=s, v=v, weights=post_weights)
    scale_length = float(cp.mean(s))
    n_scales = float(cp.sum(s)**2 / cp.sum(s**2))
    print(f"Mean eigenvalue: {scale_length:.2f}, scale length: {scale_length:.2f}, path length: {n_scales * scale_length:.2f}")

    cov = sel_covar
    u, s, v = cp.linalg.svd(cov)

    effs = list()
    for _ in range(100):
        weights = cp.exp(cp.dot(cp.random.normal(0, 1, len(s)), cp.sqrt(s)[:, None] * v))
        eff = float(cp.sum(post_weights * weights)**2 / cp.sum(post_weights * weights**2))
        effs.append(eff)
    print(np.median(effs))

    post_weights = cp.asnumpy(post_weights)[:, None]

    xvals = np.linspace(1e-5, 1 - 1e-5, 1000)
    curves = list()
    for ii in trange(len(posterior)):
        curves.append(beta(posterior["alpha_chi"].iloc[ii], posterior["beta_chi"].iloc[ii]).pdf(xvals))
    curves = np.array(curves)
    ppd = np.sum(post_weights * curves, axis=0)

    if outdir == Path("production"):
        alt_posterior, _ = load_result("../no_limits/production/result/o1o2o3_IMR_mass_c_iid_spin_magnitude_mag_iid_spin_orientati_tilt_powerlaw_redshift_result.json", nest=nest)
    elif outdir == Path("more_injections"):
        alt_posterior, _ = load_result("../no_limits/production_more_injections/result/o1o2o3_IMR_mass_c_iid_spin_magnitude_mag_iid_spin_orientati_tilt_powerlaw_redshift_result.json", nest=nest)
    elif outdir == Path("tuned"):
        alt_posterior, _ = load_result("../no_limits/n_events_scaling/tuned/result/69_mass_c_iid_spin_magnitude_mag_iid_spin_orientation_tilt_powerlaw_redshift_result.json", nest=nest)
    elif outdir == Path("LVK"):
        alt_posterior, _ = load_result("/home/jacob.golomb/O3b/mar15/init/result/o1o2o3_IMR_mass_c_iid_mag_iid_tilt_powerlaw_redshift_result.json", nest=nest)
    else:
        print(f"Didn't recognize outdir: {outdir}")
        sys.exit()
    alt_ppd = xvals * 0
    for ii in range(len(alt_posterior)):
        alt_ppd += beta(alt_posterior["alpha_chi"].iloc[ii], alt_posterior["beta_chi"].iloc[ii]).pdf(xvals)
    alt_ppd /= len(alt_posterior)

    alt_curve = list()
    divergences = list()
    for _ in trange(1000):
        weights = cp.asnumpy(cp.exp(cp.dot(cp.random.normal(0, 1, len(s)), cp.sqrt(s)[:, None] * v)))
        curve = np.sum(curves * post_weights * weights[:, np.newaxis], axis=0)
        curve /= np.trapz(curve, xvals)
        alt_curve.append(curve)
        divergences.append(jensenshannon(curve, ppd))
    alt_curve = np.array(alt_curve)
    sorted_curves = np.sort(curves, axis=0)
    bounds = list()
    for ii, val in enumerate(np.cumsum(post_weights[:, 0][np.argsort(curves, axis=0)], axis=0).T):
        bounds.append(sorted_curves[:, ii][interp1d(val, np.arange(len(val)), kind="nearest")([0.05, 0.95]).astype(int)])
    bounds = np.array(bounds)

    if nest:
        fname = f"{outdir}/data_nest"
    else:
        fname = f"{outdir}/data"
    np.savez(fname, ppd=ppd, alt_ppd=alt_ppd, alt_curve=alt_curve, bounds=bounds, alt_bounds=np.percentile(alt_curve, [5, 95], axis=0).T)


if __name__ == "__main__":
    base = sys.argv[1]
    outdir = sys.argv[2]
    nest = True

    posterior, models = load_result(base, n_samples=3000, nest=nest)

    args = Dump(models)

    samples = load_samples(base)
    vt_data = load_injections(outdir)
    n_events = samples["prior"].shape[0]

    kwargs = dict(
        samples=samples,
        vt_data=vt_data,
        posterior=posterior,
        nest=nest,
        n_events=n_events,
    )

    model = load_model(args)
    vt_model = load_model(args)
    optimized(
        reduce_vt=1,
        reduce_population=1,
        outdir=outdir,
        vt_model=vt_model,
        model=model,
        **kwargs,
    )
