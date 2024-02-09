using Pkg
Pkg.activate(abspath((@__DIR__)*"/../../"))
Pkg.instantiate()

using Distributed

include(abspath((@__DIR__)*"/../models/model_list.jl"))
include(abspath((@__DIR__)*"/../models/gauss_comp_model_list.jl"))

Stretched_M2Ring_And_Emission_Floor_And_2_Elliptical_Gaussians(prior) = Stretched_M2Ring_And_Emission_Floor_And_N_Elliptical_Gaussians(prior, 2)

using Comrade
using Pyehtim
using Plots
using Chain
using Distributions
using VLBIImagePriors

using ComradeDynesty
using StatsBase
using TypedTables 
using CSV

models = [
    Circular_Gaussian, 
    Elliptical_Gaussian, 
    TopDashHat_Disk, 
    TopDashHat_Ring, 
    Two_Circular_Gaussians, 
    Two_Elliptical_Gaussians, 
    Three_Elliptical_Gaussians, 
    Four_Elliptical_Gaussians, 
    TopDashHat_Crescent, 
    TopDashHat_Crescent_And_Emission_Floor,
    Blurred_Crescent_And_Emission_Floor, 
    BlurredAndSlashed_Crescent, 
    BlurredAndSlashed_Crescent_And_Elliptical_Gaussian, 
    BlurredAndSlashed_Crescent_And_2_Elliptical_Gaussians, 
    BlurredAndSlashed_Crescent_And_3_Elliptical_Gaussians,  
    BlurredAndSlashed_Crescent_And_Emission_Floor, 
    BlurredAndSlashed_Crescent_And_Emission_Floor_And_2_Elliptical_Gaussians, 
    BlurredAndSlashed_Crescent_And_Emission_Floor_And_3_Elliptical_Gaussians, 
    M1Ring, 
    M2Ring, 
    M3Ring, 
    M4Ring,
    M1Ring_And_Elliptical_Gaussian,
    M1Ring_And_2_Elliptical_Gaussians,
    M2Ring_And_Elliptical_Gaussian,
    M2Ring_And_2_Elliptical_Gaussians,
    M3Ring_And_Elliptical_Gaussian,
    M3Ring_And_2_Elliptical_Gaussians,
    Stretched_M1Ring,
    Stretched_M2Ring,
    Stretched_M3Ring,
    Stretched_M1Ring_And_Elliptical_Gaussian,
    Stretched_M1Ring_And_2_Elliptical_Gaussians,
    Stretched_M2Ring_And_Elliptical_Gaussian,
    Stretched_M2Ring_And_2_Elliptical_Gaussians,
    Stretched_M3Ring_And_Elliptical_Gaussian,
    Stretched_M3Ring_And_2_Elliptical_Gaussians,
    Stretched_M1Ring_And_Emission_Floor,
    Stretched_M2Ring_And_Emission_Floor,
    Stretched_M2Ring_And_Emission_Floor_And_Elliptical_Gaussian,
    Stretched_M2Ring_And_Emission_Floor_And_2_Elliptical_Gaussians,
    Two_M1Rings,
    Two_M2Rings,
    M1Ring_And_M2Ring,
    M0Ring_And_M1Ring
   ]

addprocs(9)
@everywhere begin

    using Pkg
    Pkg.activate(abspath((@__DIR__)*"/../../"))

    Pkg.instantiate()
    include(abspath((@__DIR__)*"/../models/model_list.jl"))
    include(abspath((@__DIR__)*"/../models/gauss_comp_model_list.jl"))

    Stretched_M2Ring_And_Emission_Floor_And_2_Elliptical_Gaussians(prior) = Stretched_M2Ring_And_Emission_Floor_And_N_Elliptical_Gaussians(prior, 2)

    using Comrade
    using Pyehtim
    using Plots
    using Chain
    using Distributions

    using ComradeDynesty
    using StatsBase
    using TypedTables 
    using CSV


    struct SampleRun
        band
        uvfile
        obs
        cphase
        lcamp
    end

    function SampleRun(band) 
        uvfile = abspath((@__DIR__)*"/../../data/M87+3C279_Pre-L2/hops/hops-"*band*"/3644/hops_3644_M87.freqavg.polcal.uvfits")

        obs = ehtim.obsdata.load_uvfits(joinpath(@__DIR__, uvfile))

        obs.add_scans()
        obs = scan_average(obs.flag_uvdist(uv_min=0.1e9)).add_fractional_noise(0.01) # add 1% fractional systematic error
        
        cphase = Comrade.extract_cphase(obs, ;snrcut=3) # extract minimal set of closure phases
        lcamp = Comrade.extract_lcamp(obs, ;snrcut=3) # extract minimal set of log-closure amplitudes

        return SampleRun(band, uvfile, obs, cphase, lcamp)
    end

    SampleRun_list = [SampleRun(band) for band in ["b1", "b2", "b3", "b4"]]
end

pmap(models) do i
    for sr in SampleRun_list
        model, prior = i(PRIOR)
        lklhd = RadioLikelihood(model, sr.lcamp, sr.cphase)
        model_prior = VLBIImagePriors.NamedDist(prior)
        posterior = Posterior(lklhd, model_prior)
        tpost = ascube(posterior) # Transforms into unit hypercube
        ndim = dimension(tpost) # Gets diminsions of parameter space

        smplr = NestedSampler(ndim, nlive=10_000, sample="rwalk", walks=100)
        chain, stats =  sample(posterior, smplr)
        echain = sample(chain, Weights(stats.weights), 10_000) |> Table

        outbase = first(splitext(sr.uvfile))*"/"
        outbase = replace(outbase, "data"=>"runs")
        mkpath(dirname(outbase))
            
        CSV.write(outbase*string(i)*"_dynesty_chain.csv", chain)
        CSV.write(outbase*string(i)*"_echain.csv", echain)
        open(outbase*string(i)*"_stats.csv", "w") do io
            println(io,"logl,weights")
            println(io, "logZ: ", stats.logz)
            println(io, "logZerr: ", stats.logzerr)
            tbl = (logl = stats.logl, weights=stats.weights)
            CSV.write(io, tbl, append=true, writeheader=false)
            nothing
        end


        out = (chain=chain, stats=stats)
        evidence = stats.logz

        model_name = replace(replace(replace(string(i), "Dash"=>"-"), "And"=>"+"), "_"=>" ")

        # Save MAP images
        outbase = first(splitext(sr.uvfile))*"/"
        outbase = replace(outbase, "data"=>"plots")
        mkpath(dirname(outbase))
        plt = plot(model(chain[end]), title="MAP")
        savefig(plt, outbase*string(i)*"_MAP.png")


        print("echain length: $(length(echain))\n")
        msamples = model.(echain[8_000:5:end])
        imgs = intensitymap.(msamples, μas2rad(100.0), μas2rad(100.0), 256, 256)
        img_mean, img_std = mean_and_std(imgs)
        plt = plot(img_mean, title="Mean", clims=(0.0, maximum(img_mean)))
        savefig(plt, outbase*string(i)*"_Mean.png")
        plt = plot(img_std, title="STD", clims=(0.0, maximum(img_mean)))
        savefig(plt, outbase*string(i)*"_STD.png")
    end
end
