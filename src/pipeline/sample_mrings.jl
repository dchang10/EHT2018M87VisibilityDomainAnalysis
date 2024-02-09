using Pkg
Pkg.activate(abspath((@__DIR__)*"/../../"))
Pkg.instantiate()

include(abspath((@__DIR__)*"/../models/gauss_comp_model_list.jl"))

using Distributed
using Comrade
using VLBIImagePriors
using Pyehtim
using Plots
using Chain
using Distributions

using ComradeDynesty
using StatsBase
using TypedTables 
using CSV

Stretched_M2Ring_And_Emission_Floor_And_2_Elliptical_Gaussians(prior) = Stretched_M2Ring_And_Emission_Floor_And_N_Elliptical_Gaussians(prior, 2)
Stretched_M2Ring_And_Emission_Floor_And_3_Elliptical_Gaussians(prior) = Stretched_M2Ring_And_Emission_Floor_And_N_Elliptical_Gaussians(prior, 3)
Stretched_M2Ring_And_Emission_Floor_And_4_Elliptical_Gaussians(prior) = Stretched_M2Ring_And_Emission_Floor_And_N_Elliptical_Gaussians(prior, 4)
Stretched_M2Ring_And_Emission_Floor_And_5_Elliptical_Gaussians(prior) = Stretched_M2Ring_And_Emission_Floor_And_N_Elliptical_Gaussians(prior, 5)
Stretched_M2Ring_And_Emission_Floor_And_6_Elliptical_Gaussians(prior) = Stretched_M2Ring_And_Emission_Floor_And_N_Elliptical_Gaussians(prior, 6)

models = [
    Stretched_M2Ring_And_Emission_Floor,
    Stretched_M2Ring_And_Emission_Floor_And_Elliptical_Gaussian,
    Stretched_M2Ring_And_Emission_Floor_And_2_Elliptical_Gaussians,
    Stretched_M2Ring_And_Emission_Floor_And_3_Elliptical_Gaussians,
    Stretched_M2Ring_And_Emission_Floor_And_4_Elliptical_Gaussians,
    Stretched_M2Ring_And_Emission_Floor_And_5_Elliptical_Gaussians,
    Stretched_M2Ring_And_Emission_Floor_And_6_Elliptical_Gaussians    
    ]


addprocs(9)
@everywhere begin

    using Pkg
    Pkg.activate(abspath((@__DIR__)*"/../../"))
    Pkg.instantiate()
    include(abspath((@__DIR__)*"/../models/gauss_comp_model_list.jl"))

    using Comrade
    using VLBIImagePriors
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
        
        cphase = extract_table(obs, ClosurePhases(;snrcut=3)) # extract minimal set of closure phases
        lcamp =  extract_table(obs, LogClosureAmplitudes(;snrcut=3)) # extract minimal set of log-closure amplitudes

        return SampleRun(band, uvfile, obs, cphase, lcamp)
    end

    SampleRun_list = [SampleRun(band) for band in ["b3"]]
end

pmap(models) do i
    for sr in SampleRun_list
        model, prior = i(PRIOR)
        lklhd = RadioLikelihood(model, sr.lcamp, sr.cphase)
        model_prior = VLBIImagePriors.NamedDist(prior)
        posterior = Posterior(lklhd, model_prior)
        tpost = ascube(posterior) # Transforms into unit hypercube
        ndim = dimension(tpost) # Gets diminsions of parameter space

        smplr = NestedSampler(ndim, nlive=10_000, sample="rwalk")
        chain, stats =  sample(posterior, smplr)
        echain = sample(chain, Weights(stats.weights), 10_000, walks=100) |> Table

        outbase = first(splitext(sr.uvfile))*"/"
        outbase = replace(outbase, "data"=>"comp_runs")
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