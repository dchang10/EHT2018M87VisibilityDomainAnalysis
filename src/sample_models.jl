using Pkg
Pkg.activate(abspath((@__DIR__)*"/../"))
Pkg.instantiate()

include(abspath((@__DIR__)*"/models/model_list.jl"))
include(abspath((@__DIR__)*"/models/gauss_comp_model_list.jl"))

# Point to directory containing uvfits files
dir = abspath((@__DIR__)*"/../data")
files = filter(x-> occursin(r".uvfits$", x), readdir(dir))

using Distributed
# Uncomment this line if you want to use parallel processing
#addprocs(9)

Stretched_M2Ring_And_Emission_Floor_And_2_Elliptical_Gaussians(prior) = Stretched_M2Ring_And_Emission_Floor_And_N_Elliptical_Gaussians(prior, 2)
Stretched_M2Ring_And_Emission_Floor_And_3_Elliptical_Gaussians(prior) = Stretched_M2Ring_And_Emission_Floor_And_N_Elliptical_Gaussians(prior, 3)
Stretched_M2Ring_And_Emission_Floor_And_4_Elliptical_Gaussians(prior) = Stretched_M2Ring_And_Emission_Floor_And_N_Elliptical_Gaussians(prior, 4)
Stretched_M2Ring_And_Emission_Floor_And_5_Elliptical_Gaussians(prior) = Stretched_M2Ring_And_Emission_Floor_And_N_Elliptical_Gaussians(prior, 5)
Stretched_M2Ring_And_Emission_Floor_And_6_Elliptical_Gaussians(prior) = Stretched_M2Ring_And_Emission_Floor_And_N_Elliptical_Gaussians(prior, 6)

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
        Stretched_M2Ring_And_Emission_Floor_And_3_Elliptical_Gaussians,
        Stretched_M2Ring_And_Emission_Floor_And_4_Elliptical_Gaussians,
        Stretched_M2Ring_And_Emission_Floor_And_5_Elliptical_Gaussians,
        Stretched_M2Ring_And_Emission_Floor_And_6_Elliptical_Gaussians,
        Two_M1Rings,
        Two_M2Rings,
        M1Ring_And_M2Ring,
        M0Ring_And_M1Ring
    ]

@everywhere begin

    using Pkg
    Pkg.activate(abspath((@__DIR__)*"/../"))

    Pkg.instantiate()
    include(abspath((@__DIR__)*"/models/model_list.jl"))
    include(abspath((@__DIR__)*"/models/gauss_comp_model_list.jl"))

    Stretched_M2Ring_And_Emission_Floor_And_2_Elliptical_Gaussians(prior) = Stretched_M2Ring_And_Emission_Floor_And_N_Elliptical_Gaussians(prior, 2)
    Stretched_M2Ring_And_Emission_Floor_And_2_Elliptical_Gaussians(prior) = Stretched_M2Ring_And_Emission_Floor_And_N_Elliptical_Gaussians(prior, 2)
    Stretched_M2Ring_And_Emission_Floor_And_3_Elliptical_Gaussians(prior) = Stretched_M2Ring_And_Emission_Floor_And_N_Elliptical_Gaussians(prior, 3)
    Stretched_M2Ring_And_Emission_Floor_And_4_Elliptical_Gaussians(prior) = Stretched_M2Ring_And_Emission_Floor_And_N_Elliptical_Gaussians(prior, 4)
    Stretched_M2Ring_And_Emission_Floor_And_5_Elliptical_Gaussians(prior) = Stretched_M2Ring_And_Emission_Floor_And_N_Elliptical_Gaussians(prior, 5)
    Stretched_M2Ring_And_Emission_Floor_And_6_Elliptical_Gaussians(prior) = Stretched_M2Ring_And_Emission_Floor_And_N_Elliptical_Gaussians(prior, 6)
    

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
        uvfile
        obs
        cphase
        lcamp
    end

    function SampleRun(uvfile) 

        obs = ehtim.obsdata.load_uvfits(uvfile)

        obs.add_scans()
        obs = scan_average(obs.flag_uvdist(uv_min=0.1e9)).add_fractional_noise(0.01) # add 1% fractional systematic error
        
        cphase = Comrade.extract_cphase(obs, ;snrcut=3) # extract minimal set of closure phases
        lcamp = Comrade.extract_lcamp(obs, ;snrcut=3) # extract minimal set of log-closure amplitudes

        return SampleRun(uvfile, obs, cphase, lcamp)
    end

    SampleRun_list = [SampleRun(joinpath(dir, file)) for file in files]
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
