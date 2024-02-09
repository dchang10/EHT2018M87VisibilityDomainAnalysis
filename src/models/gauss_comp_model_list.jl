using Comrade
using Chain
using Distributions

_pd = product_distribution
function _model_prior(param, prior)
    dist_list = map(x-> if param[x] > 1 _pd([prior[x] for _ in 1:param[x]]) else prior[x] end, keys(param))
    return NamedTuple{keys(param)}(dist_list)
end

# Helper functions for μas units
shiftedμas(model, x, y) = shifted(model, μas2rad(x), μas2rad(y))
smoothedμas(model, width) = smoothed(model, μas2rad(width))
stretchedμas(model, rx, ry) = stretched(model, μas2rad(rx), μas2rad(ry))

const PRIOR = (
    rad = Uniform(10.0, 30.0),# Radius of primary component
    inner_rad_ratio = Uniform(0.1, 0.99),# Radius ratio of Crescent cutout
    τm = Uniform(0.1, 1.),# Ratio of Stretched MRing major axis with respect to minor axis
    ξm = Uniform(0, π),# Angular position of Stretched MRing major axis or Crescent PA
    rg = Uniform(2.0, 40.0),# Radial size of Gaussians
    τg = Uniform(0.1, 1),# Ratio of Elliptical Gaussian major axis with respect to minor axis
    ξg = Uniform(0, π),# Angular position of Elliptical Gaussian major axis
    xg = Uniform(-50.0, 50.0),# Gaussian horizontal displacement with respect to primary
    yg = Uniform(-50.0, 50.0),# Gaussian vertical displacement with respect to primary
    f = Uniform(0., 1.),# Flux ratio between primary and secondary components
    fg = Uniform(0.51, 1),# Flux ratio between Gaussians (Forces ordering based on flux)
    shift_ratio = Uniform(0.01, 0.99),# Crescent cutout offset from center
    slash= Uniform(0.0, 1.0),# Crescent slash
    floor = Uniform(0.0, 1.0),# Crescent Emission floor flux ratio
    width = Uniform(0.1, 50.0),# Gaussian blur width
    width_ratio =  Uniform(0.1, 0.99),# Double MRing Gaussian blur ratio between primary and secondary MRings. (Used in 2 MRing models)
    amp = Uniform(0.0, 0.5),# Amplitude of MRing modes
    phase = Uniform(0, 2π),# Phase of MRing modes
)

#=
    Other
=#
function Circular_Gaussian(prior)
    param_count = (rg=1,)
    cir_gauss(θ) = begin
        (;rg) = θ 
         stretchedμas(Gaussian(), rg, rg)
    end
    return (model=cir_gauss, prior=_model_prior(param_count, prior))
end

function Elliptical_Gaussian(prior)
    param_count = (rg=1, ξg=1, τg=1)
    ell_gauss(θ) =  begin 
        (;rg, ξg, τg) = θ
        @chain Gaussian() begin
            stretchedμas(_, rg, rg*(τg+1))
            rotated(_, ξg)
        end
    end

    return (model=ell_gauss, prior=_model_prior(param_count, prior))
end

function Two_Circular_Gaussians(prior)
    param_count = (rg=2, xg=1, yg=1, fg=1)
    gauss1, _ = Circular_Gaussian(prior)
    gauss2, _ = Circular_Gaussian(prior)

    two_circ_gauss(θ) = begin 
        (;rg, xg, yg, fg) = θ
        prm1 = (rg=rg[1],)
        prm2 = (rg=rg[2],)

        fg*gauss1(prm1) + (1-fg)*(shiftedμas(gauss2(prm2), xg, yg))
    end

    return (model=two_circ_gauss, prior=_model_prior(param_count, prior))
end

function N_Elliptical_Gaussians(prior, n)
    n == 1 && return Elliptical_Gaussian(prior)
    
    param_count = (rg=n, ξg=n, τg=n, xg=(n-1), yg=(n-1), fg=(n-1))
    el_gauss, _ = Elliptical_Gaussian(prior)
    n_m_1_el_gauss, _ = N_Elliptical_Gaussians(prior, n-1)

    n_el_gauss(θ) = begin
        (;rg, ξg, τg, xg, yg, fg) = θ 
        prm = (rg=rg[1], ξg=ξg[1], τg=τg[1])
        #prm = (rg=rg[2:end], ξg=ξg[2:end], τg=τg[2:end], fg=fg[2:end], xg=xg[2:end], yg=yg[2:end])
        prm2 = n > 2 ? (rg=rg[2:end], ξg=ξg[2:end], τg=τg[2:end], xg=xg[2:end], yg=yg[2:end], fg=fg[2:end]) : 
            (rg=rg[2], ξg=ξg[2], τg=τg[2])

        fg[1]*el_gauss(prm) + (1-fg[1])*shiftedμas(n_m_1_el_gauss(prm2), xg[1], yg[1])
    end

    return (model=n_el_gauss, prior=_model_prior(param_count, prior))
end

##----------------------------------------------------------------------------------------------------------------------
#   Crescents
##----------------------------------------------------------------------------------------------------------------------
function TopDashHat_Crescent(prior)
    param_count = (rad=1, inner_rad_ratio=1, shift_ratio=1, ξm=1)
    
    th_crescent(θ) = begin
        (;rad, inner_rad_ratio, shift_ratio, ξm) = θ
        radμas = μas2rad(rad)
        shift = shift_ratio*(radμas*(1-inner_rad_ratio))
        r_inner = inner_rad_ratio*radμas

        rotated(Crescent(radμas, r_inner, shift, 0.0), ξm)
    end

    return (model=th_crescent, prior=_model_prior(param_count, prior))
end

function TopDashHat_Crescent_And_Emission_Floor(prior)
    param_count = (rad=1, inner_rad_ratio=1, shift_ratio=1, ξm=1, floor=1)
    th_crescent_floored(θ) = begin
        (;rad, inner_rad_ratio, shift_ratio, ξm, floor) = θ
            radμas = μas2rad(rad)
            rotated(Crescent(radμas, inner_rad_ratio*radμas, shift_ratio*(radμas*(1-inner_rad_ratio)), floor), ξm)
    end 

    return (model=th_crescent_floored, prior=_model_prior(param_count, prior))
end

function Blurred_Crescent_And_Emission_Floor(prior)
    
    param_count = (rad=1, inner_rad_ratio=1, shift_ratio=1, ξm=1, floor=1, width=1)
    th_crescent_floored, _ = TopDashHat_Crescent_And_Emission_Floor(prior)

    bl_crescent_floored(θ) = begin
        (;width) = θ
        smoothedμas(th_crescent_floored(θ), width)
    end

    return (model=bl_crescent_floored, prior=_model_prior(param_count, prior))
end

function BlurredAndSlashed_Crescent(prior)
    param_count = (rad=1, inner_rad_ratio=1, shift_ratio=1, ξm=1, width=1, slash=1)

    bl_sl_crescent(θ) = begin 
        (;rad, inner_rad_ratio, shift_ratio, ξm, width, slash) = θ

        radμas = μas2rad(rad)
        shift = shift_ratio*(radμas *(1-inner_rad_ratio))
        inner_rad = inner_rad_ratio*radμas

        @chain ConcordanceCrescent(radμas, inner_rad ,shift, slash) begin
            smoothedμas(_, width)
            rotated(_, ξm)
        end
    end

    return (model=bl_sl_crescent, prior=_model_prior(param_count, prior))
end

function BlurredAndSlashed_Crescent_And_Elliptical_Gaussian(prior)
    param_count = (rad=1, inner_rad_ratio=1, shift_ratio=1, ξm=1, ξg=1, width=1, slash=1, rg=1, τg=1, xg=1, yg=1, f=1)
    b_s_cresc, _ = BlurredAndSlashed_Crescent(prior)
    ell_gauss, _ = Elliptical_Gaussian(prior)
    
    bl_sl_crescent_ell_gauss(θ) = begin 
        (;rg, τg, ξg, xg, yg, f) = θ
        gauss_prm = (rg=rg, τg=τg, ξg=ξg)
        f*b_s_cresc(θ) + (1-f)*shiftedμas(ell_gauss(gauss_prm),xg, yg)
    end
    return (model=bl_sl_crescent_ell_gauss, prior=_model_prior(param_count, prior))
end

function BlurredAndSlashed_Crescent_And_2_Elliptical_Gaussians(prior)
    
    param_count = (rad=1, inner_rad_ratio=1, shift_ratio=1, ξm=1, ξg=2, width=1, slash=1,rg=2, τg=2, xg=2, yg=2, f=1,fg=1)

    bl_sl_crescent_2_ell_gauss(θ) = begin
        (;rg, τg, ξg, xg, yg, f, fg) = θ
        b_s_cresc, _ = BlurredAndSlashed_Crescent(prior)
        el_gauss, _ = Two_Elliptical_Gaussians(prior)

        gauss_prm = (rg=rg, τg=τg, ξg=ξg, xg=xg[2], yg=yg[2], fg=fg)
        comp1 = f*b_s_cresc(θ)
        comp2 = (1-f)*shiftedμas(el_gauss(gauss_prm), xg[1], yg[1])

        comp1 + comp2
    end

    return (model=bl_sl_crescent_2_ell_gauss, prior=_model_prior(param_count, prior))
end

function BlurredAndSlashed_Crescent_And_3_Elliptical_Gaussians(prior)
    
    param_count = (rad=1, inner_rad_ratio=1, shift_ratio=1, ξm=1, ξg=3, width=1, slash=1, rg=3, τg=3, xg=3, yg=3, f=1,fg=2)

    bl_sl_crescent_3_ell_gauss(θ) = begin
        (;rg, τg, ξg, xg, yg, f, fg) = θ
        b_s_cresc, _ = BlurredAndSlashed_Crescent(prior)
        el_gauss, _ = Three_Elliptical_Gaussians(prior)

        gauss_prm = (rg=rg, τg=τg, ξg=ξg, xg=xg[2:end], yg=yg[2:end], fg=fg)
        comp1 = f*b_s_cresc(θ)
        comp2 = (1-f)*shiftedμas(el_gauss(gauss_prm), xg[1], yg[1])

        comp1 + comp2
    end

    return (model=bl_sl_crescent_3_ell_gauss, prior=_model_prior(param_count, prior))
end

function BlurredAndSlashed_Crescent_And_Emission_Floor(prior)
    param_count = (rad=1, inner_rad_ratio=1, shift_ratio=1, ξm=1, width=1, slash=1, floor=1)

    bl_sl_crescent_floored(θ) = begin 
        (;rad, inner_rad_ratio, shift_ratio, ξm, width, floor) = θ
        b_s_cresc, _ = BlurredAndSlashed_Crescent(prior)
        inner_rad = inner_rad_ratio*rad
        shift = shift_ratio*(rad*(1-inner_rad_ratio))

        comp1 = b_s_cresc(θ)

        cresc_floor = @chain Disk() begin
                stretchedμas(_, inner_rad, inner_rad)
                shiftedμas(_, shift, 0.)
                rotated(_,ξm)
                smoothedμas(_,width[1])
        end
    
        comp1 + floor*cresc_floor
    end

    return (model=bl_sl_crescent_floored, prior=_model_prior(param_count, prior))
end



function BlurredAndSlashed_Crescent_And_Emission_Floor_And_2_Elliptical_Gaussians(prior)
    param_count = (rad=1, inner_rad_ratio=1, shift_ratio=1, ξm=1, ξg=2, width=1, slash=1, rg=2, τg=2, floor=1, xg=2, yg=2, f=1, fg=1)

    bl_sl_crescent_floored_2_ell_gauss(θ) = begin 
        (;rad, inner_rad_ratio, shift_ratio, ξm, width, floor) = θ
        b_s_cresc_3_el_gauss, _ = BlurredAndSlashed_Crescent_And_2_Elliptical_Gaussians(prior)
        inner_rad = inner_rad_ratio*rad
        shift = shift_ratio*(rad*(1-inner_rad_ratio))

        comp1 = b_s_cresc_3_el_gauss(θ)

        cresc_floor = @chain Disk() begin
                stretchedμas(_, inner_rad, inner_rad)
                shiftedμas(_, shift, 0.)
                rotated(_,ξm)
                smoothedμas(_,width[1])
        end
    
        comp1 + floor*cresc_floor
    end

    return (model=bl_sl_crescent_floored_2_ell_gauss, prior=_model_prior(param_count, prior))
end


function BlurredAndSlashed_Crescent_And_Emission_Floor_And_3_Elliptical_Gaussians(prior)
    param_count = (rad=1, inner_rad_ratio=1, shift_ratio=1, ξm=1, ξg=3, width=1, slash=1, rg=3, τg=3, floor=1, xg=3, yg=3, f=1, fg=2)

    bl_sl_crescent_floored_3_ell_gauss(θ) = begin 
        (;rad, inner_rad_ratio, shift_ratio, ξm, width, floor) = θ
        b_s_cresc_3_el_gauss, _ = BlurredAndSlashed_Crescent_And_3_Elliptical_Gaussians(prior)
        inner_rad = inner_rad_ratio*rad
        shift = shift_ratio*(rad*(1-inner_rad_ratio))

        comp1 = b_s_cresc_3_el_gauss(θ)

        cresc_floor = @chain Disk() begin
                stretchedμas(_, inner_rad, inner_rad)
                shiftedμas(_, shift, 0.)
                rotated(_,ξm)
                smoothedμas(_,width[1])
        end
    
        comp1 + floor*cresc_floor
    end

    return (model=bl_sl_crescent_floored_3_ell_gauss, prior=_model_prior(param_count, prior))
end


function XS_Dash_Ringauss(prior)
    param_count = (rad=1, inner_rad_ratio=1, ξm=1, shift_ratio=1, width=1, slash=1, rg=2, τg=1, fg=2)

    xs_dash_ringauss(θ) = begin
        (;rad, inner_rad_ratio, ξm, shift_ratio, width, slash, rg, τg, fg) = θ
        crescent_model, _ = BlurredAndSlashed_Crescent_And_2_Elliptical_Gaussians(PRIOR)

        xg1 = [shift_ratio*rad*(1-inner_rad_ratio), -rad*(inner_rad_ratio)]

        crescentprms = (
            rad=rad,
            inner_rad_ratio=inner_rad_ratio,
            shift_ratio=shift_ratio,
            ξm=0,
            ξg=[0,0],
            width=width,
            slash=slash,
            rg=rg,
            τg=[0, τg],
            xg=[0,sum(xg1)],
            yg=[0, 0],
            f=fg[1],
            fg=fg[2]
        )
        
        return rotated(crescent_model(crescentprms), ξm)

    end

    return (model=xs_dash_ringauss, prior=_model_prior(param_count, prior))
end

function XS_Dash_Ringauss_And_2_Elliptical_Gaussians(prior)
             #(rad=1, inner_rad_ratio=1, ξm=1, shift_ratio=1, ξg=2, width=1, slash=1, rg=2, τg=2, xg=2, yg=2, fc=1, fg=1)
    param_count = (rad=1, inner_rad_ratio=1, ξm=1, shift_ratio=1, width=1, slash=1, ξg=2, rg=4, τg=3, xg=2, yg=2, fg=3, f=1)

    xs_dash_ringauss_and_2_gauss(θ) = begin
        (;rad, inner_rad_ratio, ξm, shift_ratio, ξg, width, slash, rg, τg, xg, yg, fg, f) = θ
        crescent_model, _ = XS_Dash_Ringauss(PRIOR)
        el_gauss, _ = Two_Elliptical_Gaussians(prior)

        crescentprms = 
        (
            rad=rad,
            inner_rad_ratio=inner_rad_ratio,
            ξm=ξm,
            shift_ratio=shift_ratio,
            width=width,
            slash=slash,
            rg=rg[1:2],
            τg=τg[1],
            fg=fg[1:2]
        )
        gaussprm = (rg=rg[3:4], ξg=ξg[1:2], τg=τg[2:3], xg=xg[2], yg=yg[2], fg=fg[end])
        return f*crescent_model(crescentprms)+ (1-f)*shiftedμas(el_gauss(gaussprm), xg[1], yg[1])

    end

    return (model=xs_dash_ringauss_and_2_gauss, prior=_model_prior(param_count, prior))
end

##----------------------------------------------------------------------------------------------------------------------
#   MRings
##----------------------------------------------------------------------------------------------------------------------
function M1Ring(prior)
    param_count = (rad=1, width=1, amp=1, phase=1)
    mring(θ) = begin
        
        (;rad, width, amp, phase) = θ
        c = amp*cis(phase)

        @chain MRing([real(c)], [imag(c)]) begin
        stretchedμas(_,rad, rad)
        smoothedμas(_, width)
        end
    end 

    return (model=mring, prior=_model_prior(param_count, prior))

end

function M2Ring(prior)
    param_count = (rad=1, width=1, amp=2, phase=2)

    mring(θ) = begin
        (;rad, width, amp, phase) = θ
        c = amp .* cis.(phase) 

        @chain MRing(real.(c), imag.(c)) begin
        stretchedμas(_,rad, rad)
        smoothedμas(_, width)
        end
    end 

    return (model=mring, prior=_model_prior(param_count, prior))
end

##----------------------------------------------------------------------------------------------------------------------
#   Stretched MRings with Emission Floors
##----------------------------------------------------------------------------------------------------------------------

function Stretched_M1Ring_And_Emission_Floor(prior)
    param_count = (rad=1, width=1, amp=1, phase=1, ξm=1, τm=1, floor=1)

    floored_mring(θ) = begin
        (;rad, width, amp, phase, ξm, τm, floor) = θ
        flatfloor = @chain Disk() begin
            stretchedμas(_, rad, rad*τm)
            rotated(_, ξm)
            smoothedμas(_, width)
        end
            
        c = amp * cis(phase) 
        mring = @chain MRing(real(c), imag(c)) begin
            stretchedμas(_,rad, rad*τm)
            rotated(_, ξm)
            smoothedμas(_, width)
        end

        floor*mring + (1-floor)*flatfloor
    end

    return (model=floored_mring, prior=_model_prior(param_count, prior))
end



function Stretched_M2Ring_And_Emission_Floor(prior)
    param_count = (rad=1, width=1, amp=2, phase=2, ξm=1, τm=1, floor=1)

    floored_mring(θ) = begin
        (;rad, width, amp, phase, ξm, τm, floor) = θ
        flatfloor = @chain Disk() begin
            stretchedμas(_, rad, rad*τm)
            rotated(_, ξm)
            smoothedμas(_, width)
        end
            
        c = amp .* cis.(phase) 
        mring = @chain MRing(real.(c), imag.(c)) begin
            stretchedμas(_,rad, rad*τm)
            rotated(_, ξm)
            smoothedμas(_, width)
        end

        floor*mring + (1-floor)*flatfloor
    end

    return (model=floored_mring, prior=_model_prior(param_count, prior))
end

function Stretched_M2Ring_And_Emission_Floor_And_Elliptical_Gaussian(prior)
    param_count = (rad=1, rg=1, width=1, amp=2, phase=2, ξg=1, τg=1, ξm=1, τm=1, xg=1, yg=1, f=1, floor=1)
    el_gauss, _ = Elliptical_Gaussian(prior)
    mring_with_floor, _ = Stretched_M2Ring_And_Emission_Floor(prior)

    mring_and_ell_gauss(θ) = begin
        (;xg, yg, f) = θ

        f*mring_with_floor(θ) + (1-f)*shiftedμas(el_gauss(θ), xg, yg)
    end 

    return (model=mring_and_ell_gauss, prior=_model_prior(param_count, prior))
end

function Stretched_M2Ring_And_Emission_Floor_And_N_Elliptical_Gaussians(prior, n)
    param_count = (rad=1, width=1, amp=2, phase=2, ξm=1, τm=1, rg=n, ξg=n, τg=n, xg=n, yg=n, fg=(n-1), f=1, floor=1)
    el_gauss, _ = N_Elliptical_Gaussians(prior, n)
    mring_with_floor, _ = Stretched_M2Ring_And_Emission_Floor(prior)

    fiducial(θ) = begin
        (;rg, ξg, τg, xg, yg, fg, f) = θ
        gaus_prm = n > 1 ? (rg=rg, ξg=ξg, τg=τg, xg=xg[2:end], yg=yg[2:end], fg=fg, f=f) : 
            (rg=rg, ξg=ξg, τg=τg, fg=fg, f=f)

        f*mring_with_floor(θ) + (1-f)*shiftedμas(el_gauss(gaus_prm), xg[1], yg[1])
    end 

    return (model=fiducial, prior=_model_prior(param_count, prior))
end


##----------------------------------------------------------------------------------------------------------------------
#   Stretched MRings
##----------------------------------------------------------------------------------------------------------------------
function Stretched_M1Ring(prior)
    param_count = (rad=1, width=1, amp=1, phase=1, ξm=1, τm=1)

    mring(θ) = begin
        
        (;rad, width, amp, phase, ξm, τm) = θ
        c = amp*cis(phase)

        @chain MRing([real(c)], [imag(c)]) begin
        stretchedμas(_,rad, rad*τm)
        rotated(_, ξm)
        smoothedμas(_, width)
        end
    end 

    return (model=mring, prior=_model_prior(param_count, prior))

end

function Stretched_M2Ring(prior)
    param_count = (rad=1, width=1, amp=2, phase=2, ξm=1, τm=1)

    mring(θ) = begin
        (;rad, width, amp, phase, ξm, τm) = θ
        c = amp .* cis.(phase) 

        @chain MRing(real.(c), imag.(c)) begin
        stretchedμas(_,rad, rad*τm)
        rotated(_, ξm)
        smoothedμas(_, width)
        end
    end 

    return (model=mring, prior=_model_prior(param_count, prior))
end

function Stretched_M1Ring_And_Elliptical_Gaussian(prior)
    param_count = (rad=1, rg=1, width=1, amp=1, phase=1, ξg=1, τg=1, ξm=1, τm=1, xg=1, yg=1, f=1, fg=1)
    el_gauss, _ = Elliptical_Gaussian(prior)

    mring_and_ell_gauss(θ) = begin
        (;rad, rg, width, amp, phase, ξg, τg, ξm, τm, xg, yg, f, fg) = θ

        c = amp * cis(phase) 

        mring = @chain MRing([real(c)], [imag(c)]) begin
        stretchedμas(_,rad, rad*τm)
        rotated(_, ξm)
        smoothedμas(_, width)
        end
        prm = (rg=rg, ξg=ξg, τg=τg, fg=fg)
        f*mring + (1-f)*shiftedμas(el_gauss(prm), xg, yg)
    end 

    return (model=mring_and_ell_gauss, prior=_model_prior(param_count, prior))
end

function Stretched_M1Ring_And_2_Elliptical_Gaussians(prior)
    param_count = (rad=1, rg=2, width=1, amp=1, phase=1, ξg=2, τg=2, ξm=1, τm=1, xg=2, yg=2, f=1, fg=1)
    el_gauss, _ = Two_Elliptical_Gaussians(prior)

    mring_and_2_ell_gauss(θ) = begin
        (;rad, rg, width, amp, phase, ξg, τg, ξm, τm, xg, yg, f, fg) = θ

        c = amp * cis(phase) 

        mring = @chain MRing([real(c)], [imag(c)]) begin
        stretchedμas(_,rad, rad*τm)
        rotated(_, ξm)
        smoothedμas(_, width)
        end
        prm = (rg=rg, ξg=ξg, τg=τg, xg=xg[2], yg=yg[2], fg=fg)
        f*mring + (1-f)*shiftedμas(el_gauss(prm), xg[1], yg[1])
    end 

    return (model=mring_and_2_ell_gauss, prior=_model_prior(param_count, prior))
end

function Stretched_M2Ring_And_Elliptical_Gaussian(prior)
    param_count = (rad=1, rg=1, width=1, amp=2, phase=2, ξg=1, τg=1, ξm=1, τm=1, xg=1, yg=1, f=1, fg=1)
    el_gauss, _ = Elliptical_Gaussian(prior)

    mring_and_ell_gauss(θ) = begin
        (;rad, rg, width, amp, phase, ξg, τg, ξm, τm, xg, yg, f, fg) = θ

        c = amp .* cis.(phase) 

        mring = @chain MRing(real.(c), imag.(c)) begin
        stretchedμas(_,rad, rad*τm)
        rotated(_, ξm)
        smoothedμas(_, width)
        end
        prm = (rg=rg, ξg=ξg, τg=τg, fg=fg)
        f*mring + (1-f)*shiftedμas(el_gauss(prm), xg, yg)
    end 

    return (model=mring_and_ell_gauss, prior=_model_prior(param_count, prior))
end

function Stretched_M2Ring_And_2_Elliptical_Gaussians(prior)
    param_count = (rad=1, rg=2, width=1, amp=2, phase=2, ξg=2, τg=2, ξm=1, τm=1, xg=2, yg=2, f=1, fg=1)
    el_gauss, _ = Two_Elliptical_Gaussians(prior)

    mring_and_2_ell_gauss(θ) = begin
        (;rad, rg, width, amp, phase, ξg, τg, ξm, τm, xg, yg, f, fg) = θ

        c = amp .* cis.(phase) 

        mring = @chain MRing(real.(c), imag.(c)) begin
        stretchedμas(_,rad, rad*τm)
        rotated(_, ξm)
        smoothedμas(_, width)
        end
        prm = (rg=rg, ξg=ξg, τg=τg, xg=xg[2], yg=yg[2], fg=fg)
        f*mring + (1-f)*shiftedμas(el_gauss(prm), xg[1], yg[1])
    end 

    return (model=mring_and_2_ell_gauss, prior=_model_prior(param_count, prior))
end
