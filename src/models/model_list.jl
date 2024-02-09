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
    params = (rg=1,)
    cir_gauss(θ) = begin
        (;rg) = θ 
         stretchedμas(Gaussian(), rg, rg)
    end
    return (model=cir_gauss, prior=_model_prior(params, prior))
end

function Elliptical_Gaussian(prior)
    params = (rg=1, ξg=1, τg=1)
    ell_gauss(θ) =  begin 
        (;rg, ξg, τg) = θ
        @chain Gaussian() begin
            stretchedμas(_, rg, rg*(τg+1))
            rotated(_, ξg)
        end
    end

    return (model=ell_gauss, prior=_model_prior(params, prior))
end

function TopDashHat_Disk(prior)
    params = (rad=1,)
    th_disk(θ) = begin
        (;rad) = θ 
        stretchedμas(Disk(), rad, rad)
    end
    return (model=th_disk, prior=_model_prior(params, prior))
end

function TopDashHat_Ring(prior)
    params = (rad=1, inner_rad_ratio=1)
    th_ring(θ) = begin
        (;rad, inner_rad_ratio) = θ
        radμas = μas2rad(rad)
        Crescent(radμas, inner_rad_ratio*radμas, 0.0, 0.0)
    end

    return (model=th_ring, prior=_model_prior(params, prior))
end

function Two_Circular_Gaussians(prior)
    params = (rg=2, xg=1, yg=1, fg=1)
    gauss1, _ = Circular_Gaussian(prior)
    gauss2, _ = Circular_Gaussian(prior)

    two_circ_gauss(θ) = begin 
        (;rg, xg, yg, fg) = θ
        prm1 = (rg=rg[1],)
        prm2 = (rg=rg[2],)

        fg*gauss1(prm1) + (1-fg)*(shiftedμas(gauss2(prm2), xg, yg))
    end

    return (model=two_circ_gauss, prior=_model_prior(params, prior))
end

function Two_Elliptical_Gaussians(prior)
    params = (rg=2, ξg=2, τg=2, xg=1, yg=1, fg=1)
    gauss1, _ = Elliptical_Gaussian(prior)
    gauss2, _ = Elliptical_Gaussian(prior)

    two_ellip_gauss(θ) = begin
        (;rg, ξg, τg, xg, yg, fg) = θ 
        println(θ)
        prm1 = (rg=rg[1], ξg=ξg[1], τg=τg[1])
        prm2 = (rg=rg[2], ξg=ξg[2], τg=τg[2])

        fg*gauss1(prm1) + (1-fg)*(shiftedμas(gauss2(prm2), xg, yg))
    end

    return (model=two_ellip_gauss, prior=_model_prior(params, prior))
end

function Three_Elliptical_Gaussians(prior)
    params = (rg=3, ξg=3, τg=3, xg=2, yg=2, fg=2)
    gauss1, _ = Elliptical_Gaussian(prior)
    gauss2, _ = Elliptical_Gaussian(prior)
    gauss3, _ = Elliptical_Gaussian(prior)

    three_ellip_gauss(θ) = begin
        (;rg, ξg, τg, xg, yg, fg) = θ 
        prm1 = (rg=rg[1], ξg=ξg[1], τg=τg[1])
        prm2 = (rg=rg[2], ξg=ξg[2], τg=τg[2])
        prm3 = (rg=rg[3], ξg=ξg[3], τg=τg[3])


        fg[1]*gauss1(prm1) + (1-fg[1])*(fg[2]*(shiftedμas(gauss2(prm2), xg[1], yg[1])) + (1-fg[2])*(shiftedμas(gauss3(prm3), xg[2], yg[2])))

    end

    return (model=three_ellip_gauss, prior=_model_prior(params, prior))
end

function Four_Elliptical_Gaussians(prior)
    params = (rg=4, ξg=4, τg=4, xg=3, yg=3, fg=3)
    gauss1, _ = Elliptical_Gaussian(prior)
    gauss2, _ = Elliptical_Gaussian(prior)
    gauss3, _ = Elliptical_Gaussian(prior)
    gauss4, _ = Elliptical_Gaussian(prior)

    four_ellip_gauss(θ) = begin
        (;rg, ξg, τg, xg, yg, fg) = θ 
        prm1 = (rg=rg[1], ξg=ξg[1], τg=τg[1])
        prm2 = (rg=rg[2], ξg=ξg[2], τg=τg[2])
        prm3 = (rg=rg[3], ξg=ξg[3], τg=τg[3])
        prm4 = (rg=rg[4], ξg=ξg[4], τg=τg[4])


        fg[1]*gauss1(prm1) + (1-fg[1])*(
                fg[2]*(shiftedμas(gauss2(prm2), xg[1], yg[1])) + (1-fg[2])*(
                    fg[3]*shiftedμas(gauss3(prm3), xg[2], yg[2]) + (1-fg[3])*shiftedμas(gauss4(prm4), xg[3], yg[3])
                )
            )

    end

    return (model=four_ellip_gauss, prior=_model_prior(params, prior))
end

#=
    Crescents
=#
function TopDashHat_Crescent(prior)
    params = (rad=1, inner_rad_ratio=1, shift_ratio=1, ξm=1)
    
    th_crescent(θ) = begin
        (;rad, inner_rad_ratio, shift_ratio, ξm) = θ
        radμas = μas2rad(rad)
        shift = shift_ratio*(radμas*(1-inner_rad_ratio))
        r_inner = inner_rad_ratio*radμas

        rotated(Crescent(radμas, r_inner, shift, 0.0), ξm)
    end

    return (model=th_crescent, prior=_model_prior(params, prior))
end

function TopDashHat_Crescent_And_Emission_Floor(prior)
    params = (rad=1, inner_rad_ratio=1, shift_ratio=1, ξm=1, floor=1)
    th_crescent_floored(θ) = begin
        (;rad, inner_rad_ratio, shift_ratio, ξm, floor) = θ
            radμas = μas2rad(rad)
            rotated(Crescent(radμas, inner_rad_ratio*radμas, shift_ratio*(radμas*(1-inner_rad_ratio)), floor), ξm)
    end 

    return (model=th_crescent_floored, prior=_model_prior(params, prior))
end

function Blurred_Crescent_And_Emission_Floor(prior)
    
    params = (rad=1, inner_rad_ratio=1, shift_ratio=1, ξm=1, floor=1, width=1)
    th_crescent_floored, _ = TopDashHat_Crescent_And_Emission_Floor(prior)

    bl_crescent_floored(θ) = begin
        (;width) = θ
        smoothedμas(th_crescent_floored(θ), width)
    end

    return (model=bl_crescent_floored, prior=_model_prior(params, prior))
end

function BlurredAndSlashed_Crescent(prior)
    params = (rad=1, inner_rad_ratio=1, shift_ratio=1, ξm=1, width=1, slash=1)

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

    return (model=bl_sl_crescent, prior=_model_prior(params, prior))
end

function BlurredAndSlashed_Crescent_And_Elliptical_Gaussian(prior)
    params = (rad=1, inner_rad_ratio=1, shift_ratio=1, ξm=1, ξg=1, width=1, slash=1, rg=1, τg=1, xg=1, yg=1, f=1)
    b_s_cresc, _ = BlurredAndSlashed_Crescent(prior)
    ell_gauss, _ = Elliptical_Gaussian(prior)
    
    bl_sl_crescent_ell_gauss(θ) = begin 
        (;rg, τg, ξg, xg, yg, f) = θ
        gauss_prm = (rg=rg, τg=τg, ξg=ξg)
        f*b_s_cresc(θ) + (1-f)*shiftedμas(ell_gauss(gauss_prm),xg, yg)
    end
    return (model=bl_sl_crescent_ell_gauss, prior=_model_prior(params, prior))
end

function BlurredAndSlashed_Crescent_And_2_Elliptical_Gaussians(prior)
    
    params = (rad=1, inner_rad_ratio=1, shift_ratio=1, ξm=1, ξg=2, width=1, slash=1,rg=2, τg=2, xg=2, yg=2, f=1,fg=1)

    bl_sl_crescent_2_ell_gauss(θ) = begin
        (;rg, τg, ξg, xg, yg, f, fg) = θ
        b_s_cresc, _ = BlurredAndSlashed_Crescent(prior)
        el_gauss, _ = Two_Elliptical_Gaussians(prior)

        gauss_prm = (rg=rg, τg=τg, ξg=ξg, xg=xg[2], yg=yg[2], fg=fg)
        comp1 = f*b_s_cresc(θ)
        comp2 = (1-f)*shiftedμas(el_gauss(gauss_prm), xg[1], yg[1])

        comp1 + comp2
    end

    return (model=bl_sl_crescent_2_ell_gauss, prior=_model_prior(params, prior))
end

function BlurredAndSlashed_Crescent_And_3_Elliptical_Gaussians(prior)
    
    params = (rad=1, inner_rad_ratio=1, shift_ratio=1, ξm=1, ξg=3, width=1, slash=1, rg=3, τg=3, xg=3, yg=3, f=1,fg=2)

    bl_sl_crescent_3_ell_gauss(θ) = begin
        (;rg, τg, ξg, xg, yg, f, fg) = θ
        b_s_cresc, _ = BlurredAndSlashed_Crescent(prior)
        el_gauss, _ = Three_Elliptical_Gaussians(prior)

        gauss_prm = (rg=rg, τg=τg, ξg=ξg, xg=xg[2:end], yg=yg[2:end], fg=fg)
        comp1 = f*b_s_cresc(θ)
        comp2 = (1-f)*shiftedμas(el_gauss(gauss_prm), xg[1], yg[1])

        comp1 + comp2
    end

    return (model=bl_sl_crescent_3_ell_gauss, prior=_model_prior(params, prior))
end

function BlurredAndSlashed_Crescent_And_Emission_Floor_And_2_Elliptical_Gaussians(prior)
    params = (rad=1, inner_rad_ratio=1, shift_ratio=1, ξm=1, ξg=2, width=1, slash=1, rg=2, τg=2, floor=1, xg=2, yg=2, f=1, fg=1)

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

    return (model=bl_sl_crescent_floored_2_ell_gauss, prior=_model_prior(params, prior))
end


function BlurredAndSlashed_Crescent_And_Emission_Floor_And_3_Elliptical_Gaussians(prior)
    params = (rad=1, inner_rad_ratio=1, shift_ratio=1, ξm=1, ξg=3, width=1, slash=1, rg=3, τg=3, floor=1, xg=3, yg=3, f=1, fg=2)

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

    return (model=bl_sl_crescent_floored_3_ell_gauss, prior=_model_prior(params, prior))
end

#=
    MRings
=#
function M1Ring(prior)
    params = (rad=1, width=1, amp=1, phase=1)
    mring(θ) = begin
        
        (;rad, width, amp, phase) = θ
        c = amp*cis(phase)

        @chain MRing([real(c)], [imag(c)]) begin
        stretchedμas(_,rad, rad)
        smoothedμas(_, width)
        end
    end 

    return (model=mring, prior=_model_prior(params, prior))

end

function M2Ring(prior)
    params = (rad=1, width=1, amp=2, phase=2)

    mring(θ) = begin
        (;rad, width, amp, phase) = θ
        c = amp .* cis.(phase) 

        @chain MRing(real.(c), imag.(c)) begin
        stretchedμas(_,rad, rad)
        smoothedμas(_, width)
        end
    end 

    return (model=mring, prior=_model_prior(params, prior))
end

function M3Ring(prior)
    params = (rad=1, width=1, amp=3, phase=3)

    mring(θ) = begin
        (;rad, width, amp, phase) = θ
        c = amp .* cis.(phase) 

        @chain MRing(real.(c), imag.(c)) begin
        stretchedμas(_,rad, rad)
        smoothedμas(_, width)
        end
    end 

    return (model=mring, prior=_model_prior(params, prior))
end


function M4Ring(prior)
    params = (rad=1, width=1, amp=4, phase=4)

    mring(θ) = begin
        (;rad, width, amp, phase) = θ
        c = amp .* cis.(phase) 

        @chain MRing(real.(c), imag.(c)) begin
        stretchedμas(_,rad, rad)
        smoothedμas(_, width)
        end
    end 

    return (model=mring, prior=_model_prior(params, prior))
end


function M1Ring_And_Elliptical_Gaussian(prior)
    params = (rad=1, rg=1, width=1, amp=1, phase=1, ξg=1, τg=1, xg=1, yg=1, f=1, fg=1)
    el_gauss, _ = Elliptical_Gaussian(prior)

    mring_and_ell_gauss(θ) = begin
        (;rad, rg, width, amp, phase, ξg, τg, xg, yg, f, fg) = θ

        c = amp * cis(phase) 

        mring = @chain MRing([real(c)], [imag(c)]) begin
        stretchedμas(_,rad, rad)
        smoothedμas(_, width)
        end
        prm = (rg=rg, ξg=ξg, τg=τg, fg=fg)
        f*mring + (1-f)*shiftedμas(el_gauss(prm), xg, yg)
    end 

    return (model=mring_and_ell_gauss, prior=_model_prior(params, prior))
end

function M1Ring_And_2_Elliptical_Gaussians(prior)
    params = (rad=1, rg=2, width=1, amp=1, phase=1, ξg=2, τg=2, xg=2, yg=2, f=1, fg=1)
    el_gauss, _ = Two_Elliptical_Gaussians(prior)

    mring_and_2_ell_gauss(θ) = begin
        (;rad, rg, width, amp, phase, ξg, τg, xg, yg, f, fg) = θ

        c = amp * cis(phase) 

        mring = @chain MRing([real(c)], [imag(c)]) begin
        stretchedμas(_,rad, rad)
        smoothedμas(_, width)
        end
        prm = (rg=rg, ξg=ξg, τg=τg, xg=xg[2], yg=yg[2], fg=fg)
        f*mring + (1-f)*shiftedμas(el_gauss(prm), xg[1], yg[1])
    end 

    return (model=mring_and_2_ell_gauss, prior=_model_prior(params, prior))
end


function M2Ring_And_Elliptical_Gaussian(prior)
    params = (rad=1, rg=1, width=1, amp=2, phase=2, ξg=1, τg=1, xg=1, yg=1, f=1, fg=1)
    el_gauss, _ = Elliptical_Gaussian(prior)

    mring_and_ell_gauss(θ) = begin
        (;rad, rg, width, amp, phase, ξg, τg, xg, yg, f, fg) = θ

        c = amp .* cis.(phase) 

        mring = @chain MRing(real.(c), imag.(c)) begin
        stretchedμas(_,rad, rad)
        smoothedμas(_, width)
        end
        prm = (rg=rg, ξg=ξg, τg=τg, fg=fg)
        f*mring + (1-f)*shiftedμas(el_gauss(prm), xg, yg)
    end 

    return (model=mring_and_ell_gauss, prior=_model_prior(params, prior))
end

function M2Ring_And_2_Elliptical_Gaussians(prior)
    params = (rad=1, rg=2, width=1, amp=2, phase=2, ξg=2, τg=2, xg=2, yg=2, f=1, fg=1)
    el_gauss, _ = Two_Elliptical_Gaussians(prior)

    mring_and_2_ell_gauss(θ) = begin
        (;rad, rg, width, amp, phase, ξg, τg, xg, yg, f, fg) = θ

        c = amp .* cis.(phase) 

        mring = @chain MRing(real.(c), imag.(c)) begin
        stretchedμas(_,rad, rad)
        smoothedμas(_, width)
        end
        prm = (rg=rg, ξg=ξg, τg=τg, xg=xg[2], yg=yg[2], fg=fg)
        f*mring + (1-f)*shiftedμas(el_gauss(prm), xg[1], yg[1])
    end 

    return (model=mring_and_2_ell_gauss, prior=_model_prior(params, prior))
end

function M3Ring_And_Elliptical_Gaussian(prior)
    params = (rad=1, rg=1, width=1, amp=3, phase=3, ξg=1, τg=1, xg=1, yg=1, f=1, fg=1)
    el_gauss, _ = Elliptical_Gaussian(prior)

    mring_and_ell_gauss(θ) = begin
        (;rad, rg, width, amp, phase, ξg, τg, xg, yg, f, fg) = θ

        c = amp .* cis.(phase) 

        mring = @chain MRing(real.(c), imag.(c)) begin
        stretchedμas(_,rad, rad)
        smoothedμas(_, width)
        end
        prm = (rg=rg, ξg=ξg, τg=τg, fg=fg)
        f*mring + (1-f)*shiftedμas(el_gauss(prm), xg, yg)
    end 

    return (model=mring_and_ell_gauss, prior=_model_prior(params, prior))
end


function M3Ring_And_2_Elliptical_Gaussians(prior)
    params = (rad=1, rg=2, width=1, amp=3, phase=3, ξg=2, τg=2, xg=2, yg=2, f=1, fg=1)
    el_gauss, _ = Two_Elliptical_Gaussians(prior)

    mring_and_2_ell_gauss(θ) = begin
        (;rad, rg, width, amp, phase, ξg, τg, xg, yg, f, fg) = θ

        c = amp .* cis.(phase) 

        mring = @chain MRing(real.(c), imag.(c)) begin
        stretchedμas(_,rad, rad)
        smoothedμas(_, width)
        end
        prm = (rg=rg, ξg=ξg, τg=τg, xg=xg[2], yg=yg[2], fg=fg)
        f*mring + (1-f)*shiftedμas(el_gauss(prm), xg[1], yg[1])
    end 

    return (model=mring_and_2_ell_gauss, prior=_model_prior(params, prior))
end

#=
    Stretched MRings
=#
function Stretched_M1Ring(prior)
    params = (rad=1, width=1, amp=1, phase=1, ξm=1, τm=1)

    mring(θ) = begin
        
        (;rad, width, amp, phase, ξm, τm) = θ
        c = amp*cis(phase)

        @chain MRing([real(c)], [imag(c)]) begin
        stretchedμas(_,rad, rad*τm)
        rotated(_, ξm)
        smoothedμas(_, width)
        end
    end 

    return (model=mring, prior=_model_prior(params, prior))

end

function Stretched_M2Ring(prior)
    params = (rad=1, width=1, amp=2, phase=2, ξm=1, τm=1)

    mring(θ) = begin
        (;rad, width, amp, phase, ξm, τm) = θ
        c = amp .* cis.(phase) 

        @chain MRing(real.(c), imag.(c)) begin
        stretchedμas(_,rad, rad*τm)
        rotated(_, ξm)
        smoothedμas(_, width)
        end
    end 

    return (model=mring, prior=_model_prior(params, prior))
end

function Stretched_M3Ring(prior)
    params = (rad=1, width=1, amp=3, phase=3, ξm=1, τm=1)

    mring(θ) = begin
        (;rad, width, amp, phase, ξm, τm) = θ
        c = amp .* cis.(phase) 

        @chain MRing(real.(c), imag.(c)) begin
        stretchedμas(_,rad, rad*τm)
        rotated(_, ξm)
        smoothedμas(_, width)
        end
    end 

    return (model=mring, prior=_model_prior(params, prior))
end

function Stretched_M1Ring_And_Elliptical_Gaussian(prior)
    params = (rad=1, rg=1, width=1, amp=1, phase=1, ξg=1, τg=1, ξm=1, τm=1, xg=1, yg=1, f=1, fg=1)
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

    return (model=mring_and_ell_gauss, prior=_model_prior(params, prior))
end

function Stretched_M1Ring_And_2_Elliptical_Gaussians(prior)
    params = (rad=1, rg=2, width=1, amp=1, phase=1, ξg=2, τg=2, ξm=1, τm=1, xg=2, yg=2, f=1, fg=1)
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

    return (model=mring_and_2_ell_gauss, prior=_model_prior(params, prior))
end


function Stretched_M2Ring_And_Elliptical_Gaussian(prior)
    params = (rad=1, rg=1, width=1, amp=2, phase=2, ξg=1, τg=1, ξm=1, τm=1, xg=1, yg=1, f=1, fg=1)
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

    return (model=mring_and_ell_gauss, prior=_model_prior(params, prior))
end

function Stretched_M2Ring_And_2_Elliptical_Gaussians(prior)
    params = (rad=1, rg=2, width=1, amp=2, phase=2, ξg=2, τg=2, ξm=1, τm=1, xg=2, yg=2, f=1, fg=1)
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

    return (model=mring_and_2_ell_gauss, prior=_model_prior(params, prior))
end

function Stretched_M3Ring_And_Elliptical_Gaussian(prior)
    params = (rad=1, rg=1, width=1, amp=3, phase=3, ξg=1, τg=1,  ξm=1, τm=1, xg=1, yg=1, f=1, fg=1)
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

    return (model=mring_and_ell_gauss, prior=_model_prior(params, prior))
end


function Stretched_M3Ring_And_2_Elliptical_Gaussians(prior)
    params = (rad=1, rg=2, width=1, amp=3, phase=3, ξg=3, τg=2, τm=1, ξm=1, xg=2, yg=2, f=1, fg=1)
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

    return (model=mring_and_2_ell_gauss, prior=_model_prior(params, prior))
end

#=
    Two MRings
=#
function Two_M1Rings(prior)
    params = (rad=2, width=1, width_ratio=1, amp=2, phase=2, f=1)

    two_mring(θ) = begin
        
        (;rad, width, width_ratio, amp, phase, f) = θ
        c1 = amp[1]*cis(phase[1])
        c2 = amp[1]*cis(phase[2])


        ring1 = @chain MRing([real(c1)], [imag(c1)]) begin
        stretchedμas(_,rad[1], rad[1])
        smoothedμas(_, width)
        end

        ring2 = @chain MRing([real(c2)], [imag(c2)]) begin
        stretchedμas(_,rad[2], rad[2])
        smoothedμas(_, width*width_ratio)
        end

        f*ring1 + (1-f)*ring2
    end 

    return (model=two_mring, prior=_model_prior(params, prior))

end

function Two_M2Rings(prior)
    params = (rad=2, width=1, width_ratio=1, amp=4, phase=4, f=1)

    two_mring(θ) = begin

        (;rad, width, width_ratio, amp, phase, f) = θ
        amp1 = amp[begin:begin+1]
        phase1 = phase[begin:begin+1]

        amp2 = amp[begin+2:end]
        phase2 = phase[begin+2:end]

        c1 = amp1 .* cis.(phase1)
        c2 = amp2 .* cis.(phase2)

        ring1 = @chain MRing(real.(c1), imag.(c1)) begin
        stretchedμas(_,rad[1], rad[1])
        smoothedμas(_, width)
        end

        ring2 = @chain MRing(real.(c2), imag.(c2)) begin
        stretchedμas(_,rad[2], rad[2])
        smoothedμas(_, width*width_ratio)
        end

        f*ring1 + (1-f)*ring2
    end 

    return (model=two_mring, prior=_model_prior(params, prior))

end

function M0Ring_And_M1Ring(prior)
    params = (rad=2, width=1, width_ratio=1, amp=1, phase=1, f=1)

    two_mring(θ) = begin
        
        (;rad, width, width_ratio,  amp, phase, f) = θ
        c1 = amp*cis(phase)


        ring1 = @chain Ring() begin
        stretchedμas(_,rad[1], rad[1])
        smoothedμas(_, width)
        end

        ring2 = @chain MRing([real(c1)], [imag(c1)]) begin
        stretchedμas(_,rad[2], rad[2])
        smoothedμas(_, width*width_ratio)
        end

        f*ring1 + (1-f)*ring2
    end 

    return (model=two_mring, prior=_model_prior(params, prior))

end


function M1Ring_And_M2Ring(prior)
    params = (rad=2, width=2, amp=3, phase=3, f=1)

    two_mring(θ) = begin
        
        (;rad, width, amp, phase, f) = θ
        c1 = amp[1]*cis(phase[1])
        c2 = amp[begin+1:end] .* cis.(phase[begin+2:end])


        ring1 = @chain MRing([real(c1)], [imag(c1)]) begin
        stretchedμas(_,rad[1], rad[1])
        smoothedμas(_, width[1])
        end

        ring2 = @chain MRing(real.(c2), imag.(c2)) begin
        stretchedμas(_,rad[2], rad[2])
        smoothedμas(_, width[2])
        end

        f*ring1 + (1-f)*ring2
    end 

    return (model=two_mring, prior=_model_prior(params, prior))

end

