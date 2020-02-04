module Moments
export m

    using SymPy
    using HCubature
    using Distributions
    using LinearAlgebra
    using SpecialFunctions
    using Memoization
    using Combinatorics

    function sigma(l, k)
        sigma_l = factorial(l) / doublefactorial(2l + 1)
        lambda_lk = factorial(l + k + 1/2) / (factorial(big(k)) * factorial(1/2))
        return Float64(sigma_l * lambda_lk)
    end

    function c(l, k, m)
        num = (-1)^m * factorial(k + l + 1/2)
        denom = factorial(big(m)) * factorial(big(k - m)) * factorial(l + m + 1/2)
        Float64(num / denom)
    end

    function L(l, k)
        return x -> sum([c(l, k, m) * x^m for m in 0:k])
    end

    function p_sym(l::Int64)
        vx, vy, vz = symbols("vx vy vz")
        v = [vx, vy, vz]
        if l == 0
            return [Sym(1.)]
        elseif l == 1
            return v
        else
            lower_order = p_sym(l - 1)
            speed_squared = vx^2 + vy^2 + vz^2
            lcat = (l) -> (m1, m2) -> cat(m1, m2; dims=l)
            outer_product = reduce(lcat(l), [vi .* lower_order for vi in v])
            jac = reduce(lcat(l), [diff.(lower_order, vi) for vi in v])
            return outer_product .- (jac .* (speed_squared / (2l - 1)))
        end
    end

    function p(l::Int64)
        vx, vy, vz = symbols("vx vy vz")
        v = [vx, vy, vz]
        varargsify = p_l -> function(v)
            return p_l(v...)
        end
        varargsify.(lambdify.(p_sym(l), (v,))) # Prevents broadcast over vars
    end

    function p(l, k)
        p_l = p(l)
        L_lk = L(l, k)
        combine = p -> function(v)
            c = norm(v)
            return p(v) * L_lk(c^2)
        end
        combine.(p_l)
    end

    function integrand_func(p, sigma, f)
        return function(v)
            return p(v) * f(v) / sqrt(sigma)
        end
    end

    function m(l, k, f, a, b)
        p_lk = p(l, k)
        s = sigma(l, k)
        m = function(p_expr)
            integrand = integrand_func(p_expr, s, f)
            return hcubature(integrand, a, b, atol=0.001)[1]
        end
        m.(p_lk)
    end

    function f_lk_hat(l, k, m_lk)
        p_lk = p(l, k)
        sigma_lk = sigma(l, k)
        f_M = maxwellian([0., 0., 0.])
        fuse = v -> (p -> p(v))
        fused = v -> fuse(v).(p_lk)
        v -> sum(fused(v) .* m_lk * f_M(v) / sqrt(sigma_lk))
    end

    function moment_reconstruction(f0, lmax, kmax)
        a = [-10., -10., -10.]
        b = [10., 10., 10.]
        hats = []
        for l in 0:lmax
            for k in 0:kmax
                mlk = m(l, k, f0, a, b)
                flk_hat = f_lk_hat(l, k, mlk)
                push!(hats, flk_hat)
            end
        end
        return v -> sum([hat(v) for hat in hats])
    end

    function maxwellian(V)
        return function(v)
            c = norm(V - v)
            return pi^(-3/2) * exp(-c^2)
        end
    end

    fM = maxwellian([0., 0., 0.])
    f0 = fM

    a = [-10., -10., -10.]
    b = [10., 10., 10.]

    m00 = m(0, 0, f0, a, b)
    f00_hat = f_lk_hat(0, 0, m00)

    m01 = m(0, 1, f0, a, b)
    f01_hat = f_lk_hat(0, 1, m01)
    m02 = m(0, 2, f0, a, b)
    f02_hat = f_lk_hat(0, 2, m01)
    m03 = m(0, 3, f0, a, b)
    f03_hat = f_lk_hat(0, 3, m01)

    m10 = m(1, 0, f0, a, b)
    f10_hat = f_lk_hat(1, 0, m10)

    f_hat = v -> f00_hat(v) + f01_hat(v) + f10_hat(v) + f02_hat(v) + f03_hat(v)
end
