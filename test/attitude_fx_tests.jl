using LinearAlgebra, Test

@testset "randq" begin
    let
        for i = 1:100
            q = randq()

            @test isequal(ndims(q),1)
            @test isequal(size(q,1),4)
            @test isequal(length(q),4)

            @test isapprox(norm(q),1.0,rtol = 1e-6)
        end
    end
end

@testset "qshorter" begin
    let
        for i = 1:100
            q = randq()
            qs = q_shorter(q)

            # rotation between the two
            b1_q_b2 = qconj(q) ⊙ qs

            # axis angle between the two
            ϕ = phi_from_q(b1_q_b2)

            @test isapprox(norm(ϕ),0.0,rtol = 1e-6)


        end
    end
end

@testset "q <-> DCM" begin
    let
        for i = 1:1000

            q = randq()
            Q = dcm_from_q(q)
            q2 = q_from_dcm(Q)

            q = q_shorter(q)
            q2 = q_shorter(q2)

            @test isapprox(q,q2,rtol = 1e-6)


        end
    end
end

@testset "pure quaternion" begin
    let
        for i = 1:1000

            x = randn(3)

            x_quaternion = H_mat()*x

            @test isapprox(x,x_quaternion[1:3],rtol = 1e-6)

            @test isapprox(x_quaternion[1:3],H_mat()'*x_quaternion,rtol = 1e-6)

        end
    end
end


@testset "axis angle rotations" begin
    let
        for i = 1:1000

            ϕ = randn(3)

            x_old = randn(3)

            q = q_from_phi(ϕ)
            Q = dcm_from_phi(ϕ)

            x_new_expm = exp(skew_from_vec(ϕ))*x_old

            x_new_dcm = Q*x_old

            x_new_quaternion = H()'*(q ⊙ (H()*x_old) ⊙ qconj(q))

            @test isapprox(x_new_expm,x_new_dcm,rtol = 1e-6)

            @test isapprox(x_new_expm,x_new_quaternion,rtol = 1e-6)

        end
    end
end

@testset "shorter quaternion" begin
    let
        for i = 1:1000
            q = randq()
            q = q_shorter(q)

            phi_shorter = phi_from_q(q)
            phi_longer = phi_from_q(-q)

            @test norm(phi_shorter) < norm(phi_longer)
        end
    end
end

@testset "q - p - g" begin
    let
        for i = 1:1000

            q = randq()
            g = g_from_q(q)
            p = p_from_q(q)

            q_g = q_from_g(g)
            q_p = q_from_p(p)

            @test isapprox(q_shorter(q),q_shorter(q_g),rtol = 1e-6)
            @test isapprox(q_shorter(q),q_shorter(q_p),rtol = 1e-6)

        end
    end
end


@testset "DCM conversions" begin
    let
        for i = 1:1000

            x = randn(3)

            q = randq()
            g = g_from_q(q)
            p = p_from_q(q)

            dcm_q = dcm_from_q(q)
            dcm_g = dcm_from_g(g)
            dcm_p = dcm_from_p(p)

            @test matrix_isapprox(dcm_q,dcm_g,1e-7)
            @test matrix_isapprox(dcm_q,dcm_p,1e-7)

            @test isapprox(dcm_q*x,dcm_p*x,rtol = 1e-6)
            @test isapprox(dcm_q*x,dcm_g*x,rtol = 1e-6)

        end
    end
end



@testset "mrp derivative" begin
    let
        ω = deg2rad.([1;2;3])

        tf = 5
        dt = .001
        t_vec = 0:dt:tf
        # integrate the attitudes
        p = zeros(3)
        q = [0;0;0;1]
        for i = 1:length(t_vec)
            p += dt*pdot_from_w(p,ω)
            q += dt*.5*q⊙[ω;0]
        end


        dcm_p = dcm_from_p(p)
        dcm_q = dcm_from_q(q)


        err_phi = phi_from_dcm(dcm_p'*dcm_q)

        @test rad2deg(norm(err_phi)) < 1e-5

    end
end


@testset "phi shorter" begin
    let
        for i = 1:100

            r = normalize(randn(3))
            θ = rand_in_range(-pi,pi) + 2*pi
            phi = θ*r
            phi_shorter1 = phi_shorter(phi)
            phi_shorter2 = (θ - 2*pi)*r

            @test isapprox(phi_shorter2,phi_shorter1,rtol = 1e-9)

            phi = rand_in_range(0,2*pi)*normalize(randn(3))

            dcm1 = dcm_from_phi(phi)
            dcm2 = dcm_from_phi(phi_shorter(phi))

            n = norm(dcm1-dcm2)

            @test n<1e-10


            q = randq()

            phi1 = phi_shorter(phi_from_q(q))
            phi2 = phi_from_q(q_shorter(q))

            @test isapprox(phi1,phi2,rtol = 1e-9)
        end
    end
end

@testset "phi from p" begin
    let
        for i = 1:100
            q = randq()
            p = p_from_q(q)
            phi1 = phi_shorter(phi_from_q(q))
            phi2 = phi_shorter(phi_from_p(p))

            @test isapprox(phi1,phi2,rtol=1e-9)

        end
    end
end

@testset "p from phi" begin
    let
        for i = 1:100
            q = q_shorter(randq())
            phi = phi_from_q(q)
            p1 = (p_from_q(q))
            p2 = (p_from_phi(phi_shorter(phi)))

            @test isapprox(p1,p2,rtol=1e-9)

        end
    end
end

@testset "hasnan" begin
    let
        M = randn(100,100)
        idx = Int(round(rand_in_range(1,100)))
        idy = Int(round(rand_in_range(1,100)))
        idx2 = Int(round(rand_in_range(1,100)))
        idy2 = Int(round(rand_in_range(1,100)))
        M[idx,idy] = NaN
        M[idx2,idy2] = NaN

        @test hasnan(M)

    end
end

@testset "clamp3d" begin
    let
        for i = 1:100

            max = abs.(randn(3))

            x = randn(3)

            xc = clamp3d(max,x)

            for ii = 1:3
                @test xc[ii] <= max[ii]
                @test xc[ii] >= -max[ii]
            end

        end
    end
end

@testset "vec from mat" begin
    let

        i = 3
        j = 10
        X = randn(i,j)

        Xv = vec_from_mat(X)

        for ii = 1:i
            for jj = 1:j
                @test isapprox(X[ii,jj],Xv[jj][ii],rtol = 1e-9)
            end
        end

        X2 = mat_from_vec(Xv)

        @test norm(X2-X) < 1e-9
    end
end

@testset "mat from vec" begin
    let
        for i = 1:100
            X = randn(30,50)

            Xv = vec_from_mat(X)
            X2 = mat_from_vec(Xv)

            @test norm(X2-X) < 1e-9
        end
    end
end

@testset "interp1" begin
    let
        N = 30
        B_save = [[i;2*i;3*i] for i = 0:N-1]
        t = 0:1.0:(N-1)*1.0

        # check at 1.5
        input_t = 1.5

        expected = [1.5;3;4.5]

        actual = interp1(t,B_save,input_t)

        @test isapprox(expected,actual,rtol = 1e-6)


        for i = 1:N
            input_t = t[i]
            expected = B_save[i]
            actual = interp1(t,B_save,input_t)
            @test isapprox(expected,actual,atol = 1e-9)
        end

        # check at the start
        # input_t = 0.0
        # expected = zeros(3)
        #
        # actual = interp1(t,B_save,input_t)
        #
        # @test isapprox(expected,actual,atol = 1e-9)
        #
        # # check at the beginning
        # input_t = t[1]
        # expected = B_save[1]
        #
        # actual = interp1(t,B_save,input_t)
        #
        # @test isapprox(expected,actual,rtol = 1e-6)
        #
        # # check in the middle
        # input_t = t[10]
        # expected = B_save[10]
        #
        # actual = interp1(t,B_save,input_t)
        #
        # @test isapprox(expected,actual,rtol = 1e-6)
        # # check at the end
        # input_t = t[end]
        # expected = B_save[end]
        #
        # actual = interp1(t,B_save,input_t)
        #
        # @test isapprox(expected,actual,rtol = 1e-6)

    end
end
