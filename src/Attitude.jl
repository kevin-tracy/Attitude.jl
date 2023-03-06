__precompile__(true)
module Attitude

using LinearAlgebra
using StaticArrays
# greet() = print("Hello World!")

# get the types
include(joinpath(dirname(@__FILE__),"types.jl"))


export wrap_to_2pi

export cfill
function cfill(nx,N)
    """clean version of fill without any references"""
    return [zeros(nx) for i = 1:N]
end
function cfill(nu,nx,N)
    """clena version of fill without any references for vec of mats"""
    return [zeros(nu,nx) for i = 1:N]
end
function wrap_to_2pi(theta::Real)::Real
    """Takes an angle theta and returns the same angle theta ∈ [0,2π].

    Args:
        theta: angle in radians :: Float64

    Returns:
        theta: angle in radians wrapped :: Float64
    """

    # if angle is negative
    if theta < 0.0
        theta = -2*pi*(abs(theta)/(2*pi)-floor(abs(theta/(2*pi)))) + 2*pi

    # if angle is positive
    else
        theta = 2*pi*(abs(theta)/(2*pi)-floor(abs(theta/(2*pi))))
    end

    return theta
end

export wrap_to_pm_pi

function wrap_to_pm_pi(theta::Real)::Real
    """Takes an angle theta and returns the same angle theta ∈ [-π,π].

    Args:
        theta: angle in radians :: Float64

    Returns:
        theta: angle in radians wrapped :: Float64

    Comments:
        'pm' in the function title stands for 'plus minus', or ±
    """

    # wrap the angle to 2π
    theta = wrap_to_2pi(theta)

    # if the angle is > pi, subtract 2π s.t. it's between ±π
    if theta>pi
        theta -= 2*pi
    end

    return theta
end

export skew_from_vec

function skew_from_vec(v::Vec)::Mat
    """Skew symmetric matrix from a vector.

    Summary:
        Takes 3 component vector and returns matrix after skew_from_vec
        operator has been applied. This is the same as the cross product
        matrix. cross(a,b) = skew_from_vec(a)*b

    Args:
        v::Vector

    Returns:
        Skew symmetric matrix from the given vector :: AbstractArray{Float64,2}
    """

    # v = float(vec(v))
    return @SArray [0 -v[3] v[2]; v[3] 0 -v[1]; -v[2] v[1] 0]
end

export hat

function hat(v::Vec)::Mat
    return skew_from_vec(v)
end

export vec_from_skew

function vec_from_skew(mat::Mat)::Vec
    """Converts 3x3 skew symmetric matrix to a 3x1 vector.

    Args:
        mat: 3x3 skew symmetric matrix

    Returns:
        3x1 vector
    """

    return SVector(mat[3, 2], mat[1, 3], mat[2, 1])
end


function product_of_diagonals(Q::Mat)::Float64
    sum = 1.0
    for i = 1:size(Q,1)
        sum *= Q[i,i]
    end
    return sum
end

export mvnrnd

function mvnrnd(μ::Vec,Σ::Mat)::Vec

    if product_of_diagonals(Q) != 0.0
        return Matrix(cholesky(Σ))*randn(length(μ))+vec(μ)
    else
        return sqrt(Σ)*randn(length(μ))+vec(μ)
    end

end

export unhat

function unhat(mat::Mat)::Vec
    return vec_from_skew(mat)
end

export dcm_from_phi

function dcm_from_phi(phi::Vec)::Mat
    """DCM from axis angle (phi)"""
    return skew_expm(skew_from_vec(phi))
end

export skew_expm

function skew_expm(B::Mat)::Mat
    """Expm for skew symmetric matrices.

    Summary:
        matrix exponential for skew symmetric matrices. about 40% faster than
        the standard exp.jl function. This function can be used to take the
        skew symmetric skew_from_vec matrix of an axis angle vector, and
        producing the orthogonal Direction Cosine Matrix (DCM)
        skew_from_vec corresponds.

    Args:
        B: skew symmetric matrix :: AbstractArray{Float64,2}

    Returns:
        orthognal matrix :: AbstractArray{Float64,2}
    """

    # axis (from axis-angle)
    phi = vec_from_skew(B)

    # angle (from axis-angle)
    theta = norm(phi)

    # axis
    if theta == 0
        r = SVector(0.0, 0.0, 0.0)
    else
        r = phi / theta
    end

    # closed form skew symmetric matrix exponential
    return (I + sin(theta) * skew_from_vec(r) + (1.0 - cos(theta)) *
    skew_from_vec(r) * skew_from_vec(r))
end

export ortho_logm

function ortho_logm(Q::Mat)::Mat
    """Matrix logarithm for 3x3 orthogonal matrices (like DCM's).

    Summary:
        This is both faster and more robust than log.jl (180 degree rotations)

    Args:
        Q: orthogonal matrix (like a DCM) :: AbstractArray{Float64,2}

    Returns:
        skew symmetric matrix :: AbstractArray{Float64,2}
    """

    val = (tr(Q) - 1) / 2

    if abs(val - 1) < 1e-10
        # no rotation
        phi = [0.0; 0.0; 0.0]
    elseif abs(val + 1) < 1e-10
        # 180 degree rotation
        M = I + Q
        r = M[1, :] / norm(M[1, :])
        theta = pi
        phi = r * theta
    else
        # normal rotation (0-180 degrees)
        theta = acos(val)
        r = -(1 / (2 * sin(theta))) *
            [Q[2, 3] - Q[3, 2]; Q[3, 1] - Q[1, 3]; Q[1, 2] - Q[2, 1]]
        phi = r * theta
    end

    return skew_from_vec(phi)
end

export H_mat

function H_mat()::Mat
    """matrix for converting vector to pure quaternion. Scalar first"""
    return @SArray [0 0 0;1 0 0;
            0 1 0;
            0 0 1]
end

export phi_from_dcm

function phi_from_dcm(Q::Mat)::Vec
    # TODO: test this
    """Matrix logarithm for 3x3 orthogonal matrices (like DCM's).

    Summary:
        This is both faster and more robust than log.jl (180 degree rotations)

    Args:
        Q: orthogonal matrix (like a DCM) :: AbstractArray{Float64,2}

    Returns:
        skew symmetric matrix :: AbstractArray{Float64,2}
    """

    val = (tr(Q) - 1) / 2

    if abs(val - 1) < 1e-10
        # no rotation
        phi = SVector(0.0, 0.0, 0.0)
    elseif abs(val + 1) < 1e-10
        # 180 degree rotation
        M = I + Q
        r = M[1, :] / norm(M[1, :])
        theta = pi
        phi = r * theta
    else
        # normal rotation (0-180 degrees)
        theta = acos(val)
        r = -(1 / (2 * sin(theta))) *
            SVector(Q[2, 3] - Q[3, 2], Q[3, 1] - Q[1, 3], Q[1, 2] - Q[2, 1])
        phi = r * theta
    end

    return phi
end

export rand_in_range

function rand_in_range(lower::Real, upper::Real)::Real
    """Random number within range with a uniform distribution.

    Args:
        lower: lower value in range
        upper: upper value in range

    Returns:
        random number in the specified range
    """

    delta = upper - lower
    return lower + rand() * delta
end

export active_rotation_axis_angle

function active_rotation_axis_angle(axis::Vec,theta::Real,vector::Vec)::Vec
    """Actively rotate a vector using an axis and angle.

    Args:
        axis: axis of rotation (unit norm)
        theta: angle to rotate (rad)

    Returns:
        rotated vector (unit norm)
    """

    # check if axis is unit norm
    if !isapprox(norm(axis),1.0,rtol=1e-6)
        error("axis is not unit norm")
    end

    return skew_expm(skew_from_vec(theta*axis))*vector
end

export ⊙

function ⊙(q1::Vec, q2::Vec)::Vec
    """Quaternion multiplication, hamilton product, scalar last"""

    v1 = @views q1[2:4]
    s1 = q1[1]
    v2 = @views q2[2:4]
    s2 = q2[1]

    return SVector{4}([(s1 * s2 - dot(v1, v2));s1 * v2 + s2 * v1 + cross(v1, v2)])

end

export qdot

function qdot(q1::Vec,q2::Vec)::Vec
    return (q1 ⊙ q2)
end

export dcm_from_q

function dcm_from_q(q::Vec)::Mat
    """DCM from quaternion, hamilton product, scalar last"""

    # pull our the parameters from the quaternion
    # q1,q2,q3,q4 = normalize(q)
    # q = normalize(q)
    # # DCM
    # return @SArray [(2*q[1]^2+2*q[4]^2-1)   2*(q[1]*q[2] - q[3]*q[4])   2*(q[1]*q[3] + q[2]*q[4]);
    #       2*(q[1]*q[2] + q[3]*q[4])  (2*q[2]^2+2*q[4]^2-1)   2*(q[2]*q[3] - q[1]*q[4]);
    #       2*(q[1]*q[3] - q[2]*q[4])   2*(q[2]*q[3] + q[1]*q[4])  (2*q[3]^2+2*q[4]^2-1)]
    # pull our the parameters from the quaternion
    q4,q1,q2,q3 = normalize(q)

    # DCM
    Q = @SArray [(2*q1^2+2*q4^2-1)   2*(q1*q2 - q3*q4)   2*(q1*q3 + q2*q4);
          2*(q1*q2 + q3*q4)  (2*q2^2+2*q4^2-1)   2*(q2*q3 - q1*q4);
          2*(q1*q3 - q2*q4)   2*(q2*q3 + q1*q4)  (2*q3^2+2*q4^2-1)]
end

export qconj

function qconj(q::Vec)::Vec
    """Conjugate of the quaternion (scalar first)"""

    # return [-q[1:3]; q[4]]
    return SVector(q[1],-q[2],-q[3],-q[4])
end

export phi_from_q

function phi_from_q(q::Vec)::Vec
    """axis angle from quaternion (scalar last)"""

    # v = @views q[1:3]
    v = SVector(q[2],q[3],q[4])
    s = q[1]
    normv = norm(v)

    if normv == 0.0
        return zeros(3)
    else
        r = v / normv
        θ = (2 * atan(normv, s))
        return r * θ
    end
end

export q_from_phi

function q_from_phi(ϕ::Vec)::Vec
    """Quaternion from axis angle vector, scalar last"""

    θ = norm(ϕ)
    if abs(θ) < 0.0000000001
        return [1; 0; 0; 0]
    else
        r = ϕ / θ
        return [cos(θ / 2);r * sin(θ / 2)]
    end
end

export q_from_dcm

function q_from_dcm(dcm::Mat)::Vec
    """Kane/Levinson convention, scalar last"""
    R = dcm
    T = R[1,1] + R[2,2] + R[3,3]
    if T> R[1,1] && T > R[2,2] && T>R[3,3]
        q4 = .5*sqrt(1+T)
        r  = .25/q4
        q1 = (R[3,2] - R[2,3])*r
        q2 = (R[1,3] - R[3,1])*r
        q3 = (R[2,1] - R[1,2])*r
    elseif R[1,1]>R[2,2] && R[1,1]>R[3,3]
        q1 = .5*sqrt(1-T + 2*R[1,1])
        r  = .25/q1
        q4 = (R[3,2] - R[2,3])*r
        q2 = (R[1,2] + R[2,1])*r
        q3 = (R[1,3] + R[3,1])*r
    elseif R[2,2]>R[3,3]
        q2 = .5*sqrt(1-T + 2*R[2,2])
        r  = .25/q2
        q4 = (R[1,3] - R[3,1])*r
        q1 = (R[1,2] + R[2,1])*r
        q3 = (R[2,3] + R[3,2])*r
    else
        q3 = .5*sqrt(1-T + 2*R[3,3])
        r  = .25/q3
        q4 = (R[2,1] - R[1,2])*r
        q1 = (R[1,3] + R[3,1])*r
        q2 = (R[2,3] + R[3,2])*r
    end
    q = [q4;q1;q2;q3]
    if q4<0
        q = -q
    end

    return q
end

export randq

function randq()::Vec
    return SVector{4}(normalize(randn(4)))
end

export q_shorter

function q_shorter(q::Vec)::Vec

    if q[1]<0
        q = -q
    end
    return q
end

export g_from_q

function g_from_q(q::Vec)::Vec
    """Rodgrigues parameter from quaternion (scalar last)"""
    return q[2:4]/q[1]
end

export q_from_g

function q_from_g(g::Vec)::Vec
    """Quaternion (scalar last) from Rodrigues parameter"""
    return (1/sqrt(1+dot(g,g)))*[1;g]
end

export dcm_from_g

function dcm_from_g(g::Vec)::Mat
    """DCM form Rodrigues parameter"""
    return I +  2*(skew_from_vec(g)^2 + skew_from_vec(g))/(1 + dot(g,g))
end

export p_from_q

function p_from_q(q::Vec)::Vec
    """MRP from quaternion (scalar first)"""
    return q[2:4]/(1+q[1])
end

export q_from_p

function q_from_p(p::Vec)::Vec
    """Quaternion (scalar first) from MRP"""
    return (1/(1+dot(p,p)))*[(1-dot(p,p));2*p]
end

export dcm_from_p

function dcm_from_p(p::Vec)::Mat
    """DCM from MRP"""
    sp = skew_from_vec(p)
    return I + (8*sp^2 + 4*(1 - dot(p,p))*sp)/(1 + dot(p,p))^2
end

export pdot_from_w

function pdot_from_w(p::Vec,w::Vec)::Vec
    """kinematics of the modified rodrigues parameter assuming that
    attitude is being denoted as N_R_B using the kane/levinson convention

    Arguments:
        p: ᴺpᴮ, MRP, Kane/Levinson convention
        w: ᴺωᴮ expressed in B

    Returns:
        ᴺṗᴮ
    """

    return ((1+norm(p)^2)/4) *(   I + 2*(hat(p)^2 + hat(p))/(1+norm(p)^2)   )*w

end

export p_from_phi

function p_from_phi(phi::Vec)::Vec
    """rodrigues parameter from axis angle"""
    q = q_from_phi(phi)
    p = p_from_q(q)

    return p
end

export phi_from_p

function phi_from_p(p::Vec)::Vec
    """axis angle from rodrigues parameter"""
    q = q_from_p(p)
    phi = phi_from_q(q)

    return phi
end

export vec_from_mat

function vec_from_mat(mat::Mat)
    #vector of vectors from matrix of column vectors

    s = size(mat)
    if length(s) == 3
        a,b,c = size(mat)

        V = fill(zeros(a,b),c)

        for i = 1:c
            V[i] = mat[:,:,i]
        end
    else
        a,b = size(mat)

        V = fill(zeros(a),b)

        for i = 1:b
            V[i] = mat[:,i]
        end
    end


    return V
end

export mat_from_vec

function mat_from_vec(a::Vector)
    "Turn a vector of vectors into a matrix"


    rows = length(a[1])
    columns = length(a)
    A = zeros(rows,columns)

    for i = 1:columns
        A[:,i] = a[i]
    end

    return A
end

export clamp3d

function clamp3d(max_moments::Vec,m::Vec)
    #3d vector clamp function

    if minimum(max_moments)<0
        error("max moments has negative in it")
    end

    m_out = zeros(3)

    for i = 1:3
        m_out[i] = clamp(m[i],-max_moments[i],max_moments[i])
    end

    return m_out
end

export hasnan

function hasnan(mat)
    #check if matrix has NaN present

    if norm(isnan.(vec(mat)))>0.0
        return true
    else
        return false
    end
end

export phi_shorter

function phi_shorter(phi)
    #axis angle for angles less than ±π
    θ = norm(phi)
    r = phi/θ

    return r*wrap_to_pm_pi(θ)
end

# tests
export run_all_attitude_tests

function run_all_attitude_tests()
    include(joinpath(dirname(@__DIR__),"test/attitude_fx_tests.jl"))
end

export Lmat

function Lmat(q)
    qs = q[1]
    qv = SVector(q[2],q[3],q[4])
    return [qs  -qv'; qv (qs*I + skew_from_vec(qv))]
end

export Rmat

function Rmat(q)
    qs = q[1]
    qv = SVector(q[2],q[3],q[4])
    return [qs  -qv'; qv (qs*I - skew_from_vec(qv))]
end


export Gmat

function Gmat(q)
    """Quaternion to rodrigues parameter Jacobian"""
    s = q[1]
    v = SVector(q[2],q[3],q[4])
    return [-v'; (s*I + skew_from_vec(v))]
end

end # module
