"""
    JopSincRegular(dom, rng, dtdom, dtrng)

Return a sinc interpolation filter applied to a signal in `dom::JotSpace`, regularly resampling 
from dtdom to dtrng. It is assumed that the forward operation maps from coarse to dense sampling
along the fast dimension of arrays.
"""
# TODO remove constraint number of dimensions == 2
function JopSincRegular(dom::JetSpace{T,2}, rng::JetSpace{T,2}, dtdom, dtrng, nthreads=12) where {T}
	n1dom,n2dom = size(dom)
	n1rng,n2rng = size(rng)
    @assert n1dom < n1rng   # coarse to dense
    @assert n2dom == n2rng  # same size slow dimension
    filters = Wave.interpfilters(T(dtrng), T(dtdom), 0, Wave.LangC(), nthreads)
	JopLn(dom = dom, rng = rng, df! = JopSincRegular_df!, df′! = JopSincRegular_df′!, s = (filters=filters, nthreads=nthreads))
end

export JopSincRegular

function JopSincRegular_df!(d::AbstractArray{T,2}, m::AbstractArray{T,2}; filters, nthreads, kwargs...) where {T}
    Wave.interpadjoint!(filters, d, m) # from coarse to dense
    d
end

function JopSincRegular_df′!(m::AbstractArray{T,2}, d::AbstractArray{T,2}; filters, nthreads, kwargs...) where {T}
    Wave.interpforward!(filters, m, d) # from dense to coarse
    m
end
