# JetPackWaveFD

## Space and time discretization
These operators are implemented second order in time and 8th order in space.

## Remarks on attentuation and propagator evolution 
These propagators were originally used in monochromatic full waveform inversion. They incorporate a simple type of *dissipation only* Maxwell body attenuation, that when combined with a per-frequency wavelet estimation can provide a good approximation to visco-acoustic propagation for bandwidth within about an octave of dominant frequency. 

For time domain full waveform inversion with bandwidth inside an octave of the center frequency of the attenuation model, the approximation used here performs well. To accurately model visco-acoustic attenuation with a larger bandwidth, a more sophisticated mechanism will be required: for example multiple filter banks implementing the widely used standard linear solid (SLS) model. 

Note that we are in the process of open sourcing the description of our attenuation implementation. It derives from first principles of Maxwell Bodies, as shown in the Fung reference below. 

## Self-adjoint operators
These propagators are *self-adjoint*, meaning the same equations used for the nonlinear forward operations are also used for the linearized Jacobian forward and linearized Jacobian adjoint operations. 

## Source aperture considerations
This package includes a set of operations `Ginsu` that help in easily handling field seismic experiments. For typical narrow azimuth towed streamer marine field experiments the aperture required for a single source location may be *much* smaller than the entire model. 

`Ginsu` provides methods to easily cut out part of the model that will be the correct size for the modeling aperture associated with individual source and receiver arrays. There are forward methods (`sub` and `sub!`) provided to extract a piece of a large model for use with individual shots. Similarly there are adjoint methods (`super` and `super!`) provided to add the contributions from individuals sources back to a larger model, for example when summing model perturbation contributions over sources for reverse time migration or full waveform inversion. 

These `Ginsu` mechanisms are part of the operation of the `JetPackWaveFD` propagators. 

## Support for simultaneous sources
The `JetPackWaveFD` propagators can take an array of source locations, and an array of per source delay times and wavelets, in order to ease the simulation of simultaneous (or *blended*) sources. See `sz`, `sy`, `sx`, and `st` in the help docs for the operators for more information.

## Compression and serialization
In order to perform the linearized Jacobian operations, interactions with the nonlinear source wavefield are required. We compress and serialize the nonlinear source wavefield during computation, and then deserialize and decompress during the finite difference evolution for the Jacobian forward and adjoint operation. 

We use the package `CvxCompress.jl`, which is built on the C++ library `CvxCompress` for wavelet compression.

## Illumination compensation
`JetPackWaveFD` operators provide the `srcillum` method to compute the source side illumination array, which is typically used as *illumination compensation* in full waveform inversion and reverse time migration. See help docs for `srcillum` for more information.


## See Also
* `WaveFD` single time step modeling implementations wrapping high performance C++ kernels.
* `Source Wavelets` section in the `WaveFD` documentation, discussing a selection of wavelets commonly used in seismic modeling.
* `JetPackWaveDevito` package, implementing these same 6 operators using the Devito domain specific language (https://www.devitoproject.org/).

## References
* Our implementation of self adjoint energy conserving propagators 
    * *Self-adjoint, energy-conserving second-order pseudoacoustic systems for VTI and TTI media for reverse migration and full-waveform inversion* (2016)
    SEG Technical Program Expanded Abstracts
    https://library.seg.org/doi/10.1190/segam2016-13878451.1
    * PDF documents in the `WaveFD` package describing derivations for time update equations and linearization of constant density acoustics: https://github.com/ChevronETC/WaveFD.jl/blob/master/docs/latex_notes.

* Maxwell body reference
    Fung, Y.C., A First Course in Continuum Mechanics, Prentice-Hall, 1977.

* Interpolation in `JetPackWaveFD` for off-grid physical locations 
    * Bilinear: https://en.wikipedia.org/wiki/Bilinear_interpolation
    * Trilinear: https://en.wikipedia.org/wiki/Trilinear_interpolation
    * Sinc: *Arbitrary source and receiver positioning in finite‐difference schemes using Kaiser windowed sinc functions* (2002)
    Graham Hicks, Geophysics, Vol. 67
    https://library.seg.org/doi/10.1190/1.1451454

* Stability condition for temporal discretization in finite difference solution of PDEs
    * https://en.wikipedia.org/wiki/Courant%E2%80%93Friedrichs%E2%80%93Lewy_condition

* Weak elastic anisotropy
    * *Weak elastic anisotropy* (1986)
    Leon Thomsen, Geophysics, Vol. 51
    https://library.seg.org/doi/abs/10.1190/1.1442051
    * *Acoustic approximations for processing in transversely isotropic media* (1998)
    Tariq Alkhalifah
    Geophysics, Vol. 63
    https://library.seg.org/doi/pdf/10.1190/1.1444361




