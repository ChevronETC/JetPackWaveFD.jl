# JetPackWaveFD.jl

| **Documentation** | **Action Statuses** |
|:---:|:---:|
| [![][docs-dev-img]][docs-dev-url] [![][docs-stable-img]][docs-stable-url] | [![][doc-build-status-img]][doc-build-status-url] [![][build-status-img]][build-status-url] [![][code-coverage-img]][code-coverage-results] |

`JetPackWaveFD` is a `Jets.jl` wave propagation operator pack which includes a reference implementation of isotropic and anisotropic second order in time pseudo- visco- acoustic variable density self adjoint operators and related infrastructure. These operators rely on the `WaveFD` package to supply high performance c++ kernels for the propagators.

| Operator | Physical Dimension | Type | Description |
|:---|:---:|:---|:---|
| `JopNlProp2DAcoIsoDenQ_DEO2_FDTD` <br> `JopLnProp2DAcoIsoDenQ_DEO2_FDTD` | 2D | Nonlinear <br> Linearized | Isotropic modeling  |
| `JopNlProp2DAcoVTIDenQ_DEO2_FDTD` <br> `JopLnProp2DAcoVTIDenQ_DEO2_FDTD` | 2D | Nonlinear <br> Linearized | Vertical transverse isotropy modeling |
| `JopNlProp2DAcoTTIDenQ_DEO2_FDTD` <br> `JopLnProp2DAcoTTIDenQ_DEO2_FDTD` | 2D | Nonlinear <br> Linearized | Tilted transverse isotropy modeling |
| `JopNlProp3DAcoIsoDenQ_DEO2_FDTD` <br> `JopLnProp3DAcoIsoDenQ_DEO2_FDTD` | 3D | Nonlinear <br> Linearized | Isotropic  modeling |
| `JopNlProp3DAcoVTIDenQ_DEO2_FDTD` <br> `JopLnProp3DAcoVTIDenQ_DEO2_FDTD` | 3D | Nonlinear <br> Linearized | Vertical transverse isotropy modeling |
| `JopNlProp3DAcoTTIDenQ_DEO2_FDTD` <br> `JopLnProp3DAcoTTIDenQ_DEO2_FDTD` | 3D | Nonlinear <br> Linearized | Tilted transverse isotropy modeling |
| `JopSincRegular` | - | Linear | Regular → regular sinc interpolation |
| `Ginsu` | - | Linear | Interpolate model domains for shot aperture |

## Companion packages in the COFII framework
- Jets - https://github.com/ChevronETC/Jets.jl
- WaveFD - https://github.com/ChevronETC/WaveFD.jl
- DistributedJets - https://github.com/ChevronETC/DistributedJets.jl
- JetPack - https://github.com/ChevronETC/JetPack.jl
- JetPackDSP - https://github.com/ChevronETC/JetPackDSP.jl
- JetPackTransforms - https://github.com/ChevronETC/JetPackTransforms.jl

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://chevronetc.github.io/JetPackWaveFD.jl/dev/

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://ChevronETC.github.io/JetPackWaveFD.jl/stable

[doc-build-status-img]: https://github.com/ChevronETC/JetPackWaveFD.jl/workflows/Documentation/badge.svg
[doc-build-status-url]: https://github.com/ChevronETC/JetPackWaveFD.jl/actions?query=workflow%3ADocumentation

[build-status-img]: https://github.com/ChevronETC/JetPackWaveFD.jl/workflows/Tests/badge.svg
[build-status-url]: https://github.com/ChevronETC/JetPackWaveFD.jl/actions?query=workflow%3A"Tests"

[code-coverage-img]: https://codecov.io/gh/ChevronETC/JetPackWaveFD.jl/branch/master/graph/badge.svg
[code-coverage-results]: https://codecov.io/gh/ChevronETC/JetPackWaveFD.jl
