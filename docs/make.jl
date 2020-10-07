using Documenter, JetPackWaveFD

makedocs(
    sitename="JetPackWaveFD.jl",
    modules=[JetPackWaveFD],
    pages = [ "index.md", "reference.md" ]
)

deploydocs(
    repo = "github.com/ChevronETC/JetPackWaveFD.jl.git"
)
