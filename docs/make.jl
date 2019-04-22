using Documenter, JetPackWaveFD

makedocs(
    sitename="JetPackWaveFD.jl",
    modules=[Jets],
    pages = [ "index.md", "reference.md" ]
)

deploydocs(
    repo = "github.com/ChevronETC/JetPackWaveFD.jl.git"
)
