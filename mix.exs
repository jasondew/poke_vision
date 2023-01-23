defmodule PokeVision.MixProject do
  use Mix.Project

  def project do
    [
      app: :poke_vision,
      version: "0.1.0",
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger],
      mod: {PokeVision.Application, []}
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:axon, "~> 0.3.0"},
      {:exla, "~> 0.4.1", runtime: false},
      {:finch, "~> 0.14"},
      {:image, "~> 0.22.1"},
      {:nimble_csv, "~> 1.1"},
      {:nx, "~> 0.4.1"},
      {:nx_image, "~> 0.1"},
      {:table_rex, "~> 3.1.1", only: :dev},
      {:vix, github: "akash-akya/vix", branch: "dev", override: true}
    ]
  end
end
