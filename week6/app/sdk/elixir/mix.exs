defmodule StandardWebhooks.MixProject do
  use Mix.Project

  def project do
    [
      app: :standard_webhooks,
      version: "1.3.0",
      elixir: "~> 1.15",
      start_permanent: Mix.env() == :prod,
      description: description(),
      package: package(),
      deps: deps(),
      aliases: aliases(),
      source_url:
        "https://github.com/standard-webhooks/standard-webhooks/tree/main/libraries/elixir",
      test_coverage: [tool: ExCoveralls]
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:plug, "~> 1.15"},

      # Dev
      {:ex_doc, "~> 0.34", only: :dev, runtime: false},

      # Test
      {:sobelow, "~> 0.12", only: [:dev, :test], runtime: false},
      {:credo, "~> 1.6", only: [:dev, :test], runtime: false},
      {:excoveralls, "~> 0.10", only: [:dev, :test], runtime: false}
    ]
  end

  defp description() do
    "Elixir library for creating and verifying Standard Webhooks signatures."
  end

  defp package() do
    [
      files: ~w(lib .formatter.exs mix.exs README.md LICENSE),
      licenses: ["Apache-2.0"],
      links: %{
        "GitHub" =>
          "https://github.com/standard-webhooks/standard-webhooks/tree/main/libraries/elixir"
      }
    ]
  end

  defp aliases do
    [
      # Run tests and check coverage
      test: ["test", "coveralls"],
      # Run to check the quality of your code
      quality: [
        "format --check-formatted",
        "sobelow --config",
        "credo"
      ]
    ]
  end
end
