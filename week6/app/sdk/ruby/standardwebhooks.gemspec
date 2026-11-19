# frozen_string_literal: true

Gem::Specification.new do |spec|
  spec.name = "standardwebhooks"
  spec.version = "1.3.0"
  spec.authors = ["Standard Webhooks"]
  spec.license = "MIT"
  spec.summary = "Library for creating and verifying webhook signatures for the Standard Webhook spec."

  spec.required_ruby_version = ">= 3.3"

  spec.metadata = {
    "allowed_push_host" => "https://rubygems.org",
    "homepage_uri" => "https://www.standardwebhooks.com/",
    "bug_tracker_uri" => "https://github.com/standard-webhooks/standard-webhooks/issues",
    "source_code_uri" => "https://github.com/standard-webhooks/standard-webhooks/tree/main/libraries/ruby",
    "documentation_uri" => "https://github.com/standard-webhooks/standard-webhooks/blob/main/libraries/ruby/README.md"
  }

  spec.files = Dir["lib/**/*.rb", "README.md"]
  spec.require_paths = ["lib"]

  spec.add_dependency "base64", "~> 0.3.0"

  spec.add_development_dependency "bundler", ">= 2.2.10"
  spec.add_development_dependency "rake", "~> 13.0"
  spec.add_development_dependency "rspec", "~> 3.2"
end
