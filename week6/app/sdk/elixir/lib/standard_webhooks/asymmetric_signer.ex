defmodule StandardWebhooks.AsymmetricSigner do
  @moduledoc """
  Sign Standard Webhooks with the asymmetric `v2a` signature scheme, for producers that hold the
  private key themselves.

      signer = StandardWebhooks.AsymmetricSigner.new(secret_key)
      StandardWebhooks.AsymmetricSigner.sign(endpoint_id, msg_id, timestamp, payload, signer)

  It's recommended to hold the private key in a KMS or an HSM instead, and sign using
  `StandardWebhooks.Asymmetric.signed_content_hash/4` and
  `StandardWebhooks.Asymmetric.format_signature/1`.

  A key pair isn't tied to an endpoint: one key pair can serve all endpoints, as the endpoint ID
  is part of the signed content.
  """

  alias StandardWebhooks.Asymmetric

  @secret_key_prefix "whsk_"

  @derive {Inspect, only: [:public_key]}
  @enforce_keys [:seed, :public_key]
  defstruct [:seed, :public_key]

  @type t :: %__MODULE__{seed: binary(), public_key: binary()}

  @doc """
  Create a signer given a secret key

  `secret_key` is a `whsk_` prefixed, base64 encoded 32 byte ed25519 private key (seed).
  """
  @spec new(secret_key :: String.t()) :: t()
  def new(@secret_key_prefix <> secret_key) do
    case Base.decode64(secret_key) do
      {:ok, <<_::binary-size(32)>> = seed} ->
        {public_key, _private_key} = :crypto.generate_key(:eddsa, :ed25519, seed)
        %__MODULE__{seed: seed, public_key: public_key}

      _ ->
        raise ArgumentError, message: "Secret key must be a base64 encoded 32 byte key"
    end
  end

  def new(_secret_key) do
    raise ArgumentError, message: "Secret key must start with \"#{@secret_key_prefix}\""
  end

  @doc """
  Sign a webhook given an endpoint ID, message ID, timestamp, payload and signer created with
  `new/1`

  Takes the same arguments as `StandardWebhooks.Asymmetric.signed_content_hash/4`, and returns
  the `webhook-signature` header value (`v2a,<base64 signature>`).
  """
  @spec sign(
          endpoint_id :: String.t(),
          msg_id :: String.t(),
          timestamp :: integer(),
          payload :: binary() | String.t(),
          signer :: t()
        ) :: String.t()
  def sign(endpoint_id, msg_id, timestamp, payload, %__MODULE__{} = signer) do
    digest = Asymmetric.signed_content_hash(endpoint_id, msg_id, timestamp, payload)

    :crypto.sign(:eddsa, :none, digest, [signer.seed, :ed25519])
    |> Asymmetric.format_signature()
  end

  @doc """
  Returns the `whpk_` prefixed public key to share with consumers
  """
  @spec public_key(signer :: t()) :: String.t()
  def public_key(%__MODULE__{} = signer) do
    Asymmetric.format_public_key(signer.public_key)
  end
end
