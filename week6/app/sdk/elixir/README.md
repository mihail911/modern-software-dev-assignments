Elixir library for Standard Webhooks

# Example

Verifying a webhook payload:

```elixir
StandardWebhooks.verify(webhook_payload, plug_conn, webhook_secret)
```

Note: [Plug](https://hexdocs.pm/plug/Plug.Conn.html#t:headers/0) contains the request headers.

The payload must be the raw request body, exactly as received, as bytes or a string. Don't pass the parsed and re-encoded JSON, as even minor differences in serialization will make verification fail. Since `Plug.Parsers` consumes the body, keep a copy of the raw body using a custom [`:body_reader`](https://hexdocs.pm/plug/Plug.Parsers.html#module-custom-body-reader), for example:

```elixir
defmodule MyAppWeb.CacheBodyReader do
  def read_body(conn, opts) do
    {:ok, body, conn} = Plug.Conn.read_body(conn, opts)
    conn = Plug.Conn.assign(conn, :raw_body, body)
    {:ok, body, conn}
  end
end

# In your endpoint
plug Plug.Parsers,
  parsers: [:json],
  body_reader: {MyAppWeb.CacheBodyReader, :read_body, []},
  json_decoder: Jason

# Then
StandardWebhooks.verify(conn.assigns.raw_body, conn, webhook_secret)
```

Signing a webhook (the payload is signed byte for byte, so pass the exact body that will be sent):

```elixir
StandardWebhooks.sign(webhook_id, webhook_timestamp, webhook_payload, webhook_secret)
```

Responding to [endpoint verification](../../spec/extensions/endpoint-verification.md):

```elixir
case Plug.Conn.get_req_header(conn, "webhook-challenge") do
  [challenge] ->
    response = StandardWebhooks.endpoint_verification_response(challenge, webhook_secret)

    # Not a webhook delivery: respond without processing the body
    conn
    |> Plug.Conn.put_resp_header("webhook-challenge-response", response)
    |> Plug.Conn.send_resp(204, "")
    |> Plug.Conn.halt()

  _ ->
    # Not a challenge: verify and process the webhook as usual
    handle_webhook(conn)
end
```

Producers can verify the `webhook-challenge-response` header with `StandardWebhooks.verify_endpoint_verification_response(challenge, header, webhook_secret)`.

## Asymmetric signatures

Verifying a webhook signed with asymmetric (`v2a`) signatures, using the list of trusted public keys (`whpk_...`) and the endpoint ID the producer shared with you:

```elixir
webhook = StandardWebhooks.Asymmetric.new(webhook_endpoint_id, [webhook_public_key])
StandardWebhooks.Asymmetric.verify(webhook_payload, plug_conn, webhook)
```

As with `StandardWebhooks.verify/3`, the payload must be the raw request body (bytes or a string). A webhook is accepted if it's signed by any of the keys, so keys can be rotated by adding the new key to the list. Put the newest key first, as `StandardWebhooks.Asymmetric.endpoint_verification_response/2` uses the first key.

### Signing (asymmetric)

It's recommended to generate and hold the private key in a KMS or an HSM, and never export it. In that case, use `StandardWebhooks.Asymmetric.signed_content_hash/4` to get the 32 byte SHA-256 hash to sign, sign it (pure ed25519) with your HSM, and format the 64 byte signature with `StandardWebhooks.Asymmetric.format_signature/1`:

```elixir
hash = StandardWebhooks.Asymmetric.signed_content_hash(endpoint_id, webhook_id, webhook_timestamp, webhook_payload)
signature = MyHsm.ed25519_sign(hash) # Your HSM / KMS sign call
StandardWebhooks.Asymmetric.format_signature(signature) # "v2a,..."

# The public key to share with consumers. Accepts the raw 32 byte key or its DER SubjectPublicKeyInfo.
StandardWebhooks.Asymmetric.format_public_key(MyHsm.ed25519_public_key()) # "whpk_..."
```

If you hold the private key yourself (`whsk_...`), use `StandardWebhooks.AsymmetricSigner`. One key pair can serve all of your endpoints, as the endpoint ID is part of the signed content:

```elixir
signer = StandardWebhooks.AsymmetricSigner.new(webhook_secret_key)

# Share with consumers, together with the endpoint ID
StandardWebhooks.AsymmetricSigner.public_key(signer) # "whpk_..."

StandardWebhooks.AsymmetricSigner.sign(endpoint_id, webhook_id, webhook_timestamp, webhook_payload, signer) # "v2a,..."
```

# Development

## Installation

The package can be installed by adding `standard_webhooks` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    # ... your other deps
    {:standard_webhooks, "~> 0.1.0"}
  ]
end
```

Then run:

```sh
mix deps.get
```

## Elixir Console

```sh
iex -S mix
```

## Running Tests

Simply run:

```sh
mix test
```

## Code Quality

```sh
mix quality
```