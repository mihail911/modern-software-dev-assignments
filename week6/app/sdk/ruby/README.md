Ruby library for Standard Webhooks

# Installation

Add `standardwebhooks` to your `Gemfile` and run `bundle install`.

Or install it without bundler:

```sh
gem install standardwebhooks
```

# Verifying webhooks

`StandardWebhooks::Webhook#verify` expects:

- The exact raw request body as a string.
- A hash containing `webhook-id`, `webhook-timestamp`, and `webhook-signature`.

It returns the parsed JSON payload as a Ruby hash and raises
`StandardWebhooks::WebhookVerificationError` when verification fails.

```ruby
require 'standardwebhooks'

webhook = StandardWebhooks::Webhook.new(ENV.fetch('WEBHOOK_SECRET'))

payload = webhook.verify(raw_body, {
	'webhook-id' => request_headers.fetch('webhook-id'),
	'webhook-timestamp' => request_headers.fetch('webhook-timestamp'),
	'webhook-signature' => request_headers.fetch('webhook-signature'),
})

puts payload['type']
```

Pass the raw body exactly as received. Verifying a parsed-and-reserialized JSON body
can fail because the signature is computed over the original bytes.

## Responding to endpoint verification

To support the [endpoint verification](../../spec/extensions/endpoint-verification.md) extension,
use `#endpoint_verification_response` when the request includes a `webhook-challenge` header:

```ruby
challenge = request_headers['webhook-challenge']
if challenge
	response_headers['webhook-challenge-response'] = webhook.endpoint_verification_response(challenge)
	# Not a webhook delivery: respond (e.g. with 204 No Content) without processing the body
	return
end
```

Producers can verify the response with `#verify_endpoint_verification_response`, which raises
`StandardWebhooks::WebhookVerificationError` if it doesn't match:

```ruby
webhook.verify_endpoint_verification_response(challenge, response_headers['webhook-challenge-response'])
```

## Asymmetric signatures

Webhooks signed with asymmetric (`v2a`) signatures are verified with `StandardWebhooks::WebhookAsymmetric`,
which takes the endpoint ID and the list of trusted public keys (`whpk_...`) the producer shared with you.
A webhook is accepted if it's signed by any of the keys, so keys can be rotated by adding the new key to the list.
Put the newest key first, as endpoint verification responses are calculated with the first key.

```ruby
webhook = StandardWebhooks::WebhookAsymmetric.new(
	ENV.fetch('WEBHOOK_ENDPOINT_ID'),
	[ENV.fetch('WEBHOOK_PUBLIC_KEY')],
)

payload = webhook.verify(raw_body, {
	'webhook-id' => request_headers.fetch('webhook-id'),
	'webhook-timestamp' => request_headers.fetch('webhook-timestamp'),
	'webhook-signature' => request_headers.fetch('webhook-signature'),
})
```

`WebhookAsymmetric` also supports `#endpoint_verification_response` and `#verify_endpoint_verification_response`.

### Signing (asymmetric)

It's recommended to generate and hold the private key in a KMS or an HSM, and never export it.
In that case, use `WebhookAsymmetric.signed_content_hash` to get the 32-byte SHA-256 hash to sign,
sign it (pure ed25519) with your HSM, and format the 64-byte signature with `WebhookAsymmetric.format_signature`:

```ruby
digest = StandardWebhooks::WebhookAsymmetric.signed_content_hash(endpoint_id, msg_id, timestamp, payload)
signature = my_hsm.ed25519_sign(digest) # Your HSM / KMS sign call
headers['webhook-signature'] = StandardWebhooks::WebhookAsymmetric.format_signature(signature)

# The `whpk_...` public key to share with consumers. Accepts the raw 32-byte key or its DER SubjectPublicKeyInfo.
public_key = StandardWebhooks::WebhookAsymmetric.format_public_key(my_hsm.ed25519_public_key)
```

If you hold the private key yourself (`whsk_...`), use `StandardWebhooks::WebhookAsymmetricSigner`.
One key pair can serve all of your endpoints, as the endpoint ID is part of the signed content:

```ruby
signer = StandardWebhooks::WebhookAsymmetricSigner.new(ENV.fetch('WEBHOOK_SECRET_KEY'))

# Share with consumers, together with the endpoint ID
public_key = signer.public_key # "whpk_..."

headers['webhook-signature'] = signer.sign(endpoint_id, msg_id, Time.now.to_i, payload) # "v2a,..."
```

## Rails example

```ruby
class WebhooksController < ApplicationController
	def create
		webhook = StandardWebhooks::Webhook.new(ENV.fetch('WEBHOOK_SECRET'))

		payload = webhook.verify(request.raw_post, {
			'webhook-id' => request.headers.fetch('webhook-id'),
			'webhook-timestamp' => request.headers.fetch('webhook-timestamp'),
			'webhook-signature' => request.headers.fetch('webhook-signature'),
		})

		process_event(payload)

		head :ok
	rescue StandardWebhooks::WebhookVerificationError => error
		Rails.logger.warn("Invalid webhook: #{error.message}")

		head :bad_request
	end

	private

	def process_event(payload)
		case payload['type']
		when 'customer.created'
			CustomerCreateJob.perform_later(payload.fetch('data'))
		end
	end
end
```

## Rack or Sinatra example

```ruby
post '/webhooks' do
	request.body.rewind
	raw_body = request.body.read

	webhook = StandardWebhooks::Webhook.new(ENV.fetch('WEBHOOK_SECRET'))

	payload = webhook.verify(raw_body, {
		'webhook-id' => request.env.fetch('HTTP_WEBHOOK_ID'),
		'webhook-timestamp' => request.env.fetch('HTTP_WEBHOOK_TIMESTAMP'),
		'webhook-signature' => request.env.fetch('HTTP_WEBHOOK_SIGNATURE'),
	})

	handle_event(payload)

	status 204
rescue StandardWebhooks::WebhookVerificationError
	halt 400
end
```

# Signing payloads

If you are producing Standard Webhooks or need deterministic signatures in tests,
use `#sign`:

```ruby
require 'standardwebhooks'

webhook = StandardWebhooks::Webhook.new(ENV.fetch('WEBHOOK_SECRET'))

msg_id = 'msg_2KWPBgLlAfxdpx2AI54pPJ85f4W'
timestamp = Time.now.to_i
payload = '{"type":"example.event","timestamp":"2022-11-03T20:26:10.344522Z","data":{"foo":"bar"}}'

signature = webhook.sign(msg_id, timestamp, payload)

headers = {
	"webhook-id" => msg_id,
	"webhook-timestamp" => timestamp.to_s,
	"webhook-signature" => signature,
}

# …
```

# Development

## Building

```sh
bundler exec rake build
```

## Running tests

```sh
bundle exec rspec spec
```
