# frozen_string_literal: true

require "json"
require "openssl"
require "base64"
require "uri"

module StandardWebhooks
  class Webhook
    include TimestampVerification

    def self.new_using_raw_bytes(secret)
      new(secret.pack("C*").force_encoding("UTF-8"))
    end

    def initialize(secret)
      if secret.start_with?(SECRET_PREFIX)
        secret = secret[SECRET_PREFIX.length..-1]
      end

      @secret = Base64.decode64(secret)

      if @secret.empty?
        raise EmptyWebhookSecretError, "Webhook secret must not be blank"
      end
    end

    def verify(payload, headers)
      msg_id = headers["webhook-id"]
      msg_signature = headers["webhook-signature"]
      msg_timestamp = headers["webhook-timestamp"]

      if !msg_signature || !msg_id || !msg_timestamp
        raise WebhookVerificationError, "Missing required headers"
      end

      verify_timestamp(msg_timestamp)

      signature = sign(msg_id, msg_timestamp, payload)

      passed_signatures = msg_signature.split(" ")
      matched = passed_signatures.any? do |versioned_signature|
        ::StandardWebhooks.secure_compare(versioned_signature, signature)
      end

      unless matched
        raise WebhookVerificationError, "No matching signature found"
      end

      JSON.parse(payload)
    end

    def sign(msg_id, timestamp, payload)
      begin
        timestamp = Integer(timestamp)
      rescue
        raise WebhookSigningError, "Invalid timestamp"
      end

      to_sign = "#{msg_id}.#{timestamp}.#{payload}"
      signature = Base64.encode64(OpenSSL::HMAC.digest(OpenSSL::Digest.new("sha256"), @secret, to_sign)).strip

      "v1,#{signature}"
    end

    # Calculate the response to an endpoint verification challenge. Takes the value of the
    # `webhook-challenge` request header and returns the value for the
    # `webhook-challenge-response` response header.
    def endpoint_verification_response(challenge)
      verification_key = OpenSSL::HMAC.digest(OpenSSL::Digest.new("sha256"), @secret, "wev-v1")
      digest = OpenSSL::HMAC.digest(OpenSSL::Digest.new("sha256"), verification_key, challenge)
      signature = Base64.encode64(digest).strip

      "v1,#{signature}"
    end

    # Verify the response to an endpoint verification challenge. Takes the challenge sent in
    # the `webhook-challenge` request header and the value of the `webhook-challenge-response`
    # response header. Raises WebhookVerificationError if no matching signature is found.
    def verify_endpoint_verification_response(challenge, header)
      signature = endpoint_verification_response(challenge)

      unless ::StandardWebhooks.secure_compare(header, signature)
        raise WebhookVerificationError, "No matching signature found"
      end
    end

    SECRET_PREFIX = "whsec_"
    private_constant :SECRET_PREFIX
  end
end
