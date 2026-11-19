# frozen_string_literal: true

require "openssl"
require "base64"

module StandardWebhooks
  # Signs webhooks with the asymmetric `v2a` signature scheme, for producers that hold
  # the private key themselves. It's recommended to hold the private key in a KMS or an
  # HSM instead, and sign using WebhookAsymmetric.signed_content_hash and
  # WebhookAsymmetric.format_signature.
  #
  # A key pair isn't tied to an endpoint: one key pair can serve all endpoints, as the
  # endpoint ID is part of the signed content.
  class WebhookAsymmetricSigner
    # secret_key: a `whsk_` prefixed, base64 encoded 32-byte ed25519 private key (seed).
    def initialize(secret_key)
      seed = decode_secret_key(secret_key)
      @private_key = OpenSSL::PKey.read(ED25519_PKCS8_PREFIX + seed)
      # The raw public key is the last 32 bytes of the DER encoded SubjectPublicKeyInfo
      raw_public_key = @private_key.public_to_der[-PUBLIC_KEY_LENGTH..]
      @public_key = WebhookAsymmetric.format_public_key(raw_public_key)
    end

    # Returns the `webhook-signature` header value (`v2a,<base64 signature>`) for the
    # given webhook. Takes the same arguments as WebhookAsymmetric.signed_content_hash.
    def sign(endpoint_id, msg_id, timestamp, payload)
      digest = WebhookAsymmetric.signed_content_hash(endpoint_id, msg_id, timestamp, payload)
      WebhookAsymmetric.format_signature(@private_key.sign(nil, digest))
    end

    # Returns the `whpk_` prefixed public key to share with consumers.
    def public_key
      @public_key
    end

    def inspect
      "#<#{self.class.name} public_key=#{@public_key}>"
    end

    SECRET_KEY_PREFIX = "whsk_"
    SECRET_KEY_LENGTH = 32
    PUBLIC_KEY_LENGTH = 32
    # DER encoded PKCS#8 PrivateKeyInfo header for a raw ed25519 private key (seed).
    ED25519_PKCS8_PREFIX = ["302e020100300506032b657004220420"].pack("H*").freeze
    private_constant :SECRET_KEY_PREFIX, :SECRET_KEY_LENGTH,
      :PUBLIC_KEY_LENGTH, :ED25519_PKCS8_PREFIX

    private

    def decode_secret_key(key)
      unless key.is_a?(String) && key.start_with?(SECRET_KEY_PREFIX)
        raise EmptyWebhookSecretError, "Secret key must start with \"#{SECRET_KEY_PREFIX}\""
      end

      begin
        seed = Base64.strict_decode64(key[SECRET_KEY_PREFIX.length..])
      rescue ArgumentError
        raise EmptyWebhookSecretError, "Secret key must be valid base64"
      end

      unless seed.bytesize == SECRET_KEY_LENGTH
        raise EmptyWebhookSecretError, "Secret key must be #{SECRET_KEY_LENGTH} bytes"
      end

      seed
    end
  end
end
