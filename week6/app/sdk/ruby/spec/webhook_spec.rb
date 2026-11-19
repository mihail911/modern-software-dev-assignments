# frozen_string_literal: true

require "securerandom"
require "standardwebhooks"

DEFAULT_MSG_ID = "msg_p5jXN8AQM9LWM0D4loKWxJek"
DEFAULT_PAYLOAD = '{"test": 2432232314}'
DEFAULT_SECRET = "MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw"
TOLERANCE = 5 * 60

class TestPayload
  attr_accessor :secret
  attr_accessor :id
  attr_accessor :timestamp
  attr_accessor :payload
  attr_accessor :signature
  attr_accessor :headers

  def initialize(timestamp = Time.now.to_i)
    @secret = DEFAULT_SECRET

    @id = DEFAULT_MSG_ID
    @timestamp = timestamp

    @payload = DEFAULT_PAYLOAD
    @secret = DEFAULT_SECRET

    toSign = "#{@id}.#{@timestamp}.#{@payload}"
    @signature = Base64.encode64(
      OpenSSL::HMAC.digest(OpenSSL::Digest.new("sha256"), Base64.decode64(DEFAULT_SECRET), toSign)
    ).strip

    @headers = {
      "webhook-id" => @id,
      "webhook-signature" => "v1,#{@signature}",
      "webhook-timestamp" => @timestamp
    }
  end
end

describe StandardWebhooks::Webhook do
  describe ".new" do
    it "rejects empty webhook secrets" do
      expect { StandardWebhooks::Webhook.new("") }.to raise_error(StandardWebhooks::EmptyWebhookSecretError)
      expect { StandardWebhooks::Webhook.new("whsec_") }.to raise_error(StandardWebhooks::EmptyWebhookSecretError)
    end
  end

  describe ".new_using_raw_bytes" do
    it "rejects empty webhook secrets" do
      expect { StandardWebhooks::Webhook.new("") }.to raise_error(StandardWebhooks::EmptyWebhookSecretError)
    end
  end

  it "missing id raises error" do
    testPayload = TestPayload.new
    testPayload.headers.delete("webhook-id")

    wh = StandardWebhooks::Webhook.new(testPayload.secret)

    expect {
      wh.verify(testPayload.payload, testPayload.headers)
    }.to raise_error(StandardWebhooks::WebhookVerificationError)
  end

  it "missing timestamp raises error" do
    testPayload = TestPayload.new
    testPayload.headers.delete("webhook-timestamp")

    wh = StandardWebhooks::Webhook.new(testPayload.secret)

    expect {
      wh.verify(testPayload.payload, testPayload.headers)
    }.to raise_error(StandardWebhooks::WebhookVerificationError)
  end

  it "missing signature raises error" do
    testPayload = TestPayload.new
    testPayload.headers.delete("webhook-signature")

    wh = StandardWebhooks::Webhook.new(testPayload.secret)

    expect {
      wh.verify(testPayload.payload, testPayload.headers)
    }.to raise_error(StandardWebhooks::WebhookVerificationError)
  end

  it "invalid signature raises error" do
    testPayload = TestPayload.new
    testPayload.headers["webhook-signature"] = "v1,g0hM9SsE+OTPJTGt/tmIKtSyZlE3uFJELVlNIOLawdd"

    wh = StandardWebhooks::Webhook.new(testPayload.secret)

    expect {
      wh.verify(testPayload.payload, testPayload.headers)
    }.to raise_error(StandardWebhooks::WebhookVerificationError)
  end

  it "valid signature is valid and returns valid json" do
    testPayload = TestPayload.new
    wh = StandardWebhooks::Webhook.new(testPayload.secret)

    json = wh.verify(testPayload.payload, testPayload.headers)
    expect(json["test"]).to eq(2432232314)
  end

  it "valid unbranded signature is valid and returns valid json" do
    testPayload = TestPayload.new
    unbrandedHeaders = {
      "webhook-id" => testPayload.headers["webhook-id"],
      "webhook-signature" => testPayload.headers["webhook-signature"],
      "webhook-timestamp" => testPayload.headers["webhook-timestamp"]
    }
    testPayload.headers = unbrandedHeaders

    wh = StandardWebhooks::Webhook.new(testPayload.secret)

    json = wh.verify(testPayload.payload, testPayload.headers)
    expect(json["test"]).to eq(2432232314)
  end

  it "old timestamp raises error" do
    testPayload = TestPayload.new(Time.now.to_i - TOLERANCE - 1)

    wh = StandardWebhooks::Webhook.new(testPayload.secret)

    expect {
      wh.verify(testPayload.payload, testPayload.headers)
    }.to raise_error(StandardWebhooks::WebhookVerificationError)
  end

  it "new timestamp raises error" do
    testPayload = TestPayload.new(Time.now.to_i + TOLERANCE + 1)

    wh = StandardWebhooks::Webhook.new(testPayload.secret)

    expect {
      wh.verify(testPayload.payload, testPayload.headers)
    }.to raise_error(StandardWebhooks::WebhookVerificationError)
  end

  it "invalid timestamp raises error" do
    testPayload = TestPayload.new("teadwd")

    wh = StandardWebhooks::Webhook.new(testPayload.secret)

    expect {
      wh.verify(testPayload.payload, testPayload.headers)
    }.to raise_error(StandardWebhooks::WebhookVerificationError)
  end

  it "multi sig pyload is valid" do
    testPayload = TestPayload.new
    sigs = [
      "v1,Ceo5qEr07ixe2NLpvHk3FH9bwy/WavXrAFQ/9tdO6mc=",
      "v2,Ceo5qEr07ixe2NLpvHk3FH9bwy/WavXrAFQ/9tdO6mc=",
      testPayload.headers["webhook-signature"], # valid signature
      "v1,Ceo5qEr07ixe2NLpvHk3FH9bwy/WavXrAFQ/9tdO6mc="
    ]
    testPayload.headers["webhook-signature"] = sigs.join(" ")

    wh = StandardWebhooks::Webhook.new(testPayload.secret)

    json = wh.verify(testPayload.payload, testPayload.headers)
    expect(json["test"]).to eq(2432232314)
  end

  it "malformed signatures are skipped" do
    testPayload = TestPayload.new
    sigs = [
      "v1",
      "v1,not-base64!",
      testPayload.headers["webhook-signature"] # valid signature
    ]
    testPayload.headers["webhook-signature"] = sigs.join(" ")

    wh = StandardWebhooks::Webhook.new(testPayload.secret)

    json = wh.verify(testPayload.payload, testPayload.headers)
    expect(json["test"]).to eq(2432232314)
  end

  it "valid signature with an unknown version raises error" do
    testPayload = TestPayload.new
    testPayload.headers["webhook-signature"] = "v2,#{testPayload.signature}"

    wh = StandardWebhooks::Webhook.new(testPayload.secret)

    expect {
      wh.verify(testPayload.payload, testPayload.headers)
    }.to raise_error(StandardWebhooks::WebhookVerificationError)
  end

  it "signature with a version and no signature raises error" do
    testPayload = TestPayload.new
    testPayload.headers["webhook-signature"] = "v1"

    wh = StandardWebhooks::Webhook.new(testPayload.secret)

    expect {
      wh.verify(testPayload.payload, testPayload.headers)
    }.to raise_error(StandardWebhooks::WebhookVerificationError)
  end

  it "signature verification works with and without prefix" do
    testPayload = TestPayload.new

    wh = StandardWebhooks::Webhook.new(testPayload.secret)
    json = wh.verify(testPayload.payload, testPayload.headers)
    expect(json["test"]).to eq(2432232314)

    wh = StandardWebhooks::Webhook.new("whsec_#{testPayload.secret}")
    json = wh.verify(testPayload.payload, testPayload.headers)
    expect(json["test"]).to eq(2432232314)
  end

  it "sign function works" do
    key = "whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw"
    msg_id = "msg_p5jXN8AQM9LWM0D4loKWxJek"
    timestamp = 1614265330
    payload = '{"test": 2432232314}'
    expected = "v1,g0hM9SsE+OTPJTGt/tmIKtSyZlE3uFJELVlNIOLJ1OE="

    wh = StandardWebhooks::Webhook.new(key)
    signature = wh.sign(msg_id, timestamp, payload)
    expect(signature).to eq(expected)
  end

  it "challenge response matches test vector" do
    wh = StandardWebhooks::Webhook.new("whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw")
    response = wh.endpoint_verification_response("0njH7N3kV5eXPB+/eh+eoA2Ahvz2JgNp0iC/BLOl6yM=")
    expect(response).to eq("v1,2WoO4Z1kjgoudcvk4PX/8LDD6FNJDEUHMtYqZpNx/FA=")
  end

  it "verify challenge response accepts test vector" do
    wh = StandardWebhooks::Webhook.new("whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw")
    challenge = "0njH7N3kV5eXPB+/eh+eoA2Ahvz2JgNp0iC/BLOl6yM="
    header = "v1,2WoO4Z1kjgoudcvk4PX/8LDD6FNJDEUHMtYqZpNx/FA="
    expect { wh.verify_endpoint_verification_response(challenge, header) }.not_to raise_error
  end

  it "verify challenge response rejects wrong challenge" do
    wh = StandardWebhooks::Webhook.new("whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw")
    challenge = "1njH7N3kV5eXPB+/eh+eoA2Ahvz2JgNp0iC/BLOl6yM="
    header = "v1,2WoO4Z1kjgoudcvk4PX/8LDD6FNJDEUHMtYqZpNx/FA="
    expect { wh.verify_endpoint_verification_response(challenge, header) }
      .to raise_error(StandardWebhooks::WebhookVerificationError)
  end

  it "verify challenge response rejects lists" do
    wh = StandardWebhooks::Webhook.new("whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw")
    challenge = "0njH7N3kV5eXPB+/eh+eoA2Ahvz2JgNp0iC/BLOl6yM="
    header = "v2,abc v1,2WoO4Z1kjgoudcvk4PX/8LDD6FNJDEUHMtYqZpNx/FA="
    expect { wh.verify_endpoint_verification_response(challenge, header) }
      .to raise_error(StandardWebhooks::WebhookVerificationError)
  end

  it "challenge response depends on the secret" do
    wh = StandardWebhooks::Webhook.new("whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSx")
    response = wh.endpoint_verification_response("0njH7N3kV5eXPB+/eh+eoA2Ahvz2JgNp0iC/BLOl6yM=")
    expect(response).not_to eq("v1,2WoO4Z1kjgoudcvk4PX/8LDD6FNJDEUHMtYqZpNx/FA=")
  end
end

ASYMMETRIC_SECRET_KEY = "whsk_4uKPjji/dhB8w6GW3YYDnIMYvXcdHrvH6lidTr4Z+EY="
ASYMMETRIC_PUBLIC_KEY = "whpk_uWKK13PfwqyOgrqT0OrWKcS2uVD7e89+oUKnkoPjx7w="
ASYMMETRIC_ENDPOINT_ID = "ep_2fVZdC8Kx9TR4ZoCqHWt3NnLgRb"
ASYMMETRIC_CHALLENGE = "0njH7N3kV5eXPB+/eh+eoA2Ahvz2JgNp0iC/BLOl6yM="
ASYMMETRIC_CHALLENGE_RESPONSE = "v2a,t0RDFhzk7OtVzgvMsjuXnodSWuby2BVX+zPqw2/ksUk="
ED25519_PKCS8_PREFIX = ["302e020100300506032b657004220420"].pack("H*")

def ed25519_key_from_secret(secret_key)
  seed = Base64.strict_decode64(secret_key.delete_prefix("whsk_"))
  OpenSSL::PKey.read(ED25519_PKCS8_PREFIX + seed)
end

def whpk_from_key(key)
  # The raw public key is the last 32 bytes of the DER encoded SubjectPublicKeyInfo
  "whpk_#{Base64.strict_encode64(key.public_to_der[-32..])}"
end

def random_asymmetric_signer
  StandardWebhooks::WebhookAsymmetricSigner.new("whsk_#{Base64.strict_encode64(SecureRandom.random_bytes(32))}")
end

class AsymmetricTestPayload
  attr_accessor :id
  attr_accessor :timestamp
  attr_accessor :payload
  attr_accessor :signature
  attr_accessor :headers

  def initialize(signer: StandardWebhooks::WebhookAsymmetricSigner.new(ASYMMETRIC_SECRET_KEY), endpoint_id: ASYMMETRIC_ENDPOINT_ID,
    timestamp: Time.now.to_i)
    @id = DEFAULT_MSG_ID
    @timestamp = timestamp
    @payload = DEFAULT_PAYLOAD

    @signature = signer.sign(endpoint_id, @id, @timestamp, @payload)

    @headers = {
      "webhook-id" => @id,
      "webhook-signature" => @signature,
      "webhook-timestamp" => @timestamp.to_s
    }
  end
end

describe StandardWebhooks::WebhookAsymmetric do
  let(:wh) { StandardWebhooks::WebhookAsymmetric.new(ASYMMETRIC_ENDPOINT_ID, [ASYMMETRIC_PUBLIC_KEY]) }

  describe ".new" do
    it "rejects an empty list of public keys" do
      expect {
        StandardWebhooks::WebhookAsymmetric.new(ASYMMETRIC_ENDPOINT_ID, [])
      }.to raise_error(StandardWebhooks::EmptyWebhookSecretError)
    end

    it "rejects public keys without the whpk_ prefix" do
      expect {
        StandardWebhooks::WebhookAsymmetric.new(ASYMMETRIC_ENDPOINT_ID, [ASYMMETRIC_PUBLIC_KEY.delete_prefix("whpk_")])
      }.to raise_error(StandardWebhooks::EmptyWebhookSecretError)
      expect {
        StandardWebhooks::WebhookAsymmetric.new(ASYMMETRIC_ENDPOINT_ID, [ASYMMETRIC_SECRET_KEY])
      }.to raise_error(StandardWebhooks::EmptyWebhookSecretError)
    end

    it "rejects public keys of the wrong length" do
      expect {
        StandardWebhooks::WebhookAsymmetric.new(ASYMMETRIC_ENDPOINT_ID, ["whpk_#{Base64.strict_encode64("a" * 31)}"])
      }.to raise_error(StandardWebhooks::EmptyWebhookSecretError)
      expect {
        StandardWebhooks::WebhookAsymmetric.new(ASYMMETRIC_ENDPOINT_ID, ["whpk_#{Base64.strict_encode64("a" * 33)}"])
      }.to raise_error(StandardWebhooks::EmptyWebhookSecretError)
      expect {
        StandardWebhooks::WebhookAsymmetric.new(ASYMMETRIC_ENDPOINT_ID, ["whpk_"])
      }.to raise_error(StandardWebhooks::EmptyWebhookSecretError)
    end

    it "rejects invalid base64 public keys" do
      expect {
        StandardWebhooks::WebhookAsymmetric.new(ASYMMETRIC_ENDPOINT_ID, ["whpk_not-base64!"])
      }.to raise_error(StandardWebhooks::EmptyWebhookSecretError)
    end

    it "rejects an empty endpoint id" do
      expect {
        StandardWebhooks::WebhookAsymmetric.new("", [ASYMMETRIC_PUBLIC_KEY])
      }.to raise_error(StandardWebhooks::EmptyWebhookSecretError)
    end

    it "rejects an endpoint id containing a dot" do
      expect {
        StandardWebhooks::WebhookAsymmetric.new("ep_1.2", [ASYMMETRIC_PUBLIC_KEY])
      }.to raise_error(StandardWebhooks::EmptyWebhookSecretError)
    end
  end

  it "test key pair matches the spec" do
    expect(whpk_from_key(ed25519_key_from_secret(ASYMMETRIC_SECRET_KEY))).to eq(ASYMMETRIC_PUBLIC_KEY)
  end

  it "verifies the spec test vector" do
    allow(Time).to receive(:now).and_return(Time.at(1674087231))

    payload = '{"type":"contact.created","timestamp":"2022-11-03T20:26:10.344522Z","data":{"id":"1f81eb52-5198-4599-803e-771906343485"}}'
    headers = {
      "webhook-id" => "msg_2KWPBgLlAfxdpx2AI54pPJ85f4W",
      "webhook-timestamp" => "1674087231",
      "webhook-signature" => "v1,K5oZfzN95Z9UVu1EsfQmfVNQhnkZ2pj9o9NDN/H/pI4= v2a,kG9XaxZjAEpKB8KeQQb5G1Xp73YH5Rf5gxoWe7cg1kYMiyspdtZXe9w+dAwKeOurAwxfqmlOG/QR4m9s37DzAg=="
    }

    json = wh.verify(payload, headers)
    expect(json["type"]).to eq("contact.created")
  end

  it "valid signature is valid and returns valid json" do
    testPayload = AsymmetricTestPayload.new

    json = wh.verify(testPayload.payload, testPayload.headers)
    expect(json["test"]).to eq(2432232314)
  end

  it "valid signature alongside other signatures is valid" do
    testPayload = AsymmetricTestPayload.new
    testPayload.headers["webhook-signature"] = [
      "v1,Ceo5qEr07ixe2NLpvHk3FH9bwy/WavXrAFQ/9tdO6mc=",
      "v2a,not-base64!",
      "v2a,Ceo5qEr07ixe2NLpvHk3FH9bwy/WavXrAFQ/9tdO6mc=",
      testPayload.signature
    ].join(" ")

    json = wh.verify(testPayload.payload, testPayload.headers)
    expect(json["test"]).to eq(2432232314)
  end

  it "signature made by the second public key is valid" do
    signer = random_asymmetric_signer
    testPayload = AsymmetricTestPayload.new(signer: signer)

    wh = StandardWebhooks::WebhookAsymmetric.new(ASYMMETRIC_ENDPOINT_ID, [ASYMMETRIC_PUBLIC_KEY, signer.public_key])
    json = wh.verify(testPayload.payload, testPayload.headers)
    expect(json["test"]).to eq(2432232314)
  end

  it "signature by an untrusted key raises error" do
    testPayload = AsymmetricTestPayload.new(signer: random_asymmetric_signer)

    expect {
      wh.verify(testPayload.payload, testPayload.headers)
    }.to raise_error(StandardWebhooks::WebhookVerificationError, "No matching signature found")
  end

  it "signature for another endpoint raises error" do
    testPayload = AsymmetricTestPayload.new(endpoint_id: "ep_other")

    expect {
      wh.verify(testPayload.payload, testPayload.headers)
    }.to raise_error(StandardWebhooks::WebhookVerificationError, "No matching signature found")
  end

  it "tampered payload raises error" do
    testPayload = AsymmetricTestPayload.new

    expect {
      wh.verify('{"test": 2432232315}', testPayload.headers)
    }.to raise_error(StandardWebhooks::WebhookVerificationError, "No matching signature found")
  end

  it "valid signature with another version raises error" do
    testPayload = AsymmetricTestPayload.new
    testPayload.headers["webhook-signature"] = testPayload.signature.sub("v2a,", "v1a,")

    expect {
      wh.verify(testPayload.payload, testPayload.headers)
    }.to raise_error(StandardWebhooks::WebhookVerificationError, "No matching signature found")
  end

  %w[webhook-id webhook-timestamp webhook-signature].each do |header|
    it "missing #{header} raises error" do
      testPayload = AsymmetricTestPayload.new
      testPayload.headers.delete(header)

      expect {
        wh.verify(testPayload.payload, testPayload.headers)
      }.to raise_error(StandardWebhooks::WebhookVerificationError, "Missing required headers")
    end
  end

  it "old timestamp raises error" do
    testPayload = AsymmetricTestPayload.new(timestamp: Time.now.to_i - TOLERANCE - 1)

    expect {
      wh.verify(testPayload.payload, testPayload.headers)
    }.to raise_error(StandardWebhooks::WebhookVerificationError, "Message timestamp too old")
  end

  it "new timestamp raises error" do
    testPayload = AsymmetricTestPayload.new(timestamp: Time.now.to_i + TOLERANCE + 1)

    expect {
      wh.verify(testPayload.payload, testPayload.headers)
    }.to raise_error(StandardWebhooks::WebhookVerificationError, "Message timestamp too new")
  end

  it "challenge response matches test vector" do
    expect(wh.endpoint_verification_response(ASYMMETRIC_CHALLENGE)).to eq(ASYMMETRIC_CHALLENGE_RESPONSE)
  end

  it "challenge response uses the first public key" do
    other_key = random_asymmetric_signer.public_key

    wh = StandardWebhooks::WebhookAsymmetric.new(ASYMMETRIC_ENDPOINT_ID, [ASYMMETRIC_PUBLIC_KEY, other_key])
    expect(wh.endpoint_verification_response(ASYMMETRIC_CHALLENGE)).to eq(ASYMMETRIC_CHALLENGE_RESPONSE)

    wh = StandardWebhooks::WebhookAsymmetric.new(ASYMMETRIC_ENDPOINT_ID, [other_key, ASYMMETRIC_PUBLIC_KEY])
    expect(wh.endpoint_verification_response(ASYMMETRIC_CHALLENGE)).not_to eq(ASYMMETRIC_CHALLENGE_RESPONSE)
  end

  it "verify challenge response accepts any of the public keys" do
    other_key = random_asymmetric_signer.public_key

    wh = StandardWebhooks::WebhookAsymmetric.new(ASYMMETRIC_ENDPOINT_ID, [other_key, ASYMMETRIC_PUBLIC_KEY])
    expect {
      wh.verify_endpoint_verification_response(ASYMMETRIC_CHALLENGE, ASYMMETRIC_CHALLENGE_RESPONSE)
    }.not_to raise_error
  end

  it "verify challenge response rejects wrong header" do
    expect {
      wh.verify_endpoint_verification_response(ASYMMETRIC_CHALLENGE, "v2a,abc")
    }.to raise_error(StandardWebhooks::WebhookVerificationError, "No matching signature found")
    expect {
      wh.verify_endpoint_verification_response(ASYMMETRIC_CHALLENGE, ASYMMETRIC_CHALLENGE_RESPONSE.sub("v2a,", "v1,"))
    }.to raise_error(StandardWebhooks::WebhookVerificationError, "No matching signature found")
    expect {
      wh.verify_endpoint_verification_response("wrong", ASYMMETRIC_CHALLENGE_RESPONSE)
    }.to raise_error(StandardWebhooks::WebhookVerificationError, "No matching signature found")
    expect {
      wh.verify_endpoint_verification_response(ASYMMETRIC_CHALLENGE, nil)
    }.to raise_error(StandardWebhooks::WebhookVerificationError, "No matching signature found")
  end
end

ASYMMETRIC_MSG_ID = "msg_2KWPBgLlAfxdpx2AI54pPJ85f4W"
ASYMMETRIC_TIMESTAMP = 1674087231
ASYMMETRIC_PAYLOAD = '{"type":"contact.created","timestamp":"2022-11-03T20:26:10.344522Z","data":{"id":"1f81eb52-5198-4599-803e-771906343485"}}'
ASYMMETRIC_CONTENT_HASH = "YXCi5iN+rjB+iEb6bvg1XbWbVBL+r2S+as3k3n1Ih/A="
ASYMMETRIC_SIGNATURE = "v2a,kG9XaxZjAEpKB8KeQQb5G1Xp73YH5Rf5gxoWe7cg1kYMiyspdtZXe9w+dAwKeOurAwxfqmlOG/QR4m9s37DzAg=="

describe "StandardWebhooks::WebhookAsymmetric signing helpers" do
  describe ".signed_content_hash" do
    it "matches the spec test vector" do
      digest = StandardWebhooks::WebhookAsymmetric.signed_content_hash(
        ASYMMETRIC_ENDPOINT_ID, ASYMMETRIC_MSG_ID, ASYMMETRIC_TIMESTAMP, ASYMMETRIC_PAYLOAD
      )
      expect(digest.bytesize).to eq(32)
      expect(Base64.strict_encode64(digest)).to eq(ASYMMETRIC_CONTENT_HASH)
    end

    it "accepts a Time timestamp" do
      digest = StandardWebhooks::WebhookAsymmetric.signed_content_hash(
        ASYMMETRIC_ENDPOINT_ID, ASYMMETRIC_MSG_ID, Time.at(ASYMMETRIC_TIMESTAMP), ASYMMETRIC_PAYLOAD
      )
      expect(Base64.strict_encode64(digest)).to eq(ASYMMETRIC_CONTENT_HASH)
    end

    it "rejects an invalid endpoint id" do
      ["", "ep_1.2", nil].each do |endpoint_id|
        expect {
          StandardWebhooks::WebhookAsymmetric.signed_content_hash(
            endpoint_id, ASYMMETRIC_MSG_ID, ASYMMETRIC_TIMESTAMP, ASYMMETRIC_PAYLOAD
          )
        }.to raise_error(StandardWebhooks::EmptyWebhookSecretError)
      end
    end

    it "rejects an invalid timestamp" do
      expect {
        StandardWebhooks::WebhookAsymmetric.signed_content_hash(
          ASYMMETRIC_ENDPOINT_ID, ASYMMETRIC_MSG_ID, "not a timestamp", ASYMMETRIC_PAYLOAD
        )
      }.to raise_error(StandardWebhooks::WebhookSigningError)
    end
  end

  describe ".format_signature" do
    it "round-trips a signature" do
      raw = Base64.strict_decode64(ASYMMETRIC_SIGNATURE.delete_prefix("v2a,"))
      expect(StandardWebhooks::WebhookAsymmetric.format_signature(raw)).to eq(ASYMMETRIC_SIGNATURE)
    end

    it "rejects signatures of the wrong length" do
      ["a" * 63, "a" * 65, ""].each do |signature|
        expect {
          StandardWebhooks::WebhookAsymmetric.format_signature(signature)
        }.to raise_error(StandardWebhooks::WebhookSigningError)
      end
    end
  end

  describe ".format_public_key" do
    let(:raw) { Base64.strict_decode64(ASYMMETRIC_PUBLIC_KEY.delete_prefix("whpk_")) }
    let(:der_prefix) { ["302a300506032b6570032100"].pack("H*") }

    it "formats a raw public key" do
      expect(StandardWebhooks::WebhookAsymmetric.format_public_key(raw)).to eq(ASYMMETRIC_PUBLIC_KEY)
    end

    it "formats a DER encoded SubjectPublicKeyInfo" do
      expect(StandardWebhooks::WebhookAsymmetric.format_public_key(der_prefix + raw)).to eq(ASYMMETRIC_PUBLIC_KEY)
    end

    it "rejects invalid public keys" do
      bad_der = ["312a300506032b6570032100"].pack("H*") + raw
      ["a" * 31, "a" * 33, bad_der].each do |public_key|
        expect {
          StandardWebhooks::WebhookAsymmetric.format_public_key(public_key)
        }.to raise_error(StandardWebhooks::WebhookSigningError)
      end
    end
  end
end

describe StandardWebhooks::WebhookAsymmetricSigner do
  let(:signer) { StandardWebhooks::WebhookAsymmetricSigner.new(ASYMMETRIC_SECRET_KEY) }

  describe ".new" do
    it "rejects secret keys without the whsk_ prefix" do
      expect {
        StandardWebhooks::WebhookAsymmetricSigner.new(ASYMMETRIC_SECRET_KEY.delete_prefix("whsk_"))
      }.to raise_error(StandardWebhooks::EmptyWebhookSecretError)
    end

    it "rejects a public key" do
      expect {
        StandardWebhooks::WebhookAsymmetricSigner.new(ASYMMETRIC_PUBLIC_KEY)
      }.to raise_error(StandardWebhooks::EmptyWebhookSecretError)
    end

    it "rejects invalid base64 secret keys" do
      expect {
        StandardWebhooks::WebhookAsymmetricSigner.new("whsk_not-base64!")
      }.to raise_error(StandardWebhooks::EmptyWebhookSecretError)
    end

    it "rejects secret keys of the wrong length" do
      ["whsk_#{Base64.strict_encode64("a" * 31)}", "whsk_#{Base64.strict_encode64("a" * 33)}", "whsk_"].each do |key|
        expect {
          StandardWebhooks::WebhookAsymmetricSigner.new(key)
        }.to raise_error(StandardWebhooks::EmptyWebhookSecretError)
      end
    end
  end

  it "public key matches the spec test vector" do
    expect(signer.public_key).to eq(ASYMMETRIC_PUBLIC_KEY)
  end

  it "signature matches the spec test vector" do
    signature = signer.sign(ASYMMETRIC_ENDPOINT_ID, ASYMMETRIC_MSG_ID, ASYMMETRIC_TIMESTAMP, ASYMMETRIC_PAYLOAD)
    expect(signature).to eq(ASYMMETRIC_SIGNATURE)
  end

  it "rejects an invalid endpoint id" do
    ["", "ep_1.2"].each do |endpoint_id|
      expect {
        signer.sign(endpoint_id, ASYMMETRIC_MSG_ID, ASYMMETRIC_TIMESTAMP, ASYMMETRIC_PAYLOAD)
      }.to raise_error(StandardWebhooks::EmptyWebhookSecretError)
    end
  end

  it "signatures verify with WebhookAsymmetric" do
    timestamp = Time.now.to_i
    signature = signer.sign(ASYMMETRIC_ENDPOINT_ID, DEFAULT_MSG_ID, timestamp, DEFAULT_PAYLOAD)

    wh = StandardWebhooks::WebhookAsymmetric.new(ASYMMETRIC_ENDPOINT_ID, [signer.public_key])
    json = wh.verify(DEFAULT_PAYLOAD, {
      "webhook-id" => DEFAULT_MSG_ID,
      "webhook-timestamp" => timestamp.to_s,
      "webhook-signature" => signature
    })
    expect(json["test"]).to eq(2432232314)
  end
end
