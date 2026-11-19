defmodule StandardWebhooksTest do
  use ExUnit.Case
  import Plug.Test
  import Plug.Conn

  alias StandardWebhooks, as: StandardWebhooks

  @id "msg_p5jXN8AQM9LWM0D4loKWxJek"
  @timestamp :os.system_time(:second)
  @tolerance 5 * 60
  @payload ~s({"event_type":"ping"})
  @secret_prefix "whsec_"
  @secret "MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw"
  @prefixed_secret @secret_prefix <> @secret

  describe "sign/4" do
    test "raises error when message id is not a String" do
      assert_raise ArgumentError, "Message id must be a string", fn ->
        StandardWebhooks.sign(123, @timestamp, @payload, @secret)
      end
    end

    test "raises error when message timestamp is not an Integer" do
      assert_raise ArgumentError, "Message timestamp must be an integer", fn ->
        StandardWebhooks.sign(@id, to_string(@timestamp), @payload, @secret)
      end
    end

    test "raises error when message timestamp is too old" do
      assert_raise ArgumentError, "Message timestamp too old", fn ->
        timestamp = :os.system_time(:second) - @tolerance - 1
        StandardWebhooks.sign(@id, timestamp, @payload, @secret)
      end
    end

    test "raises error when message timestamp is too new" do
      assert_raise ArgumentError, "Message timestamp too new", fn ->
        timestamp = :os.system_time(:second) + @tolerance + 1
        StandardWebhooks.sign(@id, timestamp, @payload, @secret)
      end
    end

    test "raises error when message payload is not a Map" do
      assert_raise ArgumentError, "Message payload must be a binary or a string", fn ->
        StandardWebhooks.sign(@id, @timestamp, %{"event_type" => "ping"}, @secret)
      end
    end

    test "raises error when secret is not a String" do
      assert_raise ArgumentError, "Secret must be a string", fn ->
        StandardWebhooks.sign(@id, @timestamp, @payload, [])
      end
    end

    test "raises error when secret is empty" do
      assert_raise ArgumentError, "Secret must not be empty", fn ->
        StandardWebhooks.sign(@id, @timestamp, @payload, "")
      end
    end

    test "returns valid signature when unencoded secret" do
      [signature_identifier, signature] =
        StandardWebhooks.sign(@id, @timestamp, @payload, @secret) |> String.split(",")

      {:ok, decoded_signature} = Base.decode64(signature)

      assert "v1" == signature_identifier
      assert is_binary(decoded_signature)
    end

    test "returns valid signature when prefixed secret" do
      [signature_identifier, signature] =
        StandardWebhooks.sign(@id, @timestamp, @payload, @prefixed_secret) |> String.split(",")

      {:ok, decoded_signature} = Base.decode64(signature)

      assert "v1" == signature_identifier
      assert is_binary(decoded_signature)
    end

    test "returns the same signature for prefixed and unprefixed secret" do
      assert StandardWebhooks.sign(@id, @timestamp, @payload, @prefixed_secret) ==
               StandardWebhooks.sign(@id, @timestamp, @payload, @secret)
    end

    test "raises error when prefixed secret is empty" do
      assert_raise ArgumentError, "Secret must not be empty", fn ->
        StandardWebhooks.sign(@id, @timestamp, @payload, @secret_prefix)
      end
    end

    test "raises error when secret has a duplicate prefix" do
      assert_raise ArgumentError, ~s(Secret has a duplicate "whsec_" prefix), fn ->
        StandardWebhooks.sign(@id, @timestamp, @payload, @secret_prefix <> @prefixed_secret)
      end
    end
  end

  describe "verify/3" do
    setup do
      signature = StandardWebhooks.sign(@id, @timestamp, @payload, @secret)

      {:ok, signature: signature}
    end

    test "raises error when empty secret", %{signature: signature} do
      conn = setup_webhook(signature)

      assert_raise ArgumentError, "Secret must not be empty", fn ->
        assert StandardWebhooks.verify(@payload, conn, "")
      end
    end

    test "return true when valid prefixed_secret and signature", %{signature: signature} do
      conn = setup_webhook(signature)

      assert StandardWebhooks.verify(@payload, conn, @prefixed_secret)
    end

    test "return true when valid secret and signature", %{signature: signature} do
      conn = setup_webhook(signature)

      assert StandardWebhooks.verify(@payload, conn, @secret)
    end

    test "return false when valid secret and invalid signature" do
      conn = setup_webhook("invalid signature")

      assert false == StandardWebhooks.verify(@payload, conn, @secret)
    end

    test "return true when multiple signatures and the valid one is not last", %{
      signature: signature
    } do
      conn = setup_webhook("#{signature} v1,Ceo5qEr07ixe2NLpvHk3FH9bwy/WavXrAFQ/9tdO6mc=")

      assert StandardWebhooks.verify(@payload, conn, @secret)
    end

    test "return true when entry without a comma followed by valid signature", %{
      signature: signature
    } do
      conn = setup_webhook("v1 #{signature}")

      assert StandardWebhooks.verify(@payload, conn, @secret)
    end

    test "return true when malformed entries followed by valid signature", %{
      signature: signature
    } do
      conn = setup_webhook(",,, v1,not,base64  #{signature}")

      assert StandardWebhooks.verify(@payload, conn, @secret)
    end

    test "return false when valid signature with another version", %{signature: signature} do
      conn = setup_webhook(String.replace(signature, "v1,", "v2,", global: false))

      assert false == StandardWebhooks.verify(@payload, conn, @secret)
    end

    test "return false when the payload is serialized differently", %{signature: signature} do
      conn = setup_webhook(signature)

      assert false == StandardWebhooks.verify(~s({"event_type": "ping"}), conn, @secret)
    end

    test "return true for a binary (non UTF-8) payload" do
      payload = <<0xFF, 0x00, 0xFE, 0x80>>
      signature = StandardWebhooks.sign(@id, @timestamp, payload, @secret)
      conn = setup_webhook(signature)

      assert StandardWebhooks.verify(payload, conn, @secret)
      assert false == StandardWebhooks.verify(<<0xFF, 0x00, 0xFE, 0x81>>, conn, @secret)
    end

    test "return true for a string payload with non-ASCII characters" do
      payload = ~s({"name":"Zoë 🚀"})
      signature = StandardWebhooks.sign(@id, @timestamp, payload, @secret)
      conn = setup_webhook(signature)

      assert StandardWebhooks.verify(payload, conn, @secret)
    end

    test "signs the payload byte for byte", %{signature: signature} do
      expected =
        :crypto.mac(
          :hmac,
          :sha256,
          Base.decode64!(@secret),
          "#{@id}.#{@timestamp}." <> @payload
        )
        |> Base.encode64()

      assert "v1," <> expected == signature
    end

    test "raises error when missing webhook header", %{signature: signature} do
      connection =
        conn(:post, "/_incoming", @payload)
        |> put_req_header("webhook-timestamp", to_string(@timestamp))
        |> put_req_header("webhook-signature", signature)

      assert_raise ArgumentError, "Missing required headers", fn ->
        StandardWebhooks.verify(@payload, connection, @secret)
      end
    end
  end

  describe "endpoint_verification_response/2" do
    @challenge "0njH7N3kV5eXPB+/eh+eoA2Ahvz2JgNp0iC/BLOl6yM="
    @challenge_response "v1,2WoO4Z1kjgoudcvk4PX/8LDD6FNJDEUHMtYqZpNx/FA="

    test "returns valid response when secret" do
      assert @challenge_response ==
               StandardWebhooks.endpoint_verification_response(@challenge, @secret)
    end

    test "returns valid response when prefixed secret" do
      assert @challenge_response ==
               StandardWebhooks.endpoint_verification_response(
                 @challenge,
                 @secret_prefix <> @secret
               )
    end

    test "raises error when secret is empty" do
      assert_raise ArgumentError, "Secret must not be empty", fn ->
        StandardWebhooks.endpoint_verification_response(@challenge, "")
      end
    end
  end

  describe "verify_endpoint_verification_response/3" do
    test "returns true when valid response" do
      assert StandardWebhooks.verify_endpoint_verification_response(
               @challenge,
               @challenge_response,
               @secret
             )
    end

    test "returns false when a list of signatures" do
      refute StandardWebhooks.verify_endpoint_verification_response(
               @challenge,
               "v2,abc " <> @challenge_response,
               @secret
             )
    end

    test "returns false when wrong challenge" do
      refute StandardWebhooks.verify_endpoint_verification_response(
               "wrong-challenge",
               @challenge_response,
               @secret
             )
    end
  end

  defp setup_webhook(signature) do
    conn(:post, "/_incoming", @payload)
    |> put_req_header("webhook-id", @id)
    |> put_req_header("webhook-timestamp", to_string(@timestamp))
    |> put_req_header("webhook-signature", signature)
  end
end
