# frozen_string_literal: true

module StandardWebhooks
  # Secure string comparison for strings of fixed length
  #
  # While a timing attack would not be able to discern the content of
  # a secret compared via secure_compare, it is possible to determine
  # the secret length. This should be considered when using secure_compare
  # to compare weak, short secrets to user input.
  def secure_compare(a, b)
    return false unless a.bytesize == b.bytesize

    OpenSSL.fixed_length_secure_compare(a, b)
  end

  module_function :secure_compare

  # Shared timestamp tolerance check used by the webhook verifiers.
  module TimestampVerification
    TOLERANCE = 5 * 60
    private_constant :TOLERANCE

    private

    # Returns the integer timestamp, or raises WebhookVerificationError if it is
    # invalid or outside of the allowed tolerance.
    def verify_timestamp(timestamp_header)
      begin
        now = Integer(Time.now)
        timestamp = Integer(timestamp_header)
      rescue
        raise WebhookVerificationError, "Invalid Signature Headers"
      end

      if timestamp < (now - TOLERANCE)
        raise WebhookVerificationError, "Message timestamp too old"
      end

      if timestamp > (now + TOLERANCE)
        raise WebhookVerificationError, "Message timestamp too new"
      end

      timestamp
    end
  end
end
