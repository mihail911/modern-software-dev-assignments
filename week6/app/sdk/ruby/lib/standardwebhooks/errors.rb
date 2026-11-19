# frozen_string_literal: true

module StandardWebhooks
  class StandardWebhooksError < StandardError
    attr_reader :message

    def initialize(message = nil)
      @message = message
    end
  end

  class WebhookVerificationError < StandardWebhooksError
  end

  class WebhookSigningError < StandardWebhooksError
  end

  class EmptyWebhookSecretError < StandardWebhooksError
  end
end
