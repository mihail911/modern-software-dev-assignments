using System;

namespace StandardWebhooks.Exceptions
{

    [Serializable]
    public class WebhookVerificationException : Exception
    {
        public WebhookVerificationException() : base() { }
        public WebhookVerificationException(string message) : base(message) { }
        public WebhookVerificationException(string message, Exception inner) : base(message, inner) { }

        protected WebhookVerificationException(System.Runtime.Serialization.SerializationInfo info,
            System.Runtime.Serialization.StreamingContext context) : base(info, context) { }
    }
}
