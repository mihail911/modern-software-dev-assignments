using System;

namespace StandardWebhooks.Exceptions
{

    [Serializable]
    public class EmptyWebhookSecretException : Exception
    {
        public EmptyWebhookSecretException() : base() { }
        public EmptyWebhookSecretException(string message) : base(message) { }
        public EmptyWebhookSecretException(string message, Exception inner) : base(message, inner) { }

        protected EmptyWebhookSecretException(System.Runtime.Serialization.SerializationInfo info,
            System.Runtime.Serialization.StreamingContext context) : base(info, context) { }
    }
}
