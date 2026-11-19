using System;
using System.Runtime.CompilerServices;

namespace StandardWebhooks
{
    internal static class Utils
    {

        // Borrowed from Stripe-dotnet https://github.com/stripe/stripe-dotnet/blob/7b62c461d7c0cf2c9e06dce5e564b374a9d232e0/src/Stripe.net/Infrastructure/StringUtils.cs#L30
        // basically identical to SecureCompare from Rails::ActiveSupport used in our ruby lib
        [MethodImpl(MethodImplOptions.NoOptimization)]
        public static bool SecureCompare(string a, string b)
        {
            if (a == null)
            {
                throw new ArgumentNullException(nameof(a));
            }

            if (b == null)
            {
                throw new ArgumentNullException(nameof(b));
            }

            if (a.Length != b.Length)
            {
                return false;
            }

            var result = 0;
            for (var i = 0; i < a.Length; i++)
            {
                result |= a[i] ^ b[i];
            }

            return result == 0;
        }
    }
}
