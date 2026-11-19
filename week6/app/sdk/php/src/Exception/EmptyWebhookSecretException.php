<?php

namespace StandardWebhooks\Exception;

class EmptyWebhookSecretException extends \Exception
{
    public function __construct($message, $code = 0, ?\Throwable $previous = null)
    {
        parent::__construct($message, $code, $previous);
    }
}
