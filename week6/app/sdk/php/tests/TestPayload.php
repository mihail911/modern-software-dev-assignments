<?php

namespace StandardWebhooks;

final class TestPayload
{
    private const DEFAULT_MSG_ID = "msg_p5jXN8AQM9LWM0D4loKWxJek";
    private const DEFAULT_PAYLOAD = '{"test": 2432232315}';
    private const DEFAULT_SECRET = "MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw";

    public $id;
    public $timestamp;
    public $payload;
    public $secret;
    public $header;

    public function __construct(int $timestamp)
    {
        $this->id = self::DEFAULT_MSG_ID;
        $this->timestamp = strval($timestamp);

        $this->payload = self::DEFAULT_PAYLOAD;
        $this->secret = self::DEFAULT_SECRET;

        $toSign = "{$this->id}.{$this->timestamp}.{$this->payload}";
        $signature =  base64_encode(pack('H*', hash_hmac('sha256', $toSign, base64_decode($this->secret))));

        $this->header = array(
            "webhook-id" => $this->id,
            "webhook-signature" => "v1,{$signature}",
            "webhook-timestamp" => $this->timestamp,
        );
    }
}
