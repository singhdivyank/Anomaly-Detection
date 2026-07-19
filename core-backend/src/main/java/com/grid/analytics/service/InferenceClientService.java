package com.grid.analytics.service;

import java.time.Duration;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import reactor.core.publisher.Mono;
import reactor.util.retry.Retry;

import com.grid.analytics.dto.InferenceRequest;
import com.grid.analytics.dto.InferenceResponse;
import com.grid.analytics.dto.TelemetryMessage;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.client.WebClientResponseException;

/**
 * Calls the Python FastAPI ML inference sidecar over HTTP
 * (POST {ml-inference}/api/v1/predict), fully asynchronously
 * via the reactive WebClient so the Kafka consumer thread is
 * never blocked waiting on model inference latency.
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class InferenceClientService {

    private final WebClient mlInferenceWebClient;

    @Value("${grid.ml-inference.predict-path:/api/v1/predict}")
    private String predictPath;

    @Value("${grid.ml-inference.timeout-ms:2000}")
    private int timeoutMs;

    public Mono<InferenceResponse> predict(TelemetryMessage message) {
        InferenceRequest request = InferenceRequest.from(message);

        return mlInferenceWebClient.post()
                .uri(predictPath)
                .bodyValue(request)
                .retrieve()
                .bodyToMono(InferenceResponse.class)
                .timeout(Duration.ofMillis(timeoutMs))
                .retryWhen(Retry.backoff(2, Duration.ofMillis(200))
                        .filter(this::isTransient))
                .doOnError(err -> log.warn("ML inference call failed for household={}: {}",
                        message.householdId(), err.getMessage()))
                .onErrorResume(err -> Mono.just(fallbackResponse()));
    }

    private boolean isTransient(Throwable throwable) {
        // Retry on timeouts/connection errors; don't retry on 4xx (bad request shape).
        return !(throwable instanceof WebClientResponseException wcre)
                || wcre.getStatusCode().is5xxServerError();
    }

    /**
     * Fail-open fallback: if the ML sidecar is unreachable, we do not want
     * to stall the telemetry pipeline. Readings are still persisted with
     * anomaly_detected=false and flagged via a low confidence score so they
     * can be reprocessed later rather than silently dropped.
     */
    private InferenceResponse fallbackResponse() {
        return new InferenceResponse(false, 0.0, 0.0);
    }

}
