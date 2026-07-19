package com.grid.analytics.config;

import java.util.concurrent.Executor;
import java.time.Duration;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.client.reactive.ReactorClientHttpConnector;
import org.springframework.scheduling.annotation.EnableAsync;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.netty.http.client.HttpClient;
import reactor.netty.resources.ConnectionProvider;

/**
 * Async Network Slicing runs Spring Boot -> Python inference calls on a
 * dedicated connection pool, and @Async-annotated service methods run on
 * a bounded thread pool -- so a slow/backed-up ML sidecar can never block
 * the Kafka consumer threads.
 */
@Configuration
@EnableAsync
public class WebClientConfig {

    @Value("${grid.ml-inference.base-url:http://localhost:8000}")
    private String mlInferenceBaseUrl;

    @Value("${grid.ml-inference.timeout-ms:2000}")
    private int timeoutMs;

    @Bean
    public WebClient mlInferenceWebClient() {
        ConnectionProvider connectionProvider = ConnectionProvider.builder("ml-inference-pool")
                .maxConnections(50)
                .pendingAcquireTimeout(Duration.ofMillis(timeoutMs))
                .build();

        HttpClient httpClient = HttpClient.create(connectionProvider)
                .responseTimeout(Duration.ofMillis(timeoutMs));

        return WebClient.builder()
                .baseUrl(mlInferenceBaseUrl)
                .clientConnector(new ReactorClientHttpConnector(httpClient))
                .build();
    }

    @Bean(name = "inferenceTaskExecutor")
    public Executor inferenceTaskExecutor() {
        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
        executor.setCorePoolSize(8);
        executor.setMaxPoolSize(32);
        executor.setQueueCapacity(500);
        executor.setThreadNamePrefix("inference-async-");
        executor.initialize();
        return executor;
    }

}
