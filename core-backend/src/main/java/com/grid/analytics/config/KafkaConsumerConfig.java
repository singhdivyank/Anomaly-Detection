package main.java.com.grid.analytics.config;

import java.util.HashMap;
import java.util.Map;

import org.apache.kafka.clients.consumer.ConsumerConfig;
import org.apache.kafka.common.serialization.StringDeserializer;
import org.apache.kafka.common.TopicPartition;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.kafka.KafkaProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.kafka.core.*;
import org.springframework.kafka.listener.ConsumerRecordRecoverer;
import org.springframework.kafka.listener.DeadLetterPublishingRecoverer;
import org.springframework.kafka.listener.DefaultErrorHandler;
import org.springframework.util.backoff.FixedBackOff;

/**
 * Configure a DeadLetterPublishingRecoverer + DefaultErrorHandler so a
 * single malformed corrupted message never stalls the stream.
 */
@Configuration
public class KafkaConsumerConfig {

    @Value("${grid.kafka.dlt-suffix:.DLT}")
    private String dltSuffix;

    @Bean
    public ConsumerFactory<String, String> consumerFactory(KafkaProperties kafkaProperties) {
        Map<String, Object> props = new HashMap<>(kafkaProperties.buildConsumerProperties(null));
        props.put(ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class);
        props.put(ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class);
        props.put(ConsumerConfig.ENABLE_AUTO_COMMIT_CONFIG, false);
        return new DefaultKafkaConsumerFactory<>(props);
    }

    @Bean
    public ProducerFactory<String, String> producerFactory(KafkaProperties KafkaProperties) {
        return new DefaultKafkaProducerFactory<>(KafkaProperties.buildProducerProperties(null));
    }

    @Bean
    public KafkaTemplate<String, String> kafkaTemplate(ProducerFactory<String, String> producerFactory) {
        return new KafkaTemplate<>(producerFactory);
    }

    /**
     * Routes any record that repeatedly fails deserialization/processing to
     * a `<topic>.DLT` dead-letter topic instead of blocking the partition.
     */
    @Bean
    public ConsumerRecordRecoverer deadLetterRecoverer(KafkaTemplate<String, String> kafkaTemplate) {
        return new DeadLetterPublishingRecoverer(kafkaTemplate,
                (record, ext) -> new TopicPartition(record.topic() + dltSuffix, record.partition()));
    }

    @Bean
    public DefaultErrorHandler kafkaErrorHandler(ConsumerRecordRecoverer deadLetterRecoverer) {
        DefaultErrorHandler handler = new DefaultErrorHandler(deadLetterRecoverer, new FixedBackOff(1000L, 3L));
        handler.addNotRetryableExceptions(IllegalArgumentException.class);
        return handler;
    }

    /**
     * Overrides Spring Boot's autoconfigured listener container factory
     * so the DefaultErrorHandler above is actually wired in.
     */
    @Bean
    public ConcurrentKafkaListenerContainerFactory<String, String> kafkaListenerContainerFactory(
            ConsumerFactory<String, String> consumerFactory,
            DefaultErrorHandler kafkaErrorHandler,
            KafkaProperties kafkaProperties) {

        ConcurrentKafkaListenerContainerFactory<String, String> factory = new ConcurrentKafkaListenerContainerFactory<>();
        factory.setConsumerFactory(consumerFactory);
        factory.setCommonErrorHandler(kafkaErrorHandler);
        factory.setConcurrency(kafkaProperties.getListener().getConcurrency() != null
                ? kafkaProperties.getListener().getConcurrency()
                : 3);
        factory.getContainerProperties().setAckMode(
                org.springframework.kafka.listener.ContainerProperties.AckMode.MANUAL);
        return factory;
    }
}
