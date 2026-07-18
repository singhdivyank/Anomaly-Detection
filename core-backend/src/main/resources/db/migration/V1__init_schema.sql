CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS meter_reading (
    id               BIGSERIAL,
    household_id     VARCHAR(64)      NOT NULL,
    reading_time     TIMESTAMPTZ      NOT NULL,
    kw_consumed      DOUBLE PRECISION NOT NULL,
    pricing_tier     VARCHAR(32),
    is_weekend       BOOLEAN          NOT NULL DEFAULT FALSE,
    anomaly_score    DOUBLE PRECISION,
    confidence_score DOUBLE PRECISION,
    anomaly_detected BOOLEAN          NOT NULL DEFAULT FALSE,
    ingested_at      TIMESTAMPTZ      NOT NULL DEFAULT now(),
    PRIMARY KEY (id, reading_time)
);

SELECT create_hypertable('meter_reading', 'reading_time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_meter_reading_household_time
    ON meter_reading (household_id, reading_time DESC);

CREATE INDEX IF NOT EXISTS idx_meter_reading_anomaly
    ON meter_reading (anomaly_detected, reading_time DESC)
    WHERE anomaly_detected = TRUE;

CREATE TABLE IF NOT EXISTS household (
    household_id VARCHAR(64) PRIMARY KEY,
    region       VARCHAR(64),
    tenant_id    VARCHAR(64),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS anomaly_alert (
    id                BIGSERIAL PRIMARY KEY,
    household_id      VARCHAR(64)      NOT NULL,
    severity          VARCHAR(16)      NOT NULL,
    alert_type        VARCHAR(64)      NOT NULL,
    kw_consumed       DOUBLE PRECISION NOT NULL,
    anomaly_score     DOUBLE PRECISION NOT NULL,
    confidence_score  DOUBLE PRECISION NOT NULL,
    occurred_at       TIMESTAMPTZ      NOT NULL,
    created_at        TIMESTAMPTZ      NOT NULL DEFAULT now(),
    acknowledged      BOOLEAN          NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_anomaly_alert_occurred_at
    ON anomaly_alert (occurred_at DESC);

CREATE INDEX IF NOT EXISTS idx_anomaly_alert_severity
    ON anomaly_alert (severity);
