// IoT driver that streams smart meter readings to Kafka. It reproduces the
// London Smart Meter dataset shape (LCLid, DateTime, KWH/hh) and re-maps it
// onto the platform's ingestion contract (household_id, timestamp,
// kw_consumed, pricing_tier, is_weekend), partitioned by household_id.
package main

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"cloud.google.com/go/storage"
	"github.com/joho/godotenv"
	"github.com/segmentio/kafka-go"
)

// MeterReading is the wire payload published to the smart-meter-telemetry topic
type MeterReading struct {
	HouseholdID string  `json:"household_id"`
	Timestamp   string  `json:"timestamp"`
	KWConsumed  float64 `json:"kw_consumed"`
	PricingTier string  `json:"pricing_tier"`
	IsWeekend   bool    `json:"is_weekend"`
}

type config struct {
	kafkaBrokers  string
	kafkaTopic    string
	gcsBucket     string
	gcsObject     string
	streamRateMs  int
	channelBuffer   int
}

func loadConfig() config {
	cfg := config{
		kafkaBrokers: getEnv("KAFKA_BROKERS", ""),
		kafkaTopic:   getEnv("KAFKA_TOPIC", ""),
		gcsBucket:     getEnv("GCS_BUCKET", ""),
		gcsObject:     getEnv("GCS_OBJECT", ""),
		streamRateMs: getEnvInt("STREAM_RATE_MS", 200),
		channelBuffer:  getEnvInt("HOUSEHOLD_CHANNEL_BUFFER", 32),
	}
	return cfg
}

func getEnv(key, fallback string) string {
	fmt.Println(os.LookupEnv(key))
	if v, ok := os.LookupEnv(key); ok && v != "" {
		return v
	}
	return fallback
}

func getEnvInt(key string, fallback int) int {
	if v, ok := os.LookupEnv(key); ok {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return fallback
}

func main() {
	if err := godotenv.Load(); err != nil && !os.IsNotExist(err) {
		log.Fatalf("failed to load .env: %v", err)
	}
	cfg := loadConfig()
	log.Printf("iot-simulator starting: brokers=%s topic=%s rate_ms=%d",
		cfg.kafkaBrokers, cfg.kafkaTopic, cfg.streamRateMs)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	source, closeSource, err := openCSVSource(ctx, cfg)
	if err != nil {
		log.Fatalf("failed to open CSV source: %v", err)
	}
	defer closeSource()

	writer := &kafka.Writer{
		Addr:         kafka.TCP(strings.Split(cfg.kafkaBrokers, ",")...),
		Topic:        cfg.kafkaTopic,
		Balancer:     &kafka.Hash{}, // ensures household_id -> same partition ordering
		BatchTimeout: 50 * time.Millisecond,
		RequiredAcks: kafka.RequireOne,
	}
	defer writer.Close()

	if err := streamAllHouseholds(ctx, source, writer, cfg); err != nil {
		log.Fatalf("streaming failed: %v", err)
	}
	log.Println("iot-simulator finished streaming all rows")
}

func openCSVSource(ctx context.Context, cfg config) (io.Reader, func() error, error) {
	fmt.Println("CFG:", cfg)
	client, err := storage.NewClient(ctx)
	if err != nil {
		return nil, nil, fmt.Errorf(
			"creating GCS client (check GOOGLE_APPLICATION_CREDENTIALS is set and points to a valid service account key): %w", err)
	}

	rc, err := client.Bucket(cfg.gcsBucket).Object(cfg.gcsObject).NewReader(ctx)
	if err != nil {
		_ = client.Close()
		return nil, nil, fmt.Errorf("opening gs://%s/%s: %w", cfg.gcsBucket, cfg.gcsObject, err)
	}

	log.Printf("streaming CSV from gs://%s/%s", cfg.gcsBucket, cfg.gcsObject)
	return rc, func() error {
		_ = rc.Close()
		return client.Close()
	}, nil
}

type headerIndex struct {
	lclid, tou, datetime, kwh int
}

func indexHeader(header []string) headerIndex {
	idx := headerIndex{-1, -1, -1, -1}
	for i, h := range header {
		switch strings.TrimSpace(strings.ToLower(h)) {
		case "lclid":
			idx.lclid = i
		case "stdortou":
			idx.tou = i
		case "datetime":
			idx.datetime = i
		case "kwh/hh (per half hour)", "kwh/hh":
			idx.kwh = i
		}
	}
	return idx
}

func mapRow(record []string, idx headerIndex) (MeterReading, bool) {
	if idx.lclid < 0 || idx.datetime < 0 || idx.kwh < 0 {
		return MeterReading{}, false
	}
	kwh, err := strconv.ParseFloat(strings.TrimSpace(record[idx.kwh]), 64)
	if err != nil {
		return MeterReading{}, false
	}

	ts, err := time.Parse("2006-01-02 15:04:05", strings.TrimSpace(record[idx.datetime]))
	if err != nil {
		// fall back to RFC3339 in case the source has already been normalized
		ts, err = time.Parse(time.RFC3339, strings.TrimSpace(record[idx.datetime]))
		if err != nil {
			return MeterReading{}, false
		}
	}

	tier := "Standard"
	if idx.tou >= 0 {
		if strings.EqualFold(strings.TrimSpace(record[idx.tou]), "ToU") {
			tier = "dToU_High"
		}
	}
	isWeekend := ts.Weekday() == time.Saturday || ts.Weekday() == time.Sunday

	return MeterReading{
		HouseholdID: strings.TrimSpace(record[idx.lclid]),
		Timestamp:   ts.UTC().Format(time.RFC3339),
		KWConsumed:  kwh,
		PricingTier: tier,
		IsWeekend:   isWeekend,
	}, true
}

func streamAllHouseholds(ctx context.Context, source io.Reader, writer *kafka.Writer, cfg config) error {
	r := csv.NewReader(source)
	r.ReuseRecord = true

	header, err := r.Read()
	if err != nil {
		return fmt.Errorf("reading CSV header: %w", err)
	}
	colIdx := indexHeader(header)
	if colIdx.lclid < 0 || colIdx.datetime < 0 || colIdx.kwh < 0 {
		return errors.New("CSV header missing required columns (LCLid, DateTime, KWH/hh)")
	}

	rate := time.Duration(cfg.streamRateMs) * time.Millisecond

	var mu sync.Mutex
	routes := make(map[string]chan MeterReading)
	var wg sync.WaitGroup

	rowCount := 0
	for {
		record, readErr := r.Read()
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			log.Printf("skipping malformed row: %v", readErr)
			continue
		}

		reading, ok := mapRow(record, colIdx)
		if !ok {
			continue
		}
		rowCount++

		mu.Lock()
		ch, exists := routes[reading.HouseholdID]
		if !exists {
			ch = make(chan MeterReading, cfg.channelBuffer)
			routes[reading.HouseholdID] = ch
			wg.Add(1)
			go func(householdID string, readings <-chan MeterReading) {
				defer wg.Done()
				streamHousehold(ctx, writer, householdID, readings, rate)
			}(reading.HouseholdID, ch)
		}
		mu.Unlock()

		select {
		case ch <- reading:
		case <-ctx.Done():
			closeAllRoutes(&mu, routes)
			wg.Wait()
			return ctx.Err()
		}

		if rowCount%500_000 == 0 {
			mu.Lock()
			active := len(routes)
			mu.Unlock()
			log.Printf("dispatched %d rows across %d households so far", rowCount, active)
		}
	}

	closeAllRoutes(&mu, routes)
	wg.Wait()

	log.Printf("dispatched %d total rows across %d households", rowCount, len(routes))
	return nil
}

func closeAllRoutes(mu *sync.Mutex, routes map[string]chan MeterReading) {
	mu.Lock()
	defer mu.Unlock()
	for _, ch := range routes {
		close(ch)
	}
}

func streamHousehold(ctx context.Context, writer *kafka.Writer, householdID string, readings <-chan MeterReading, rate time.Duration) {
	ticker := time.NewTicker(rate)
	defer ticker.Stop()

	for reading := range readings {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			payload, err := json.Marshal(reading)
			if err != nil {
				log.Printf("[%s] marshal error: %v", householdID, err)
				continue
			}
			if err := writer.WriteMessages(ctx, kafka.Message{
				Key:   []byte(householdID),
				Value: payload,
				Time:  time.Now(),
			}); err != nil {
				log.Printf("[%s] kafka write error: %v", householdID, err)
			}
		}
	}
}
