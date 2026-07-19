// IoT driver that streams smart meter readings to Kafka. It reproduces the
// London Smart Meter dataset shape (LCLid, DateTime, KWH/hh) and re-maps it
// onto the platform's ingestion contract (household_id, timestamp,
// kw_consumed, pricing_tier, is_weekend), partitioned by household_id.
package main

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"io"
	"log"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

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
	csvPath       string
	streamRateMs  int
	workerCount   int
}

func loadConfig() config {
	cfg := config{
		kafkaBrokers: getEnv("KAFKA_BROKERS", "localhost:9092"),
		kafkaTopic:   getEnv("KAFKA_TOPIC", "smart-meter-telemetry"),
		csvPath:      getEnv("CSV_PATH", "./data/smart_meter_data.csv"),
		streamRateMs: getEnvInt("STREAM_RATE_MS", 200),
		workerCount:  getEnvInt("WORKER_COUNT", 8),
	}
	return cfg
}

func getEnv(key, fallback string) string {
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
	cfg := loadConfig()
	log.Printf("iot-simulator starting: brokers=%s topic=%s csv=%s rate_ms=%d workers=%d",
		cfg.kafkaBrokers, cfg.kafkaTopic, cfg.csvPath, cfg.streamRateMs, cfg.workerCount)

	rows, err := readCSV(cfg.csvPath)
	if err != nil {
		log.Fatalf("failed to read source CSV: %v", err)
	}
	log.Printf("loaded %d rows from %s", len(rows), cfg.csvPath)

	writer := &kafka.Writer{
		Addr:         kafka.TCP(strings.Split(cfg.kafkaBrokers, ",")...),
		Topic:        cfg.kafkaTopic,
		Balancer:     &kafka.Hash{}, // ensures household_id -> same partition ordering
		BatchTimeout: 50 * time.Millisecond,
		RequiredAcks: kafka.RequireOne,
	}
	defer writer.Close()

	// Fan the rows out across N concurrent household "device" goroutines
	byHousehold := groupByHousehold(rows)
	var wg sync.WaitGroup
	sem := make(chan struct{}, cfg.workerCount)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	for householdID, readings := range byHousehold {
		wg.Add(1)
		sem <- struct{}{}
		go func(id string, readings []MeterReading) {
			defer wg.Done()
			defer func() { <-sem }()
			streamHousehold(ctx, writer, id, readings, time.Duration(cfg.streamRateMs)*time.Millisecond)
		}(householdID, readings)
	}

	wg.Wait()
	log.Println("iot-simulator finished streaming all rows")
}

// streamHousehold publishes one household's time-ordered readings to Kafka
// at a fixed cadence, using household_id as the partition/message key so
// Kafka guarantees per-household ordering downstream.
func streamHousehold(ctx context.Context, writer *kafka.Writer, householdID string, readings []MeterReading, rate time.Duration) {
	ticker := time.NewTicker(rate)
	defer ticker.Stop()

	for _, reading := range readings {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			payload, err := json.Marshal(reading)
			if err != nil {
				log.Printf("[%s] marshal error: %v", householdID, err)
				continue
			}
			err = writer.WriteMessages(ctx, kafka.Message{
				Key:   []byte(householdID),
				Value: payload,
				Time:  time.Now(),
			})
			if err != nil {
				log.Printf("[%s] kafka write error: %v", householdID, err)
			}
		}
	}
}

// readCSV parses the LCL-FullData-style CSV and maps each row onto the
// MeterReading ingestion contract.
func readCSV(path string) ([]MeterReading, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.ReuseRecord = true

	header, err := r.Read()
	if err != nil {
		return nil, err
	}
	colIdx := indexHeader(header)

	var out []MeterReading
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Printf("skipping malformed row: %v", err)
			continue
		}

		reading, ok := mapRow(record, colIdx)
		if !ok {
			continue
		}
		out = append(out, reading)
	}
	return out, nil
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

func groupByHousehold(rows []MeterReading) map[string][]MeterReading {
	grouped := make(map[string][]MeterReading)
	for _, r := range rows {
		grouped[r.HouseholdID] = append(grouped[r.HouseholdID], r)
	}
	return grouped
}
