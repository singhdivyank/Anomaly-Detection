## System Objective: Smart Meter Grid Reliability & Demand-Response Analytics Platform

A high-throughput, low-latency B2B enterprise SaaS platform designed for electrical grid operators to monitor multi-tenant smart meter IoT streams in real-time. The platform ingests energy usage data, handles stateful streaming data routing, runs inline machine learning inference to detect demand non-adherence and technical anomalies, and broadcasts instant dashboard alerts.

### High-Level Architecture Flow

$$\text{Data Simulator (Go)} \xrightarrow{\text{Protobuf Streaming}} \text{Apache Kafka} \xrightarrow{\text{Pub/Sub Ingestion}} \text{Spring Boot Application} \xrightarrow{\text{Async Sidecar Proxy}} \text{FastAPI (Python ML Inference)}$$

$$\text{Spring Boot Layer} \xrightarrow{\text{WebSockets / STOMP}} \text{React.js Real-time Dashboard}$$

---

## Technical Stack Matrix

| Layer                    | Technology                        | Key Responsibility                                                                                  |
| ------------------------ | --------------------------------- | --------------------------------------------------------------------------------------------------- |
| **Data Simulation**      | **Go (Golang 1.24+)**             | High-performance CSV parser and concurrent mock IoT driver streaming to Kafka clusters.             |
| **Message Broker**       | **Apache Kafka**                  | Multi-topic ingestion framework partitioned by `household_id` to guarantee ordering.                |
| **Core Backend**         | **Java 21 / Spring Boot 3.4+**    | Stream consumption, alert state aggregation, PostgreSQL/TimescaleDB persistence, WebSocket routing. |
| **ML Inference Sidecar** | **Python 3.11+ / FastAPI**        | High-throughput encapsulation of pretrained Isolation Forest, LightGBM, and Neural Network models.  |
| **Database Engine**      | **TimescaleDB / PostgreSQL**      | Hybrid storage optimization: Timescale hypertables for raw ticks; relational schemas for entities.  |
| **Frontend Framework**   | **React.js 19 (Vite) + Tailwind** | Reactive UI utilizing WebSockets, Recharts time-series mapping, and critical alert modals.          |
| **Infrastructure / IaC** | **Docker Compose / Terraform**    | Reproducible multi-service deployment targeting Google Cloud Run and GKE.                           |

---

## Monorepo Folder Structure

```filesystem
anomaly-detection-platform/
├── docker-compose.yml                 # Local orchestrator for Kafka, TimescaleDB, and backend services
├── terraform/                         # IaC for GCP infrastructure
│   ├── main.tf
│   ├── variables.tf
│   └── outputs.tf
│
├── iot-simulator/                     # [Go Service]
│   ├── main.go                        # Fast CSV reader and Kafka producer loop
│   ├── go.mod
│   └── data/                          # Mount location for raw smart_meter_data.csv
│
├── core-backend/                      # [Spring Boot Service]
│   ├── pom.xml
│   └── src/main/java/com/grid/analytics/
│       ├── config/                    # KafkaConsumerConfig, WebSocketConfig, SecurityConfig
│       ├── consumer/                  # SmartMeterStreamConsumer.java (Listens to Kafka)
│       ├── controller/                # AlertController.java, DashboardController.java
│       ├── model/                     # MeterReading.java, AnomalyAlert.java (JPA Entities)
│       ├── repository/                # MeterReadingRepository.java (Timescale hypertables query)
│       ├── service/                   # InferenceClientService.java (Calls Python sidecar over HTTP/gRPC)
│       └── websocket/                 # WebSocketAlertBroker.java
│
├── ml-inference/                      # [Python FastAPI Service - Ported from your ML Base]
│   ├── main.py                        # FastAPI microservice exposing HTTP validation endpoints
│   ├── requirements.txt
│   ├── models/                        # Pretrained serialized binaries mounted here
│   │   ├── isolation_forest_model.pkl
│   │   ├── lightgbm_model.pkl
│   │   └── nn_scaler.pkl
│   └── services/
│       └── predictor.py               # Feature reconstruction matrix processing and score generation
│
└── dashboard-ui/                      # [React.js App via Vite]
    ├── package.json
    ├── vite.config.js
    ├── tailwind.config.js
    └── src/
        ├── components/                # RealTimeChart.jsx, AlertBanner.jsx, MetricCard.jsx
        ├── hooks/                     # useWebSocket.js (Handles connection to Spring Boot STOMP broker)
        └── App.jsx

```

---

## System Specs & Integration Boundaries

### 1. Ingestion Contract (Apache Kafka)

- **Topic:** `smart-meter-telemetry`
- **Partition Key:** `household_id` (Ensures time-series ordering per home node).
- **Payload Structure (JSON):**

```json
{
  "household_id": "MAC000123",
  "timestamp": "2026-06-18T02:35:10Z",
  "kw_consumed": 0.432,
  "pricing_tier": "dToU_High",
  "is_weekend": false
}
```

### 2. Spring Boot to Python Inference Contract

- **Protocol:** Synchronous REST POST (or gRPC) targeting `http://ml-inference:8000/api/v1/predict`
- **FastAPI Processing Logic:** The Python microservice takes incoming raw stats, applies the `nn_scaler.pkl`, and formats input vectors matching your original feature engineering structure (`hour`, `day_of_week`, `rolling_mean_3h`).
- **Inference Response Payload:**

```json
{
  "anomaly_detected": true,
  "anomaly_score": -0.142,
  "confidence_score": 0.89
}
```

### 3. Frontend WebSockets Contract

- **Broker Endpoint:** `ws://localhost:8080/grid-ws-broker`
- **Subscription Destination:** `/topic/live-alerts`
- **Action:** When Spring Boot evaluates `anomaly_detected == true`, it stores the record inside the database and pushes the payload directly to the dashboard client without database polling.

---

## Development Commands Reference

Claude Code can execute these within the workspace directories to build or verify components:

- **Spin up local infrastructure dependencies:**
  `docker-compose up -d kafka zookeeper postgresql-timescale`
- **Run Go IoT Simulator:**
  `cd iot-simulator && go run main.go`
- **Build/Run Spring Boot Backend:**
  `cd core-backend && ./mvnw spring-boot:run`
- **Run Python ML Service:**
  `cd ml-inference && pip install -r requirements.txt && uvicorn main:app --reload --port 8000`
- **Run React Development Workspace:**
  `cd dashboard-ui && npm install && npm run dev`

---

## Coding Guidelines for Tool Automation

1. **Strict Type Separation:** Do not mix raw IoT tracking streams inside PostgreSQL's primary transaction block. Relational contexts must strictly contain user definitions, tenant constraints, and alert logging thresholds.
2. **Resilience & Backpressure:** The Spring Boot Kafka configuration must use a configured `DeadLetterPublishingRecoverer` and `DefaultErrorHandler` to avoid stalling the message stream if corrupted parsing logs arrive.
3. **Async Network Slicing:** Web requests sent from Spring Boot to Python's inference system must run asynchronously inside a dedicated thread pool or using Spring's reactive `WebClient` to guarantee thread protection.
