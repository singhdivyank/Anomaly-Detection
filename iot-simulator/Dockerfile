FROM golang:1.24-alpine AS build
WORKDIR /src
COPY go.mod ./
RUN go mod download
COPY main.go ./
RUN CGO_ENABLED=0 GOOS=linux go build -o /iot-simulator main.go

FROM alpine:3.20
RUN adduser -D -H simuser
COPY --from=build /iot-simulator /usr/local/bin/iot-simulator
USER simuser
ENTRYPOINT ["/usr/local/bin/iot-simulator"]
