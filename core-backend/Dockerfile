FROM maven:3.9-eclipse-temurin-21 AS build

WORKDIR /src

COPY pom.xml ./

RUN mvn -B dependency:go-offline

COPY src ./src

RUN mvn -B clean package -DskipTests


FROM eclipse-temurin:21-jre-alpine

WORKDIR /app

RUN addgroup -S spring && adduser -S spring -G spring

COPY --from=build /src/target/core-backend-*.jar app.jar

USER spring

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=5s --retries=5 \
    CMD wget -qO- http://localhost:8080/actuator/health || exit 1

ENTRYPOINT ["java", "-jar", "/app/app.jar"]
