FROM openjdk:17-jdk-slim

# Install curl
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Cache the gradle dependencies
ADD ./build.gradle /build.gradle
ADD ./settings.gradle /settings.gradle
ADD ./gradlew /gradlew
ADD ./gradle /gradle
RUN ["./gradlew", "init"]

# Add source files
ADD ./src /src

# Handle dd-java-agent.jar
COPY ./dd-java-agent.jar* /dd-java-agent.jar
RUN if [ ! -f /dd-java-agent.jar ]; then curl -Lo /dd-java-agent.jar 'https://dtdg.co/latest-java-tracer'; fi

RUN ["./gradlew", "build"]

ENV JAVA_TOOL_OPTIONS="-javaagent:/dd-java-agent.jar"
ENTRYPOINT ["./gradlew", "run", "--no-daemon"]