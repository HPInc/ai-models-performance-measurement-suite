#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "Starting pre-commit checks..."

# 1. Run type checking
echo "Running mypy..."
if uv run mypy; then
    echo -e "${GREEN}Mypy passed${NC}"
else
    echo -e "${RED}Mypy failed${NC}"
    exit 1
fi

# 2. Generate JSON Schema
echo "Generating JSON Schema..."
if uv run scripts/generate_schema.py; then
    echo -e "${GREEN}Schema generation done${NC}"
else
    echo -e "${RED}Schema generation failed${NC}"
    exit 1
fi

# 3. Generate sample JSON
echo "Starting mock server..."
# Start the mock server in the background and silence output
uv run --extra dev tests/mock_server.py > /dev/null 2>&1 &
SERVER_PID=$!
# Ensure the server is killed when the script exits
trap "kill $SERVER_PID" EXIT

echo "Waiting for mock server to be ready..."
max_retries=30
count=0
while ! curl -s http://localhost:8000/v1/models > /dev/null; do
    sleep 1
    count=$((count+1))
    if [ $count -ge $max_retries ]; then
        echo -e "${RED}Mock server failed to start (timeout)${NC}"
        exit 1
    fi
done
echo "Mock server is ready."

echo "Generating sample JSON..."
# Store command in a variable to use it in the header later
BENCH_CMD="uv run -m llama_benchy --base-url http://localhost:8000/v1 --model test --concurrency 1 2 --enable-prefix-caching --save-total-throughput-timeseries --save-all-throughput-timeseries --runs 2 --format json --save-result schemas/sample.json"

if $BENCH_CMD; then
    echo -e "${GREEN}Sample JSON generated${NC}"
else
    echo -e "${RED}Sample JSON generation failed${NC}"
    exit 1
fi

# 4. Generate JSONC Documentation
echo "Generating JSONC documentation..."
# Generate plain JSONC first to a temp file
if uv run scripts/enrich_json.py schemas/sample.json schemas/benchmark_report_schema.json schemas/sample.jsonc.tmp; then
    # Add command as header
    echo "// Generated with command: $BENCH_CMD" > schemas/sample.jsonc
    cat schemas/sample.jsonc.tmp >> schemas/sample.jsonc
    rm schemas/sample.jsonc.tmp
    echo -e "${GREEN}JSONC documentation generated: schemas/sample.jsonc${NC}"
else
    echo -e "${RED}JSONC generation failed${NC}"
    exit 1
fi

echo -e "${GREEN}All good! Ready to commit.${NC}"
