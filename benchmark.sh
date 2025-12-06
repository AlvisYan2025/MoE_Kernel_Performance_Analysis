#the default benchmark script for varying basic configs (batch_size, sequence length, etc)
#change config in lauch scripts 
#u can run this in a seprate terminal (ssh to the same node)
#check with curl http://localhost:8000/health

PORT=${1:-8000}
NUM_PROMPTS=${2:-100}

echo "Waiting for server to be ready..."
sleep 30

#check server
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "Server is ready!"
        break
    fi
    echo "Waiting for server... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 2
    RETRY_COUNT=$((RETRY_COUNT + 1))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "Server failed to start!"
    exit 1
fi

echo "========================================="
echo "Running benchmark with $NUM_PROMPTS prompts"
echo "========================================="

python -m sglang.bench_serving \
  --backend vllm \
  --host localhost \
  --port $PORT \
  --dataset-name random \
  --num-prompts $NUM_PROMPTS \
  --random-input 1024 \
  --random-output 512 \
  --max-concurrency 8 \
  | tee outputs/benchmark_results.txt

echo "========================================="
echo "Benchmark complete! Results saved to benchmark_results.txt"
echo "========================================="