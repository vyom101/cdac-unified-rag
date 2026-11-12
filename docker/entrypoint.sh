#!/bin/sh
set -e

if [ "${WAIT_FOR_ELASTICSEARCH:-0}" = "1" ]; then
  ES_URL="${ELASTICSEARCH_URL:-http://elasticsearch:9200}"
  echo "Waiting for Elasticsearch at ${ES_URL}..."
  until curl -fsS "${ES_URL}/_cluster/health?wait_for_status=yellow&timeout=1s" >/dev/null 2>&1; do
    sleep 2
  done
  echo "Elasticsearch is ready."
fi

exec "$@"
