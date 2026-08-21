import json
from pathlib import Path
import sys

# Add src to path so we can import llama_benchy
sys.path.append(str(Path(__file__).parent.parent / "src"))

from llama_benchy.config import BenchmarkConfig
from llama_benchy.results import BenchmarkReport

def generate_json_schemas():
    output_dir = Path("schemas")
    output_dir.mkdir(exist_ok=True)

    models = [
        (BenchmarkReport, "benchmark_report_schema.json"),
    ]

    for model, filename in models:
        schema = model.model_json_schema()
        with open(output_dir / filename, "w") as f:
            json.dump(schema, f, indent=2)
        print(f"Generated schema for {model.__name__} at {output_dir / filename}")

if __name__ == "__main__":
    generate_json_schemas()
