import typer
import os
import time
import pandas as pd
from dotenv import load_dotenv
from typing import Optional
from .engines.textract import TextractEngine
from .engines.openai import OpenAIVLMEngine
from .data.loader import SUPPORTED_DATASETS, load_dataset_samples
from .utils.metrics import calculate_wer, calculate_cer

load_dotenv()

app = typer.Typer()

@app.command()
def evaluate(
    dataset: str = typer.Option("docvqa", help=f"Dataset to use: {', '.join(SUPPORTED_DATASETS)}"),
    split: Optional[str] = typer.Option(None, help="HF split to load (defaults vary by dataset)"),
    engine: str = typer.Option("all", help="Engine to use: textract, openai, or all"),
    samples: int = typer.Option(10, help="Number of samples to evaluate"),
    output: str = typer.Option("results.md", help="Output file for the report"),
):
    """
    Run OCR evaluation.
    """
    print(f"Loading dataset: {dataset} ({samples} samples)...")
    try:
        data = load_dataset_samples(name=dataset, split=split, num_samples=samples)
    except Exception as e:
        print(f"Failed to load dataset '{dataset}': {e}")
        return
    
    engines = {}
    if engine in ["textract", "all"]:
        try:
            engines["Textract"] = TextractEngine()
        except Exception as e:
            print(f"Failed to initialize Textract: {e}")

    if engine in ["openai", "all"]:
        try:
            engines["OpenAI"] = OpenAIVLMEngine()
        except Exception as e:
            print(f"Failed to initialize OpenAI: {e}")
            
    if not engines:
        print("No engines available. Exiting.")
        return

    results = []

    for sample in data:
        image_path = sample["image_path"]
        ground_truth = sample["ground_truth"]
        
        print(f"Processing sample {sample['id']}...")
        
        for name, engine_instance in engines.items():
            start_time = time.time()
            try:
                prediction = engine_instance.process_image(image_path)
                latency = time.time() - start_time
                
                wer = calculate_wer(ground_truth, prediction)
                cer = calculate_cer(ground_truth, prediction)
                
                results.append({
                    "Sample ID": sample["id"],
                    "Engine": name,
                    "Latency (s)": round(latency, 2),
                    "WER": round(wer, 4),
                    "CER": round(cer, 4),
                    "Ground Truth": ground_truth[:50] + "...", # Truncate for display
                    "Prediction": prediction[:50] + "..."
                })
            except Exception as e:
                print(f"Error processing sample {sample['id']} with {name}: {e}")
                results.append({
                    "Sample ID": sample["id"],
                    "Engine": name,
                    "Latency (s)": -1,
                    "WER": -1,
                    "CER": -1,
                    "Ground Truth": "Error",
                    "Prediction": str(e)
                })

    df = pd.DataFrame(results)
    
    # Calculate averages
    summary = df.groupby("Engine")[["Latency (s)", "WER", "CER"]].mean().reset_index()
    
    print("\nEvaluation Complete!")
    print(summary)
    
    # Generate Markdown report
    with open(output, "w") as f:
        f.write("# OCR Evaluation Results\n\n")
        f.write("## Summary\n\n")
        f.write(summary.to_markdown(index=False))
        f.write("\n\n## Detailed Results\n\n")
        f.write(df.to_markdown(index=False))
        
    print(f"Report saved to {output}")

if __name__ == "__main__":
    app()
