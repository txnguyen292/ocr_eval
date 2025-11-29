from datasets import load_dataset

def inspect_cord():
    dataset = load_dataset("naver-clova-ix/cord-v2", split="test", trust_remote_code=True, streaming=True)
    item = next(iter(dataset))
    print("Keys:", item.keys())
    gt = item.get("ground_truth")
    print(f"Ground Truth Type: {type(gt)}")
    if isinstance(gt, dict):
        print(f"Ground Truth Keys: {gt.keys()}")
        if "lines" in gt:
            print(f"First line: {gt['lines'][0] if gt['lines'] else 'Empty'}")
    else:
        print(f"Ground Truth Content (first 100 chars): {str(gt)[:100]}")

if __name__ == "__main__":
    inspect_cord()
