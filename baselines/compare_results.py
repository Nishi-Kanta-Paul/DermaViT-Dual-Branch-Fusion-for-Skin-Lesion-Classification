"""
Compare Results
Reads summary_metrics.txt from DermaViT and all Baselines and generates a Markdown table
equivalent to Table I in the research paper specification.
"""
import os

def parse_summary_file(filepath):
    metrics = {}
    if not os.path.exists(filepath):
        return metrics
        
    with open(filepath, 'r') as f:
        for line in f:
            if ':' in line:
                key, val = line.strip().split(':')
                key = key.strip().lower()
                val = val.strip().replace('%', '')
                try:
                    metrics[key] = float(val)
                except ValueError:
                    pass
    return metrics

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_root = os.path.abspath(os.getenv("DERMAVIT_OUTPUT_DIR", os.path.join(root_dir, "outputs")))
    
    # Define models to look for, order matches Table I format
    models = [
        {"name": "ResNet-50", "path": os.path.join(output_root, "ResNet-50", "results", "summary_metrics.txt")},
        {"name": "EfficientNet-B2", "path": os.path.join(output_root, "EfficientNet-B2", "results", "summary_metrics.txt")},
        {"name": "ViT-B/16", "path": os.path.join(output_root, "ViT-B16", "results", "summary_metrics.txt")},
        {"name": "Swin-T", "path": os.path.join(output_root, "Swin-T", "results", "summary_metrics.txt")},
        {"name": "DermaViT", "path": os.path.join(output_root, "results", "summary_metrics.txt")}
    ]
    
    print("\nTABLE I")
    print("PERFORMANCE COMPARISON ON TEST SET")
    print("-" * 65)
    print(f"| {'Model':<18} | {'Acc(%)':<8} | {'Prec.':<8} | {'Rec.':<8} | {'F1':<8} | {'AUC':<8} |")
    print("-" * 65)
    
    for model in models:
        metrics = parse_summary_file(model['path'])
        if not metrics:
            print(f"| {model['name']:<18} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8} |")
            continue
            
        acc = metrics.get('accuracy', 0.0)
        prec = metrics.get('precision', 0.0)
        rec = metrics.get('recall', 0.0)
        f1 = metrics.get('f1_score', 0.0)
        auc = metrics.get('auc', 0.0)
        
        # Highlight DermaViT by wrapping in ** if writing to markdown (or just print normally)
        if model['name'] == "DermaViT":
            print(f"| **{model['name']:<14}** | **{acc:<4.1f}** | **{prec:<4.1f}** | **{rec:<4.1f}** | **{f1:<4.1f}** | **{auc:<4.1f}** |")
        else:
            print(f"| {model['name']:<18} | {acc:<8.1f} | {prec:<8.1f} | {rec:<8.1f} | {f1:<8.1f} | {auc:<8.1f} |")
            
    print("-" * 65)
    print("\n✓ Run all scripts using the following commands:")
    print("  python Baselines/train_resnet50.py")
    print("  python Baselines/train_efficientnet_b2.py")
    print("  python Baselines/train_vit_b16.py")
    print("  python Baselines/train_swin_t.py")
    print("  python DermaViT/main.py")

if __name__ == "__main__":
    main()
