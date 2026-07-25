import glob
import json

for f in glob.glob("training_presets/*.json"):
    try:
        c = json.load(open(f, encoding="utf-8"))
    except Exception as e:
        print("unreadable:", f, e)
        continue
    if (c.get("model_type") == "KREA_2" and c.get("training_method") == "FINE_TUNE"
            and c.get("output_model_format") == "SAFETENSORS"):
        c["output_model_format"] = "ORIGINAL_TRANSFORMER"
        json.dump(c, open(f, "w", encoding="utf-8"), indent=4)
        print("fixed:", f)
