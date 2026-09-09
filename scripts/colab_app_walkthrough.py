# OpenTune - app.py live walkthrough in a real browser
# Exposes the Streamlit app running in Colab via a cloudflared quick tunnel
# (no signup/account needed) so you can actually click through Setup ->
# Train -> Evaluate -> Chat. Run this as ONE script in a fresh Colab GPU runtime.
#
# Uses cloudflared instead of localtunnel: localtunnel injects an
# interstitial "click to continue" page on first visit, which does NOT get
# shown to Streamlit's background fetch() calls for its own dynamically
# imported JS chunks (Selectbox/TextInput/Button) - the browser gets that
# interstitial's HTML back instead of real JS and throws "Failed to fetch
# dynamically imported module". cloudflared's quick tunnels don't have this
# interstitial step at all.

!pip install -q transformers peft trl bitsandbytes accelerate datasets huggingface_hub rouge_score "nltk<3.10" streamlit
!pip uninstall -y -q torchao  # see colab_full_verification.py - Colab's pre-installed version is incompatible with current peft

from huggingface_hub import login
login()

import os
os.makedirs("/content/opentune", exist_ok=True)
os.chdir("/content/opentune")

print("Select ALL SEVEN files: config.py, model_registry.py, data_pipeline.py, train.py, evaluate.py, inference.py, app.py")
from google.colab import files
files.upload()

required = ["config.py", "model_registry.py", "data_pipeline.py", "train.py", "evaluate.py", "inference.py", "app.py"]
missing = [f for f in required if not os.path.exists(f)]
if missing:
    raise RuntimeError(f"Missing after upload: {missing}. Re-run this cell and select all seven files.")
print("All seven files confirmed present.\n")

# Launch Streamlit in the background
get_ipython().system_raw("streamlit run app.py --server.port 8501 &> /content/logs.txt &")

import time
time.sleep(5)

# Download cloudflared (no account/signup needed for a quick tunnel)
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
!chmod +x cloudflared

# Launch the tunnel in the background, log to a file since we're not
# watching the stream live, then extract the assigned URL from that log.
get_ipython().system_raw("./cloudflared tunnel --url http://localhost:8501 &> /content/cloudflared_log.txt &")
time.sleep(8)

print("Your app URL (open this directly - no password step):\n")
found = get_ipython().getoutput(
    r"grep -o 'https://[a-zA-Z0-9.-]*\.trycloudflare\.com' /content/cloudflared_log.txt | head -1"
)
if found:
    print(found[0])
else:
    print("URL not found yet - run this cell to check the raw log directly:")
    print("  !cat /content/cloudflared_log.txt")

