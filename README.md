# TabPFN Benchmark

A simple benchmark script set for running ML experiments compared to [TabPFN](https://github.com/automl/TabPFN) across CPU/GPU.

---

## 🔧 Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/Mycrax/TabPFN_Benchmark.git
   cd TabPFN_Benchmark
   ```

2. **Install dependencies**

   ```bash
   pip install uv
   uv sync
   source .venv/bin/activate
   ```

3. **Download [TabPFN Weights](https://huggingface.co/Prior-Labs/tabpfn_2_5/blob/main/tabpfn-v2.5-classifier-v2.5_default.ckpt)**
   ```
   Go to the TabPFN hugging face and download the weights for TabPFN V2.5
   ```

4. **Define path to TabPFN**

   ```
   tabpfn_model_path = [//tabpfn-v2.5-classifier-v2.5_default.ckpt]
   located in the TabPFN_CommonScript-Bottom_GPU.py
   line 148
   ```


5. **Test Setup**
   ```
   cd Arasteh_amyloidosis
   python TabPFN_CommonScript-Bottom_CPU.py
   python TabPFN_CommonScript-Bottom_GPU.py

   If these run, then the pipeline is working, and the settings can be changed in each .py above in the first few lines
   ```

6. **Aquire Data**

   ```
   Each target has a different publically available data source described in 
   the methods section of the manuscript in detail.

   There is sample data for Arasteh_Amyloidosis to test the pipeline out on, 
   replace with the full data set for each target/dir.
   ```



---
## Usage

1. **Prepare your datasets**

   Place your DataFrames (`.csv`, `.xlsx`, etc.) into their corresponding folders. Each folder represents a separate target.

   ```
   TabPFN_Benchmark/
   ├── Arasteh_amyliodosis/
   │   ├── cpu_run.py
   │   ├── gpu_run.py
   │   └── amyloidosis_data.csv
   ├── SEER_RCC/
   │   ├── cpu_run.py
   │   ├── gpu_run.py
   │   └── SEER_RCC_data.csv
   ...
   ```

3. **Modify the path variable to the path of your data file**

   ```bash
   # Load Data
   path = "" <-- Here
   df = pd.read_csv(path)
   ```
3. **Run the benchmark scripts**

   In each dataset folder, run the appropriate script:

   ```bash
   python XXX_CPU.py    # For ML models
   python XXX_GPU.py    # For TabPFN Models
   ```

---

## Notes

- Each subfolder contains a `_CPU.py` or `_GPU.py` script customized for that dataset.

---

## Tips

- Keeping all experiments in separate folders helps manage, datasets, and outputs cleanly.

---
