import json
import pandas as pd
import matplotlib.pyplot as plt
import os

try:
    # قراءة ملف الـ JSON
    with open('google_results.json') as f:
        res = json.load(f)
    
    # استخراج البيانات
    benchmarks = res['benchmarks']
    data_list = []
    for b in benchmarks:
        if 'aggregate_name' not in b: # نتجاهل التجميعات الإحصائية
            name = b['name']
            size = int(name.split('/')[-1])
            gflops = b['GFLOPS']
            kernel_type = 'ME' if 'BM_MyKernel' in name else 'DNNL'
            data_list.append({'Size': size, 'GFLOPS': gflops, 'Type': kernel_type})
    
    df = pd.DataFrame(data_list)
    me_df = df[df['Type'] == 'ME'].sort_values('Size')
    dnnl_df = df[df['Type'] == 'DNNL'].sort_values('Size')
    
    # الرسم
    plt.figure(figsize=(14, 7))
    plt.plot(dnnl_df['Size'], dnnl_df['GFLOPS'], label='Intel oneDNN (Google Bench)', marker='o', linestyle='--')
    plt.plot(me_df['Size'], me_df['GFLOPS'], label='My Kernel (Google Bench)', marker='s', linestyle='-')
    
    cpu_model = os.getenv('CPU_MODEL', 'AVX2 CPU')
    plt.title(f'Google Benchmark Performance: ME vs Intel\nHardware: {cpu_model}', fontsize=14, fontweight='bold')
    plt.xlabel('Matrix Size (N)', fontsize=12)
    plt.ylabel('Performance (GFLOPS)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig('google_bench_plot.png', dpi=300)
    print("✅ Google Benchmark plot saved as google_bench_plot.png")
except Exception as e:
    print(f"❌ Error in Google Plot: {e}")