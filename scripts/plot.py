import pandas as pd
import matplotlib.pyplot as plt

# 1. قراءة البيانات
data = pd.read_csv('benchmark_results.csv')

# 2. إعداد الرسم البياني
plt.figure(figsize=(12, 6))
plt.plot(data['Size'], data['DNNL_GFLOPS'], label='Intel oneDNN (DNNL)', marker='o', linestyle='--', linewidth=2)
plt.plot(data['Size'], data['ME_GFLOPS'], label='My Kernel (ME)', marker='s', linestyle='-', linewidth=2)

# 3. تجميل الرسم البياني
plt.title('Performance Comparison: My GEMM vs Intel oneDNN', fontsize=14)
plt.xlabel('Matrix Size (M=N=K)', fontsize=12)
plt.ylabel('Performance (GFLOPS)', fontsize=12)
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()

# 4. حفظ الرسم البياني وعرضه
plt.savefig('performance_plot.png')
plt.show()

print("Plot saved as performance_plot.png")